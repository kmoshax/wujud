//! Train one classifier per configuration and score it on held-out rooms, with room-level bootstrap errors.

use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use wujud_model::gbdt::{train_classifier, Gbdt, TrainOptions};
use wujud_model::metrics::roc_auc;
use wujud_model::numeric::{argmax, median, percentile};
use wujud_model::occupancy::{LatchConfig, Occupancy, OccupancyLatch};

use crate::scenes::{DayWindows, StudyRoom, EMPTY, MOTION, STILL};
use crate::windows::{LinkWindows, GLRT_COLUMN};

/// research/bench/breath_visibility.py: breathing is "visible" above the 95th percentile of empty windows.
pub const FALSE_ALARM: f64 = 0.05;
/// research/bench/occupancy_sim.py threshold grid.
const LATCH_THRESHOLDS: [f64; 8] = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0];

/// One quiet or moving window of a test session.
#[derive(Clone, Debug)]
pub struct WindowRecord {
    pub room: usize,
    pub class: i64,
    pub features: Vec<f64>,
    pub glrt: f64,
    pub rate_error_bpm: f64,
}

/// `[mean, max, min]` over links of every feature (as `RoomEngine`); a single link is used as is.
fn aggregate_over_links(rows: &[&Vec<f64>]) -> Vec<f64> {
    if rows.len() == 1 {
        return rows[0].clone();
    }
    let columns = rows[0].len();
    let column = |c: usize| rows.iter().map(move |r| r[c]);
    let mean = (0..columns).map(|c| column(c).sum::<f64>() / rows.len() as f64);
    let max = (0..columns).map(|c| column(c).fold(f64::NEG_INFINITY, f64::max));
    let min = (0..columns).map(|c| column(c).fold(f64::INFINITY, f64::min));
    mean.chain(max).chain(min).collect()
}

fn common_window_count(links: &[LinkWindows]) -> usize {
    links.iter().map(|l| l.rows.len()).min().unwrap_or(0)
}

/// The link whose breathing detector fires hardest in window `w`.
fn loudest_link(links: &[LinkWindows], w: usize) -> usize {
    let glrt: Vec<f64> = links.iter().map(|l| l.rows[w][GLRT_COLUMN]).collect();
    argmax(&glrt)
}

pub fn session_records(rooms: &[StudyRoom]) -> Vec<WindowRecord> {
    let mut records = Vec::new();
    for (room_index, room) in rooms.iter().enumerate() {
        for session in &room.sessions {
            for w in 0..common_window_count(&session.links) {
                let rows: Vec<&Vec<f64>> = session.links.iter().map(|l| &l.rows[w]).collect();
                let loudest = loudest_link(&session.links, w);
                records.push(WindowRecord {
                    room: room_index,
                    class: session.class,
                    features: aggregate_over_links(&rows),
                    glrt: session.links[loudest].rows[w][GLRT_COLUMN],
                    rate_error_bpm: (session.links[loudest].breathing_rates[w]
                        - session.breath_bpm)
                        .abs(),
                });
            }
        }
    }
    records
}

fn train_options() -> TrainOptions {
    TrainOptions {
        max_iter: 300,
        learning_rate: 0.05,
        min_samples_leaf: 40,
        balanced_class_weight: true,
        ..TrainOptions::default()
    }
}

/// The classifier of research/bench/sim_calibrate.py, trained with the native trainer.
pub fn train(records: &[WindowRecord]) -> anyhow::Result<Gbdt> {
    let columns = records[0].features.len();
    let x = Array2::from_shape_fn((records.len(), columns), |(i, c)| records[i].features[c]);
    let y: Vec<i64> = records.iter().map(|r| r.class).collect();
    Ok(train_classifier(x.view(), &y, &train_options())?)
}

pub fn probabilities(model: &Gbdt, rows: &[Vec<f64>]) -> anyhow::Result<Vec<Vec<f64>>> {
    Ok(rows
        .iter()
        .map(|row| model.predict_proba(row))
        .collect::<Result<_, _>>()?)
}

fn still_share(p: &[f64]) -> f64 {
    p[1] / (p[0] + p[1] + 1e-9)
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Estimate {
    pub value: f64,
    pub se: f64,
}

/// Window records with their class probabilities, scored per room so rooms can be resampled.
pub struct ScoredWindows {
    records: Vec<WindowRecord>,
    probabilities: Vec<Vec<f64>>,
    rooms: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct WindowMetrics {
    pub still_vs_empty_auc: f64,
    pub breathing_visible: f64,
    pub rate_error_median_bpm: f64,
    pub motion_recall: f64,
    pub motion_false_alarm: f64,
}

impl ScoredWindows {
    pub fn new(records: Vec<WindowRecord>, model: &Gbdt) -> anyhow::Result<Self> {
        let rows: Vec<Vec<f64>> = records.iter().map(|r| r.features.clone()).collect();
        let rooms = records.iter().map(|r| r.room + 1).max().unwrap_or(0);
        Ok(ScoredWindows {
            probabilities: probabilities(model, &rows)?,
            records,
            rooms,
        })
    }

    fn windows_of(&self, rooms: &[usize]) -> Vec<usize> {
        let mut by_room = vec![Vec::new(); self.rooms];
        for (i, r) in self.records.iter().enumerate() {
            by_room[r.room].push(i);
        }
        rooms
            .iter()
            .flat_map(|&room| by_room[room].clone())
            .collect()
    }

    fn metrics(&self, rooms: &[usize]) -> WindowMetrics {
        let windows = self.windows_of(rooms);
        let quiet: Vec<usize> = windows
            .iter()
            .copied()
            .filter(|&i| self.records[i].class != MOTION)
            .collect();
        let moving: Vec<usize> = windows
            .iter()
            .copied()
            .filter(|&i| self.records[i].class == MOTION)
            .collect();
        let says_motion = |i: usize| argmax(&self.probabilities[i]) == MOTION as usize;
        let is_still: Vec<bool> = quiet
            .iter()
            .map(|&i| self.records[i].class == STILL)
            .collect();
        let scores: Vec<f64> = quiet
            .iter()
            .map(|&i| still_share(&self.probabilities[i]))
            .collect();
        let (visible, rate_error) = self.breathing(&quiet);
        WindowMetrics {
            still_vs_empty_auc: roc_auc(&is_still, &scores),
            breathing_visible: visible,
            rate_error_median_bpm: rate_error,
            motion_recall: share(&moving, says_motion),
            motion_false_alarm: share(&quiet, says_motion),
        }
    }

    /// Share of still windows whose GLRT clears the 5 %-false-alarm threshold of the empty windows, and the
    /// median breathing-rate error of those.
    fn breathing(&self, quiet: &[usize]) -> (f64, f64) {
        let glrt_of = |class: i64| -> Vec<f64> {
            quiet
                .iter()
                .filter(|&&i| self.records[i].class == class)
                .map(|&i| self.records[i].glrt)
                .collect()
        };
        let threshold = percentile(&glrt_of(EMPTY), 100.0 * (1.0 - FALSE_ALARM));
        let still: Vec<usize> = quiet
            .iter()
            .copied()
            .filter(|&i| self.records[i].class == STILL)
            .collect();
        let visible: Vec<usize> = still
            .iter()
            .copied()
            .filter(|&i| self.records[i].glrt > threshold)
            .collect();
        let errors: Vec<f64> = visible
            .iter()
            .map(|&i| self.records[i].rate_error_bpm)
            .collect();
        let rate_error = if errors.is_empty() {
            f64::NAN
        } else {
            median(&errors)
        };
        (visible.len() as f64 / still.len() as f64, rate_error)
    }
}

fn share(windows: &[usize], hit: impl Fn(usize) -> bool) -> f64 {
    windows.iter().filter(|&&i| hit(i)).count() as f64 / windows.len() as f64
}

fn resample(rng: &mut ChaCha8Rng, count: usize) -> Vec<usize> {
    (0..count).map(|_| rng.gen_range(0..count)).collect()
}

fn standard_error(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (finite.len() as f64 - 1.0)).sqrt()
}

#[derive(Clone, Debug, Serialize)]
pub struct WindowReport {
    pub still_vs_empty_auc: Estimate,
    pub breathing_visible: Estimate,
    pub rate_error_median_bpm: Estimate,
    pub motion_recall: Estimate,
    pub motion_false_alarm: Estimate,
    pub test_windows: usize,
}

/// Point estimates on all test rooms; standard errors from resampling whole rooms.
pub fn window_report(scored: &ScoredWindows, bootstrap: usize) -> WindowReport {
    let all: Vec<usize> = (0..scored.rooms).collect();
    let point = scored.metrics(&all);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let draws: Vec<WindowMetrics> = (0..bootstrap)
        .map(|_| scored.metrics(&resample(&mut rng, scored.rooms)))
        .collect();
    let estimate = |value: f64, pick: fn(&WindowMetrics) -> f64| Estimate {
        value,
        se: standard_error(&draws.iter().map(pick).collect::<Vec<_>>()),
    };
    WindowReport {
        still_vs_empty_auc: estimate(point.still_vs_empty_auc, |m| m.still_vs_empty_auc),
        breathing_visible: estimate(point.breathing_visible, |m| m.breathing_visible),
        rate_error_median_bpm: estimate(point.rate_error_median_bpm, |m| m.rate_error_median_bpm),
        motion_recall: estimate(point.motion_recall, |m| m.motion_recall),
        motion_false_alarm: estimate(point.motion_false_alarm, |m| m.motion_false_alarm),
        test_windows: scored.records.len(),
    }
}

/// A day reduced to what the latch reads: the static profile (all links side by side) and the class probabilities.
pub struct LatchDay {
    profiles: Vec<Vec<f64>>,
    probabilities: Vec<Vec<f64>>,
    occupied: Vec<bool>,
    walking: Vec<bool>,
}

pub fn latch_day(day: &DayWindows, model: &Gbdt) -> anyhow::Result<LatchDay> {
    let windows = common_window_count(&day.links).min(day.occupied.len());
    let rows: Vec<Vec<f64>> = (0..windows)
        .map(|w| aggregate_over_links(&day.links.iter().map(|l| &l.rows[w]).collect::<Vec<_>>()))
        .collect();
    let profiles = (0..windows)
        .map(|w| {
            day.links
                .iter()
                .flat_map(|l| l.profiles[w].clone())
                .collect()
        })
        .collect();
    Ok(LatchDay {
        profiles,
        probabilities: probabilities(model, &rows)?,
        occupied: day.occupied[..windows].to_vec(),
        walking: day.walking[..windows].to_vec(),
    })
}

/// One latch window of a day: the latch's verdict and its distance (with a fresh reference only).
struct LatchStep {
    day: usize,
    occupied: bool,
    walking: bool,
    says_present: bool,
    says_moving: bool,
    distance: Option<f64>,
}

fn run_latch(days: &[LatchDay], threshold: f64) -> Vec<LatchStep> {
    let mut steps = Vec::new();
    for (index, day) in days.iter().enumerate() {
        let mut latch = OccupancyLatch::new(LatchConfig {
            threshold,
            ..LatchConfig::default()
        });
        for w in 0..day.profiles.len() {
            let p = &day.probabilities[w];
            let (state, distance) = latch.step(&day.profiles[w], p[2], Some(still_share(p)));
            steps.push(LatchStep {
                day: index,
                occupied: day.occupied[w],
                walking: day.walking[w],
                says_present: state != Occupancy::Empty,
                says_moving: state == Occupancy::Moving,
                distance,
            });
        }
    }
    steps
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct LatchMetrics {
    pub distance_auc_still_vs_empty: f64,
    pub accuracy: f64,
    pub still_detected: f64,
    pub false_present_when_empty: f64,
    pub walking_flagged_as_motion: f64,
}

fn latch_metrics(steps: &[&LatchStep]) -> LatchMetrics {
    let quiet: Vec<&&LatchStep> = steps.iter().filter(|s| !s.walking).collect();
    let fresh: Vec<&&&LatchStep> = quiet.iter().filter(|s| s.distance.is_some()).collect();
    let auc = roc_auc(
        &fresh.iter().map(|s| s.occupied).collect::<Vec<_>>(),
        &fresh
            .iter()
            .map(|s| s.distance.unwrap_or(0.0))
            .collect::<Vec<_>>(),
    );
    let rate = |filter: &dyn Fn(&LatchStep) -> bool, hit: &dyn Fn(&LatchStep) -> bool| {
        let chosen: Vec<&&&LatchStep> = quiet.iter().filter(|s| filter(s)).collect();
        chosen.iter().filter(|s| hit(s)).count() as f64 / chosen.len() as f64
    };
    let walking: Vec<&&LatchStep> = steps.iter().filter(|s| s.walking).collect();
    LatchMetrics {
        distance_auc_still_vs_empty: auc,
        accuracy: rate(&|_| true, &|s| s.says_present == s.occupied),
        still_detected: rate(&|s| s.occupied, &|s| s.says_present),
        false_present_when_empty: rate(&|s| !s.occupied, &|s| s.says_present),
        walking_flagged_as_motion: walking.iter().filter(|s| s.says_moving).count() as f64
            / walking.len() as f64,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LatchReport {
    pub threshold: f64,
    pub tune_accuracy: Vec<(f64, f64)>,
    pub distance_auc_still_vs_empty: Estimate,
    pub test: LatchMetrics,
    pub windows_with_fresh_reference: usize,
}

/// research/bench/occupancy_sim.py: threshold chosen for accuracy on tuning days, reported on test days.
pub fn latch_report(tune: &[LatchDay], test: &[LatchDay], bootstrap: usize) -> LatchReport {
    let accuracy_at = |threshold: f64| {
        let steps = run_latch(tune, threshold);
        latch_metrics(&steps.iter().collect::<Vec<_>>()).accuracy
    };
    let tune_accuracy: Vec<(f64, f64)> = LATCH_THRESHOLDS
        .iter()
        .map(|&t| (t, accuracy_at(t)))
        .collect();
    let threshold = tune_accuracy
        .iter()
        .fold((LATCH_THRESHOLDS[0], f64::NEG_INFINITY), |best, &(t, a)| {
            if a > best.1 {
                (t, a)
            } else {
                best
            }
        })
        .0;
    let steps = run_latch(test, threshold);
    let all: Vec<&LatchStep> = steps.iter().collect();
    let point = latch_metrics(&all);
    let mut by_day: Vec<Vec<&LatchStep>> = vec![Vec::new(); test.len()];
    for step in &steps {
        by_day[step.day].push(step);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let draws: Vec<f64> = (0..bootstrap)
        .map(|_| {
            let chosen: Vec<&LatchStep> = resample(&mut rng, test.len())
                .into_iter()
                .flat_map(|d| by_day[d].clone())
                .collect();
            latch_metrics(&chosen).distance_auc_still_vs_empty
        })
        .collect();
    LatchReport {
        threshold,
        tune_accuracy,
        distance_auc_still_vs_empty: Estimate {
            value: point.distance_auc_still_vs_empty,
            se: standard_error(&draws),
        },
        test: point,
        windows_with_fresh_reference: steps
            .iter()
            .filter(|s| !s.walking && s.distance.is_some())
            .count(),
    }
}
