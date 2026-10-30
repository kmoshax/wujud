//! Laptop beacon sensing (`research/apsense.py::Engine`): access-point beacon RSSI per laptop antenna, every second
//! motion from the wobble, and (after a calibration walk) which spot the room looks like.

use std::collections::{BTreeMap, HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::numeric::{median, nan_median, nan_percentile, pairwise_sum, percentile, round_to};
use crate::scene::Scene;

const BUFFER_SECONDS: f64 = 60.0;
const WINDOW_SECONDS: f64 = 4.0;
/// The card hops away for ~1 s between dwell bursts.
const BURST_GAP_SECONDS: f64 = 0.4;
const MIN_SAMPLES: usize = 4;
const LAPTOP_CHAINS: usize = 2;
const HISTORY_LEN: usize = 900;
const MOTION_HISTORY_LEN: usize = 3 * 3600;
const LEARNING_SECONDS: usize = 120;
const FLOOR_LOOKBACK: usize = 1800;
/// Links are correlated: the likelihood is tempered.
const LIKELIHOOD_TEMPER: f64 = 4.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccessPoint {
    pub name: String,
    pub chan: i64,
    pub bssids: Vec<String>,
    pub xyz: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeaconRoom {
    #[serde(rename = "W")]
    pub width: f64,
    #[serde(rename = "D")]
    pub depth: f64,
    #[serde(rename = "H", default = "default_height")]
    pub height: f64,
    pub laptop: [f64; 3],
    pub aps: Vec<AccessPoint>,
    #[serde(default)]
    pub furniture: Vec<serde_json::Value>,
}

fn default_height() -> f64 {
    2.8
}

impl BeaconRoom {
    pub fn scene(&self) -> Scene {
        let nodes: Vec<[f64; 3]> = std::iter::once(self.laptop)
            .chain(self.aps.iter().map(|ap| ap.xyz))
            .collect();
        let roles = std::iter::once("esp".to_string())
            .chain(self.aps.iter().map(|_| "router".to_string()))
            .collect();
        Scene::room(
            self.width,
            self.depth,
            self.height,
            &nodes,
            self.furniture.clone(),
            roles,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationWindow {
    pub label: String,
    pub xy: Option<[f64; 2]>,
    pub t: f64,
    pub sess: i64,
    pub m: Vec<f64>,
    pub mot: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Calibration {
    pub windows: Vec<CalibrationWindow>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CrossValidation {
    pub acc: f64,
    pub n: usize,
    pub loc_err_m: Option<f64>,
    pub rounds: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PendingReport {
    pub label: String,
    pub wait: i64,
    pub left: i64,
    pub n: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpotSummary {
    pub label: String,
    pub n: usize,
    pub xy: Option<[f64; 2]>,
}

/// Per-link diagnostics for the calibration page (`Engine.last`).
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct BeaconDiagnostics {
    pub m: Vec<f64>,
    pub mot: Vec<f64>,
    pub keys: Vec<(usize, String, usize)>,
    pub pending: Option<PendingReport>,
    pub cv: Option<CrossValidation>,
    pub spots: Vec<SpotSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AutoReport {
    pub move_ratio: Vec<f64>,
    pub excess: Vec<f64>,
    pub floor: Vec<f64>,
}

/// `state` / `state3`: "moving", "still", "empty" when calibrated, else "moving" or "quiet".
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct BeaconReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub motion: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probs: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xy: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spot: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto: Option<AutoReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zone: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub still_hint: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state3: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibrated: Option<bool>,
}

/// `[t, motion, null]`.
pub type BeaconHistoryPoint = (f64, Option<f64>, Option<f64>);

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BeaconState {
    pub mode: String,
    pub t: f64,
    pub scene: Scene,
    pub room: BeaconReport,
    pub link_activity: Vec<f64>,
    pub alive: usize,
    pub history: Vec<BeaconHistoryPoint>,
}

/// One link: (access point, BSSID, laptop antenna). Two BSSIDs of one radio sit ~0.5 dB apart; mixing them fakes wobble.
type LinkKey = (usize, String, usize);

struct Sample {
    time: f64,
    link: usize,
    dbm: f64,
}

struct PendingCalibration {
    label: String,
    xy: Option<[f64; 2]>,
    start: f64,
    end: f64,
    windows: usize,
    session: i64,
}

struct SpotModel {
    labels: Vec<String>,
    means: Vec<Vec<f64>>,
    spread: Vec<f64>,
    xy: Vec<Option<[f64; 2]>>,
}

/// Mean level and wobble of every link over the last window.
struct LinkWindow {
    level: Vec<f64>,
    motion: Vec<f64>,
    alive: Vec<bool>,
}

pub struct BeaconEngine {
    room: BeaconRoom,
    links: Vec<LinkKey>,
    access_point_of: HashMap<String, usize>,
    samples: VecDeque<Sample>,
    calibration: Calibration,
    pending: Option<PendingCalibration>,
    motion_threshold: f64,
    spot_model: Option<SpotModel>,
    cross_validation: Option<CrossValidation>,
    smoothed: Option<Vec<f64>>,
    history: VecDeque<BeaconHistoryPoint>,
    first_tick: Option<f64>,
    motion_history: VecDeque<Vec<f64>>,
    last: BeaconDiagnostics,
}

impl BeaconEngine {
    pub fn new(room: BeaconRoom, calibration: Calibration) -> Self {
        let links: Vec<LinkKey> = room
            .aps
            .iter()
            .enumerate()
            .flat_map(|(i, ap)| {
                ap.bssids
                    .iter()
                    .flat_map(move |b| (0..LAPTOP_CHAINS).map(move |c| (i, b.to_lowercase(), c)))
            })
            .collect();
        let access_point_of = room
            .aps
            .iter()
            .enumerate()
            .flat_map(|(i, ap)| ap.bssids.iter().map(move |b| (b.to_lowercase(), i)))
            .collect();
        let mut engine = BeaconEngine {
            room,
            links,
            access_point_of,
            samples: VecDeque::new(),
            calibration,
            pending: None,
            motion_threshold: 1.0,
            spot_model: None,
            cross_validation: None,
            smoothed: None,
            history: VecDeque::new(),
            first_tick: None,
            motion_history: VecDeque::new(),
            last: BeaconDiagnostics::default(),
        };
        engine.fit();
        engine
    }

    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    pub fn diagnostics(&self) -> &BeaconDiagnostics {
        &self.last
    }

    pub fn delete_spot(&mut self, label: &str) {
        self.calibration.windows.retain(|w| w.label != label);
        self.fit();
    }

    /// One beacon: BSSID and the dBm of each laptop antenna. The channel tag is unreliable around a hop; the MAC isn't.
    pub fn add(&mut self, t: f64, bssid: &str, chains: &[i64]) {
        let Some(&ap) = self.access_point_of.get(bssid) else {
            return;
        };
        for (chain, &dbm) in chains.iter().take(LAPTOP_CHAINS).enumerate() {
            if let Some(link) = self
                .links
                .iter()
                .position(|k| k.0 == ap && k.1 == bssid && k.2 == chain)
            {
                self.samples.push_back(Sample {
                    time: t,
                    link,
                    dbm: dbm as f64,
                });
            }
        }
        while self
            .samples
            .front()
            .is_some_and(|s| s.time < t - BUFFER_SECONDS)
        {
            self.samples.pop_front();
        }
    }

    /// Records the room as `label` (at `xy`, or `None` for "empty") for `seconds`, starting `delay` s after `now`.
    pub fn start_calibration(
        &mut self,
        now: f64,
        label: &str,
        xy: Option<[f64; 2]>,
        seconds: f64,
        delay: f64,
    ) {
        self.pending = Some(PendingCalibration {
            label: label.into(),
            xy,
            start: now + delay,
            end: now + delay + seconds,
            windows: 0,
            session: now.round_ties_even() as i64,
        });
    }

    fn link_window(&self, now: f64) -> LinkWindow {
        let mut window = LinkWindow {
            level: vec![],
            motion: vec![],
            alive: vec![],
        };
        for link in 0..self.links.len() {
            let mut points: Vec<(f64, f64)> = self
                .samples
                .iter()
                .filter(|s| s.time >= now - WINDOW_SECONDS && s.link == link)
                .map(|s| (s.time, s.dbm))
                .collect();
            if points.len() < MIN_SAMPLES {
                window.level.push(f64::NAN);
                window.motion.push(f64::NAN);
                window.alive.push(false);
                continue;
            }
            points.sort_by(|a, b| a.0.total_cmp(&b.0));
            let values: Vec<f64> = points.iter().map(|p| p.1).collect();
            window
                .level
                .push(pairwise_sum(&values) / values.len() as f64);
            window.motion.push(wobble(&points));
            window.alive.push(true);
        }
        window
    }

    fn calibration_tick(&mut self, now: f64) {
        let Some(pending) = &self.pending else {
            return;
        };
        if now < pending.start {
            return;
        }
        if now - pending.start >= WINDOW_SECONDS {
            let window = self.link_window(now);
            if window.alive.iter().all(|&a| a) {
                let pending = self.pending.as_mut().unwrap();
                self.calibration.windows.push(CalibrationWindow {
                    label: pending.label.clone(),
                    xy: pending.xy,
                    t: now,
                    sess: pending.session,
                    m: window.level,
                    mot: window.motion,
                });
                pending.windows += 1;
            }
        }
        if now >= self.pending.as_ref().unwrap().end {
            self.pending = None;
            self.fit();
        }
    }

    fn labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = self
            .calibration
            .windows
            .iter()
            .map(|w| w.label.clone())
            .collect();
        labels.sort();
        labels.dedup();
        labels
    }

    /// Per-spot mean level vectors, pooled per-link spread; the motion threshold from the empty room.
    fn fit(&mut self) {
        self.spot_model = None;
        self.cross_validation = None;
        let windows = &self.calibration.windows;
        if windows.is_empty() {
            self.motion_threshold = 1.0;
            return;
        }
        let empty_wobble: Vec<f64> = windows
            .iter()
            .filter(|w| w.label == "empty")
            .map(|w| median(&w.mot))
            .collect();
        self.motion_threshold = if empty_wobble.len() > 5 {
            (1.8 * percentile(&empty_wobble, 99.0)).max(0.6)
        } else {
            1.0
        };
        let labels = self.labels();
        if labels.len() < 2 {
            return;
        }
        let all: Vec<&CalibrationWindow> = windows.iter().collect();
        self.spot_model = Some(fit_spots(&all, &labels));
        self.cross_validation = cross_validate(&all, &labels);
    }

    pub fn tick(&mut self, now: f64) -> BeaconState {
        self.calibration_tick(now);
        let window = self.link_window(now);
        let mut room = BeaconReport::default();
        if window.alive.iter().any(|&a| a) {
            let wobble = nan_median(&window.motion);
            room.motion = Some(round_to(wobble, 2));
            room.thr = Some(round_to(self.motion_threshold, 2));
            let state = if self.spot_model.is_some() && window.alive.iter().all(|&a| a) {
                self.calibrated_state(&window, wobble > self.motion_threshold, &mut room)
            } else {
                self.self_calibrating_state(&window.motion, &mut room)
            };
            room.state = Some(state.clone());
            room.state3 = Some(state);
            room.calibrated = Some(self.spot_model.is_some());
        }
        let first_tick = *self.first_tick.get_or_insert(now);
        let t = round_to(now - first_tick, 1);
        if self.history.len() == HISTORY_LEN {
            self.history.pop_front();
        }
        self.history.push_back((t, room.motion, None));
        let link_activity = self.access_point_activity(&window);
        let alive = (0..self.room.aps.len())
            .filter(|&ap| self.access_point_alive(&window, ap))
            .count();
        self.last = self.diagnostics_at(now, &window);
        BeaconState {
            mode: "LIVE".into(),
            t,
            scene: self.room.scene(),
            room,
            link_activity,
            alive,
            history: self.history.iter().copied().collect(),
        }
    }

    fn links_of(&self, ap: usize) -> Vec<usize> {
        (0..self.links.len())
            .filter(|&i| self.links[i].0 == ap)
            .collect()
    }

    fn access_point_alive(&self, window: &LinkWindow, ap: usize) -> bool {
        self.links_of(ap).iter().any(|&i| window.alive[i])
    }

    fn access_point_activity(&self, window: &LinkWindow) -> Vec<f64> {
        (0..self.room.aps.len())
            .map(|ap| {
                if !self.access_point_alive(window, ap) {
                    return 0.0;
                }
                let wobble: Vec<f64> = self
                    .links_of(ap)
                    .iter()
                    .map(|&i| window.motion[i])
                    .collect();
                (sequential_nan_mean(&wobble) / (2.0 * self.motion_threshold)).clamp(0.0, 1.0)
            })
            .collect()
    }

    fn calibrated_state(
        &mut self,
        window: &LinkWindow,
        moving: bool,
        room: &mut BeaconReport,
    ) -> String {
        let model = self.spot_model.as_ref().unwrap();
        let posterior = spot_posterior(model, &window.level);
        let smoothed = match &self.smoothed {
            Some(previous) if previous.len() == posterior.len() => previous
                .iter()
                .zip(&posterior)
                .map(|(s, p)| 0.7 * s + 0.3 * p)
                .collect(),
            _ => posterior,
        };
        self.smoothed = Some(smoothed.clone());
        room.probs = Some(
            model
                .labels
                .iter()
                .zip(&smoothed)
                .map(|(l, p)| (l.clone(), round_to(*p, 3)))
                .collect(),
        );
        let p_empty = model
            .labels
            .iter()
            .position(|l| l == "empty")
            .map_or(0.0, |i| smoothed[i]);
        if let Some((xy, spot)) = expected_position(model, &smoothed) {
            room.xy = Some(xy.map(|v| round_to(v, 2)));
            room.spot = Some(model.labels[spot].clone());
        }
        let state = if moving {
            "moving"
        } else if p_empty > 0.5 {
            "empty"
        } else {
            "still"
        };
        if state == "empty" {
            room.xy = None;
        }
        state.into()
    }

    /// No calibration: each link's empty-room wobble is learnt as a low percentile of its own 20-s average over the
    /// last hours. Walk test (6 min, the user's room): walking caught 109/120 s; still vs empty NOT separable.
    fn self_calibrating_state(&mut self, motion: &[f64], room: &mut BeaconReport) -> String {
        if self.motion_history.len() == MOTION_HISTORY_LEN {
            self.motion_history.pop_front();
        }
        self.motion_history.push_back(motion.to_vec());
        let history: Vec<&Vec<f64>> = self.motion_history.iter().collect();
        let n = history.len();
        let recent_4 = column_nan_means(&history[n.saturating_sub(4)..]);
        let recent_20 = column_nan_means(&history[n.saturating_sub(20)..]);
        if n < LEARNING_SECONDS {
            room.learning = Some(format!("learning this room: {n}/{LEARNING_SECONDS} s"));
            return if sequential_nan_mean(&recent_4) > 1.5 {
                "moving"
            } else {
                "quiet"
            }
            .into();
        }
        let averages_20s: Vec<Vec<f64>> = (n.saturating_sub(FLOOR_LOOKBACK)..n)
            .step_by(5)
            .map(|i| column_nan_means(&history[i.saturating_sub(19)..=i]))
            .collect();
        let columns = motion.len();
        let floor: Vec<f64> = (0..columns)
            .map(|c| nan_percentile(&averages_20s.iter().map(|r| r[c]).collect::<Vec<_>>(), 5.0))
            .collect();
        let typical: Vec<f64> = (0..columns)
            .map(|c| {
                nan_median(
                    &history[n.saturating_sub(FLOOR_LOOKBACK)..]
                        .iter()
                        .map(|r| r[c])
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let pick = |values: &[f64], links: &[usize]| {
            links.iter().map(|&i| values[i]).collect::<Vec<f64>>()
        };
        let access_points: Vec<Vec<usize>> = (0..self.room.aps.len())
            .map(|ap| self.links_of(ap))
            .collect();
        let move_ratio: Vec<f64> = access_points
            .iter()
            .map(|k| {
                sequential_nan_mean(&pick(&recent_4, k))
                    / python_max(0.3, sequential_nan_mean(&pick(&typical, k)))
            })
            .collect();
        let excess_per_link: Vec<f64> = recent_20.iter().zip(&floor).map(|(r, f)| r - f).collect();
        let excess: Vec<f64> = access_points
            .iter()
            .map(|k| sequential_nan_mean(&pick(&excess_per_link, k)))
            .collect();
        room.auto = Some(AutoReport {
            move_ratio: move_ratio.iter().map(|v| round_to(*v, 2)).collect(),
            excess: excess.iter().map(|v| round_to(*v, 2)).collect(),
            floor: floor.iter().map(|v| round_to(*v, 2)).collect(),
        });
        if python_max_of(&move_ratio) > 2.0 {
            let ap = &self.room.aps[nan_argmax(&move_ratio)];
            let (laptop, target) = (
                [self.room.laptop[0], self.room.laptop[1]],
                [ap.xyz[0], ap.xyz[1]],
            );
            room.zone = Some(format!("near the laptop-{} line", ap.name));
            // where that link is most sensitive: the middle of its Fresnel zone
            room.xy = Some([0, 1].map(|i| round_to(laptop[i] + 0.45 * (target[i] - laptop[i]), 2)));
            return "moving".into();
        }
        // still vs empty from beacon RSSI failed the walk test (60 %), so only a hint is shown
        let quietest = excess_per_link
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .fold(f64::NAN, f64::min);
        room.still_hint = Some(quietest > 0.1);
        "quiet".into()
    }

    fn diagnostics_at(&self, now: f64, window: &LinkWindow) -> BeaconDiagnostics {
        BeaconDiagnostics {
            m: window.level.iter().map(|v| round_to(*v, 2)).collect(),
            mot: window.motion.iter().map(|v| round_to(*v, 2)).collect(),
            keys: self.links.clone(),
            pending: self.pending.as_ref().map(|p| PendingReport {
                label: p.label.clone(),
                wait: (p.start - now).max(0.0).round_ties_even() as i64,
                left: (p.end - now).max(0.0).round_ties_even() as i64,
                n: p.windows,
            }),
            cv: self.cross_validation.clone(),
            spots: self.spots(),
        }
    }

    /// Calibrated spots in the order they were first recorded.
    pub fn spots(&self) -> Vec<SpotSummary> {
        let mut spots: Vec<SpotSummary> = Vec::new();
        for w in &self.calibration.windows {
            match spots.iter_mut().find(|s| s.label == w.label) {
                Some(spot) => spot.n += 1,
                None => spots.push(SpotSummary {
                    label: w.label.clone(),
                    n: 1,
                    xy: w.xy,
                }),
            }
        }
        spots
    }
}

/// Within-burst RMS wobble plus half the mean jump between bursts.
fn wobble(points: &[(f64, f64)]) -> f64 {
    let bursts = split_bursts(points);
    let burst_means: Vec<f64> = bursts
        .iter()
        .map(|b| pairwise_sum(b) / b.len() as f64)
        .collect();
    let deviations: Vec<f64> = bursts
        .iter()
        .zip(&burst_means)
        .flat_map(|(b, m)| b.iter().map(move |v| (v - m) * (v - m)))
        .collect();
    let within = (pairwise_sum(&deviations) / deviations.len() as f64).sqrt();
    let between = if burst_means.len() > 1 {
        let jumps: Vec<f64> = burst_means
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();
        pairwise_sum(&jumps) / jumps.len() as f64
    } else {
        0.0
    };
    within + 0.5 * between
}

fn split_bursts(points: &[(f64, f64)]) -> Vec<Vec<f64>> {
    let mut bursts: Vec<Vec<f64>> = vec![vec![points[0].1]];
    for pair in points.windows(2) {
        if pair[1].0 - pair[0].0 > BURST_GAP_SECONDS {
            bursts.push(vec![]);
        }
        bursts.last_mut().unwrap().push(pair[1].1);
    }
    bursts
}

/// Column means over rows, added row by row like NumPy's axis-0 reduction.
fn column_means(rows: &[&Vec<f64>]) -> Vec<f64> {
    let columns = rows[0].len();
    (0..columns)
        .map(|c| rows.iter().fold(0.0, |acc, r| acc + r[c]) / rows.len() as f64)
        .collect()
}

fn column_nan_means(rows: &[&Vec<f64>]) -> Vec<f64> {
    let columns = rows[0].len();
    (0..columns)
        .map(|c| {
            let finite = rows.iter().filter(|r| !r[c].is_nan()).count();
            let sum = rows
                .iter()
                .fold(0.0, |acc, r| acc + if r[c].is_nan() { 0.0 } else { r[c] });
            sum / finite as f64
        })
        .collect()
}

fn sequential_nan_mean(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    finite.iter().fold(0.0, |acc, v| acc + v) / finite.len() as f64
}

/// Python's `max(a, b)`: keeps `a` unless `b > a` (so a NaN `b` is ignored).
fn python_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

fn python_max_of(values: &[f64]) -> f64 {
    values[1..]
        .iter()
        .fold(values[0], |acc, &v| python_max(acc, v))
}

fn nan_argmax(values: &[f64]) -> usize {
    crate::numeric::nan_argmax(values)
}

fn fit_spots(windows: &[&CalibrationWindow], labels: &[String]) -> SpotModel {
    let mut means = Vec::new();
    let mut residuals: Vec<Vec<f64>> = Vec::new();
    let mut xy = Vec::new();
    for label in labels {
        let spot: Vec<&Vec<f64>> = windows
            .iter()
            .filter(|w| &w.label == label)
            .map(|w| &w.m)
            .collect();
        let mean = column_means(&spot);
        residuals.extend(
            spot.iter()
                .map(|row| row.iter().zip(&mean).map(|(v, m)| v - m).collect()),
        );
        means.push(mean);
        xy.push(
            windows
                .iter()
                .find(|w| &w.label == label)
                .and_then(|w| w.xy),
        );
    }
    let residual_rows: Vec<&Vec<f64>> = residuals.iter().collect();
    let residual_mean = column_means(&residual_rows);
    let squared: Vec<Vec<f64>> = residuals
        .iter()
        .map(|row| {
            row.iter()
                .zip(&residual_mean)
                .map(|(v, m)| (v - m) * (v - m))
                .collect()
        })
        .collect();
    let spread = column_means(&squared.iter().collect::<Vec<_>>())
        .iter()
        .map(|v| v.sqrt().max(0.5))
        .collect();
    SpotModel {
        labels: labels.to_vec(),
        means,
        spread,
        xy,
    }
}

fn spot_posterior(model: &SpotModel, level: &[f64]) -> Vec<f64> {
    let live_links = level.iter().filter(|v| v.is_finite()).count().max(1) as f64;
    let log_likelihood: Vec<f64> = model
        .means
        .iter()
        .map(|mean| {
            let squared_z = level
                .iter()
                .zip(mean)
                .zip(&model.spread)
                .map(|((m, mu), sd)| ((m - mu) / sd).powi(2))
                .fold(0.0, |acc, z| if z.is_nan() { acc } else { acc + z });
            -0.5 * squared_z / live_links * LIKELIHOOD_TEMPER
        })
        .collect();
    let best = log_likelihood
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_likelihood.iter().map(|ll| (ll - best).exp()).collect();
    let total = weights.iter().fold(0.0, |acc, w| acc + w);
    weights.iter().map(|w| w / total).collect()
}

/// Probability-weighted position over the spots that have one, and the most likely of those spots.
fn expected_position(model: &SpotModel, probabilities: &[f64]) -> Option<([f64; 2], usize)> {
    let placed: Vec<usize> = (0..model.xy.len())
        .filter(|&i| model.xy[i].is_some())
        .collect();
    if placed.is_empty() {
        return None;
    }
    let total = placed.iter().fold(0.0, |acc, &i| acc + probabilities[i]);
    let weights: Vec<f64> = placed.iter().map(|&i| probabilities[i] / total).collect();
    let mut xy = [0.0, 0.0];
    for (w, &i) in weights.iter().zip(&placed) {
        let spot = model.xy[i].unwrap();
        xy[0] += w * spot[0];
        xy[1] += w * spot[1];
    }
    Some((xy, placed[crate::numeric::argmax(&weights)]))
}

/// Honest accuracy: RSSI drifts over minutes, so test on each spot's last recording round (or its last third in
/// time when every spot has only one round, flagged by `rounds: false` as optimistic).
fn cross_validate(windows: &[&CalibrationWindow], labels: &[String]) -> Option<CrossValidation> {
    let sessions = |label: &String| {
        let mut s: Vec<i64> = windows
            .iter()
            .filter(|w| &w.label == label)
            .map(|w| w.sess)
            .collect();
        s.sort_unstable();
        s.dedup();
        s
    };
    let rounds = labels.iter().all(|l| sessions(l).len() >= 2);
    let (mut train, mut test): (Vec<&CalibrationWindow>, Vec<&CalibrationWindow>) =
        (vec![], vec![]);
    for label in labels {
        let mut spot: Vec<&CalibrationWindow> = windows
            .iter()
            .copied()
            .filter(|w| &w.label == label)
            .collect();
        spot.sort_by(|a, b| a.t.total_cmp(&b.t));
        if rounds {
            let last = *sessions(label).last().unwrap();
            train.extend(spot.iter().filter(|w| w.sess != last));
            test.extend(spot.iter().filter(|w| w.sess == last));
        } else {
            let split = spot.len() * 2 / 3;
            train.extend(&spot[..split]);
            test.extend(&spot[split..]);
        }
    }
    let trained_labels = labels
        .iter()
        .filter(|l| train.iter().any(|w| &&w.label == l))
        .count();
    if test.is_empty() || trained_labels < labels.len() {
        return None;
    }
    let model = fit_spots(&train, labels);
    let mut correct = 0;
    let mut errors = Vec::new();
    for w in &test {
        let posterior = spot_posterior(&model, &w.m);
        correct += (model.labels[crate::numeric::argmax(&posterior)] == w.label) as usize;
        if let (Some(truth), Some((xy, _))) = (w.xy, expected_position(&model, &posterior)) {
            errors.push((xy[0] - truth[0]).hypot(xy[1] - truth[1]));
        }
    }
    Some(CrossValidation {
        acc: correct as f64 / test.len() as f64,
        n: test.len(),
        loc_err_m: (!errors.is_empty()).then(|| median(&errors)),
        rounds,
    })
}
