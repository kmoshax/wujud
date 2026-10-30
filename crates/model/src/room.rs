//! Room engine (`research/live.py::RoomEngine`): per-packet CSI of router -> board links in, a state every second out.

use std::collections::VecDeque;

use ndarray::{Array2, ArrayView2};
use serde::Serialize;

use crate::features::{amplitude_from_iq, profile_windows, windows, GLRT_COLUMN_WITH_RSSI};
use crate::gbdt::Gbdt;
use crate::numeric::{argmax, round_to};
use crate::occupancy::{LatchConfig, Occupancy, OccupancyLatch};
use crate::roomloc::locate;
use crate::scene::Scene;
use crate::Result;

const BUFFER_SECONDS: f64 = 32.0;
const MIN_PACKETS: usize = 50;
const MIN_SPAN_SECONDS: f64 = 18.0;
const PROFILE_SPAN_SECONDS: f64 = 20.5;
const BREATHING_SPAN_SECONDS: f64 = 29.0;
const LATCH_THRESHOLD: f64 = 0.4;
const LATCH_EVERY_TICKS: u64 = 5;
const HISTORY_LEN: usize = 900;
/// GLRT > 1.8 fires on ~8 % of real empty 20-s windows (ESPectre), hence 10 s in a row and 25 s calm.
const BREATHING_OVERRIDE_GLRT: f64 = 1.8;
const BREATHING_OVERRIDE_RUN: u32 = 10;
const CALM_SECONDS_FOR_BREATHING: u32 = 25;
const BREATHING_REPORT_GLRT: f64 = 1.4;
const POSE_CONFIDENCE: f64 = 0.55;

/// The model file `research/live.py::models` picks for a room with `links` boards.
pub fn activity_model_file(links: usize) -> &'static str {
    if links < 3 {
        "room_activity_esp1_random.json"
    } else {
        "room_activity_esp3_random.json"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RoomState3 {
    Empty,
    Still,
    Moving,
}

/// `state`: the 3-state decision, or for a confident still person the pose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Pose {
    Empty,
    Still,
    Moving,
    Lie,
    Sit,
    Stand,
}

impl From<RoomState3> for Pose {
    fn from(state: RoomState3) -> Self {
        match state {
            RoomState3::Empty => Pose::Empty,
            RoomState3::Still => Pose::Still,
            RoomState3::Moving => Pose::Moving,
        }
    }
}

const STILL_POSES: [Pose; 3] = [Pose::Lie, Pose::Sit, Pose::Stand];

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LatchReport {
    pub distance: Option<f64>,
    pub thr: f64,
    pub reference_windows: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RoomDecision {
    pub state: Pose,
    pub state3: RoomState3,
    /// Smoothed probabilities of away, lie, sit, stand, walk.
    pub probs: Vec<f64>,
    pub latch: LatchReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub breath_bpm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xy: Option<[f64; 2]>,
}

/// Ground truth of a simulation, shown next to the decision.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RoomTruth {
    pub truth: String,
    pub truth_bpm: Option<f64>,
    pub truth_xy: Option<[f64; 2]>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct RoomReport {
    #[serde(flatten)]
    pub decision: Option<RoomDecision>,
    #[serde(flatten)]
    pub truth: Option<RoomTruth>,
}

/// `[t, breath_bpm, truth_bpm]`.
pub type RoomHistoryPoint = (f64, Option<f64>, Option<f64>);

/// What `live.py::publish` sends to the dashboard every second.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RoomState {
    pub mode: String,
    pub t: f64,
    pub scene: Scene,
    pub room: RoomReport,
    pub link_activity: Vec<f64>,
    pub alive: usize,
    pub history: Vec<RoomHistoryPoint>,
}

struct Packet {
    time: f64,
    iq: Vec<i8>,
    rssi: f64,
}

/// One link's newest windows: activity features, static profile, breathing GLRT and rate.
struct LinkWindows {
    features: Option<Vec<f64>>,
    profile: Option<Vec<f64>>,
    breathing: Option<(f64, f64)>,
}

pub struct RoomEngine {
    scene: Scene,
    mode: String,
    links: Vec<(usize, usize)>,
    nodes: Vec<[f64; 3]>,
    model: Gbdt,
    buffers: Vec<VecDeque<Packet>>,
    history: VecDeque<RoomHistoryPoint>,
    first_tick: Option<f64>,
    smoothed: Option<Vec<f64>>,
    pub locate_person: bool,
    latch: OccupancyLatch,
    latch_state: Option<Occupancy>,
    latch_distance: Option<f64>,
    profile_width: Option<usize>,
    ticks: u64,
    breathing_run: u32,
    calm_for: u32,
}

impl RoomEngine {
    /// `links[k]` are the (router, board) node indices of link `k`; `model` from `activity_model_file`.
    pub fn new(
        scene: Scene,
        links: Vec<(usize, usize)>,
        mode: &str,
        model: Gbdt,
        nodes: Vec<[f64; 3]>,
    ) -> Self {
        RoomEngine {
            scene,
            mode: mode.into(),
            buffers: links.iter().map(|_| VecDeque::new()).collect(),
            links,
            nodes,
            model,
            history: VecDeque::new(),
            first_tick: None,
            smoothed: None,
            locate_person: false,
            latch: new_latch(),
            latch_state: None,
            latch_distance: None,
            profile_width: None,
            ticks: 0,
            breathing_run: 0,
            calm_for: 0,
        }
    }

    /// One CSI packet of link `link`: `iq` interleaved bytes as the ESP32 sends them, `rssi` in dBm.
    pub fn add(&mut self, link: usize, t_seconds: f64, iq: &[i8], rssi: f64) {
        let buffer = &mut self.buffers[link];
        if buffer.back().is_some_and(|p| p.iq.len() != iq.len()) {
            buffer.clear();
        }
        buffer.push_back(Packet {
            time: t_seconds,
            iq: iq.to_vec(),
            rssi,
        });
        while buffer
            .front()
            .is_some_and(|p| p.time < t_seconds - BUFFER_SECONDS)
        {
            buffer.pop_front();
        }
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// The floor-plan ends of every link, for `MotionTracker`.
    pub fn link_ends(&self) -> Vec<([f64; 2], [f64; 2])> {
        let plan = |node: usize| [self.nodes[node][0], self.nodes[node][1]];
        self.links
            .iter()
            .map(|&(a, b)| (plan(a), plan(b)))
            .collect()
    }

    pub fn tick(&mut self, now: f64) -> Result<RoomState> {
        self.tick_with_truth(now, None)
    }

    pub fn tick_with_truth(&mut self, now: f64, truth: Option<RoomTruth>) -> Result<RoomState> {
        let first_tick = *self.first_tick.get_or_insert(now);
        let per_link: Vec<LinkWindows> = self.buffers.iter().filter_map(link_windows).collect();
        let features: Vec<Vec<f64>> = per_link.iter().filter_map(|w| w.features.clone()).collect();
        let mut report = RoomReport {
            decision: None,
            truth,
        };
        let mut link_activity = vec![0.0; self.links.len()];
        if !features.is_empty() {
            report.decision = Some(self.decide(&features, &per_link)?);
            link_activity = features.iter().map(|row| activity_level(row)).collect();
        }
        let t = round_to(now - first_tick, 1);
        let breath = report.decision.as_ref().and_then(|d| d.breath_bpm);
        let truth_bpm = report.truth.as_ref().and_then(|t| t.truth_bpm);
        if self.history.len() == HISTORY_LEN {
            self.history.pop_front();
        }
        self.history.push_back((t, breath, truth_bpm));
        Ok(RoomState {
            mode: self.mode.clone(),
            t,
            scene: self.scene.clone(),
            room: report,
            link_activity,
            alive: features.len(),
            history: self.history.iter().copied().collect(),
        })
    }

    fn decide(&mut self, features: &[Vec<f64>], per_link: &[LinkWindows]) -> Result<RoomDecision> {
        let probabilities = self.model.predict_proba(&aggregate_over_links(features))?;
        let smoothed = match &self.smoothed {
            None => probabilities,
            Some(previous) => previous
                .iter()
                .zip(&probabilities)
                .map(|(s, p)| 0.6 * s + 0.4 * p)
                .collect(),
        };
        self.smoothed = Some(smoothed.clone());
        let p3 = [
            smoothed[0],
            smoothed[1] + smoothed[2] + smoothed[3],
            smoothed[4],
        ];
        let mut state3 = [RoomState3::Empty, RoomState3::Still, RoomState3::Moving][argmax(&p3)];

        let breathing: Vec<(f64, f64)> = per_link.iter().filter_map(|w| w.breathing).collect();
        let strongest_glrt = breathing
            .iter()
            .map(|b| b.0)
            .fold(f64::NEG_INFINITY, f64::max);
        self.breathing_run = if strongest_glrt > BREATHING_OVERRIDE_GLRT {
            self.breathing_run + 1
        } else {
            0
        };
        self.calm_for = if state3 != RoomState3::Moving {
            self.calm_for + 1
        } else {
            0
        };
        if state3 == RoomState3::Empty
            && self.breathing_run >= BREATHING_OVERRIDE_RUN
            && self.calm_for >= CALM_SECONDS_FOR_BREATHING
        {
            state3 = RoomState3::Still;
        }

        self.ticks += 1;
        let profiles: Vec<&Vec<f64>> = per_link.iter().filter_map(|w| w.profile.as_ref()).collect();
        if !profiles.is_empty() && profiles.len() == features.len() {
            self.step_latch(&profiles, &p3);
        }
        if state3 != RoomState3::Moving && self.latch_distance.is_some() {
            state3 = match self.latch_state {
                Some(Occupancy::Still) => RoomState3::Still,
                Some(Occupancy::Empty) => RoomState3::Empty,
                _ => state3,
            };
        }

        Ok(RoomDecision {
            state: pose(state3, &smoothed, p3[1]),
            state3,
            probs: smoothed.iter().map(|p| round_to(*p, 3)).collect(),
            latch: LatchReport {
                distance: self.latch_distance.map(|d| round_to(d, 2)),
                thr: self.latch.config.threshold,
                reference_windows: self.latch.reference_len(),
            },
            breath_bpm: self.breathing_rate(state3, &breathing),
            xy: self.position(state3, features),
        })
    }

    /// Every 5 s (the latch counts in 5-s windows); a link that came or went starts a new reference.
    fn step_latch(&mut self, profiles: &[&Vec<f64>], p3: &[f64; 3]) {
        let profile: Vec<f64> = profiles.iter().flat_map(|p| p.iter().copied()).collect();
        if self.profile_width != Some(profile.len()) {
            self.latch = new_latch();
            self.profile_width = Some(profile.len());
        }
        if self.ticks.is_multiple_of(LATCH_EVERY_TICKS) {
            let p_still = p3[1] / (p3[0] + p3[1]).max(1e-9);
            let (state, distance) = self.latch.step(&profile, p3[2], Some(p_still));
            self.latch_state = Some(state);
            self.latch_distance = distance;
        }
    }

    fn breathing_rate(&self, state3: RoomState3, breathing: &[(f64, f64)]) -> Option<f64> {
        if state3 != RoomState3::Still
            || self.calm_for < CALM_SECONDS_FOR_BREATHING
            || breathing.is_empty()
        {
            return None;
        }
        let glrts: Vec<f64> = breathing.iter().map(|b| b.0).collect();
        let (glrt, rate) = breathing[argmax(&glrts)];
        (glrt > BREATHING_REPORT_GLRT && (6.0..=36.0).contains(&rate)).then(|| round_to(rate, 1))
    }

    /// Off by default: without an empty-room calibration it is no better than guessing the centre (README).
    fn position(&self, state3: RoomState3, features: &[Vec<f64>]) -> Option<[f64; 2]> {
        if !self.locate_person || features.len() < 3 || state3 == RoomState3::Empty {
            return None;
        }
        let xy = locate(
            features,
            &self.links,
            &self.nodes,
            self.scene.width,
            self.scene.depth,
        );
        Some(xy.map(|v| round_to(v, 2)))
    }
}

fn new_latch() -> OccupancyLatch {
    OccupancyLatch::new(LatchConfig {
        threshold: LATCH_THRESHOLD,
        ..LatchConfig::default()
    })
}

fn pose(state3: RoomState3, smoothed: &[f64], p_still: f64) -> Pose {
    if state3 != RoomState3::Still {
        return state3.into();
    }
    let best = argmax(&smoothed[1..4]);
    if smoothed[1 + best] / p_still.max(1e-9) >= POSE_CONFIDENCE {
        STILL_POSES[best]
    } else {
        Pose::Still
    }
}

fn activity_level(row: &[f64]) -> f64 {
    round_to((row[3].max(row[1]) / 1.5).clamp(0.0, 1.0), 2)
}

/// `[mean, max, min]` of every feature over the links.
fn aggregate_over_links(rows: &[Vec<f64>]) -> Vec<f64> {
    let columns = rows[0].len();
    let column = |c: usize| rows.iter().map(move |r| r[c]);
    let mean = (0..columns).map(|c| column(c).sum::<f64>() / rows.len() as f64);
    let max = (0..columns).map(|c| column(c).fold(f64::NEG_INFINITY, f64::max));
    let min = (0..columns).map(|c| column(c).fold(f64::INFINITY, f64::min));
    mean.chain(max).chain(min).collect()
}

fn link_windows(buffer: &VecDeque<Packet>) -> Option<LinkWindows> {
    let (first, last) = (buffer.front()?, buffer.back()?);
    let span = last.time - first.time;
    if buffer.len() < MIN_PACKETS || span < MIN_SPAN_SECONDS {
        return None;
    }
    let times: Vec<f64> = buffer.iter().map(|p| p.time - first.time).collect();
    let rssi: Vec<f64> = buffer.iter().map(|p| p.rssi).collect();
    let iq = stacked_iq(buffer);
    let amplitude = amplitude_from_iq(iq.view()).mapv(f64::from);
    let newest_window =
        |seconds: f64| windows(&times, amplitude.view(), Some(&rssi), seconds, 100.0);
    let activity = newest_window((span - 0.5).min(20.0));
    let profile = (span >= PROFILE_SPAN_SECONDS)
        .then(|| profile_windows(&times, amplitude.view(), 20.0, 100.0).pop())
        .flatten();
    let breathing = (span >= BREATHING_SPAN_SECONDS)
        .then(|| {
            let long = newest_window((span - 0.5).min(30.0));
            let glrt = long.rows.last()?[GLRT_COLUMN_WITH_RSSI];
            Some((glrt, *long.breathing_rates.last()?))
        })
        .flatten();
    Some(LinkWindows {
        features: activity.rows.last().cloned(),
        profile,
        breathing,
    })
}

fn stacked_iq(buffer: &VecDeque<Packet>) -> Array2<i8> {
    let width = buffer.front().map_or(0, |p| p.iq.len());
    let flat: Vec<i8> = buffer.iter().flat_map(|p| p.iq.iter().copied()).collect();
    ArrayView2::from_shape((buffer.len(), width), &flat)
        .unwrap()
        .to_owned()
}
