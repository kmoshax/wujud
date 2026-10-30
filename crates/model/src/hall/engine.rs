use std::collections::VecDeque;

use ndarray::{stack, Array2, Array3, ArrayView2, Axis};
use serde::Serialize;

use super::seat_features::{
    distance_bins, link_features, scene_vector, seat_descriptor, SceneEvidence,
};
use super::{HallLayout, SeatModel};
use crate::numeric::round_to;
use crate::scene::Scene;
use crate::Result;

const MAX_WINDOWS: usize = 60;
const MIN_WINDOWS: usize = 10;
const HISTORY_LEN: usize = 900;
const COUNT_SMOOTHING: f64 = 0.3;

#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct HallReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truth: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub true_count: Option<i64>,
}

/// `[t, count, true_count]`.
pub type HallHistoryPoint = (f64, Option<f64>, Option<i64>);

/// What `live.py::HallEngine.push` publishes to the dashboard.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HallState {
    pub mode: String,
    pub t: f64,
    pub scene: Scene,
    pub hall: HallReport,
    pub link_activity: Vec<f64>,
    pub alive: usize,
    pub history: Vec<HallHistoryPoint>,
}

/// Every second: the last minute of link windows vs the empty-hall calibration -> seat probabilities and a count.
pub struct HallEngine {
    layout: HallLayout,
    chests: Vec<[f64; 3]>,
    bins: Array2<Option<usize>>,
    model: SeatModel,
    calibration_mean: Array3<f32>,
    calibration_std: Array3<f32>,
    mode: String,
    window_means: VecDeque<Array2<f32>>,
    window_stds: VecDeque<Array2<f32>>,
    smoothed_count: Option<f64>,
    history: VecDeque<HallHistoryPoint>,
    scene: Scene,
}

impl HallEngine {
    /// Calibration arrays are `[links, windows, 13]` recorded in the empty hall.
    pub fn new(
        layout: HallLayout,
        model: SeatModel,
        calibration_mean: Array3<f32>,
        calibration_std: Array3<f32>,
        mode: &str,
    ) -> Self {
        let chests = layout.chest_points();
        HallEngine {
            bins: distance_bins(&layout.nodes, &layout.links, &chests),
            scene: layout.scene(),
            chests,
            layout,
            model,
            calibration_mean,
            calibration_std,
            mode: mode.into(),
            window_means: VecDeque::new(),
            window_stds: VecDeque::new(),
            smoothed_count: None,
            history: VecDeque::new(),
        }
    }

    pub fn evidence(&self) -> Option<SceneEvidence> {
        if self.window_means.len() < MIN_WINDOWS {
            return None;
        }
        let stacked = |windows: &VecDeque<Array2<f32>>| {
            let views: Vec<ArrayView2<f32>> = windows.iter().map(|w| w.view()).collect();
            stack(Axis(1), &views).unwrap()
        };
        let (session_mean, session_std) = (stacked(&self.window_means), stacked(&self.window_stds));
        let features = link_features(
            self.calibration_mean.view(),
            self.calibration_std.view(),
            session_mean.view(),
            session_std.view(),
        );
        Some(SceneEvidence {
            seat_descriptor: seat_descriptor(
                &features,
                &self.bins,
                &self.layout.nodes,
                &self.chests,
            ),
            scene_vector: scene_vector(
                &features,
                &self.layout.nodes,
                &self.layout.links,
                self.chests.len(),
                self.layout.width,
                self.layout.depth,
            ),
            link_features: features,
        })
    }

    /// One 1-s window `[links, 13]` (from `Windower::tick`); `truth` per seat in a simulation.
    pub fn push(
        &mut self,
        t_seconds: f64,
        mean: Array2<f32>,
        std: Array2<f32>,
        truth: Option<&[i64]>,
    ) -> Result<HallState> {
        let alive = mean.column(0).iter().filter(|v| v.is_finite()).count();
        let links = mean.nrows();
        if self.window_means.len() == MAX_WINDOWS {
            self.window_means.pop_front();
            self.window_stds.pop_front();
        }
        self.window_means.push_back(mean);
        self.window_stds.push_back(std);
        let mut report = HallReport::default();
        let mut link_activity = vec![0.0; links];
        if let Some(evidence) = self.evidence() {
            let prediction = self.model.predict(&evidence, &self.bins, &self.chests)?;
            let count = match self.smoothed_count {
                None => prediction.count,
                Some(previous) => {
                    (1.0 - COUNT_SMOOTHING) * previous + COUNT_SMOOTHING * prediction.count
                }
            };
            self.smoothed_count = Some(count);
            report.count = Some(round_to(count, 1));
            report.p = Some(
                prediction
                    .occupied
                    .iter()
                    .map(|p| round_to(*p, 3))
                    .collect(),
            );
            link_activity = activity(&evidence);
        }
        if let Some(truth) = truth {
            report.truth = Some(truth.to_vec());
            report.true_count = Some(truth.iter().sum());
        }
        let t = round_to(t_seconds, 1);
        if self.history.len() == HISTORY_LEN {
            self.history.pop_front();
        }
        self.history.push_back((t, report.count, report.true_count));
        Ok(HallState {
            mode: self.mode.clone(),
            t,
            scene: self.scene.clone(),
            hall: report,
            link_activity,
            alive,
            history: self.history.iter().copied().collect(),
        })
    }
}

/// How much each link is shadowed or moving, 0..1, for drawing; float32 like NumPy's, rounded to 0.01.
fn activity(evidence: &SceneEvidence) -> Vec<f64> {
    let zero_nan = |v: f64| if v.is_nan() { 0.0 } else { v as f32 };
    let shadow = evidence.link_features.get("sh_mean");
    let motion = evidence.link_features.get("mo_mean");
    shadow
        .iter()
        .zip(motion)
        .map(|(&s, &m)| {
            let level = (zero_nan(-s) / 6.0 + zero_nan(m) / 1.5).clamp(0.0, 1.0);
            ((level * 100.0).round_ties_even() / 100.0) as f64
        })
        .collect()
}
