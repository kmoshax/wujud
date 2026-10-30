use std::path::Path;

use ndarray::{concatenate, Array2, Axis};

use super::seat_features::{context_features, SceneEvidence};
use crate::gbdt::SeatModelTrees;
use crate::Result;

/// Two-stage seat classifier plus head-count regressor (`research/hallsense/model.py::SeatModel.predict`).
pub struct SeatModel {
    pub trees: SeatModelTrees,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SeatPrediction {
    pub stage1: Vec<f64>,
    pub occupied: Vec<f64>,
    pub count_sum: f64,
    pub count_global: f64,
    /// Mean of the two counts.
    pub count: f64,
}

fn positive_class(model: &crate::gbdt::Gbdt, rows: &Array2<f32>) -> Result<Vec<f64>> {
    rows.rows()
        .into_iter()
        .map(|row| Ok(model.predict_proba(&row.iter().map(|&v| v as f64).collect::<Vec<_>>())?[1]))
        .collect()
}

impl SeatModel {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(SeatModel {
            trees: SeatModelTrees::load(path)?,
        })
    }

    pub fn predict(
        &self,
        evidence: &SceneEvidence,
        bins: &Array2<Option<usize>>,
        chests: &[[f64; 3]],
    ) -> Result<SeatPrediction> {
        let stage1 = positive_class(&self.trees.stage1, &evidence.seat_descriptor)?;
        let context = context_features(&stage1, bins, chests);
        let with_context =
            concatenate(Axis(1), &[evidence.seat_descriptor.view(), context.view()]).unwrap();
        let occupied = positive_class(&self.trees.stage2, &with_context)?;
        let scene: Vec<f64> = evidence
            .scene_vector
            .iter()
            .map(|&v| nan_to_num(v) as f64)
            .collect();
        let count_global = self
            .trees
            .count
            .predict_value(&scene)?
            .clamp(0.0, chests.len() as f64);
        let count_sum: f64 = occupied.iter().sum();
        Ok(SeatPrediction {
            stage1,
            occupied,
            count_sum,
            count_global,
            count: 0.5 * (count_sum + count_global),
        })
    }
}

/// `np.nan_to_num` for float32: NaN -> 0, +-inf -> +-f32::MAX.
fn nan_to_num(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else {
        v.clamp(f32::MIN, f32::MAX)
    }
}
