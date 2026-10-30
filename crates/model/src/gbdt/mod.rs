//! Gradient-boosted trees: inference of sklearn `HistGradientBoosting*` models exported to JSON, and training.

mod binning;
mod grower;
mod model_file;
mod train;

use std::path::Path;

use ndarray::{Array2, ArrayView2};

use crate::{ModelError, Result};
pub use model_file::{FORMAT, SEAT_MODEL_FORMAT};
pub use train::{train_classifier, train_regressor, EarlyStopping, TrainOptions};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    Multinomial,
    Binomial,
    SquaredError,
}

/// One regression tree. Split nodes and leaves live in separate arrays; a child `c >= 0` is split node `c`,
/// `c < 0` is leaf `-1 - c`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    pub feature: Vec<usize>,
    pub threshold: Vec<f64>,
    pub missing_goes_left: Vec<bool>,
    pub left: Vec<i32>,
    pub right: Vec<i32>,
    pub leaf_value: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Node {
    Split(usize),
    Leaf(usize),
}

fn node_of(child: i32) -> Node {
    if child >= 0 {
        Node::Split(child as usize)
    } else {
        Node::Leaf((-1 - child) as usize)
    }
}

impl Tree {
    pub fn single_leaf(value: f64) -> Self {
        Tree {
            feature: vec![],
            threshold: vec![],
            missing_goes_left: vec![],
            left: vec![],
            right: vec![],
            leaf_value: vec![value],
        }
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        let mut node = if self.feature.is_empty() {
            Node::Leaf(0)
        } else {
            Node::Split(0)
        };
        loop {
            match node {
                Node::Leaf(leaf) => return self.leaf_value[leaf],
                Node::Split(split) => node = node_of(self.child(split, x[self.feature[split]])),
            }
        }
    }

    fn child(&self, split: usize, value: f64) -> i32 {
        let goes_left = if value.is_nan() {
            self.missing_goes_left[split]
        } else {
            value <= self.threshold[split]
        };
        if goes_left {
            self.left[split]
        } else {
            self.right[split]
        }
    }

    fn validate(&self, n_features: usize) -> Result<()> {
        let splits = self.feature.len();
        let same_lengths = [
            self.threshold.len(),
            self.missing_goes_left.len(),
            self.left.len(),
            self.right.len(),
        ]
        .iter()
        .all(|&len| len == splits);
        let child_ok = |c: &i32| match node_of(*c) {
            Node::Split(s) => s < splits && s > 0,
            Node::Leaf(l) => l < self.leaf_value.len(),
        };
        let valid = same_lengths
            && !self.leaf_value.is_empty()
            && self.feature.iter().all(|&f| f < n_features)
            && self.left.iter().chain(&self.right).all(child_ok);
        if valid {
            Ok(())
        } else {
            Err(ModelError::InvalidModel("malformed tree".into()))
        }
    }
}

/// A boosted ensemble: `raw[k] = baseline[k] + sum of tree k of every iteration`.
#[derive(Debug, Clone, PartialEq)]
pub struct Gbdt {
    pub loss: Loss,
    pub n_features: usize,
    pub classes: Vec<i64>,
    pub baseline: Vec<f64>,
    pub iterations: Vec<Vec<Tree>>,
}

impl Gbdt {
    pub fn load(path: &Path) -> Result<Self> {
        model_file::read_gbdt(path)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        model_file::parse_gbdt(text)
    }

    pub fn to_json(&self) -> String {
        model_file::gbdt_to_json(self, "wujud-model")
    }

    pub fn trees_per_iteration(&self) -> usize {
        self.baseline.len()
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let k = self.trees_per_iteration();
        let shape_ok = match self.loss {
            Loss::Multinomial => k == self.classes.len() && k > 2,
            Loss::Binomial => k == 1 && self.classes.len() == 2,
            Loss::SquaredError => k == 1 && self.classes.is_empty(),
        };
        if !shape_ok || self.iterations.iter().any(|trees| trees.len() != k) {
            return Err(ModelError::InvalidModel(
                "baseline, classes and trees per iteration disagree".into(),
            ));
        }
        self.iterations
            .iter()
            .flatten()
            .try_for_each(|tree| tree.validate(self.n_features))
    }

    fn check_features(&self, x: &[f64]) -> Result<()> {
        if x.len() == self.n_features {
            Ok(())
        } else {
            Err(ModelError::FeatureCount {
                expected: self.n_features,
                got: x.len(),
            })
        }
    }

    /// sklearn's `_raw_predict`: one score per tree of an iteration.
    pub fn raw_scores(&self, x: &[f64]) -> Result<Vec<f64>> {
        self.check_features(x)?;
        let mut raw = self.baseline.clone();
        for trees in &self.iterations {
            for (score, tree) in raw.iter_mut().zip(trees) {
                *score += tree.predict(x);
            }
        }
        Ok(raw)
    }

    /// Class probabilities in the order of `classes` (`predict_proba`); a regressor returns its single value.
    pub fn predict_proba(&self, x: &[f64]) -> Result<Vec<f64>> {
        let raw = self.raw_scores(x)?;
        Ok(match self.loss {
            Loss::Multinomial => softmax(&raw),
            Loss::Binomial => {
                let p = sigmoid(raw[0]);
                vec![1.0 - p, p]
            }
            Loss::SquaredError => raw,
        })
    }

    pub fn predict_proba_rows(&self, rows: ArrayView2<f64>) -> Result<Array2<f64>> {
        let width = if self.loss == Loss::SquaredError {
            1
        } else {
            self.classes.len()
        };
        let mut out = Array2::zeros((rows.nrows(), width));
        for (i, row) in rows.rows().into_iter().enumerate() {
            let proba = self.predict_proba(&row.to_vec())?;
            out.row_mut(i).assign(&ndarray::Array1::from(proba));
        }
        Ok(out)
    }

    /// The regression value (`predict` of a `HistGradientBoostingRegressor`).
    pub fn predict_value(&self, x: &[f64]) -> Result<f64> {
        Ok(self.raw_scores(x)?[0])
    }

    /// `predict` of a classifier: the most probable class label.
    pub fn predict_class(&self, x: &[f64]) -> Result<i64> {
        let proba = self.predict_proba(x)?;
        Ok(self.classes[crate::numeric::argmax(&proba)])
    }
}

pub fn softmax(raw: &[f64]) -> Vec<f64> {
    let max = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = raw.iter().map(|r| (r - max).exp()).collect();
    let total: f64 = exp.iter().sum();
    exp.iter().map(|e| e / total).collect()
}

pub fn sigmoid(raw: f64) -> f64 {
    1.0 / (1.0 + (-raw).exp())
}

/// Two-stage seat model of the hall (`research/wujud/model.py::SeatModel`), as stored in `hall3d.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct SeatModelTrees {
    pub stage1: Gbdt,
    pub stage2: Gbdt,
    pub count: Gbdt,
}

impl SeatModelTrees {
    pub fn load(path: &Path) -> Result<Self> {
        model_file::read_seat_model(path)
    }
}
