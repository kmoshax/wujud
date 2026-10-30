//! The JSON layout written by `research/tools/export_models.py` (see the crate README).

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{Gbdt, Loss, SeatModelTrees, Tree};
use crate::{ModelError, Result};

pub const FORMAT: &str = "wujud-gbdt-v2";
pub const SEAT_MODEL_FORMAT: &str = "wujud-seat-model-v1";

#[derive(Serialize, Deserialize)]
struct TreeFile {
    feature: Vec<usize>,
    threshold: Vec<Option<f64>>,
    missing_left: Vec<u8>,
    left: Vec<i32>,
    right: Vec<i32>,
    leaf_value: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct GbdtFile {
    format: String,
    source: String,
    loss: String,
    n_features: usize,
    classes: Option<Vec<i64>>,
    baseline: Vec<f64>,
    trees_per_iteration: usize,
    iterations: Vec<Vec<TreeFile>>,
}

#[derive(Deserialize)]
struct SeatModelFile {
    format: String,
    stage1: GbdtFile,
    stage2: GbdtFile,
    count: GbdtFile,
}

fn read_text(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|source| ModelError::Io {
        path: path.into(),
        source,
    })
}

fn parse<T: for<'de> Deserialize<'de>>(text: &str, path: &Path) -> Result<T> {
    serde_json::from_str(text).map_err(|source| ModelError::Json {
        path: path.into(),
        source,
    })
}

pub(super) fn read_gbdt(path: &Path) -> Result<Gbdt> {
    gbdt_from_file(parse(&read_text(path)?, path)?)
}

pub(super) fn parse_gbdt(text: &str) -> Result<Gbdt> {
    gbdt_from_file(parse(text, Path::new("<string>"))?)
}

pub(super) fn read_seat_model(path: &Path) -> Result<SeatModelTrees> {
    let file: SeatModelFile = parse(&read_text(path)?, path)?;
    if file.format != SEAT_MODEL_FORMAT {
        return Err(ModelError::InvalidModel(format!(
            "format {} is not {SEAT_MODEL_FORMAT}",
            file.format
        )));
    }
    Ok(SeatModelTrees {
        stage1: gbdt_from_file(file.stage1)?,
        stage2: gbdt_from_file(file.stage2)?,
        count: gbdt_from_file(file.count)?,
    })
}

fn loss_from_name(name: &str) -> Result<Loss> {
    match name {
        "multinomial" => Ok(Loss::Multinomial),
        "binomial" => Ok(Loss::Binomial),
        "squared_error" => Ok(Loss::SquaredError),
        other => Err(ModelError::InvalidModel(format!("unknown loss {other}"))),
    }
}

fn loss_name(loss: Loss) -> &'static str {
    match loss {
        Loss::Multinomial => "multinomial",
        Loss::Binomial => "binomial",
        Loss::SquaredError => "squared_error",
    }
}

fn gbdt_from_file(file: GbdtFile) -> Result<Gbdt> {
    if file.format != FORMAT {
        return Err(ModelError::InvalidModel(format!(
            "format {} is not {FORMAT}",
            file.format
        )));
    }
    if file.trees_per_iteration != file.baseline.len() {
        return Err(ModelError::InvalidModel(
            "trees_per_iteration != baseline length".into(),
        ));
    }
    let model = Gbdt {
        loss: loss_from_name(&file.loss)?,
        n_features: file.n_features,
        classes: file.classes.unwrap_or_default(),
        baseline: file.baseline,
        iterations: file
            .iterations
            .into_iter()
            .map(|trees| trees.into_iter().map(tree_from_file).collect())
            .collect(),
    };
    model.validate()?;
    Ok(model)
}

fn tree_from_file(file: TreeFile) -> Tree {
    Tree {
        feature: file.feature,
        threshold: file
            .threshold
            .iter()
            .map(|t| t.unwrap_or(f64::INFINITY))
            .collect(),
        missing_goes_left: file.missing_left.iter().map(|&m| m != 0).collect(),
        left: file.left,
        right: file.right,
        leaf_value: file.leaf_value,
    }
}

fn tree_to_file(tree: &Tree) -> TreeFile {
    TreeFile {
        feature: tree.feature.clone(),
        threshold: tree
            .threshold
            .iter()
            .map(|&t| t.is_finite().then_some(t))
            .collect(),
        missing_left: tree.missing_goes_left.iter().map(|&m| m as u8).collect(),
        left: tree.left.clone(),
        right: tree.right.clone(),
        leaf_value: tree.leaf_value.clone(),
    }
}

pub(super) fn gbdt_to_json(model: &Gbdt, source: &str) -> String {
    let file = GbdtFile {
        format: FORMAT.into(),
        source: source.into(),
        loss: loss_name(model.loss).into(),
        n_features: model.n_features,
        classes: (model.loss != Loss::SquaredError).then(|| model.classes.clone()),
        baseline: model.baseline.clone(),
        trees_per_iteration: model.trees_per_iteration(),
        iterations: model
            .iterations
            .iter()
            .map(|trees| trees.iter().map(tree_to_file).collect())
            .collect(),
    };
    serde_json::to_string(&file).expect("plain numbers and strings always serialize")
}
