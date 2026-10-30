//! Native training of histogram gradient-boosted trees, following sklearn's `HistGradientBoosting*`.

use ndarray::{ArrayView2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::binning::BinMapper;
use super::grower::{grow_tree, GrowerOptions, TreeData};
use super::{sigmoid, softmax, Gbdt, Loss, Tree};
use crate::{ModelError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStopping {
    /// On when there are more than 10 000 samples (sklearn's default).
    Auto,
    On,
    Off,
}

/// sklearn's hyper-parameters and defaults; `seed` drives the validation split.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainOptions {
    pub max_iter: usize,
    pub learning_rate: f64,
    pub max_leaf_nodes: usize,
    pub min_samples_leaf: usize,
    pub l2_regularization: f64,
    pub balanced_class_weight: bool,
    pub early_stopping: EarlyStopping,
    pub validation_fraction: f64,
    pub n_iter_no_change: usize,
    pub tol: f64,
    pub seed: u64,
}

impl Default for TrainOptions {
    fn default() -> Self {
        TrainOptions {
            max_iter: 100,
            learning_rate: 0.1,
            max_leaf_nodes: 31,
            min_samples_leaf: 20,
            l2_regularization: 0.0,
            balanced_class_weight: false,
            early_stopping: EarlyStopping::Auto,
            validation_fraction: 0.1,
            n_iter_no_change: 10,
            tol: 1e-7,
            seed: 0,
        }
    }
}

const EARLY_STOPPING_MIN_SAMPLES: usize = 10_000;

/// What is being fitted: class index per sample (classification) or a value (regression).
enum Target {
    Classes { index: Vec<usize>, count: usize },
    Values(Vec<f64>),
}

impl Target {
    fn loss(&self) -> Loss {
        match self {
            Target::Classes { count, .. } if *count > 2 => Loss::Multinomial,
            Target::Classes { .. } => Loss::Binomial,
            Target::Values(_) => Loss::SquaredError,
        }
    }

    fn trees_per_iteration(&self) -> usize {
        match self {
            Target::Classes { count, .. } if *count > 2 => *count,
            _ => 1,
        }
    }
}

/// A classifier: `y` holds any integer labels; 2 classes give a binomial model, more a multinomial one.
pub fn train_classifier(x: ArrayView2<f64>, y: &[i64], options: &TrainOptions) -> Result<Gbdt> {
    let mut classes = y.to_vec();
    classes.sort_unstable();
    classes.dedup();
    if classes.len() < 2 {
        return Err(ModelError::InvalidTrainingData(
            "need at least two classes".into(),
        ));
    }
    let index = y
        .iter()
        .map(|label| classes.binary_search(label).unwrap())
        .collect();
    let target = Target::Classes {
        index,
        count: classes.len(),
    };
    let weights = sample_weights(&target, options.balanced_class_weight);
    fit(x, &target, &weights, classes, options)
}

pub fn train_regressor(x: ArrayView2<f64>, y: &[f64], options: &TrainOptions) -> Result<Gbdt> {
    let weights = vec![1.0; y.len()];
    fit(x, &Target::Values(y.to_vec()), &weights, vec![], options)
}

/// sklearn `class_weight="balanced"`: `n_samples / (n_classes * count(class))`.
fn sample_weights(target: &Target, balanced: bool) -> Vec<f64> {
    match target {
        Target::Classes { index, count } if balanced => {
            let mut per_class = vec![0usize; *count];
            index.iter().for_each(|&c| per_class[c] += 1);
            index
                .iter()
                .map(|&c| index.len() as f64 / (*count as f64 * per_class[c] as f64))
                .collect()
        }
        _ => vec![1.0; sample_count(target)],
    }
}

fn sample_count(target: &Target) -> usize {
    match target {
        Target::Classes { index, .. } => index.len(),
        Target::Values(values) => values.len(),
    }
}

/// Stratified (per class) or plain random split into (training, validation) sample indices.
fn validation_split(target: &Target, fraction: f64, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let groups: Vec<Vec<u32>> = match target {
        Target::Classes { index, count } => (0..*count)
            .map(|c| {
                (0..index.len() as u32)
                    .filter(|&i| index[i as usize] == c)
                    .collect()
            })
            .collect(),
        Target::Values(values) => vec![(0..values.len() as u32).collect()],
    };
    let (mut train, mut validation) = (Vec::new(), Vec::new());
    for mut group in groups {
        group.shuffle(&mut rng);
        let held_out = (group.len() as f64 * fraction).round() as usize;
        validation.extend_from_slice(&group[..held_out]);
        train.extend_from_slice(&group[held_out..]);
    }
    train.sort_unstable();
    validation.sort_unstable();
    (train, validation)
}

fn baseline(target: &Target, weights: &[f64], samples: &[u32]) -> Vec<f64> {
    let total_weight: f64 = samples.iter().map(|&i| weights[i as usize]).sum();
    match target {
        Target::Classes { index, count } => {
            let eps = f64::EPSILON;
            let prior: Vec<f64> = (0..*count)
                .map(|c| {
                    let w: f64 = samples
                        .iter()
                        .filter(|&&i| index[i as usize] == c)
                        .map(|&i| weights[i as usize])
                        .sum();
                    (w / total_weight).clamp(eps, 1.0 - eps)
                })
                .collect();
            if *count == 2 {
                vec![(prior[1] / (1.0 - prior[1])).ln()]
            } else {
                let logs: Vec<f64> = prior.iter().map(|p| p.ln()).collect();
                let mean_log = logs.iter().sum::<f64>() / logs.len() as f64;
                logs.iter().map(|l| l - mean_log).collect()
            }
        }
        Target::Values(values) => {
            vec![
                samples
                    .iter()
                    .map(|&i| weights[i as usize] * values[i as usize])
                    .sum::<f64>()
                    / total_weight,
            ]
        }
    }
}

/// Weighted mean loss (sklearn's validation score is its negative).
fn mean_loss(target: &Target, raw: &[Vec<f64>], weights: &[f64], samples: &[u32]) -> f64 {
    let mut total = 0.0;
    let mut total_weight = 0.0;
    for &i in samples {
        let i = i as usize;
        let loss = match target {
            Target::Classes { index, count } if *count > 2 => {
                let max = raw[i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let log_sum: f64 = raw[i].iter().map(|r| (r - max).exp()).sum::<f64>().ln() + max;
                log_sum - raw[i][index[i]]
            }
            Target::Classes { index, .. } => {
                let r = raw[i][0];
                (1.0 + r.exp()).ln() - if index[i] == 1 { r } else { 0.0 }
            }
            Target::Values(values) => 0.5 * (raw[i][0] - values[i]).powi(2),
        };
        total += weights[i] * loss;
        total_weight += weights[i];
    }
    total / total_weight
}

/// Gradient and hessian of tree `k` for every sample.
fn gradients(target: &Target, raw: &[Vec<f64>], weights: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    raw.iter()
        .enumerate()
        .map(|(i, scores)| {
            let w = weights[i];
            match target {
                Target::Classes { index, count } if *count > 2 => {
                    let p = softmax(scores)[k];
                    let y = (index[i] == k) as u8 as f64;
                    (w * (p - y), w * p * (1.0 - p))
                }
                Target::Classes { index, .. } => {
                    let p = sigmoid(scores[0]);
                    (w * (p - index[i] as f64), w * p * (1.0 - p))
                }
                Target::Values(values) => (w * (scores[0] - values[i]), w),
            }
        })
        .unzip()
}

/// sklearn `_should_stop`: none of the last `n` scores beat the one before them by more than `tol`.
fn should_stop(scores: &[f64], n_iter_no_change: usize, tol: f64) -> bool {
    if scores.len() < n_iter_no_change + 1 {
        return false;
    }
    let reference = scores[scores.len() - n_iter_no_change - 1] + tol;
    scores[scores.len() - n_iter_no_change..]
        .iter()
        .all(|&s| s <= reference)
}

fn fit(
    x: ArrayView2<f64>,
    target: &Target,
    weights: &[f64],
    classes: Vec<i64>,
    options: &TrainOptions,
) -> Result<Gbdt> {
    let n = sample_count(target);
    if x.nrows() != n || n == 0 {
        return Err(ModelError::InvalidTrainingData(format!(
            "{} rows for {n} targets",
            x.nrows()
        )));
    }
    let early_stopping = match options.early_stopping {
        EarlyStopping::Auto => n > EARLY_STOPPING_MIN_SAMPLES,
        EarlyStopping::On => true,
        EarlyStopping::Off => false,
    };
    let (train, validation) = if early_stopping {
        validation_split(target, options.validation_fraction, options.seed)
    } else {
        ((0..n as u32).collect(), vec![])
    };
    let mapper = BinMapper::fit(
        x.select(
            Axis(0),
            &train.iter().map(|&i| i as usize).collect::<Vec<_>>(),
        )
        .view(),
    );
    let bins = mapper.transform(x);
    let base = baseline(target, weights, &train);
    let mut raw = vec![base.clone(); n];
    let grower = GrowerOptions {
        max_leaf_nodes: options.max_leaf_nodes,
        min_samples_leaf: options.min_samples_leaf,
        l2_regularization: options.l2_regularization,
        learning_rate: options.learning_rate,
    };
    let mut iterations: Vec<Vec<Tree>> = Vec::new();
    let mut scores = vec![-mean_loss(target, &raw, weights, &validation)];
    for _ in 0..options.max_iter {
        let trees: Vec<Tree> = (0..target.trees_per_iteration())
            .map(|k| {
                let (g, h) = gradients(target, &raw, weights, k);
                let data = TreeData {
                    bins: &bins,
                    mapper: &mapper,
                    gradients: &g,
                    hessians: &h,
                };
                grow_tree(train.clone(), &data, &grower)
            })
            .collect();
        for (i, scores_i) in raw.iter_mut().enumerate() {
            let row: Vec<f64> = x.row(i).to_vec();
            for (score, tree) in scores_i.iter_mut().zip(&trees) {
                *score += tree.predict(&row);
            }
        }
        iterations.push(trees);
        if early_stopping {
            scores.push(-mean_loss(target, &raw, weights, &validation));
            if should_stop(&scores, options.n_iter_no_change, options.tol) {
                break;
            }
        }
    }
    let model = Gbdt {
        loss: target.loss(),
        n_features: x.ncols(),
        classes,
        baseline: base,
        iterations,
    };
    model.validate()?;
    Ok(model)
}
