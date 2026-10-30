//! Best-first growth of one histogram tree on gradients and hessians (sklearn `TreeGrower`).

use rayon::prelude::*;

use super::binning::{BinMapper, MISSING_BIN};
use super::Tree;

#[derive(Debug, Clone, Copy)]
pub struct GrowerOptions {
    pub max_leaf_nodes: usize,
    pub min_samples_leaf: usize,
    pub l2_regularization: f64,
    pub learning_rate: f64,
}

const MIN_HESSIAN_TO_SPLIT: f64 = 1e-3;
const HISTOGRAM_BINS: usize = 256;

#[derive(Clone, Copy, Default)]
struct BinStats {
    gradient: f64,
    hessian: f64,
    count: usize,
}

impl BinStats {
    fn add(&mut self, other: BinStats) {
        self.gradient += other.gradient;
        self.hessian += other.hessian;
        self.count += other.count;
    }

    fn minus(self, other: BinStats) -> BinStats {
        BinStats {
            gradient: self.gradient - other.gradient,
            hessian: self.hessian - other.hessian,
            count: self.count - other.count,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Split {
    gain: f64,
    feature: usize,
    bin: usize,
    missing_goes_left: bool,
}

struct Node {
    samples: Vec<u32>,
    stats: BinStats,
    split: Option<Split>,
    children: Option<(usize, usize)>,
}

pub struct TreeData<'a> {
    pub bins: &'a [Vec<u8>],
    pub mapper: &'a BinMapper,
    pub gradients: &'a [f64],
    pub hessians: &'a [f64],
}

fn score(stats: BinStats, l2: f64) -> f64 {
    stats.gradient * stats.gradient / (stats.hessian + l2)
}

impl TreeData<'_> {
    fn stats_of(&self, samples: &[u32]) -> BinStats {
        let mut stats = BinStats::default();
        for &i in samples {
            stats.add(BinStats {
                gradient: self.gradients[i as usize],
                hessian: self.hessians[i as usize],
                count: 1,
            });
        }
        stats
    }

    fn histogram(&self, feature: usize, samples: &[u32]) -> Vec<BinStats> {
        let mut histogram = vec![BinStats::default(); HISTOGRAM_BINS];
        let column = &self.bins[feature];
        for &i in samples {
            let i = i as usize;
            histogram[column[i] as usize].add(BinStats {
                gradient: self.gradients[i],
                hessian: self.hessians[i],
                count: 1,
            });
        }
        histogram
    }
}

fn admissible(left: BinStats, right: BinStats, options: &GrowerOptions) -> bool {
    left.count >= options.min_samples_leaf
        && right.count >= options.min_samples_leaf
        && left.hessian >= MIN_HESSIAN_TO_SPLIT
        && right.hessian >= MIN_HESSIAN_TO_SPLIT
}

/// Scans the value bins left to right (missing values right) and, when the feature had missing values in training,
/// right to left (missing values left). The last value bin with missing right is the "split on NaN".
fn best_split_of_feature(
    feature: usize,
    histogram: &[BinStats],
    total: BinStats,
    data: &TreeData,
    options: &GrowerOptions,
) -> Option<Split> {
    let has_missing = data.mapper.has_missing[feature];
    let value_bins = data.mapper.value_bins(feature);
    let parent_score = score(total, options.l2_regularization);
    let gain = |left: BinStats, right: BinStats| {
        score(left, options.l2_regularization) + score(right, options.l2_regularization)
            - parent_score
    };
    let mut best: Option<Split> = None;
    let mut consider = |left: BinStats, bin: usize, missing_goes_left: bool| {
        let right = total.minus(left);
        if !admissible(left, right, options) {
            return;
        }
        let g = gain(left, right);
        if g > 0.0 && best.is_none_or(|b| g > b.gain) {
            best = Some(Split {
                gain: g,
                feature,
                bin,
                missing_goes_left,
            });
        }
    };
    let mut left = BinStats::default();
    let last_bin = value_bins - 1 + has_missing as usize;
    for (bin, stats) in histogram.iter().enumerate().take(last_bin) {
        left.add(*stats);
        consider(left, bin, false);
    }
    if has_missing {
        let mut right = BinStats::default();
        for bin in (0..value_bins - 1).rev() {
            right.add(histogram[bin + 1]);
            consider(total.minus(right), bin, true);
        }
    }
    best
}

fn best_split(
    samples: &[u32],
    total: BinStats,
    data: &TreeData,
    options: &GrowerOptions,
) -> Option<Split> {
    if samples.len() < 2 * options.min_samples_leaf || total.hessian < MIN_HESSIAN_TO_SPLIT {
        return None;
    }
    let candidates: Vec<Option<Split>> = (0..data.bins.len())
        .into_par_iter()
        .map(|feature| {
            let histogram = data.histogram(feature, samples);
            best_split_of_feature(feature, &histogram, total, data, options)
        })
        .collect();
    candidates
        .into_iter()
        .flatten()
        .fold(None, |best: Option<Split>, s| match best {
            Some(b) if b.gain >= s.gain => Some(b),
            _ => Some(s),
        })
}

fn goes_left(bin: u8, split: &Split) -> bool {
    if bin == MISSING_BIN {
        split.missing_goes_left
    } else {
        bin as usize <= split.bin
    }
}

pub fn grow_tree(samples: Vec<u32>, data: &TreeData, options: &GrowerOptions) -> Tree {
    let stats = data.stats_of(&samples);
    let split = best_split(&samples, stats, data, options);
    let mut nodes = vec![Node {
        samples,
        stats,
        split,
        children: None,
    }];
    let mut leaves = 1;
    while leaves < options.max_leaf_nodes {
        let Some(parent) = best_open_leaf(&nodes) else {
            break;
        };
        let split = nodes[parent].split.unwrap();
        let column = &data.bins[split.feature];
        let (left, right): (Vec<u32>, Vec<u32>) = nodes[parent]
            .samples
            .iter()
            .partition(|&&i| goes_left(column[i as usize], &split));
        let mut split = split;
        if !data.mapper.has_missing[split.feature] {
            // no missing values in training: at prediction time they follow the bigger child (sklearn)
            split.missing_goes_left = left.len() > right.len();
            nodes[parent].split = Some(split);
        }
        let children = [left, right].map(|child| {
            let stats = data.stats_of(&child);
            let split = best_split(&child, stats, data, options);
            nodes.push(Node {
                samples: child,
                stats,
                split,
                children: None,
            });
            nodes.len() - 1
        });
        nodes[parent].children = Some((children[0], children[1]));
        nodes[parent].samples = Vec::new();
        leaves += 1;
    }
    to_tree(&nodes, data.mapper, options)
}

fn best_open_leaf(nodes: &[Node]) -> Option<usize> {
    (0..nodes.len())
        .filter(|&i| nodes[i].children.is_none() && nodes[i].split.is_some())
        .fold(None, |best: Option<usize>, i| match best {
            Some(b) if nodes[b].split.unwrap().gain >= nodes[i].split.unwrap().gain => Some(b),
            _ => Some(i),
        })
}

/// Arena -> exported layout: split nodes numbered in creation order (root 0), leaves `-1 - k`.
fn to_tree(nodes: &[Node], mapper: &BinMapper, options: &GrowerOptions) -> Tree {
    let mut split_id = vec![0usize; nodes.len()];
    let mut leaf_id = vec![0usize; nodes.len()];
    let (mut splits, mut leaves) = (0, 0);
    for (i, node) in nodes.iter().enumerate() {
        if node.children.is_some() {
            split_id[i] = splits;
            splits += 1;
        } else {
            leaf_id[i] = leaves;
            leaves += 1;
        }
    }
    let child_ref = |i: usize| -> i32 {
        if nodes[i].children.is_some() {
            split_id[i] as i32
        } else {
            -1 - leaf_id[i] as i32
        }
    };
    let mut tree = Tree::single_leaf(0.0);
    tree.leaf_value.clear();
    for node in nodes {
        match (node.children, node.split) {
            (Some((left, right)), Some(split)) => {
                tree.feature.push(split.feature);
                tree.threshold
                    .push(mapper.threshold(split.feature, split.bin));
                tree.missing_goes_left.push(split.missing_goes_left);
                tree.left.push(child_ref(left));
                tree.right.push(child_ref(right));
            }
            _ => tree.leaf_value.push(
                -options.learning_rate * node.stats.gradient
                    / (node.stats.hessian + options.l2_regularization),
            ),
        }
    }
    tree
}
