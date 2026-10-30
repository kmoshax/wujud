//! Geometry-invariant seat descriptors: link evidence binned by each seat's Fresnel excess path to the link.

use ndarray::{Array2, ArrayView3, Axis};

use crate::numeric::{
    mean, nan_max, nan_mean, nan_mean_f32, nan_median, nan_median_f32, nan_min, nan_percentile,
    nan_std_f32,
};

/// Excess-path bin edges (m); the first Fresnel zone at 2.4 GHz is lambda / 2 = 0.061 m.
pub const DISTANCE_BIN_EDGES_M: [f64; 6] = [0.0, 0.015, 0.035, 0.07, 0.14, 0.28];
const DISTANCE_BINS: usize = DISTANCE_BIN_EDGES_M.len() - 1;
pub const LINK_FEATURE_NAMES: [&str; 8] = [
    "sh_mean",
    "sh_min",
    "sh_abs",
    "sh_spread",
    "mo_mean",
    "mo_p90",
    "mo_frac",
    "slow",
];
const BINNED_FEATURE_NAMES: [&str; 6] =
    ["sh_mean", "sh_min", "sh_abs", "mo_mean", "mo_frac", "slow"];
const NEIGHBOUR_RADIUS_M: f64 = 1.0;
const CEILING_FOR_SURFACE_M: f64 = 3.7;
const MOTION_RATIO_THRESHOLD: f64 = 1.5;

/// Per-link evidence, one value per link for each of `LINK_FEATURE_NAMES` (NaN for a dead link).
#[derive(Debug, Clone, PartialEq)]
pub struct LinkFeatures {
    pub values: [Vec<f64>; 8],
}

impl LinkFeatures {
    pub fn get(&self, name: &str) -> &[f64] {
        let index = LINK_FEATURE_NAMES
            .iter()
            .position(|&n| n == name)
            .expect("known link feature");
        &self.values[index]
    }
}

/// `np.maximum`: NaN propagates (`f64::max` would drop it).
fn maximum(x: f64, floor: f64) -> f64 {
    if x.is_nan() {
        x
    } else {
        x.max(floor)
    }
}

fn to_f32(x: f64) -> f64 {
    x as f32 as f64
}

fn as_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|&v| v as f32).collect()
}

fn f32_mean(values: &[f64]) -> f64 {
    nan_mean_f32(&as_f32(values)) as f64
}

fn f32_std(values: &[f64]) -> f64 {
    nan_std_f32(&as_f32(values)) as f64
}

fn f32_median(values: &[f64]) -> f64 {
    nan_median_f32(&as_f32(values)) as f64
}

/// `[links, groups]` NaN-median over the windows (axis 1) of `[links, windows, groups]`.
fn median_over_windows(x: &ArrayView3<f32>) -> Array2<f64> {
    let (links, _, groups) = x.dim();
    Array2::from_shape_fn((links, groups), |(l, g)| {
        let lane: Vec<f32> = x.index_axis(Axis(0), l).column(g).to_vec();
        nan_median_f32(&lane) as f64
    })
}

/// `[links, windows]` NaN-mean over the groups (axis 2).
fn mean_over_groups(x: &ArrayView3<f32>) -> Array2<f64> {
    let (links, windows, _) = x.dim();
    Array2::from_shape_fn((links, windows), |(l, w)| {
        let groups: Vec<f32> = x.slice(ndarray::s![l, w, ..]).to_vec();
        nan_mean_f32(&groups) as f64
    })
}

fn per_row(x: &Array2<f64>, statistic: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    x.rows()
        .into_iter()
        .map(|row| statistic(&row.to_vec()))
        .collect()
}

/// `research/wujud/model.py::link_features` on `[links, windows, 13]` calibration and session windows.
pub fn link_features(
    cal_mean: ArrayView3<f32>,
    cal_std: ArrayView3<f32>,
    ses_mean: ArrayView3<f32>,
    ses_std: ArrayView3<f32>,
) -> LinkFeatures {
    let shift = (&median_over_windows(&ses_mean) - &median_over_windows(&cal_mean)).mapv(to_f32);
    let baseline_spread = per_row(&median_over_windows(&cal_std), f32_mean);
    let ses_spread = mean_over_groups(&ses_std);
    let motion_ratio = Array2::from_shape_fn(ses_spread.dim(), |(l, w)| {
        to_f32(ses_spread[[l, w]] / maximum(baseline_spread[l], 1e-3))
    });
    let log_ratio = motion_ratio.mapv(|m| to_f32(maximum(m, 1e-3).ln()));
    let slow_session = per_row(&mean_over_groups(&ses_mean), f32_std);
    let slow_calibration = per_row(&mean_over_groups(&cal_mean), f32_std);
    LinkFeatures {
        values: [
            per_row(&shift, f32_mean),
            per_row(&shift, nan_min),
            per_row(&shift.mapv(f64::abs), f32_mean),
            per_row(&shift, f32_std),
            per_row(&log_ratio, f32_mean),
            per_row(&log_ratio, |r| to_f32(nan_percentile(r, 90.0))),
            per_row(&motion_ratio, |r| {
                mean(
                    &r.iter()
                        .map(|&m| (m > MOTION_RATIO_THRESHOLD) as u8 as f64)
                        .collect::<Vec<_>>(),
                )
            }),
            slow_session
                .iter()
                .zip(&slow_calibration)
                .map(|(s, c)| to_f32((maximum(*s, 1e-3) / maximum(*c, 1e-3)).ln()))
                .collect(),
        ],
    }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]))
        .sqrt()
}

/// `|a - p| + |p - b| - |a - b|`: how much longer the path via `p` is than the direct link.
pub fn excess_path(a: [f64; 3], b: [f64; 3], p: [f64; 3]) -> f64 {
    distance(p, a) + distance(p, b) - distance(b, a)
}

/// `[links, seats]` excess-path bin of every seat for every link; `None` outside the last edge.
pub fn distance_bins(
    nodes: &[[f64; 3]],
    links: &[(usize, usize)],
    chests: &[[f64; 3]],
) -> Array2<Option<usize>> {
    Array2::from_shape_fn((links.len(), chests.len()), |(l, s)| {
        let (a, b) = links[l];
        let e = excess_path(nodes[a], nodes[b], chests[s]);
        DISTANCE_BIN_EDGES_M
            .windows(2)
            .position(|edge| edge[0] <= e && e < edge[1])
    })
}

fn zero_if_nan(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x
    }
}

/// Mean over the links in the bin; NumPy does it in float32 for float32 features (all but `mo_frac`).
fn bin_mean(values: &[f64], members: &[usize], in_float32: bool) -> f64 {
    if members.is_empty() {
        return f64::NAN;
    }
    if in_float32 {
        let sum = members
            .iter()
            .fold(0.0f32, |acc, &l| acc + zero_if_nan(values[l]) as f32);
        (sum / members.len() as f32) as f64
    } else {
        members.iter().map(|&l| zero_if_nan(values[l])).sum::<f64>() / members.len() as f64
    }
}

/// `[seats, 49]` stage-1 descriptor (`model.py::seat_descriptor`), float32 values.
pub fn seat_descriptor(
    features: &LinkFeatures,
    bins: &Array2<Option<usize>>,
    nodes: &[[f64; 3]],
    chests: &[[f64; 3]],
) -> Array2<f32> {
    let seats = bins.ncols();
    let rows: Vec<Vec<f64>> = (0..seats)
        .map(|s| seat_row(features, bins, nodes, chests, s))
        .collect();
    Array2::from_shape_fn((seats, rows[0].len()), |(s, c)| rows[s][c] as f32)
}

fn seat_row(
    features: &LinkFeatures,
    bins: &Array2<Option<usize>>,
    nodes: &[[f64; 3]],
    chests: &[[f64; 3]],
    seat: usize,
) -> Vec<f64> {
    let mut row = Vec::with_capacity(49);
    for bin in 0..DISTANCE_BINS {
        let members: Vec<usize> = (0..bins.nrows())
            .filter(|&l| bins[[l, seat]] == Some(bin))
            .collect();
        row.push(members.len() as f64);
        for name in BINNED_FEATURE_NAMES {
            row.push(bin_mean(features.get(name), &members, name != "mo_frac"));
        }
        if bin <= 1 {
            let in_bin = |name| {
                members
                    .iter()
                    .map(|&l| zero_if_nan(features.get(name)[l]))
                    .collect::<Vec<_>>()
            };
            row.push(nan_min(&in_bin("sh_mean")));
            row.push(nan_max(&in_bin("mo_mean")));
        }
    }
    for values in &features.values {
        row.push(f32_median(values));
    }
    let nearest_node = nodes
        .iter()
        .map(|&n| distance(chests[seat], n))
        .fold(f64::INFINITY, f64::min);
    row.push(nearest_node);
    row.push(nodes.len() as f64);
    row
}

/// `[seats, 4]` stage-2 context from the stage-1 probabilities: neighbours and "explaining away" on shared links.
pub fn context_features(
    p1: &[f64],
    bins: &Array2<Option<usize>>,
    chests: &[[f64; 3]],
) -> Array2<f32> {
    let seats = chests.len();
    let near = |l: usize, s: usize| matches!(bins[[l, s]], Some(0 | 1));
    let link_load: Vec<f64> = (0..bins.nrows())
        .map(|l| (0..seats).filter(|&s| near(l, s)).map(|s| p1[s]).sum())
        .collect();
    let mut out = Array2::zeros((seats, 4));
    for s in 0..seats {
        let neighbours: Vec<usize> = (0..seats)
            .filter(|&o| {
                let d = distance(chests[s], chests[o]);
                d < NEIGHBOUR_RADIUS_M && d > 0.0
            })
            .collect();
        let neighbour_mean =
            zero_if_nan(neighbours.iter().map(|&o| p1[o]).sum::<f64>() / neighbours.len() as f64);
        let links: Vec<usize> = (0..bins.nrows()).filter(|&l| near(l, s)).collect();
        let others = links.iter().map(|&l| link_load[l] - p1[s]).sum::<f64>() / links.len() as f64;
        let row = [p1[s], neighbour_mean, others, p1[s] - neighbour_mean];
        for (c, v) in row.iter().enumerate() {
            out[[s, c]] = *v as f32;
        }
    }
    out
}

/// Inputs of the head-count regressor (`model.py::scene_vector`), 41 float32 values.
pub fn scene_vector(
    features: &LinkFeatures,
    nodes: &[[f64; 3]],
    links: &[(usize, usize)],
    seats: usize,
    width: f64,
    depth: f64,
) -> Vec<f32> {
    let mut v = Vec::with_capacity(41);
    for values in &features.values {
        v.extend([
            nan_median(values),
            nan_mean(values),
            nan_percentile(values, 10.0),
            nan_percentile(values, 90.0),
        ]);
    }
    let area = width * depth;
    let surface = 2.0 * area + 2.0 * (width + depth) * CEILING_FOR_SURFACE_M;
    let lengths: Vec<f64> = links
        .iter()
        .map(|&(i, j)| distance(nodes[i], nodes[j]))
        .collect();
    let shadow = features.get("sh_mean");
    // Sabine-style inversion: a d dB drop of the reverberant field ~ surface * (10^(d/10) - 1) absorbers
    let absorbers = |drop_db: f64| surface * (10f64.powf(-drop_db / 10.0) - 1.0);
    v.extend([
        nodes.len() as f64,
        seats as f64,
        area,
        surface,
        mean(&lengths),
        absorbers(nan_median(shadow)),
        absorbers(nan_mean(shadow)),
        nan_median(features.get("mo_mean")) * seats as f64,
        shadow_slope_per_metre(&lengths, shadow),
    ]);
    v.iter().map(|&x| x as f32).collect()
}

/// Least-squares slope of link shadowing over link length (`np.polyfit(..., 1)[0]`), 0 with <= 3 live links.
fn shadow_slope_per_metre(lengths: &[f64], shadow: &[f64]) -> f64 {
    let live: Vec<(f64, f64)> = lengths
        .iter()
        .zip(shadow)
        .filter(|(_, s)| s.is_finite())
        .map(|(&l, &s)| (l, s))
        .collect();
    if live.len() <= 3 {
        return 0.0;
    }
    let mean_length = live.iter().map(|p| p.0).sum::<f64>() / live.len() as f64;
    let mean_shadow = live.iter().map(|p| p.1).sum::<f64>() / live.len() as f64;
    let covariance: f64 = live
        .iter()
        .map(|(l, s)| (l - mean_length) * (s - mean_shadow))
        .sum();
    let variance: f64 = live.iter().map(|(l, _)| (l - mean_length).powi(2)).sum();
    covariance / variance
}

/// Everything the seat model needs about one moment of the hall.
#[derive(Debug, Clone)]
pub struct SceneEvidence {
    pub link_features: LinkFeatures,
    pub seat_descriptor: Array2<f32>,
    pub scene_vector: Vec<f32>,
}
