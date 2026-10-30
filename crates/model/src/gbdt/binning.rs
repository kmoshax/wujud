//! Feature binning for training: up to 255 value bins per feature plus one bin for missing values.

use ndarray::ArrayView2;

pub const MAX_BINS: usize = 255;
pub const MISSING_BIN: u8 = 255;
/// sklearn keeps thresholds finite except for the "split on NaN" case.
const ALMOST_INF: f64 = 1e300;

/// Right edges of the value bins of each feature: bin `b` holds `thresholds[b-1] < x <= thresholds[b]`.
#[derive(Debug, Clone)]
pub struct BinMapper {
    pub thresholds: Vec<Vec<f64>>,
    pub has_missing: Vec<bool>,
}

/// `np.percentile(sorted, q, method="averaged_inverted_cdf")`, `q` in 0..=1.
fn averaged_inverted_cdf(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let position = n as f64 * q;
    let below = position.floor() as usize;
    if position == below as f64 && below > 0 && below < n {
        (sorted[below - 1] + sorted[below]) / 2.0
    } else {
        sorted[below.min(n - 1)]
    }
}

fn distinct(sorted: &[f64]) -> Vec<f64> {
    let mut values = sorted.to_vec();
    values.dedup();
    values
}

/// sklearn `_find_binning_thresholds` without sample weights: midpoints of the distinct values when there are few,
/// otherwise 254 quantiles.
fn feature_thresholds(column: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut values: Vec<f64> = column.filter(|v| !v.is_nan()).collect();
    values.sort_by(f64::total_cmp);
    let unique = distinct(&values);
    if unique.len() <= 1 {
        return vec![];
    }
    let thresholds: Vec<f64> = if unique.len() <= MAX_BINS {
        unique.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect()
    } else {
        (1..MAX_BINS)
            .map(|i| averaged_inverted_cdf(&values, i as f64 / MAX_BINS as f64))
            .collect()
    };
    distinct(&thresholds)
        .into_iter()
        .map(|t| t.min(ALMOST_INF))
        .collect()
}

impl BinMapper {
    pub fn fit(x: ArrayView2<f64>) -> Self {
        BinMapper {
            thresholds: x
                .columns()
                .into_iter()
                .map(|c| feature_thresholds(c.iter().copied()))
                .collect(),
            has_missing: x
                .columns()
                .into_iter()
                .map(|c| c.iter().any(|v| v.is_nan()))
                .collect(),
        }
    }

    pub fn value_bins(&self, feature: usize) -> usize {
        self.thresholds[feature].len() + 1
    }

    pub fn bin(&self, feature: usize, value: f64) -> u8 {
        if value.is_nan() {
            return MISSING_BIN;
        }
        self.thresholds[feature].partition_point(|&t| t < value) as u8
    }

    /// Column-major `[feature][sample]` bins.
    pub fn transform(&self, x: ArrayView2<f64>) -> Vec<Vec<u8>> {
        x.columns()
            .into_iter()
            .enumerate()
            .map(|(f, column)| column.iter().map(|&v| self.bin(f, v)).collect())
            .collect()
    }

    /// The raw-value threshold of a split after bin `bin`; past the last value bin it is the "split on NaN".
    pub fn threshold(&self, feature: usize, bin: usize) -> f64 {
        self.thresholds[feature]
            .get(bin)
            .copied()
            .unwrap_or(f64::INFINITY)
    }
}
