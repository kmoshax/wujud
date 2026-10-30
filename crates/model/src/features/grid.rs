use ndarray::{Array2, ArrayView2};

use crate::numeric::interp_at_integers;

/// Resamples packets at arbitrary times onto a uniform grid: bin means, empty bins linearly filled.
/// `times` must be sorted; `values` is `[packets, columns]`.
pub fn to_grid(times: &[f64], values: ArrayView2<f64>, rate_hz: f64) -> Array2<f64> {
    let columns = values.ncols();
    if times.len() < 2 {
        return Array2::zeros((0, columns));
    }
    let rows = ((times[times.len() - 1] - times[0]) * rate_hz) as usize + 1;
    let (sums, counts) = bin_sums(times, values, rate_hz, rows);
    let filled: Vec<usize> = (0..rows).filter(|&r| counts[r] > 0).collect();
    let positions: Vec<f64> = filled.iter().map(|&r| r as f64).collect();
    let mut grid = Array2::zeros((rows, columns));
    for c in 0..columns {
        let means: Vec<f64> = filled
            .iter()
            .map(|&r| sums[[r, c]] / counts[r] as f64)
            .collect();
        for (r, v) in interp_at_integers(rows, &positions, &means)
            .into_iter()
            .enumerate()
        {
            grid[[r, c]] = v;
        }
    }
    grid
}

fn bin_sums(
    times: &[f64],
    values: ArrayView2<f64>,
    rate_hz: f64,
    rows: usize,
) -> (Array2<f64>, Vec<usize>) {
    let mut sums = Array2::zeros((rows, values.ncols()));
    let mut counts = vec![0usize; rows];
    let start = times[0];
    for (i, &t) in times.iter().enumerate() {
        let row = (((t - start) * rate_hz) as usize).min(rows - 1);
        counts[row] += 1;
        for c in 0..values.ncols() {
            sums[[row, c]] += values[[i, c]];
        }
    }
    (sums, counts)
}
