use ndarray::{s, Array2, ArrayView2, Axis};

use super::{round_like_float32, to_grid, BreathingRate, GRID_RATE_HZ};
use crate::numeric::{argmax, mean, median, percentile, std};
use crate::spectrum::{hann_power_spectrum, rfft_frequencies};

pub const RSSI_FEATURE_NAMES: [&str; 10] = [
    "std",
    "p90_absdiff",
    "br",
    "mi",
    "hi",
    "br_mi",
    "peak",
    "glrt",
    "range",
    "slow",
];

/// Ten features of one RSSI window `[samples, chains]` (dBm on the 20 Hz grid), plus a breathing-rate guess.
pub fn rssi_window(grid: ArrayView2<f64>, rate_hz: f64) -> (Vec<f64>, BreathingRate) {
    let n = grid.nrows();
    let centered = &grid - &grid.mean_axis(Axis(0)).unwrap();
    let power = padded_power(&centered);
    let freqs = rfft_frequencies(4 * n, rate_hz);
    let band_mean = |lo: f64, hi: f64| -> Vec<f64> { band_mean_per_chain(&power, &freqs, lo, hi) };
    let breath = band_mean(0.12, 0.6);
    let micro = band_mean(0.6, 3.0);
    let high: Vec<f64> = band_mean(4.0, 9.5).iter().map(|p| p + 1e-6).collect();
    let breath_line = summed_over_chains(&power, &freqs, 0.12, 0.6);
    let micro_floor = median(&summed_over_chains(&power, &freqs, 1.0, 3.0).1);
    let strongest = argmax(&breath_line.1);
    let peak_power = breath_line.1[strongest];
    let (breath, micro, high) = (mean(&breath), mean(&micro), mean(&high));
    let mut features = vec![
        (mean(&chain_stds(&centered)) + 1e-3).log10(),
        (percentile(&abs_steps(&centered), 90.0) + 1e-3).log10(),
        (breath / high).log10(),
        (micro / high).log10(),
        high.log10(),
        (breath / (micro + 1e-6)).log10(),
        peak_power / (breath_line.1.iter().sum::<f64>() + 1e-9),
        (peak_power / (micro_floor + 1e-9) + 1e-9).ln(),
        peak_to_peak(&centered),
        half_to_half_shift(&centered),
    ];
    round_like_float32(&mut features);
    (features, breath_line.0[strongest] * 60.0)
}

/// `[frequency, chain]` power of each chain, zero-padded to 4x the window for a finer breathing line.
fn padded_power(centered: &Array2<f64>) -> Array2<f64> {
    let n = centered.nrows();
    let mut power = Array2::zeros((2 * n + 1, centered.ncols()));
    for (c, column) in centered.columns().into_iter().enumerate() {
        let p = hann_power_spectrum(&column.to_vec(), 4 * n);
        power.column_mut(c).assign(&ndarray::Array1::from(p));
    }
    power
}

fn band_rows(freqs: &[f64], lo: f64, hi: f64) -> Vec<usize> {
    (0..freqs.len())
        .filter(|&i| freqs[i] >= lo && freqs[i] < hi)
        .collect()
}

fn band_mean_per_chain(power: &Array2<f64>, freqs: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    let rows = band_rows(freqs, lo, hi);
    (0..power.ncols())
        .map(|c| rows.iter().map(|&r| power[[r, c]]).sum::<f64>() / rows.len() as f64)
        .collect()
}

/// (frequencies, power summed over chains) inside `[lo, hi)`.
fn summed_over_chains(
    power: &Array2<f64>,
    freqs: &[f64],
    lo: f64,
    hi: f64,
) -> (Vec<f64>, Vec<f64>) {
    band_rows(freqs, lo, hi)
        .into_iter()
        .map(|r| (freqs[r], power.row(r).sum()))
        .unzip()
}

fn chain_stds(centered: &Array2<f64>) -> Vec<f64> {
    centered
        .columns()
        .into_iter()
        .map(|c| std(&c.to_vec()))
        .collect()
}

fn abs_steps(centered: &Array2<f64>) -> Vec<f64> {
    let mut steps = Vec::new();
    for i in 1..centered.nrows() {
        for c in 0..centered.ncols() {
            steps.push((centered[[i, c]] - centered[[i - 1, c]]).abs());
        }
    }
    steps
}

fn peak_to_peak(centered: &Array2<f64>) -> f64 {
    let max = centered.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = centered.iter().copied().fold(f64::INFINITY, f64::min);
    max - min
}

fn half_to_half_shift(centered: &Array2<f64>) -> f64 {
    let half = centered.nrows() / 2;
    let shifts: Vec<f64> = centered
        .columns()
        .into_iter()
        .map(|c| (mean(&c.slice(s![..half]).to_vec()) - mean(&c.slice(s![half..]).to_vec())).abs())
        .collect();
    mean(&shifts)
}

/// RSSI feature rows over 20 s windows every 5 s; `rssi` is `[packets, chains]` in dBm.
pub fn rssi_windows(
    times: &[f64],
    rssi: ArrayView2<f64>,
    window_seconds: f64,
    hop_seconds: f64,
) -> (Vec<Vec<f64>>, Vec<BreathingRate>) {
    let grid = to_grid(times, rssi, GRID_RATE_HZ);
    let window = (window_seconds * GRID_RATE_HZ) as usize;
    let hop = (hop_seconds * GRID_RATE_HZ) as usize;
    let mut rows = Vec::new();
    let mut rates = Vec::new();
    let mut start = 0;
    while grid.nrows() >= window && start + window <= grid.nrows() {
        let (row, rate) = rssi_window(grid.slice(s![start..start + window, ..]), GRID_RATE_HZ);
        rows.push(row);
        rates.push(rate);
        start += hop;
    }
    (rows, rates)
}
