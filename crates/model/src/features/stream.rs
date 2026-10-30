use ndarray::{s, Array2, ArrayView2, Axis};

use super::{
    extra_features, spectral_features, to_grid, window_features, BreathingRate, GRID_RATE_HZ,
};
use crate::numeric::std;

/// Feature rows of consecutive windows, with one breathing-rate guess per window.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WindowSet {
    pub rows: Vec<Vec<f64>>,
    pub breathing_rates: Vec<BreathingRate>,
}

fn window_starts(grid_len: usize, window: usize, hop: usize) -> impl Iterator<Item = usize> {
    let count = if grid_len >= window {
        (grid_len - window) / hop + 1
    } else {
        0
    };
    (0..count).map(move |i| i * hop)
}

fn samples(seconds: f64) -> usize {
    (seconds * GRID_RATE_HZ) as usize
}

/// Window feature rows `[core 10, RSSI wobble (if rssi given), extra 6]` of a linear-amplitude stream.
pub fn windows(
    times: &[f64],
    amplitude: ArrayView2<f64>,
    rssi: Option<&[f64]>,
    window_seconds: f64,
    hop_seconds: f64,
) -> WindowSet {
    let grid = to_grid(times, amplitude, GRID_RATE_HZ);
    let rssi_grid = rssi.map(|r| {
        let column = ArrayView2::from_shape((r.len(), 1), r).expect("one RSSI per packet");
        to_grid(times, column, GRID_RATE_HZ)
    });
    let (window, hop) = (samples(window_seconds), samples(hop_seconds));
    let mut set = WindowSet::default();
    for start in window_starts(grid.nrows(), window, hop) {
        let segment = grid.slice(s![start..start + window, ..]);
        let (mut row, rate) = window_features(segment, GRID_RATE_HZ);
        if let Some(rssi_grid) = &rssi_grid {
            let wobble = rssi_grid.slice(s![start..start + window, 0]).to_vec();
            row.push((std(&wobble) + 0.1).log10());
        }
        row.extend(extra_features(segment, GRID_RATE_HZ));
        set.rows.push(row);
        set.breathing_rates.push(rate);
    }
    set
}

/// The ten `spectral_features` of a sanitized-phase stream, on the same windows as `windows`.
pub fn phase_windows(
    times: &[f64],
    phase: ArrayView2<f64>,
    window_seconds: f64,
    hop_seconds: f64,
) -> Vec<Vec<f64>> {
    let grid = to_grid(times, phase, GRID_RATE_HZ);
    window_starts(grid.nrows(), samples(window_seconds), samples(hop_seconds))
        .map(|start| {
            let segment = grid.slice(s![start..start + samples(window_seconds), ..]);
            spectral_features(segment, GRID_RATE_HZ).0
        })
        .collect()
}

/// The room's static fingerprint per window: mean amplitude (dB) of every subcarrier.
pub fn profile_windows(
    times: &[f64],
    amplitude: ArrayView2<f64>,
    window_seconds: f64,
    hop_seconds: f64,
) -> Vec<Vec<f64>> {
    let grid = to_grid(times, amplitude, GRID_RATE_HZ);
    let decibels: Array2<f64> = grid.mapv(|a| 20.0 * a.max(1e-3).log10());
    window_starts(grid.nrows(), samples(window_seconds), samples(hop_seconds))
        .map(|start| {
            decibels
                .slice(s![start..start + samples(window_seconds), ..])
                .mean_axis(Axis(0))
                .unwrap()
                .to_vec()
        })
        .collect()
}
