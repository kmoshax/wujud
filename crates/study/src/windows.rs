//! Window features of one link, also when the link hops over several channels.
//!
//! With one channel this is exactly `wujud_model::features::windows` + `profile_windows`. With N channels
//! each channel's packets are turned into amplitude and put on the 20 Hz grid separately, and the grids are
//! placed side by side as N × K subcarriers; RSSI is centred per channel before its wobble is measured.

use ndarray::{concatenate, s, Array2, ArrayView2, Axis};
use wujud_model::features::{
    amplitude_from_iq, extra_features, to_grid, window_features, GRID_RATE_HZ, HOP_SECONDS,
    WINDOW_SECONDS,
};
use wujud_model::numeric::{median, std};
use wujud_sim::capture::Capture;

pub const GLRT_COLUMN: usize = 16;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LinkWindows {
    /// `[core 10, RSSI wobble, extra 6]` per window.
    pub rows: Vec<Vec<f64>>,
    pub breathing_rates: Vec<f64>,
    /// Mean amplitude (dB) of every subcarrier of every channel per window: the static profile.
    pub profiles: Vec<Vec<f64>>,
}

struct ChannelPackets {
    times: Vec<f64>,
    iq: Array2<i8>,
    rssi: Vec<f64>,
}

fn split_by_channel(capture: &Capture, channels: usize) -> Vec<ChannelPackets> {
    let subcarriers = capture.iq.dim().1;
    (0..channels)
        .map(|channel| {
            let packets: Vec<usize> = (0..capture.len())
                .filter(|&n| capture.frame[n] as usize % channels == channel)
                .collect();
            let iq = Array2::from_shape_fn((packets.len(), 2 * subcarriers), |(n, column)| {
                capture.iq[[packets[n], column / 2, column % 2]]
            });
            ChannelPackets {
                times: packets.iter().map(|&n| capture.t[n]).collect(),
                iq,
                rssi: packets
                    .iter()
                    .map(|&n| f64::from(capture.rssi[n]))
                    .collect(),
            }
        })
        .collect()
}

fn amplitude_grid(packets: &ChannelPackets) -> Array2<f64> {
    let amplitude = amplitude_from_iq(packets.iq.view()).mapv(f64::from);
    to_grid(&packets.times, amplitude.view(), GRID_RATE_HZ)
}

/// The RSSI of all channels on one grid, each channel's median removed (channels differ by their fading).
fn centred_rssi_grid(channels: &[ChannelPackets]) -> Array2<f64> {
    let mut samples: Vec<(f64, f64)> = channels
        .iter()
        .flat_map(|packets| {
            let level = median(&packets.rssi);
            let centre = if channels.len() > 1 { level } else { 0.0 };
            packets
                .times
                .iter()
                .zip(&packets.rssi)
                .map(move |(&t, &r)| (t, r - centre))
        })
        .collect();
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let times: Vec<f64> = samples.iter().map(|s| s.0).collect();
    let values: Vec<f64> = samples.iter().map(|s| s.1).collect();
    let column = ArrayView2::from_shape((values.len(), 1), &values).expect("one RSSI per packet");
    to_grid(&times, column, GRID_RATE_HZ)
}

fn side_by_side(grids: &[Array2<f64>]) -> Array2<f64> {
    let rows = grids.iter().map(|g| g.nrows()).min().unwrap_or(0);
    let views: Vec<ArrayView2<f64>> = grids.iter().map(|g| g.slice(s![..rows, ..])).collect();
    concatenate(Axis(1), &views).expect("same row count")
}

fn grid_samples(seconds: f64) -> usize {
    (seconds * GRID_RATE_HZ) as usize
}

fn profile(segment: ArrayView2<f64>) -> Vec<f64> {
    segment
        .mapv(|a| 20.0 * a.max(1e-3).log10())
        .mean_axis(Axis(0))
        .expect("non-empty window")
        .to_vec()
}

pub fn link_windows(capture: &Capture, channels: usize) -> LinkWindows {
    let packets = split_by_channel(capture, channels);
    if packets.iter().any(|p| p.times.len() < 2) {
        return LinkWindows::default();
    }
    let grids: Vec<Array2<f64>> = packets.iter().map(amplitude_grid).collect();
    let amplitude = side_by_side(&grids);
    let rssi = centred_rssi_grid(&packets);
    let (window, hop) = (grid_samples(WINDOW_SECONDS), grid_samples(HOP_SECONDS));
    let rows = amplitude.nrows().min(rssi.nrows());
    let mut result = LinkWindows::default();
    let mut start = 0;
    while start + window <= rows {
        let segment = amplitude.slice(s![start..start + window, ..]);
        let (mut row, rate) = window_features(segment, GRID_RATE_HZ);
        let wobble = rssi.slice(s![start..start + window, 0]).to_vec();
        row.push((std(&wobble) + 0.1).log10());
        row.extend(extra_features(segment, GRID_RATE_HZ));
        result.rows.push(row);
        result.breathing_rates.push(rate);
        result.profiles.push(profile(segment));
        start += hop;
    }
    result
}
