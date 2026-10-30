//! Feature ports against `research/wujud/roomfeat.py` on real ESP32 recordings.

mod common;

use common::*;
use ndarray::{s, Array2};
use wujud_model::features::{
    amplitude_from_iq, phase_from_iq, phase_windows, profile_windows, to_grid, windows,
    GRID_RATE_HZ, HOP_SECONDS, WINDOW_SECONDS,
};

const BIT_EXACT: f64 = 0.0;
/// Measured: f32-rounded features identical, the f64 RSSI column within 1.2e-15.
const FEATURE_TOLERANCE: f64 = 1e-12;
/// Measured: 2.9e-14 (lstsq vs normal equations).
const PHASE_TOLERANCE: f64 = 1e-12;
/// Measured: 2.2e-15.
const GRID_TOLERANCE: f64 = 1e-13;

fn amplitude_f64(recording: &Recording) -> Array2<f64> {
    amplitude_from_iq(recording.iq.view()).mapv(f64::from)
}

fn odd_window_seconds(times: &[f64]) -> f64 {
    let span = times[times.len() - 1] - times[0];
    (span - 0.5).min(20.0) - 0.37
}

#[test]
fn amplitude_matches_numpy_bit_for_bit() {
    for name in RECORDINGS {
        let amplitude = amplitude_from_iq(recording(name).iq.view());
        let expected = read_2d::<f32>(&format!("feat_{name}_amp_head"));
        let head = amplitude.slice(s![..expected.nrows(), ..]);
        assert_eq!(head.dim(), expected.dim(), "{name}");
        assert_eq!(head, expected, "{name}");
    }
}

#[test]
fn grid_matches() {
    for name in RECORDINGS {
        let recording = recording(name);
        let grid = to_grid(
            &recording.times,
            amplitude_f64(&recording).view(),
            GRID_RATE_HZ,
        );
        let expected = read_2d::<f64>(&format!("feat_{name}_grid_head"));
        let head = grid.slice(s![..expected.nrows(), ..]);
        assert_close(
            name,
            head.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            GRID_TOLERANCE,
        );
    }
}

#[test]
fn phase_matches() {
    for name in RECORDINGS {
        let phase = phase_from_iq(recording(name).iq.view());
        let expected = read_2d::<f64>(&format!("feat_{name}_phase_head"));
        let head = phase.slice(s![..expected.nrows(), ..]).to_owned();
        assert_eq!(phase.ncols(), expected.ncols(), "{name}");
        assert_close(
            name,
            head.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            PHASE_TOLERANCE,
        );
    }
}

fn assert_rows(label: &str, rows: &[Vec<f64>], expected: &Array2<f64>, tolerance: f64) {
    assert_eq!(rows.len(), expected.nrows(), "{label}: window count");
    let actual = rows_to_array(rows, expected.ncols());
    assert_close(
        label,
        actual.as_slice().unwrap(),
        expected.as_slice().unwrap(),
        tolerance,
    );
}

#[test]
fn amplitude_windows_match() {
    for name in RECORDINGS {
        let recording = recording(name);
        let amplitude = amplitude_f64(&recording);
        let with_rssi = windows(
            &recording.times,
            amplitude.view(),
            Some(&recording.rssi),
            WINDOW_SECONDS,
            HOP_SECONDS,
        );
        let label = format!("{name} windows");
        assert_rows(
            &label,
            &with_rssi.rows,
            &read_2d(&format!("feat_{name}_windows")),
            FEATURE_TOLERANCE,
        );
        let rates = read_1d::<f64>(&format!("feat_{name}_rates")).to_vec();
        assert_close(&label, &with_rssi.breathing_rates, &rates, BIT_EXACT);

        let without_rssi = windows(
            &recording.times,
            amplitude.view(),
            None,
            WINDOW_SECONDS,
            HOP_SECONDS,
        );
        let expected = read_2d_f32_as_f64(&format!("feat_{name}_windows_no_rssi"));
        assert_rows(
            &format!("{name} no rssi"),
            &without_rssi.rows,
            &expected,
            FEATURE_TOLERANCE,
        );

        let odd = windows(
            &recording.times,
            amplitude.view(),
            Some(&recording.rssi),
            odd_window_seconds(&recording.times),
            100.0,
        );
        let label = format!("{name} odd window");
        assert_rows(
            &label,
            &odd.rows,
            &read_2d(&format!("feat_{name}_windows_odd")),
            FEATURE_TOLERANCE,
        );
        let rates = read_1d::<f64>(&format!("feat_{name}_rates_odd")).to_vec();
        assert_close(&label, &odd.breathing_rates, &rates, BIT_EXACT);
    }
}

#[test]
fn phase_and_profile_windows_match() {
    for name in RECORDINGS {
        let recording = recording(name);
        let phase = phase_from_iq(recording.iq.view());
        let rows = phase_windows(&recording.times, phase.view(), WINDOW_SECONDS, HOP_SECONDS);
        let expected = read_2d_f32_as_f64(&format!("feat_{name}_phase_windows"));
        assert_rows(
            &format!("{name} phase"),
            &rows,
            &expected,
            FEATURE_TOLERANCE,
        );

        let amplitude = amplitude_f64(&recording);
        let rows = profile_windows(
            &recording.times,
            amplitude.view(),
            WINDOW_SECONDS,
            HOP_SECONDS,
        );
        let expected = read_2d::<f64>(&format!("feat_{name}_profile_windows"));
        assert_rows(&format!("{name} profile"), &rows, &expected, GRID_TOLERANCE);
    }
}
