//! Window features from the per-packet CSI / RSSI stream of one link.

mod csi;
mod grid;
mod rssi;
mod stream;
mod window;

pub use csi::{amplitude_from_iq, phase_from_iq, signal_subcarriers};
pub use grid::to_grid;
pub use rssi::{rssi_window, rssi_windows, RSSI_FEATURE_NAMES};
pub use stream::{phase_windows, profile_windows, windows, WindowSet};
pub use window::{extra_features, spectral_features, window_features, BreathingRate};

pub const GRID_RATE_HZ: f64 = 20.0;
pub const WINDOW_SECONDS: f64 = 20.0;
pub const HOP_SECONDS: f64 = 5.0;

pub const BREATH_BAND_HZ: (f64, f64) = (0.12, 0.6);
pub const MICRO_BAND_HZ: (f64, f64) = (0.6, 3.0);
pub const NOISE_BAND_HZ: (f64, f64) = (4.0, 9.5);

pub const CORE_FEATURE_NAMES: [&str; 10] = [
    "br_med", "br_top", "mi_med", "mi_top", "std_med", "std_p90", "br_peak", "pc_share", "br_frac",
    "mi_frac",
];
pub const RSSI_COLUMN_NAME: &str = "rssi_std";
pub const EXTRA_FEATURE_NAMES: [&str; 6] = ["line", "flat", "step", "burst", "spread", "glrt"];
pub const PHASE_FEATURE_NAMES: [&str; 10] = [
    "ph_br_med",
    "ph_br_top",
    "ph_mi_med",
    "ph_mi_top",
    "ph_std_med",
    "ph_std_p90",
    "ph_br_peak",
    "ph_pc_share",
    "ph_br_frac",
    "ph_mi_frac",
];

/// Index of the breathing GLRT in a 17-column (core + RSSI + extra) window row.
pub const GLRT_COLUMN_WITH_RSSI: usize = 16;

/// Python returns these features as float32; the trees were trained on the rounded values.
fn round_like_float32(values: &mut [f64]) {
    for v in values {
        *v = *v as f32 as f64;
    }
}
