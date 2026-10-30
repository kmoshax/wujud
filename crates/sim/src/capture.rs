//! What a host receives from one transmitter -> receiver direction: the packets that were not lost.

use ndarray::{Array3, Axis};
use serde::{Deserialize, Serialize};

use crate::radio::{measure_with_noise, LinkRadio, Measurement, ReceiverNoise};
use ndarray::Array2;
use num_complex::Complex64;

/// Kept packets only. `frame[n]` is the simulation frame packet `n` was sent in (index into labels).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Capture {
    pub frame: Vec<u32>,
    /// Receive time in seconds.
    pub t: Vec<f64>,
    /// `[packets, subcarriers, 2]` int8 I/Q as the ESP32 reports it.
    pub iq: Array3<i8>,
    pub rssi: Vec<i16>,
}

impl Capture {
    /// Keeps the packets of `measurement` that were not lost; `times[t]` is frame `t`'s receive time.
    pub fn from_measurement(measurement: &Measurement, first_frame: usize, times: &[f64]) -> Self {
        let kept: Vec<usize> = (0..measurement.kept.len())
            .filter(|&t| measurement.kept[t])
            .collect();
        Capture {
            frame: kept.iter().map(|&t| (first_frame + t) as u32).collect(),
            t: kept.iter().map(|&t| times[t]).collect(),
            iq: measurement.iq.select(Axis(0), &kept),
            rssi: kept.iter().map(|&t| measurement.rssi[t]).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.t.len()
    }

    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }
}

/// Both captures of one direction: the empty-scene calibration and the session with people.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkCapture {
    pub tx: usize,
    pub rx: usize,
    pub calibration: Capture,
    pub session: Capture,
}

/// `first_frame / fs, ...`: packet times without jitter (calibration and hall captures).
pub fn frame_times(frames: std::ops::Range<usize>, fs: f64) -> Vec<f64> {
    frames.map(|t| t as f64 / fs).collect()
}

/// Measures `h` (frames `first_frame..`) with the receiver's continuing noise; packet times get
/// `time_jitter_s` of Gaussian host jitter.
pub fn receive(
    h: &Array2<Complex64>,
    radio: &LinkRadio,
    receiver: &mut ReceiverNoise,
    first_frame: usize,
    fs: f64,
    time_jitter_s: f64,
) -> Capture {
    let (frames, subcarriers) = h.dim();
    let (noise, jitter) = receiver.draw(frames, subcarriers, radio, fs);
    let times: Vec<f64> = jitter
        .iter()
        .enumerate()
        .map(|(t, z)| (first_frame + t) as f64 / fs + time_jitter_s * z)
        .collect();
    Capture::from_measurement(
        &measure_with_noise(h, radio, fs, &noise),
        first_frame,
        &times,
    )
}
