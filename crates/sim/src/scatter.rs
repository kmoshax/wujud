//! Bistatic point scatterers: static furniture and the moving chest/limb of a body.

use ndarray::Array2;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::band::{carrier_of, wavelength_of};
use crate::constants::{SPEED_OF_LIGHT, WAVELENGTH};
use crate::vec3::{distance, is_present, Vec3};

/// A point reflector with radar cross-section `rcs` (m²).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scatterer {
    pub position: Vec3,
    pub rcs: f64,
}

/// Bistatic radar equation in amplitude: `λ sqrt(σ) / ((4π)^1.5 d1 d2)`.
pub fn bistatic_amplitude(rcs: f64, to_tx: f64, to_rx: f64) -> f64 {
    bistatic_amplitude_at(WAVELENGTH, rcs, to_tx, to_rx)
}

pub fn bistatic_amplitude_at(wavelength: f64, rcs: f64, to_tx: f64, to_rx: f64) -> f64 {
    wavelength * rcs.sqrt() / ((4.0 * PI).powf(1.5) * to_tx * to_rx)
}

fn delay_phasor(frequency: f64, length: f64) -> Complex64 {
    Complex64::from_polar(1.0, -2.0 * PI * frequency * length / SPEED_OF_LIGHT)
}

/// Sum of the responses of static scatterers at each frequency (the clutter of a link).
pub fn static_rest(a: Vec3, b: Vec3, clutter: &[Scatterer], frequencies: &[f64]) -> Vec<Complex64> {
    let mut rest = vec![Complex64::new(0.0, 0.0); frequencies.len()];
    let wavelength = wavelength_of(frequencies);
    for scatterer in clutter {
        let to_tx = distance(scatterer.position, a);
        let to_rx = distance(scatterer.position, b);
        let amplitude = bistatic_amplitude_at(wavelength, scatterer.rcs, to_tx, to_rx);
        for (value, &f) in rest.iter_mut().zip(frequencies) {
            *value += amplitude * delay_phasor(f, to_tx + to_rx);
        }
    }
    rest
}

/// A moving point scatterer: its track and the position its spectrum is evaluated at.
#[derive(Clone, Copy, Debug)]
pub struct MovingScatterer<'a> {
    pub track: &'a [Vec3],
    /// The mean present position of the whole session's track, so any split into calls agrees.
    pub reference: Vec3,
    pub rcs: f64,
}

/// Adds `weight ×` the return of one moving point scatterer to `channel[t, k]`.
///
/// The spectrum is evaluated at the reference position; motion around it changes only the
/// carrier phase and the 1/L amplitude (the in-band delay change of a few cm is negligible).
pub fn add_moving_scatterer(
    channel: &mut Array2<Complex64>,
    a: Vec3,
    b: Vec3,
    scatterer: &MovingScatterer,
    frequencies: &[f64],
    weight: f64,
) {
    let reference = scatterer.reference;
    let to_tx = distance(reference, a);
    let to_rx = distance(reference, b);
    let reference_length = to_tx + to_rx;
    let carrier = carrier_of(frequencies);
    let amplitude = bistatic_amplitude_at(wavelength_of(frequencies), scatterer.rcs, to_tx, to_rx);
    let spectrum: Vec<Complex64> = frequencies
        .iter()
        .map(|&f| amplitude * delay_phasor(f, reference_length))
        .collect();
    for (t, &point) in scatterer.track.iter().enumerate() {
        if !is_present(point) {
            continue;
        }
        let length = distance(point, a) + distance(point, b);
        let modulation =
            delay_phasor(carrier, length - reference_length) * (reference_length / length.max(0.3));
        for (value, s) in channel.row_mut(t).iter_mut().zip(&spectrum) {
            *value += weight * modulation * s;
        }
    }
}

pub fn mean_present_position(track: &[Vec3]) -> Option<Vec3> {
    let present: Vec<&Vec3> = track.iter().filter(|p| is_present(**p)).collect();
    if present.is_empty() {
        return None;
    }
    let n = present.len() as f64;
    let sum = present
        .iter()
        .fold([0.0; 3], |s, p| [s[0] + p[0], s[1] + p[1], s[2] + p[2]]);
    Some([sum[0] / n, sum[1] / n, sum[2] / n])
}
