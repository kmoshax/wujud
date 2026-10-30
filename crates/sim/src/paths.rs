//! Propagation paths of one link: line of sight, first-order reflections, diffuse paths.

use num_complex::Complex64;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::band::wavelength_of;
use crate::constants::{SPEED_OF_LIGHT, WAVELENGTH};
use crate::random::{uniform, unit};
use crate::vec3::{add, distance, scale, sub, Vec3};

pub const FLOOR_GAIN: f64 = -0.5;
pub const CEILING_GAIN: f64 = -0.35;
/// Bodies block the floor bounce only partly (value from physics3d.py::link_paths3).
pub const FLOOR_SHADOW_SCALE: f64 = 0.7;

/// The axis-aligned box a scene lives in: x in `0..width`, y in `0..depth`, z in `0..height`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoomShape {
    pub width: f64,
    pub depth: f64,
    pub height: f64,
}

impl RoomShape {
    pub fn surface_area(&self) -> f64 {
        2.0 * self.width * self.depth + 2.0 * (self.width + self.depth) * self.height
    }
}

/// What the static paths of a scene need to know: its box, whether the floor is a flat mirror
/// (not in a tiered hall) and the reflection gain of the walls x=0, x=W, y=0, y=D.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Enclosure {
    pub shape: RoomShape,
    pub floor_reflects: bool,
    pub wall_gains: [f64; 4],
}

/// A polyline from transmitter to receiver with its complex reflection gain.
#[derive(Clone, Debug, PartialEq)]
pub struct PropagationPath {
    pub points: Vec<Vec3>,
    pub gain: Complex64,
    pub shadow_scale: f64,
    pub shadowable: bool,
}

impl PropagationPath {
    fn new(points: Vec<Vec3>, gain: Complex64) -> Self {
        PropagationPath {
            points,
            gain,
            shadow_scale: 1.0,
            shadowable: true,
        }
    }

    pub fn length(&self) -> f64 {
        self.points.windows(2).map(|s| distance(s[0], s[1])).sum()
    }

    /// Free-space response at each frequency: `g λ/(4πL) exp(-j2πfL/c)`, `λ` of the grid's carrier.
    pub fn response(&self, frequencies: &[f64]) -> Vec<Complex64> {
        path_response(self.length(), self.gain, frequencies)
    }
}

pub fn path_response(length: f64, gain: Complex64, frequencies: &[f64]) -> Vec<Complex64> {
    let spreading = gain * (wavelength_of(frequencies) / (4.0 * PI * length));
    frequencies
        .iter()
        .map(|f| spreading * Complex64::from_polar(1.0, -2.0 * PI * f * length / SPEED_OF_LIGHT))
        .collect()
}

/// Point on the plane `coordinate[axis] = plane` where `a -> b` reflects (image method).
pub fn reflection_point(a: Vec3, b: Vec3, axis: usize, plane: f64) -> Vec3 {
    let mut image = a;
    image[axis] = 2.0 * plane - a[axis];
    let along = (plane - image[axis]) / (b[axis] - image[axis] + 1e-12);
    let mut point = add(image, scale(sub(b, image), along));
    point[axis] = plane;
    point
}

/// Line of sight, the four wall reflections, the floor (if flat) and the ceiling reflection.
pub fn geometric_paths(a: Vec3, b: Vec3, enclosure: &Enclosure) -> Vec<PropagationPath> {
    let shape = enclosure.shape;
    let mut paths = vec![PropagationPath::new(vec![a, b], Complex64::new(1.0, 0.0))];
    let walls = [(0, 0.0), (0, shape.width), (1, 0.0), (1, shape.depth)];
    for ((axis, plane), gain) in walls.into_iter().zip(enclosure.wall_gains) {
        let point = reflection_point(a, b, axis, plane);
        paths.push(PropagationPath::new(
            vec![a, point, b],
            Complex64::new(gain, 0.0),
        ));
    }
    if enclosure.floor_reflects {
        let point = reflection_point(a, b, 2, 0.0);
        let mut floor = PropagationPath::new(vec![a, point, b], Complex64::new(FLOOR_GAIN, 0.0));
        floor.shadow_scale = FLOOR_SHADOW_SCALE;
        paths.push(floor);
    }
    let point = reflection_point(a, b, 2, shape.height);
    paths.push(PropagationPath::new(
        vec![a, point, b],
        Complex64::new(CEILING_GAIN, 0.0),
    ));
    paths
}

/// The random numbers behind one link's diffuse paths, drawn up front so tests can inject them.
#[derive(Clone, Debug, PartialEq)]
pub struct DiffuseDraws {
    pub via_points: Vec<Vec3>,
    pub rician_k_db: f64,
    pub delay_spread_s: f64,
    pub phase_turns: Vec<f64>,
}

impl DiffuseDraws {
    pub fn sample(rng: &mut ChaCha8Rng, shape: &RoomShape, count: usize) -> Self {
        let xs: Vec<f64> = (0..count)
            .map(|_| uniform(rng, 0.1, shape.width - 0.1))
            .collect();
        let ys: Vec<f64> = (0..count)
            .map(|_| uniform(rng, 0.1, shape.depth - 0.1))
            .collect();
        let zs: Vec<f64> = (0..count)
            .map(|_| uniform(rng, 0.2, shape.height - 0.2))
            .collect();
        let rician_k_db = uniform(rng, -3.0, 6.0);
        let delay_spread_s = uniform(rng, 15e-9, 40e-9);
        let phase_turns = (0..count).map(|_| unit(rng)).collect();
        DiffuseDraws {
            via_points: (0..count).map(|k| [xs[k], ys[k], zs[k]]).collect(),
            rician_k_db,
            delay_spread_s,
            phase_turns,
        }
    }
}

/// The reverberant field as paths `a -> random point -> b` with an exponential power-delay
/// profile whose total is `10^(-K/10)` times the line-of-sight power.
pub fn diffuse_paths(a: Vec3, b: Vec3, draws: &DiffuseDraws) -> Vec<PropagationPath> {
    let direct = distance(a, b);
    let lengths: Vec<f64> = draws
        .via_points
        .iter()
        .map(|&via| distance(via, a) + distance(via, b))
        .collect();
    let relative_power: Vec<f64> = lengths
        .iter()
        .map(|length| (-(length - direct) / SPEED_OF_LIGHT / draws.delay_spread_s).exp())
        .collect();
    let power_sum: f64 = relative_power.iter().sum();
    let total_power =
        (WAVELENGTH / (4.0 * PI * direct)).powi(2) * 10f64.powf(-draws.rician_k_db / 10.0);
    draws
        .via_points
        .iter()
        .enumerate()
        .map(|(k, &via)| {
            let amplitude = (relative_power[k] / power_sum * total_power).sqrt();
            let field = Complex64::from_polar(amplitude, 2.0 * PI * draws.phase_turns[k]);
            let gain = field / (WAVELENGTH / (4.0 * PI * lengths[k]));
            PropagationPath::new(vec![a, via, b], gain)
        })
        .collect()
}
