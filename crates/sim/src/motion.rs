//! Walking trajectories and the standing/walking body built around a trajectory.

use std::f64::consts::{FRAC_PI_2, PI};

use crate::bodies::Body;
use crate::random::{exponential, uniform, SimRng};

/// A rectangle `x in x_range, y in y_range` that random walk targets are drawn from.
#[derive(Clone, Copy, Debug)]
pub struct TargetArea {
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
}

impl TargetArea {
    pub fn sample(&self, rng: &mut SimRng) -> [f64; 2] {
        let x = uniform(rng, self.x_range.0, self.x_range.1);
        let y = uniform(rng, self.y_range.0, self.y_range.1);
        [x, y]
    }
}

/// How a walker moves: speed, how close counts as arrived, and the mean pause on arrival.
#[derive(Clone, Copy, Debug)]
pub struct Gait {
    pub speed_m_s: f64,
    pub arrive_within_m: f64,
    pub mean_pause_s: f64,
}

/// Walks from `start` towards `first_target`, then to random targets in `area`, pausing on arrival.
pub fn wander(
    rng: &mut SimRng,
    start: [f64; 2],
    first_target: [f64; 2],
    area: &TargetArea,
    gait: &Gait,
    frames: usize,
    fs: f64,
) -> Vec<[f64; 2]> {
    let mut position = start;
    let mut target = first_target;
    let mut pause_frames = 0usize;
    let mut track = Vec::with_capacity(frames);
    for _ in 0..frames {
        if pause_frames > 0 {
            pause_frames -= 1;
        } else {
            let offset = [target[0] - position[0], target[1] - position[1]];
            let remaining = offset[0].hypot(offset[1]);
            if remaining < gait.arrive_within_m {
                target = area.sample(rng);
                if gait.mean_pause_s > 0.0 {
                    pause_frames = (exponential(rng, gait.mean_pause_s) * fs) as usize;
                }
            } else {
                position = step_towards(position, offset, remaining, gait.speed_m_s / fs);
            }
        }
        track.push(position);
    }
    track
}

pub fn step_towards(
    position: [f64; 2],
    offset: [f64; 2],
    remaining: f64,
    max_step: f64,
) -> [f64; 2] {
    let step = remaining.min(max_step) / remaining;
    [
        position[0] + offset[0] * step,
        position[1] + offset[1] * step,
    ]
}

/// Velocity by backward difference (zero at the first frame), as `[vx, vy]` in m/s.
pub fn velocities(track: &[[f64; 2]], fs: f64) -> Vec<[f64; 2]> {
    let mut velocity = vec![[0.0, 0.0]; track.len()];
    for t in 1..track.len() {
        velocity[t] = [
            (track[t][0] - track[t - 1][0]) * fs,
            (track[t][1] - track[t - 1][1]) * fs,
        ];
    }
    velocity
}

/// A standing or walking person following `track` (scenes3d.py::_standing): faces where it
/// walks, swings an arm while moving, breathes.
pub fn standing_body(rng: &mut SimRng, track: &[[f64; 2]], fs: f64, floor_z: f64) -> Body {
    let breath_amplitude = uniform(rng, 0.003, 0.006);
    let breath_hz = uniform(rng, 0.2, 0.33);
    let velocity = velocities(track, fs);
    let mut body = Body {
        centre: track.to_vec(),
        z_bottom: vec![floor_z; track.len()],
        z_top: vec![floor_z + 1.70; track.len()],
        half_width: vec![0.21; track.len()],
        half_depth: Vec::with_capacity(track.len()),
        yaw: Vec::with_capacity(track.len()),
        absorption: 0.9,
        chest: Vec::with_capacity(track.len()),
        limb: Vec::with_capacity(track.len()),
        chest_rcs: 0.6,
        limb_rcs: 0.1,
    };
    for (t, (&[x, y], &[vx, vy])) in track.iter().zip(&velocity).enumerate() {
        let time = t as f64 / fs;
        let breath = breath_amplitude * (2.0 * PI * breath_hz * time).sin();
        let yaw = vy.atan2(vx) - FRAC_PI_2;
        let speed = vx.hypot(vy);
        let swing = if speed > 0.1 {
            0.25 * (2.0 * PI * speed.max(0.3) / 0.8 * time).sin()
        } else {
            0.0
        };
        let forward = [(yaw + FRAC_PI_2).cos(), (yaw + FRAC_PI_2).sin()];
        body.yaw.push(yaw);
        body.half_depth.push(0.13 + breath);
        body.limb.push([
            x + forward[0] * swing,
            y + forward[1] * swing,
            floor_z + 0.9,
        ]);
        body.chest.push([
            x + forward[0] * breath,
            y + forward[1] * breath,
            floor_z + 1.3,
        ]);
    }
    body
}
