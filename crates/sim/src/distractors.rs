//! Things that move but are not people: fans, pets, robot vacuums, doors (the classic false alarms).

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::bodies::Body;
use crate::motion::{velocities, wander, Gait, TargetArea};
use crate::paths::RoomShape;
use crate::random::{chance, pick, uniform, SimRng};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistractorKind {
    Fan,
    Pet,
    Vacuum,
    Door,
}

/// Port of scenes3d.py::distractor.
pub fn distractor(
    rng: &mut SimRng,
    kind: DistractorKind,
    shape: &RoomShape,
    frames: usize,
    fs: f64,
) -> Body {
    match kind {
        DistractorKind::Fan => fan(rng, shape, frames, fs),
        DistractorKind::Pet | DistractorKind::Vacuum => roamer(rng, kind, shape, frames, fs),
        DistractorKind::Door => door(rng, shape, frames, fs),
    }
}

fn constant_box(
    centre: Vec<[f64; 2]>,
    z: (f64, f64),
    half_size: (f64, f64),
    yaw: Vec<f64>,
    absorption: f64,
) -> Body {
    let frames = centre.len();
    Body {
        z_bottom: vec![z.0; frames],
        z_top: vec![z.1; frames],
        half_width: vec![half_size.0; frames],
        half_depth: vec![half_size.1; frames],
        chest: Vec::new(),
        limb: Vec::new(),
        centre,
        yaw,
        absorption,
        chest_rcs: 0.0,
        limb_rcs: 0.0,
    }
}

/// A standing or ceiling fan: three blades turning at 12-22 rev/s act as a rotating scatterer.
fn fan(rng: &mut SimRng, shape: &RoomShape, frames: usize, fs: f64) -> Body {
    let spot = [
        uniform(rng, 0.5, shape.width - 0.5),
        uniform(rng, 0.5, shape.depth - 0.5),
    ];
    let hub_z = if chance(rng, 0.3) {
        pick(rng, &[1.1, 2.6])
    } else {
        uniform(rng, 1.0, 1.4)
    };
    let blade_hz = uniform(rng, 12.0, 22.0) * 3.0;
    let radius = uniform(rng, 0.15, 0.25);
    let mut body = constant_box(
        vec![spot; frames],
        (hub_z - 0.25, hub_z + 0.25),
        (0.25, 0.05),
        vec![0.0; frames],
        0.15,
    );
    body.chest = vec![[spot[0], spot[1], hub_z]; frames];
    body.limb = (0..frames)
        .map(|t| {
            let angle = 2.0 * PI * blade_hz * t as f64 / fs;
            [
                spot[0] + radius * angle.cos(),
                spot[1],
                hub_z + radius * angle.sin(),
            ]
        })
        .collect();
    body.chest_rcs = 0.02;
    body.limb_rcs = uniform(rng, 0.03, 0.12);
    body
}

/// A cat/small dog (low, fast breathing, small RCS) or a robot vacuum, wandering with pauses.
fn roamer(
    rng: &mut SimRng,
    kind: DistractorKind,
    shape: &RoomShape,
    frames: usize,
    fs: f64,
) -> Body {
    let is_pet = kind == DistractorKind::Pet;
    let speed = if is_pet {
        uniform(rng, 0.2, 0.9)
    } else {
        uniform(rng, 0.2, 0.35)
    };
    let area = TargetArea {
        x_range: (0.4, shape.width - 0.4),
        y_range: (0.4, shape.depth - 0.4),
    };
    let start = area.sample(rng);
    let gait = Gait {
        speed_m_s: speed,
        arrive_within_m: 0.1,
        mean_pause_s: if is_pet { 6.0 } else { 0.5 },
    };
    let track = wander(rng, start, start, &area, &gait, frames, fs);
    let yaw: Vec<f64> = velocities(&track, fs)
        .iter()
        .map(|v| v[1].atan2(v[0]))
        .collect();
    if is_pet {
        pet_body(rng, track, yaw, fs)
    } else {
        let mut body = constant_box(track.clone(), (0.0, 0.1), (0.17, 0.17), yaw, 0.9);
        body.chest = track.iter().map(|&[x, y]| [x, y, 0.06]).collect();
        body.limb = body.chest.clone();
        body.chest_rcs = uniform(rng, 0.05, 0.15);
        body
    }
}

fn pet_body(rng: &mut SimRng, track: Vec<[f64; 2]>, yaw: Vec<f64>, fs: f64) -> Body {
    let height = uniform(rng, 0.2, 0.45);
    let breath_hz = uniform(rng, 0.4, 0.8);
    let half_width = uniform(rng, 0.18, 0.3);
    let chest = track
        .iter()
        .enumerate()
        .map(|(t, &[x, y])| {
            [
                x,
                y,
                height * 0.6 + 0.0015 * (2.0 * PI * breath_hz * t as f64 / fs).sin(),
            ]
        })
        .collect();
    let limb = track
        .iter()
        .zip(&yaw)
        .enumerate()
        .map(|(t, (&[x, y], &heading))| {
            let gait = 0.1 * (2.0 * PI * 2.5 * t as f64 / fs).sin();
            [x + heading.cos() * gait, y + heading.sin() * gait, 0.1]
        })
        .collect();
    let mut body = constant_box(track, (0.05, height), (half_width, 0.08), yaw, 0.6);
    body.chest = chest;
    body.limb = limb;
    body.chest_rcs = uniform(rng, 0.02, 0.08);
    body.limb_rcs = 0.01;
    body
}

/// A 0.85 m door on the y=0 wall that swings open or closed every 15-40 s.
fn door(rng: &mut SimRng, shape: &RoomShape, frames: usize, fs: f64) -> Body {
    const DOOR_WIDTH: f64 = 0.85;
    let hinge = [uniform(rng, 0.5, shape.width - 1.5), 0.05];
    let angle = door_angles(rng, frames, fs);
    let centre: Vec<[f64; 2]> = angle
        .iter()
        .map(|a| {
            [
                hinge[0] + 0.5 * DOOR_WIDTH * a.cos(),
                hinge[1] + 0.5 * DOOR_WIDTH * a.sin(),
            ]
        })
        .collect();
    let mut body = constant_box(
        centre.clone(),
        (0.0, 2.0),
        (DOOR_WIDTH / 2.0, 0.02),
        angle,
        0.4,
    );
    body.chest = centre.iter().map(|&[x, y]| [x, y, 1.0]).collect();
    body.limb = body.chest.clone();
    body.chest_rcs = 0.3;
    body
}

fn door_angles(rng: &mut SimRng, frames: usize, fs: f64) -> Vec<f64> {
    let mut angle = vec![0.0; frames];
    let mut k = 0;
    while k < frames {
        k += (uniform(rng, 15.0, 40.0) * fs) as usize;
        let duration = (uniform(rng, 1.0, 2.5) * fs) as usize;
        let current = angle[k.min(frames - 1)];
        let target = if current < 0.1 {
            uniform(rng, 0.3, 1.5)
        } else {
            0.0
        };
        for (i, value) in angle
            .iter_mut()
            .enumerate()
            .take(frames.min(k + duration))
            .skip(k)
        {
            let progress = (i - k) as f64 / duration as f64;
            *value = current + (target - current) * (0.5 - 0.5 * (PI * progress).cos());
        }
        if k + duration < frames {
            angle[k + duration..].fill(target);
        }
        k += duration;
    }
    angle
}
