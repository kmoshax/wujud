//! The person as a skeleton: joints placed by forward kinematics for each thing the model can tell
//! (walking, standing, sitting or crouching, lying), animated over time and breathing with the
//! measured rate. The limbs are drawn, not measured: two boards cannot see where a hand is.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use crate::camera::Point3;

const PELVIS_Z: f32 = 0.95;
const THIGH: f32 = 0.44;
const SHIN: f32 = 0.43;
const UPPER_ARM: f32 = 0.29;
const FOREARM: f32 = 0.27;
const HIP_HALF_WIDTH: f32 = 0.1;
const SHOULDER_HALF_WIDTH: f32 = 0.19;
pub const HEAD_RADIUS: f32 = 0.1;
const STEPS_PER_SECOND: f32 = 0.9;
const SEAT_Z: f32 = 0.5;
const LYING_Z: f32 = 0.12;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Gait {
    Walk,
    Stand,
    Sit,
    Lie,
}

/// Left side first in every pair.
pub struct Joints {
    pub head: Point3,
    pub neck: Point3,
    pub chest: Point3,
    pub pelvis: Point3,
    pub shoulder: [Point3; 2],
    pub elbow: [Point3; 2],
    pub wrist: [Point3; 2],
    pub hip: [Point3; 2],
    pub knee: [Point3; 2],
    pub ankle: [Point3; 2],
    pub toe: [Point3; 2],
}

impl Joints {
    /// Bones as (from, to, thickness in metres), trunk first.
    pub fn bones(&self) -> Vec<(Point3, Point3, f32)> {
        let mut bones = vec![
            (self.pelvis, self.chest, 0.07),
            (self.chest, self.neck, 0.06),
            (self.neck, self.head, 0.035),
        ];
        for side in 0..2 {
            bones.extend([
                (self.chest, self.shoulder[side], 0.04),
                (self.shoulder[side], self.elbow[side], 0.04),
                (self.elbow[side], self.wrist[side], 0.032),
                (self.pelvis, self.hip[side], 0.05),
                (self.hip[side], self.knee[side], 0.055),
                (self.knee[side], self.ankle[side], 0.045),
                (self.ankle[side], self.toe[side], 0.03),
            ]);
        }
        bones
    }

    pub fn all(&self) -> Vec<Point3> {
        let mut joints = vec![self.neck, self.chest, self.pelvis];
        for side in 0..2 {
            joints.extend([
                self.shoulder[side],
                self.elbow[side],
                self.wrist[side],
                self.hip[side],
                self.knee[side],
                self.ankle[side],
            ]);
        }
        joints
    }
}

/// `breath` runs -1..1 with each breath; `heading` is where the body faces, radians from +x;
/// `under` is the height of what the body sits or lies on where it is (0 on the floor).
pub fn pose(gait: Gait, at: [f32; 2], heading: f32, seconds: f32, breath: f32, under: f32) -> Joints {
    let local = match gait {
        Gait::Walk => walking(seconds, breath),
        Gait::Stand => standing(seconds, breath),
        Gait::Sit => sitting(seconds, breath),
        Gait::Lie => lying(seconds, breath),
    };
    let (sin, cos) = heading.sin_cos();
    let raise = if gait == Gait::Lie { under } else { 0.0 };
    let place = |[f, l, z]: Point3| [at[0] + f * cos - l * sin, at[1] + f * sin + l * cos, z + raise];
    Joints {
        head: place(local.head),
        neck: place(local.neck),
        chest: place(local.chest),
        pelvis: place(local.pelvis),
        shoulder: local.shoulder.map(place),
        elbow: local.elbow.map(place),
        wrist: local.wrist.map(place),
        hip: local.hip.map(place),
        knee: local.knee.map(place),
        ankle: local.ankle.map(place),
        toe: local.toe.map(place),
    }
}

/// Angles in the body's forward/up plane, from straight down, forward positive.
struct Limb {
    swing: f32,
    bend: f32,
}

/// Body-local coordinates: forward, left, up.
fn body(pelvis: Point3, lean: f32, breath: f32, legs: [Limb; 2], arms: [Limb; 2]) -> Joints {
    let along = |from: Point3, angle: f32, length: f32| {
        [from[0] + length * angle.sin(), from[1], from[2] - length * angle.cos()]
    };
    let up = |from: Point3, length: f32| along(from, PI + lean, length);
    let chest = add(up(pelvis, 0.36), [0.015 * breath, 0.0, 0.01 * breath]);
    let neck = up(chest, 0.17);
    let head = up(neck, 0.15);
    let shoulder = [1.0, -1.0].map(|side| {
        [chest[0], chest[1] + side * (SHOULDER_HALF_WIDTH + 0.006 * breath), chest[2] + 0.11 + 0.008 * breath]
    });
    let hip = [1.0, -1.0].map(|side| [pelvis[0], pelvis[1] + side * HIP_HALF_WIDTH, pelvis[2] - 0.03]);
    let elbow = [0, 1].map(|s| along(shoulder[s], arms[s].swing, UPPER_ARM));
    let wrist = [0, 1].map(|s| along(elbow[s], arms[s].swing + arms[s].bend, FOREARM));
    let knee = [0, 1].map(|s| along(hip[s], legs[s].swing, THIGH));
    let ankle = [0, 1].map(|s| along(knee[s], legs[s].swing - legs[s].bend, SHIN));
    let toe = [0, 1].map(|s| add(ankle[s], [0.14, 0.0, -0.04]));
    Joints {
        head,
        neck,
        chest,
        pelvis,
        shoulder,
        elbow,
        wrist,
        hip,
        knee,
        ankle,
        toe,
    }
}

fn walking(seconds: f32, breath: f32) -> Joints {
    let phase = seconds * TAU * STEPS_PER_SECOND / 2.0;
    let leg = |p: f32| Limb {
        swing: 0.42 * p.sin(),
        bend: 0.08 + 0.65 * p.cos().max(0.0),
    };
    let arm = |p: f32| Limb {
        swing: -0.38 * p.sin(),
        bend: 0.25 + 0.15 * (-p.sin()).max(0.0),
    };
    let bob = 0.025 * (2.0 * phase).cos();
    body(
        [0.0, 0.0, PELVIS_Z - 0.02 + bob],
        0.06,
        breath,
        [leg(phase), leg(phase + PI)],
        [arm(phase), arm(phase + PI)],
    )
}

fn standing(seconds: f32, breath: f32) -> Joints {
    let sway = 0.012 * (seconds * 0.7).sin();
    let still = |swing: f32, bend: f32| Limb { swing, bend };
    body(
        [0.0, sway, PELVIS_Z],
        0.0,
        breath,
        [still(0.03, 0.05), still(-0.02, 0.04)],
        [still(0.06, 0.15), still(0.04, 0.18)],
    )
}

/// On a seat, or crouching: thighs forward, shins down, hands on the knees.
fn sitting(seconds: f32, breath: f32) -> Joints {
    let settle = 0.006 * (seconds * 0.5).sin();
    let leg = |spread: f32| Limb {
        swing: FRAC_PI_2 - 0.08 + spread,
        bend: FRAC_PI_2 - 0.08,
    };
    let arm = Limb {
        swing: 0.55,
        bend: 0.75,
    };
    body(
        [0.0, 0.0, SEAT_Z + settle],
        -0.08,
        breath,
        [leg(0.0), leg(0.0)],
        [Limb { ..arm }, Limb { ..arm }],
    )
}

/// On the back: the standing body turned so it points along the floor, chest up.
fn lying(seconds: f32, breath: f32) -> Joints {
    let rest = |swing: f32, bend: f32| Limb { swing, bend };
    let upright = body(
        [0.0, 0.0, PELVIS_Z],
        0.0,
        breath * 1.6,
        [rest(0.0, 0.05), rest(0.0, 0.05)],
        [rest(0.1, 0.1), rest(0.1, 0.1)],
    );
    let _ = seconds;
    let lay = |[f, l, z]: Point3| [z - PELVIS_Z, l * 1.1, LYING_Z + f];
    Joints {
        head: lay(upright.head),
        neck: lay(upright.neck),
        chest: lay(upright.chest),
        pelvis: lay(upright.pelvis),
        shoulder: upright.shoulder.map(lay),
        elbow: upright.elbow.map(lay),
        wrist: upright.wrist.map(lay),
        hip: upright.hip.map(lay),
        knee: upright.knee.map(lay),
        ankle: upright.ankle.map(lay),
        toe: upright.toe.map(|[f, l, z]| lay([f - 0.14, l, z + 0.14])),
    }
}

fn add(a: Point3, b: Point3) -> Point3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standing_feet_on_the_floor_and_head_up_high() {
        let body = pose(Gait::Stand, [1.0, 1.0], 0.0, 0.0, 0.0, 0.0);
        assert!(body.ankle.iter().all(|a| a[2] > 0.0 && a[2] < 0.15));
        assert!(body.head[2] > 1.55 && body.head[2] < 1.85);
    }

    #[test]
    fn sitting_is_lower_and_lying_is_flat() {
        let stand = pose(Gait::Stand, [0.0, 0.0], 0.0, 0.0, 0.0, 0.0);
        let sit = pose(Gait::Sit, [0.0, 0.0], 0.0, 0.0, 0.0, 0.0);
        let lie = pose(Gait::Lie, [0.0, 0.0], 0.0, 0.0, 0.0, 0.0);
        assert!(sit.head[2] < stand.head[2] - 0.35);
        assert!(sit.ankle.iter().all(|a| a[2] > -0.05 && a[2] < 0.2));
        assert!(lie.all().iter().all(|j| j[2] < 0.45));
        assert!((lie.head[0] - lie.ankle[0][0]).abs() > 1.2);
    }

    #[test]
    fn walking_swings_the_legs_in_turn() {
        let quarter = 1.0 / (2.0 * STEPS_PER_SECOND);
        let body = pose(Gait::Walk, [0.0, 0.0], 0.0, quarter, 0.0, 0.0);
        assert!((body.ankle[0][0] - body.ankle[1][0]).abs() > 0.3);
    }
}
