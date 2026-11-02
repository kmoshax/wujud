//! The person: the skeleton of skeleton.rs walking to where the network puts them, a fading trail of
//! where they went, and a ring that breathes with them while they hold still.

use std::collections::VecDeque;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

use bevy::prelude::*;
use wujud_model::room::Pose;

use crate::live_core::{RoomStatus, Snapshot};
use crate::skeleton::Gait;

use super::palette::{FAINT, MUTED, TEXT};
use super::{ease, floor, Latest};

const FOLLOW_RATE: f32 = 3.8;
const TURN_RATE: f32 = 3.0;
const TURN_ABOVE_M_PER_S: f32 = 0.06;
const TRAIL_SECONDS: f32 = 12.0;
const TRAIL_STEP_M: f32 = 0.04;
/// room_view::BREATHING_AFTER_STILL_SECONDS: the ring fills while the breathing is read.
pub const BREATHING_AFTER_STILL_SECONDS: f64 = 30.0;
const FADE_RATE: f32 = 3.0;

pub struct PersonPlugin;

impl Plugin for PersonPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Figure>()
            .add_systems(Update, (follow, trail_and_rings).chain());
    }
}

/// The drawn person: eased towards the tracked spot so it walks instead of jumping.
#[derive(Resource)]
pub struct Figure {
    pub at: Option<[f32; 2]>,
    heading: f32,
    pub gait: Gait,
    pub presence: f32,
    trail: VecDeque<([f32; 2], f32)>,
    pub head: Vec3,
    /// The spot is the tracker's (not the room's middle, used when the boards are not placed).
    pub located: bool,
}

impl Figure {
    /// Where the body faces, radians from +x: the way it last walked.
    pub fn heading(&self) -> f32 {
        self.heading
    }
}

impl Default for Figure {
    fn default() -> Self {
        Figure {
            at: None,
            heading: -FRAC_PI_2,
            gait: Gait::Stand,
            presence: 0.0,
            trail: VecDeque::new(),
            head: Vec3::ZERO,
            located: false,
        }
    }
}

fn gait_of(pose: Pose) -> Gait {
    match pose {
        Pose::Moving => Gait::Walk,
        Pose::Lie => Gait::Lie,
        Pose::Sit => Gait::Sit,
        _ => Gait::Stand,
    }
}

fn follow(time: Res<Time>, latest: Res<Latest>, mut figure: ResMut<Figure>) {
    let snapshot = &latest.0;
    let dt = time.delta_secs();
    let present = snapshot.person.is_some();
    figure.presence += (f32::from(u8::from(present)) - figure.presence) * ease(FADE_RATE, dt);
    let Some(person) = &snapshot.person else {
        if figure.presence < 0.01 {
            figure.at = None;
        }
        return;
    };
    let [width, depth, _] = snapshot.room_size;
    let target = person.at.unwrap_or([width / 2.0, depth / 2.0]);
    figure.located = person.at.is_some();
    figure.gait = gait_of(person.pose);
    let drawn = *figure.at.get_or_insert(target);
    let share = ease(FOLLOW_RATE, dt);
    let step = [(target[0] - drawn[0]) * share, (target[1] - drawn[1]) * share];
    let now = [drawn[0] + step[0], drawn[1] + step[1]];
    figure.at = Some(now);
    if step[0].hypot(step[1]) > TURN_ABOVE_M_PER_S * dt {
        let towards = step[1].atan2(step[0]);
        let turn = (towards - figure.heading + PI).rem_euclid(TAU) - PI;
        figure.heading += turn * ease(TURN_RATE, dt);
    }
    let seconds = time.elapsed_secs();
    let moved_enough = figure.trail.back().is_none_or(|(last, _)| (now[0] - last[0]).hypot(now[1] - last[1]) > TRAIL_STEP_M);
    if figure.located && moved_enough {
        figure.trail.push_back((now, seconds));
    }
    while figure.trail.front().is_some_and(|(_, t)| seconds - t > TRAIL_SECONDS) {
        figure.trail.pop_front();
    }
}

fn breath_phase(snapshot: &Snapshot, seconds: f32) -> f32 {
    snapshot.breath_bpm.map_or(0.0, |bpm| (TAU * bpm as f32 / 60.0 * seconds).sin())
}

fn trail_and_rings(time: Res<Time>, latest: Res<Latest>, figure: Res<Figure>, mut gizmos: Gizmos) {
    let snapshot = &latest.0;
    let seconds = time.elapsed_secs();
    let points: Vec<(Vec3, Color)> = figure
        .trail
        .iter()
        .chain(figure.at.filter(|_| figure.located).map(|at| (at, seconds)).iter())
        .map(|&(at, t)| {
            let freshness = (1.0 - (seconds - t) / TRAIL_SECONDS).clamp(0.0, 1.0);
            (floor(at, 0.01), TEXT.with_alpha(0.8 * freshness.powf(1.5)))
        })
        .collect();
    if points.len() >= 2 {
        gizmos.linestrip_gradient(points);
    }
    if let Some(truth) = snapshot.truth_xy {
        let centre = floor(truth, 0.015);
        let flat = Quat::from_rotation_x(FRAC_PI_2);
        let color = MUTED;
        gizmos.circle(Isometry3d::new(centre, flat), 0.18, color);
        gizmos.line(centre - Vec3::X * 0.1, centre + Vec3::X * 0.1, color);
        gizmos.line(centre - Vec3::Z * 0.1, centre + Vec3::Z * 0.1, color);
    }
    let Some(at) = figure.at else { return };
    let flat = Quat::from_rotation_x(FRAC_PI_2);
    let centre = floor(at, 0.02);
    if !matches!(snapshot.status, RoomStatus::Still { .. }) {
        return;
    }
    match snapshot.breath_bpm {
        Some(_) => {
            let radius = 0.5 + 0.1 * breath_phase(snapshot, seconds);
            gizmos.circle(Isometry3d::new(centre, flat), radius, TEXT.with_alpha(figure.presence));
        }
        None => {
            let filled = (snapshot.still_seconds / BREATHING_AFTER_STILL_SECONDS).min(1.0) as f32;
            gizmos.circle(Isometry3d::new(centre, flat), 0.5, FAINT);
            if filled > 0.0 {
                gizmos.arc_3d(TAU * filled, 0.5, Isometry3d::new(centre, Quat::IDENTITY), TEXT);
            }
        }
    }
}
