//! The person as a simple, smooth body: rounded capsules on human proportions, posed by forward
//! kinematics from a handful of joint angles that ease from one posture into the next.

use std::f32::consts::{FRAC_PI_2, TAU};

use bevy::prelude::*;

use crate::skeleton::Gait;

use super::palette::{glowing, mix, FAINT, MUTED, RED, TEXT};
use super::person::Figure;
use super::{ease, floor, world, Latest};

/// Proportions of a 1.75 m adult (Drillis and Contini's segment lengths as shares of height).
const HEIGHT: f32 = 1.75;
const HIP_HEIGHT: f32 = 0.53 * HEIGHT;
const TORSO: f32 = 0.29 * HEIGHT;
const NECK: f32 = 0.05 * HEIGHT;
const HEAD_RADIUS: f32 = HEIGHT / 7.5 / 2.0;
const SHOULDER_HALF_WIDTH: f32 = 0.125 * HEIGHT - 0.04;
const HIP_HALF_WIDTH: f32 = 0.05 * HEIGHT;
const UPPER_ARM: f32 = 0.186 * HEIGHT;
const FOREARM: f32 = 0.146 * HEIGHT;
const THIGH: f32 = 0.245 * HEIGHT;
const SHIN: f32 = 0.246 * HEIGHT;
const ANKLE_HEIGHT: f32 = 0.039 * HEIGHT;
const FOOT: f32 = 0.15 * HEIGHT;
const STRIDES_PER_SECOND: f32 = 0.95;
const SEAT_HEIGHT: f32 = 0.46;
const SETTLED_RATE: f32 = 14.0;
const CHANGING_RATE: f32 = 2.5;
const SETTLE_SECONDS: f32 = 1.2;

pub struct BodyPlugin;

impl Plugin for BodyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Motion>().add_systems(Startup, spawn_body).add_systems(Update, pose_body);
    }
}

/// Joint angles in radians, forward positive, left side first; `recline` turns the whole body back
/// onto the floor (a quarter turn is lying on the back).
#[derive(Clone, Copy, Default)]
struct Posture {
    pelvis_height: f32,
    recline: f32,
    lean: f32,
    hip: [f32; 2],
    knee: [f32; 2],
    shoulder: [f32; 2],
    elbow: [f32; 2],
    arm_out: f32,
}

impl Posture {
    fn toward(&mut self, goal: &Posture, share: f32) {
        let step = |now: &mut f32, to: f32| *now += (to - *now) * share;
        step(&mut self.pelvis_height, goal.pelvis_height);
        step(&mut self.recline, goal.recline);
        step(&mut self.lean, goal.lean);
        step(&mut self.arm_out, goal.arm_out);
        for side in 0..2 {
            step(&mut self.hip[side], goal.hip[side]);
            step(&mut self.knee[side], goal.knee[side]);
            step(&mut self.shoulder[side], goal.shoulder[side]);
            step(&mut self.elbow[side], goal.elbow[side]);
        }
    }
}

fn walking(seconds: f32) -> Posture {
    let phase = seconds * TAU * STRIDES_PER_SECOND;
    let leg = |p: f32| (0.38 * p.sin(), 0.08 + 0.75 * p.cos().max(0.0).powf(1.5));
    let ([hip_l, hip_r], [knee_l, knee_r]) = {
        let (l, r) = (leg(phase), leg(phase + std::f32::consts::PI));
        ([l.0, r.0], [l.1, r.1])
    };
    Posture {
        pelvis_height: HIP_HEIGHT - 0.035 + 0.022 * (2.0 * phase).cos(),
        recline: 0.0,
        lean: 0.06,
        hip: [hip_l, hip_r],
        knee: [knee_l, knee_r],
        shoulder: [-0.32 * phase.sin(), 0.32 * phase.sin()],
        elbow: [0.3 + 0.2 * (-phase.sin()).max(0.0), 0.3 + 0.2 * phase.sin().max(0.0)],
        arm_out: 0.06,
    }
}

fn standing(seconds: f32) -> Posture {
    let sway = 0.015 * (seconds * 0.6).sin();
    Posture {
        pelvis_height: HIP_HEIGHT,
        recline: 0.0,
        lean: sway,
        hip: [0.02, -0.02],
        knee: [0.04, 0.04],
        shoulder: [0.04, 0.02],
        elbow: [0.18, 0.2],
        arm_out: 0.09,
    }
}

fn sitting() -> Posture {
    Posture {
        pelvis_height: SEAT_HEIGHT + 0.06,
        recline: 0.0,
        lean: -0.06,
        hip: [FRAC_PI_2 - 0.05; 2],
        knee: [FRAC_PI_2 - 0.05; 2],
        shoulder: [0.35; 2],
        elbow: [0.85; 2],
        arm_out: 0.1,
    }
}

fn lying(under: f32) -> Posture {
    Posture {
        pelvis_height: under + 0.11,
        recline: FRAC_PI_2,
        lean: 0.0,
        hip: [0.0; 2],
        knee: [0.06; 2],
        shoulder: [0.05; 2],
        elbow: [0.1; 2],
        arm_out: 0.16,
    }
}

/// The body being drawn, easing towards the posture of what the person does.
#[derive(Resource, Default)]
struct Motion {
    posture: Option<Posture>,
    gait: Option<Gait>,
    gait_seconds: f32,
}

/// Where every joint is, in the room's metres (x, y, z up).
struct Joints {
    head: Vec3,
    neck: Vec3,
    chest: Vec3,
    pelvis: Vec3,
    shoulder: [Vec3; 2],
    elbow: [Vec3; 2],
    wrist: [Vec3; 2],
    hip: [Vec3; 2],
    knee: [Vec3; 2],
    ankle: [Vec3; 2],
    toe: [Vec3; 2],
}

/// Down a limb from `from`, turned forward by `angle` and out to the side by `out` (body frame:
/// x forward, y left, z up).
fn limb(from: Vec3, angle: f32, out: f32, length: f32) -> Vec3 {
    from + Vec3::new(angle.sin() * out.cos(), out.sin(), -angle.cos() * out.cos()) * length
}

fn joints(posture: &Posture, breath: f32) -> Joints {
    let pelvis = Vec3::ZERO;
    let up = Vec3::new(posture.lean.sin(), 0.0, posture.lean.cos());
    let chest = pelvis + up * (TORSO + 0.006 * breath);
    let neck = chest + up * NECK;
    let head = neck + up * (HEAD_RADIUS + 0.02);
    let shoulder = [1.0, -1.0].map(|side| chest + Vec3::new(0.0, side * (SHOULDER_HALF_WIDTH + 0.004 * breath), -0.03));
    let hip = [1.0, -1.0].map(|side| pelvis + Vec3::new(0.0, side * HIP_HALF_WIDTH, -0.02));
    let side_of = |s: usize| if s == 0 { 1.0 } else { -1.0 };
    let elbow = [0, 1].map(|s| limb(shoulder[s], posture.shoulder[s], side_of(s) * posture.arm_out, UPPER_ARM));
    let wrist = [0, 1].map(|s| limb(elbow[s], posture.shoulder[s] + posture.elbow[s], side_of(s) * posture.arm_out * 0.5, FOREARM));
    let knee = [0, 1].map(|s| limb(hip[s], posture.hip[s], 0.0, THIGH));
    let ankle = [0, 1].map(|s| limb(knee[s], posture.hip[s] - posture.knee[s], 0.0, SHIN));
    let foot_angle = |s: usize| posture.hip[s] - posture.knee[s] + FRAC_PI_2;
    let toe = [0, 1].map(|s| limb(ankle[s], foot_angle(s), 0.0, FOOT) + Vec3::new(0.0, 0.0, -ANKLE_HEIGHT * 0.3));
    let recline = Quat::from_rotation_y(-posture.recline);
    let lift = Vec3::new(0.0, 0.0, posture.pelvis_height);
    let place = |p: Vec3| recline * p + lift;
    Joints {
        head: place(head),
        neck: place(neck),
        chest: place(chest),
        pelvis: place(pelvis),
        shoulder: shoulder.map(place),
        elbow: elbow.map(place),
        wrist: wrist.map(place),
        hip: hip.map(place),
        knee: knee.map(place),
        ankle: ankle.map(place),
        toe: toe.map(place),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Part {
    Head,
    Neck,
    Chest,
    Belly,
    Shoulders,
    Hips,
    UpperArm(usize),
    Forearm(usize),
    Thigh(usize),
    Shin(usize),
    Foot(usize),
}

const PARTS: [Part; 16] = [
    Part::Head,
    Part::Neck,
    Part::Chest,
    Part::Belly,
    Part::Shoulders,
    Part::Hips,
    Part::UpperArm(0),
    Part::UpperArm(1),
    Part::Forearm(0),
    Part::Forearm(1),
    Part::Thigh(0),
    Part::Thigh(1),
    Part::Shin(0),
    Part::Shin(1),
    Part::Foot(0),
    Part::Foot(1),
];

impl Part {
    /// Its two ends, and how thick it is.
    fn span(self, j: &Joints) -> (Vec3, Vec3, f32) {
        let middle = j.pelvis.lerp(j.chest, 0.45);
        match self {
            Part::Head => (j.head, j.head, HEAD_RADIUS),
            Part::Neck => (j.neck, j.head, 0.04),
            Part::Chest => (middle, j.chest.lerp(j.neck, 0.15), 0.11),
            Part::Belly => (j.pelvis, middle, 0.1),
            Part::Shoulders => (j.shoulder[0], j.shoulder[1], 0.05),
            Part::Hips => (j.hip[0], j.hip[1], 0.085),
            Part::UpperArm(s) => (j.shoulder[s], j.elbow[s], 0.04),
            Part::Forearm(s) => (j.elbow[s], j.wrist[s], 0.033),
            Part::Thigh(s) => (j.hip[s], j.knee[s], 0.064),
            Part::Shin(s) => (j.knee[s], j.ankle[s], 0.047),
            Part::Foot(s) => (j.ankle[s], j.toe[s], 0.036),
        }
    }

    /// The far side a shade darker, so crossing limbs stay readable in flat colour.
    fn is_right(self) -> bool {
        matches!(self, Part::UpperArm(1) | Part::Forearm(1) | Part::Thigh(1) | Part::Shin(1) | Part::Foot(1))
    }
}

#[derive(Component)]
struct BodyPart(Part);

#[derive(Component)]
struct Seat;

#[derive(Resource)]
struct Skin {
    near: Handle<StandardMaterial>,
    far: Handle<StandardMaterial>,
    head: Handle<StandardMaterial>,
}

fn spawn_body(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let flat = |color: Color| StandardMaterial { base_color: color, unlit: true, ..default() };
    let matte = |color: Color| StandardMaterial { base_color: color, perceptual_roughness: 1.0, reflectance: 0.05, ..default() };
    let skin = Skin { near: materials.add(matte(TEXT)), far: materials.add(matte(mix(TEXT, MUTED, 0.35))), head: materials.add(matte(TEXT)) };
    commands.spawn((
        DirectionalLight { illuminance: 3500.0, shadows_enabled: false, ..default() },
        Transform::from_xyz(2.0, 6.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(AmbientLight { color: Color::WHITE, brightness: 900.0, ..default() });
    let rest = joints(&standing(0.0), 0.0);
    for part in PARTS {
        let (a, b, radius) = part.span(&rest);
        let mesh = if part == Part::Head {
            meshes.add(Sphere::new(radius).mesh().uv(32, 18))
        } else {
            meshes.add(Capsule3d::new(radius, a.distance(b).max(0.001)).mesh().rings(4).latitudes(12).longitudes(20))
        };
        let material = match part {
            Part::Head => skin.head.clone(),
            _ if part.is_right() => skin.far.clone(),
            _ => skin.near.clone(),
        };
        commands.spawn((BodyPart(part), Mesh3d(mesh), MeshMaterial3d(material), Transform::default(), Visibility::Hidden));
    }
    commands.spawn((
        Seat,
        Mesh3d(meshes.add(Cuboid::new(0.42, SEAT_HEIGHT, 0.42))),
        MeshMaterial3d(materials.add(flat(FAINT))),
        Transform::default(),
        Visibility::Hidden,
    ));
    commands.insert_resource(skin);
}

/// The room's z-up point in Bevy's world.
fn room_point(p: Vec3) -> Vec3 {
    world([p.x, p.y, p.z])
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn pose_body(
    time: Res<Time>,
    latest: Res<Latest>,
    mut figure: ResMut<Figure>,
    mut motion: ResMut<Motion>,
    skin: Res<Skin>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut parts: Query<(&BodyPart, &mut Transform, &mut Visibility), Without<Seat>>,
    mut seats: Query<(&mut Transform, &mut Visibility), With<Seat>>,
) {
    let snapshot = &latest.0;
    let shown = figure.presence > 0.3 && figure.at.is_some();
    for (_, _, mut visibility) in &mut parts {
        *visibility = if shown { Visibility::Inherited } else { Visibility::Hidden };
    }
    let Some(at) = figure.at.filter(|_| shown) else {
        for (_, mut visibility) in &mut seats {
            *visibility = Visibility::Hidden;
        }
        return;
    };
    let seconds = time.elapsed_secs();
    let dt = time.delta_secs();
    let under = snapshot.furniture.iter().filter(|item| item.kind == "bed").filter_map(|item| item.top_under(at)).fold(0.0, f32::max);
    let goal = match figure.gait {
        Gait::Walk => walking(seconds),
        Gait::Stand => standing(seconds),
        Gait::Sit => sitting(),
        Gait::Lie => lying(under),
    };
    if motion.gait != Some(figure.gait) {
        motion.gait = Some(figure.gait);
        motion.gait_seconds = 0.0;
    }
    motion.gait_seconds += dt;
    let rate = if motion.gait_seconds > SETTLE_SECONDS { SETTLED_RATE } else { CHANGING_RATE };
    let posture = motion.posture.get_or_insert(goal);
    posture.toward(&goal, ease(rate, dt));
    let breath = snapshot.breath_bpm.map_or(0.0, |bpm| (TAU * bpm as f32 / 60.0 * seconds).sin());
    let body = joints(posture, breath);
    let heading = Quat::from_rotation_z(figure.heading());
    let origin = Vec3::new(at[0], at[1], 0.0);
    let to_room = |p: Vec3| room_point(origin + heading * p);
    figure.head = to_room(body.head);
    for (part, mut transform, _) in &mut parts {
        let (a, b, _) = part.0.span(&body);
        let (a, b) = (to_room(a), to_room(b));
        let along = b - a;
        let rotation = if along.length() > 1e-4 { Quat::from_rotation_arc(Vec3::Y, along.normalize()) } else { Quat::IDENTITY };
        *transform = Transform::from_translation((a + b) / 2.0).with_rotation(rotation);
    }
    let moving = figure.gait == Gait::Walk;
    if let Some(head) = materials.get_mut(&skin.head) {
        head.base_color = if moving { Color::LinearRgba(glowing(mix(TEXT, RED, 0.6), 1.0)) } else { TEXT };
    }
    let sitting = figure.gait == Gait::Sit && motion.gait_seconds > 0.5;
    for (mut transform, mut visibility) in &mut seats {
        let back = origin + heading * Vec3::new(-0.08, 0.0, 0.0);
        *transform = Transform::from_translation(floor([back.x, back.y], SEAT_HEIGHT / 2.0)).with_rotation(Quat::from_rotation_y(figure.heading()));
        *visibility = if sitting { Visibility::Inherited } else { Visibility::Hidden };
    }
}
