//! A camera that orbits the room: drag to turn, scroll to zoom, 1/2/3 for set views, space for a slow
//! turn of its own (off until asked for), which then also starts after a while without input.

use std::f32::consts::FRAC_PI_2;

use bevy::core_pipeline::bloom::Bloom;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;
use bevy::window::{MonitorSelection, PrimaryWindow, WindowMode};

use super::person::Figure;
use super::{ease, palette, Latest};

const DRAG_RADIANS_PER_PIXEL: f32 = 0.006;
const ZOOM_PER_LINE: f32 = 0.08;
const PITCH_RANGE: (f32, f32) = (0.06, FRAC_PI_2 - 0.02);
const IDLE_BEFORE_AUTO_SECONDS: f32 = 8.0;
const AUTO_RADIANS_PER_SECOND: f32 = 0.09;
const GLIDE_RATE: f32 = 3.5;
const FIELD_OF_VIEW: f32 = 0.75;
/// The HUD's side columns, which the room should fit between.
const SIDE_COLUMNS_PX: f32 = 900.0;

pub struct OrbitPlugin;

impl Plugin for OrbitPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(Update, (keyboard, mouse, glide).chain());
    }
}

/// `zoom` 1 fits the room between the HUD's columns.
#[derive(Clone, Copy)]
struct View {
    yaw: f32,
    pitch: f32,
    zoom: f32,
}

#[derive(Component)]
pub struct Orbit {
    target: Vec3,
    room_centre: Vec3,
    /// View 4: the camera keeps the person in the middle.
    following: bool,
    shown: View,
    goal: View,
    auto: bool,
    idle_seconds: f32,
    room_span: f32,
}

impl Orbit {
    fn preset(&self, number: usize) -> View {
        match number {
            2 => View { yaw: 0.0, pitch: 1.5, zoom: 0.95 },
            3 => View { yaw: self.shown.yaw + 0.9, pitch: 0.2, zoom: 0.7 },
            4 => View { yaw: self.shown.yaw, pitch: 0.3, zoom: 0.6 },
            _ => View { yaw: 0.3, pitch: 0.62, zoom: 1.0 },
        }
    }
}

#[derive(Resource)]
pub struct FirstView(pub usize);

fn spawn_camera(mut commands: Commands, latest: Res<Latest>, first: Res<FirstView>) {
    let [width, depth, _] = latest.0.room_size;
    let target = super::world([width / 2.0, depth / 2.0, 0.6]);
    let mut orbit = Orbit {
        target,
        room_centre: target,
        following: first.0 == 4,
        shown: View { yaw: 0.0, pitch: 0.0, zoom: 1.0 },
        goal: View { yaw: 0.0, pitch: 0.0, zoom: 1.0 },
        auto: false,
        idle_seconds: 0.0,
        room_span: width.max(depth),
    };
    orbit.goal = orbit.preset(first.0);
    orbit.shown = View { yaw: orbit.goal.yaw - 0.6, pitch: orbit.goal.pitch + 0.3, zoom: 1.6 };
    commands.spawn((
        Camera3d::default(),
        Camera { hdr: true, clear_color: ClearColorConfig::Custom(palette::BACKGROUND), ..default() },
        Tonemapping::TonyMcMapface,
        Bloom { intensity: 0.12, ..Bloom::NATURAL },
        Projection::Perspective(PerspectiveProjection { fov: FIELD_OF_VIEW, ..default() }),
        place(&orbit, 10.0),
        orbit,
    ));
}

/// How far the camera stands for the room to fill the space between the HUD's columns.
fn fitting_distance(room_span: f32, window: &Window) -> f32 {
    let (width, height) = (window.width().max(1.0), window.height().max(1.0));
    let free = (width - SIDE_COLUMNS_PX).max(0.6 * width);
    let half_vertical = (FIELD_OF_VIEW / 2.0).tan();
    let half_horizontal = half_vertical * width / height;
    let across = 1.0 * room_span * width / (free * half_horizontal);
    let up = 0.8 * room_span / half_vertical;
    across.max(up)
}

fn place(orbit: &Orbit, fitting: f32) -> Transform {
    let View { yaw, pitch, zoom } = orbit.shown;
    let distance = zoom * fitting;
    let offset = Vec3::new(pitch.cos() * yaw.sin(), pitch.sin(), pitch.cos() * yaw.cos()) * distance;
    Transform::from_translation(orbit.target + offset).looking_at(orbit.target, Vec3::Y)
}

fn keyboard(
    keys: Res<ButtonInput<KeyCode>>,
    mut orbits: Query<&mut Orbit>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut exit: EventWriter<AppExit>,
) {
    let Ok(mut orbit) = orbits.single_mut() else { return };
    if keys.just_pressed(KeyCode::Space) {
        orbit.auto = !orbit.auto;
        orbit.idle_seconds = if orbit.auto { IDLE_BEFORE_AUTO_SECONDS } else { 0.0 };
    }
    let presets = [(KeyCode::Digit1, 1), (KeyCode::Digit2, 2), (KeyCode::Digit3, 3), (KeyCode::Digit4, 4)];
    for (key, number) in presets {
        if keys.just_pressed(key) {
            orbit.goal = orbit.preset(number);
            orbit.following = number == 4;
            orbit.idle_seconds = 0.0;
        }
    }
    let Ok(mut window) = windows.single_mut() else { return };
    let fullscreen = matches!(window.mode, WindowMode::BorderlessFullscreen(_));
    if keys.just_pressed(KeyCode::KeyF) {
        window.mode = if fullscreen {
            WindowMode::Windowed
        } else {
            WindowMode::BorderlessFullscreen(MonitorSelection::Current)
        };
    }
    if keys.just_pressed(KeyCode::Escape) && fullscreen {
        window.mode = WindowMode::Windowed;
    }
    if keys.just_pressed(KeyCode::KeyQ) {
        exit.write(AppExit::Success);
    }
}

fn mouse(
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    time: Res<Time>,
    mut orbits: Query<&mut Orbit>,
) {
    let Ok(mut orbit) = orbits.single_mut() else { return };
    let dragging = buttons.any_pressed([MouseButton::Left, MouseButton::Right, MouseButton::Middle]);
    let lines = match scroll.unit {
        MouseScrollUnit::Line => scroll.delta.y,
        MouseScrollUnit::Pixel => scroll.delta.y / 40.0,
    };
    if dragging && motion.delta != Vec2::ZERO {
        let (yaw, pitch) = (
            orbit.goal.yaw - motion.delta.x * DRAG_RADIANS_PER_PIXEL,
            (orbit.goal.pitch + motion.delta.y * DRAG_RADIANS_PER_PIXEL).clamp(PITCH_RANGE.0, PITCH_RANGE.1),
        );
        orbit.goal.yaw = yaw;
        orbit.goal.pitch = pitch;
        orbit.idle_seconds = 0.0;
    }
    if lines != 0.0 {
        orbit.goal.zoom = (orbit.goal.zoom * (1.0 - lines * ZOOM_PER_LINE)).clamp(0.15, 4.0);
        orbit.idle_seconds = 0.0;
    }
    orbit.idle_seconds += time.delta_secs();
    if orbit.auto && orbit.idle_seconds >= IDLE_BEFORE_AUTO_SECONDS {
        orbit.goal.yaw += AUTO_RADIANS_PER_SECOND * time.delta_secs();
    }
}

fn glide(
    time: Res<Time>,
    figure: Res<Figure>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cameras: Query<(&mut Orbit, &mut Transform)>,
) {
    let Ok((mut orbit, mut transform)) = cameras.single_mut() else { return };
    let Ok(window) = windows.single() else { return };
    let share = ease(GLIDE_RATE, time.delta_secs());
    let aim = match figure.at.filter(|_| orbit.following && figure.presence > 0.3) {
        Some(at) => super::floor(at, 0.9),
        None => orbit.room_centre,
    };
    orbit.target = orbit.target.lerp(aim, share);
    let View { yaw, pitch, zoom } = orbit.goal;
    let shown = &mut orbit.shown;
    shown.yaw += (yaw - shown.yaw) * share;
    shown.pitch += (pitch - shown.pitch) * share;
    shown.zoom += (zoom - shown.zoom) * share;
    *transform = place(&orbit, fitting_distance(orbit.room_span, window));
}
