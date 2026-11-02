//! The room: floor, grid, the boards, and each link as a line that thickens and warms with its
//! disturbance. While someone moves, the Fresnel zones of the most disturbed links light up on the
//! floor: the person is where they cross.

use std::f32::consts::FRAC_PI_2;

use bevy::prelude::*;

use crate::live_core::{RoomStatus, Snapshot};

use super::palette::{self, RED, EDGE, FAINT, GRID, TEXT};
use super::{ease, floor, world, Latest};

/// 2.4 GHz channel 6.
const WAVELENGTH_M: f32 = 0.123;
/// The zone drawn: the path through its edge is this many half-wavelengths longer than the direct one.
const FRESNEL_ZONE: usize = 2;
const GRID_STEP_M: f32 = 1.0;
const ACTIVITY_RATE: f32 = 6.0;
const ZONE_RATE: f32 = 2.5;
/// A link's zone lights when it is this share of the most disturbed link and above `ZONE_MIN_ACTIVITY`.
const ZONE_SHARE_OF_TOP: f32 = 0.75;
const ZONE_MIN_ACTIVITY: f32 = 0.12;
const ZONE_FILL_ALPHA: f32 = 0.07;

pub struct RoomPlugin;

impl Plugin for RoomPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Layout>()
            .add_systems(Update, (rebuild_on_new_layout, beams, fresnel_zones, outlines).chain());
    }
}

/// Where the boards and links stand; when it changes (a board joins, the boards are placed again)
/// the room's entities are made anew.
#[derive(Resource, Default, PartialEq)]
pub struct Layout {
    boards: Vec<([f32; 3], String)>,
    links: Vec<[usize; 2]>,
    size: [f32; 3],
}

#[derive(Component)]
struct RoomPart;

#[derive(Component)]
struct Beam {
    link: usize,
    shown: f32,
}

#[derive(Component)]
struct Zone {
    link: usize,
    shown: f32,
}

fn layout_of(snapshot: &Snapshot) -> Layout {
    Layout {
        boards: snapshot.boards.iter().map(|b| (b.at, b.name.clone())).collect(),
        links: snapshot.links.iter().map(|l| l.ends).collect(),
        size: snapshot.room_size,
    }
}

fn flat(color: Color) -> StandardMaterial {
    StandardMaterial { base_color: color, unlit: true, ..default() }
}

fn link_ends(snapshot: &Snapshot, link: usize) -> Option<([f32; 3], [f32; 3])> {
    let ends = snapshot.links.get(link)?.ends;
    Some((snapshot.boards.get(ends[0])?.at, snapshot.boards.get(ends[1])?.at))
}

fn rebuild_on_new_layout(
    mut commands: Commands,
    latest: Res<Latest>,
    mut layout: ResMut<Layout>,
    parts: Query<Entity, With<RoomPart>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let now = layout_of(&latest.0);
    if *layout == now && !parts.is_empty() {
        return;
    }
    for entity in &parts {
        commands.entity(entity).despawn();
    }
    let snapshot = &latest.0;
    let [width, depth, _] = snapshot.room_size;
    commands.spawn((
        RoomPart,
        Mesh3d(meshes.add(Plane3d::default().mesh().size(width, depth))),
        MeshMaterial3d(materials.add(flat(palette::FLOOR))),
        Transform::from_translation(floor([width / 2.0, depth / 2.0], 0.0)),
    ));
    let board_mesh = meshes.add(Cuboid::new(0.1, 0.1, 0.1));
    let board_material = materials.add(flat(TEXT));
    for board in &snapshot.boards {
        commands.spawn((RoomPart, Mesh3d(board_mesh.clone()), MeshMaterial3d(board_material.clone()), Transform::from_translation(world(board.at))));
    }
    let beam_mesh = meshes.add(Cylinder::new(1.0, 1.0));
    let zone_mesh = meshes.add(Circle::new(1.0).mesh().resolution(96));
    for link in 0..snapshot.links.len() {
        let Some((a, b)) = link_ends(snapshot, link) else { continue };
        commands.spawn((
            RoomPart,
            Beam { link, shown: 0.0 },
            Mesh3d(beam_mesh.clone()),
            MeshMaterial3d(materials.add(flat(EDGE))),
            segment_transform(world(a), world(b), 0.006),
        ));
        commands.spawn((
            RoomPart,
            Zone { link, shown: 0.0 },
            Mesh3d(zone_mesh.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: RED.with_alpha(0.0),
                unlit: true,
                alpha_mode: AlphaMode::Add,
                cull_mode: None,
                ..default()
            })),
            zone_transform([a[0], a[1]], [b[0], b[1]], 0.01),
        ));
    }
    *layout = now;
}

/// A unit cylinder stretched from `a` to `b` with this radius.
pub fn segment_transform(a: Vec3, b: Vec3, radius: f32) -> Transform {
    let along = b - a;
    let length = along.length().max(1e-4);
    Transform {
        translation: (a + b) / 2.0,
        rotation: Quat::from_rotation_arc(Vec3::Y, along / length),
        scale: Vec3::new(radius, length, radius),
    }
}

/// The Fresnel zone between two points on the floor: the ellipse whose foci are the ends and where
/// the path through it is `FRESNEL_ZONE` half-wavelengths longer than the direct one.
fn fresnel_axes(a: [f32; 2], b: [f32; 2]) -> Vec2 {
    let distance = (b[0] - a[0]).hypot(b[1] - a[1]);
    let semi_major = (distance + FRESNEL_ZONE as f32 * WAVELENGTH_M / 2.0) / 2.0;
    let semi_minor = (semi_major.powi(2) - (distance / 2.0).powi(2)).max(0.0).sqrt();
    Vec2::new(semi_major, semi_minor)
}

fn zone_transform(a: [f32; 2], b: [f32; 2], height: f32) -> Transform {
    let axes = fresnel_axes(a, b);
    let centre = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
    let angle = (b[1] - a[1]).atan2(b[0] - a[0]);
    Transform {
        translation: floor(centre, height),
        rotation: Quat::from_rotation_y(angle) * Quat::from_rotation_x(-FRAC_PI_2),
        scale: axes.extend(1.0),
    }
}

fn beams(
    time: Res<Time>,
    latest: Res<Latest>,
    mut beams: Query<(&mut Beam, &mut Transform, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let snapshot = &latest.0;
    let share = ease(ACTIVITY_RATE, time.delta_secs());
    let most_disturbed = snapshot.most_disturbed_link(ZONE_MIN_ACTIVITY);
    for (mut beam, mut transform, material) in &mut beams {
        let (Some(link), Some((a, b))) = (snapshot.links.get(beam.link), link_ends(snapshot, beam.link)) else {
            continue;
        };
        beam.shown += (link.activity - beam.shown) * share;
        let activity = beam.shown.clamp(0.0, 1.0);
        *transform = segment_transform(world(a), world(b), 0.004 + 0.010 * activity);
        if let Some(material) = materials.get_mut(&material.0) {
            let color = if most_disturbed == Some(beam.link) { RED } else { palette::mix(EDGE, TEXT, activity * 1.5) };
            material.base_color = Color::LinearRgba(palette::glowing(color, 1.0 + 2.0 * activity));
        }
    }
}

/// The links that place the person right now: the most disturbed ones, while someone moves.
fn lit_links(snapshot: &Snapshot) -> Vec<f32> {
    let moving = snapshot.status == RoomStatus::Moving;
    let top = snapshot.links.iter().map(|l| l.activity).fold(0.0, f32::max);
    snapshot
        .links
        .iter()
        .map(|link| {
            let lit = moving && link.activity >= ZONE_MIN_ACTIVITY && link.activity >= ZONE_SHARE_OF_TOP * top;
            if lit { (link.activity / top.max(1e-3)).clamp(0.4, 1.0) } else { 0.0 }
        })
        .collect()
}

fn fresnel_zones(
    time: Res<Time>,
    latest: Res<Latest>,
    mut zones: Query<(&mut Zone, &mut Transform, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let snapshot = &latest.0;
    let lit = lit_links(snapshot);
    let share = ease(ZONE_RATE, time.delta_secs());
    for (mut zone, mut transform, material) in &mut zones {
        let Some((a, b)) = link_ends(snapshot, zone.link) else { continue };
        let goal = lit.get(zone.link).copied().unwrap_or(0.0);
        zone.shown += (goal - zone.shown) * share;
        *transform = zone_transform([a[0], a[1]], [b[0], b[1]], 0.004 + 0.001 * zone.link as f32);
        if let Some(material) = materials.get_mut(&material.0) {
            material.base_color = RED.with_alpha(ZONE_FILL_ALPHA * zone.shown);
        }
    }
}

fn outlines(latest: Res<Latest>, zones: Query<&Zone>, mut gizmos: Gizmos) {
    let snapshot = &latest.0;
    let [width, depth, height] = snapshot.room_size;
    let mut x = GRID_STEP_M;
    while x < width - 1e-3 {
        gizmos.line(floor([x, 0.0], 0.001), floor([x, depth], 0.001), GRID);
        x += GRID_STEP_M;
    }
    let mut y = GRID_STEP_M;
    while y < depth - 1e-3 {
        gizmos.line(floor([0.0, y], 0.001), floor([width, y], 0.001), GRID);
        y += GRID_STEP_M;
    }
    let corners = [[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]];
    for (i, &corner) in corners.iter().enumerate() {
        let next = corners[(i + 1) % 4];
        gizmos.line(floor(corner, 0.002), floor(next, 0.002), EDGE);
        gizmos.line(floor(corner, height), floor(next, height), GRID);
        gizmos.line(floor(corner, 0.0), floor(corner, height), GRID);
    }
    for item in &snapshot.furniture {
        let size = Vec3::new(item.size[0], item.size[2], item.size[1]);
        gizmos.cuboid(Transform::from_translation(world(item.centre)).with_scale(size), FAINT);
    }
    for board in &snapshot.boards {
        gizmos.line(floor([board.at[0], board.at[1]], 0.0), world(board.at), FAINT);
    }
    for zone in &zones {
        if zone.shown < 0.02 {
            continue;
        }
        let Some((a, b)) = link_ends(snapshot, zone.link) else { continue };
        let transform = zone_transform([a[0], a[1]], [b[0], b[1]], 0.006);
        let color = RED.with_alpha(zone.shown);
        gizmos.ellipse(Isometry3d::new(transform.translation, transform.rotation), fresnel_axes([a[0], a[1]], [b[0], b[1]]), color);
    }
    for phone in &snapshot.phones {
        for &(board, metres) in &phone.boards {
            let Some(board) = snapshot.boards.get(board) else { continue };
            let centre = floor([board.at[0], board.at[1]], 0.003);
            gizmos.circle(Isometry3d::new(centre, Quat::from_rotation_x(FRAC_PI_2)), metres as f32, palette::MUTED);
        }
    }
}
