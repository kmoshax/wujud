use std::f32::consts::TAU;

use eframe::egui::{self, Color32, Mesh, Pos2, Rect, Sense, Shape, Stroke, Vec2};
use wujud_model::motion_tracker::Track;
use wujud_model::room::{Pose, RoomState, RoomState3};
use wujud_model::scene::Scene;

use crate::camera::{add, lerp, Camera, Point3, Projection};
use crate::phone_range::PhoneDistance;
use crate::room_view;
use crate::skeleton::{self, Gait};
use crate::sensor_feed::SimulatedTruth;
use crate::theme;

const SKY_TOP: Color32 = Color32::from_rgb(16, 16, 16);
const SKY_BOTTOM: Color32 = theme::BACKGROUND;
const FLOOR: Color32 = Color32::from_rgb(18, 18, 18);
const FLOOR_LINE: Color32 = Color32::from_rgba_premultiplied(30, 30, 29, 110);
const WALL: Color32 = Color32::from_rgba_premultiplied(18, 18, 18, 150);
const EDGE: Color32 = Color32::from_rgb(72, 72, 70);
const FURNITURE: Color32 = Color32::from_rgb(62, 61, 59);
const BED: Color32 = Color32::from_rgb(74, 73, 70);
const DESK: Color32 = Color32::from_rgb(68, 67, 64);
const WARDROBE: Color32 = Color32::from_rgb(50, 49, 47);
const BUFFET: Color32 = Color32::from_rgb(64, 63, 60);
const MIRROR: Color32 = theme::MUTED;
const HUB: Color32 = Color32::from_rgb(210, 206, 198);
const DEVICE: Color32 = Color32::from_rgb(140, 138, 133);
const LABEL: Color32 = Color32::from_rgb(170, 167, 160);
const HEAT: Color32 = theme::RED;
const SHADOW: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 90);
pub(crate) const BOARD: Color32 = HUB;
pub(crate) const LINK_IDLE: Color32 = Color32::from_rgb(52, 52, 50);
pub(crate) const LINK_ACTIVE: Color32 = theme::TEXT;
pub(crate) const TRUE_PERSON: Color32 = Color32::from_rgba_premultiplied(70, 70, 70, 70);

const FLOOR_TILE_M: f32 = 0.5;
pub(crate) const HEAD_RADIUS: f32 = 0.11;
const SEAT: Color32 = Color32::from_rgb(40, 40, 39);
const SKELETON_LINE_M: f32 = 0.035;
/// Share of the screen distance the drawn person closes each frame towards the tracked position.
const FOLLOW: f32 = 0.12;
/// The body turns to face where it walks, once a frame's step is longer than this.
const TURN_ABOVE_M: f32 = 0.002;
const TURN_RATE: f32 = 0.1;
const HEAT_SHOWN_ABOVE: f32 = 0.04;
/// Beyond the boards' line by this much the body still disturbs the link clearly (first Fresnel zone, roughly).
const NEAR_LINE_M: f32 = 0.45;
const FAR_LINE_M: f32 = 1.4;
const NEAR: Color32 = theme::RED;
const FAR: Color32 = theme::MUTED;
const PHONE: Color32 = theme::TEXT;
const PHONE_HEIGHT_M: f64 = 0.9;
const BREATH: Color32 = theme::TEXT;
const NEAREST_BOARD: Color32 = LABEL;
/// room_view::BREATHING_AFTER_STILL_SECONDS, as the ring that fills while the breathing is read.
const BREATHING_AFTER_STILL_SECONDS: f32 = 30.0;

pub(crate) struct Drawable {
    pub depth: f32,
    pub shape: Shape,
}

/// Where the tracker believes the person is, as a floor heat map.
pub struct Belief<'a> {
    pub xs: &'a [f64],
    pub ys: &'a [f64],
    pub values: &'a [f64],
}

pub struct SceneInputs<'a> {
    pub scene: &'a Scene,
    pub node_names: &'a [String],
    pub state: Option<&'a RoomState>,
    pub track: Option<&'a Track>,
    pub belief: Option<Belief<'a>>,
    pub truth: Option<&'a SimulatedTruth>,
    pub seconds: f32,
    /// Two boards: draw how near the person is to their line instead of a position.
    pub one_line: bool,
    pub phones: &'a [PhoneDistance],
    pub still_seconds: f32,
}

#[derive(Default)]
pub struct Scene3d {
    camera: Option<Camera>,
    drawn_at: Option<[f32; 2]>,
    heading: Option<f32>,
}

impl Scene3d {
    pub fn show(&mut self, ui: &mut egui::Ui, inputs: &SceneInputs) {
        let scene = inputs.scene;
        let (width, depth, height) = (scene.width as f32, scene.depth as f32, scene.height as f32);
        let (viewport, response) = ui.allocate_exact_size(ui.available_size(), Sense::drag());
        let camera = self
            .camera
            .get_or_insert_with(|| Camera::looking_at_room(width, depth, height));
        camera.follow_mouse(&response, ui.input(|i| i.smooth_scroll_delta.y));
        let view = camera.projection(viewport);
        let painter = ui.painter_at(viewport);
        painter.add(vertical_gradient(viewport, SKY_TOP, SKY_BOTTOM));

        let mut drawables = Vec::new();
        room_shell(&view, width, depth, height, &mut drawables);
        floor_tiles(&view, width, depth, &mut drawables);
        if let Some(belief) = &inputs.belief {
            floor_heat(&view, belief, &mut drawables);
        }
        if inputs.one_line {
            closeness_zone(&view, inputs, &mut drawables);
        }
        phone_rings(&view, inputs, &mut drawables);
        for item in &scene.furniture {
            let number = |key: &str| item.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let size = [number("w"), number("d"), number("h")];
            let color = match item.get("kind").and_then(|k| k.as_str()) {
                Some("bed") => BED,
                Some("desk") => DESK,
                Some("wardrobe") => WARDROBE,
                Some("buffet") => BUFFET,
                Some("mirror") => MIRROR,
                Some("laptop") => DEVICE,
                _ => FURNITURE,
            };
            solid_box(
                &view,
                [number("x"), number("y"), number("z") + size[2] / 2.0],
                size,
                color,
                &mut drawables,
            );
        }
        for (link, [a, b]) in scene.links.iter().enumerate() {
            let activity = link_activity(inputs, link);
            radio_link(
                &view,
                point(scene.nodes[*a]),
                point(scene.nodes[*b]),
                activity,
                inputs.seconds,
                &mut drawables,
            );
        }
        for (index, node) in scene.nodes.iter().enumerate() {
            let is_hub = scene.links.iter().all(|link| link.contains(&index));
            node_model(&view, point(*node), is_hub, &mut drawables);
        }
        let figure = self.figure(inputs);
        if let Some(figure) = &figure {
            person(&view, figure, inputs.seconds, &mut drawables);
            breathing_ring(&view, figure, inputs, &mut drawables);
            if !inputs.one_line && inputs.track.is_some_and(|t| t.xy.is_some()) {
                nearest_board_line(&view, figure, scene, &mut drawables);
            }
        }
        if let Some(truth) = inputs.truth {
            true_person(&view, truth, &mut drawables);
        }

        drawables.sort_by(|a, b| b.depth.total_cmp(&a.depth));
        painter.extend(drawables.into_iter().map(|d| d.shape));
        node_labels(&painter, &view, &scene.nodes, inputs.node_names);
        if let Some(figure) = &figure {
            figure_labels(&painter, &view, figure, inputs);
        }
        if inputs.one_line {
            closeness_label(&painter, &view, inputs);
        }
        phone_labels(&painter, &view, inputs);
        painter.text(
            viewport.left_bottom() + Vec2::new(10.0, -10.0),
            egui::Align2::LEFT_BOTTOM,
            if inputs.one_line {
                "drag to rotate · scroll to zoom · lit floor = how near the person is to the boards' line"
            } else {
                "drag to rotate · scroll to zoom · red floor = where the person may be"
            },
            egui::FontId::proportional(11.0),
            theme::MUTED,
        );
    }

    /// The person to draw: at the tracked position (eased, so it walks instead of jumping), shaped by
    /// the activity model, and faded when the position is uncertain.
    fn figure(&mut self, inputs: &SceneInputs) -> Option<Figure> {
        let decision = inputs.state.and_then(|s| s.room.decision.as_ref());
        let moving_now = inputs.track.is_some_and(|t| t.moving);
        let someone_here = moving_now || decision.is_some_and(|d| d.state3 != RoomState3::Empty);
        if !someone_here {
            return None;
        }
        let scene = inputs.scene;
        let tracked = inputs
            .track
            .and_then(|t| t.xy)
            .map(|[x, y]| [x as f32, y as f32]);
        let target = tracked.unwrap_or([scene.width as f32 / 2.0, scene.depth as f32 / 2.0]);
        let drawn = self.drawn_at.get_or_insert(target);
        let step = [(target[0] - drawn[0]) * FOLLOW, (target[1] - drawn[1]) * FOLLOW];
        drawn[0] += step[0];
        drawn[1] += step[1];
        let heading = self.heading.get_or_insert(-std::f32::consts::FRAC_PI_2);
        if step[0].hypot(step[1]) > TURN_ABOVE_M {
            let towards = step[1].atan2(step[0]);
            let turn = (towards - *heading + std::f32::consts::PI).rem_euclid(TAU) - std::f32::consts::PI;
            *heading += turn * TURN_RATE;
        }
        let heading = *heading;
        let confidence = inputs.track.map_or(0.0, |t| t.confidence) as f32;
        let pose = match decision.map(|d| d.state) {
            Some(pose @ (Pose::Moving | Pose::Lie | Pose::Sit)) => pose,
            _ => Pose::Stand,
        };
        let under = bed_top(scene, *drawn);
        Some(Figure {
            at: *drawn,
            heading,
            under,
            pose,
            breath_bpm: decision.and_then(|d| d.breath_bpm),
            opacity: if tracked.is_some() {
                0.45 + 0.55 * confidence
            } else {
                0.35
            },
        })
    }
}

/// Name pills above the boards, pushed up when two would overlap.
fn node_labels(painter: &egui::Painter, view: &Projection, nodes: &[[f64; 3]], names: &[String]) {
    let font = egui::FontId::proportional(12.0);
    let mut labels: Vec<(Pos2, &String)> = nodes
        .iter()
        .zip(names)
        .filter_map(|(node, name)| Some((view.project(add(point(*node), [0.0, 0.0, 0.22]))?.at, name)))
        .collect();
    labels.sort_by(|a, b| b.0.y.total_cmp(&a.0.y));
    let mut placed: Vec<Rect> = Vec::new();
    for (anchor, name) in labels {
        let galley = painter.layout_no_wrap(name.clone(), font.clone(), LABEL);
        let size = galley.size() + Vec2::new(14.0, 6.0);
        let mut rect = Rect::from_center_size(anchor - Vec2::new(0.0, size.y / 2.0), size);
        while let Some(hit) = placed.iter().find(|p| p.expand(2.0).intersects(rect)) {
            rect = rect.translate(Vec2::new(0.0, hit.top() - rect.bottom() - 3.0));
        }
        if rect.center().y < anchor.y - size.y {
            painter.line_segment(
                [anchor, Pos2::new(rect.center().x, rect.bottom())],
                Stroke::new(1.0_f32, LABEL.gamma_multiply(0.4)),
            );
        }
        painter.rect_filled(rect, egui::CornerRadius::same(1), Color32::from_rgba_unmultiplied(9, 9, 9, 210));
        painter.rect_stroke(
            rect,
            egui::CornerRadius::same(1),
            Stroke::new(1.0_f32, theme::CARD_EDGE),
            egui::StrokeKind::Inside,
        );
        painter.galley(rect.center() - galley.size() / 2.0, galley, LABEL);
        placed.push(rect);
    }
}

fn link_activity(inputs: &SceneInputs, link: usize) -> f32 {
    let fast = inputs
        .track
        .and_then(|t| t.link_excess.get(link))
        .map(|&e| (e / 2.0).clamp(0.0, 1.0) as f32);
    let slow = inputs
        .state
        .and_then(|s| s.link_activity.get(link))
        .map(|&a| a as f32);
    fast.or(slow).unwrap_or(0.0)
}

struct Figure {
    at: [f32; 2],
    /// Where the body faces, radians from +x: the way it last walked.
    heading: f32,
    /// Top of the bed under the person, 0 on the floor.
    under: f32,
    pose: Pose,
    breath_bpm: Option<f64>,
    opacity: f32,
}

pub(crate) fn point(p: [f64; 3]) -> Point3 {
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

fn vertical_gradient(rect: Rect, top: Color32, bottom: Color32) -> Shape {
    let mut mesh = Mesh::default();
    mesh.colored_vertex(rect.left_top(), top);
    mesh.colored_vertex(rect.right_top(), top);
    mesh.colored_vertex(rect.right_bottom(), bottom);
    mesh.colored_vertex(rect.left_bottom(), bottom);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);
    Shape::mesh(mesh)
}

pub(crate) fn polygon(
    view: &Projection,
    corners: &[Point3],
    fill: Color32,
    stroke: Stroke,
    out: &mut Vec<Drawable>,
) {
    let projected: Option<Vec<_>> = corners.iter().map(|&c| view.project(c)).collect();
    let Some(projected) = projected else { return };
    let depth = projected.iter().map(|p| p.depth).sum::<f32>() / projected.len() as f32;
    let points: Vec<Pos2> = projected.iter().map(|p| p.at).collect();
    out.push(Drawable {
        depth,
        shape: Shape::convex_polygon(points, fill, stroke),
    });
}

/// Something lying flat on the floor: drawn right after the floor, under everything standing on it.
fn on_floor(
    view: &Projection,
    corners: &[Point3],
    fill: Color32,
    layer: f32,
    out: &mut Vec<Drawable>,
) {
    polygon(view, corners, fill, Stroke::NONE, out);
    if let Some(last) = out.last_mut() {
        last.depth = f32::MAX / 2.0 - layer;
    }
}

pub(crate) fn segment(
    view: &Projection,
    a: Point3,
    b: Point3,
    stroke: Stroke,
    out: &mut Vec<Drawable>,
) {
    if let (Some(pa), Some(pb)) = (view.project(a), view.project(b)) {
        out.push(Drawable {
            depth: (pa.depth + pb.depth) / 2.0,
            shape: Shape::line_segment([pa.at, pb.at], stroke),
        });
    }
}

pub(crate) fn sphere(
    view: &Projection,
    centre: Point3,
    radius: f32,
    color: Color32,
    out: &mut Vec<Drawable>,
) {
    if let Some(p) = view.project(centre) {
        let pixels = view.world_to_pixels(radius, p.depth);
        let rim = shade(color, 0.6);
        out.push(Drawable {
            depth: p.depth - radius,
            shape: Shape::Vec(vec![
                Shape::circle_filled(p.at, pixels, rim),
                Shape::circle_filled(p.at + Vec2::new(-0.2, -0.25) * pixels, pixels * 0.72, color),
            ]),
        });
    }
}

fn shade(color: Color32, light: f32) -> Color32 {
    let [r, g, b, a] = color.to_array();
    let lit = |c: u8| (f32::from(c) * light).min(255.0) as u8;
    Color32::from_rgba_premultiplied(lit(r), lit(g), lit(b), a)
}

/// Floor, the walls on the far side of the room (the near ones are cut away), and the room's edges.
pub(crate) fn room_shell(
    view: &Projection,
    width: f32,
    depth: f32,
    height: f32,
    out: &mut Vec<Drawable>,
) {
    let floor = [
        [0.0, 0.0, 0.0],
        [width, 0.0, 0.0],
        [width, depth, 0.0],
        [0.0, depth, 0.0],
    ];
    polygon(view, &floor, FLOOR, Stroke::new(1.5_f32, EDGE), out);
    if let Some(floor) = out.last_mut() {
        floor.depth = f32::MAX;
    }
    let walls: [([Point3; 4], Point3); 4] = [
        (
            [
                [0.0, 0.0, 0.0],
                [width, 0.0, 0.0],
                [width, 0.0, height],
                [0.0, 0.0, height],
            ],
            [0.0, 1.0, 0.0],
        ),
        (
            [
                [width, 0.0, 0.0],
                [width, depth, 0.0],
                [width, depth, height],
                [width, 0.0, height],
            ],
            [-1.0, 0.0, 0.0],
        ),
        (
            [
                [width, depth, 0.0],
                [0.0, depth, 0.0],
                [0.0, depth, height],
                [width, depth, height],
            ],
            [0.0, -1.0, 0.0],
        ),
        (
            [
                [0.0, depth, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, height],
                [0.0, depth, height],
            ],
            [1.0, 0.0, 0.0],
        ),
    ];
    for (corners, inward) in walls {
        if view.faces_camera(corners[0], inward) {
            polygon(view, &corners, WALL, Stroke::new(1.0_f32, EDGE), out);
        }
    }
}

fn floor_tiles(view: &Projection, width: f32, depth: f32, out: &mut Vec<Drawable>) {
    let lines = |extent: f32| {
        (1..)
            .map(|i| i as f32 * FLOOR_TILE_M)
            .take_while(move |&v| v < extent)
    };
    for x in lines(width) {
        segment(
            view,
            [x, 0.0, 0.002],
            [x, depth, 0.002],
            Stroke::new(1.0_f32, FLOOR_LINE),
            out,
        );
        if let Some(last) = out.last_mut() {
            last.depth = f32::MAX / 2.0;
        }
    }
    for y in lines(depth) {
        segment(
            view,
            [0.0, y, 0.002],
            [width, y, 0.002],
            Stroke::new(1.0_f32, FLOOR_LINE),
            out,
        );
        if let Some(last) = out.last_mut() {
            last.depth = f32::MAX / 2.0;
        }
    }
}

/// Only what stands out above an even spread is lit, so "no idea where" leaves the floor dark.
fn floor_heat(view: &Projection, belief: &Belief, out: &mut Vec<Drawable>) {
    let even = 1.0 / belief.values.len().max(1) as f64;
    let peak = belief.values.iter().copied().fold(0.0, f64::max);
    if peak <= even {
        return;
    }
    let concentration = (1.0 - even / peak) as f32;
    let half = |axis: &[f64]| {
        axis.get(1)
            .zip(axis.first())
            .map_or(0.05, |(b, a)| (b - a) as f32 / 2.0)
    };
    let (hx, hy) = (half(belief.xs), half(belief.ys));
    for (cell, &value) in belief.values.iter().enumerate() {
        let strength = ((value - even) / (peak - even)).max(0.0) as f32 * concentration;
        if strength < HEAT_SHOWN_ABOVE {
            continue;
        }
        let x = belief.xs[cell % belief.xs.len()] as f32;
        let y = belief.ys[cell / belief.xs.len()] as f32;
        let corners = [
            [x - hx, y - hy, 0.004],
            [x + hx, y - hy, 0.004],
            [x + hx, y + hy, 0.004],
            [x - hx, y + hy, 0.004],
        ];
        on_floor(
            view,
            &corners,
            HEAT.gamma_multiply(0.75 * strength.sqrt()),
            1.0,
            out,
        );
    }
}

pub(crate) fn solid_box(
    view: &Projection,
    centre: Point3,
    size: [f32; 3],
    color: Color32,
    out: &mut Vec<Drawable>,
) {
    let [hx, hy, hz] = size.map(|s| s / 2.0);
    let corner = |sx: f32, sy: f32, sz: f32| add(centre, [sx * hx, sy * hy, sz * hz]);
    let faces: [([Point3; 4], Point3); 6] = [
        (
            [
                corner(-1.0, -1.0, 1.0),
                corner(1.0, -1.0, 1.0),
                corner(1.0, 1.0, 1.0),
                corner(-1.0, 1.0, 1.0),
            ],
            [0.0, 0.0, 1.0],
        ),
        (
            [
                corner(-1.0, -1.0, -1.0),
                corner(-1.0, 1.0, -1.0),
                corner(1.0, 1.0, -1.0),
                corner(1.0, -1.0, -1.0),
            ],
            [0.0, 0.0, -1.0],
        ),
        (
            [
                corner(1.0, -1.0, -1.0),
                corner(1.0, 1.0, -1.0),
                corner(1.0, 1.0, 1.0),
                corner(1.0, -1.0, 1.0),
            ],
            [1.0, 0.0, 0.0],
        ),
        (
            [
                corner(-1.0, 1.0, -1.0),
                corner(-1.0, -1.0, -1.0),
                corner(-1.0, -1.0, 1.0),
                corner(-1.0, 1.0, 1.0),
            ],
            [-1.0, 0.0, 0.0],
        ),
        (
            [
                corner(1.0, 1.0, -1.0),
                corner(-1.0, 1.0, -1.0),
                corner(-1.0, 1.0, 1.0),
                corner(1.0, 1.0, 1.0),
            ],
            [0.0, 1.0, 0.0],
        ),
        (
            [
                corner(-1.0, -1.0, -1.0),
                corner(1.0, -1.0, -1.0),
                corner(1.0, -1.0, 1.0),
                corner(-1.0, -1.0, 1.0),
            ],
            [0.0, -1.0, 0.0],
        ),
    ];
    for (corners, normal) in faces {
        if view.faces_camera(corners[0], normal) {
            let fill = shade(color, view.lighting(normal));
            polygon(
                view,
                &corners,
                fill,
                Stroke::new(0.5_f32, shade(color, 0.5)),
                out,
            );
        }
    }
}

fn floor_disc(
    view: &Projection,
    centre: [f32; 2],
    radius: f32,
    color: Color32,
    out: &mut Vec<Drawable>,
) {
    let corners: Vec<Point3> = (0..20)
        .map(|i| {
            let angle = TAU * i as f32 / 20.0;
            [
                centre[0] + radius * angle.cos(),
                centre[1] + radius * angle.sin(),
                0.006,
            ]
        })
        .collect();
    on_floor(view, &corners, color, 2.0, out);
}

/// The shared board is the light box with a red light; every other end (laptop, phone) is a light slab.
fn node_model(view: &Projection, at: Point3, is_hub: bool, out: &mut Vec<Drawable>) {
    floor_disc(view, [at[0], at[1]], 0.12, SHADOW, out);
    if is_hub {
        solid_box(view, at, [0.12, 0.08, 0.05], HUB, out);
        sphere(
            view,
            add(at, [0.0, 0.0, 0.05]),
            0.025,
            theme::RED,
            out,
        );
    } else {
        solid_box(view, at, [0.26, 0.18, 0.025], DEVICE, out);
    }
}

/// The radio path glows and thickens with motion; pulses travel along it, faster when something moves.
pub(crate) fn radio_link(
    view: &Projection,
    a: Point3,
    b: Point3,
    activity: f32,
    seconds: f32,
    out: &mut Vec<Drawable>,
) {
    let activity = activity.clamp(0.0, 1.0);
    let color = LINK_IDLE.lerp_to_gamma(LINK_ACTIVE, activity);
    segment(
        view,
        a,
        b,
        Stroke::new(3.0 + 10.0 * activity, color.gamma_multiply(0.18)),
        out,
    );
    segment(view, a, b, Stroke::new(1.2 + 2.5 * activity, color), out);
    let speed = 0.25 + 1.5 * activity;
    for pulse in 0..3 {
        let phase = (seconds * speed + pulse as f32 / 3.0).fract();
        sphere(view, lerp(a, b, phase), 0.025 + 0.03 * activity, color, out);
    }
}

/// The skeleton as thin flat lines: off-white while still, red while moving.
fn person(view: &Projection, figure: &Figure, seconds: f32, out: &mut Vec<Drawable>) {
    let breath = figure
        .breath_bpm
        .map_or(0.0, |bpm| (TAU * bpm as f32 / 60.0 * seconds).sin());
    let gait = match figure.pose {
        Pose::Moving => Gait::Walk,
        Pose::Lie => Gait::Lie,
        Pose::Sit => Gait::Sit,
        _ => Gait::Stand,
    };
    let body = skeleton::pose(gait, figure.at, figure.heading, seconds, breath, figure.under);
    floor_disc(view, figure.at, 0.32, SHADOW, out);
    if gait == Gait::Sit {
        let [x, y] = figure.at;
        let back = [x - 0.06 * figure.heading.cos(), y - 0.06 * figure.heading.sin()];
        solid_box(view, [back[0], back[1], 0.22], [0.38, 0.38, 0.44], SEAT, out);
    }
    let color = if gait == Gait::Walk { theme::MOVING } else { theme::STILL };
    let line = color.gamma_multiply(figure.opacity);
    for (from, to, _) in body.bones() {
        bone(view, from, to, SKELETON_LINE_M, line, out);
    }
    sphere(view, body.head, skeleton::HEAD_RADIUS, line, out);
}

/// A limb as a line as thick as `thickness` metres at its distance from the camera.
fn bone(view: &Projection, from: Point3, to: Point3, thickness: f32, color: Color32, out: &mut Vec<Drawable>) {
    let (Some(a), Some(b)) = (view.project(from), view.project(to)) else { return };
    let depth = (a.depth + b.depth) / 2.0;
    let width = view.world_to_pixels(thickness, depth).max(1.0);
    out.push(Drawable {
        depth,
        shape: Shape::Vec(vec![
            Shape::line_segment([a.at, b.at], Stroke::new(width, color)),
            Shape::circle_filled(a.at, width / 2.0, color),
            Shape::circle_filled(b.at, width / 2.0, color),
        ]),
    });
}

/// In simulation the real position is known: a grey ghost shows where the person actually is.
fn true_person(view: &Projection, truth: &SimulatedTruth, out: &mut Vec<Drawable>) {
    let Some([x, y]) = truth.xy else { return };
    let (x, y) = (x as f32, y as f32);
    let (centre, size, head) = match truth.activity.as_str() {
        "lie" => ([x, y, 0.62], [1.6, 0.42, 0.22], [x - 0.95, y, 0.66]),
        "sit" => ([x, y, 0.85], [0.42, 0.3, 0.9], [x, y, 1.42]),
        _ => ([x, y, 0.85], [0.42, 0.28, 1.7], [x, y, 1.82]),
    };
    solid_box(view, centre, size, TRUE_PERSON, out);
    sphere(view, head, HEAD_RADIUS, TRUE_PERSON, out);
}

fn closeness(inputs: &SceneInputs) -> Option<f32> {
    room_view::closeness(inputs.state, inputs.track).map(|c| c as f32)
}

fn line_ends(scene: &Scene) -> Option<([f32; 2], [f32; 2])> {
    let [a, b] = *scene.links.first()?;
    let at = |i: usize| [scene.nodes[i][0] as f32, scene.nodes[i][1] as f32];
    Some((at(a), at(b)))
}

/// Points around the boards' line at `margin` metres from it: a stadium, so it stays convex.
fn around_line(a: [f32; 2], b: [f32; 2], margin: f32, z: f32) -> Vec<Point3> {
    let heading = (b[1] - a[1]).atan2(b[0] - a[0]);
    let cap = |end: [f32; 2], from: f32| {
        (0..=12).map(move |i| {
            let angle = from + std::f32::consts::PI * i as f32 / 12.0;
            [end[0] + margin * angle.cos(), end[1] + margin * angle.sin(), z]
        })
    };
    let quarter = std::f32::consts::FRAC_PI_2;
    cap(b, heading - quarter).chain(cap(a, heading + quarter)).collect()
}

/// Two floor bands around the boards' line: the inner one lights when the person moves close to
/// the line, the outer one when they move further away.
fn closeness_zone(view: &Projection, inputs: &SceneInputs, out: &mut Vec<Drawable>) {
    let Some((a, b)) = line_ends(inputs.scene) else { return };
    let near = closeness(inputs);
    let (outer, inner) = match near {
        None => (FAR.gamma_multiply(0.05), NEAR.gamma_multiply(0.06)),
        Some(c) if c >= room_view::CLOSE as f32 => {
            (FAR.gamma_multiply(0.06), NEAR.gamma_multiply(0.2 + 0.4 * c))
        }
        Some(c) => (FAR.gamma_multiply(0.3 - 0.3 * c), FLOOR),
    };
    let scene = inputs.scene;
    let inside = |p: Point3| -> Point3 {
        [p[0].clamp(0.0, scene.width as f32), p[1].clamp(0.0, scene.depth as f32), p[2]]
    };
    let band = |margin: f32, z: f32| -> Vec<Point3> { around_line(a, b, margin, z).into_iter().map(inside).collect() };
    on_floor(view, &band(FAR_LINE_M, 0.003), outer, 0.5, out);
    on_floor(view, &band(NEAR_LINE_M, 0.004), inner, 0.8, out);
}

fn closeness_label(painter: &egui::Painter, view: &Projection, inputs: &SceneInputs) {
    let Some((a, b)) = line_ends(inputs.scene) else { return };
    let text = match closeness(inputs) {
        None => return,
        Some(c) if c >= room_view::VERY_CLOSE as f32 => "very close to the line",
        Some(c) if c >= room_view::CLOSE as f32 => "close to the line",
        Some(_) => "far from the line",
    };
    let middle = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, 0.05];
    if let Some(p) = view.project(middle) {
        pill(painter, p.at + Vec2::new(0.0, 22.0), text, NEAR);
    }
}

/// A ring on the floor around the person that swells and shrinks with each breath; while the
/// breathing is still being read, a ring that fills up.
fn breathing_ring(view: &Projection, figure: &Figure, inputs: &SceneInputs, out: &mut Vec<Drawable>) {
    match figure.breath_bpm {
        Some(bpm) => {
            let breath = (TAU * bpm as f32 / 60.0 * inputs.seconds).sin();
            let radius = 0.5 + 0.12 * breath;
            ring(view, figure.at, radius, 360.0, Stroke::new(3.0_f32, BREATH.gamma_multiply(0.85)), out);
            ring(view, figure.at, radius + 0.06, 360.0, Stroke::new(8.0_f32, BREATH.gamma_multiply(0.15)), out);
        }
        None if inputs.still_seconds > 0.0 => {
            let filled = (inputs.still_seconds / BREATHING_AFTER_STILL_SECONDS).min(1.0);
            ring(view, figure.at, 0.5, 360.0, Stroke::new(2.0_f32, BREATH.gamma_multiply(0.15)), out);
            ring(view, figure.at, 0.5, 360.0 * filled, Stroke::new(3.0_f32, BREATH.gamma_multiply(0.7)), out);
        }
        None => {}
    }
}

fn ring(view: &Projection, centre: [f32; 2], radius: f32, degrees: f32, stroke: Stroke, out: &mut Vec<Drawable>) {
    let steps = ((degrees / 10.0).ceil() as usize).max(1);
    let points: Option<Vec<Pos2>> = (0..=steps)
        .map(|i| {
            let angle = (degrees.to_radians() * i as f32 / steps as f32) - std::f32::consts::FRAC_PI_2;
            view.project([centre[0] + radius * angle.cos(), centre[1] + radius * angle.sin(), 0.01])
                .map(|p| p.at)
        })
        .collect();
    if let Some(points) = points {
        out.push(Drawable {
            depth: f32::MAX / 2.0 - 3.0,
            shape: Shape::line(points, stroke),
        });
    }
}

fn nearest_board_line(view: &Projection, figure: &Figure, scene: &Scene, out: &mut Vec<Drawable>) {
    let Some(board) = nearest_board(figure.at, scene) else { return };
    let to = scene.nodes[board];
    segment(
        view,
        [figure.at[0], figure.at[1], 0.02],
        [to[0] as f32, to[1] as f32, 0.02],
        Stroke::new(1.5_f32, NEAREST_BOARD.gamma_multiply(0.8)),
        out,
    );
    if let Some(last) = out.last_mut() {
        last.depth = f32::MAX / 2.0 - 3.0;
    }
    ring(view, [to[0] as f32, to[1] as f32], 0.2, 360.0, Stroke::new(1.5_f32, NEAREST_BOARD), out);
}

fn nearest_board(at: [f32; 2], scene: &Scene) -> Option<usize> {
    let distance = |n: &[f64; 3]| (n[0] as f32 - at[0]).hypot(n[1] as f32 - at[1]);
    (0..scene.nodes.len()).min_by(|&i, &j| distance(&scene.nodes[i]).total_cmp(&distance(&scene.nodes[j])))
}

/// Breathing above the head, and the distance halfway along the line to the nearest board.
fn figure_labels(painter: &egui::Painter, view: &Projection, figure: &Figure, inputs: &SceneInputs) {
    let [x, y] = figure.at;
    let above_head = |text: &str, color: Color32| {
        if let Some(p) = view.project([x, y, 1.95]) {
            pill(painter, p.at, text, color);
        }
    };
    match figure.breath_bpm {
        Some(bpm) => above_head(&format!("breathing {bpm:.0} / min"), BREATH),
        None if inputs.still_seconds > 0.0 => {
            let left = BREATHING_AFTER_STILL_SECONDS - inputs.still_seconds;
            if left > 0.0 {
                above_head(&format!("reading breath… {left:.0} s"), BREATH.gamma_multiply(0.8));
            } else {
                above_head("breath too faint here", LABEL);
            }
        }
        None => {}
    }
    if inputs.one_line || !inputs.track.is_some_and(|t| t.xy.is_some()) {
        return;
    }
    let Some(board) = nearest_board(figure.at, inputs.scene) else { return };
    let to = inputs.scene.nodes[board];
    let metres = (to[0] as f32 - x).hypot(to[1] as f32 - y);
    let middle = [(x + to[0] as f32) / 2.0, (y + to[1] as f32) / 2.0, 0.05];
    if let Some(p) = view.project(middle) {
        pill(painter, p.at, &format!("{metres:.1} m"), NEAREST_BOARD);
    }
}

fn pill(painter: &egui::Painter, at: Pos2, text: &str, color: Color32) {
    let galley = painter.layout_no_wrap(text.to_string(), egui::FontId::proportional(13.0), color);
    let rect = Rect::from_center_size(at, galley.size() + Vec2::new(16.0, 8.0));
    painter.rect(
        rect,
        egui::CornerRadius::same(1),
        Color32::from_rgba_premultiplied(9, 9, 9, 220),
        Stroke::new(1.0_f32, color.gamma_multiply(0.6)),
        egui::StrokeKind::Inside,
    );
    painter.galley(rect.center() - galley.size() / 2.0, galley, color);
}

/// Around each board, a ring on the floor at the phone's distance from it: the person is where the rings meet.
fn phone_rings(view: &Projection, inputs: &SceneInputs, out: &mut Vec<Drawable>) {
    for phone in inputs.phones {
        for &(board, metres) in &phone.boards {
            let at = inputs.scene.nodes[board];
            let centre = [at[0] as f32, at[1] as f32];
            let radius = floor_distance(metres, at[2]);
            ring(view, centre, radius, 360.0, Stroke::new(2.5_f32, PHONE), out);
            ring(view, centre, radius, 360.0, Stroke::new(9.0_f32, PHONE.gamma_multiply(0.15)), out);
        }
    }
}

fn phone_labels(painter: &egui::Painter, view: &Projection, inputs: &SceneInputs) {
    for phone in inputs.phones {
        for &(board, metres) in &phone.boards {
            let at = inputs.scene.nodes[board];
            let radius = floor_distance(metres, at[2]);
            let towards_room = [
                inputs.scene.width as f32 / 2.0 - at[0] as f32,
                inputs.scene.depth as f32 / 2.0 - at[1] as f32,
            ];
            let length = towards_room[0].hypot(towards_room[1]).max(1e-3);
            let label_at = [
                at[0] as f32 + radius * towards_room[0] / length,
                at[1] as f32 + radius * towards_room[1] / length,
                0.05,
            ];
            if let Some(p) = view.project(label_at) {
                pill(painter, p.at, &format!("{metres:.1} m"), PHONE);
            }
        }
    }
}

/// The phone is carried around waist height, the boards hang higher: the straight-line distance on the floor.
fn floor_distance(metres: f64, board_height: f64) -> f32 {
    let rise = (board_height - PHONE_HEIGHT_M).max(0.0);
    (metres * metres - rise * rise).max(0.04).sqrt() as f32
}

fn bed_top(scene: &Scene, [x, y]: [f32; 2]) -> f32 {
    scene
        .furniture
        .iter()
        .filter(|item| item.get("kind").and_then(|k| k.as_str()) == Some("bed"))
        .filter_map(|item| {
            let number = |key: &str| item.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let (cx, cy, w, d) = (number("x"), number("y"), number("w"), number("d"));
            ((x - cx).abs() <= w / 2.0 && (y - cy).abs() <= d / 2.0).then(|| number("z") + number("h"))
        })
        .fold(0.0, f32::max)
}
