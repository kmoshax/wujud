use std::f32::consts::TAU;

use eframe::egui::{self, Color32, Mesh, Pos2, Rect, Sense, Shape, Stroke, Vec2};
use hallsense_model::motion_tracker::Track;
use hallsense_model::room::{Pose, RoomState, RoomState3};
use hallsense_model::scene::Scene;

use crate::camera::{add, lerp, Camera, Point3, Projection};
use crate::sensor_feed::SimulatedTruth;

const SKY_TOP: Color32 = Color32::from_rgb(22, 26, 36);
const SKY_BOTTOM: Color32 = Color32::from_rgb(10, 12, 18);
const FLOOR: Color32 = Color32::from_rgb(30, 35, 45);
const FLOOR_LINE: Color32 = Color32::from_rgba_premultiplied(60, 70, 88, 110);
const WALL: Color32 = Color32::from_rgba_premultiplied(34, 40, 54, 150);
const EDGE: Color32 = Color32::from_rgb(78, 90, 110);
const FURNITURE: Color32 = Color32::from_rgb(112, 118, 132);
const BED: Color32 = Color32::from_rgb(96, 112, 150);
const DESK: Color32 = Color32::from_rgb(150, 116, 84);
const WARDROBE: Color32 = Color32::from_rgb(82, 88, 100);
const BUFFET: Color32 = Color32::from_rgb(128, 98, 74);
const MIRROR: Color32 = Color32::from_rgb(190, 214, 230);
const HUB: Color32 = Color32::from_rgb(255, 138, 76);
const DEVICE: Color32 = Color32::from_rgb(200, 206, 218);
const LABEL: Color32 = Color32::from_rgb(170, 178, 192);
const HEAT: Color32 = Color32::from_rgb(90, 200, 255);
const SHADOW: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 90);
pub(crate) const BOARD: Color32 = HUB;
pub(crate) const LINK_IDLE: Color32 = Color32::from_rgb(45, 70, 105);
pub(crate) const LINK_ACTIVE: Color32 = Color32::from_rgb(110, 210, 255);
pub(crate) const PERSON: Color32 = Color32::from_rgb(79, 179, 255);
pub(crate) const TRUE_PERSON: Color32 = Color32::from_rgba_premultiplied(70, 70, 70, 70);

const FLOOR_TILE_M: f32 = 0.5;
pub(crate) const HEAD_RADIUS: f32 = 0.11;
const BREATH_SWELL: f32 = 0.06;
/// Share of the screen distance the drawn person closes each frame towards the tracked position.
const FOLLOW: f32 = 0.12;
const HEAT_SHOWN_ABOVE: f32 = 0.04;

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
}

#[derive(Default)]
pub struct Scene3d {
    camera: Option<Camera>,
    drawn_at: Option<[f32; 2]>,
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
        if let Some(figure) = self.figure(inputs) {
            person(&view, &figure, inputs.seconds, &mut drawables);
        }
        if let Some(truth) = inputs.truth {
            true_person(&view, truth, &mut drawables);
        }

        drawables.sort_by(|a, b| b.depth.total_cmp(&a.depth));
        painter.extend(drawables.into_iter().map(|d| d.shape));
        for (node, name) in scene.nodes.iter().zip(inputs.node_names) {
            if let Some(at) = view.project(add(point(*node), [0.0, 0.0, 0.22])) {
                painter.text(
                    at.at,
                    egui::Align2::CENTER_BOTTOM,
                    name,
                    egui::FontId::proportional(13.0),
                    LABEL,
                );
            }
        }
        painter.text(
            viewport.left_bottom() + Vec2::new(10.0, -10.0),
            egui::Align2::LEFT_BOTTOM,
            "drag to rotate · scroll to zoom · blue floor = where the person may be",
            egui::FontId::proportional(11.0),
            Color32::from_gray(110),
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
        drawn[0] += (target[0] - drawn[0]) * FOLLOW;
        drawn[1] += (target[1] - drawn[1]) * FOLLOW;
        let confidence = inputs.track.map_or(0.0, |t| t.confidence) as f32;
        // Posture is not shown: with one or two links the lie / sit / stand guess is unreliable.
        let pose = match decision.map(|d| d.state3) {
            Some(RoomState3::Moving) => Pose::Moving,
            _ => Pose::Stand,
        };
        Some(Figure {
            at: *drawn,
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

/// The shared board is the orange box; every other end (laptop, phone) is a light slab.
fn node_model(view: &Projection, at: Point3, is_hub: bool, out: &mut Vec<Drawable>) {
    floor_disc(view, [at[0], at[1]], 0.12, SHADOW, out);
    if is_hub {
        solid_box(view, at, [0.12, 0.08, 0.05], HUB, out);
        sphere(
            view,
            add(at, [0.0, 0.0, 0.05]),
            0.025,
            Color32::from_rgb(90, 160, 255),
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

fn person(view: &Projection, figure: &Figure, seconds: f32, out: &mut Vec<Drawable>) {
    let [x, y] = figure.at;
    let color = PERSON.gamma_multiply(figure.opacity);
    let breathing = figure
        .breath_bpm
        .map_or(0.0, |bpm| (TAU * bpm as f32 / 60.0 * seconds).sin());
    let chest = 1.0 + BREATH_SWELL * breathing;
    floor_disc(view, figure.at, 0.32, SHADOW, out);
    match figure.pose {
        Pose::Lie => {
            solid_box(view, [x, y, 0.62], [1.5, 0.4 * chest, 0.2], color, out);
            sphere(view, [x - 0.88, y, 0.68], HEAD_RADIUS, color, out);
        }
        Pose::Sit | Pose::Still => {
            solid_box(view, [x, y, 0.62], [0.36, 0.26 * chest, 0.55], color, out);
            solid_box(view, [x + 0.18, y, 0.45], [0.4, 0.24, 0.14], color, out);
            sphere(view, [x, y, 1.03], HEAD_RADIUS, color, out);
        }
        _ => {
            let stride = if figure.pose == Pose::Moving {
                (seconds * 6.0).sin()
            } else {
                0.0
            };
            let bob = 0.03 * stride.abs();
            for side in [-1.0f32, 1.0] {
                let swing = 0.12 * stride * side;
                solid_box(
                    view,
                    [x + swing, y + 0.09 * side, 0.42 + bob],
                    [0.13, 0.13, 0.84],
                    color,
                    out,
                );
            }
            solid_box(
                view,
                [x, y, 1.15 + bob],
                [0.38, 0.24 * chest, 0.62],
                color,
                out,
            );
            sphere(view, [x, y, 1.6 + bob], HEAD_RADIUS, color, out);
        }
    }
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
