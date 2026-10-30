use std::collections::BTreeMap;

use eframe::egui::{self, Color32, Sense, Vec2};
use hallsense_model::hall::HallState;

use crate::camera::{Camera, Point3};
use crate::scene_3d::{
    point, radio_link, room_shell, solid_box, sphere, Drawable, BOARD, HEAD_RADIUS, PERSON,
    TRUE_PERSON,
};

const STEP: Color32 = Color32::from_rgb(52, 58, 70);
const SEAT_EMPTY: Color32 = Color32::from_rgb(70, 76, 88);
const SEAT_LIKELY: Color32 = Color32::from_rgb(79, 179, 255);
const ROW_DEPTH: f32 = 0.9;
const HALL_ZOOM_OUT: f32 = 1.35;
const SEAT_SIZE: [f32; 3] = [0.42, 0.42, 0.45];
const STUDENT_SIZE: [f32; 3] = [0.38, 0.26, 0.75];

#[derive(Default)]
pub struct HallView {
    camera: Option<Camera>,
}

impl HallView {
    pub fn show(&mut self, ui: &mut egui::Ui, state: &HallState, seconds: f32) {
        let scene = &state.scene;
        let (width, depth, height) = (scene.width as f32, scene.depth as f32, scene.height as f32);
        let (viewport, response) = ui.allocate_exact_size(ui.available_size(), Sense::drag());
        let camera = self.camera.get_or_insert_with(|| {
            let mut whole_hall = Camera::looking_at_room(width, depth, height * 0.6);
            whole_hall.distance *= HALL_ZOOM_OUT;
            whole_hall
        });
        camera.follow_mouse(&response, ui.input(|i| i.smooth_scroll_delta.y));
        let view = camera.projection(viewport);

        let mut drawables = Vec::new();
        room_shell(&view, width, depth, height, &mut drawables);
        for SeatRow { floor, front, back } in seat_rows(&scene.seats) {
            let rise = floor.max(0.02);
            solid_box(
                &view,
                [width / 2.0, (front + back) / 2.0, rise / 2.0],
                [width, back - front + ROW_DEPTH, rise],
                STEP,
                &mut drawables,
            );
        }
        let occupancy = state.hall.p.clone().unwrap_or_default();
        let detected =
            most_likely_seats(&occupancy, state.hall.count.unwrap_or(0.0).round() as usize);
        for (index, seat) in scene.seats.iter().enumerate() {
            let [x, y, floor] = point(*seat);
            let likelihood = occupancy.get(index).copied().unwrap_or(0.0) as f32;
            solid_box(
                &view,
                [x, y, floor + SEAT_SIZE[2] / 2.0],
                SEAT_SIZE,
                SEAT_EMPTY.lerp_to_gamma(SEAT_LIKELY, likelihood.clamp(0.0, 1.0)),
                &mut drawables,
            );
            if detected.contains(&index) {
                student(&view, [x, y, floor], PERSON, &mut drawables);
            }
            if state
                .hall
                .truth
                .as_ref()
                .is_some_and(|truth| truth.get(index) == Some(&1))
            {
                student(&view, [x, y + 0.05, floor], TRUE_PERSON, &mut drawables);
            }
        }
        for (link, [a, b]) in scene.links.iter().enumerate() {
            let activity = state.link_activity.get(link).copied().unwrap_or(0.0) as f32 * 0.4;
            radio_link(
                &view,
                point(scene.nodes[*a]),
                point(scene.nodes[*b]),
                activity,
                seconds,
                &mut drawables,
            );
        }
        for node in &scene.nodes {
            solid_box(
                &view,
                point(*node),
                [0.18, 0.18, 0.12],
                BOARD,
                &mut drawables,
            );
        }

        drawables.sort_by(|a, b| b.depth.total_cmp(&a.depth));
        let painter = ui.painter_at(viewport);
        painter.extend(drawables.into_iter().map(|d| d.shape));
        painter.text(
            viewport.left_bottom() + Vec2::new(8.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            "drag to rotate · scroll to zoom · blue = detected, grey = really there",
            egui::FontId::proportional(11.0),
            Color32::from_gray(110),
        );
    }
}

fn student(
    view: &crate::camera::Projection,
    seat: Point3,
    color: Color32,
    out: &mut Vec<Drawable>,
) {
    let [x, y, floor] = seat;
    solid_box(
        view,
        [x, y, floor + 0.45 + STUDENT_SIZE[2] / 2.0],
        STUDENT_SIZE,
        color,
        out,
    );
    sphere(
        view,
        [x, y, floor + 0.45 + STUDENT_SIZE[2] + HEAD_RADIUS],
        HEAD_RADIUS,
        color,
        out,
    );
}

struct SeatRow {
    floor: f32,
    front: f32,
    back: f32,
}

/// One step of a tiered hall per distinct seat-floor height (a flat hall has a single step).
fn seat_rows(seats: &[[f64; 3]]) -> Vec<SeatRow> {
    let mut rows: BTreeMap<i64, SeatRow> = BTreeMap::new();
    for seat in seats {
        let (y, floor) = (seat[1] as f32, seat[2] as f32);
        let row = rows
            .entry((seat[2] * 100.0).round() as i64)
            .or_insert(SeatRow {
                floor,
                front: y,
                back: y,
            });
        row.front = row.front.min(y);
        row.back = row.back.max(y);
    }
    rows.into_values().collect()
}

fn most_likely_seats(occupancy: &[f64], count: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..occupancy.len()).collect();
    order.sort_by(|&a, &b| occupancy[b].total_cmp(&occupancy[a]));
    order.truncate(count);
    order
}
