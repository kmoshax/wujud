use eframe::egui;
use wujud_model::motion_tracker::Track;
use wujud_model::room::{RoomState, RoomState3};

use crate::phone_range::PhoneDistance;
use crate::sensor_feed::SimulatedTruth;
use crate::theme;

/// The model reads breathing after 25 calm seconds; a few more for its 30 s window to settle.
const BREATHING_AFTER_STILL_SECONDS: f64 = 30.0;

/// Only what a few links can tell reliably: empty, someone still, someone moving. The model also
/// guesses lying / sitting / standing, but on real rooms with one or two links that guess is often
/// wrong (a person lying on the bed came out as "sitting"), so it is not shown.
pub fn state_banner(ui: &mut egui::Ui, state: Option<&RoomState>, still_seconds: f64) {
    let Some(decision) = state.and_then(|s| s.room.decision.as_ref()) else {
        theme::card(ui, |ui| {
            theme::caption(ui, "Room");
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    egui::RichText::new("Learning the room…")
                        .size(22.0)
                        .strong(),
                );
            });
            ui.label(
                egui::RichText::new("The first answer comes after 20 seconds").color(theme::MUTED),
            );
        });
        return;
    };
    let (headline, color) = match decision.state3 {
        RoomState3::Empty => ("Room empty", theme::EMPTY),
        RoomState3::Still => ("Someone here, still", theme::STILL),
        RoomState3::Moving => ("Someone moving", theme::MOVING),
    };
    theme::card(ui, |ui| {
        theme::caption(ui, "Room");
        ui.horizontal(|ui| {
            theme::status_dot(ui, color, 6.0, 30.0);
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(headline)
                    .size(22.0)
                    .strong()
                    .color(color),
            );
        });
        match decision.breath_bpm {
            Some(bpm) => {
                ui.label(egui::RichText::new(format!("Breathing {bpm:.0} / min")).size(15.0));
            }
            None if decision.state3 == RoomState3::Still => {
                let left = BREATHING_AFTER_STILL_SECONDS - still_seconds;
                let hint = if left > 0.0 {
                    format!("Breathing: stay still, reading in {left:.0} s")
                } else {
                    "Breathing: too faint, sit nearer the boards' line".to_string()
                };
                ui.label(egui::RichText::new(hint).color(theme::MUTED));
            }
            None => {}
        }
    });
    theme::card(ui, |ui| {
        theme::caption(ui, "How sure");
        ui.add_space(4.0);
        let p = &decision.probs;
        let shares = [
            ("empty", p[0], theme::EMPTY),
            ("still", p[1] + p[2] + p[3], theme::STILL),
            ("moving", p[4], theme::MOVING),
        ];
        for (name, probability, color) in shares {
            theme::meter(
                ui,
                name,
                probability as f32,
                &format!("{:.0}%", probability * 100.0),
                color,
            );
        }
        if let Some(distance) = decision.latch.distance {
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(format!(
                    "change from empty {distance:.2} · stays above {:.1}",
                    decision.latch.thr
                ))
                .small()
                .color(theme::MUTED),
            );
        }
    });
}

/// Where the person is, from the fast per-link disturbance (motion_tracker).
pub struct Boards<'a> {
    pub names: &'a [String],
    pub at: &'a [[f64; 3]],
    /// Every board on one line (two boards): the links tell how near someone is to it, not where.
    pub one_line: bool,
    pub state: Option<&'a RoomState>,
}

pub const CLOSE: f64 = 0.33;
pub const VERY_CLOSE: f64 = 0.66;

/// How near a moving person is to the boards' line, 0 (far) .. 1 (on it); None while nobody moves.
/// The larger of the fast disturbance (above each link's learnt quiet level, quick but blind to
/// motion that never stopped) and the model's per-second activity (absolute, a second behind).
pub fn closeness(state: Option<&RoomState>, track: Option<&Track>) -> Option<f64> {
    let room_moving = state
        .and_then(|s| s.room.decision.as_ref())
        .is_some_and(|d| d.state3 == RoomState3::Moving);
    if !room_moving && !track.is_some_and(|t| t.moving) {
        return None;
    }
    let fast = track.map_or(0.0, |t| t.link_excess.iter().copied().fold(0.0, f64::max) / 2.0);
    let slow = state.map_or(0.0, |s| s.link_activity.iter().copied().fold(0.0, f64::max));
    Some(fast.max(slow).min(1.0))
}

pub fn tracking(ui: &mut egui::Ui, track: Option<&Track>, link_names: &[String], boards: &Boards) {
    let Some(track) = track else { return };
    theme::card(ui, |ui| {
        theme::caption(ui, "Position");
        if boards.one_line {
            let (words, color) = match closeness(boards.state, Some(track)) {
                None => ("no movement right now", theme::MUTED),
                Some(c) if c >= VERY_CLOSE => ("very close to the line between the boards", theme::MOVING),
                Some(c) if c >= CLOSE => ("close to the line between the boards", theme::ACCENT),
                Some(_) => ("moving, far from the line between the boards", theme::EMPTY),
            };
            ui.label(egui::RichText::new(words).size(16.0).strong());
            let value = closeness(boards.state, Some(track)).unwrap_or(0.0);
            theme::meter(ui, "closeness", value as f32, &format!("{:.0}%", value * 100.0), color);
            ui.label(
                egui::RichText::new(
                    "two boards measure one line: how near to it, not where along it",
                )
                .small()
                .color(theme::MUTED),
            );
            return;
        }
        match (track.xy, track.moving) {
            (Some([x, y]), true) => {
                ui.label(
                    egui::RichText::new(format!("({x:.1}, {y:.1}) m"))
                        .size(20.0)
                        .strong(),
                );
                ui.label(
                    egui::RichText::new(format!(
                        "moving · sure within 60 cm: {:.0}%",
                        track.confidence * 100.0
                    ))
                    .color(theme::MUTED),
                );
            }
            (Some([x, y]), false) => {
                ui.label(
                    egui::RichText::new(format!("({x:.1}, {y:.1}) m"))
                        .size(20.0)
                        .strong(),
                );
                ui.label(egui::RichText::new("last seen moving here").color(theme::MUTED));
            }
            (None, _) => {
                ui.label(egui::RichText::new("waiting for the first movement").color(theme::MUTED));
            }
        }
        if let Some(xy) = track.xy {
            if let Some((name, metres)) = nearest_board(xy, boards) {
                ui.label(
                    egui::RichText::new(format!("nearest board: {name}, {metres:.1} m")).size(15.0),
                );
            }
        }
    });
    theme::card(ui, |ui| {
        theme::caption(ui, "Disturbance per link");
        ui.add_space(4.0);
        for (name, excess) in link_names.iter().zip(&track.link_excess) {
            let color = if *excess >= 1.0 {
                theme::ACCENT
            } else {
                theme::EMPTY
            };
            theme::meter(
                ui,
                name,
                (*excess / 3.0) as f32,
                &format!("{excess:.1}×"),
                color,
            );
        }
        ui.label(
            egui::RichText::new("above each link's quiet level")
                .small()
                .color(theme::MUTED),
        );
    });
}

/// How far the phone the person carries is from each board, from each board's signal strength.
pub fn phone_distances(ui: &mut egui::Ui, phones: &[PhoneDistance], node_names: &[String]) {
    theme::card(ui, |ui| {
        theme::caption(ui, "Distance from each board");
        if phones.is_empty() {
            ui.label(
                egui::RichText::new("join your phone to Wi-Fi \"wujud\" and keep it on you")
                    .color(theme::MUTED),
            );
            return;
        }
        for phone in phones {
            for &(board, metres) in &phone.boards {
                let name = node_names.get(board).map_or("board", String::as_str);
                ui.label(egui::RichText::new(format!("{name}: {metres:.1} m")).size(16.0).strong());
            }
            ui.label(
                egui::RichText::new(format!("phone {} · from signal strength, about ±1 m", phone.phone))
                    .small()
                    .color(theme::MUTED),
            );
        }
    });
}

pub fn simulator_truth(ui: &mut egui::Ui, truth: &SimulatedTruth) {
    theme::card(ui, |ui| {
        theme::caption(ui, "Simulator ground truth");
        ui.label(egui::RichText::new(&truth.activity).size(18.0).strong());
        if let Some(bpm) = truth.breath_bpm {
            ui.label(egui::RichText::new(format!("breathing {bpm:.0} / min")).color(theme::MUTED));
        }
    });
}

fn nearest_board<'a>([x, y]: [f64; 2], boards: &Boards<'a>) -> Option<(&'a str, f64)> {
    boards
        .names
        .iter()
        .zip(boards.at)
        .map(|(name, at)| (name.as_str(), (at[0] - x).hypot(at[1] - y)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}
