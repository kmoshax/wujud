use eframe::egui;
use wujud_model::motion_tracker::Track;
use wujud_model::room::{RoomState, RoomState3};

use crate::sensor_feed::SimulatedTruth;
use crate::theme;

/// Only what a few links can tell reliably: empty, someone still, someone moving. The model also
/// guesses lying / sitting / standing, but on real rooms with one or two links that guess is often
/// wrong (a person lying on the bed came out as "sitting"), so it is not shown.
pub fn state_banner(ui: &mut egui::Ui, state: Option<&RoomState>, _track: Option<&Track>) {
    let Some(decision) = state.and_then(|s| s.room.decision.as_ref()) else {
        theme::card(ui, |ui| {
            theme::caption(ui, "Room");
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(egui::RichText::new("Learning the room…").size(22.0).strong());
            });
            ui.label(egui::RichText::new("The first answer comes after 20 seconds").color(theme::MUTED));
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
            ui.label(egui::RichText::new(headline).size(22.0).strong().color(color));
        });
        if let Some(bpm) = decision.breath_bpm {
            ui.label(egui::RichText::new(format!("Breathing {bpm:.0} / min")).size(15.0));
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
pub fn tracking(ui: &mut egui::Ui, track: Option<&Track>, link_names: &[String]) {
    let Some(track) = track else { return };
    theme::card(ui, |ui| {
        theme::caption(ui, "Position");
        match (track.xy, track.moving) {
            (Some([x, y]), true) => {
                ui.label(egui::RichText::new(format!("({x:.1}, {y:.1}) m")).size(20.0).strong());
                ui.label(
                    egui::RichText::new(format!(
                        "moving · sure within 60 cm: {:.0}%",
                        track.confidence * 100.0
                    ))
                    .color(theme::MUTED),
                );
            }
            (Some([x, y]), false) => {
                ui.label(egui::RichText::new(format!("({x:.1}, {y:.1}) m")).size(20.0).strong());
                ui.label(egui::RichText::new("last seen moving here").color(theme::MUTED));
            }
            (None, _) => {
                ui.label(egui::RichText::new("waiting for the first movement").color(theme::MUTED));
            }
        }
    });
    theme::card(ui, |ui| {
        theme::caption(ui, "Disturbance per link");
        ui.add_space(4.0);
        for (name, excess) in link_names.iter().zip(&track.link_excess) {
            let color = if *excess >= 1.0 { theme::ACCENT } else { theme::EMPTY };
            theme::meter(
                ui,
                name,
                (*excess / 3.0) as f32,
                &format!("{excess:.1}×"),
                color,
            );
        }
        ui.label(egui::RichText::new("above each link's quiet level").small().color(theme::MUTED));
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
