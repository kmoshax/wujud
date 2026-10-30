use eframe::egui;
use wujud_model::motion_tracker::Track;
use wujud_model::room::{RoomState, RoomState3};

use crate::sensor_feed::SimulatedTruth;

/// Only what a few links can tell reliably: empty, someone still, someone moving. The model also
/// guesses lying / sitting / standing, but on real rooms with one or two links that guess is often
/// wrong (a person lying on the bed came out as "sitting"), so it is not shown.
pub fn state_banner(ui: &mut egui::Ui, state: Option<&RoomState>, _track: Option<&Track>) {
    let Some(decision) = state.and_then(|s| s.room.decision.as_ref()) else {
        ui.label(
            egui::RichText::new("Learning the room…")
                .size(26.0)
                .strong(),
        );
        ui.weak("The first answer comes after 20 seconds");
        return;
    };
    let headline = match decision.state3 {
        RoomState3::Empty => "Room empty",
        RoomState3::Still => "Someone here, still",
        RoomState3::Moving => "Someone moving",
    };
    ui.label(egui::RichText::new(headline).size(26.0).strong());
    if let Some(bpm) = decision.breath_bpm {
        ui.label(egui::RichText::new(format!("Breathing {bpm:.0} / min")).size(16.0));
    }
    ui.add_space(14.0);
    ui.weak("How sure");
    let p = &decision.probs;
    let shares = [
        ("empty", p[0]),
        ("still", p[1] + p[2] + p[3]),
        ("moving", p[4]),
    ];
    for (name, probability) in shares {
        ui.horizontal(|ui| {
            ui.add_sized([70.0, 18.0], egui::Label::new(name));
            ui.add(
                egui::ProgressBar::new(probability as f32)
                    .desired_width(140.0)
                    .text(format!("{:.0}%", probability * 100.0)),
            );
        });
    }
    ui.add_space(14.0);
    if let Some(distance) = decision.latch.distance {
        ui.weak("Change from the empty room");
        ui.label(format!(
            "{distance:.2}  (someone stayed above {:.1})",
            decision.latch.thr
        ));
    }
}

/// Where the person is, from the fast per-link disturbance (motion_tracker).
pub fn tracking(ui: &mut egui::Ui, track: Option<&Track>, link_names: &[String]) {
    let Some(track) = track else { return };
    ui.add_space(14.0);
    ui.separator();
    ui.weak("Position");
    match (track.xy, track.moving) {
        (Some([x, y]), true) => {
            ui.label(egui::RichText::new(format!("moving at ({x:.1}, {y:.1}) m")).size(16.0));
            ui.label(format!(
                "sure within 60 cm: {:.0}%",
                track.confidence * 100.0
            ));
        }
        (Some([x, y]), false) => {
            ui.label(format!("last seen moving at ({x:.1}, {y:.1}) m"));
        }
        (None, _) => {
            ui.label("waiting for the first movement");
        }
    }
    ui.add_space(8.0);
    ui.weak("Disturbance per link, above its quiet level");
    for (name, excess) in link_names.iter().zip(&track.link_excess) {
        ui.horizontal(|ui| {
            ui.add_sized([70.0, 18.0], egui::Label::new(name.as_str()).truncate());
            ui.add(
                egui::ProgressBar::new((*excess / 3.0).clamp(0.0, 1.0) as f32)
                    .desired_width(140.0)
                    .text(format!("{excess:.1}x")),
            );
        });
    }
}

pub fn simulator_truth(ui: &mut egui::Ui, truth: &SimulatedTruth) {
    ui.add_space(14.0);
    ui.separator();
    ui.weak("What the simulator is really doing");
    ui.label(egui::RichText::new(&truth.activity).size(18.0));
    if let Some(bpm) = truth.breath_bpm {
        ui.label(format!("breathing {bpm:.0} / min"));
    }
}
