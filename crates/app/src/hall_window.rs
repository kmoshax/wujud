use std::sync::mpsc::Receiver;
use std::time::Instant;

use eframe::egui::{self, Pos2, RichText, Sense, Stroke, Vec2};
use hallsense_model::hall::{HallEngine, HallState};

use crate::hall_feed::HallSecond;
use crate::hall_view::HallView;

const SIDE_PANEL_WIDTH: f32 = 260.0;
const HISTORY_HEIGHT: f32 = 140.0;

pub struct HallWindow {
    seconds: Receiver<HallSecond>,
    engine: HallEngine,
    state: Option<HallState>,
    view: HallView,
    started: Instant,
}

impl HallWindow {
    pub fn new(seconds: Receiver<HallSecond>, engine: HallEngine) -> Self {
        HallWindow {
            seconds,
            engine,
            state: None,
            view: HallView::default(),
            started: Instant::now(),
        }
    }

    fn take_new_seconds(&mut self) {
        for second in self.seconds.try_iter() {
            let truth: Vec<i64> = second.present.iter().map(|&p| i64::from(p)).collect();
            if let Ok(state) =
                self.engine
                    .push(second.second as f64, second.mean, second.std, Some(&truth))
            {
                self.state = Some(state);
            }
        }
    }
}

impl eframe::App for HallWindow {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.take_new_seconds();
        egui::SidePanel::left("count")
            .exact_width(SIDE_PANEL_WIDTH)
            .show(ctx, |ui| count_panel(ui, self.state.as_ref()));
        egui::CentralPanel::default().show(ctx, |ui| match &self.state {
            Some(state) => self
                .view
                .show(ui, state, self.started.elapsed().as_secs_f32()),
            None => {
                ui.centered_and_justified(|ui| {
                    ui.weak("Collecting the first seconds of the lecture")
                });
            }
        });
        ctx.request_repaint();
    }
}

fn count_panel(ui: &mut egui::Ui, state: Option<&HallState>) {
    let Some(state) = state else { return };
    ui.weak("Students detected");
    match state.hall.count {
        Some(count) => ui.label(RichText::new(format!("{count:.0}")).size(48.0).strong()),
        None => ui.label(RichText::new("…").size(48.0)),
    };
    ui.weak(format!("of {} seats", state.scene.seats.len()));
    if let Some(truth) = state.hall.true_count {
        ui.add_space(12.0);
        ui.weak("Really there (simulator)");
        ui.label(RichText::new(truth.to_string()).size(28.0));
    }
    ui.add_space(16.0);
    ui.weak("Over time: detected (blue), real (grey)");
    history_chart(ui, &state.history, state.scene.seats.len() as f64);
}

fn history_chart(ui: &mut egui::Ui, history: &[(f64, Option<f64>, Option<i64>)], seats: f64) {
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), HISTORY_HEIGHT),
        Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_stroke(
        rect,
        egui::CornerRadius::ZERO,
        Stroke::new(1.0_f32, ui.visuals().weak_text_color()),
        egui::StrokeKind::Inside,
    );
    let n = history.len().max(2) as f32 - 1.0;
    let to_screen = |i: usize, value: f64| {
        Pos2::new(
            rect.left() + rect.width() * i as f32 / n,
            rect.bottom() - rect.height() * (value / seats.max(1.0)) as f32,
        )
    };
    let line = |values: Vec<Pos2>, color| painter.line(values, Stroke::new(1.5_f32, color));
    line(
        history
            .iter()
            .enumerate()
            .filter_map(|(i, h)| h.2.map(|t| to_screen(i, t as f64)))
            .collect(),
        egui::Color32::from_gray(130),
    );
    line(
        history
            .iter()
            .enumerate()
            .filter_map(|(i, h)| h.1.map(|c| to_screen(i, c)))
            .collect(),
        crate::scene_3d::PERSON,
    );
}
