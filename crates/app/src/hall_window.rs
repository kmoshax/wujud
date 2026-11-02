use std::sync::mpsc::Receiver;
use std::time::Instant;

use eframe::egui::{self, Pos2, RichText, Sense, Stroke, Vec2};
use wujud_model::hall::{HallEngine, HallState};

use crate::hall_feed::HallSecond;
use crate::hall_view::HallView;
use crate::live_core::Source;
use crate::theme;

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
            .frame(
                egui::Frame::new()
                    .fill(theme::PANEL)
                    .inner_margin(egui::Margin::same(16))
                    .stroke(Stroke::new(1.0_f32, theme::CARD_EDGE)),
            )
            .show(ctx, |ui| count_panel(ui, self.state.as_ref()));
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::BACKGROUND))
            .show(ctx, |ui| match &self.state {
                Some(state) => self
                    .view
                    .show(ui, state, self.started.elapsed().as_secs_f32()),
                None => {
                    ui.centered_and_justified(|ui| {
                        ui.weak("Collecting the first seconds of the lecture")
                    });
                }
            });
        ctx.request_repaint_after(crate::FRAME_INTERVAL);
    }
}

fn count_panel(ui: &mut egui::Ui, state: Option<&HallState>) {
    let Some(state) = state else { return };
    ui.label(theme::pixel_text("WUJUD", 2.0, theme::TEXT));
    ui.add_space(8.0);
    theme::source_tag(ui, Source::Simulation);
    ui.add_space(16.0);
    theme::caption(ui, "Students detected");
    match state.hall.count {
        Some(count) => ui.label(theme::pixel_text(format!("{count:.0}"), 6.0, theme::TEXT)),
        None => ui.label(theme::pixel_text("...", 6.0, theme::MUTED)),
    };
    ui.label(RichText::new(format!("of {} seats", state.scene.seats.len())).color(theme::MUTED));
    if let Some(truth) = state.hall.true_count {
        ui.add_space(16.0);
        theme::caption(ui, "Really there (simulator)");
        ui.label(theme::pixel_text(truth.to_string(), 3.0, theme::MUTED));
    }
    ui.add_space(20.0);
    theme::caption(ui, "Over time");
    ui.label(
        RichText::new("red = detected, grey = really there")
            .small()
            .color(theme::MUTED),
    );
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
        Stroke::new(1.0_f32, theme::CARD_EDGE),
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
        theme::MUTED,
    );
    line(
        history
            .iter()
            .enumerate()
            .filter_map(|(i, h)| h.1.map(|c| to_screen(i, c)))
            .collect(),
        theme::RED,
    );
}
