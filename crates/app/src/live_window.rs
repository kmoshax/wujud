use std::time::Instant;

use eframe::egui::{self, Color32, ColorImage, Pos2, Sense, Stroke, TextureHandle, TextureOptions, Vec2};

use crate::link_history::LinkHistory;
use crate::live_core::LiveCore;
use crate::room_view;
use crate::scene_3d::{Belief, Scene3d, SceneInputs};
use crate::theme;

const CHANGE_RANGE_DB: f32 = 4.0;
const HEATMAP_HEIGHT: f32 = 150.0;
const RSSI_HEIGHT: f32 = 60.0;
const SIDE_PANEL_WIDTH: f32 = 300.0;
const SIGNAL_PANEL_HEIGHT: f32 = 290.0;

pub struct LiveWindow {
    core: LiveCore,
    heatmaps: Vec<Option<TextureHandle>>,
    scene: Scene3d,
    last_frame: Instant,
}

impl LiveWindow {
    pub fn new(core: LiveCore) -> Self {
        LiveWindow {
            core,
            heatmaps: Vec::new(),
            scene: Scene3d::default(),
            last_frame: Instant::now(),
        }
    }

    fn link_panel(&mut self, ui: &mut egui::Ui, index: usize) {
        let link = &self.core.links[index];
        let alive = link.packets_per_second() >= 10.0;
        ui.horizontal_wrapped(|ui| {
            theme::status_dot(
                ui,
                if alive { theme::ALIVE } else { theme::SILENT },
                4.0,
                18.0,
            );
            ui.label(egui::RichText::new(&link.name).strong());
            let detail = match link.latest_rssi() {
                Some(rssi) => format!("{:.0} packets/s · {rssi} dBm", link.packets_per_second()),
                None => format!("{:.0} packets/s", link.packets_per_second()),
            };
            ui.label(egui::RichText::new(detail).color(theme::MUTED).monospace());
            if let Some(banner) = &self.core.banners[index] {
                ui.label(egui::RichText::new(banner).color(theme::MUTED));
            }
        });
        let image = heatmap_image(&link.amplitude_changes_db());
        let texture = match &mut self.heatmaps[index] {
            Some(texture) => {
                texture.set(image, TextureOptions::NEAREST);
                texture.clone()
            }
            empty => empty
                .insert(ui.ctx().load_texture(
                    format!("csi-{index}"),
                    image,
                    TextureOptions::NEAREST,
                ))
                .clone(),
        };
        let width = ui.available_width();
        ui.add(
            egui::Image::new((texture.id(), Vec2::new(width, HEATMAP_HEIGHT)))
                .corner_radius(egui::CornerRadius::ZERO),
        );
        ui.label(
            egui::RichText::new("CSI per subcarrier · red = stronger than usual, grey = weaker")
                .small()
                .color(theme::MUTED),
        );
        draw_rssi(ui, &self.core.links[index], width);
        ui.add_space(14.0);
    }
}

impl eframe::App for LiveWindow {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let gap = self.last_frame.elapsed();
        let step = Instant::now();
        self.core.tick();
        if std::env::var_os("WUJUD_LOG").is_some() && (gap.as_millis() > 500 || step.elapsed().as_millis() > 200) {
            println!(
                "slow frame: {} ms since the last one, all work {} ms",
                gap.as_millis(),
                step.elapsed().as_millis()
            );
        }
        self.last_frame = Instant::now();
        let core = &mut self.core;
        if self.heatmaps.len() != core.links.len() {
            self.heatmaps = vec![None; core.links.len()];
        }
        egui::SidePanel::left("state")
            .exact_width(SIDE_PANEL_WIDTH)
            .frame(panel_frame())
            .show(ctx, |ui| {
                ui.label(theme::pixel_text("WUJUD", 2.0, theme::TEXT));
                ui.add_space(8.0);
                theme::source_tag(ui, core.source());
                ui.add_space(8.0);
                room_view::state_banner(
                    ui,
                    core.room_state.as_ref(),
                    core.still_seconds(),
                    core.sense.as_ref().map(|s| &s.reading),
                );
                if let Some(sense) = &mut core.sense {
                    room_view::room_sense(ui, sense);
                }
                let boards = room_view::Boards {
                    names: &core.node_names,
                    at: &core.engine.scene().nodes,
                    one_line: core.one_line,
                    placed: core.placed,
                    state: core.room_state.as_ref(),
                };
                if core.is_growing() {
                    room_view::phone_distances(ui, &core.phone_distances(), &core.node_names);
                }
                room_view::tracking(ui, core.shown_track(), &core.link_names, &boards);
                if let Some(truth) = &core.truth {
                    room_view::simulator_truth(ui, truth);
                }
            });
        egui::TopBottomPanel::bottom("signals")
            .exact_height(SIGNAL_PANEL_HEIGHT)
            .frame(panel_frame())
            .show(ctx, |ui| {
                let count = self.core.links.len();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.columns(count.clamp(1, 3), |columns| {
                        for index in 0..count {
                            self.link_panel(&mut columns[index % columns.len()], index);
                        }
                    });
                });
            });
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let core = &self.core;
                let (xs, ys) = core.tracker.grid();
                let phones = core.phone_distances();
                let track = core.shown_track();
                let inputs = SceneInputs {
                    scene: core.engine.scene(),
                    node_names: &core.node_names,
                    state: core.room_state.as_ref(),
                    track,
                    belief: track.and_then(|t| t.xy).filter(|_| !core.has_locator()).map(|_| Belief {
                        xs,
                        ys,
                        values: core.tracker.belief(),
                    }),
                    truth: core.truth.as_ref(),
                    seconds: core.seconds() as f32,
                    one_line: core.one_line,
                    phones: &phones,
                    still_seconds: core.still_seconds() as f32,
                };
                self.scene.show(ui, &inputs);
            });
        ctx.request_repaint_after(crate::FRAME_INTERVAL);
    }
}

/// Time runs left to right, subcarriers low to high frequency bottom to top; red = stronger than usual, grey = weaker.
fn heatmap_image(changes: &[Vec<f32>]) -> ColorImage {
    let width = changes.len().max(1);
    let height = changes.first().map_or(1, Vec::len).max(1);
    let mut image = ColorImage::new([width, height], theme::BACKGROUND);
    for (x, packet) in changes.iter().enumerate() {
        for (k, &change) in packet.iter().enumerate() {
            image[(x, height - 1 - k)] = diverging_color(change / CHANGE_RANGE_DB);
        }
    }
    image
}

fn diverging_color(value: f32) -> Color32 {
    let t = value.clamp(-1.0, 1.0);
    let toward = |from: Color32, to: Color32, amount: f32| {
        let mix = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * amount) as u8;
        Color32::from_rgb(
            mix(from.r(), to.r()),
            mix(from.g(), to.g()),
            mix(from.b(), to.b()),
        )
    };
    let base = theme::BACKGROUND;
    let amount = t.abs();
    if t >= 0.0 {
        toward(base, theme::RED, amount)
    } else {
        toward(base, theme::MUTED, amount)
    }
}

fn panel_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(theme::PANEL)
        .inner_margin(egui::Margin::same(14))
        .stroke(Stroke::new(1.0_f32, theme::CARD_EDGE))
}

fn draw_rssi(ui: &mut egui::Ui, link: &LinkHistory, width: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, RSSI_HEIGHT), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, egui::CornerRadius::ZERO, theme::BACKGROUND);
    let values: Vec<f32> = link.rssi_history().map(f32::from).collect();
    if values.len() < 2 {
        return;
    }
    let (low, high) = values
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let span = (high - low).max(4.0);
    let inner = rect.shrink2(Vec2::new(6.0, 8.0));
    let y_of = |v: f32| inner.bottom() - inner.height() * (v - low) / span;
    let columns = (inner.width() / 2.0).max(2.0) as usize;
    let mut means = Vec::with_capacity(columns);
    for column in 0..columns {
        let from = column * values.len() / columns;
        let to = ((column + 1) * values.len() / columns)
            .max(from + 1)
            .min(values.len());
        let bucket = &values[from..to];
        let x = inner.left() + inner.width() * column as f32 / (columns - 1) as f32;
        let (lo, hi) = bucket
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        painter.line_segment(
            [Pos2::new(x, y_of(lo)), Pos2::new(x, y_of(hi))],
            Stroke::new(2.0_f32, theme::TEXT.gamma_multiply(0.12)),
        );
        let mean = bucket.iter().sum::<f32>() / bucket.len() as f32;
        means.push(Pos2::new(x, y_of(mean)));
    }
    painter.line(means, Stroke::new(1.5_f32, theme::TEXT));
    painter.text(
        rect.min + Vec2::new(8.0, 4.0),
        egui::Align2::LEFT_TOP,
        format!("RSSI {low:.0} to {high:.0} dBm"),
        egui::FontId::proportional(11.0),
        theme::MUTED,
    );
}

