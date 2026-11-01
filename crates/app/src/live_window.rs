use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Color32, ColorImage, Pos2, Sense, Stroke, TextureHandle, TextureOptions, Vec2,
};
use wujud_model::motion_tracker::{MotionTracker, Track};
use wujud_model::room::{RoomEngine, RoomState};

use crate::link_history::LinkHistory;
use crate::room_setup::LiveRoom;
use crate::room_view;
use crate::scene_3d::{Belief, Scene3d, SceneInputs};
use crate::sensor_feed::{FeedEvent, SimulatedTruth};
use crate::theme;

const CHANGE_RANGE_DB: f32 = 4.0;
const HEATMAP_HEIGHT: f32 = 150.0;
const RSSI_HEIGHT: f32 = 60.0;
const ENGINE_TICK: Duration = Duration::from_secs(1);
const TRACKER_TICK: Duration = Duration::from_millis(100);
const SIDE_PANEL_WIDTH: f32 = 300.0;
const SIGNAL_PANEL_HEIGHT: f32 = 290.0;

pub struct LiveWindow {
    feed: Receiver<FeedEvent>,
    links: Vec<LinkHistory>,
    banners: Vec<Option<String>>,
    heatmaps: Vec<Option<TextureHandle>>,
    newest_board_seconds: f64,
    engine: RoomEngine,
    node_names: Vec<String>,
    link_names: Vec<String>,
    room_state: Option<RoomState>,
    tracker: MotionTracker,
    track: Option<Track>,
    last_track: Instant,
    truth: Option<SimulatedTruth>,
    scene: Scene3d,
    last_tick: Instant,
    started: Instant,
}

impl LiveWindow {
    pub fn new(feed: Receiver<FeedEvent>, room: LiveRoom) -> Self {
        let LiveRoom {
            engine,
            node_names,
            link_names,
            ..
        } = room;
        let scene = engine.scene();
        let tracker = MotionTracker::new(engine.link_ends(), scene.width, scene.depth);
        LiveWindow {
            node_names,
            link_names: link_names.clone(),
            tracker,
            track: None,
            last_track: Instant::now(),
            feed,
            links: link_names.iter().cloned().map(LinkHistory::new).collect(),
            banners: vec![None; link_names.len()],
            heatmaps: vec![None; link_names.len()],
            newest_board_seconds: f64::NAN,
            engine,
            room_state: None,
            truth: None,
            scene: Scene3d::default(),
            last_tick: Instant::now(),
            started: Instant::now(),
        }
    }

    /// The engine runs on the boards' own clock, like research/live.py; one decision per wall-clock second.
    fn tick_engine(&mut self) {
        if self.last_tick.elapsed() < ENGINE_TICK {
            return;
        }
        self.last_tick = Instant::now();
        if self.newest_board_seconds.is_finite() {
            self.room_state = self.engine.tick(self.newest_board_seconds).ok();
        }
    }

    fn tick_tracker(&mut self) {
        let elapsed = self.last_track.elapsed();
        if elapsed < TRACKER_TICK {
            return;
        }
        self.last_track = Instant::now();
        let fresh = |value: Option<f64>| value.unwrap_or(f64::NAN);
        let motion: Vec<f64> = self
            .links
            .iter()
            .map(|l| fresh(l.disturbance_db()))
            .collect();
        let shadowing: Vec<f64> = self
            .links
            .iter()
            .map(|l| fresh(l.rssi_spread_db()))
            .collect();
        if motion.iter().any(|m| !m.is_nan()) {
            self.track = Some(
                self.tracker
                    .update(&motion, &shadowing, elapsed.as_secs_f64()),
            );
        }
    }

    fn take_new_events(&mut self) {
        let now = self.started.elapsed().as_secs_f64();
        for event in self.feed.try_iter() {
            match event {
                FeedEvent::Frame(frame) => {
                    self.engine.add(
                        frame.link,
                        frame.board_seconds,
                        &frame.iq,
                        f64::from(frame.rssi),
                    );
                    self.newest_board_seconds = self.newest_board_seconds.max(frame.board_seconds);
                    self.links[frame.link].push(&frame, now);
                }
                FeedEvent::Banner { link, text } => self.banners[link] = Some(text),
                FeedEvent::Truth(truth) => self.truth = Some(truth),
            }
        }
    }

    fn link_panel(&mut self, ui: &mut egui::Ui, index: usize) {
        let link = &self.links[index];
        let alive = link.packets_per_second() >= 10.0;
        ui.horizontal(|ui| {
            theme::status_dot(ui, if alive { theme::ALIVE } else { theme::SILENT }, 4.0, 18.0);
            ui.label(egui::RichText::new(&link.name).strong());
            let detail = match link.latest_rssi() {
                Some(rssi) => format!("{:.0} packets/s · {rssi} dBm", link.packets_per_second()),
                None => format!("{:.0} packets/s", link.packets_per_second()),
            };
            ui.label(egui::RichText::new(detail).color(theme::MUTED).monospace());
            if let Some(banner) = &self.banners[index] {
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
                .corner_radius(egui::CornerRadius::same(6)),
        );
        ui.label(
            egui::RichText::new("CSI per subcarrier · warm = stronger than usual, cyan = weaker")
                .small()
                .color(theme::MUTED),
        );
        draw_rssi(ui, &self.links[index], width);
        ui.add_space(14.0);
    }
}

impl eframe::App for LiveWindow {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.take_new_events();
        self.tick_engine();
        self.tick_tracker();
        egui::SidePanel::left("state")
            .exact_width(SIDE_PANEL_WIDTH)
            .frame(panel_frame())
            .show(ctx, |ui| {
                ui.label(egui::RichText::new("wujud").size(13.0).strong().color(theme::ACCENT).extra_letter_spacing(3.0));
                ui.add_space(8.0);
                room_view::state_banner(ui, self.room_state.as_ref(), self.track.as_ref());
                room_view::tracking(ui, self.track.as_ref(), &self.link_names);
                if let Some(truth) = &self.truth {
                    room_view::simulator_truth(ui, truth);
                }
            });
        egui::TopBottomPanel::bottom("signals")
            .exact_height(SIGNAL_PANEL_HEIGHT)
            .frame(panel_frame())
            .show(ctx, |ui| {
                let count = self.links.len();
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
                let (xs, ys) = self.tracker.grid();
                let inputs = SceneInputs {
                    scene: self.engine.scene(),
                    node_names: &self.node_names,
                    state: self.room_state.as_ref(),
                    track: self.track.as_ref(),
                    belief: self.track.as_ref().and_then(|t| t.xy).map(|_| Belief {
                        xs,
                        ys,
                        values: self.tracker.belief(),
                    }),
                    truth: self.truth.as_ref(),
                    seconds: self.started.elapsed().as_secs_f32(),
                };
                self.scene.show(ui, &inputs);
            });
        ctx.request_repaint();
    }
}

/// Time runs left to right, subcarriers low to high frequency bottom to top; warm = stronger than usual, cyan = weaker.
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
        Color32::from_rgb(mix(from.r(), to.r()), mix(from.g(), to.g()), mix(from.b(), to.b()))
    };
    let base = theme::BACKGROUND;
    let amount = t.abs();
    if t >= 0.0 {
        let hot = if amount < 0.6 {
            toward(Color32::from_rgb(180, 40, 90), Color32::from_rgb(255, 120, 50), amount / 0.6)
        } else {
            toward(Color32::from_rgb(255, 120, 50), Color32::from_rgb(255, 230, 120), (amount - 0.6) / 0.4)
        };
        toward(base, hot, amount.min(1.0))
    } else {
        toward(base, Color32::from_rgb(40, 200, 255), amount)
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
    painter.rect_filled(rect, egui::CornerRadius::same(6), theme::BACKGROUND);
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
        let to = ((column + 1) * values.len() / columns).max(from + 1).min(values.len());
        let bucket = &values[from..to];
        let x = inner.left() + inner.width() * column as f32 / (columns - 1) as f32;
        let (lo, hi) = bucket
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        painter.line_segment(
            [Pos2::new(x, y_of(lo)), Pos2::new(x, y_of(hi))],
            Stroke::new(2.0_f32, theme::ACCENT.gamma_multiply(0.18)),
        );
        let mean = bucket.iter().sum::<f32>() / bucket.len() as f32;
        means.push(Pos2::new(x, y_of(mean)));
    }
    painter.line(means, Stroke::new(1.6_f32, theme::ACCENT));
    painter.text(
        rect.min + Vec2::new(8.0, 4.0),
        egui::Align2::LEFT_TOP,
        format!("RSSI {low:.0} to {high:.0} dBm"),
        egui::FontId::proportional(11.0),
        theme::MUTED,
    );
}
