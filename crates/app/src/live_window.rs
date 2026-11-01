use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Color32, ColorImage, Pos2, Sense, Stroke, TextureHandle, TextureOptions, Vec2,
};
use wujud_model::motion_tracker::{MotionTracker, Track};
use wujud_model::room::{Pose, RoomEngine, RoomState, RoomState3};

use crate::learned_tracker::LearnedTracker;
use crate::link_history::LinkHistory;
use crate::link_net::{LinkNetTracker, LinkSecond};
use crate::phone_range::PhoneRange;
use crate::room_sense::RoomSense;
use crate::room_setup::{self, LiveRoom};
use crate::room_view;
use crate::scene_3d::{Belief, Scene3d, SceneInputs};
use crate::sensor_feed::{FeedEvent, SimulatedTruth};
use crate::theme;

const CHANGE_RANGE_DB: f32 = 4.0;
const HEATMAP_HEIGHT: f32 = 150.0;
const RSSI_HEIGHT: f32 = 60.0;
const ENGINE_TICK: Duration = Duration::from_secs(1);
const TRACKER_TICK: Duration = Duration::from_millis(100);
/// Share of the way the shown position moves towards each new answer of the network (once a second).
const FOLLOW_ANSWER: f64 = 0.5;
const SIDE_PANEL_WIDTH: f32 = 300.0;
const SIGNAL_PANEL_HEIGHT: f32 = 290.0;
/// A board's links show up one by one; the room is rebuilt once none has come for this long.
const SETTLE_NEW_LINKS: Duration = Duration::from_secs(3);
const RELAYOUT_AFTER: Duration = Duration::from_secs(30);

/// Live boards only: a board switched on while the window is open joins the room.
pub struct Growth {
    /// Every link heard so far, read or not (a link whose ends are not placed yet may be placed later).
    pub seen: Vec<String>,
    /// The links the feed hands over, in the engine's order.
    pub reading: Arc<Mutex<Vec<String>>>,
    pub room_file: Option<PathBuf>,
    /// Each link's signal strength, for placing the boards (refreshed from the live links on a rebuild).
    pub rssi: std::collections::BTreeMap<String, f64>,
}

pub struct LiveWindow {
    feed: Receiver<FeedEvent>,
    links: Vec<LinkHistory>,
    banners: Vec<Option<String>>,
    heatmaps: Vec<Option<TextureHandle>>,
    newest_board_seconds: f64,
    engine: RoomEngine,
    node_names: Vec<String>,
    node_ids: Vec<String>,
    phones: PhoneRange,
    link_names: Vec<String>,
    one_line: bool,
    /// The boards' real places are known (a room file, or the simulator): only then a position.
    placed: bool,
    still_since: Option<Instant>,
    room_state: Option<RoomState>,
    tracker: MotionTracker,
    /// The network trained on simulated rooms; it places a walking person when the boards are placed.
    learned: Option<Locator>,
    learned_xy: Option<[f64; 2]>,
    /// The network's newest answer (where, and the activity it thinks), refreshed once a second.
    learned_answer: Option<([f64; 2], String)>,
    relaid: bool,
    last_learned: Instant,
    track: Option<Track>,
    last_track: Instant,
    truth: Option<SimulatedTruth>,
    scene: Scene3d,
    last_tick: Instant,
    started: Instant,
    growth: Option<Growth>,
    new_link_at: Option<Instant>,
    /// Live boards: the room read from its own links (drift, and the empty room it learns by itself).
    sense: Option<RoomSense>,
    last_frame: Instant,
}

impl LiveWindow {
    pub fn new(feed: Receiver<FeedEvent>, room: LiveRoom) -> Self {
        let scene = room.engine.scene();
        let tracker = MotionTracker::new(room.engine.link_ends(), scene.width, scene.depth);
        let mut window = LiveWindow {
            one_line: false,
            placed: room.placed,
            still_since: None,
            node_names: Vec::new(),
            node_ids: room.node_ids,
            phones: PhoneRange::default(),
            link_names: Vec::new(),
            tracker,
            learned: None,
            learned_xy: None,
            learned_answer: None,
            relaid: false,
            last_learned: Instant::now(),
            track: None,
            last_track: Instant::now(),
            feed,
            links: Vec::new(),
            banners: Vec::new(),
            heatmaps: Vec::new(),
            newest_board_seconds: f64::NAN,
            engine: room.engine,
            room_state: None,
            truth: None,
            scene: Scene3d::default(),
            last_tick: Instant::now(),
            started: Instant::now(),
            growth: None,
            new_link_at: None,
            sense: None,
            last_frame: Instant::now(),
        };
        window.read_links(room.node_names, room.link_names);
        window
    }

    pub fn growing(mut self, growth: Growth) -> Self {
        let reading = growth.reading.lock().unwrap().clone();
        self.sense = Some(RoomSense::new(&reading));
        self.growth = Some(growth);
        self
    }

    fn read_links(&mut self, node_names: Vec<String>, link_names: Vec<String>) {
        let ends = self.engine.link_ends();
        let scene = self.engine.scene();
        self.tracker = MotionTracker::new(ends.clone(), scene.width, scene.depth);
        let previous = self.learned.take();
        self.learned = Locator::load(&ends, &link_names, scene.width, scene.depth).map(|mut learned| {
            if let Some(previous) = &previous {
                learned.carry_over(previous);
            }
            learned
        });
        self.one_line = on_one_line(&ends);
        self.node_names = node_names;
        // A room laid out again keeps what it saw on the links it still has: nothing disappears.
        let mut old: Vec<LinkHistory> = std::mem::take(&mut self.links);
        self.links = link_names
            .iter()
            .map(|name| match old.iter().position(|l| &l.name == name) {
                Some(i) => old.swap_remove(i),
                None => LinkHistory::new(name.clone()),
            })
            .collect();
        self.banners = vec![None; link_names.len()];
        self.heatmaps = vec![None; link_names.len()];
        self.link_names = link_names;
    }

    /// A board switched on later: once its links stop arriving, the room is laid out again with them.
    fn join_new_links(&mut self) {
        let settled = self.new_link_at.is_some_and(|t| t.elapsed() >= SETTLE_NEW_LINKS);
        // A few seconds of RSSI at the start place the boards roughly; half a minute places them better.
        let relayout = self.growth.is_some() && !self.relaid && self.started.elapsed() >= RELAYOUT_AFTER;
        if !(settled || relayout) || self.growth.is_none() {
            return;
        }
        self.new_link_at = None;
        self.relaid |= relayout;
        let heard: Vec<(String, f64)> = {
            let reading = self.growth.as_ref().expect("growth").reading.lock().unwrap().clone();
            reading
                .into_iter()
                .zip(&self.links)
                .filter_map(|(name, link)| {
                    let mut rssi: Vec<f64> = link.rssi_history().map(f64::from).collect();
                    rssi.sort_by(f64::total_cmp);
                    Some((name, *rssi.get(rssi.len() / 2)?))
                })
                .collect()
        };
        let growth = self.growth.as_mut().expect("growth");
        growth.rssi.extend(heard);
        let growth = self.growth.as_ref().expect("growth");
        let reading = growth.reading.lock().unwrap().clone();
        let room = match room_setup::room_engine(&growth.seen, growth.room_file.as_deref(), &growth.rssi) {
            Ok(room) if relayout || room.links != reading => room,
            Ok(_) => return,
            Err(error) => return eprintln!("new links: {error}"),
        };
        println!("links: {}", room.links.join(", "));
        *growth.reading.lock().unwrap() = room.links.clone();
        let sensed = self.engine.sensed;
        self.engine = room.engine;
        self.engine.sensed = sensed;
        self.node_ids = room.node_ids;
        self.placed = room.placed;
        let sense = RoomSense::new(&room.links);
        self.sense = Some(match &self.sense {
            Some(previous) => sense.carry_over(previous),
            None => sense,
        });
        self.read_links(room.node_names, room.link_names);
    }

    /// The engine runs on the boards' own clock, like research/live.py; one decision per wall-clock second.
    fn tick_engine(&mut self) {
        if self.last_tick.elapsed() < ENGINE_TICK {
            return;
        }
        self.last_tick = Instant::now();
        if self.newest_board_seconds.is_finite() {
            if let Some(sense) = &mut self.sense {
                // RoomSense holds its last state through short gaps itself; None is "learning".
                self.engine.sensed = sense.tick(self.newest_board_seconds);
            }
            let started = Instant::now();
            self.room_state = self.engine.tick(self.newest_board_seconds).ok();
            if std::env::var_os("WUJUD_LOG").is_some() {
                println!("engine tick {} ms", started.elapsed().as_millis());
            }
            if self.sense.is_some() && self.engine.sensed.is_none() {
                // Nothing read from the room yet: "learning", not the model's guess.
                self.room_state = None;
            }
            // The posture is the one taught in this room, or none: the model's own guess comes from a
            // model that read a real empty room as moving all the time.
            let taught = self.sense.as_ref().map(|s| s.reading.posture);
            let decision = self.room_state.as_mut().and_then(|s| s.room.decision.as_mut());
            if let (Some(taught), Some(decision)) = (taught, decision) {
                if decision.state3 == RoomState3::Still {
                    decision.state = taught.unwrap_or(Pose::Still);
                }
            }
        }
        let still = self
            .room_state
            .as_ref()
            .and_then(|s| s.room.decision.as_ref())
            .is_some_and(|d| d.state3 == RoomState3::Still);
        self.still_since = still.then(|| self.still_since.unwrap_or_else(Instant::now));
        if std::env::var_os("WUJUD_LOG").is_some() {
            self.log_second();
        }
    }

    /// `WUJUD_LOG=1`: one line a second on stdout, to see why the room reads the way it does.
    fn log_second(&self) {
        let Some(state) = &self.room_state else {
            if let Some(r) = self.sense.as_ref().map(|s| &s.reading) {
                println!(
                    "t learning {:?}/{} drift {:?} above {:.3} stirring {} glrt {:?}",
                    r.learning,
                    crate::room_sense::LEARN_EMPTY_SECONDS,
                    r.drift_db.map(|d| (d * 1000.0).round() / 1000.0),
                    r.moving_above,
                    r.stirring,
                    self.engine.breathing_strength.map(|g| (g * 100.0).round() / 100.0),
                );
            }
            return;
        };
        let Some(decision) = &state.room.decision else { return };
        let rates: Vec<String> = self.links.iter().map(|l| format!("{:.0}", l.packets_per_second())).collect();
        let activity: Vec<String> = state.link_activity.iter().map(|a| format!("{a:.2}")).collect();
        let sensed = self.sense.as_ref().map(|s| s.reading.clone()).unwrap_or_default();
        println!(
            "t {:.0} {:?} drift {:?} above {:.3} presence {:?} learning {:?} glrt {:?} probs {:?} activity [{}] packets/s [{}] breath {:?} net {:?}",
            state.t,
            decision.state,
            sensed.drift_db.map(|d| (d * 1000.0).round() / 1000.0),
            sensed.moving_above,
            sensed.presence,
            sensed.learning,
            self.engine.breathing_strength.map(|g| (g * 100.0).round() / 100.0),
            decision.probs,
            activity.join(" "),
            rates.join(" "),
            decision.breath_bpm,
            self.learned_answer.as_ref().map(|([x, y], what)| format!("{x:.2},{y:.2} {what}")),
        );
    }

    /// Live boards: moving as the drift detector reads it, not every flicker of a single link, so a
    /// person sitting still is neither walked around nor hidden and shown again.
    fn sensed_moving(&self) -> bool {
        match &self.sense {
            Some(sense) => sense.reading.state == Some(RoomState3::Moving),
            None => self.track.as_ref().is_some_and(|t| t.moving),
        }
    }

    fn room_empty(&self) -> bool {
        self.room_state
            .as_ref()
            .and_then(|s| s.room.decision.as_ref())
            .is_some_and(|d| d.state3 == RoomState3::Empty)
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
        if self.last_learned.elapsed() >= Duration::from_secs(1) {
            self.last_learned = Instant::now();
            if let Some(learned) = &mut self.learned {
                learned.add_second(&self.links);
                self.learned_answer = learned.answer();
            }
            if let (true, Some((xy, _))) = (self.sensed_moving(), &self.learned_answer) {
                self.learned_xy = Some(match self.learned_xy {
                    Some(old) => [old[0] + (xy[0] - old[0]) * FOLLOW_ANSWER, old[1] + (xy[1] - old[1]) * FOLLOW_ANSWER],
                    None => *xy,
                });
            }
        }
        if motion.iter().any(|m| !m.is_nan()) {
            let mut track = self
                .tracker
                .update(&motion, &shadowing, elapsed.as_secs_f64());
            if self.sense.is_some() {
                track.moving = self.sensed_moving();
            }
            if self.learned_xy.is_some() {
                track.xy = self.learned_xy;
            }
            if self.one_line || !self.placed {
                track.xy = None;
            }
            self.track = Some(track);
        }
    }

    fn take_new_events(&mut self) {
        let now = self.started.elapsed().as_secs_f64();
        for event in self.feed.try_iter() {
            match event {
                FeedEvent::Frame(frame) if frame.link >= self.links.len() => {}
                FeedEvent::Frame(frame) => {
                    self.engine.add(
                        frame.link,
                        frame.board_seconds,
                        &frame.iq,
                        f64::from(frame.rssi),
                    );
                    if let Some(sense) = &mut self.sense {
                        sense.add(frame.link, frame.board_seconds, &frame.iq);
                    }
                    self.newest_board_seconds = self.newest_board_seconds.max(frame.board_seconds);
                    self.links[frame.link].push(&frame, now);
                }
                FeedEvent::Banner { link, text } => {
                    if let Some(banner) = self.banners.get_mut(link) {
                        *banner = Some(text);
                    }
                }
                FeedEvent::Truth(truth) => self.truth = Some(truth),
                FeedEvent::Heard { link, rssi } => {
                    self.phones.hear(&link, f64::from(rssi), &self.node_ids);
                }
                FeedEvent::NewLink(link) => {
                    if let Some(growth) = &mut self.growth {
                        growth.seen.push(link);
                        self.new_link_at = Some(Instant::now());
                    }
                }
            }
        }
    }

    fn link_panel(&mut self, ui: &mut egui::Ui, index: usize) {
        let link = &self.links[index];
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
        let logging = std::env::var_os("WUJUD_LOG").is_some();
        let gap = self.last_frame.elapsed();
        let step = Instant::now();
        self.take_new_events();
        let events = step.elapsed();
        self.join_new_links();
        self.tick_engine();
        self.tick_tracker();
        if logging && (gap.as_millis() > 500 || step.elapsed().as_millis() > 200) {
            println!(
                "slow frame: {} ms since the last one, events {} ms, all work {} ms",
                gap.as_millis(),
                events.as_millis(),
                step.elapsed().as_millis()
            );
        }
        self.last_frame = Instant::now();
        egui::SidePanel::left("state")
            .exact_width(SIDE_PANEL_WIDTH)
            .frame(panel_frame())
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new("wujud")
                        .size(13.0)
                        .strong()
                        .color(theme::ACCENT)
                        .extra_letter_spacing(3.0),
                );
                ui.add_space(8.0);
                let still_seconds = self.still_since.map_or(0.0, |t| t.elapsed().as_secs_f64());
                room_view::state_banner(
                    ui,
                    self.room_state.as_ref(),
                    still_seconds,
                    self.sense.as_ref().map(|s| &s.reading),
                );
                if let Some(sense) = &mut self.sense {
                    room_view::room_sense(ui, sense);
                }
                let boards = room_view::Boards {
                    names: &self.node_names,
                    at: &self.engine.scene().nodes,
                    one_line: self.one_line,
                    placed: self.placed,
                    state: self.room_state.as_ref(),
                };
                let phones = self.phones.distances(&self.node_ids);
                if self.growth.is_some() {
                    room_view::phone_distances(ui, &phones, &self.node_names);
                }
                let track = self.track.as_ref().filter(|_| !self.room_empty());
                room_view::tracking(ui, track, &self.link_names, &boards);
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
                let phones = self.phones.distances(&self.node_ids);
                let inputs = SceneInputs {
                    scene: self.engine.scene(),
                    node_names: &self.node_names,
                    state: self.room_state.as_ref(),
                    track: self.track.as_ref().filter(|_| !self.room_empty()),
                    belief: self.track.as_ref().and_then(|t| t.xy).filter(|_| !self.room_empty() && self.learned.is_none()).map(|_| Belief {
                        xs,
                        ys,
                        values: self.tracker.belief(),
                    }),
                    truth: self.truth.as_ref(),
                    seconds: self.started.elapsed().as_secs_f32(),
                    one_line: self.one_line,
                    phones: &phones,
                    still_seconds: self.still_since.map_or(0.0, |t| t.elapsed().as_secs_f32()),
                };
                self.scene.show(ui, &inputs);
            });
        ctx.request_repaint_after(crate::FRAME_INTERVAL);
    }
}

fn on_one_line(links: &[([f64; 2], [f64; 2])]) -> bool {
    let ends: Vec<[f64; 2]> = links.iter().flat_map(|&(a, b)| [a, b]).collect();
    let Some(&first) = ends.first() else {
        return true;
    };
    let Some(&second) = ends
        .iter()
        .find(|p| (p[0] - first[0]).hypot(p[1] - first[1]) > 0.1)
    else {
        return true;
    };
    let (dx, dy) = (second[0] - first[0], second[1] - first[1]);
    let length = dx.hypot(dy);
    ends.iter()
        .all(|p| ((p[0] - first[0]) * dy - (p[1] - first[1]) * dx).abs() / length < 0.1)
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
        Color32::from_rgb(
            mix(from.r(), to.r()),
            mix(from.g(), to.g()),
            mix(from.b(), to.b()),
        )
    };
    let base = theme::BACKGROUND;
    let amount = t.abs();
    if t >= 0.0 {
        let hot = if amount < 0.6 {
            toward(
                Color32::from_rgb(180, 40, 90),
                Color32::from_rgb(255, 120, 50),
                amount / 0.6,
            )
        } else {
            toward(
                Color32::from_rgb(255, 120, 50),
                Color32::from_rgb(255, 230, 120),
                (amount - 0.6) / 0.4,
            )
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

/// The network that places a walking person: every link at once (link_net), or with
/// `WUJUD_TRACKER=points` the earlier one that scores each link on its own (learned_tracker).
enum Locator {
    Links(LinkNetTracker),
    Points(LearnedTracker),
}

impl Locator {
    fn load(ends: &[([f64; 2], [f64; 2])], names: &[String], width: f64, depth: f64) -> Option<Self> {
        let models = room_setup::models_dir();
        if std::env::var("WUJUD_TRACKER").as_deref() != Ok("points") {
            match LinkNetTracker::load(&models.join("link_net.json"), ends, names, width, depth) {
                Ok(tracker) => return Some(Locator::Links(tracker)),
                Err(error) => eprintln!("link_net.json: {error}"),
            }
        }
        LearnedTracker::load(&models.join("learned_tracker.json"), ends, names, width, depth)
            .map(Locator::Points)
            .map_err(|error| eprintln!("learned_tracker.json: {error}"))
            .ok()
    }

    fn carry_over(&mut self, previous: &Locator) {
        match (self, previous) {
            (Locator::Links(now), Locator::Links(before)) => now.carry_over(before),
            (Locator::Points(now), Locator::Points(before)) => now.carry_over(before),
            _ => {}
        }
    }

    fn add_second(&mut self, links: &[LinkHistory]) {
        match self {
            Locator::Links(tracker) => {
                let second: Vec<Option<LinkSecond>> = links
                    .iter()
                    .map(|l| {
                        let (spread, shape) = l.subcarrier_db()?;
                        Some(LinkSecond { motion: l.disturbance_db()?, shadow: l.rssi_spread_db()?, spread, shape })
                    })
                    .collect();
                tracker.add_second(&second);
            }
            Locator::Points(tracker) => {
                let second: Vec<(Option<f64>, Option<f64>)> =
                    links.iter().map(|l| (l.disturbance_db(), l.rssi_spread_db())).collect();
                tracker.add_second(&second);
            }
        }
    }

    fn answer(&self) -> Option<([f64; 2], String)> {
        match self {
            Locator::Links(tracker) => tracker.position().map(|(xy, what)| (xy, what.to_owned())),
            Locator::Points(tracker) => tracker.position().map(|xy| (xy, String::from("?"))),
        }
    }
}
