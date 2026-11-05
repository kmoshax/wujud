//! The live room without any window: packets in, a plain `Snapshot` of what the room looks like out.
//! The egui window and the Bevy view both draw on top of it.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use wujud_model::motion_tracker::{MotionTracker, Track};
use wujud_model::room::{Pose, RoomEngine, RoomState, RoomState3};

use crate::learned_tracker::LearnedTracker;
use crate::link_history::LinkHistory;
use crate::link_net::{LinkNetTracker, LinkSecond};
use crate::phone_range::{PhoneDistance, PhoneRange};
use crate::room_sense::RoomSense;
use crate::room_setup::{self, LiveRoom};
use crate::sensor_feed::{FeedEvent, SimulatedTruth};

const ENGINE_TICK: Duration = Duration::from_secs(1);
const TRACKER_TICK: Duration = Duration::from_millis(100);
/// A gap in the boards' data (a hub hiccup, a relayout) keeps the last reading this long, so the
/// person does not vanish; someone moving is then shown still until data comes back.
const HOLD_THROUGH_GAP: Duration = Duration::from_secs(30);
/// A board's links show up one by one; the room is rebuilt once none has come for this long.
const SETTLE_NEW_LINKS: Duration = Duration::from_secs(3);
/// The boards are placed again from the newest signal strengths this often, in the background.
const RELAYOUT_EVERY: Duration = Duration::from_secs(10);
/// A new layout replaces the shown one only when a board moved this far: no jitter on screen.
const RELAYOUT_MOVE_M: f64 = 0.3;
/// Each new median signal strength moves the remembered one by this share (about a minute of memory).
const RSSI_MEMORY_SHARE: f64 = 0.3;
const DRIFT_HISTORY_SECONDS: usize = 120;

/// Where the packets come from, shown on screen so the simulator is never taken for the real room.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    Simulation,
    Live,
    Replay,
}

impl Source {
    pub fn tag(self) -> &'static str {
        match self {
            Source::Simulation => "SIMULATION",
            Source::Live => "LIVE · REAL BOARDS",
            Source::Replay => "REPLAY · REAL RECORDING",
        }
    }

    pub fn note(self) -> &'static str {
        match self {
            Source::Simulation => "simulated room · no boards",
            Source::Live => "the ESP32 boards, right now",
            Source::Replay => "real boards, recorded earlier",
        }
    }
}

/// Live boards only: a board switched on while the window is open joins the room.
pub struct Growth {
    /// Every link heard so far, read or not (a link whose ends are not placed yet may be placed later).
    pub seen: Vec<String>,
    /// The links the feed hands over, in the engine's order.
    pub reading: Arc<Mutex<Vec<String>>>,
    pub room_file: Option<PathBuf>,
    /// Each link's signal strength, for placing the boards (refreshed from the live links on a rebuild).
    pub rssi: std::collections::BTreeMap<String, f64>,
    pub recorded: bool,
}

/// One second of the drift detector: the links' recent change, and the line above which it is movement.
#[derive(Clone, Copy)]
pub struct DriftSecond {
    pub drift_db: f64,
    pub moving_above: f64,
}

pub struct LiveCore {
    feed: Receiver<FeedEvent>,
    pub links: Vec<LinkHistory>,
    pub banners: Vec<Option<String>>,
    newest_board_seconds: f64,
    pub engine: RoomEngine,
    pub node_names: Vec<String>,
    pub node_ids: Vec<String>,
    pub phones: PhoneRange,
    pub link_names: Vec<String>,
    pub one_line: bool,
    /// The boards' real places are known (a room file, or the simulator): only then a position.
    pub placed: bool,
    still_since: Option<Instant>,
    pub room_state: Option<RoomState>,
    /// When the room last had a reading, to hold it through a gap in the boards' data.
    last_reading: Option<Instant>,
    pub tracker: MotionTracker,
    /// The network trained on simulated rooms; it places a walking person when the boards are placed.
    learned: Option<Locator>,
    learned_xy: Option<[f64; 2]>,
    /// The network's newest answer (where, and the activity it thinks), refreshed once a second.
    learned_answer: Option<([f64; 2], String)>,
    last_learned: Instant,
    places: std::collections::BTreeMap<String, [f64; 3]>,
    notes: Vec<String>,
    last_layout: Instant,
    layout_job: Option<Receiver<std::collections::BTreeMap<String, [f64; 3]>>>,
    pub track: Option<Track>,
    last_track: Instant,
    pub truth: Option<SimulatedTruth>,
    last_tick: Instant,
    started: Instant,
    growth: Option<Growth>,
    new_link_at: Option<Instant>,
    /// Live boards: the room read from its own links (drift, and the empty room it learns by itself).
    pub sense: Option<RoomSense>,
    /// The simulator: the same drift detector, only to show its reading; it decides nothing.
    drift_meter: Option<RoomSense>,
    drift_history: VecDeque<DriftSecond>,
}

impl LiveCore {
    pub fn new(feed: Receiver<FeedEvent>, room: LiveRoom) -> Self {
        let scene = room.engine.scene();
        let tracker = MotionTracker::new(room.engine.link_ends(), scene.width, scene.depth);
        let drift_meter = Some(RoomSense::new(&room.links));
        let mut core = LiveCore {
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
            last_learned: Instant::now(),
            places: room.places,
            notes: room.notes,
            last_layout: Instant::now(),
            layout_job: None,
            track: None,
            last_track: Instant::now(),
            feed,
            links: Vec::new(),
            banners: Vec::new(),
            newest_board_seconds: f64::NAN,
            engine: room.engine,
            room_state: None,
            last_reading: None,
            truth: None,
            last_tick: Instant::now(),
            started: Instant::now(),
            growth: None,
            new_link_at: None,
            sense: None,
            drift_meter,
            drift_history: VecDeque::new(),
        };
        core.read_links(room.node_names, room.link_names);
        core
    }

    pub fn growing(mut self, growth: Growth) -> Self {
        let reading = growth.reading.lock().unwrap().clone();
        self.sense = Some(RoomSense::new(&reading));
        self.drift_meter = None;
        self.growth = Some(growth);
        self
    }

    /// Everything that is due: new packets, a board joining, the once-a-second engine, the tracker.
    pub fn tick(&mut self) {
        let logging = std::env::var_os("WUJUD_LOG").is_some();
        let step = Instant::now();
        self.take_new_events();
        let events = step.elapsed();
        self.join_new_links();
        self.tick_engine();
        self.tick_tracker();
        if logging && step.elapsed().as_millis() > 200 {
            println!(
                "slow tick: events {} ms, all work {} ms",
                events.as_millis(),
                step.elapsed().as_millis()
            );
        }
    }

    pub fn seconds(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub fn still_seconds(&self) -> f64 {
        self.still_since.map_or(0.0, |t| t.elapsed().as_secs_f64())
    }

    pub fn is_growing(&self) -> bool {
        self.growth.is_some()
    }

    pub fn source(&self) -> Source {
        match &self.growth {
            None => Source::Simulation,
            Some(growth) if growth.recorded => Source::Replay,
            Some(_) => Source::Live,
        }
    }

    /// The network that places people is the trained one, so no belief map is drawn.
    pub fn has_locator(&self) -> bool {
        self.learned.is_some()
    }

    pub fn phone_distances(&self) -> Vec<PhoneDistance> {
        self.phones.distances(&self.node_ids)
    }

    pub fn room_empty(&self) -> bool {
        self.room_state
            .as_ref()
            .and_then(|s| s.room.decision.as_ref())
            .is_some_and(|d| d.state3 == RoomState3::Empty)
    }

    /// The track to draw: none while the room is empty.
    pub fn shown_track(&self) -> Option<&Track> {
        self.track.as_ref().filter(|_| !self.room_empty())
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
        self.link_names = link_names;
    }

    /// A board switched on later: once its links stop arriving, the room is laid out again with them.
    /// Without a room file, the boards are also placed again every few seconds from the newest signal
    /// strengths (read while nobody walks, who would shadow them), and shown there once one has moved.
    fn join_new_links(&mut self) {
        let settled = self
            .new_link_at
            .is_some_and(|t| t.elapsed() >= SETTLE_NEW_LINKS);
        let moved = self
            .finished_layout()
            .is_some_and(|places| self.boards_moved(&places));
        if self.last_layout.elapsed() >= RELAYOUT_EVERY {
            self.last_layout = Instant::now();
            self.remember_rssi();
            self.start_layout();
        }
        if settled || moved {
            self.new_link_at = None;
            self.rebuild_room();
        }
    }

    fn remember_rssi(&mut self) {
        if self.sensed_moving() {
            return;
        }
        let Some(growth) = self.growth.as_mut() else {
            return;
        };
        let reading = growth.reading.lock().unwrap().clone();
        for (name, link) in reading.iter().zip(&self.links) {
            let mut rssi: Vec<f64> = link.rssi_history().map(f64::from).collect();
            rssi.sort_by(f64::total_cmp);
            if let Some(&median) = rssi.get(rssi.len() / 2) {
                let remembered = growth.rssi.entry(name.clone()).or_insert(median);
                *remembered += RSSI_MEMORY_SHARE * (median - *remembered);
            }
        }
    }

    fn start_layout(&mut self) {
        let Some(growth) = self.growth.as_ref().filter(|g| g.room_file.is_none()) else {
            return;
        };
        if self.layout_job.is_some() {
            return;
        }
        let (seen, rssi) = (growth.seen.clone(), growth.rssi.clone());
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = send.send(room_setup::board_places(
                &seen,
                &rssi,
                room_setup::ESP32_RADIO,
            ));
        });
        self.layout_job = Some(receive);
    }

    fn finished_layout(&mut self) -> Option<std::collections::BTreeMap<String, [f64; 3]>> {
        let places = self.layout_job.as_ref()?.try_recv().ok()?;
        self.layout_job = None;
        Some(places)
    }

    fn boards_moved(&self, places: &std::collections::BTreeMap<String, [f64; 3]>) -> bool {
        if places.keys().ne(self.places.keys()) {
            return true;
        }
        let flat = |p: &std::collections::BTreeMap<String, [f64; 3]>| -> Vec<[f64; 2]> {
            p.values().map(|v| [v[0], v[1]]).collect()
        };
        let (new, shown) = (flat(places), flat(&self.places));
        wujud_model::board_layout::largest_move(
            &wujud_model::board_layout::align_to(&new, &shown),
            &shown,
        ) > RELAYOUT_MOVE_M
    }

    fn rebuild_room(&mut self) {
        let Some(growth) = self.growth.as_mut() else {
            return;
        };
        let reading = growth.reading.lock().unwrap().clone();
        let room = match room_setup::room_engine(
            &growth.seen,
            growth.room_file.as_deref(),
            &growth.rssi,
        ) {
            Ok(room) if room.links != reading || room.places != self.places => room,
            Ok(_) => return,
            Err(error) => return eprintln!("new links: {error}"),
        };
        *growth.reading.lock().unwrap() = room.links.clone();
        println!("links: {}", room.links.join(", "));
        let sensed = self.engine.sensed;
        self.engine = room.engine;
        self.engine.sensed = sensed;
        self.node_ids = room.node_ids;
        self.placed = room.placed;
        self.places = room.places;
        self.notes = room.notes;
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
            if let Some(meter) = &mut self.drift_meter {
                meter.tick(self.newest_board_seconds);
            }
            self.remember_drift();
            let started = Instant::now();
            let previous = self.room_state.take();
            self.room_state = self.engine.tick(self.newest_board_seconds).ok();
            if std::env::var_os("WUJUD_LOG").is_some() {
                println!("engine tick {} ms", started.elapsed().as_millis());
            }
            if self.sense.is_some() && self.engine.sensed.is_none() {
                // Nothing read from the room yet: "learning", not the model's guess.
                self.room_state = None;
            }
            let decided = |state: &Option<RoomState>| state.as_ref().is_some_and(|s| s.room.decision.is_some());
            if decided(&self.room_state) {
                self.last_reading = Some(Instant::now());
            } else if decided(&previous) && self.last_reading.is_some_and(|t| t.elapsed() < HOLD_THROUGH_GAP) {
                self.room_state = previous.map(held_through_gap);
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

    fn drift_reader(&self) -> Option<&RoomSense> {
        self.sense.as_ref().or(self.drift_meter.as_ref())
    }

    fn remember_drift(&mut self) {
        let Some(reading) = self.drift_reader().map(|s| &s.reading) else { return };
        let Some(drift_db) = reading.drift_db else { return };
        let second = DriftSecond { drift_db, moving_above: reading.moving_above };
        if self.drift_history.len() == DRIFT_HISTORY_SECONDS {
            self.drift_history.pop_front();
        }
        self.drift_history.push_back(second);
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

    fn tick_tracker(&mut self) {
        let elapsed = self.last_track.elapsed();
        if elapsed < TRACKER_TICK {
            return;
        }
        self.last_track = Instant::now();
        let fresh = |value: Option<f64>| value.unwrap_or(f64::NAN);
        let motion: Vec<f64> = self.links.iter().map(|l| fresh(l.disturbance_db())).collect();
        let shadowing: Vec<f64> = self.links.iter().map(|l| fresh(l.rssi_spread_db())).collect();
        if self.last_learned.elapsed() >= Duration::from_secs(1) {
            self.last_learned = Instant::now();
            let moving = self.sensed_moving();
            if let Some(learned) = &mut self.learned {
                learned.add_second(&self.links);
                self.learned_answer = learned.answer(moving);
            }
            if let (true, Some((xy, _))) = (moving, &self.learned_answer) {
                self.learned_xy = Some(*xy);
            }
        }
        if motion.iter().any(|m| !m.is_nan()) {
            let mut track = self.tracker.update(&motion, &shadowing, elapsed.as_secs_f64());
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
                    self.engine.add(frame.link, frame.board_seconds, &frame.iq, f64::from(frame.rssi));
                    if let Some(sense) = &mut self.sense {
                        sense.add(frame.link, frame.board_seconds, &frame.iq);
                    }
                    if let Some(meter) = &mut self.drift_meter {
                        meter.add(frame.link, frame.board_seconds, &in_64_bins(&frame.iq));
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

    /// How stirred up a link is, 0..1: the fast disturbance above its quiet level, else the model's
    /// per-second activity.
    pub fn link_activity(&self, link: usize) -> f32 {
        let fast = self
            .shown_track()
            .and_then(|t| t.link_excess.get(link))
            .map(|&e| (e / 2.0).clamp(0.0, 1.0) as f32);
        let slow = self
            .room_state
            .as_ref()
            .and_then(|s| s.link_activity.get(link))
            .map(|&a| a as f32);
        fast.or(slow).unwrap_or(0.0)
    }

    /// What the room looks like now, for a front end to draw; `csi_packets` newest packets per link.
    pub fn snapshot(&self, csi_packets: usize) -> Snapshot {
        let scene = self.engine.scene();
        let boards = scene
            .nodes
            .iter()
            .enumerate()
            .map(|(index, at)| BoardView {
                name: self.node_names.get(index).cloned().unwrap_or_else(|| format!("board {index}")),
                at: at.map(|v| v as f32),
            })
            .collect();
        let links = self
            .engine
            .link_nodes()
            .into_iter()
            .zip(&self.links)
            .enumerate()
            .map(|(index, (ends, history))| {
                LinkView {
                    name: self.link_names.get(index).cloned().unwrap_or_default(),
                    ends,
                    activity: self.link_activity(index),
                    packets_per_second: history.packets_per_second(),
                    rssi: history.latest_rssi(),
                    csi_db: history.newest_amplitudes_db(csi_packets),
                }
            })
            .collect();
        let decision = self.room_state.as_ref().and_then(|s| s.room.decision.as_ref());
        let reading = self.drift_reader().map(|s| &s.reading);
        Snapshot {
            room_size: [scene.width as f32, scene.depth as f32, scene.height as f32],
            boards,
            links,
            furniture: scene.furniture.iter().map(furniture_view).collect(),
            status: self.status(),
            person: self.person(),
            breath_bpm: decision.and_then(|d| d.breath_bpm),
            still_seconds: self.still_seconds(),
            phones: self.phone_distances(),
            drift_history: self.drift_history.iter().copied().collect(),
            moving_above: reading.map(|r| r.moving_above),
            truth_xy: self.truth.as_ref().and_then(|t| t.xy).map(|xy| xy.map(|v| v as f32)),
            one_line: self.one_line,
            placed: self.placed,
            notes: self.notes.clone(),
            source: self.source(),
        }
    }

    fn status(&self) -> RoomStatus {
        let decision = self.room_state.as_ref().and_then(|s| s.room.decision.as_ref());
        let since_motion = self.sense.as_ref().and_then(|s| s.reading.since_motion);
        match decision.map(|d| d.state3) {
            None if self.sense.is_some() => RoomStatus::Waiting,
            None => RoomStatus::Learning,
            Some(RoomState3::Empty) => RoomStatus::Empty { quiet_for: since_motion },
            Some(RoomState3::Still) => RoomStatus::Still { last_moved: since_motion },
            Some(RoomState3::Moving) => RoomStatus::Moving,
        }
    }

    /// Whoever is in the room: where the tracker last put them and the pose the model reads.
    fn person(&self) -> Option<PersonView> {
        let decision = self.room_state.as_ref().and_then(|s| s.room.decision.as_ref());
        let track = self.shown_track();
        let moving = track.is_some_and(|t| t.moving);
        let someone_here = moving || decision.is_some_and(|d| d.state3 != RoomState3::Empty);
        if !someone_here {
            return None;
        }
        let pose = match decision.map(|d| d.state) {
            Some(pose @ (Pose::Moving | Pose::Lie | Pose::Sit)) => pose,
            None if moving => Pose::Moving,
            _ => Pose::Stand,
        };
        Some(PersonView {
            at: track.and_then(|t| t.xy).map(|xy| xy.map(|v| v as f32)),
            pose,
        })
    }
}

pub struct Snapshot {
    /// Width (x), depth (y), height (z) in metres; z is up.
    pub room_size: [f32; 3],
    pub boards: Vec<BoardView>,
    pub links: Vec<LinkView>,
    pub furniture: Vec<FurnitureView>,
    pub status: RoomStatus,
    pub person: Option<PersonView>,
    pub breath_bpm: Option<f64>,
    pub still_seconds: f64,
    pub phones: Vec<PhoneDistance>,
    pub drift_history: Vec<DriftSecond>,
    pub moving_above: Option<f64>,
    /// The simulator only: where its person really is.
    pub truth_xy: Option<[f32; 2]>,
    pub one_line: bool,
    pub placed: bool,
    /// What placing the boards found worth telling.
    pub notes: Vec<String>,
    pub source: Source,
}

impl Snapshot {
    /// The link a moving body disturbs most, when someone moves and it stands out above `at_least`.
    pub fn most_disturbed_link(&self, at_least: f32) -> Option<usize> {
        if self.status != RoomStatus::Moving {
            return None;
        }
        self.links
            .iter()
            .enumerate()
            .filter(|(_, link)| link.activity >= at_least)
            .max_by(|a, b| a.1.activity.total_cmp(&b.1.activity))
            .map(|(index, _)| index)
    }
}

pub struct BoardView {
    pub name: String,
    pub at: [f32; 3],
}

pub struct FurnitureView {
    pub kind: String,
    pub centre: [f32; 3],
    pub size: [f32; 3],
}

impl FurnitureView {
    /// The height of its top, when `[x, y]` is over it.
    pub fn top_under(&self, [x, y]: [f32; 2]) -> Option<f32> {
        let inside = (x - self.centre[0]).abs() <= self.size[0] / 2.0 && (y - self.centre[1]).abs() <= self.size[1] / 2.0;
        inside.then_some(self.centre[2] + self.size[2] / 2.0)
    }
}

fn furniture_view(item: &serde_json::Value) -> FurnitureView {
    let number = |key: &str| item.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let size = [number("w"), number("d"), number("h")];
    FurnitureView {
        kind: item.get("kind").and_then(|k| k.as_str()).unwrap_or("furniture").to_owned(),
        centre: [number("x"), number("y"), number("z") + size[2] / 2.0],
        size,
    }
}

pub struct LinkView {
    pub name: String,
    /// Indices into `Snapshot::boards`.
    pub ends: [usize; 2],
    /// 0 (quiet) .. 1 (strongly disturbed).
    pub activity: f32,
    pub packets_per_second: f64,
    pub rssi: Option<i8>,
    /// Oldest first: each packet's subcarriers (low to high frequency) in dB over the packet's mean.
    pub csi_db: Vec<Vec<f32>>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum RoomStatus {
    /// The simulator's model needs its first seconds.
    Learning,
    /// Live boards: nothing has moved yet.
    Waiting,
    Empty { quiet_for: Option<f64> },
    Still { last_moved: Option<f64> },
    Moving,
}

pub struct PersonView {
    /// Where the tracker puts them; None when the boards' places are unknown or on one line.
    pub at: Option<[f32; 2]>,
    pub pose: Pose,
}

fn on_one_line(links: &[([f64; 2], [f64; 2])]) -> bool {
    let ends: Vec<[f64; 2]> = links.iter().flat_map(|&(a, b)| [a, b]).collect();
    let Some(&first) = ends.first() else {
        return true;
    };
    let Some(&second) = ends.iter().find(|p| (p[0] - first[0]).hypot(p[1] - first[1]) > 0.1) else {
        return true;
    };
    let (dx, dy) = (second[0] - first[0], second[1] - first[1]);
    let length = dx.hypot(dy);
    ends.iter()
        .all(|p| ((p[0] - first[0]) * dy - (p[1] - first[1]) * dx).abs() / length < 0.1)
}

/// The drift detector reads the ESP32's 64-bin layout; the simulator's 56 subcarriers are centred in it.
fn in_64_bins(iq: &[i8]) -> Cow<'_, [i8]> {
    const SIMULATED_BYTES: usize = 112;
    const ESP32_BYTES: usize = 128;
    if iq.len() != SIMULATED_BYTES {
        return Cow::Borrowed(iq);
    }
    let offset = (ESP32_BYTES - SIMULATED_BYTES) / 2;
    let mut bins = vec![0i8; ESP32_BYTES];
    bins[offset..offset + SIMULATED_BYTES].copy_from_slice(iq);
    Cow::Owned(bins)
}

/// The network that places a walking person: every link at once (link_net), or with
/// `WUJUD_TRACKER=points` the earlier one that scores each link on its own (learned_tracker).
enum Locator {
    Links(Box<LinkNetTracker>),
    Points(Box<LearnedTracker>),
}

impl Locator {
    fn load(ends: &[([f64; 2], [f64; 2])], names: &[String], width: f64, depth: f64) -> Option<Self> {
        let models = room_setup::models_dir();
        if std::env::var("WUJUD_TRACKER").as_deref() != Ok("points") {
            match LinkNetTracker::load(&models.join("link_net.json"), ends, names, width, depth) {
                Ok(tracker) => return Some(Locator::Links(Box::new(tracker))),
                Err(error) => eprintln!("link_net.json: {error}"),
            }
        }
        LearnedTracker::load(&models.join("learned_tracker.json"), ends, names, width, depth)
            .map(|tracker| Locator::Points(Box::new(tracker)))
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

    fn answer(&mut self, moving: bool) -> Option<([f64; 2], String)> {
        match self {
            Locator::Links(tracker) => tracker.position(moving).map(|(xy, what)| (xy, what.to_owned())),
            Locator::Points(tracker) => tracker.position().map(|xy| (xy, String::from("?"))),
        }
    }
}

fn held_through_gap(mut state: RoomState) -> RoomState {
    if let Some(decision) = state.room.decision.as_mut() {
        if decision.state3 == RoomState3::Moving {
            decision.state3 = RoomState3::Still;
            decision.state = Pose::Still;
        }
    }
    state
}
