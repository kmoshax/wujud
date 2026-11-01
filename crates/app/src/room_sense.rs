//! Empty, still or moving, read from the room's own links: nothing to install, train or calibrate.
//!
//! Moving: research/tools/drift_detector.py. Each second the packets of a link are averaged into one
//! spectrum shape (dB per subcarrier, its mean removed, so gain steps cancel); drift is the mean change
//! from one second's shape to the next. It is relative, so it carries over to any room, and averaging
//! per second makes it blind to lost packets, which the trained model read as motion: on a real, empty
//! room it said "moving" 100% of the time. The line between still and moving is not a fixed number
//! either: it is a multiple of how quiet this room has been over the last minutes.
//!
//! Empty or someone still: a body standing or sitting changes the multipath, so the shape of the last
//! ten seconds moves away from the empty room's shape. The empty shape is learnt by itself in the
//! first quiet minute after the boards start, and keeps following the room while it stays empty.

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use wujud_model::room::{Pose, RoomState3};

/// A body in the way thins a link out; ten packets still make a usable average.
const MIN_PACKETS_IN_A_SECOND: usize = 10;
/// A second with no usable link keeps the last state this long.
const HOLD_SECONDS: i64 = 8;
const PRESENCE_SECONDS: usize = 6;
const SETTLED_SECONDS: usize = 4;
/// Tuned on a guided recording (research/tools/guided_room.py: away, walk, sit, lie, walk, stand,
/// away; every second labelled). Walking drifts 0.43-0.80 dB/s, anyone still or nobody 0.05-0.10.
/// Moving: the links' median drift over five seconds above three times the room's typical drift of the
/// last five minutes: 100% of the walking, 98% of the stillness.
const MOVING_SECONDS: usize = 5;
const MOVING_TIMES_TYPICAL: f64 = 3.0;
const MOVING_ABOVE_MIN_DB: f64 = 0.12;
/// Smaller movements (sitting down, bowing in prayer): over three seconds, three times the room's
/// quietest fifth. They mean someone is here, not walking.
const STIRRING_SECONDS: usize = 3;
const STIRRING_TIMES_QUIET: f64 = 3.0;
const STIRRING_ABOVE_MIN_DB: f64 = 0.15;
const HISTORY_SECONDS: usize = 300;
const HISTORY_NEEDED: usize = 30;
/// Until there is a history.
const MOVING_ABOVE_START_DB: f64 = 0.25;
const STIRRING_ABOVE_START_DB: f64 = 0.2;
const STILL_BELOW_SHARE: f64 = 0.8;
/// Quiet seconds of ten-second shapes that make the empty room.
pub const LEARN_EMPTY_SECONDS: usize = 45;
pub const TEACH_SETTLE: Duration = Duration::from_secs(8);
pub const TEACH: Duration = Duration::from_secs(20);
pub const POSTURES: [Pose; 3] = [Pose::Stand, Pose::Sit, Pose::Lie];
/// Empty-room wander times this is still empty.
const PRESENCE_MARGIN: f64 = 1.5;
const PRESENCE_MIN: f64 = 0.05;
/// Present when at least this share of the links say so.
const PRESENT_SHARE: f64 = 0.5;
/// While the room is clearly empty its shapes creep (heat, a door left open): each second the empty
/// shape moves this share of the way towards what the link sees now. Never while anyone is seen.
const EMPTY_FOLLOWS: f64 = 0.01;
const FOLLOW_BELOW_THRESHOLD: f64 = 0.6;
/// Whoever moved is taken to be still here this long after their last movement: a person sitting
/// completely still reads like an empty room, and nobody leaves without moving.
const STILL_AFTER_MOTION_SECONDS: f64 = 600.0;
/// Someone arrives or leaves only after the links agree for this many seconds in a row.
const PRESENCE_HOLD_SECONDS: u32 = 2;

/// The empty room last learnt, per link, so a room learnt once is known on the next start.
fn empty_room_file() -> PathBuf {
    if cfg!(test) {
        return std::env::temp_dir().join("wujud-test-empty-room.json");
    }
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".local/share/wujud/empty_room.json")
}

fn remembered_empty_room() -> BTreeMap<String, EmptyLink> {
    std::fs::read_to_string(empty_room_file())
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn remember_empty_room(empty: &BTreeMap<String, EmptyLink>) {
    let file = empty_room_file();
    let saved = file
        .parent()
        .map(std::fs::create_dir_all)
        .transpose()
        .and_then(|_| std::fs::write(&file, serde_json::to_string(empty).unwrap_or_default()));
    if let Err(error) = saved {
        eprintln!("{}: {error}", file.display());
    }
}

/// A link's name without the board it was read from (`/dev/ttyACM0@` or `udp:…@`).
fn link_key(link: &str) -> &str {
    crate::serial::split_link_name(link).1.unwrap_or(link)
}

/// One packet's spectrum shape, as drift_detector.py: dB per signal subcarrier, mean removed.
fn shape_db(iq: &[i8]) -> Option<Vec<f64>> {
    if iq.len() != 128 {
        return None;
    }
    let db: Vec<f64> = (0..64usize)
        .filter(|&i| {
            let bin = i as i64 - 32;
            bin.abs() <= 26 && bin != 0
        })
        .map(|i| {
            let (a, b) = (f64::from(iq[2 * i]), f64::from(iq[2 * i + 1]));
            20.0 * (a.hypot(b) + 1.0).log10()
        })
        .collect();
    let mean = db.iter().sum::<f64>() / db.len() as f64;
    Some(db.into_iter().map(|v| v - mean).collect())
}

fn mean_abs_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len().max(1) as f64
}

fn average(shapes: impl ExactSizeIterator<Item = impl AsRef<[f64]>>) -> Option<Vec<f64>> {
    let count = shapes.len();
    let mut sum: Option<Vec<f64>> = None;
    for shape in shapes {
        let shape = shape.as_ref();
        let total = sum.get_or_insert_with(|| vec![0.0; shape.len()]);
        total.iter_mut().zip(shape).for_each(|(t, v)| *t += v);
    }
    sum.map(|s| s.into_iter().map(|v| v / count as f64).collect())
}

fn median(values: &mut [f64]) -> Option<f64> {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied()
}

#[derive(Default)]
struct LinkSense {
    second: Option<i64>,
    packets: Vec<Vec<f64>>,
    /// Each full second's shape, the last ten seconds of them; a thin second is left out, not the rest.
    shapes: VecDeque<(i64, Vec<f64>)>,
    /// The newest second's drift.
    drift: Option<f64>,
    /// When the newest shape closed: a link that stops sending stops counting.
    fresh_at: Option<Instant>,
}

impl LinkSense {
    fn add(&mut self, t: f64, iq: &[i8]) {
        let Some(shape) = shape_db(iq) else { return };
        let second = t.floor() as i64;
        if self.second != Some(second) {
            self.close();
            self.second = Some(second);
        }
        self.packets.push(shape);
    }

    fn close(&mut self) {
        let packets = std::mem::take(&mut self.packets);
        let Some(second) = self.second else { return };
        if packets.len() < MIN_PACKETS_IN_A_SECOND {
            self.drift = None;
            return;
        }
        let shape = average(packets.iter()).expect("packets");
        self.drift = self
            .shapes
            .back()
            .filter(|(s, _)| *s == second - 1)
            .map(|(_, previous)| mean_abs_difference(previous, &shape));
        while self.shapes.front().is_some_and(|(s, _)| *s <= second - PRESENCE_SECONDS as i64) {
            self.shapes.pop_front();
        }
        self.shapes.push_back((second, shape));
        self.fresh_at = Some(Instant::now());
    }

    fn fresh(&self) -> bool {
        self.fresh_at.is_some_and(|t| t.elapsed() < Duration::from_secs(3))
    }

    fn drift(&self) -> Option<f64> {
        self.drift.filter(|_| self.fresh())
    }

    /// The last ten seconds' shape, once most of them are there.
    fn settled_shape(&self) -> Option<Vec<f64>> {
        (self.fresh() && self.shapes.len() >= SETTLED_SECONDS).then(|| average(self.shapes.iter().map(|(_, s)| s)))?
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct EmptyLink {
    shape: Vec<f64>,
    present_above: f64,
}

fn posture_name(pose: Pose) -> &'static str {
    match pose {
        Pose::Lie => "lie",
        Pose::Sit => "sit",
        _ => "stand",
    }
}

struct Teaching {
    pose: Pose,
    started: Instant,
    shapes: BTreeMap<String, Vec<Vec<f64>>>,
}

pub struct RoomSense {
    names: Vec<String>,
    links: Vec<LinkSense>,
    /// Each link's empty-room shape, once learnt.
    empty: BTreeMap<String, EmptyLink>,
    /// Ten-second shapes gathered towards the empty room, while it is being learnt.
    learning: Option<BTreeMap<String, Vec<Vec<f64>>>>,
    learnt_quiet_seconds: usize,
    postures: BTreeMap<String, BTreeMap<String, Vec<f64>>>,
    teaching: Option<Teaching>,
    history: VecDeque<f64>,
    moving: bool,
    stirring: bool,
    present: bool,
    present_votes: u32,
    last_motion: Option<i64>,
    last_second: Option<i64>,
    last_fresh_second: Option<i64>,
    pub reading: Reading,
}

/// What the side panel shows of it.
#[derive(Default, Clone)]
pub struct Reading {
    pub state: Option<RoomState3>,
    pub drift_db: Option<f64>,
    /// Change above this is movement, as this room sets it now.
    pub moving_above: f64,
    /// Small movements: someone is here, not walking.
    pub stirring: bool,
    /// Seconds since anyone last moved, once someone has.
    pub since_motion: Option<f64>,
    /// Share of links that see someone, once the empty room is learnt.
    pub presence: Option<f64>,
    /// Learning the empty room: quiet seconds so far, out of LEARN_EMPTY_SECONDS.
    pub learning: Option<usize>,
    /// While someone is still: the taught posture this room's links look most like.
    pub posture: Option<Pose>,
}

impl RoomSense {
    pub fn new(link_names: &[String]) -> Self {
        let names: Vec<String> = link_names.iter().map(|l| link_key(l).to_owned()).collect();
        // Nothing is learnt about the room by itself any more: someone sitting at the laptop was learnt
        // as the empty room and then never seen. Empty-vs-still only comes from the button.
        RoomSense {
            links: link_names.iter().map(|_| LinkSense::default()).collect(),
            learning: None,
            names,
            empty: BTreeMap::new(),
            learnt_quiet_seconds: 0,
            postures: BTreeMap::new(),
            teaching: None,
            history: VecDeque::new(),
            moving: false,
            stirring: false,
            present: false,
            present_votes: 0,
            last_motion: None,
            last_second: None,
            last_fresh_second: None,
            reading: Reading::default(),
        }
    }

    /// A room that changed links (a board joined) keeps what it learnt about the links it had.
    pub fn carry_over(mut self, previous: &RoomSense) -> Self {
        self.empty = previous.empty.clone();
        self.postures = previous.postures.clone();
        self.history = previous.history.clone();
        self.last_motion = previous.last_motion;
        self.moving = previous.moving;
        self.stirring = previous.stirring;
        self.present = previous.present;
        self.reading = previous.reading.clone();
        if !self.empty.is_empty() {
            self.learning = None;
        }
        self
    }

    pub fn add(&mut self, link: usize, t: f64, iq: &[i8]) {
        if let Some(sense) = self.links.get_mut(link) {
            sense.add(t, iq);
        }
    }

    /// "The room is empty now": learn it again from the next quiet seconds.
    pub fn learn_empty_room(&mut self) {
        self.learning = Some(BTreeMap::new());
        self.learnt_quiet_seconds = 0;
    }

    /// The room's drift at a percentile of the last five minutes.
    fn typical(&self, percentile: usize) -> Option<f64> {
        if self.history.len() < HISTORY_NEEDED {
            return None;
        }
        let mut history: Vec<f64> = self.history.iter().copied().collect();
        history.sort_by(f64::total_cmp);
        Some(history[history.len() * percentile / 100])
    }

    fn recent(&self, seconds: usize) -> Option<f64> {
        let mut last: Vec<f64> = self.history.iter().rev().take(seconds).copied().collect();
        median(&mut last)
    }

    /// Once a second: the room's state, or None while it cannot tell yet.
    pub fn tick(&mut self, now: f64) -> Option<RoomState3> {
        let second = now.floor() as i64;
        if self.last_second == Some(second) {
            return self.reading.state;
        }
        self.last_second = Some(second);
        let mut drifts: Vec<f64> = self.links.iter().filter_map(LinkSense::drift).collect();
        let drift = median(&mut drifts);
        if let Some(d) = drift {
            if self.history.len() == HISTORY_SECONDS {
                self.history.pop_front();
            }
            self.history.push_back(d);
        }
        let moving_above = self
            .typical(50)
            .map_or(MOVING_ABOVE_START_DB, |t| (t * MOVING_TIMES_TYPICAL).max(MOVING_ABOVE_MIN_DB));
        let stirring_above = self
            .typical(20)
            .map_or(STIRRING_ABOVE_START_DB, |t| (t * STIRRING_TIMES_QUIET).max(STIRRING_ABOVE_MIN_DB));
        let fresh = drift.is_some();
        if fresh {
            self.last_fresh_second = Some(second);
        } else if self.last_fresh_second.is_some_and(|s| second - s <= HOLD_SECONDS) {
            return self.reading.state;
        }
        let over = |level: Option<f64>, above: f64, was: bool| match level {
            Some(l) if fresh => l > above || (was && l > above * STILL_BELOW_SHARE),
            _ => false,
        };
        self.moving = over(self.recent(MOVING_SECONDS), moving_above, self.moving);
        self.stirring = !self.moving && over(self.recent(STIRRING_SECONDS), stirring_above, self.stirring);
        if self.moving || self.stirring {
            self.last_motion = Some(second);
        }
        let since_motion = self.last_motion.map(|m| (second - m) as f64);
        self.keep_learning();
        let presence = self.presence();
        if let Some(share) = presence {
            let says_present = share >= PRESENT_SHARE;
            self.present_votes = if says_present == self.present { 0 } else { self.present_votes + 1 };
            if self.present_votes >= PRESENCE_HOLD_SECONDS {
                self.present = says_present;
                self.present_votes = 0;
            }
        }
        if !self.moving && !self.stirring && !self.present && presence == Some(0.0) {
            self.follow_empty_room();
        }
        let learnt = !self.empty.is_empty();
        let recently_moved = since_motion.is_some_and(|s| s <= STILL_AFTER_MOTION_SECONDS);
        let mut state = match (fresh, self.moving, self.stirring) {
            (false, _, _) => None,
            (_, true, _) => Some(RoomState3::Moving),
            (_, false, true) => Some(RoomState3::Still),
            _ if self.present || recently_moved => Some(RoomState3::Still),
            _ if learnt || since_motion.is_some() => Some(RoomState3::Empty),
            // Nobody has moved yet and the empty room is not known: nothing to say.
            _ => None,
        };
        if self.keep_teaching().is_some() {
            state = Some(RoomState3::Still);
        }
        self.reading = Reading {
            state,
            drift_db: self.recent(MOVING_SECONDS),
            moving_above,
            stirring: self.stirring,
            presence,
            since_motion,
            learning: self.learning.as_ref().map(|_| self.learnt_quiet_seconds),
            posture: (state == Some(RoomState3::Still)).then(|| self.posture()).flatten(),
        };
        state
    }

    /// Quiet seconds go towards the empty room; any movement starts the count again, so the room is
    /// learnt from the first stretch of 45 s in which nothing moved at all.
    fn keep_learning(&mut self) {
        if self.learning.is_none() {
            return;
        }
        if self.moving || self.stirring {
            self.learning = Some(BTreeMap::new());
            self.learnt_quiet_seconds = 0;
            return;
        }
        let shapes: Vec<(String, Vec<f64>)> = self
            .names
            .iter()
            .zip(&self.links)
            .filter_map(|(name, sense)| Some((name.clone(), sense.settled_shape()?)))
            .collect();
        if shapes.is_empty() {
            return;
        }
        let learning = self.learning.as_mut().expect("learning");
        for (name, shape) in shapes {
            learning.entry(name).or_default().push(shape);
        }
        self.learnt_quiet_seconds += 1;
        if self.learnt_quiet_seconds < LEARN_EMPTY_SECONDS {
            return;
        }
        let learnt = finish(self.learning.take().expect("learning"));
        if learnt.is_empty() {
            self.learn_empty_room();
        } else {
            println!("learnt the empty room on {} links", learnt.len());
            let mut remembered = remembered_empty_room();
            remembered.extend(learnt.iter().map(|(k, v)| (k.clone(), v.clone())));
            remember_empty_room(&remembered);
            self.empty = learnt;
            self.present = false;
        }
    }

    /// Each link's shift from empty right now, for the links that are learnt and settled.
    fn shifts(&self) -> BTreeMap<String, Vec<f64>> {
        self.names
            .iter()
            .zip(&self.links)
            .filter_map(|(name, sense)| {
                let empty = self.empty.get(name)?;
                let shape = sense.settled_shape()?;
                Some((name.clone(), shape.iter().zip(&empty.shape).map(|(s, e)| s - e).collect()))
            })
            .collect()
    }

    /// The nearest taught posture, over the links both know.
    fn posture(&self) -> Option<Pose> {
        let now = self.shifts();
        POSTURES
            .into_iter()
            .filter_map(|pose| {
                let taught = self.postures.get(posture_name(pose))?;
                let distances: Vec<f64> = now
                    .iter()
                    .filter_map(|(link, shift)| Some(mean_abs_difference(shift, taught.get(link)?)))
                    .collect();
                (!distances.is_empty()).then(|| (pose, distances.iter().sum::<f64>() / distances.len() as f64))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(pose, _)| pose)
    }

    /// Collects the posture being taught; keeps it when the time is up.
    fn keep_teaching(&mut self) -> Option<(Pose, String)> {
        let shifts = self.shifts();
        let teaching = self.teaching.as_mut()?;
        let elapsed = teaching.started.elapsed();
        if elapsed < TEACH_SETTLE {
            let left = (TEACH_SETTLE - elapsed).as_secs();
            return Some((teaching.pose, format!("get into position… {left} s")));
        }
        if elapsed < TEACH_SETTLE + TEACH {
            for (link, shift) in shifts {
                teaching.shapes.entry(link).or_default().push(shift);
            }
            let left = (TEACH_SETTLE + TEACH - elapsed).as_secs();
            return Some((teaching.pose, format!("hold still… {left} s")));
        }
        let teaching = self.teaching.take()?;
        let learnt: BTreeMap<String, Vec<f64>> = teaching
            .shapes
            .into_iter()
            .filter_map(|(link, shifts)| Some((link, average(shifts.iter())?)))
            .collect();
        if !learnt.is_empty() {
            self.postures.insert(posture_name(teaching.pose).to_owned(), learnt);
        }
        None
    }

    fn presence(&self) -> Option<f64> {
        let votes: Vec<bool> = self
            .names
            .iter()
            .zip(&self.links)
            .filter_map(|(name, sense)| {
                let empty = self.empty.get(name)?;
                let shape = sense.settled_shape()?;
                Some(mean_abs_difference(&shape, &empty.shape) > empty.present_above)
            })
            .collect();
        (!votes.is_empty()).then(|| votes.iter().filter(|&&v| v).count() as f64 / votes.len() as f64)
    }

    fn follow_empty_room(&mut self) {
        for (name, sense) in self.names.iter().zip(&self.links) {
            let (Some(empty), Some(shape)) = (self.empty.get_mut(name), sense.settled_shape()) else {
                continue;
            };
            if mean_abs_difference(&shape, &empty.shape) < empty.present_above * FOLLOW_BELOW_THRESHOLD {
                for (e, now) in empty.shape.iter_mut().zip(&shape) {
                    *e += EMPTY_FOLLOWS * (now - *e);
                }
            }
        }
    }
}

/// Each link's empty shape is the mean of its ten-second shapes; it is "someone here" once the
/// current ten seconds sit further from it than any empty ten seconds did, with a margin.
fn finish(shapes: BTreeMap<String, Vec<Vec<f64>>>) -> BTreeMap<String, EmptyLink> {
    shapes
        .into_iter()
        .filter(|(_, shapes)| shapes.len() >= LEARN_EMPTY_SECONDS / 2)
        .filter_map(|(name, shapes)| {
            let shape = average(shapes.iter())?;
            let wander = shapes
                .iter()
                .map(|s| mean_abs_difference(s, &shape))
                .fold(0.0, f64::max);
            Some((
                name,
                EmptyLink {
                    shape,
                    present_above: (wander * PRESENCE_MARGIN).max(PRESENCE_MIN),
                },
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(level: i8, tilt: f64) -> Vec<i8> {
        (0..64)
            .flat_map(|k| [(f64::from(level) + tilt * f64::from(k)) as i8, 0])
            .collect()
    }

    fn feed(sense: &mut RoomSense, seconds: std::ops::Range<i64>, tilt: impl Fn(i64) -> f64) {
        for second in seconds {
            for p in 0..50 {
                sense.add(0, second as f64 + p as f64 / 50.0, &packet(20, tilt(second)));
            }
        }
    }

    #[test]
    fn a_steady_link_is_not_moving_and_a_changing_one_is() {
        let mut sense = RoomSense::new(&["udp:1@aa:aa:aa:aa:aa:01".to_owned()]);
        feed(&mut sense, 0..6, |_| 0.0);
        sense.tick(6.0);
        assert!(!sense.moving);
        feed(&mut sense, 6..12, |s| if s % 2 == 0 { 0.3 } else { -0.3 });
        sense.tick(12.0);
        assert!(sense.moving);
    }

    #[test]
    fn shape_ignores_overall_gain() {
        let quiet = shape_db(&packet(10, 0.2)).unwrap();
        let loud = shape_db(&packet(40, 0.8)).unwrap();
        assert_eq!(quiet.len(), 52);
        assert!(mean_abs_difference(&quiet, &loud) < 0.5);
    }

    /// `WUJUD_GUIDED=research/rec/guided_room_<time> cargo test --release -p wujud guided -- --ignored --nocapture`:
    /// the recording through this detector, second by second against what the person was doing.
    #[test]
    #[ignore]
    fn guided_recording() {
        let Some(base) = std::env::var_os("WUJUD_GUIDED") else { return };
        let base = base.to_string_lossy().into_owned();
        let plan: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(format!("{base}_plan.json")).unwrap()).unwrap();
        let text = std::fs::read_to_string(format!("{base}.csv")).unwrap();
        // As on the day: the view starts, then the person walks out.
        let started = if std::env::var_os("WUJUD_GUIDED_FROM_FILE_START").is_some() {
            0.0
        } else {
            plan[0]["start"].as_f64().unwrap() - 30.0
        };
        let mut names: Vec<String> = Vec::new();
        let mut rows = Vec::new();
        for line in text.lines() {
            let parts: Vec<&str> = line.splitn(6, ',').collect();
            if parts.len() < 6 || parts[2] != "P" || parts[5].len() != 256 {
                continue;
            }
            let link = parts[1].to_owned();
            let index = names.iter().position(|n| *n == link).unwrap_or_else(|| {
                names.push(link);
                names.len() - 1
            });
            let iq: Vec<i8> = (0..128).map(|i| u8::from_str_radix(&parts[5][2 * i..2 * i + 2], 16).unwrap() as i8).collect();
            let t = parts[0].parse::<f64>().unwrap();
            if t >= started {
                rows.push((t, index, iq));
            }
        }
        let mut sense = RoomSense::new(&names);
        let mut room = crate::room_setup::room_engine(&names, None, &BTreeMap::new()).unwrap();
        let order: Vec<Option<usize>> = names.iter().map(|n| room.links.iter().position(|l| l == n)).collect();
        let mut said: Vec<(f64, Option<RoomState3>)> = Vec::new();
        let mut strength: Vec<(f64, f64, bool)> = Vec::new();
        let mut last = None;
        for (t, link, iq) in rows {
            sense.add(link, t, &iq);
            if let Some(engine_link) = order[link] {
                room.engine.add(engine_link, t, &iq, -60.0);
            }
            if last != Some(t.floor()) {
                last = Some(t.floor());
                said.push((t, sense.tick(t)));
                room.engine.sensed = sense.reading.state;
                let _ = room.engine.tick(t);
                if let Some(g) = room.engine.breathing_strength {
                    strength.push((t, g, room.engine.breathing_seen()));
                }
            }
        }
        for step in &plan {
            let (start, end) = (step["start"].as_f64().unwrap() + 5.0, step["end"].as_f64().unwrap());
            let mut g: Vec<f64> = strength.iter().filter(|(t, _, _)| *t >= start && *t < end).map(|(_, g, _)| *g).collect();
            let seen = strength.iter().filter(|(t, _, s)| *t >= start && *t < end && *s).count();
            g.sort_by(f64::total_cmp);
            if !g.is_empty() {
                println!(
                    "breathing {:34} glrt p10 {:.2} median {:.2} p90 {:.2}  breathing seen {seen}/{} s",
                    step["english"].as_str().unwrap(),
                    g[g.len() / 10],
                    g[g.len() / 2],
                    g[g.len() * 9 / 10],
                    g.len()
                );
            }
        }
        for step in &plan {
            let (start, end) = (step["start"].as_f64().unwrap() + 5.0, step["end"].as_f64().unwrap());
            let label = step["label"].as_str().unwrap();
            let got: Vec<&str> = said
                .iter()
                .filter(|(t, _)| *t >= start && *t < end)
                .map(|(_, s)| match s {
                    Some(RoomState3::Moving) => "moving",
                    Some(RoomState3::Still) => "still",
                    Some(RoomState3::Empty) => "empty",
                    None => "learning",
                })
                .collect();
            let right = got.iter().filter(|&&g| g == label).count();
            let mut counts = BTreeMap::new();
            for g in &got {
                *counts.entry(*g).or_insert(0) += 1;
            }
            println!(
                "{:34} {label:6} {:5.1}% {counts:?}",
                step["english"].as_str().unwrap(),
                100.0 * right as f64 / got.len().max(1) as f64
            );
        }
    }
}
