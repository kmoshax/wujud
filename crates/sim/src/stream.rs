//! Second-by-second simulation for live demos (research/live.py demo-room / demo-hall): each call to
//! `next()` simulates one more second of every link, so a dashboard can start immediately.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::capture::Capture;
use crate::hall::{random_hall, Hall, HallOptions, NodePlacement};
use crate::hall_sim::{ArrivalOptions, HallSession};
use crate::random::seeded;
use crate::room::{random_room, Activity, PlanStep, Room, RoomOptions};
use crate::room_sim::{RoomSession, RoomSimOptions};

/// live.py demo-room: away, walk, sit, walk, lie (breathing), walk, away.
pub fn demo_room_plan() -> Vec<PlanStep> {
    [
        (Activity::Away, 25.0),
        (Activity::Walk, 15.0),
        (Activity::Sit, 50.0),
        (Activity::Walk, 10.0),
        (Activity::Lie, 80.0),
        (Activity::Walk, 12.0),
        (Activity::Away, 25.0),
    ]
    .map(|(activity, seconds)| PlanStep { activity, seconds })
    .to_vec()
}

fn frames_of_second(second: usize, fs: f64, total: usize) -> Range<usize> {
    let start = ((second as f64 * fs) as usize).min(total);
    let end = (((second + 1) as f64 * fs) as usize).min(total);
    start..end
}

/// Ground truth at the middle of a streamed second.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoomTruth {
    pub activity: Activity,
    /// Breaths per minute while lying, sitting or standing still.
    pub breath_bpm: Option<f64>,
    pub xy: Option<[f64; 2]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoomSecond {
    pub second: usize,
    /// One capture per `room.links`.
    pub links: Vec<Capture>,
    pub truth: RoomTruth,
}

pub struct RoomStream {
    session: RoomSession,
    fs: f64,
    next_second: usize,
}

impl RoomStream {
    pub fn new(room: &Room, options: &RoomSimOptions, seed: u64) -> Self {
        let session = RoomSession::new(room, options, &mut seeded(seed, 1));
        RoomStream {
            session,
            fs: options.fs,
            next_second: 0,
        }
    }

    /// live.py demo-room: smart placement, `boards` ESP32s, 50 packets/s, `demo_room_plan()`.
    pub fn demo(seed: u64, boards: usize) -> Self {
        let room = random_room(
            &mut seeded(seed, 0),
            &RoomOptions {
                boards,
                mesh: false,
                smart_placement: true,
            },
        );
        let plan = demo_room_plan();
        let seconds = plan.iter().map(|step| step.seconds).sum();
        let options = RoomSimOptions {
            seconds,
            fs: 50.0,
            plan: Some(plan),
            ..RoomSimOptions::default()
        };
        RoomStream::new(&room, &options, seed)
    }

    pub fn room(&self) -> &Room {
        self.session.room()
    }

    pub fn seconds(&self) -> usize {
        (self.session.frames() as f64 / self.fs).ceil() as usize
    }

    /// Empty-room packets of every link (the calibration a real install records first).
    pub fn calibration(&mut self) -> Vec<Capture> {
        self.session.calibrate()
    }

    fn truth_at(&self, frame: usize) -> RoomTruth {
        let labels = self.session.labels();
        let frame = frame.min(labels.activity.len().saturating_sub(1));
        let activity = labels.activity[frame];
        let breathing_visible = matches!(activity, Activity::Lie | Activity::Sit | Activity::Stand);
        let xy = labels.xy[frame];
        RoomTruth {
            activity,
            breath_bpm: breathing_visible.then_some(labels.breath_hz * 60.0),
            xy: xy[0].is_finite().then_some(xy),
        }
    }
}

impl Iterator for RoomStream {
    type Item = RoomSecond;

    fn next(&mut self) -> Option<RoomSecond> {
        let frames = frames_of_second(self.next_second, self.fs, self.session.frames());
        if frames.is_empty() {
            return None;
        }
        let second = self.next_second;
        self.next_second += 1;
        let links = self.session.capture(frames);
        let truth = self.truth_at(((second as f64 + 0.5) * self.fs) as usize);
        Some(RoomSecond {
            second,
            links,
            truth,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HallSecond {
    pub second: usize,
    /// One capture per direction, in `HallStream::directions()` order.
    pub links: Vec<Capture>,
    /// Which seats hold a student at the start of this second.
    pub present: Vec<bool>,
}

pub struct HallStream {
    session: HallSession,
    next_second: usize,
}

impl HallStream {
    pub fn new(hall: &Hall, options: &ArrivalOptions, seed: u64) -> Self {
        let session = HallSession::arrivals(hall, options, &mut seeded(seed, 1));
        HallStream {
            session,
            next_second: 0,
        }
    }

    /// live.py demo-hall: a 13 × 12 m tiered hall, 12 wall boards, a 4-minute lecture filling to 75 %.
    pub fn demo(seed: u64) -> Self {
        let options = HallOptions {
            nodes: 12,
            tiered: Some(true),
            placement: Some(NodePlacement::Walls),
            width: Some(13.0),
            depth: Some(12.0),
        };
        let hall = random_hall(&mut seeded(seed, 0), &options);
        HallStream::new(&hall, &ArrivalOptions::default(), seed)
    }

    pub fn hall(&self) -> &Hall {
        self.session.hall()
    }

    /// `(tx, rx)` of each capture in `HallSecond::links` and `calibration()`.
    pub fn directions(&self) -> Vec<(usize, usize)> {
        self.session.directions()
    }

    pub fn seconds(&self) -> usize {
        (self.session.frames() as f64 / self.session.fs()).ceil() as usize
    }

    /// Empty-hall packets of every direction.
    pub fn calibration(&mut self) -> Vec<Capture> {
        self.session.calibrate()
    }
}

impl Iterator for HallStream {
    type Item = HallSecond;

    fn next(&mut self) -> Option<HallSecond> {
        let fs = self.session.fs();
        let frames = frames_of_second(self.next_second, fs, self.session.frames());
        if frames.is_empty() {
            return None;
        }
        let second = self.next_second;
        self.next_second += 1;
        let present = self.session.present_at(frames.start);
        let links = self.session.capture(frames);
        Some(HallSecond {
            second,
            links,
            present,
        })
    }
}
