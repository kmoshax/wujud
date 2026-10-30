//! The simulated rooms of the study: per room, one minute of each activity and a "day" of entering,
//! staying still and leaving; each reduced to window features right away.

use wujud_sim::random::{pick, seeded, uniform, SimRng};
use wujud_sim::room::{random_room, Activity, PlanStep, RealismKnobs, Room, RoomOptions};
use wujud_sim::room_sim::{simulate_room_on_band, RoomSimOptions, RoomSimulation};

use crate::configs::StudyConfig;
use crate::windows::{link_windows, LinkWindows};

/// The calibrated realism of research/results/sim_calibrate.json (`chosen_knobs`).
pub const CALIBRATED_KNOBS: RealismKnobs = RealismKnobs {
    rcs_scale: 0.1,
    breath_scale: 0.5,
    evm_shift_db: 6.0,
};

pub const SESSION_SECONDS: f64 = 60.0;
const SESSION_ACTIVITIES: [Activity; 5] = [
    Activity::Away,
    Activity::Lie,
    Activity::Sit,
    Activity::Stand,
    Activity::Walk,
];

pub const EMPTY: i64 = 0;
pub const STILL: i64 = 1;
pub const MOTION: i64 = 2;

pub fn room_class(activity: Activity) -> i64 {
    match activity {
        Activity::Away => EMPTY,
        Activity::Walk => MOTION,
        _ => STILL,
    }
}

/// One activity session: per-link windows, its class and the true breathing rate.
pub struct SessionWindows {
    pub class: i64,
    pub links: Vec<LinkWindows>,
    pub breath_bpm: f64,
}

/// A day in the room: per-link windows plus, per window, whether it is occupied and whether anyone walked.
pub struct DayWindows {
    pub links: Vec<LinkWindows>,
    pub occupied: Vec<bool>,
    pub walking: Vec<bool>,
}

pub struct StudyRoom {
    pub sessions: Vec<SessionWindows>,
    pub day: Option<DayWindows>,
}

fn room_and_harshness(config: &StudyConfig, seed: u64) -> (Room, f64) {
    let mut rng = seeded(seed, 0);
    let room = random_room(
        &mut rng,
        &RoomOptions {
            boards: config.boards,
            mesh: config.mesh,
            smart_placement: config.smart_placement,
        },
    );
    (room, uniform(&mut rng, 0.8, 1.6))
}

fn simulate(
    config: &StudyConfig,
    room: &Room,
    harsh: f64,
    plan: Vec<PlanStep>,
    rng: &mut SimRng,
) -> RoomSimulation {
    let options = RoomSimOptions {
        seconds: plan.iter().map(|step| step.seconds).sum(),
        fs: config.packets_per_second,
        calibration_seconds: 0.0,
        harsh,
        plan: Some(plan),
        knobs: CALIBRATED_KNOBS,
        ..RoomSimOptions::default()
    };
    simulate_room_on_band(room, &options, &config.band, rng)
}

fn all_link_windows(config: &StudyConfig, simulation: &RoomSimulation) -> Vec<LinkWindows> {
    let channels = config.band.channel_count();
    simulation
        .links
        .iter()
        .map(|link| link_windows(&link.session, channels))
        .collect()
}

fn session(
    config: &StudyConfig,
    room: &Room,
    harsh: f64,
    seed: u64,
    index: usize,
) -> SessionWindows {
    let activity = SESSION_ACTIVITIES[index];
    let plan = vec![PlanStep {
        activity,
        seconds: SESSION_SECONDS,
    }];
    let simulation = simulate(
        config,
        room,
        harsh,
        plan,
        &mut seeded(seed, 1 + index as u64),
    );
    SessionWindows {
        class: room_class(activity),
        links: all_link_windows(config, &simulation),
        breath_bpm: simulation.labels.breath_hz * 60.0,
    }
}

/// research/bench/occupancy_sim.py::plan: away, then walk in / stay still / walk out / away, ~10 minutes.
fn day_plan(rng: &mut SimRng) -> Vec<PlanStep> {
    let step = |activity, seconds| PlanStep { activity, seconds };
    let mut plan = vec![step(Activity::Away, uniform(rng, 90.0, 150.0))];
    while plan.iter().map(|s| s.seconds).sum::<f64>() < 560.0 {
        let still = pick(rng, &[Activity::Sit, Activity::Lie, Activity::Stand]);
        plan.push(step(Activity::Walk, uniform(rng, 8.0, 15.0)));
        plan.push(step(still, uniform(rng, 60.0, 180.0)));
        plan.push(step(Activity::Walk, uniform(rng, 8.0, 15.0)));
        plan.push(step(Activity::Away, uniform(rng, 60.0, 150.0)));
    }
    plan
}

fn day(config: &StudyConfig, room: &Room, harsh: f64, seed: u64) -> DayWindows {
    let plan = day_plan(&mut seeded(seed, 10));
    let simulation = simulate(config, room, harsh, plan, &mut seeded(seed, 11));
    let links = all_link_windows(config, &simulation);
    let windows = links.iter().map(|l| l.rows.len()).min().unwrap_or(0);
    let (occupied, walking) = window_truth(
        &simulation.labels.activity,
        config.packets_per_second,
        windows,
    );
    DayWindows {
        links,
        occupied,
        walking,
    }
}

/// Window `k` covers `[5k, 5k + 20)` s: occupied when most of its frames have someone in, walking if any frame does.
fn window_truth(activity: &[Activity], fs: f64, windows: usize) -> (Vec<bool>, Vec<bool>) {
    use wujud_model::features::{HOP_SECONDS, WINDOW_SECONDS};
    (0..windows)
        .map(|k| {
            let first = (k as f64 * HOP_SECONDS * fs).ceil() as usize;
            let end = (((k as f64 * HOP_SECONDS + WINDOW_SECONDS) * fs).ceil() as usize)
                .min(activity.len());
            let frames = &activity[first.min(end)..end];
            let away = frames.iter().filter(|&&a| a == Activity::Away).count();
            let occupied = away * 2 <= frames.len();
            let walking = frames.contains(&Activity::Walk);
            (occupied, walking)
        })
        .unzip()
}

pub fn study_room(config: &StudyConfig, seed: u64, with_day: bool) -> StudyRoom {
    let (room, harsh) = room_and_harshness(config, seed);
    let sessions = (0..SESSION_ACTIVITIES.len())
        .map(|index| session(config, &room, harsh, seed, index))
        .collect();
    StudyRoom {
        sessions,
        day: with_day.then(|| day(config, &room, harsh, seed)),
    }
}
