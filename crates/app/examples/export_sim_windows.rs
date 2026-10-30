//! Exports raw single-link room sessions (router -> one board) with domain-randomised realism for the
//! deep-learning study in research/deep/. One chunk per worker: `<out>/chunk_<c>.{iq,t,rssi}.npy` + `.json`.
//! cargo run --release -p hallsense --example export_sim_windows -- <out_dir> <first_room> <rooms> [seconds]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use hallsense_sim::capture::Capture;
use hallsense_sim::distractors::DistractorKind;
use hallsense_sim::random::{chance, seeded, uniform, SimRng};
use hallsense_sim::room::{random_room, Activity, PlanStep, RealismKnobs, RoomOptions};
use hallsense_sim::room_sim::{simulate_room, RoomSimOptions};
use serde_json::json;

const WORKERS: usize = 12;
const ACTIVITIES: [Activity; 5] = [
    Activity::Away,
    Activity::Lie,
    Activity::Sit,
    Activity::Stand,
    Activity::Walk,
];
const DISTRACTORS: [DistractorKind; 4] = [
    DistractorKind::Fan,
    DistractorKind::Pet,
    DistractorKind::Vacuum,
    DistractorKind::Door,
];

struct SessionDraw {
    fs: f64,
    harsh: f64,
    knobs: RealismKnobs,
    distractors: Vec<DistractorKind>,
}

/// Spans the original simulator (1, 1, 0 dB) to the calibrated one (0.1, 0.5, +6 dB) of results/sim_calibrate.json.
fn randomised_knobs(rng: &mut SimRng) -> RealismKnobs {
    RealismKnobs {
        rcs_scale: 10f64.powf(uniform(rng, -1.0, 0.0)),
        breath_scale: uniform(rng, 0.5, 1.0),
        evm_shift_db: uniform(rng, 0.0, 6.0),
    }
}

fn draw_session(rng: &mut SimRng) -> SessionDraw {
    let distractors = if chance(rng, 0.2) {
        vec![DISTRACTORS[(uniform(rng, 0.0, 4.0) as usize).min(3)]]
    } else {
        Vec::new()
    };
    SessionDraw {
        fs: uniform(rng, 50.0, 120.0),
        harsh: uniform(rng, 0.8, 1.6),
        knobs: randomised_knobs(rng),
        distractors,
    }
}

fn activity_name(activity: Activity) -> &'static str {
    match activity {
        Activity::Away => "away",
        Activity::Lie => "lie",
        Activity::Sit => "sit",
        Activity::Stand => "stand",
        Activity::Walk => "walk",
    }
}

fn distractor_name(kind: DistractorKind) -> &'static str {
    match kind {
        DistractorKind::Fan => "fan",
        DistractorKind::Pet => "pet",
        DistractorKind::Vacuum => "vacuum",
        DistractorKind::Door => "door",
    }
}

fn simulate_room_sessions(room_seed: u64, seconds: f64) -> Vec<(serde_json::Value, Capture)> {
    let mut rng = seeded(room_seed, 0);
    let smart_placement = chance(&mut rng, 0.3);
    let room = random_room(
        &mut rng,
        &RoomOptions {
            boards: 1,
            mesh: false,
            smart_placement,
        },
    );
    ACTIVITIES
        .iter()
        .enumerate()
        .map(|(index, &activity)| {
            let mut session_rng = seeded(room_seed, 1 + index as u64);
            let draw = draw_session(&mut session_rng);
            let options = RoomSimOptions {
                seconds,
                fs: draw.fs,
                calibration_seconds: 0.0,
                harsh: draw.harsh,
                plan: Some(vec![PlanStep { activity, seconds }]),
                distractors: draw.distractors.clone(),
                wall_db: 0.0,
                outside: false,
                knobs: draw.knobs,
            };
            let simulation = simulate_room(&room, &options, &mut session_rng);
            let capture = simulation.links[0].session.clone();
            let meta = json!({
                "room": room_seed,
                "activity": activity_name(activity),
                "fs": draw.fs,
                "harsh": draw.harsh,
                "rcs_scale": draw.knobs.rcs_scale,
                "breath_scale": draw.knobs.breath_scale,
                "evm_shift_db": draw.knobs.evm_shift_db,
                "distractor": draw.distractors.first().map(|&k| distractor_name(k)),
                "packets": capture.len(),
            });
            (meta, capture)
        })
        .collect()
}

fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_text = match shape {
        [n] => format!("({n},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut dict =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}");
    let unpadded = 10 + dict.len() + 1;
    dict.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    dict.push('\n');
    let mut header = b"\x93NUMPY\x01\x00".to_vec();
    header.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    header.extend_from_slice(dict.as_bytes());
    header
}

fn write_npy(path: &Path, descr: &str, shape: &[usize], bytes: &[u8]) -> anyhow::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    file.write_all(&npy_header(descr, shape))?;
    file.write_all(bytes)?;
    Ok(())
}

fn write_chunk(stem: &Path, sessions: &[(serde_json::Value, Capture)]) -> anyhow::Result<()> {
    let packets: usize = sessions.iter().map(|(_, c)| c.len()).sum();
    let subcarriers = sessions.first().map_or(0, |(_, c)| c.iq.shape()[1]);
    let mut iq = Vec::with_capacity(packets * subcarriers * 2);
    let mut times = Vec::with_capacity(packets * 4);
    let mut rssi = Vec::with_capacity(packets * 2);
    for (_, capture) in sessions {
        iq.extend(capture.iq.iter().map(|&v| v as u8));
        times.extend(capture.t.iter().flat_map(|&t| (t as f32).to_le_bytes()));
        rssi.extend(capture.rssi.iter().flat_map(|&r| r.to_le_bytes()));
    }
    let with = |suffix: &str| PathBuf::from(format!("{}.{suffix}", stem.display()));
    write_npy(&with("iq.npy"), "|i1", &[packets, subcarriers, 2], &iq)?;
    write_npy(&with("t.npy"), "<f4", &[packets], &times)?;
    write_npy(&with("rssi.npy"), "<i2", &[packets], &rssi)?;
    let metas: Vec<_> = sessions.iter().map(|(m, _)| m.clone()).collect();
    std::fs::write(with("json"), serde_json::to_string(&metas)?)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_dir = PathBuf::from(&args[0]);
    let first_room: u64 = args[1].parse()?;
    let rooms: u64 = args[2].parse()?;
    let seconds: f64 = args.get(3).map_or(Ok(60.0), |s| s.parse())?;
    std::fs::create_dir_all(&out_dir)?;
    let per_worker = rooms.div_ceil(WORKERS as u64);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..WORKERS as u64)
            .map(|worker| {
                let out_dir = &out_dir;
                scope.spawn(move || -> anyhow::Result<()> {
                    let start = first_room + worker * per_worker;
                    let end = (start + per_worker).min(first_room + rooms);
                    if start >= end {
                        return Ok(());
                    }
                    let sessions: Vec<_> = (start..end)
                        .flat_map(|seed| simulate_room_sessions(seed, seconds))
                        .collect();
                    write_chunk(&out_dir.join(format!("chunk_{start:07}")), &sessions)
                })
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|h| h.join().expect("worker panicked"))
    })?;
    println!("exported rooms {first_room}..{}", first_room + rooms);
    Ok(())
}
