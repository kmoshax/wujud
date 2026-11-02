//! Ten-per-second link sequences of a person who prays, waves, sits down and stands up, for the movement
//! network (research/tools/movement_net.py). Rooms are random with 3 or 4 devices in a full mesh, random
//! board places, random hardware harshness and person realism, and now and then a fan, pet or door.
//!
//! Every 100 ms and link: the spectrum shape in 32 bins (dB, mean removed) and the mean RSSI, as
//! export_sequences. Every 100 ms: the movement under way (`Movement::ALL` index).
//!
//! Writes `<out>.bin` (f32, per room: steps x links x 33) and `<out>.json` (per room: offset, steps, links,
//! nodes, room size, movement per step, activity per second).
//! cargo run --release -p wujud --example export_movements -- <out> <first_seed> <rooms> [seconds]

use std::fs::File;
use std::io::{BufWriter, Write};

use rayon::prelude::*;
use serde_json::json;
use wujud_sim::distractors::DistractorKind;
use wujud_sim::random::{chance, integer, seeded, uniform};
use wujud_sim::room::{random_movement_plan, random_room, RealismKnobs, RoomOptions};
use wujud_sim::room_sim::RoomSimOptions;
use wujud_sim::stream::RoomStream;

const BINS: usize = 32;
const STEPS_PER_SECOND: usize = 10;
const FS: f64 = 50.0;
const DISTRACTORS: [DistractorKind; 3] = [DistractorKind::Fan, DistractorKind::Pet, DistractorKind::Door];

fn step_features(iq: ndarray::ArrayView3<i8>, rssi: &[i16]) -> [f32; BINS + 1] {
    let (packets, subcarriers) = (iq.len_of(ndarray::Axis(0)), iq.len_of(ndarray::Axis(1)));
    let mut level = vec![0.0f64; subcarriers];
    for p in 0..packets {
        for (k, l) in level.iter_mut().enumerate() {
            let (i, q) = (f64::from(iq[[p, k, 0]]), f64::from(iq[[p, k, 1]]));
            *l += 20.0 * (i.hypot(q) + 1.0).log10() / packets.max(1) as f64;
        }
    }
    let mean = level.iter().sum::<f64>() / subcarriers.max(1) as f64;
    let mut out = [0.0f32; BINS + 1];
    for (b, o) in out.iter_mut().take(BINS).enumerate() {
        let position = b as f64 * (subcarriers - 1) as f64 / (BINS - 1) as f64;
        let (lo, hi) = (position.floor() as usize, position.ceil() as usize);
        let w = position - lo as f64;
        *o = ((level[lo] * (1.0 - w) + level[hi] * w) - mean) as f32;
    }
    out[BINS] = (rssi.iter().map(|&r| f64::from(r)).sum::<f64>() / rssi.len().max(1) as f64) as f32;
    out
}

struct RoomExport {
    features: Vec<f32>,
    index: serde_json::Value,
}

fn export_room(seed: u64, seconds: f64) -> RoomExport {
    let mut rng = seeded(seed, 11);
    let boards = integer(&mut rng, 2, 4);
    let room = random_room(&mut seeded(seed, 0), &RoomOptions { boards, mesh: true, smart_placement: false });
    let plan = random_movement_plan(&mut rng, seconds);
    let distractors = if chance(&mut rng, 0.15) {
        vec![DISTRACTORS[integer(&mut rng, 0, DISTRACTORS.len())]]
    } else {
        Vec::new()
    };
    let options = RoomSimOptions {
        seconds: plan.iter().map(|step| step.seconds).sum(),
        fs: FS,
        plan: Some(plan),
        harsh: uniform(&mut rng, 0.7, 2.0),
        distractors,
        knobs: RealismKnobs {
            rcs_scale: 10f64.powf(uniform(&mut rng, -1.0, 0.0)),
            breath_scale: uniform(&mut rng, 0.5, 1.0),
            evm_shift_db: uniform(&mut rng, 0.0, 6.0),
        },
        ..RoomSimOptions::default()
    };
    let mut stream = RoomStream::new(&room, &options, seed);
    let (mut features, mut movement, mut activity) = (Vec::new(), Vec::new(), Vec::new());
    while let Some(second) = stream.next() {
        for step in 0..STEPS_PER_SECOND {
            for capture in &second.links {
                let packets = capture.iq.len_of(ndarray::Axis(0));
                let from = packets * step / STEPS_PER_SECOND;
                let to = (packets * (step + 1) / STEPS_PER_SECOND).max(from + 1).min(packets);
                features.extend(step_features(capture.iq.slice(ndarray::s![from..to, .., ..]), &capture.rssi[from..to]));
            }
            let frame = ((second.second as f64 + (step as f64 + 0.5) / STEPS_PER_SECOND as f64) * FS) as usize;
            let labels = stream.labels();
            movement.push(labels.movement_at(frame.min(labels.activity.len() - 1)).index());
        }
        activity.push(second.truth.activity.name());
    }
    let shape = room.shape();
    let index = json!({
        "room": seed, "steps": movement.len(), "links": room.links,
        "nodes": room.nodes.iter().map(|n| [n[0], n[1]]).collect::<Vec<_>>(),
        "width": shape.width, "depth": shape.depth, "harsh": options.harsh,
        "movement": movement, "activity": activity,
    });
    RoomExport { features, index }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = args.first().expect("out");
    let first: u64 = args.get(1).map_or(1, |a| a.parse().unwrap());
    let rooms: u64 = args.get(2).map_or(10, |a| a.parse().unwrap());
    let seconds: f64 = args.get(3).map_or(240.0, |a| a.parse().unwrap());
    let exports: Vec<RoomExport> = (first..first + rooms).into_par_iter().map(|seed| export_room(seed, seconds)).collect();
    let mut data = BufWriter::new(File::create(format!("{out}.bin")).unwrap());
    let mut index = Vec::new();
    let mut offset = 0usize;
    for mut export in exports {
        for v in &export.features {
            data.write_all(&v.to_le_bytes()).unwrap();
        }
        export.index["offset"] = json!(offset);
        offset += export.features.len();
        index.push(export.index);
    }
    std::fs::write(format!("{out}.json"), serde_json::to_string(&index).unwrap()).unwrap();
}
