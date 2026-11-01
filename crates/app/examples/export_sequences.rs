//! Ten-per-second link sequences from simulated rooms, for the temporal network (research/tools/seq_net.py).
//! Every 100 ms and link: the spectrum shape in 32 bins (dB, mean removed, so gain steps cancel) and the
//! mean RSSI. Every second: the truth (activity, position, breathing). Rooms are random, with random board
//! places, random plans (odd seeds walk more) and random hardware harshness.
//!
//! Writes `<out>.bin` (f32, per room: frames x links x 33, frames at 10 Hz) and `<out>.json` (one entry per
//! room: offset, frames, links, nodes, room size, per-second truth).
//! cargo run --release -p wujud --example export_sequences -- <out> <first_seed> <rooms>

use std::fs::File;
use std::io::{BufWriter, Write};

use serde_json::json;
use wujud_sim::random::{seeded, uniform};
use wujud_sim::room::{random_plan, random_room, Activity, PlanStep, RoomOptions};
use wujud_sim::room_sim::RoomSimOptions;
use wujud_sim::stream::RoomStream;

const BINS: usize = 32;
const STEPS_PER_SECOND: usize = 10;

fn walking_plan() -> Vec<PlanStep> {
    [
        (Activity::Away, 10.0),
        (Activity::Walk, 40.0),
        (Activity::Sit, 30.0),
        (Activity::Walk, 30.0),
        (Activity::Lie, 40.0),
        (Activity::Walk, 30.0),
        (Activity::Stand, 30.0),
        (Activity::Walk, 30.0),
    ]
    .map(|(activity, seconds)| PlanStep { activity, seconds })
    .to_vec()
}

/// The spectrum shape of a few packets on BINS points, and their mean RSSI.
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = args.first().expect("out");
    let first: u64 = args.get(1).map_or(1, |a| a.parse().unwrap());
    let rooms: u64 = args.get(2).map_or(10, |a| a.parse().unwrap());
    let mut data = BufWriter::new(File::create(format!("{out}.bin")).unwrap());
    let mut index = Vec::new();
    let mut offset = 0usize;
    for seed in first..first + rooms {
        let room = random_room(&mut seeded(seed, 0), &RoomOptions { boards: 2, mesh: true, smart_placement: false });
        let mut rng = seeded(seed, 7);
        let plan = if seed % 2 == 1 { walking_plan() } else { random_plan(&mut rng, 240.0) };
        let options = RoomSimOptions {
            seconds: plan.iter().map(|step| step.seconds).sum(),
            fs: 50.0,
            plan: Some(plan),
            harsh: uniform(&mut rng, 0.7, 2.0),
            ..RoomSimOptions::default()
        };
        let (mut frames, mut truth) = (0usize, Vec::new());
        for second in RoomStream::new(&room, &options, seed) {
            for step in 0..STEPS_PER_SECOND {
                for capture in &second.links {
                    let packets = capture.iq.len_of(ndarray::Axis(0));
                    let (from, to) = (packets * step / STEPS_PER_SECOND, packets * (step + 1) / STEPS_PER_SECOND);
                    let features = step_features(capture.iq.slice(ndarray::s![from..to.max(from + 1).min(packets), .., ..]), &capture.rssi[from..to.max(from + 1).min(packets)]);
                    for v in features {
                        data.write_all(&v.to_le_bytes()).unwrap();
                    }
                }
                frames += 1;
            }
            truth.push(json!([format!("{:?}", second.truth.activity), second.truth.xy, second.truth.breath_bpm]));
        }
        let shape = room.shape();
        index.push(json!({
            "room": seed, "offset": offset, "frames": frames, "links": room.links,
            "nodes": room.nodes.iter().map(|n| [n[0], n[1]]).collect::<Vec<_>>(),
            "width": shape.width, "depth": shape.depth, "truth": truth,
        }));
        offset += frames * room.links.len() * (BINS + 1);
        eprint!("\rroom {seed}");
    }
    eprintln!();
    std::fs::write(format!("{out}.json"), serde_json::to_string(&index).unwrap()).unwrap();
}
