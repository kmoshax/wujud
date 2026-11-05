//! Per-second training rows for learning where a person is, what they do and how they breathe from
//! three boards' links, with nothing about the room known but where the boards stand. One JSON line
//! per second: the links' disturbance and RSSI spread (as the live view computes them), each link's
//! per-subcarrier spread and spectrum shape (dB, mean removed), the boards' places, the room size, and
//! the simulator's truth. Odd seeds walk a lot; even seeds follow a random plan of every activity.
//! cargo run --release -p wujud --example export_tracking -- <out.jsonl> <first_seed> <rooms> [boards besides the hub, 2] [small]
//! `small`: bedrooms 2.3-3.3 by 2.2-3 m (the taped room of research/rooms/home_20261105.json is 2.7 by
//! 2.4), every link in both directions as the real boards measure them, and with four boards every
//! other room has them in its corners.

use std::fs::File;
use std::io::{BufWriter, Write};

use ndarray::{s, Axis};
use serde_json::json;
use wujud_model::motion_tracker::{disturbance_db, rssi_spread_db};
use wujud_sim::capture::Capture;
use wujud_sim::random::{seeded, uniform};
use wujud_sim::room::{random_plan, random_room, random_room_sized, Activity, PlanStep, Room, RoomOptions};
use wujud_sim::room_sim::RoomSimOptions;
use wujud_sim::stream::RoomStream;

/// Mostly walking, with pauses, so most seconds have a position to learn.
fn plan() -> Vec<PlanStep> {
    [
        (Activity::Away, 10.0),
        (Activity::Walk, 60.0),
        (Activity::Sit, 15.0),
        (Activity::Walk, 60.0),
        (Activity::Stand, 15.0),
        (Activity::Walk, 60.0),
    ]
    .map(|(activity, seconds)| PlanStep { activity, seconds })
    .to_vec()
}

fn small_room(seed: u64, options: &RoomOptions) -> Room {
    let mut rng = seeded(seed, 0);
    let mut room = random_room_sized(&mut rng, options, (2.3, 3.3), (2.2, 3.0));
    if room.nodes.len() == 4 && seed.is_multiple_of(2) {
        let (w, d) = (room.shape().width, room.shape().depth);
        let inset = uniform(&mut rng, 0.05, 0.3);
        let corners = [[inset, inset], [w - inset, inset], [w - inset, d - inset], [inset, d - inset]];
        let first = (seed / 2 % 4) as usize;
        let turn = if (seed / 8).is_multiple_of(2) { 1 } else { 3 };
        for (k, node) in room.nodes.iter_mut().enumerate() {
            let [x, y] = corners[(first + k * turn) % 4];
            *node = [x, y, node[2]];
        }
    }
    let n = room.nodes.len();
    room.links = (0..n).flat_map(|a| (0..n).filter(move |&b| b != a).map(move |b| (a, b))).collect();
    room
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = args.first().expect("out.jsonl");
    let first: u64 = args.get(1).map_or(1, |a| a.parse().unwrap());
    let rooms: u64 = args.get(2).map_or(10, |a| a.parse().unwrap());
    let boards: usize = args.get(3).map_or(2, |a| a.parse().unwrap());
    let small = args.get(4).is_some_and(|a| a == "small");
    let mut file = BufWriter::new(File::create(out).unwrap());
    for seed in first..first + rooms {
        let options = RoomOptions { boards, mesh: true, smart_placement: false };
        let room = if small { small_room(seed, &options) } else { random_room(&mut seeded(seed, 0), &options) };
        let mut rng = seeded(seed, 7);
        let plan = if seed % 2 == 1 { plan() } else { random_plan(&mut rng, 240.0) };
        let options = RoomSimOptions {
            seconds: plan.iter().map(|step| step.seconds).sum(),
            fs: 50.0,
            plan: Some(plan),
            harsh: uniform(&mut rng, 0.7, 2.0),
            ..RoomSimOptions::default()
        };
        let shape = room.shape();
        let nodes: Vec<[f64; 2]> = room.nodes.iter().map(|n| [n[0], n[1]]).collect();
        let links = room.links.clone();
        for second in RoomStream::new(&room, &options, seed) {
            let motion: Vec<f64> = second
                .links
                .iter()
                .map(|capture| {
                    let iq = capture.iq.slice(s![.., .., ..]);
                    let packets = iq.len_of(Axis(0));
                    let flat = iq.to_shape((packets, iq.len_of(Axis(1)) * 2)).unwrap();
                    disturbance_db(flat.view())
                })
                .collect();
            let shadowing: Vec<f64> = second
                .links
                .iter()
                .map(|capture| rssi_spread_db(&capture.rssi.iter().map(|&r| f64::from(r)).collect::<Vec<_>>()))
                .collect();
            let spread: Vec<Vec<f32>> = second.links.iter().map(|c| subcarrier_db(c).0).collect();
            let spectrum: Vec<Vec<f32>> = second.links.iter().map(|c| subcarrier_db(c).1).collect();
            let row = json!({
                "spread": spread,
                "shape": spectrum,
                "breath_bpm": second.truth.breath_bpm,
                "room": seed,
                "second": second.second,
                "width": shape.width,
                "depth": shape.depth,
                "nodes": nodes,
                "links": links,
                "motion": motion,
                "shadowing": shadowing,
                "activity": format!("{:?}", second.truth.activity),
                "xy": second.truth.xy,
            });
            writeln!(file, "{row}").unwrap();
        }
        eprint!("\rroom {seed}");
    }
    eprintln!();
}

/// Per subcarrier over the second: the spread of its dB level, and its mean level with the
/// spectrum's mean removed (gain steps cancel). Rounded to keep the files small.
fn subcarrier_db(capture: &Capture) -> (Vec<f32>, Vec<f32>) {
    let (packets, subcarriers) = (capture.iq.len_of(Axis(0)), capture.iq.len_of(Axis(1)));
    let db = |p: usize, k: usize| {
        let (i, q) = (f64::from(capture.iq[[p, k, 0]]), f64::from(capture.iq[[p, k, 1]]));
        20.0 * (i.hypot(q) + 1.0).log10()
    };
    let mut spread = Vec::with_capacity(subcarriers);
    let mut level = Vec::with_capacity(subcarriers);
    for k in 0..subcarriers {
        let values: Vec<f64> = (0..packets).map(|p| db(p, k)).collect();
        let mean = values.iter().sum::<f64>() / packets.max(1) as f64;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / packets.max(1) as f64;
        spread.push(var.sqrt());
        level.push(mean);
    }
    let overall = level.iter().sum::<f64>() / subcarriers.max(1) as f64;
    let round = |v: f64| ((v * 1000.0).round() / 1000.0) as f32;
    (spread.into_iter().map(round).collect(), level.into_iter().map(|v| round(v - overall)).collect())
}
