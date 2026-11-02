//! Per-second training rows for counting and placing several people at once: the same link features as
//! export_tracking, plus every person's activity and place. Each room holds 0 to `most_people` people
//! (drawn per room), each on their own plan of walking, sitting, standing, lying and being away, so the
//! number in the room and the number walking change over the session.
//! cargo run --release -p wujud --example export_crowd -- <out.jsonl> <first_seed> <rooms> [boards besides the hub, 2] [most people, 4]

use std::fs::File;
use std::io::{BufWriter, Write};

use ndarray::{s, Axis};
use serde_json::json;
use wujud_model::motion_tracker::{disturbance_db, rssi_spread_db};
use wujud_sim::capture::Capture;
use wujud_sim::crowd::Crowd;
use wujud_sim::random::{integer, seeded, uniform, weighted_index, SimRng};
use wujud_sim::room::{random_room, Activity, PlanStep, RoomOptions};
use wujud_sim::room_sim::RoomSimOptions;
use wujud_sim::stream::RoomStream;

const SESSION_S: f64 = 180.0;
const ACTIVITIES: [Activity; 5] = [Activity::Away, Activity::Walk, Activity::Sit, Activity::Stand, Activity::Lie];
const WEIGHTS: [f64; 5] = [0.2, 0.4, 0.15, 0.15, 0.1];

fn person_plan(rng: &mut SimRng) -> Vec<PlanStep> {
    let mut plan = Vec::new();
    let mut total = 0.0;
    while total < SESSION_S {
        let seconds = uniform(rng, 15.0, 60.0);
        plan.push(PlanStep { activity: ACTIVITIES[weighted_index(rng, &WEIGHTS)], seconds });
        total += seconds;
    }
    plan
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = args.first().expect("out.jsonl");
    let first: u64 = args.get(1).map_or(1, |a| a.parse().unwrap());
    let rooms: u64 = args.get(2).map_or(10, |a| a.parse().unwrap());
    let boards: usize = args.get(3).map_or(2, |a| a.parse().unwrap());
    let most_people: usize = args.get(4).map_or(4, |a| a.parse().unwrap());
    let mut file = BufWriter::new(File::create(out).unwrap());
    for seed in first..first + rooms {
        let room = random_room(&mut seeded(seed, 0), &RoomOptions { boards, mesh: true, smart_placement: false });
        let mut rng = seeded(seed, 7);
        let people = integer(&mut rng, 0, most_people + 1);
        let first_plan = if people == 0 {
            vec![PlanStep { activity: Activity::Away, seconds: SESSION_S }]
        } else {
            person_plan(&mut rng)
        };
        let crowd = Crowd { plans: (1..people).map(|_| Some(person_plan(&mut rng))).collect() };
        let options = RoomSimOptions {
            seconds: SESSION_S,
            fs: 50.0,
            plan: Some(first_plan),
            harsh: uniform(&mut rng, 0.7, 2.0),
            ..RoomSimOptions::default()
        };
        let shape = room.shape();
        let nodes: Vec<[f64; 2]> = room.nodes.iter().map(|n| [n[0], n[1]]).collect();
        let links = room.links.clone();
        let mut stream = RoomStream::with_crowd(&room, &options, &crowd, seed);
        while let Some(second) = stream.next() {
            let everyone = stream.people_at(second.second);
            let people_json: Vec<_> = everyone
                .iter()
                .take(people)
                .map(|truth| json!({ "activity": format!("{:?}", truth.activity), "xy": truth.xy }))
                .collect();
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
            let (spread, spectrum): (Vec<Vec<f32>>, Vec<Vec<f32>>) = second.links.iter().map(subcarrier_db).unzip();
            let row = json!({
                "spread": spread,
                "shape": spectrum,
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
                "people": people_json,
            });
            writeln!(file, "{row}").unwrap();
        }
        eprint!("\rroom {seed}");
    }
    eprintln!();
}

/// Per subcarrier over the second: the spread of its dB level, and its mean level with the
/// spectrum's mean removed (as export_tracking).
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
