//! Runs the room demo without a window and scores the engine's 3-state decision against the simulator's truth.
//! cargo run --release -p hallsense --example demo_accuracy -- <boards> <seed>

use hallsense_model::gbdt::Gbdt;
use hallsense_model::room::{activity_model_file, RoomEngine, RoomState3};
use hallsense_model::scene::Scene;
use hallsense_sim::room::Activity;
use hallsense_sim::stream::RoomStream;

const SETTLE_SECONDS: usize = 20;

fn main() -> anyhow::Result<()> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let (boards, seed) = (
        args.first().copied().unwrap_or(3),
        args.get(1).copied().unwrap_or(7) as u64,
    );
    let stream = RoomStream::demo(seed, boards);
    let room = stream.room().clone();
    let shape = room.shape();
    let scene = Scene::room(
        shape.width,
        shape.depth,
        shape.height,
        &room.nodes,
        vec![],
        vec!["router".into(); room.nodes.len()],
    );
    let models = concat!(env!("CARGO_MANIFEST_DIR"), "/../model/models/");
    let model = Gbdt::load(std::path::Path::new(&format!(
        "{models}{}",
        activity_model_file(boards)
    )))?;
    let mut engine = RoomEngine::new(
        scene,
        room.links.clone(),
        "BENCH",
        model,
        room.nodes.clone(),
    );
    let (mut right, mut scored, mut since_change, mut previous) = (0, 0, 0, None);
    for second in stream {
        for (link, capture) in second.links.iter().enumerate() {
            for p in 0..capture.t.len() {
                let iq: Vec<i8> = capture
                    .iq
                    .index_axis(ndarray::Axis(0), p)
                    .iter()
                    .copied()
                    .collect();
                engine.add(link, capture.t[p], &iq, f64::from(capture.rssi[p]));
            }
        }
        let now = second
            .links
            .iter()
            .flat_map(|c| c.t.last().copied())
            .fold(f64::NAN, f64::max);
        let state = engine.tick(now)?;
        let truth = match second.truth.activity {
            Activity::Away => RoomState3::Empty,
            Activity::Walk => RoomState3::Moving,
            _ => RoomState3::Still,
        };
        since_change = if previous == Some(truth) {
            since_change + 1
        } else {
            0
        };
        previous = Some(truth);
        let said = state.room.decision.as_ref().map(|d| d.state3);
        if since_change >= SETTLE_SECONDS {
            scored += 1;
            right += usize::from(said == Some(truth));
        }
        println!("{:4} truth {:?} said {:?}", second.second, truth, said);
    }
    println!(
        "ACCURACY {right}/{scored} = {:.1}% (seconds at least {SETTLE_SECONDS} s after a change)",
        100.0 * right as f64 / scored.max(1) as f64
    );
    Ok(())
}
