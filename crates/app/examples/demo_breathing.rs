//! Scores the engine's breathing rate against the simulator's truth while someone keeps still.
//! The room's state is given from the truth, so only the breathing reading is judged.
//! cargo run --release -p wujud --example demo_breathing -- <boards> <seeds>

use wujud_model::gbdt::Gbdt;
use wujud_model::numeric::median;
use wujud_model::room::{activity_model_file, RoomEngine, RoomState3};
use wujud_model::scene::Scene;
use wujud_sim::room::Activity;
use wujud_sim::stream::RoomStream;

fn main() -> anyhow::Result<()> {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let (boards, seeds) = (args.first().copied().unwrap_or(3), args.get(1).copied().unwrap_or(10));
    let models = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../model/models");
    let (mut errors, mut read, mut still) = (Vec::new(), 0, 0);
    for seed in 1..=seeds as u64 {
        let stream = RoomStream::demo(seed, boards);
        let room = stream.room().clone();
        let roles = std::iter::once("router".to_owned())
            .chain((1..room.nodes.len()).map(|_| "esp".to_owned()))
            .collect();
        let scene = Scene::room(room.shape().width, room.shape().depth, 2.8, &room.nodes, Vec::new(), roles);
        let model = Gbdt::load(&models.join(activity_model_file(room.links.len())))?;
        let mut engine = RoomEngine::new(scene, room.links.clone(), "SIM", model, room.nodes.clone());
        for second in stream {
            for (link, capture) in second.links.iter().enumerate() {
                for (p, &t) in capture.t.iter().enumerate() {
                    let iq: Vec<i8> = capture.iq.slice(ndarray::s![p, .., ..]).iter().copied().collect();
                    engine.add(link, t, &iq, f64::from(capture.rssi[p]));
                }
            }
            engine.sensed = Some(match second.truth.activity {
                Activity::Away => RoomState3::Empty,
                Activity::Walk => RoomState3::Moving,
                _ => RoomState3::Still,
            });
            let Ok(state) = engine.tick(second.second as f64 + 1.0) else { continue };
            let Some(truth) = second.truth.breath_bpm else { continue };
            still += 1;
            if let Some(bpm) = state.room.decision.and_then(|d| d.breath_bpm) {
                read += 1;
                errors.push((bpm - truth).abs());
            }
        }
    }
    println!(
        "{boards} boards, {seeds} rooms: breathing read in {read}/{still} still seconds; median error {:.1} /min; within 2 /min {:.0}%",
        median(&errors),
        errors.iter().filter(|&&e| e <= 2.0).count() as f64 / errors.len().max(1) as f64 * 100.0
    );
    Ok(())
}
