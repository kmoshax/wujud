//! Seconds per simulated room-minute, the recipe of research/tools/export_parity_sim.py
//! (random room, fs ~ U(50,120), 60 s of one activity + 20 s calibration).
//! Run: cargo run -p wujud-sim --release --example bench_room [rooms]

use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use wujud_sim::random::{chance, uniform};
use wujud_sim::room::{random_room, Activity, PlanStep, RoomOptions};
use wujud_sim::room_sim::{simulate_room, RoomSimOptions};

fn room_minutes(seed: u64) -> usize {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let smart_placement = chance(&mut rng, 0.3);
    let room = random_room(
        &mut rng,
        &RoomOptions {
            smart_placement,
            ..RoomOptions::default()
        },
    );
    let fs = uniform(&mut rng, 50.0, 120.0);
    let harsh = uniform(&mut rng, 0.8, 1.6);
    for activity in Activity::ALL {
        let options = RoomSimOptions {
            seconds: 60.0,
            fs,
            harsh,
            plan: Some(vec![PlanStep {
                activity,
                seconds: 60.0,
            }]),
            ..RoomSimOptions::default()
        };
        std::hint::black_box(simulate_room(&room, &options, &mut rng));
    }
    Activity::ALL.len()
}

fn main() {
    let rooms: u64 = std::env::args()
        .nth(1)
        .map_or(20, |a| a.parse().expect("room count"));
    let start = Instant::now();
    let minutes: usize = (0..rooms).map(room_minutes).sum();
    let single = start.elapsed().as_secs_f64() / minutes as f64;
    let start = Instant::now();
    let minutes: usize = (0..rooms * 5).into_par_iter().map(room_minutes).sum();
    let parallel = start.elapsed().as_secs_f64() / minutes as f64;
    println!(
        "single thread {single:.4} s/room-minute, rayon ({} threads) {parallel:.4} s/room-minute",
        rayon::current_num_threads()
    );
}
