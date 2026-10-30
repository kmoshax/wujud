//! Prints the live-demo streams second by second and how long each simulated second took.
//! Run: cargo run -p hallsense-sim --release --example stream_demo -- room|hall [seconds]

use std::time::Instant;

use hallsense_sim::stream::{HallStream, RoomStream};

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "room".to_string());
    let seconds: usize = args.next().map_or(10, |s| s.parse().expect("seconds"));
    let start = Instant::now();
    if mode == "hall" {
        let mut stream = HallStream::demo(3);
        println!(
            "hall demo ready after {:.2?}, {} directions",
            start.elapsed(),
            stream.directions().len()
        );
        for _ in 0..seconds {
            let tick = Instant::now();
            let Some(second) = stream.next() else { break };
            let packets: usize = second.links.iter().map(|c| c.len()).sum();
            let present = second.present.iter().filter(|&&p| p).count();
            println!(
                "t={:3}s packets {packets:5} students {present:3}  ({:.2?})",
                second.second,
                tick.elapsed()
            );
        }
    } else {
        let mut stream = RoomStream::demo(7, 3);
        println!("room demo ready after {:.2?}", start.elapsed());
        for _ in 0..seconds {
            let tick = Instant::now();
            let Some(second) = stream.next() else { break };
            let packets: Vec<usize> = second.links.iter().map(|c| c.len()).collect();
            println!(
                "t={:3}s packets {packets:?} truth {:?}  ({:.2?})",
                second.second,
                second.truth,
                tick.elapsed()
            );
        }
    }
}
