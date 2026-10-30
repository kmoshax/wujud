//! Scores the motion tracker's position against the simulator's truth while the person walks.
//! cargo run --release -p wujud --example demo_tracking -- <boards> <seeds>

use ndarray::{s, Axis};
use wujud_model::motion_tracker::{disturbance_db, rssi_spread_db, MotionTracker};
use wujud_model::numeric::median;
use wujud_sim::room::Activity;
use wujud_sim::stream::RoomStream;

const UPDATES_PER_SECOND: usize = 1;

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let (boards, seeds) = (
        args.first().copied().unwrap_or(3),
        args.get(1).copied().unwrap_or(10),
    );
    let (mut errors, mut centre_errors, mut found, mut walking) = (vec![], vec![], 0, 0);
    for seed in 1..=seeds as u64 {
        let stream = RoomStream::demo(seed, boards);
        let room = stream.room().clone();
        let shape = room.shape();
        let plan = |node: usize| [room.nodes[node][0], room.nodes[node][1]];
        let ends = room
            .links
            .iter()
            .map(|&(a, b)| (plan(a), plan(b)))
            .collect();
        let mut tracker = MotionTracker::new(ends, shape.width, shape.depth);
        let centre = [shape.width / 2.0, shape.depth / 2.0];
        for second in stream {
            let mut track = None;
            for part in 0..UPDATES_PER_SECOND {
                let shadowing: Vec<f64> = second
                    .links
                    .iter()
                    .map(|capture| {
                        let packets = capture.iq.len_of(Axis(0));
                        let from = packets * part / UPDATES_PER_SECOND;
                        let to = packets * (part + 1) / UPDATES_PER_SECOND;
                        let rssi: Vec<f64> = capture.rssi[from..to]
                            .iter()
                            .map(|&r| f64::from(r))
                            .collect();
                        rssi_spread_db(&rssi)
                    })
                    .collect();
                let motion: Vec<f64> = second
                    .links
                    .iter()
                    .map(|capture| {
                        let packets = capture.iq.len_of(Axis(0));
                        let from = packets * part / UPDATES_PER_SECOND;
                        let to = packets * (part + 1) / UPDATES_PER_SECOND;
                        let iq = capture.iq.slice(s![from..to, .., ..]);
                        let flat = iq.to_shape((to - from, iq.len_of(Axis(1)) * 2)).unwrap();
                        disturbance_db(flat.view())
                    })
                    .collect();
                track = Some(tracker.update(&motion, &shadowing, 1.0 / UPDATES_PER_SECOND as f64));
            }
            let (Some(track), Activity::Walk, Some(truth)) =
                (track, second.truth.activity, second.truth.xy)
            else {
                continue;
            };
            walking += 1;
            centre_errors.push(distance(centre, truth));
            if let (true, Some(xy)) = (track.moving, track.xy) {
                found += 1;
                errors.push(distance(xy, truth));
            }
        }
    }
    println!(
        "{boards} links, {seeds} rooms: tracked {found}/{walking} walking seconds; \
         median error {:.2} m (room centre {:.2} m)",
        median(&errors),
        median(&centre_errors)
    );
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
