//! Plays a recording (`live --record`) through the room engine and the motion tracker, and prints
//! what they would have said every few seconds, with the wall-clock time, to compare against what
//! really happened in the room.
//! cargo run --release -p wujud --example replay -- <recording.csv> <room.json> [seconds per line]
//! With `-` for the recording it reads lines from stdin as they come, to judge a room live.

use std::collections::{BTreeMap, VecDeque};
use std::io::BufRead;

use serde_json::Value;
use wujud_model::gbdt::Gbdt;
use wujud_model::motion_tracker::{disturbance_db, rssi_spread_db, MotionTracker};
use wujud_model::room::{activity_model_file, RoomEngine};
use wujud_model::scene::Scene;
use wujud_proto::{Line, RxClockUnwrapper};

const RECENT_PACKETS: usize = 50;

struct Link {
    clock: RxClockUnwrapper,
    recent_iq: VecDeque<Vec<i8>>,
    recent_rssi: VecDeque<f64>,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (recording, room_file) = (&args[0], &args[1]);
    let every: f64 = args.get(2).map_or(5.0, |s| s.parse().unwrap());
    let room: Value = serde_json::from_str(&std::fs::read_to_string(room_file)?)?;
    let devices: BTreeMap<String, [f64; 3]> = serde_json::from_value(room["devices"].clone())?;
    let hub: [f64; 3] = serde_json::from_value(room["hub"].clone())?;
    let number = |key: &str| room[key].as_f64().unwrap();

    let names: Vec<&String> = devices.keys().collect();
    let mut nodes = vec![hub];
    nodes.extend(devices.values());
    let scene = Scene::room(
        number("W"),
        number("D"),
        number("H"),
        &nodes,
        vec![],
        vec!["router".into(); nodes.len()],
    );
    let links: Vec<(usize, usize)> = (1..nodes.len()).map(|k| (0, k)).collect();
    let models = concat!(env!("CARGO_MANIFEST_DIR"), "/../model/models/");
    let model = Gbdt::load(std::path::Path::new(&format!(
        "{models}{}",
        activity_model_file(links.len())
    )))?;
    let mut engine = RoomEngine::new(scene, links, "REPLAY", model, nodes);
    let mut tracker = MotionTracker::new(engine.link_ends(), number("W"), number("D"));
    let mut state: Vec<Link> = names
        .iter()
        .map(|_| Link {
            clock: RxClockUnwrapper::new(),
            recent_iq: VecDeque::new(),
            recent_rssi: VecDeque::new(),
        })
        .collect();

    let (mut newest_board, mut next_tick, mut next_print) = (f64::NAN, f64::NAN, f64::NAN);
    let mut last_track = None;
    let rows: Box<dyn BufRead> = if recording == "-" {
        Box::new(std::io::stdin().lock())
    } else {
        Box::new(std::io::BufReader::new(std::fs::File::open(recording)?))
    };
    for row in rows.lines() {
        let row = row?;
        let mut fields = row.splitn(3, ',');
        let (Some(wall), Some(port), Some(text)) = (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let wall: f64 = wall.parse()?;
        let device = port.rsplit('@').next().unwrap_or(port);
        let Some(link) = names.iter().position(|n| n.as_str() == device) else {
            continue;
        };
        let Ok(Line::Csi(packet)) = Line::parse(text) else {
            continue;
        };
        let iq = packet.csi.as_bytes().to_vec();
        let l = &mut state[link];
        let board_seconds = l.clock.seconds(packet.rx_timestamp_us);
        engine.add(link, board_seconds, &iq, f64::from(packet.rssi));
        newest_board = newest_board.max(board_seconds);
        l.recent_iq.push_back(iq);
        l.recent_rssi.push_back(f64::from(packet.rssi));
        if l.recent_iq.len() > RECENT_PACKETS {
            l.recent_iq.pop_front();
            l.recent_rssi.pop_front();
        }

        if next_tick.is_nan() {
            (next_tick, next_print) = (wall + 0.1, wall + every);
        }
        if wall >= next_tick {
            let motion: Vec<f64> = state.iter().map(motion_of).collect();
            let shadowing: Vec<f64> = state
                .iter()
                .map(|l| rssi_spread_db(&l.recent_rssi.iter().copied().collect::<Vec<_>>()))
                .collect();
            last_track = Some(tracker.update(&motion, &shadowing, 0.1));
            next_tick += 0.1;
        }
        if wall >= next_print {
            let decision = engine.tick(newest_board)?.room.decision;
            let clock = chrono_like(wall);
            match (decision, &last_track) {
                (Some(d), Some(t)) => println!(
                    "{clock}  {:<7} {:<6} probs {:?}  latch {:?}  motion {:.1}x  unix {wall:.1}",
                    format!("{:?}", d.state3),
                    format!("{:?}", d.state),
                    d.probs,
                    d.latch.distance,
                    t.link_excess.iter().copied().fold(0.0, f64::max),
                ),
                _ => println!("{clock}  (warming up)"),
            }
            next_print += every;
        }
    }
    Ok(())
}

fn motion_of(link: &Link) -> f64 {
    let Some(width) = link.recent_iq.front().map(Vec::len) else {
        return f64::NAN;
    };
    let bytes: Vec<i8> = link.recent_iq.iter().flatten().copied().collect();
    ndarray::ArrayView2::from_shape((link.recent_iq.len(), width), &bytes)
        .map_or(f64::NAN, disturbance_db)
}

/// Local time of day (Cairo, UTC+2/+3 is close enough for reading a timeline: the offset is printed).
fn chrono_like(unix: f64) -> String {
    let offset = std::env::var("UTC_OFFSET_HOURS")
        .ok()
        .and_then(|h| h.parse::<f64>().ok())
        .unwrap_or(2.0);
    let seconds = ((unix + offset * 3600.0) as u64) % 86_400;
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60
    )
}
