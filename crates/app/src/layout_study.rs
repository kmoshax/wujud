//! How far from their real places the boards are drawn, without a room file: simulated bedrooms,
//! a person walking, the links' median RSSI handed to the same code the live view uses.

use std::collections::BTreeMap;

use rayon::prelude::*;
use serde::Serialize;
use wujud_model::board_layout::{align_to, largest_move};
use wujud_sim::band::{channel_2g4_hz, BandPlan, SubcarrierLayout};
use wujud_sim::random::seeded;
use wujud_sim::room::{random_room, RealismKnobs, Room, RoomOptions};
use wujud_sim::room_sim::{RoomSession, RoomSimOptions};

use crate::room_setup::{board_places, Radio, HUB};

const HUB_ADDRESS: &str = "udp:192.168.4.1";
const SECONDS: f64 = 30.0;
const PACKETS_PER_SECOND: f64 = 20.0;
/// Rooms only used to read the simulated radio's RSSI at one metre, as one would from a datasheet.
const RADIO_ROOMS: u64 = 100;
const RADIO_SEEDS_FROM: u64 = 1_000_000;
const CORNER_INSET_M: f64 = 0.1;

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Placement {
    /// Anywhere along the walls, as the simulator's rooms place them.
    Walls,
    /// One board in each corner.
    Corners,
}

impl Placement {
    fn name(self) -> &'static str {
        match self {
            Placement::Walls => "along the walls",
            Placement::Corners => "in the corners",
        }
    }
}

#[derive(Serialize)]
struct Row {
    placement: Placement,
    boards: usize,
    channels: usize,
    rooms: usize,
    mean_error_m: f64,
    median_error_m: f64,
    p90_error_m: f64,
    worst_board_median_m: f64,
}

#[derive(Serialize)]
struct Study {
    board_gain_spread_db: f64,
    simulated_rssi_at_1m: f64,
    simulated_exponent: f64,
    simulated_scatter_db: f64,
    seeds_from: u64,
    rows: Vec<Row>,
}

/// The places the live view would give the boards of a recording: each link's median RSSI.
pub fn place_recording(path: &std::path::Path) -> anyhow::Result<()> {
    let mut heard: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for line in std::fs::read_to_string(path)?.lines() {
        let fields: Vec<&str> = line.splitn(5, ',').collect();
        if let [_, link, "P", rssi, _] = fields.as_slice() {
            if let Ok(rssi) = rssi.parse::<f64>() {
                heard.entry((*link).to_owned()).or_default().push(rssi);
            }
        }
    }
    let rssi: BTreeMap<String, f64> = heard
        .into_iter()
        .map(|(link, mut values)| {
            values.sort_by(f64::total_cmp);
            (link, values[values.len() / 2])
        })
        .collect();
    let links: Vec<String> = rssi.keys().cloned().collect();
    board_places(&links, &rssi, crate::room_setup::ESP32_RADIO);
    Ok(())
}

pub fn run(rooms: usize, seeds_from: u64, board_gain_spread_db: f64, out: Option<&std::path::Path>) -> anyhow::Result<()> {
    let (radio, scatter_db) = simulated_radio(board_gain_spread_db);
    println!(
        "simulated radio: {:.1} dBm at 1 m, path-loss exponent {:.2}, links scatter {scatter_db:.1} dB around it",
        radio.rssi_at_1m, radio.exponent
    );
    let cases = [
        (Placement::Walls, 3),
        (Placement::Walls, 4),
        (Placement::Corners, 4),
    ];
    let mut rows = Vec::new();
    for (placement, boards) in cases {
        for channels in [vec![6], vec![1, 6, 11]] {
            let errors: Vec<(f64, f64)> = (seeds_from..seeds_from + rooms as u64)
                .into_par_iter()
                .map(|seed| room_error(seed, placement, boards, &channels, radio, board_gain_spread_db))
                .collect();
            let row = summarise(placement, boards, channels.len(), &errors);
            println!(
                "{} boards {}, {} channel(s): mean {:.2} m, median {:.2} m, p90 {:.2} m, worst board median {:.2} m",
                row.boards,
                placement.name(),
                row.channels,
                row.mean_error_m,
                row.median_error_m,
                row.p90_error_m,
                row.worst_board_median_m
            );
            rows.push(row);
        }
    }
    if let Some(path) = out {
        let study = Study {
            board_gain_spread_db,
            simulated_rssi_at_1m: radio.rssi_at_1m,
            simulated_exponent: radio.exponent,
            simulated_scatter_db: scatter_db,
            seeds_from,
            rows,
        };
        std::fs::write(path, serde_json::to_string_pretty(&study)?)?;
    }
    Ok(())
}

fn summarise(placement: Placement, boards: usize, channels: usize, errors: &[(f64, f64)]) -> Row {
    let mut mean: Vec<f64> = errors.iter().map(|e| e.0).collect();
    let mut worst: Vec<f64> = errors.iter().map(|e| e.1).collect();
    mean.sort_by(f64::total_cmp);
    worst.sort_by(f64::total_cmp);
    let at = |v: &[f64], q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    Row {
        placement,
        boards,
        channels,
        rooms: errors.len(),
        mean_error_m: mean.iter().sum::<f64>() / mean.len() as f64,
        median_error_m: at(&mean, 0.5),
        p90_error_m: at(&mean, 0.9),
        worst_board_median_m: at(&worst, 0.5),
    }
}

fn mac(board: usize) -> String {
    format!("aa:aa:aa:aa:aa:{board:02x}")
}

fn id(board: usize) -> String {
    if board == 0 {
        HUB.to_owned()
    } else {
        mac(board)
    }
}

/// The live link name of `tx` heard by `rx`, as the hub and the nodes tag their lines.
fn link_name(tx: usize, rx: usize) -> String {
    match (tx, rx) {
        (_, 0) => format!("{HUB_ADDRESS}@{}", mac(tx)),
        (0, _) => format!("{HUB_ADDRESS}@{}>hub", mac(rx)),
        _ => format!("{HUB_ADDRESS}@{}>{}", mac(rx), mac(tx)),
    }
}

struct Measured {
    room: Room,
    /// (tx, rx, median RSSI in dB)
    links: Vec<(usize, usize, f64)>,
}

fn measure_room(
    seed: u64,
    placement: Placement,
    boards: usize,
    channels: &[u32],
    board_gain_spread_db: f64,
) -> Measured {
    let mut rng = seeded(seed, 0x1a70);
    let mut room = random_room(
        &mut rng,
        &RoomOptions {
            boards: boards - 1,
            mesh: true,
            smart_placement: false,
        },
    );
    if let Placement::Corners = placement {
        put_in_corners(&mut room, seed);
    }
    let options = RoomSimOptions {
        seconds: SECONDS,
        fs: PACKETS_PER_SECOND,
        knobs: RealismKnobs {
            board_gain_spread_db,
            ..RealismKnobs::default()
        },
        ..RoomSimOptions::default()
    };
    let band = BandPlan::hopping(
        channels.iter().map(|&c| channel_2g4_hz(c)).collect(),
        SubcarrierLayout::HtLtf56,
    );
    let mut session = RoomSession::with_band(&room, &options, &band, &mut rng);
    let captures = session.capture(0..session.frames());
    let links = room
        .links
        .iter()
        .zip(&captures)
        .map(|(&(tx, rx), capture)| {
            (
                tx,
                rx,
                median_over_channels(&capture.rssi, &capture.frame, channels.len()),
            )
        })
        .collect();
    Measured { room, links }
}

/// Each board in its own corner, the hub in any of them, going round either way.
fn put_in_corners(room: &mut Room, seed: u64) {
    let (w, d) = (room.shape().width, room.shape().depth);
    let i = CORNER_INSET_M;
    let corners = [[i, i], [w - i, i], [w - i, d - i], [i, d - i]];
    let first = (seed % 4) as usize;
    let turn = if seed.is_multiple_of(2) { 1 } else { 3 };
    for (k, node) in room.nodes.iter_mut().enumerate().take(4) {
        let [x, y] = corners[(first + k * turn) % 4];
        *node = [x, y, node[2]];
    }
}

/// Least-squares line of RSSI against 10 log10(distance) over rooms the study never scores, and how
/// far the links scatter around it.
fn simulated_radio(board_gain_spread_db: f64) -> (Radio, f64) {
    let points: Vec<(f64, f64)> = (RADIO_SEEDS_FROM..RADIO_SEEDS_FROM + RADIO_ROOMS)
        .into_par_iter()
        .flat_map_iter(|seed| {
            let measured = measure_room(seed, Placement::Walls, 4, &[6], board_gain_spread_db);
            let nodes = measured.room.nodes.clone();
            measured.links.into_iter().map(move |(tx, rx, db)| {
                let (a, b) = (nodes[tx], nodes[rx]);
                (10.0 * (a[0] - b[0]).hypot(a[1] - b[1]).log10(), db)
            })
        })
        .collect();
    let n = points.len() as f64;
    let (mx, my) = (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let slope = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>()
        / points.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>();
    let scatter = (points
        .iter()
        .map(|p| (p.1 - my - slope * (p.0 - mx)).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    (
        Radio {
            rssi_at_1m: my - slope * mx,
            exponent: -slope,
        },
        scatter,
    )
}

/// (mean, worst) distance of a board from its real place, after the best turn, shift and mirror.
fn room_error(
    seed: u64,
    placement: Placement,
    boards: usize,
    channels: &[u32],
    radio: Radio,
    board_gain_spread_db: f64,
) -> (f64, f64) {
    let Measured {
        room,
        links: measured,
    } = measure_room(seed, placement, boards, channels, board_gain_spread_db);
    let mut links = Vec::new();
    let mut rssi = BTreeMap::new();
    for (tx, rx, db) in measured {
        let name = link_name(tx, rx);
        rssi.insert(name.clone(), db);
        links.push(name);
    }
    let places = board_places(&links, &rssi, radio);
    let truth: Vec<[f64; 2]> = room.nodes.iter().map(|p| [p[0], p[1]]).collect();
    let drawn: Vec<[f64; 2]> = (0..boards)
        .map(|b| places.get(&id(b)).map_or([0.0, 0.0], |p| [p[0], p[1]]))
        .collect();
    let aligned = align_to(&drawn, &truth);
    let mean = aligned
        .iter()
        .zip(&truth)
        .map(|(a, t)| (a[0] - t[0]).hypot(a[1] - t[1]))
        .sum::<f64>()
        / boards as f64;
    (mean, largest_move(&aligned, &truth))
}

/// Each channel's median RSSI, averaged in dB: packet `t` was sent on channel `frame mod channels`.
fn median_over_channels(rssi: &[i16], frame: &[u32], channels: usize) -> f64 {
    let medians: Vec<f64> = (0..channels)
        .filter_map(|c| {
            let mut on: Vec<f64> = rssi
                .iter()
                .zip(frame)
                .filter(|(_, &f)| f as usize % channels == c)
                .map(|(&r, _)| f64::from(r))
                .collect();
            on.sort_by(f64::total_cmp);
            on.get(on.len() / 2).copied()
        })
        .collect();
    medians.iter().sum::<f64>() / medians.len() as f64
}
