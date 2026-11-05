use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use wujud_model::board_layout::{self, Heard, Prior};
use wujud_model::gbdt::Gbdt;
use wujud_model::room::{activity_model_file, RoomEngine};
use wujud_model::scene::Scene;
use wujud_sim::room::{Furniture, Room};

/// research/room.example.json: metres, origin in a corner, z up.
/// `router` is the node every link of the hub shares (the hub board, in the room firmware's setup);
/// `boards` are the other ends, keyed by link name, by the node or device MAC, or by the board alone.
#[derive(Deserialize)]
pub struct RoomFile {
    #[serde(rename = "W")]
    pub width: f64,
    #[serde(rename = "D")]
    pub depth: f64,
    #[serde(rename = "H", default = "default_height")]
    pub height: f64,
    #[serde(alias = "hub")]
    pub router: [f64; 3],
    #[serde(alias = "devices")]
    pub boards: BTreeMap<String, [f64; 3]>,
    /// What to call each node in the 3D view, keyed like `boards`; `router` names the shared node.
    #[serde(default)]
    pub names: BTreeMap<String, String>,
    #[serde(default)]
    pub furniture: Vec<serde_json::Value>,
    /// What placing the boards found worth telling (a board that hears weakly).
    #[serde(skip)]
    pub notes: Vec<String>,
}

fn default_height() -> f64 {
    2.8
}

pub fn models_dir() -> PathBuf {
    std::env::var_os("WUJUD_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../model/models")))
}

pub struct LiveRoom {
    pub engine: RoomEngine,
    /// One per scene node: the shared node first.
    pub node_names: Vec<String>,
    /// The same nodes as the links name them (`router` for the hub, a MAC for a board).
    pub node_ids: Vec<String>,
    /// The links the engine reads, in its order, as named on the command line.
    pub links: Vec<String>,
    /// The same links as the window shows them.
    pub link_names: Vec<String>,
    /// The boards' places are known (a room file) or worked out from their signal strength.
    pub placed: bool,
    /// Where each board was put (`HUB` included), to tell when they have moved.
    pub places: BTreeMap<String, [f64; 3]>,
    pub notes: Vec<String>,
}

pub const HUB: &str = "router";

/// Both ends of a link, sender first. `<board>@<node>><sender>` is measured by a node that joined the
/// hub, from the hub (`hub`) or from another device; `<board>@<device>` by the hub from a device;
/// a bare `<board>` by the board from the router (which `HUB` then stands for).
pub fn link_ends(link: &str) -> (String, String) {
    match crate::serial::split_link_name(link) {
        (_, Some(tag)) => match tag.split_once('>') {
            Some((node, "hub")) => (HUB.to_owned(), node.to_owned()),
            Some((node, sender)) => (sender.to_owned(), node.to_owned()),
            None => (tag.to_owned(), HUB.to_owned()),
        },
        (board, None) => (HUB.to_owned(), board.to_owned()),
    }
}

/// Boards that joined the hub as nodes: the receivers of `<node>><sender>` links.
/// In the order they first appear, so a board switched on later does not take an earlier one's spot.
fn nodes_among(links: &[String]) -> Vec<String> {
    let mut nodes: Vec<String> = Vec::new();
    for tag in links.iter().filter_map(|link| crate::serial::split_link_name(link).1) {
        if let Some((node, _)) = tag.split_once('>') {
            if !nodes.iter().any(|n| n == node) {
                nodes.push(node.to_owned());
            }
        }
    }
    nodes
}

/// Links whose ends are not in the room are left out. Without a room file see `default_room`.
/// `rssi`: each link's median signal strength, to lay the boards out by themselves (`auto_layout`).
pub fn room_engine(
    links: &[String],
    room_file: Option<&Path>,
    rssi: &BTreeMap<String, f64>,
) -> anyhow::Result<LiveRoom> {
    let (room, placed) = match room_file {
        Some(path) => (serde_json::from_str::<RoomFile>(&std::fs::read_to_string(path)?)?, true),
        None => match auto_layout(links, rssi, ESP32_RADIO) {
            Some(room) => (room, true),
            None => (default_room(links), false),
        },
    };
    let mut places = room.boards.clone();
    places.insert(HUB.to_owned(), room.router);
    let notes = room.notes.clone();
    let position = |end: &str, link: &str| -> Option<[f64; 3]> {
        if end == HUB {
            return Some(room.router);
        }
        let board = crate::serial::split_link_name(link).0;
        [end, link, board]
            .into_iter()
            .find_map(|key| room.boards.get(key).copied())
    };
    let mut node_ids = vec![HUB.to_owned()];
    let mut nodes = vec![room.router];
    let mut kept = Vec::new();
    let mut pairs = Vec::new();
    for link in links {
        let (sender, receiver) = link_ends(link);
        let (Some(from), Some(to)) = (position(&sender, link), position(&receiver, link)) else {
            eprintln!("{link}: not placed in the room, left out");
            continue;
        };
        let mut node_of = |end: &str, at: [f64; 3]| {
            node_ids.iter().position(|id| id == end).unwrap_or_else(|| {
                node_ids.push(end.to_owned());
                nodes.push(at);
                nodes.len() - 1
            })
        };
        pairs.push((node_of(&sender, from), node_of(&receiver, to)));
        kept.push(link.clone());
    }
    anyhow::ensure!(!kept.is_empty(), "no link has both ends placed in the room");
    let node_names: Vec<String> = node_ids
        .iter()
        .map(|id| room.names.get(id).cloned().unwrap_or_else(|| id.clone()))
        .collect();
    let star = nodes_among(links).is_empty();
    let link_names = kept
        .iter()
        .zip(&pairs)
        .map(|(link, &(a, b))| match room.names.get(link) {
            Some(name) => name.clone(),
            None if star => node_names[a.max(b)].clone(),
            None => format!(
                "{} to {}",
                short_name(&node_ids[a]),
                short_name(&node_ids[b])
            ),
        })
        .collect();
    let roles = std::iter::once("router".to_owned())
        .chain((1..nodes.len()).map(|_| "esp".to_owned()))
        .collect();
    let scene = Scene::room(
        room.width,
        room.depth,
        room.height,
        &nodes,
        room.furniture,
        roles,
    );
    let model = Gbdt::load(&models_dir().join(activity_model_file(kept.len())))?;
    Ok(LiveRoom {
        engine: RoomEngine::new(scene, pairs, "LIVE", model, nodes),
        node_names,
        node_ids,
        links: kept,
        link_names,
        placed,
        places,
        notes,
    })
}

/// Where the room without a room file puts each board (`HUB` included), for the layout study.
/// Also the live view's check, every few seconds, of whether the boards have moved.
pub fn board_places(
    links: &[String],
    rssi: &BTreeMap<String, f64>,
    radio: Radio,
) -> BTreeMap<String, [f64; 3]> {
    let room = auto_layout(links, rssi, radio).unwrap_or_else(|| default_room(links));
    let mut places = room.boards;
    places.insert(HUB.to_owned(), room.router);
    places
}

/// The simulator's own room: its size, furniture and boards; node 0 is the router.
pub fn simulated_room_engine(room: &Room) -> anyhow::Result<LiveRoom> {
    let shape = room.shape();
    let furniture = [&room.bed, &room.desk, &room.wardrobe]
        .into_iter()
        .map(furniture_json)
        .collect();
    let boards = room.nodes.len() - 1;
    let roles = std::iter::once("router".to_owned())
        .chain((0..boards).map(|_| "esp".to_owned()))
        .collect();
    let scene = Scene::room(
        shape.width,
        shape.depth,
        shape.height,
        &room.nodes,
        furniture,
        roles,
    );
    let model = Gbdt::load(&models_dir().join(activity_model_file(boards)))?;
    let node_names: Vec<String> = std::iter::once("router".to_owned())
        .chain((1..=boards).map(|k| format!("ESP32 {k}")))
        .collect();
    let link_names = room
        .links
        .iter()
        .map(|&(a, b)| format!("{} to {}", node_names[a], node_names[b]))
        .collect();
    Ok(LiveRoom {
        engine: RoomEngine::new(
            scene,
            room.links.clone(),
            "SIMULATION",
            model,
            room.nodes.clone(),
        ),
        node_ids: node_names.clone(),
        node_names,
        links: room.links.iter().map(|(a, b)| format!("{a}-{b}")).collect(),
        link_names,
        placed: true,
        places: BTreeMap::new(),
        notes: Vec::new(),
    })
}

fn furniture_json(item: &Furniture) -> serde_json::Value {
    serde_json::json!({ "kind": item.kind, "x": item.x, "y": item.y, "z": item.z, "w": item.w, "d": item.d, "h": item.h })
}

/// The hub sits on the left wall. Its nodes go to the far corners, then mid-wall; a hub without
/// nodes has its devices spread along the walls instead, like research/live.py.
const NODE_SPOTS: [[f64; 3]; 4] = [
    [3.9, 0.3, 1.0],
    [3.9, 3.2, 1.0],
    [2.0, 0.1, 1.0],
    [2.0, 3.4, 1.0],
];
const DEVICE_SPOTS: [[f64; 3]; 4] = [
    [3.9, 1.75, 0.8],
    [2.0, 0.1, 1.0],
    [2.0, 3.4, 1.0],
    [0.1, 0.5, 1.2],
];

/// A 4 x 3.5 m room. With nodes, only the boards are placed, so only the links between boards are
/// used: where a phone lies is unknown.
fn default_room(links: &[String]) -> RoomFile {
    let nodes = nodes_among(links);
    let (boards, names) = if nodes.is_empty() {
        let devices = links.iter().map(|link| match link_ends(link) {
            (sender, receiver) if sender == HUB => receiver,
            (sender, _) => sender,
        });
        (
            devices.zip(DEVICE_SPOTS.iter().cycle().copied()).collect(),
            default_device_names(links),
        )
    } else {
        let mut names: BTreeMap<String, String> = nodes
            .iter()
            .map(|mac| (mac.clone(), format!("ESP32 · {}", mac_tail(mac))))
            .collect();
        names.insert(HUB.to_owned(), "ESP32 hub".to_owned());
        let boards = nodes
            .iter()
            .cloned()
            .zip(NODE_SPOTS.iter().cycle().copied())
            .collect();
        print_placement(&boards, &names);
        (boards, names)
    };
    RoomFile {
        width: 4.0,
        depth: 3.5,
        height: 2.8,
        router: [0.1, 1.75, 1.0],
        boards,
        furniture: vec![],
        names,
        notes: Vec::new(),
    }
}

/// Signal strength one metre from an ESP32 PCB antenna and how fast it falls indoors (log-distance).
#[derive(Clone, Copy, Debug)]
pub struct Radio {
    pub rssi_at_1m: f64,
    pub exponent: f64,
}

pub const ESP32_RADIO: Radio = Radio {
    rssi_at_1m: -45.0,
    exponent: 2.5,
};

/// How far one board's antenna and amplifier may be from another's; a phone sends at its own power.
/// research/rec/layout_room_20261105.csv: boards in a square, the hub (30:c0) heard 9 dB below the
/// others; with 3 dB the square came out as a line.
const BOARD_GAIN_SPREAD_DB: f64 = 6.0;
const COMMON_GAIN_SPREAD_DB: f64 = 1.0;
const PHONE_GAIN_SPREAD_DB: f64 = 10.0;
const EXPONENT_SPREAD: f64 = 0.5;
/// A link's own multipath fading around the path-loss line.
const LINK_FADING_DB: f64 = 3.0;
/// A bedroom's side, and how far one strays from it.
const ROOM_SIDE_M: f64 = 4.0;
const ROOM_SIDE_SPREAD_M: f64 = 1.0;
const WALL_MARGIN_M: f64 = 0.25;
/// Four boards in the corners have fewer unknowns than four anywhere on the walls: the corner layout
/// is kept unless the walls fit better by this much per board (Akaike: one per unknown).
const CORNERS_PREFERRED_BY: f64 = 1.0;
const BOARD_HEIGHT_M: f64 = 1.0;

/// Every board (and a phone every board hears) placed at once from how strongly each hears each
/// other (`wujud_model::board_layout`), squared to the walls, with the room drawn around them.
fn auto_layout(links: &[String], rssi: &BTreeMap<String, f64>, radio: Radio) -> Option<RoomFile> {
    let nodes = nodes_among(links);
    if nodes.is_empty() {
        return None;
    }
    let boards: Vec<String> = std::iter::once(HUB.to_owned())
        .chain(nodes.iter().cloned())
        .collect();
    let phones = devices_heard_by_boards(links, &boards);
    let ids: Vec<String> = boards.iter().chain(&phones).cloned().collect();
    let heard: Vec<Heard> = links
        .iter()
        .filter_map(|link| {
            let (sender, receiver) = link_ends(link);
            let from = ids.iter().position(|id| *id == sender)?;
            let to = ids.iter().position(|id| *id == receiver)?;
            Some(Heard {
                from,
                to,
                db: *rssi.get(link)?,
            })
        })
        .collect();
    let prior = Prior {
        rssi_at_1m: radio.rssi_at_1m,
        exponent: radio.exponent,
        exponent_spread: EXPONENT_SPREAD,
        gain_spread_db: ids
            .iter()
            .map(|id| {
                if boards.contains(id) {
                    BOARD_GAIN_SPREAD_DB
                } else {
                    PHONE_GAIN_SPREAD_DB
                }
            })
            .collect(),
        common_gain_spread_db: COMMON_GAIN_SPREAD_DB,
        residual_db: LINK_FADING_DB,
        room_side_m: ROOM_SIDE_M,
        room_side_spread_m: ROOM_SIDE_SPREAD_M,
    };
    let on_walls = board_layout::solve_on_walls(ids.len(), &heard, &prior)?;
    let in_corners = (ids.len() == 4)
        .then(|| board_layout::solve_in_corners(&heard, &prior))
        .flatten();
    let layout = match in_corners {
        Some(corners)
            if corners.cost <= on_walls.cost + CORNERS_PREFERRED_BY * ids.len() as f64 =>
        {
            corners
        }
        _ => on_walls,
    };
    let points = board_layout::square_to_walls(&layout.points, WALL_MARGIN_M);
    let (width, depth) = board_layout::extent(&points);
    let mut names: BTreeMap<String, String> = nodes
        .iter()
        .map(|mac| (mac.clone(), format!("ESP32 · {}", mac_tail(mac))))
        .collect();
    names.insert(HUB.to_owned(), "ESP32 hub".to_owned());
    for phone in &phones {
        names.insert(phone.clone(), format!("phone · {}", mac_tail(phone)));
    }
    println!(
        "{} boards{} placed from {} links: they fit within {:.1} dB, path-loss exponent {:.2}",
        boards.len(),
        if phones.is_empty() {
            String::new()
        } else {
            format!(" and {} phone(s)", phones.len())
        },
        heard.len(),
        layout.residual_db,
        layout.exponent
    );
    for ((id, p), gain) in ids.iter().zip(&points).zip(&layout.gains_db) {
        let name = names.get(id).map_or(id.as_str(), String::as_str);
        println!("  {name:<16} ({:.1}, {:.1}) m, gain {gain:+.1} dB", p[0], p[1]);
    }
    let notes = weak_boards(&boards, &ids, &layout.gains_db);
    for note in &notes {
        println!("  {note}");
    }
    let place = |p: [f64; 2]| [p[0], p[1], BOARD_HEIGHT_M];
    Some(RoomFile {
        width: (width + 2.0 * WALL_MARGIN_M).max(2.0),
        depth: (depth + 2.0 * WALL_MARGIN_M).max(2.0),
        height: 2.8,
        router: place(points[0]),
        boards: ids
            .iter()
            .zip(&points)
            .skip(1)
            .map(|(id, &p)| (id.clone(), place(p)))
            .collect(),
        furniture: vec![],
        names,
        notes,
    })
}

/// A board this far below the others' typical gain is worth a look: its antenna, its cable, or
/// something metal next to it. research/rec/layout_room_20261105.csv: 30:c0 12 dB below the median;
/// with another board in its place (layout_room_20261105_swapped.csv) no board was more than 5 dB off.
const WEAK_BOARD_DB: f64 = 8.0;

fn weak_boards(boards: &[String], ids: &[String], gains_db: &[f64]) -> Vec<String> {
    let mut board_gains: Vec<(String, f64)> = ids
        .iter()
        .zip(gains_db)
        .filter(|(id, _)| boards.contains(id))
        .map(|(id, &g)| (id.clone(), g))
        .collect();
    board_gains.sort_by(|a, b| a.1.total_cmp(&b.1));
    let middle = board_gains.len() / 2;
    let typical = if board_gains.len().is_multiple_of(2) {
        (board_gains[middle - 1].1 + board_gains[middle].1) / 2.0
    } else {
        board_gains[middle].1
    };
    board_gains
        .iter()
        .filter(|(_, g)| typical - g >= WEAK_BOARD_DB)
        .map(|(id, g)| format!("{} hears {:.0} dB low: check its antenna", short_name(id), typical - g))
        .collect()
}

/// Senders that are not boards but are heard by every board.
fn devices_heard_by_boards(links: &[String], boards: &[String]) -> Vec<String> {
    let mut heard: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for link in links {
        let (sender, receiver) = link_ends(link);
        if !boards.contains(&sender) && boards.contains(&receiver) {
            heard.entry(sender).or_default().insert(receiver);
        }
    }
    heard.into_iter().filter(|(_, by)| by.len() == boards.len()).map(|(device, _)| device).collect()
}

fn print_placement(boards: &BTreeMap<String, [f64; 3]>, names: &BTreeMap<String, String>) {
    println!("no --room: a 4 x 3.5 m room, x along the wall in front of you, y away from you");
    println!("  {:<16} (0.1, 1.75) m", names[HUB]);
    for (mac, [x, y, _]) in boards {
        println!("  {:<16} ({x:.1}, {y:.1}) m", names[mac]);
    }
}

/// What fits in a link's title: `hub`, or the end of a MAC.
fn short_name(id: &str) -> &str {
    if id == HUB {
        "hub"
    } else {
        mac_tail(id)
    }
}

pub fn mac_tail(mac: &str) -> &str {
    &mac[mac.len().saturating_sub(5)..]
}

/// A board measuring the devices joined to it is the room's hub; each device is named by its order
/// and the end of its MAC, which is what fits on screen.
fn default_device_names(links: &[String]) -> BTreeMap<String, String> {
    let mut names = BTreeMap::new();
    if links
        .iter()
        .any(|l| crate::serial::split_link_name(l).1.is_some())
    {
        names.insert(HUB.to_owned(), "ESP32".to_owned());
    }
    for (k, link) in links.iter().enumerate() {
        if let (_, Some(mac)) = crate::serial::split_link_name(link) {
            names.insert(
                mac.to_owned(),
                format!("Device {} · {}", k + 1, mac_tail(mac)),
            );
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    const HUB_LINK: &str = "udp:192.168.4.1@aa:aa:aa:aa:aa:01";

    fn mesh_links() -> Vec<String> {
        [
            HUB_LINK,
            "udp:192.168.4.1@aa:aa:aa:aa:aa:02",
            "udp:192.168.4.1@aa:aa:aa:aa:aa:03",
            "udp:192.168.4.1@aa:aa:aa:aa:aa:01>hub",
            "udp:192.168.4.1@aa:aa:aa:aa:aa:01>aa:aa:aa:aa:aa:02",
            "udp:192.168.4.1@aa:aa:aa:aa:aa:01>aa:aa:aa:aa:aa:03",
            "udp:192.168.4.1@aa:aa:aa:aa:aa:02>hub",
            "udp:192.168.4.1@aa:aa:aa:aa:aa:02>aa:aa:aa:aa:aa:01",
        ]
        .map(str::to_owned)
        .to_vec()
    }

    #[test]
    fn link_ends_read_hub_node_and_bare_links() {
        let ends = |link: &str| link_ends(link);
        assert_eq!(ends(HUB_LINK), ("aa:aa:aa:aa:aa:01".into(), HUB.into()));
        assert_eq!(
            ends("udp:192.168.4.1@aa:aa:aa:aa:aa:01>hub"),
            (HUB.into(), "aa:aa:aa:aa:aa:01".into())
        );
        assert_eq!(
            ends("udp:192.168.4.1@aa:aa:aa:aa:aa:01>bb:bb:bb:bb:bb:bb"),
            ("bb:bb:bb:bb:bb:bb".into(), "aa:aa:aa:aa:aa:01".into())
        );
        assert_eq!(ends("/dev/ttyACM0"), (HUB.into(), "/dev/ttyACM0".into()));
    }

    #[test]
    fn with_nodes_only_links_between_boards_are_kept() {
        let room = room_engine(&mesh_links(), None, &BTreeMap::new()).unwrap();
        // aa..03 is a phone: heard by the hub and by a node, but never a node itself.
        assert!(room.links.iter().all(|l| !l.ends_with(":03")));
        assert_eq!(room.links.len(), 6);
        assert_eq!(
            room.node_names,
            ["ESP32 hub", "ESP32 · aa:01", "ESP32 · aa:02"]
        );
        assert_eq!(room.link_names[0], "aa:01 to hub");
        assert_eq!(room.link_names[2], "hub to aa:01");
        assert_eq!(room.link_names[3], "aa:02 to aa:01");
        assert_eq!(room.engine.link_ends().len(), 6);
    }

    #[test]
    fn without_nodes_every_device_is_a_link_from_the_hub() {
        let links = mesh_links()[..3].to_vec();
        let room = room_engine(&links, None, &BTreeMap::new()).unwrap();
        assert_eq!(room.links, links);
        assert_eq!(room.link_names[0], "Device 1 · aa:01");
    }

    #[test]
    fn four_boards_in_a_square_are_drawn_as_a_square() {
        let at = |id: &str| -> [f64; 2] {
            match id {
                HUB => [0.0, 0.0],
                "aa:aa:aa:aa:aa:01" => [3.0, 0.0],
                "aa:aa:aa:aa:aa:02" => [3.0, 3.0],
                _ => [0.0, 3.0],
            }
        };
        let macs = [
            "aa:aa:aa:aa:aa:01",
            "aa:aa:aa:aa:aa:02",
            "aa:aa:aa:aa:aa:03",
        ];
        let mut links: Vec<String> = macs
            .iter()
            .map(|m| format!("udp:192.168.4.1@{m}"))
            .collect();
        for node in macs {
            links.push(format!("udp:192.168.4.1@{node}>hub"));
            links.extend(
                macs.iter()
                    .filter(|&&m| m != node)
                    .map(|m| format!("udp:192.168.4.1@{node}>{m}")),
            );
        }
        let rssi: BTreeMap<String, f64> = links
            .iter()
            .map(|link| {
                let (a, b) = link_ends(link);
                let (p, q) = (at(&a), at(&b));
                let d = (p[0] - q[0]).hypot(p[1] - q[1]);
                (
                    link.clone(),
                    ESP32_RADIO.rssi_at_1m - 10.0 * ESP32_RADIO.exponent * d.log10(),
                )
            })
            .collect();
        let places = board_places(&links, &rssi, ESP32_RADIO);
        let ids = [HUB, macs[0], macs[1], macs[2]];
        let drawn: Vec<[f64; 2]> = ids
            .iter()
            .map(|id| [places[*id][0], places[*id][1]])
            .collect();
        let truth: Vec<[f64; 2]> = ids.iter().map(|id| at(id)).collect();
        // RSSI hardly tells a small room from a big one with weaker boards: only the shape is checked.
        let size = |p: &[[f64; 2]]| (p[0][0] - p[2][0]).hypot(p[0][1] - p[2][1]);
        let scaled: Vec<[f64; 2]> = drawn
            .iter()
            .map(|p| {
                [
                    p[0] * size(&truth) / size(&drawn),
                    p[1] * size(&truth) / size(&drawn),
                ]
            })
            .collect();
        let aligned = board_layout::align_to(&scaled, &truth);
        assert!(
            board_layout::largest_move(&aligned, &truth) < 0.15,
            "{drawn:?}"
        );
    }

    #[test]
    fn the_measured_square_with_a_weak_hub_stays_a_square() {
        // research/rec/layout_room_20261105.csv: four boards in a square, the hub heard 12-15 dB low.
        let heard = [
            ("ec:e3:34:21:b5:14", -64.0),
            ("ec:e3:34:21:d9:bc", -65.0),
            ("ec:e3:34:22:74:ec", -62.0),
            ("ec:e3:34:21:b5:14>hub", -65.0),
            ("ec:e3:34:21:d9:bc>hub", -65.0),
            ("ec:e3:34:22:74:ec>hub", -62.0),
            ("ec:e3:34:21:b5:14>ec:e3:34:21:d9:bc", -49.0),
            ("ec:e3:34:21:b5:14>ec:e3:34:22:74:ec", -56.0),
            ("ec:e3:34:21:d9:bc>ec:e3:34:21:b5:14", -45.0),
            ("ec:e3:34:21:d9:bc>ec:e3:34:22:74:ec", -50.0),
            ("ec:e3:34:22:74:ec>ec:e3:34:21:b5:14", -53.0),
            ("ec:e3:34:22:74:ec>ec:e3:34:21:d9:bc", -50.0),
        ];
        let rssi: BTreeMap<String, f64> =
            heard.iter().map(|(tag, db)| (format!("/dev/ttyACM0@{tag}"), *db)).collect();
        let links: Vec<String> = rssi.keys().cloned().collect();
        let places = board_places(&links, &rssi, ESP32_RADIO);
        let at = |id: &str| [places[id][0], places[id][1]];
        let apart = |a: &str, b: &str| (at(a)[0] - at(b)[0]).hypot(at(a)[1] - at(b)[1]);
        let (b5, d9, e74) = ("ec:e3:34:21:b5:14", "ec:e3:34:21:d9:bc", "ec:e3:34:22:74:ec");
        let diagonals = [apart(HUB, d9), apart(b5, e74)];
        let sides = [apart(HUB, b5), apart(b5, d9), apart(d9, e74), apart(e74, HUB)];
        let longest_side = sides.iter().copied().fold(0.0, f64::max);
        let shortest_side = sides.iter().copied().fold(f64::MAX, f64::min);
        assert!(diagonals.iter().all(|&d| d > longest_side), "{places:?}");
        assert!(longest_side < 1.5 * shortest_side, "{places:?}");
    }
}
