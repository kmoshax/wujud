use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;
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
}

const HUB: &str = "router";

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
        None => match auto_layout(links, rssi) {
            Some(room) => (room, true),
            None => (default_room(links), false),
        },
    };
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
    })
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
    }
}

/// Signal strength one metre from an ESP32 PCB antenna and how fast it falls indoors (log-distance).
/// Good to perhaps a third of the distance; the shape of the boards' triangle matters more than its size.
const RSSI_AT_1M: f64 = -45.0;
const PATH_LOSS_EXPONENT: f64 = 2.5;
/// A bedroom: the longest distance between two boards is taken as at most this.
const LONGEST_SPAN_M: f64 = 4.5;
const WALL_MARGIN_M: f64 = 0.4;
const BOARD_HEIGHT_M: f64 = 1.0;

fn metres_from_rssi(rssi: f64) -> f64 {
    10f64.powf((RSSI_AT_1M - rssi) / (10.0 * PATH_LOSS_EXPONENT)).clamp(0.5, 10.0)
}

/// RSSI distances are rough, and three of them often cannot close a triangle (two short sides and a
/// long one put the boards on one line, which then reads as a two-board room). The longest side is
/// shortened until the triangle keeps a real height.
fn triangle(ha: f64, hb: f64, ab: f64) -> (f64, f64, f64) {
    let mut sides = [ha, hb, ab];
    let longest = (0..3).max_by(|&i, &j| sides[i].total_cmp(&sides[j])).unwrap_or(0);
    let others: f64 = sides.iter().enumerate().filter(|&(i, _)| i != longest).map(|(_, s)| s).sum();
    sides[longest] = sides[longest].min(others * MOST_OF_THE_OTHER_SIDES);
    (sides[0], sides[1], sides[2])
}

/// The longest side is at most this share of the other two together: an angle of about 100 degrees.
const MOST_OF_THE_OTHER_SIDES: f64 = 0.8;

/// Hub and two nodes placed from how strongly they hear each other: a triangle with those three
/// distances, scaled into a bedroom, with the room drawn around it. Needs every pair heard.
fn auto_layout(links: &[String], rssi: &BTreeMap<String, f64>) -> Option<RoomFile> {
    let nodes = nodes_among(links);
    let [a, b] = <[String; 2]>::try_from(nodes.clone()).ok()?;
    let distance = |x: &str, y: &str| -> Option<f64> {
        let heard: Vec<f64> = links
            .iter()
            .filter(|link| {
                let (s, r) = link_ends(link);
                (s == x && r == y) || (s == y && r == x)
            })
            .filter_map(|link| rssi.get(link).copied())
            .collect();
        (!heard.is_empty()).then(|| metres_from_rssi(heard.iter().sum::<f64>() / heard.len() as f64))
    };
    let (ha, hb, ab) = triangle(distance(HUB, &a)?, distance(HUB, &b)?, distance(&a, &b)?);
    let scale = (LONGEST_SPAN_M / ha.max(hb).max(ab)).min(1.0);
    let (ha, hb, ab) = (ha * scale, hb * scale, ab * scale);
    // Hub at the origin, node a along x, node b by the law of cosines (above x).
    let bx = (ha * ha + hb * hb - ab * ab) / (2.0 * ha);
    let by = (hb * hb - bx * bx).max(0.0).sqrt();
    let mut points = vec![[0.0, 0.0], [ha, 0.0], [bx, by]];
    let mut ids = vec![HUB.to_owned(), a.clone(), b.clone()];
    // Anything else every board hears (a phone left on a shelf) is placed where its distances to the
    // three boards fit best: three more links across the room.
    let extra: Vec<String> = devices_heard_by_boards(links, &ids)
        .into_iter()
        .filter(|device| ids.iter().all(|board| distance(device, board).is_some()))
        .collect();
    if let [device] = extra.as_slice() {
        // Four corners: RSSI is trusted for order, not distance (a phone sends at its own power), so the
        // pairing that hears itself worst is taken for the two diagonals of a rectangle.
        ids.push(device.clone());
        points = rectangle(&ids, &|x: &str, y: &str| rssi_between(links, rssi, x, y), &distance);
    } else {
        for device in extra {
            let reach: Vec<([f64; 2], f64)> = ids
                .iter()
                .zip(&points)
                .filter_map(|(board, &at)| Some((at, distance(&device, board)? * scale)))
                .collect();
            points.push(trilaterate(&reach));
            ids.push(device);
        }
    }
    let min_x = points.iter().map(|p| p[0]).fold(f64::MAX, f64::min) - WALL_MARGIN_M;
    let min_y = points.iter().map(|p| p[1]).fold(f64::MAX, f64::min) - WALL_MARGIN_M;
    let place = |p: [f64; 2]| [p[0] - min_x, p[1] - min_y, BOARD_HEIGHT_M];
    let width = points.iter().map(|p| p[0]).fold(f64::MIN, f64::max) - min_x + WALL_MARGIN_M;
    let depth = points.iter().map(|p| p[1]).fold(f64::MIN, f64::max) - min_y + WALL_MARGIN_M;
    let mut names: BTreeMap<String, String> =
        nodes.iter().map(|mac| (mac.clone(), format!("ESP32 · {}", mac_tail(mac)))).collect();
    names.insert(HUB.to_owned(), "ESP32 hub".to_owned());
    for device in &ids[3..] {
        names.insert(device.clone(), format!("phone · {}", mac_tail(device)));
    }
    println!(
        "boards placed from their signal strength: hub-{} {ha:.1} m, hub-{} {hb:.1} m, between them {ab:.1} m{}",
        mac_tail(&a),
        mac_tail(&b),
        if ids.len() > 3 { format!(", and {} more device(s)", ids.len() - 3) } else { String::new() }
    );
    Some(RoomFile {
        width: width.max(2.5),
        depth: depth.max(2.5),
        height: 2.8,
        router: place(points[0]),
        boards: ids.iter().zip(&points).skip(1).map(|(id, &p)| (id.clone(), place(p))).collect(),
        furniture: vec![],
        names,
    })
}

fn rssi_between(links: &[String], rssi: &BTreeMap<String, f64>, x: &str, y: &str) -> Option<f64> {
    let heard: Vec<f64> = links
        .iter()
        .filter(|link| {
            let (s, r) = link_ends(link);
            (s == x && r == y) || (s == y && r == x)
        })
        .filter_map(|link| rssi.get(link).copied())
        .collect();
    (!heard.is_empty()).then(|| heard.iter().sum::<f64>() / heard.len() as f64)
}

/// The longest side of four corners, and the most one side may be longer than the other.
const RECTANGLE_LONGEST_M: f64 = 4.0;
const RECTANGLE_SHORTEST_M: f64 = 2.0;
const RECTANGLE_MOST_ASPECT: f64 = 1.5;

/// Four devices as the corners of a rectangle, in `ids` order of the returned points; `ids[3]` is the
/// one that is not a board. Of the three ways
/// to split four corners into two pairs, the one whose pairs hear each other worst is the diagonals;
/// going round the rectangle then alternates between them. Side lengths come from the RSSI distances of
/// opposite sides, averaged, kept within a bedroom.
fn rectangle(
    ids: &[String],
    rssi: &dyn Fn(&str, &str) -> Option<f64>,
    distance: &dyn Fn(&str, &str) -> Option<f64>,
) -> Vec<[f64; 2]> {
    let pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))];
    let heard = |(i, j): (usize, usize)| rssi(&ids[i], &ids[j]).unwrap_or(-100.0);
    let ((p, q), (r, t)) = *pairings
        .iter()
        .min_by(|a, b| (heard(a.0) + heard(a.1)).total_cmp(&(heard(b.0) + heard(b.1))))
        .expect("three pairings");
    // Round the rectangle: p, r, q, t (each diagonal's ends opposite each other).
    let corners = [p, r, q, t];
    let side = |i: usize, j: usize| distance(&ids[corners[i]], &ids[corners[j]]).unwrap_or(3.0);
    // Opposite sides of a rectangle are equal: take the one between two boards when there is one, since
    // the fourth device (a phone) sends at its own power and reads nearer than it is.
    let between_boards = |i: usize, j: usize| corners[i] < 3 && corners[j] < 3;
    let pick = |(i, j): (usize, usize), (k, l): (usize, usize)| match (between_boards(i, j), between_boards(k, l)) {
        (true, false) => side(i, j),
        (false, true) => side(k, l),
        _ => (side(i, j) + side(k, l)) / 2.0,
    };
    let mut along = pick((0, 1), (2, 3));
    let mut across = pick((1, 2), (3, 0));
    let longest = along.max(across);
    let scale = if longest > RECTANGLE_LONGEST_M { RECTANGLE_LONGEST_M / longest } else { 1.0 };
    along = (along * scale).max(RECTANGLE_SHORTEST_M);
    across = (across * scale).max(RECTANGLE_SHORTEST_M);
    if along > across * RECTANGLE_MOST_ASPECT {
        along = across * RECTANGLE_MOST_ASPECT;
    } else if across > along * RECTANGLE_MOST_ASPECT {
        across = along * RECTANGLE_MOST_ASPECT;
    }
    let at = [[0.0, 0.0], [along, 0.0], [along, across], [0.0, across]];
    let mut points = vec![[0.0, 0.0]; 4];
    for (corner, &id) in corners.iter().enumerate() {
        points[id] = at[corner];
    }
    points
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

/// The point whose distances to the anchors fit the measured ones best (a grid search, then a finer one).
fn trilaterate(reach: &[([f64; 2], f64)]) -> [f64; 2] {
    let misfit = |p: [f64; 2]| -> f64 {
        reach
            .iter()
            .map(|(at, d)| ((p[0] - at[0]).hypot(p[1] - at[1]) - d).powi(2))
            .sum()
    };
    let mut best = reach[0].0;
    for step in [0.1, 0.02] {
        let centre = best;
        let span = if step > 0.05 { LONGEST_SPAN_M * 1.5 } else { 0.2 };
        let n = (span / step) as i32;
        for i in -n..=n {
            for j in -n..=n {
                let p = [centre[0] + f64::from(i) * step, centre[1] + f64::from(j) * step];
                if misfit(p) < misfit(best) {
                    best = p;
                }
            }
        }
    }
    best
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
    fn four_devices_become_a_rectangle_with_the_weakest_pairs_as_diagonals() {
        // Measured on 2026-11-02: hub, 30:c0, d9:bc and a phone near the corners of a bedroom.
        let ids: Vec<String> = ["router", "a", "b", "phone"].map(String::from).to_vec();
        let measured = |x: &str, y: &str| -> Option<f64> {
            let mut pair = [x, y];
            pair.sort();
            Some(match pair {
                ["a", "router"] | ["b", "router"] => -57.5,
                ["a", "b"] => -63.5,
                ["phone", "router"] => -55.0,
                ["a", "phone"] => -51.0,
                ["b", "phone"] => -40.0,
                _ => return None,
            })
        };
        let distance = |x: &str, y: &str| measured(x, y).map(metres_from_rssi);
        let points = rectangle(&ids, &measured, &distance);
        let apart = |i: usize, j: usize| (points[i][0] - points[j][0]).hypot(points[i][1] - points[j][1]);
        let diagonal = apart(0, 3);
        assert!((apart(1, 2) - diagonal).abs() < 1e-9, "a and b are the other diagonal");
        for (i, j) in [(0, 1), (0, 2), (3, 1), (3, 2)] {
            assert!(apart(i, j) < diagonal, "{i}-{j} is a side");
        }
    }
}
