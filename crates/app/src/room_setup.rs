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
    /// The links the engine reads, in its order, as named on the command line.
    pub links: Vec<String>,
    /// The same links as the window shows them.
    pub link_names: Vec<String>,
}

const HUB: &str = "router";

/// Both ends of a link, sender first. `<board>@<node>><sender>` is measured by a node that joined the
/// hub, from the hub (`hub`) or from another device; `<board>@<device>` by the hub from a device;
/// a bare `<board>` by the board from the router (which `HUB` then stands for).
fn link_ends(link: &str) -> (String, String) {
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
fn nodes_among(links: &[String]) -> BTreeSet<String> {
    links
        .iter()
        .filter_map(|link| crate::serial::split_link_name(link).1)
        .filter_map(|tag| tag.split_once('>').map(|(node, _)| node.to_owned()))
        .collect()
}

/// Links whose ends are not in the room are left out. Without a room file see `default_room`.
pub fn room_engine(links: &[String], room_file: Option<&Path>) -> anyhow::Result<LiveRoom> {
    let room = match room_file {
        Some(path) => serde_json::from_str::<RoomFile>(&std::fs::read_to_string(path)?)?,
        None => default_room(links),
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
        links: kept,
        link_names,
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
        node_names,
        links: room.links.iter().map(|(a, b)| format!("{a}-{b}")).collect(),
        link_names,
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

fn mac_tail(mac: &str) -> &str {
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
        let room = room_engine(&mesh_links(), None).unwrap();
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
        let room = room_engine(&links, None).unwrap();
        assert_eq!(room.links, links);
        assert_eq!(room.link_names[0], "Device 1 · aa:01");
    }
}
