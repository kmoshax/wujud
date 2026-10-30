use std::path::{Path, PathBuf};

use serde::Deserialize;
use wujud_model::gbdt::Gbdt;
use wujud_model::room::{activity_model_file, RoomEngine};
use wujud_model::scene::Scene;
use wujud_sim::room::{Furniture, Room};

/// research/room.example.json: metres, origin in a corner, z up.
/// `router` is the node every link shares (the board, in the room firmware's setup); `boards` are the
/// links' other ends, keyed by link name, by the device's MAC, or by the board alone.
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
    pub boards: std::collections::BTreeMap<String, [f64; 3]>,
    /// What to call each node in the 3D view, keyed like `boards`; `router` names the shared node.
    #[serde(default)]
    pub names: std::collections::BTreeMap<String, String>,
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
    /// One per scene node: the shared node first, then one per link.
    pub node_names: Vec<String>,
}

/// Without a room file the boards are spread along the walls of a 4 x 3.5 m room, like research/live.py.
pub fn room_engine(ports: &[String], room_file: Option<&Path>) -> anyhow::Result<LiveRoom> {
    let room = match room_file {
        Some(path) => serde_json::from_str::<RoomFile>(&std::fs::read_to_string(path)?)?,
        None => default_room(ports),
    };
    let mut nodes = vec![room.router];
    let mut node_names = vec![room.names.get("router").cloned().unwrap_or("router".into())];
    for port in ports {
        let key = room_file_keys(port)
            .into_iter()
            .find(|key| room.boards.contains_key(*key))
            .ok_or_else(|| anyhow::anyhow!("{port} is not in the room file's boards"))?;
        nodes.push(room.boards[key]);
        node_names.push(room.names.get(key).cloned().unwrap_or(key.to_owned()));
    }
    let roles = std::iter::once("router".to_owned())
        .chain(ports.iter().map(|_| "esp".to_owned()))
        .collect();
    let scene = Scene::room(
        room.width,
        room.depth,
        room.height,
        &nodes,
        room.furniture,
        roles,
    );
    let links = (0..ports.len()).map(|k| (0, k + 1)).collect();
    let model = Gbdt::load(&models_dir().join(activity_model_file(ports.len())))?;
    Ok(LiveRoom {
        engine: RoomEngine::new(scene, links, "LIVE", model, nodes),
        node_names,
    })
}

fn room_file_keys(link: &str) -> Vec<&str> {
    let (board, device) = crate::serial::split_link_name(link);
    [Some(link), device, Some(board)]
        .into_iter()
        .flatten()
        .collect()
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
    Ok(LiveRoom {
        engine: RoomEngine::new(
            scene,
            room.links.clone(),
            "SIMULATION",
            model,
            room.nodes.clone(),
        ),
        node_names: std::iter::once("router".to_owned())
            .chain((1..=boards).map(|k| format!("ESP32 {k}")))
            .collect(),
    })
}

fn furniture_json(item: &Furniture) -> serde_json::Value {
    serde_json::json!({ "kind": item.kind, "x": item.x, "y": item.y, "z": item.z, "w": item.w, "d": item.d, "h": item.h })
}

fn default_room(ports: &[String]) -> RoomFile {
    let wall_spots = [
        [3.9, 1.75, 0.8],
        [2.0, 0.1, 1.0],
        [2.0, 3.4, 1.0],
        [0.1, 0.5, 1.2],
    ];
    RoomFile {
        width: 4.0,
        depth: 3.5,
        height: 2.8,
        router: [0.1, 1.75, 1.0],
        boards: ports
            .iter()
            .cloned()
            .zip(wall_spots.iter().cycle().copied())
            .collect(),
        furniture: vec![],
        names: Default::default(),
    }
}
