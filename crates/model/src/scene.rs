//! The `scene` object the dashboard draws (`research/live.py::room_scene_json` / `hall_scene_json`).

use serde::{Deserialize, Serialize};

use crate::numeric::round_to;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub kind: String,
    #[serde(rename = "W")]
    pub width: f64,
    #[serde(rename = "D")]
    pub depth: f64,
    #[serde(rename = "H")]
    pub height: f64,
    pub tiers: Option<serde_json::Value>,
    pub seats: Vec<[f64; 3]>,
    pub nodes: Vec<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub node_roles: Option<Vec<String>>,
    pub links: Vec<[usize; 2]>,
    pub furniture: Vec<serde_json::Value>,
}

pub fn round_point(point: [f64; 3], digits: i32) -> [f64; 3] {
    point.map(|v| round_to(v, digits))
}

impl Scene {
    /// A room: node 0 is the router, every other node an ESP32 receiving it.
    pub fn room(
        width: f64,
        depth: f64,
        height: f64,
        nodes: &[[f64; 3]],
        furniture: Vec<serde_json::Value>,
        roles: Vec<String>,
    ) -> Self {
        Scene {
            kind: "room".into(),
            width,
            depth,
            height,
            tiers: None,
            seats: vec![],
            nodes: nodes.iter().map(|&n| round_point(n, 3)).collect(),
            node_roles: Some(roles),
            links: (1..nodes.len()).map(|k| [0, k]).collect(),
            furniture,
        }
    }
}
