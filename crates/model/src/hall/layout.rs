use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::scene::{round_point, Scene};
use crate::{ModelError, Result};

const CHEST_HEIGHT_M: f64 = 1.0;
const DEFAULT_NODE_HEIGHT_M: f64 = 1.1;

/// Hall geometry: nodes (by mesh id), seats, chest points 1 m above the seats, and every node pair as a link.
#[derive(Debug, Clone, PartialEq)]
pub struct HallLayout {
    pub width: f64,
    pub depth: f64,
    pub height: f64,
    pub front: f64,
    pub ids: Vec<i64>,
    pub nodes: Vec<[f64; 3]>,
    pub seats: Vec<[f64; 3]>,
    pub links: Vec<(usize, usize)>,
    pub tiers: Option<Value>,
}

#[derive(Deserialize)]
struct SeatGrid {
    rows: usize,
    cols: usize,
    x0: f64,
    y0: f64,
    seat_pitch: f64,
    row_pitch: f64,
    #[serde(default)]
    row_rise: f64,
    aisle_w: Option<f64>,
    #[serde(default)]
    aisles_after_col: Vec<usize>,
}

#[derive(Deserialize)]
struct HallConfig {
    #[serde(rename = "W")]
    width: f64,
    #[serde(rename = "D")]
    depth: f64,
    #[serde(rename = "H")]
    height: Option<f64>,
    front: Option<f64>,
    nodes: std::collections::BTreeMap<String, Vec<f64>>,
    seats: Option<Vec<Vec<f64>>>,
    grid: Option<SeatGrid>,
}

fn padded_point(values: &[f64], default_z: f64) -> [f64; 3] {
    [
        values[0],
        values[1],
        values.get(2).copied().unwrap_or(default_z),
    ]
}

fn all_pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n)
        .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
        .collect()
}

fn grid_seats(grid: &SeatGrid) -> Vec<[f64; 3]> {
    let aisle = grid.aisle_w.unwrap_or(1.0);
    let mut seats = Vec::with_capacity(grid.rows * grid.cols);
    for r in 0..grid.rows {
        for c in 0..grid.cols {
            let aisles_left = grid.aisles_after_col.iter().filter(|&&a| c > a).count() as f64;
            let x = grid.x0 + c as f64 * grid.seat_pitch + aisle * aisles_left;
            seats.push([
                x,
                grid.y0 + r as f64 * grid.row_pitch,
                r as f64 * grid.row_rise,
            ]);
        }
    }
    seats
}

impl HallLayout {
    /// Reads a hall description (`research/live.py::load_hall`): explicit `seats` or a seat `grid`.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| ModelError::Io {
            path: path.into(),
            source,
        })?;
        let config: HallConfig =
            serde_json::from_str(&text).map_err(|source| ModelError::Json {
                path: path.into(),
                source,
            })?;
        Self::from_config(config)
    }

    pub fn from_config_json(text: &str) -> Result<Self> {
        let config: HallConfig = serde_json::from_str(text).map_err(|source| ModelError::Json {
            path: "<string>".into(),
            source,
        })?;
        Self::from_config(config)
    }

    fn from_config(config: HallConfig) -> Result<Self> {
        let mut ids: Vec<i64> = config
            .nodes
            .keys()
            .map(|k| {
                k.parse()
                    .map_err(|_| ModelError::InvalidModel(format!("node id {k} is not an integer")))
            })
            .collect::<Result<_>>()?;
        ids.sort_unstable();
        let nodes = ids
            .iter()
            .map(|id| padded_point(&config.nodes[&id.to_string()], DEFAULT_NODE_HEIGHT_M))
            .collect();
        let (seats, tiers) = match (&config.seats, &config.grid) {
            (Some(seats), _) => (seats.iter().map(|s| padded_point(s, 0.0)).collect(), None),
            (None, Some(grid)) => {
                let tiers = serde_json::json!({
                    "y0": grid.y0, "rise": grid.row_rise, "pitch": grid.row_pitch, "front": config.front.unwrap_or(0.0)
                });
                (grid_seats(grid), Some(tiers))
            }
            (None, None) => {
                return Err(ModelError::InvalidModel(
                    "hall needs `seats` or `grid`".into(),
                ))
            }
        };
        let seats: Vec<[f64; 3]> = seats;
        let highest_seat = seats.iter().map(|s| s[2]).fold(f64::NEG_INFINITY, f64::max);
        Ok(HallLayout {
            width: config.width,
            depth: config.depth,
            height: config.height.unwrap_or(3.5 + highest_seat),
            front: config.front.unwrap_or(0.0),
            links: all_pairs(ids.len()),
            ids,
            nodes,
            seats,
            tiers,
        })
    }

    /// Where a seated person's chest is: the point the links' Fresnel zones are measured to.
    pub fn chest_points(&self) -> Vec<[f64; 3]> {
        self.seats
            .iter()
            .map(|s| [s[0], s[1], s[2] + CHEST_HEIGHT_M])
            .collect()
    }

    pub fn scene(&self) -> Scene {
        Scene {
            kind: "hall".into(),
            width: self.width,
            depth: self.depth,
            height: self.height,
            tiers: self.tiers.clone(),
            seats: self.seats.iter().map(|&s| round_point(s, 3)).collect(),
            nodes: self.nodes.iter().map(|&n| round_point(n, 3)).collect(),
            node_roles: None,
            links: self.links.iter().map(|&(a, b)| [a, b]).collect(),
            furniture: vec![],
        }
    }
}
