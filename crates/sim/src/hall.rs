//! Lecture halls (flat or tiered), seated students, the lecturer and occupancy patterns.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::bodies::Body;
use crate::motion::{standing_body, wander, Gait, TargetArea};
use crate::paths::{Enclosure, RoomShape};
use crate::random::{
    chance, integer, normal, ornstein_uhlenbeck, poisson, uniform, unit, weighted_index, SimRng,
};
use crate::scatter::Scatterer;
use crate::vec3::Vec3;

/// Row `r` of seats sits `r × rise` above the front floor; rows start at `first_row_y`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tiers {
    pub first_row_y: f64,
    pub rise: f64,
    pub row_pitch: f64,
    /// The lecturer zone is `y in 0..front`.
    pub front: f64,
}

impl Tiers {
    /// Floor height at depth `y`: the nearest row's tier, never below 0.
    pub fn floor_z(&self, y: f64) -> f64 {
        if self.rise <= 0.0 {
            return 0.0;
        }
        let row = ((y - self.first_row_y) / self.row_pitch + 0.5).floor();
        row.max(0.0) * self.rise
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodePlacement {
    /// Every board on the walls at seated-chest height.
    Walls,
    /// About a third of the boards high up (projector shelf, upper wall).
    WallsAndHigh,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hall {
    pub enclosure: Enclosure,
    pub nodes: Vec<Vec3>,
    pub links: Vec<(usize, usize)>,
    /// Seat surface points (x, y, floor z of the row).
    pub seats: Vec<Vec3>,
    /// (row, column) of every seat.
    pub seat_row_col: Vec<(usize, usize)>,
    pub tiers: Tiers,
    pub tiered: bool,
    pub aisles_x: Vec<f64>,
    pub placement: NodePlacement,
    /// The empty hall's chairs (first `seats.len()` entries, same order) and other objects.
    pub clutter: Vec<Scatterer>,
}

impl Hall {
    pub fn shape(&self) -> RoomShape {
        self.enclosure.shape
    }

    /// Where a seated student's chest is, for the 3-D view and seat models.
    pub fn chest_points(&self) -> Vec<Vec3> {
        self.seats
            .iter()
            .map(|s| [s[0], s[1], s[2] + 1.0])
            .collect()
    }
}

/// Optional overrides of `random_hall`'s random choices.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HallOptions {
    pub nodes: usize,
    pub tiered: Option<bool>,
    pub placement: Option<NodePlacement>,
    pub width: Option<f64>,
    pub depth: Option<f64>,
}

impl Default for HallOptions {
    fn default() -> Self {
        HallOptions {
            nodes: 12,
            tiered: None,
            placement: None,
            width: None,
            depth: None,
        }
    }
}

/// Port of scenes3d.py::random_hall3.
pub fn random_hall(rng: &mut SimRng, options: &HallOptions) -> Hall {
    let width = options.width.unwrap_or_else(|| uniform(rng, 9.0, 18.0));
    let depth = options.depth.unwrap_or_else(|| uniform(rng, 8.0, 16.0));
    let room_height = uniform(rng, 3.2, 5.0);
    let tiered = options.tiered.unwrap_or_else(|| chance(rng, 0.6));
    let front = uniform(rng, 2.5, 3.5);
    let row_pitch = uniform(rng, 0.85, 1.10);
    let seat_pitch = uniform(rng, 0.55, 0.70);
    let rise = if tiered {
        uniform(rng, 0.15, 0.40)
    } else {
        0.0
    };
    let tiers = Tiers {
        first_row_y: front + 0.9,
        rise,
        row_pitch,
        front,
    };
    let aisles_x = random_aisles(rng, width);
    let rows = numpy_arange(tiers.first_row_y, depth - 0.6, row_pitch);
    let (seats, seat_row_col) = seat_grid(&rows, width, seat_pitch, rise, &aisles_x);
    let wall_gains = [0; 4].map(|_| uniform(rng, -0.5, -0.25));
    let shape = RoomShape {
        width,
        depth,
        height: room_height + rows.len() as f64 * rise,
    };
    let placement = options.placement.unwrap_or_else(|| {
        if chance(rng, 0.5) {
            NodePlacement::Walls
        } else {
            NodePlacement::WallsAndHigh
        }
    });
    let nodes = wall_nodes(rng, &shape, &tiers, options.nodes, placement);
    let clutter = hall_clutter(rng, &shape, &seats);
    Hall {
        enclosure: Enclosure {
            shape,
            floor_reflects: !tiered,
            wall_gains,
        },
        links: all_pairs(nodes.len()),
        nodes,
        seats,
        seat_row_col,
        tiers,
        tiered,
        aisles_x,
        placement,
        clutter,
    }
}

/// numpy `arange(start, stop, step)`: `ceil((stop - start) / step)` values `start + i·step`.
pub fn numpy_arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let count = ((stop - start) / step).ceil().max(0.0) as usize;
    (0..count).map(|i| start + i as f64 * step).collect()
}

fn random_aisles(rng: &mut SimRng, width: f64) -> Vec<f64> {
    let (x0, x1) = (0.9, width - 0.9);
    let two_aisles = width > 11.0 && integer(rng, 1, 3) == 2;
    let cuts = if two_aisles {
        vec![0.33 + normal(rng, 0.0, 0.03), 0.67 + normal(rng, 0.0, 0.03)]
    } else {
        vec![uniform(rng, 0.3, 0.7)]
    };
    cuts.into_iter().map(|cut| x0 + cut * (x1 - x0)).collect()
}

/// Seats every `seat_pitch` across each row, skipping 1 m wide aisles.
fn seat_grid(
    rows: &[f64],
    width: f64,
    seat_pitch: f64,
    rise: f64,
    aisles_x: &[f64],
) -> (Vec<Vec3>, Vec<(usize, usize)>) {
    let columns = numpy_arange(0.9, width - 0.9 + 1e-6, seat_pitch);
    let mut seats = Vec::new();
    let mut row_col = Vec::new();
    for (row, &y) in rows.iter().enumerate() {
        let mut column = 0;
        for &x in &columns {
            if aisles_x.iter().any(|a| (x - a).abs() < 0.5) {
                continue;
            }
            seats.push([x, y, row as f64 * rise]);
            row_col.push((row, column));
            column += 1;
        }
    }
    (seats, row_col)
}

fn all_pairs(nodes: usize) -> Vec<(usize, usize)> {
    (0..nodes)
        .flat_map(|i| (i + 1..nodes).map(move |j| (i, j)))
        .collect()
}

/// Boards spread evenly (with jitter) around the perimeter, 1.0-1.3 m above the local floor.
fn wall_nodes(
    rng: &mut SimRng,
    shape: &RoomShape,
    tiers: &Tiers,
    count: usize,
    placement: NodePlacement,
) -> Vec<Vec3> {
    let (w, d) = (shape.width, shape.depth);
    let perimeter = 2.0 * (w + d);
    let offset = uniform(rng, 0.2, 0.8);
    let spacing = perimeter / count as f64;
    let positions: Vec<f64> = (0..count)
        .map(|i| (i as f64 + offset) * spacing)
        .collect::<Vec<_>>()
        .into_iter()
        .map(|s| (s + normal(rng, 0.0, 0.12 * spacing)).rem_euclid(perimeter))
        .collect();
    let points: Vec<[f64; 2]> = positions
        .iter()
        .map(|&s| perimeter_point(s, w, d))
        .collect();
    let floor: Vec<f64> = points.iter().map(|p| tiers.floor_z(p[1])).collect();
    let mut heights: Vec<f64> = floor.iter().map(|&z| z + uniform(rng, 1.0, 1.3)).collect();
    if placement == NodePlacement::WallsAndHigh {
        let high: Vec<bool> = (0..count).map(|_| chance(rng, 0.35)).collect();
        for k in (0..count).filter(|&k| high[k]) {
            heights[k] = floor[k] + uniform(rng, 2.2, 2.8);
        }
    }
    points
        .iter()
        .zip(heights)
        .map(|(p, z)| [p[0], p[1], z])
        .collect()
}

/// The point at arc length `s` along the walls, starting at (0, 0.1) and running anticlockwise.
fn perimeter_point(s: f64, w: f64, d: f64) -> [f64; 2] {
    if s < w {
        [s, 0.1]
    } else if s < w + d {
        [w - 0.1, s - w]
    } else if s < 2.0 * w + d {
        [w - (s - w - d), d - 0.1]
    } else {
        [0.1, d - (s - 2.0 * w - d)]
    }
}

fn hall_clutter(rng: &mut SimRng, shape: &RoomShape, seats: &[Vec3]) -> Vec<Scatterer> {
    let mut clutter: Vec<Scatterer> = seats
        .iter()
        .map(|s| Scatterer {
            position: [s[0], s[1], s[2] + 0.45],
            rcs: uniform(rng, 0.005, 0.03),
        })
        .collect();
    for _ in 0..integer(rng, 15, 40) {
        let position = [
            uniform(rng, 0.3, shape.width - 0.3),
            uniform(rng, 0.3, shape.depth - 0.3),
            uniform(rng, 0.0, 2.5),
        ];
        clutter.push(Scatterer {
            position,
            rcs: uniform(rng, 0.02, 0.4),
        });
    }
    clutter
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OccupancyPattern {
    Uniform,
    Front,
    Back,
    Clusters,
}

/// Who sits where: a fill fraction (random bucket if `None`) and a spatial pattern.
/// Port of sim.py::sample_occupancy.
pub fn sample_occupancy(
    rng: &mut SimRng,
    hall: &Hall,
    fill_fraction: Option<f64>,
) -> (Vec<bool>, OccupancyPattern) {
    let seats = hall.seats.len();
    let fraction = fill_fraction.unwrap_or_else(|| random_fill_fraction(rng));
    let occupied_count = (fraction * seats as f64).round_ties_even() as usize;
    let pattern = [
        OccupancyPattern::Uniform,
        OccupancyPattern::Front,
        OccupancyPattern::Back,
        OccupancyPattern::Clusters,
    ][integer(rng, 0, 4)];
    let last_row = hall
        .seat_row_col
        .iter()
        .map(|rc| rc.0)
        .max()
        .unwrap_or(0)
        .max(1) as f64;
    let mut score: Vec<f64> = (0..seats).map(|_| unit(rng)).collect();
    match pattern {
        OccupancyPattern::Front | OccupancyPattern::Back => {
            let sign = if pattern == OccupancyPattern::Front {
                -0.8
            } else {
                0.8
            };
            for (value, rc) in score.iter_mut().zip(&hall.seat_row_col) {
                *value += sign * rc.0 as f64 / last_row;
            }
        }
        OccupancyPattern::Clusters => add_cluster_scores(rng, hall, &mut score),
        OccupancyPattern::Uniform => {}
    }
    let mut order: Vec<usize> = (0..seats).collect();
    order.sort_by(|&i, &j| score[j].total_cmp(&score[i]));
    let mut occupied = vec![false; seats];
    for &seat in order.iter().take(occupied_count) {
        occupied[seat] = true;
    }
    (occupied, pattern)
}

/// Empty 8 %, sparse 2-30 %, half 30-70 %, full 70-100 %.
fn random_fill_fraction(rng: &mut SimRng) -> f64 {
    match weighted_index(rng, &[0.08, 0.27, 0.35, 0.30]) {
        0 => 0.0,
        1 => uniform(rng, 0.02, 0.3),
        2 => uniform(rng, 0.3, 0.7),
        _ => uniform(rng, 0.7, 1.0),
    }
}

fn add_cluster_scores(rng: &mut SimRng, hall: &Hall, score: &mut [f64]) {
    for _ in 0..integer(rng, 2, 6) {
        let centre = hall.seats[integer(rng, 0, hall.seats.len())];
        let radius = uniform(rng, 1.0, 3.0);
        for (value, seat) in score.iter_mut().zip(&hall.seats) {
            let d2 = (seat[0] - centre[0]).powi(2) + (seat[1] - centre[1]).powi(2);
            *value += 1.5 * (-d2 / (2.0 * radius * radius)).exp();
        }
    }
}

/// Students in `seats` facing the front, with sway, breathing and Poisson fidgets.
/// Port of scenes3d.py::seated_bodies; `restless` scales the fidget rate.
pub fn seated_students(
    rng: &mut SimRng,
    seats: &[Vec3],
    frames: usize,
    fs: f64,
    restless: f64,
) -> Vec<Body> {
    seats
        .iter()
        .map(|&seat| seated_student(rng, seat, frames, fs, restless))
        .collect()
}

fn seated_student(rng: &mut SimRng, seat: Vec3, frames: usize, fs: f64, restless: f64) -> Body {
    let settle = [normal(rng, 0.0, 0.04), normal(rng, 0.0, 0.04)];
    let sway_x = ornstein_uhlenbeck(rng, frames, 0.004, 3.0, fs);
    let sway_y = ornstein_uhlenbeck(rng, frames, 0.004, 3.0, fs);
    let mut centre: Vec<[f64; 2]> = (0..frames)
        .map(|t| {
            [
                seat[0] + settle[0] + sway_x[t],
                seat[1] + settle[1] + sway_y[t],
            ]
        })
        .collect();
    let floor = seat[2];
    let breath_hz = uniform(rng, 0.2, 0.33);
    let breath_amplitude = uniform(rng, 0.003, 0.006);
    let breath_phase = uniform(rng, 0.0, 6.3);
    let breath: Vec<f64> = (0..frames)
        .map(|t| breath_amplitude * (2.0 * PI * breath_hz * t as f64 / fs + breath_phase).sin())
        .collect();
    let hand_offset = [uniform(rng, -0.25, 0.25), -uniform(rng, 0.15, 0.35)];
    let mut limb: Vec<Vec3> = centre
        .iter()
        .map(|c| [c[0] + hand_offset[0], c[1] + hand_offset[1], floor + 0.8])
        .collect();
    let fidget_rate_hz = restless * normal(rng, (1.0f64 / 20.0).ln(), 0.7).exp();
    for _ in 0..poisson(rng, fidget_rate_hz * frames as f64 / fs) {
        add_fidget(rng, &mut centre, &mut limb, fs);
    }
    let chest = centre
        .iter()
        .zip(&breath)
        .map(|(c, b)| [c[0], c[1] - b, floor + 1.0])
        .collect();
    Body {
        z_bottom: vec![floor + 0.40; frames],
        z_top: vec![floor + 1.30; frames],
        half_width: vec![0.21; frames],
        half_depth: breath.iter().map(|b| 0.13 + b).collect(),
        yaw: vec![0.0; frames],
        absorption: uniform(rng, 0.75, 0.97),
        chest,
        limb,
        centre,
        chest_rcs: uniform(rng, 0.1, 0.5),
        limb_rcs: uniform(rng, 0.02, 0.1),
    }
}

/// A 0.8-4 s fidget: the torso shifts 2-15 cm and back, a hand oscillates at 0.5-2 Hz.
fn add_fidget(rng: &mut SimRng, centre: &mut [[f64; 2]], limb: &mut [Vec3], fs: f64) {
    let frames = centre.len();
    let first = integer(rng, 0, frames);
    let duration = (uniform(rng, 0.8, 4.0) * fs) as usize;
    let last = frames.min(first + duration);
    if last - first < 2 {
        return;
    }
    let angle = uniform(rng, 0.0, 2.0 * PI);
    let shift = uniform(rng, 0.02, 0.15);
    let hand_amplitude = uniform(rng, 0.05, 0.3);
    let hand_hz = uniform(rng, 0.5, 2.0);
    let hand_angle = uniform(rng, 0.0, 2.0 * PI);
    for t in first..last {
        let phase = (t - first) as f64 / duration as f64;
        let lean = (PI * phase).sin().powi(2) * shift;
        centre[t][0] += lean * angle.cos();
        centre[t][1] += lean * angle.sin();
        let swing = hand_amplitude
            * (2.0 * PI * hand_hz * (t - first) as f64 / fs).sin()
            * (PI * phase).sin();
        limb[t][0] += swing * hand_angle.cos();
        limb[t][1] += swing * hand_angle.sin();
    }
}

/// With probability 0.75 a lecturer walks around the front zone. Port of scenes3d.py::lecturer.
pub fn lecturer(rng: &mut SimRng, hall: &Hall, frames: usize, fs: f64) -> Option<Body> {
    if unit(rng) > 0.75 {
        return None;
    }
    let area = TargetArea {
        x_range: (1.0, hall.shape().width - 1.0),
        y_range: (0.6, hall.tiers.front - 0.3),
    };
    let start = area.sample(rng);
    let gait = Gait {
        speed_m_s: uniform(rng, 0.3, 0.9),
        arrive_within_m: 0.1,
        mean_pause_s: 4.0,
    };
    let track = wander(rng, start, start, &area, &gait, frames, fs);
    Some(standing_body(rng, &track, fs, 0.0))
}
