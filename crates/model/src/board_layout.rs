//! Where the boards stand, from how strongly they hear each other: every link fitted at once to the
//! log-distance path loss, with each board's own gain and the room's exponent unknown too.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// One direction of one link: `from` sent, `to` heard it at `db` (its median RSSI).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Heard {
    pub from: usize,
    pub to: usize,
    pub db: f64,
}

/// What is known before any packet: the radio, a room's size, how far a board's gain may stray.
#[derive(Clone, Debug, PartialEq)]
pub struct Prior {
    pub rssi_at_1m: f64,
    pub exponent: f64,
    pub exponent_spread: f64,
    /// Per node: how far its antenna and amplifier may be from the nominal radio (a phone more).
    pub gain_spread_db: Vec<f64>,
    /// How far the boards' gains may stray all together: they are one model, measured at one metre.
    pub common_gain_spread_db: f64,
    pub residual_db: f64,
    /// A room's usual side and how far it strays from it.
    pub room_side_m: f64,
    pub room_side_spread_m: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub points: Vec<[f64; 2]>,
    pub gains_db: Vec<f64>,
    pub exponent: f64,
    /// How far the links are from the fitted layout, root mean square.
    pub residual_db: f64,
    /// The fit's cost with its priors, to compare one kind of layout with another.
    pub cost: f64,
}

/// Boards closer than this read as this close: the antennas' own size.
const NEAREST_M: f64 = 0.15;
/// Past this many residual spreads a link is taken as blocked or reflected, and counts less (Huber).
const OUTLIER_SPREADS: f64 = 1.5;
const WALL_STARTS: usize = 200;
const SMALLEST_ROOM_M: f64 = 1.5;
const CORNER_GRID_STEPS: usize = 60;

/// Boards along the walls of a rectangular room: the room's width and depth and where along its
/// walls each board stands are searched; gains and exponent follow from them (`fit_radio`).
/// None when a node is heard by nobody.
pub fn solve_on_walls(nodes: usize, heard: &[Heard], prior: &Prior) -> Option<Layout> {
    if nodes < 2 || !every_node_heard(nodes, heard) {
        return None;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(nodes as u64 * 104_729 + heard.len() as u64);
    let largest = largest_side(prior);
    let mut best: Option<(f64, Walls)> = None;
    for _ in 0..WALL_STARTS {
        let width = rng.gen_range(SMALLEST_ROOM_M..largest);
        let depth = rng.gen_range(SMALLEST_ROOM_M..largest);
        let along = (0..nodes)
            .map(|_| rng.gen_range(0.0..2.0 * (width + depth)))
            .collect();
        let walls = refine(
            Walls {
                width,
                depth,
                along,
            },
            heard,
            prior,
        );
        let cost = room_cost(&walls.points(), walls.width, walls.depth, heard, prior);
        if best.as_ref().is_none_or(|(c, _)| cost < *c) {
            best = Some((cost, walls));
        }
    }
    let (cost, walls) = best?;
    Some(layout_of(walls.points(), cost, heard, prior))
}

/// Four boards in the four corners of a rectangular room: only the room's width and depth and which
/// board is in which corner are unknown. Node 0 is put in the corner at the origin.
pub fn solve_in_corners(heard: &[Heard], prior: &Prior) -> Option<Layout> {
    const ORDERS: [[usize; 3]; 6] = [
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [2, 3, 1],
        [3, 1, 2],
        [3, 2, 1],
    ];
    if !every_node_heard(4, heard) {
        return None;
    }
    let largest = largest_side(prior);
    let side_at = |k: usize| {
        SMALLEST_ROOM_M + (largest - SMALLEST_ROOM_M) * k as f64 / CORNER_GRID_STEPS as f64
    };
    let mut best: Option<(f64, Vec<[f64; 2]>)> = None;
    for order in ORDERS {
        for i in 0..=CORNER_GRID_STEPS {
            for j in 0..=CORNER_GRID_STEPS {
                let (width, depth) = (side_at(i), side_at(j));
                let mut points = vec![[0.0, 0.0]; 4];
                for (&node, corner) in
                    order
                        .iter()
                        .zip([[width, 0.0], [width, depth], [0.0, depth]])
                {
                    points[node] = corner;
                }
                let cost = room_cost(&points, width, depth, heard, prior);
                if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                    best = Some((cost, points));
                }
            }
        }
    }
    let (cost, points) = best?;
    Some(layout_of(points, cost, heard, prior))
}

fn every_node_heard(nodes: usize, heard: &[Heard]) -> bool {
    (0..nodes).all(|i| heard.iter().any(|h| h.from == i || h.to == i))
}

fn largest_side(prior: &Prior) -> f64 {
    prior.room_side_m + 2.0 * prior.room_side_spread_m
}

fn layout_of(points: Vec<[f64; 2]>, cost: f64, heard: &[Heard], prior: &Prior) -> Layout {
    let (gains_db, exponent) = fit_radio(&points, heard, prior);
    let sum: f64 = heard
        .iter()
        .map(|h| (h.db - predicted_db(&points, &gains_db, exponent, h, prior)).powi(2))
        .sum();
    let residual_db = (sum / heard.len() as f64).sqrt();
    Layout {
        points,
        gains_db,
        exponent,
        residual_db,
        cost,
    }
}

#[derive(Clone, Debug)]
struct Walls {
    width: f64,
    depth: f64,
    /// How far along the walls each board stands, anticlockwise from the corner at the origin.
    along: Vec<f64>,
}

impl Walls {
    fn points(&self) -> Vec<[f64; 2]> {
        self.along.iter().map(|&s| self.point(s)).collect()
    }

    fn point(&self, s: f64) -> [f64; 2] {
        let (w, d) = (self.width, self.depth);
        let s = s.rem_euclid(2.0 * (w + d));
        if s < w {
            [s, 0.0]
        } else if s < w + d {
            [w, s - w]
        } else if s < 2.0 * w + d {
            [2.0 * w + d - s, d]
        } else {
            [0.0, 2.0 * (w + d) - s]
        }
    }
}

/// Coordinate descent on the room's size and the boards' places along its walls.
fn refine(mut walls: Walls, heard: &[Heard], prior: &Prior) -> Walls {
    let largest = largest_side(prior);
    let cost_of = |w: &Walls| room_cost(&w.points(), w.width, w.depth, heard, prior);
    let mut cost = cost_of(&walls);
    let mut step = 0.5;
    while step > 0.02 {
        let mut improved = false;
        for k in 0..walls.along.len() + 2 {
            for sign in [-1.0, 1.0] {
                let mut trial = walls.clone();
                match k {
                    0 => trial.width = (trial.width + sign * step).clamp(SMALLEST_ROOM_M, largest),
                    1 => trial.depth = (trial.depth + sign * step).clamp(SMALLEST_ROOM_M, largest),
                    _ => trial.along[k - 2] += sign * step,
                }
                let trial_cost = cost_of(&trial);
                if trial_cost < cost {
                    (walls, cost, improved) = (trial, trial_cost, true);
                }
            }
        }
        if !improved {
            step /= 2.0;
        }
    }
    walls
}

fn separation(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + NEAREST_M * NEAREST_M).sqrt()
}

fn predicted_db(
    points: &[[f64; 2]],
    gains_db: &[f64],
    exponent: f64,
    h: &Heard,
    prior: &Prior,
) -> f64 {
    let d = separation(points[h.from], points[h.to]);
    prior.rssi_at_1m + gains_db[h.from] + gains_db[h.to] - 10.0 * exponent * d.log10()
}

fn huber(r: f64) -> f64 {
    if r.abs() <= OUTLIER_SPREADS {
        r * r / 2.0
    } else {
        OUTLIER_SPREADS * r.abs() - OUTLIER_SPREADS * OUTLIER_SPREADS / 2.0
    }
}

/// Minus the log-likelihood, up to a constant: the links (Huber), the gains and exponent against
/// their priors, and the room's size against a usual room's.
fn room_cost(points: &[[f64; 2]], width: f64, depth: f64, heard: &[Heard], prior: &Prior) -> f64 {
    let (gains_db, exponent) = fit_radio(points, heard, prior);
    let links: f64 = heard
        .iter()
        .map(|h| {
            huber((h.db - predicted_db(points, &gains_db, exponent, h, prior)) / prior.residual_db)
        })
        .sum();
    let gains: f64 = gains_db
        .iter()
        .zip(&prior.gain_spread_db)
        .map(|(g, s)| (g / s).powi(2) / 2.0)
        .sum();
    let common = (common_gain(&gains_db, prior) / prior.common_gain_spread_db).powi(2) / 2.0;
    let exponent_off = ((exponent - prior.exponent) / prior.exponent_spread).powi(2) / 2.0;
    let side = |s: f64| ((s - prior.room_side_m) / prior.room_side_spread_m).powi(2) / 2.0;
    links + gains + common + exponent_off + side(width) + side(depth)
}

/// The nodes' gains averaged, each weighted by how sure its prior is (a phone hardly counts).
fn common_gain(gains_db: &[f64], prior: &Prior) -> f64 {
    let weights = common_weights(prior);
    gains_db.iter().zip(&weights).map(|(g, w)| g * w).sum()
}

fn common_weights(prior: &Prior) -> Vec<f64> {
    let raw: Vec<f64> = prior.gain_spread_db.iter().map(|s| 1.0 / (s * s)).collect();
    let total: f64 = raw.iter().sum();
    raw.iter().map(|w| w / total).collect()
}

/// For boards at `points`, the gains and exponent that fit the links best: a linear least squares
/// with the prior as ridge, reweighted three times so a blocked link counts less (Huber).
pub fn fit_radio(points: &[[f64; 2]], heard: &[Heard], prior: &Prior) -> (Vec<f64>, f64) {
    let nodes = points.len();
    let unknowns = nodes + 1;
    let mut weights = vec![1.0; heard.len()];
    let mut solution = vec![0.0; unknowns];
    solution[nodes] = prior.exponent;
    for _ in 0..3 {
        let mut normal = vec![vec![0.0; unknowns]; unknowns];
        let mut rhs = vec![0.0; unknowns];
        for (h, &w) in heard.iter().zip(&weights) {
            let mut row = vec![0.0; unknowns];
            row[h.from] += 1.0;
            row[h.to] += 1.0;
            row[nodes] = -10.0 * separation(points[h.from], points[h.to]).log10();
            let y = h.db - prior.rssi_at_1m;
            let w = w / (prior.residual_db * prior.residual_db);
            for i in 0..unknowns {
                rhs[i] += w * row[i] * y;
                for j in 0..unknowns {
                    normal[i][j] += w * row[i] * row[j];
                }
            }
        }
        let common = common_weights(prior);
        for i in 0..nodes {
            normal[i][i] += 1.0 / prior.gain_spread_db[i].powi(2);
            for j in 0..nodes {
                normal[i][j] += common[i] * common[j] / prior.common_gain_spread_db.powi(2);
            }
        }
        normal[nodes][nodes] += 1.0 / prior.exponent_spread.powi(2);
        rhs[nodes] += prior.exponent / prior.exponent_spread.powi(2);
        solution = solve_linear(normal, rhs);
        for (h, w) in heard.iter().zip(&mut weights) {
            let fitted = predicted_db(points, &solution[..nodes], solution[nodes], h, prior);
            let r = ((h.db - fitted) / prior.residual_db).abs();
            *w = if r <= OUTLIER_SPREADS {
                1.0
            } else {
                OUTLIER_SPREADS / r
            };
        }
    }
    (solution[..nodes].to_vec(), solution[nodes])
}

/// Gaussian elimination with partial pivoting; the systems here are a handful of unknowns.
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap_or(col);
        a.swap(col, pivot);
        b.swap(col, pivot);
        let p = a[col][col];
        if p.abs() < 1e-12 {
            continue;
        }
        let pivot_row = a[col].clone();
        for row in col + 1..n {
            let f = a[row][col] / p;
            for (x, y) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *x -= f * y;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let sum: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = if a[row][row].abs() < 1e-12 {
            0.0
        } else {
            (b[row] - sum) / a[row][row]
        };
    }
    x
}

/// `points` turned, moved and if need be mirrored to lie as close as possible on `reference`
/// (Procrustes without scaling). Distances cannot tell a layout from its mirror image.
pub fn align_to(points: &[[f64; 2]], reference: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let centre = |p: &[[f64; 2]]| {
        let n = p.len() as f64;
        [
            p.iter().map(|q| q[0]).sum::<f64>() / n,
            p.iter().map(|q| q[1]).sum::<f64>() / n,
        ]
    };
    let (c, r) = (centre(points), centre(reference));
    let best = [false, true]
        .into_iter()
        .map(|mirror| {
            let moved: Vec<[f64; 2]> = points
                .iter()
                .map(|p| [p[0] - c[0], if mirror { c[1] - p[1] } else { p[1] - c[1] }])
                .collect();
            let (mut sxx, mut sxy) = (0.0, 0.0);
            for (p, q) in moved.iter().zip(reference) {
                let q = [q[0] - r[0], q[1] - r[1]];
                sxx += p[0] * q[0] + p[1] * q[1];
                sxy += p[0] * q[1] - p[1] * q[0];
            }
            let (s, co) = sxy.atan2(sxx).sin_cos();
            let placed: Vec<[f64; 2]> = moved
                .iter()
                .map(|p| [co * p[0] - s * p[1] + r[0], s * p[0] + co * p[1] + r[1]])
                .collect();
            let misfit: f64 = placed
                .iter()
                .zip(reference)
                .map(|(a, b)| separation(*a, *b))
                .sum();
            (misfit, placed)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0));
    best.map(|(_, p)| p).unwrap_or_default()
}

/// The farthest any point is from its place in `reference`.
pub fn largest_move(points: &[[f64; 2]], reference: &[[f64; 2]]) -> f64 {
    points
        .iter()
        .zip(reference)
        .map(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]))
        .fold(0.0, f64::max)
}

/// Boards stand along a room's walls, so the layout is turned until its bounding box is smallest;
/// the long side is laid along x, node 0 put in the corner at the largest x and y (ahead on the right
/// from the 3D view's first camera, which looks the way someone at the laptop by the hub faces), and
/// the box moved to start at `margin`.
pub fn square_to_walls(points: &[[f64; 2]], margin: f64) -> Vec<[f64; 2]> {
    let turned = |angle: f64| -> Vec<[f64; 2]> {
        let (s, c) = angle.sin_cos();
        points
            .iter()
            .map(|p| [c * p[0] - s * p[1], s * p[0] + c * p[1]])
            .collect()
    };
    let area = |p: &[[f64; 2]]| {
        let (w, d) = extent(p);
        w * d
    };
    let best_angle = (0..180)
        .map(|k| f64::from(k) * std::f64::consts::PI / 360.0)
        .min_by(|&a, &b| area(&turned(a)).total_cmp(&area(&turned(b))))
        .unwrap_or(0.0);
    let mut p = turned(best_angle);
    let (w, d) = extent(&p);
    if d > w {
        p.iter_mut().for_each(|q| *q = [q[1], q[0]]);
    }
    for axis in 0..2 {
        let (low, high) = span(p.iter().map(|q| q[axis]));
        let nearer_high = high - p[0][axis] < p[0][axis] - low;
        if !nearer_high {
            p.iter_mut().for_each(|q| q[axis] = -q[axis]);
        }
    }
    let (min_x, _) = span(p.iter().map(|q| q[0]));
    let (min_y, _) = span(p.iter().map(|q| q[1]));
    p.iter()
        .map(|q| [q[0] - min_x + margin, q[1] - min_y + margin])
        .collect()
}

pub fn extent(points: &[[f64; 2]]) -> (f64, f64) {
    let (a, b) = span(points.iter().map(|q| q[0]));
    let (c, d) = span(points.iter().map(|q| q[1]));
    (b - a, d - c)
}

fn span(values: impl Iterator<Item = f64>) -> (f64, f64) {
    values.fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prior(nodes: usize) -> Prior {
        Prior {
            rssi_at_1m: -45.0,
            exponent: 2.5,
            exponent_spread: 0.5,
            gain_spread_db: vec![3.0; nodes],
            common_gain_spread_db: 1.0,
            residual_db: 3.0,
            room_side_m: 4.0,
            room_side_spread_m: 1.0,
        }
    }

    fn heard_from(points: &[[f64; 2]], gains: &[f64]) -> Vec<Heard> {
        let mut heard = Vec::new();
        for from in 0..points.len() {
            for to in (0..points.len()).filter(|&to| to != from) {
                let d = separation(points[from], points[to]);
                heard.push(Heard {
                    from,
                    to,
                    db: -45.0 + gains[from] + gains[to] - 25.0 * d.log10(),
                });
            }
        }
        heard
    }

    fn worst_error(points: &[[f64; 2]], truth: &[[f64; 2]]) -> f64 {
        largest_move(&align_to(points, truth), truth)
    }

    #[test]
    fn exact_links_give_back_boards_along_the_walls() {
        let truth = [[0.0, 0.0], [3.5, 0.0], [4.5, 2.0], [1.0, 3.0]];
        let layout = solve_on_walls(4, &heard_from(&truth, &[0.0; 4]), &prior(4)).unwrap();
        assert!(
            worst_error(&layout.points, &truth) < 0.2,
            "{:?}",
            layout.points
        );
    }

    #[test]
    fn a_weak_board_in_a_corner_stays_in_its_corner() {
        let truth = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]];
        let layout =
            solve_in_corners(&heard_from(&truth, &[0.0, -6.0, 0.0, 2.0]), &prior(4)).unwrap();
        assert!(
            worst_error(&layout.points, &truth) < 0.5,
            "{:?}",
            layout.points
        );
        assert!(
            layout.gains_db[1] < layout.gains_db[3],
            "{:?}",
            layout.gains_db
        );
    }

    #[test]
    fn align_undoes_a_turn_a_shift_and_a_mirror() {
        let truth = [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.5, 2.5]];
        let moved: Vec<[f64; 2]> = truth.iter().map(|p| [5.0 - p[1], -1.0 - p[0]]).collect();
        assert!(worst_error(&moved, &truth) < 1e-9);
    }

    #[test]
    fn a_turned_rectangle_is_squared_to_the_walls() {
        let (s, c) = 0.5f64.sin_cos();
        let rect: Vec<[f64; 2]> = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]
            .iter()
            .map(|p| [c * p[0] - s * p[1], s * p[0] + c * p[1]])
            .collect();
        let squared = square_to_walls(&rect, 0.2);
        let (w, d) = extent(&squared);
        assert!((w * d - 12.0).abs() < 0.2 && w >= d, "{w} x {d}");
        assert!(
            squared[0][0] > w && squared[0][1] > d,
            "{:?}",
            squared[0]
        );
    }
}
