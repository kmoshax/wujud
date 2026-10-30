//! Where a moving person is, from how much each link is disturbed right now.
//!
//! A body disturbs a link most on the straight path between its two ends and less the longer the detour
//! through the body is (the excess path length, as in Fresnel zones). Each grid point predicts a pattern
//! of disturbance over the links; the points whose pattern matches what the links see become more likely.
//! The belief is carried over time (people walk, they do not jump), so two links that each say "somewhere
//! along me" narrow down to where the person is. With fewer links the belief stays spread out, and the
//! view shows that spread instead of pretending.

use std::collections::VecDeque;

use ndarray::ArrayView2;

use crate::features::amplitude_from_iq;

const GRID_STEP_M: f64 = 0.1;
const WALL_MARGIN_M: f64 = 0.15;
/// How fast the predicted disturbance falls with excess path; fitted by eye to a 2.4 GHz room link.
const EXCESS_PATH_SCALE_M: f64 = 0.45;
const MATCH_SHARPNESS: f64 = 2.5;
const WALKING_SPEED_M_S: f64 = 1.2;
const UNIFORM_SHARE: f64 = 0.02;
const FLOOR_HISTORY: usize = 1500;
const FLOOR_QUANTILE: f64 = 0.2;
/// Disturbance above this many times the link's own quiet level counts as motion.
const MOVING_RATIO: f64 = 0.6;
const NEAR_ESTIMATE_M: f64 = 0.6;

/// How much a link's subcarrier amplitudes move over `iq` (`[packets, interleaved bytes]`, ~0.5 s of
/// packets): the mean over subcarriers of each one's standard deviation, in dB, after removing the
/// per-packet gain. A quiet room is ~1 dB, someone walking near the link 3-5 dB.
pub fn disturbance_db(iq: ArrayView2<i8>) -> f64 {
    let amplitude = amplitude_from_iq(iq).mapv(|a| 20.0 * f64::from(a.max(1e-3)).log10());
    let spread = amplitude.std_axis(ndarray::Axis(0), 0.0);
    spread.mean().unwrap_or(0.0)
}

/// The standard deviation of a link's RSSI (dBm) over the newest packets. A body on or near the
/// straight path between the two ends shadows it on and off, so this rises most there; the CSI
/// amplitudes lose that cue because each packet's gain is normalised away.
pub fn rssi_spread_db(rssi: &[f64]) -> f64 {
    let n = rssi.len().max(1) as f64;
    let mean = rssi.iter().sum::<f64>() / n;
    (rssi.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n).sqrt()
}

pub struct MotionTracker {
    links: Vec<([f64; 2], [f64; 2])>,
    xs: Vec<f64>,
    ys: Vec<f64>,
    belief: Vec<f64>,
    quiet_motion: Vec<QuietLevel>,
    quiet_shadowing: Vec<QuietLevel>,
    estimate: Option<[f64; 2]>,
}

pub struct Track {
    /// Most likely position while someone moves, and the last one after they stop.
    pub xy: Option<[f64; 2]>,
    /// Share of the belief within NEAR_ESTIMATE_M of `xy`: 1 = sure, near 0 = no idea.
    pub confidence: f64,
    pub moving: bool,
    /// Per link, how far its disturbance is above its quiet level (0 = quiet).
    pub link_excess: Vec<f64>,
}

impl MotionTracker {
    /// `links` are the two ends of each link on the floor plan; the room spans 0..width x 0..depth.
    pub fn new(links: Vec<([f64; 2], [f64; 2])>, width: f64, depth: f64) -> Self {
        let axis = |extent: f64| -> Vec<f64> {
            let count = ((extent - 2.0 * WALL_MARGIN_M) / GRID_STEP_M).floor() as usize + 1;
            (0..count)
                .map(|i| WALL_MARGIN_M + i as f64 * GRID_STEP_M)
                .collect()
        };
        let (xs, ys) = (axis(width), axis(depth));
        let cells = xs.len() * ys.len();
        MotionTracker {
            quiet_motion: links.iter().map(|_| QuietLevel::default()).collect(),
            quiet_shadowing: links.iter().map(|_| QuietLevel::default()).collect(),
            links,
            xs,
            ys,
            belief: vec![1.0 / cells as f64; cells],
            estimate: None,
        }
    }

    pub fn grid(&self) -> (&[f64], &[f64]) {
        (&self.xs, &self.ys)
    }

    /// Belief per grid cell, row by row (y outer, x inner); sums to 1.
    pub fn belief(&self) -> &[f64] {
        &self.belief
    }

    /// `disturbance[k]`: the spread of link k's subcarrier amplitudes over the last moment (dB).
    /// Per link: `motion` from `disturbance_db` (says whether someone moves), `shadowing` from
    /// `rssi_spread_db` (says which links the body is near).
    pub fn update(&mut self, motion: &[f64], shadowing: &[f64], seconds_since_last: f64) -> Track {
        let link_excess: Vec<f64> = motion
            .iter()
            .zip(&mut self.quiet_motion)
            .map(|(&d, quiet)| quiet.excess(d))
            .collect();
        let shadow_excess: Vec<f64> = shadowing
            .iter()
            .zip(&mut self.quiet_shadowing)
            .map(|(&d, quiet)| quiet.excess(d))
            .collect();
        let moving = link_excess.iter().any(|&e| e > MOVING_RATIO);
        if moving {
            self.spread(seconds_since_last);
            self.observe(&shadow_excess);
            self.estimate = Some(self.expected_position());
        }
        Track {
            xy: self.estimate,
            confidence: self.estimate.map_or(0.0, |xy| self.mass_near(xy)),
            moving,
            link_excess,
        }
    }

    fn spread(&mut self, seconds: f64) {
        let reach_cells = (WALKING_SPEED_M_S * seconds / GRID_STEP_M).ceil().max(1.0) as usize;
        self.belief = box_blur(&self.belief, self.xs.len(), self.ys.len(), reach_cells);
        let uniform = UNIFORM_SHARE / self.belief.len() as f64;
        self.belief
            .iter_mut()
            .for_each(|b| *b = (1.0 - UNIFORM_SHARE) * *b + uniform);
    }

    /// Motion anywhere in the room stirs every link a little through reflections; a body near a link
    /// adds a bump on that link. Per cell, the observed excess is fitted as a common part plus the
    /// cell's bump pattern (both non-negative); cells whose pattern explains the differences between
    /// links fit best.
    fn observe(&mut self, link_excess: &[f64]) {
        let observed_size = link_excess.iter().map(|e| e * e).sum::<f64>().max(1e-9);
        for (cell, belief) in self.belief.iter_mut().enumerate() {
            let point = [self.xs[cell % self.xs.len()], self.ys[cell / self.xs.len()]];
            let bumps: Vec<f64> = self
                .links
                .iter()
                .map(|&(a, b)| (-excess_path(point, a, b) / EXCESS_PATH_SCALE_M).exp())
                .collect();
            let mismatch = common_plus_bump_residual(link_excess, &bumps);
            *belief *= (-MATCH_SHARPNESS * mismatch / observed_size).exp();
        }
        let total: f64 = self.belief.iter().sum();
        self.belief.iter_mut().for_each(|b| *b /= total.max(1e-300));
    }

    /// The belief's centre of mass: with a spread-out belief it stays near the middle instead of
    /// jumping between far-apart guesses.
    fn expected_position(&self) -> [f64; 2] {
        let mut xy = [0.0, 0.0];
        for (cell, belief) in self.belief.iter().enumerate() {
            xy[0] += belief * self.xs[cell % self.xs.len()];
            xy[1] += belief * self.ys[cell / self.xs.len()];
        }
        xy
    }

    fn mass_near(&self, xy: [f64; 2]) -> f64 {
        self.belief
            .iter()
            .enumerate()
            .filter(|(cell, _)| {
                let point = [self.xs[cell % self.xs.len()], self.ys[cell / self.xs.len()]];
                distance(point, xy) <= NEAR_ESTIMATE_M
            })
            .map(|(_, b)| b)
            .sum()
    }
}

/// A link's disturbance when nothing moves, learned as it runs: a low quantile of what it has seen lately.
#[derive(Default)]
struct QuietLevel {
    recent: VecDeque<f64>,
}

impl QuietLevel {
    /// A link without fresh packets passes NaN: it counts as quiet and does not shift its quiet level.
    fn excess(&mut self, disturbance: f64) -> f64 {
        if disturbance.is_nan() {
            return 0.0;
        }
        if self.recent.len() == FLOOR_HISTORY {
            self.recent.pop_front();
        }
        self.recent.push_back(disturbance);
        let mut sorted: Vec<f64> = self.recent.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let quiet = sorted[((sorted.len() - 1) as f64 * FLOOR_QUANTILE) as usize].max(1e-6);
        (disturbance / quiet - 1.0).max(0.0)
    }
}

/// Least squares of `observed ≈ common + size * bumps` with common, size >= 0; returns the squared residual.
fn common_plus_bump_residual(observed: &[f64], bumps: &[f64]) -> f64 {
    let n = observed.len() as f64;
    let (mean_o, mean_b) = (
        observed.iter().sum::<f64>() / n,
        bumps.iter().sum::<f64>() / n,
    );
    let covariance: f64 = observed
        .iter()
        .zip(bumps)
        .map(|(o, b)| (o - mean_o) * (b - mean_b))
        .sum();
    let variance: f64 = bumps.iter().map(|b| (b - mean_b).powi(2)).sum();
    let mut size = if variance > 1e-12 {
        (covariance / variance).max(0.0)
    } else {
        0.0
    };
    let mut common = mean_o - size * mean_b;
    if common < 0.0 {
        common = 0.0;
        let along: f64 = observed.iter().zip(bumps).map(|(o, b)| o * b).sum();
        let norm: f64 = bumps.iter().map(|b| b * b).sum();
        size = (along / norm.max(1e-12)).max(0.0);
    }
    observed
        .iter()
        .zip(bumps)
        .map(|(o, b)| (o - common - size * b).powi(2))
        .sum()
}

fn excess_path(point: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    distance(point, a) + distance(point, b) - distance(a, b)
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Mean over a (2r+1)^2 square, clipped at the edges; run twice it approximates a Gaussian.
fn box_blur(values: &[f64], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let pass = |input: &[f64], horizontal: bool| -> Vec<f64> {
        let mut output = vec![0.0; input.len()];
        for y in 0..height {
            for x in 0..width {
                let (centre, limit) = if horizontal { (x, width) } else { (y, height) };
                let (from, to) = (
                    centre.saturating_sub(radius),
                    (centre + radius).min(limit - 1),
                );
                let sum: f64 = (from..=to)
                    .map(|i| {
                        if horizontal {
                            input[y * width + i]
                        } else {
                            input[i * width + x]
                        }
                    })
                    .sum();
                output[y * width + x] = sum / (to - from + 1) as f64;
            }
        }
        output
    };
    let once = pass(&pass(values, true), false);
    pass(&pass(&once, true), false)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOARD: [f64; 2] = [3.8, 3.3];
    const LAPTOP: [f64; 2] = [0.4, 0.3];
    const PHONE: [f64; 2] = [0.3, 3.2];

    fn tracker() -> MotionTracker {
        MotionTracker::new(vec![(BOARD, LAPTOP), (BOARD, PHONE)], 4.0, 3.5)
    }

    fn settle(tracker: &mut MotionTracker) {
        for _ in 0..50 {
            tracker.update(&[1.0, 1.0], &[1.0, 1.0], 0.2);
        }
    }

    fn disturbance_at(point: [f64; 2]) -> Vec<f64> {
        [(BOARD, LAPTOP), (BOARD, PHONE)]
            .iter()
            .map(|&(a, b)| 1.0 + 4.0 * (-excess_path(point, a, b) / EXCESS_PATH_SCALE_M).exp())
            .collect()
    }

    #[test]
    fn quiet_links_are_not_motion() {
        let mut tracker = tracker();
        settle(&mut tracker);
        let track = tracker.update(&[1.05, 0.98], &[1.05, 0.98], 0.2);
        assert!(!track.moving);
        assert!(track.xy.is_none());
    }

    #[test]
    fn walking_along_one_link_leans_towards_that_link() {
        let mut tracker = tracker();
        settle(&mut tracker);
        let mut track = None;
        for step in 0..15 {
            let t = 0.3 + 0.03 * step as f64;
            let point = [
                LAPTOP[0] + t * (BOARD[0] - LAPTOP[0]),
                LAPTOP[1] + t * (BOARD[1] - LAPTOP[1]),
            ];
            track = Some((
                point,
                tracker.update(&disturbance_at(point), &disturbance_at(point), 0.2),
            ));
        }
        let (_, track) = track.unwrap();
        let xy = track.xy.unwrap();
        assert!(track.moving);
        assert!(
            excess_path(xy, BOARD, LAPTOP) < excess_path(xy, BOARD, PHONE),
            "estimate {xy:?} is not nearer the walked link"
        );
    }

    #[test]
    fn stopping_keeps_the_last_position() {
        let mut tracker = tracker();
        settle(&mut tracker);
        for _ in 0..10 {
            tracker.update(
                &disturbance_at([2.0, 3.2]),
                &disturbance_at([2.0, 3.2]),
                0.2,
            );
        }
        let moving = tracker
            .update(
                &disturbance_at([2.0, 3.2]),
                &disturbance_at([2.0, 3.2]),
                0.2,
            )
            .xy;
        let stopped = tracker.update(&[1.0, 1.0], &[1.0, 1.0], 0.2);
        assert!(!stopped.moving);
        assert_eq!(stopped.xy, moving);
    }
}
