//! Where a walking person is, from a small network trained on simulated rooms only
//! (research/tools/learned_tracker.py; 0.69 m median error on rooms it never saw, against 0.86 m for
//! the room's centre). It needs only where the boards stand: every input is relative to a link.
//!
//! Per grid point and link: how the point sits relative to the link (extra path via the point, distance
//! to the line, position along it, link length) and how disturbed the link is over its own quiet level.
//! The link network embeds that pair, the embeddings are averaged over links, the point network scores
//! the point, and the position is the softmax-weighted mean of the points.

use std::collections::VecDeque;
use std::path::Path;

use rayon::prelude::*;
use serde::Deserialize;

const HISTORY_SECONDS: usize = 300;
const HISTORY_NEEDED: usize = 20;

#[derive(Deserialize)]
struct File {
    grid_m: f64,
    smooth_s: usize,
    quiet_percentile: usize,
    weights: Weights,
}

#[derive(Deserialize)]
struct Weights {
    #[serde(rename = "link.0.weight")]
    link0_w: Vec<Vec<f32>>,
    #[serde(rename = "link.0.bias")]
    link0_b: Vec<f32>,
    #[serde(rename = "link.2.weight")]
    link2_w: Vec<Vec<f32>>,
    #[serde(rename = "link.2.bias")]
    link2_b: Vec<f32>,
    #[serde(rename = "point.0.weight")]
    point0_w: Vec<Vec<f32>>,
    #[serde(rename = "point.0.bias")]
    point0_b: Vec<f32>,
    #[serde(rename = "point.2.weight")]
    point2_w: Vec<Vec<f32>>,
    #[serde(rename = "point.2.bias")]
    point2_b: Vec<f32>,
}

fn dense(w: &[Vec<f32>], b: &[f32], x: &[f32], relu: bool) -> Vec<f32> {
    w.iter()
        .zip(b)
        .map(|(row, bias)| {
            let v = row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>() + bias;
            if relu {
                v.max(0.0)
            } else {
                v
            }
        })
        .collect()
}

pub struct LearnedTracker {
    weights: Weights,
    smooth: usize,
    quiet_percentile: usize,
    points: Vec<[f64; 2]>,
    /// points x links x hidden: the first layer's geometry part (bias included), which never changes;
    /// each second only the two signal inputs are added.
    geometry_part: Vec<Vec<Vec<f32>>>,
    /// Per link, the newest seconds of (disturbance, rssi spread).
    history: Vec<VecDeque<[f64; 2]>>,
    names: Vec<String>,
}

impl LearnedTracker {
    /// `names`: one per link, to carry a link's history over when the room is laid out again.
    pub fn load(
        path: &Path,
        links: &[([f64; 2], [f64; 2])],
        names: &[String],
        width: f64,
        depth: f64,
    ) -> anyhow::Result<Self> {
        let file: File = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let step = file.grid_m;
        let mut points = Vec::new();
        let mut y = step / 2.0;
        while y < depth {
            let mut x = step / 2.0;
            while x < width {
                points.push([x, y]);
                x += step;
            }
            y += step;
        }
        let geometry: Vec<Vec<[f32; 4]>> = points
            .iter()
            .map(|&[x, y]| {
                links
                    .iter()
                    .map(|&(a, b)| {
                        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                        let length = dx.hypot(dy).max(1e-9);
                        let (rx, ry) = (x - a[0], y - a[1]);
                        let extra = rx.hypot(ry) + (x - b[0]).hypot(y - b[1]) - length;
                        let cross = (dx * ry - dy * rx).abs() / length;
                        let along = ((rx * dx + ry * dy) / (length * length)).clamp(-0.5, 1.5);
                        [extra as f32, cross as f32, along as f32, length as f32]
                    })
                    .collect()
            })
            .collect();
        let w = &file.weights;
        let geometry_part = geometry
            .iter()
            .map(|links| {
                links
                    .iter()
                    .map(|g| {
                        w.link0_w
                            .iter()
                            .zip(&w.link0_b)
                            .map(|(row, b)| row[..4].iter().zip(g).map(|(a, x)| a * x).sum::<f32>() + b)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Ok(LearnedTracker {
            weights: file.weights,
            smooth: file.smooth_s,
            quiet_percentile: file.quiet_percentile,
            points,
            geometry_part,
            history: links.iter().map(|_| VecDeque::new()).collect(),
            names: names.to_vec(),
        })
    }

    /// Keeps each link's quiet level and recent seconds from the room as it was laid out before.
    pub fn carry_over(&mut self, previous: &LearnedTracker) {
        for (name, history) in self.names.iter().zip(&mut self.history) {
            if let Some(i) = previous.names.iter().position(|n| n == name) {
                *history = previous.history[i].clone();
            }
        }
    }

    /// Once a second, each link's disturbance and RSSI spread (None for a link with no fresh packets).
    pub fn add_second(&mut self, links: &[(Option<f64>, Option<f64>)]) {
        for (history, &(motion, shadow)) in self.history.iter_mut().zip(links) {
            let (Some(m), Some(s)) = (motion, shadow) else { continue };
            if history.len() == HISTORY_SECONDS {
                history.pop_front();
            }
            history.push_back([m, s]);
        }
    }

    /// Each link's (disturbance, spread) over its quiet level, in log, averaged over the newest seconds.
    fn signals(&self) -> Option<Vec<[f32; 2]>> {
        self.history
            .iter()
            .map(|history| {
                if history.len() < HISTORY_NEEDED {
                    return None;
                }
                let quiet = |k: usize| {
                    let mut v: Vec<f64> = history.iter().map(|h| h[k]).collect();
                    v.sort_by(f64::total_cmp);
                    v[v.len() * self.quiet_percentile / 100].max(1e-6)
                };
                let (qm, qs) = (quiet(0), quiet(1));
                let recent: Vec<&[f64; 2]> = history.iter().rev().take(self.smooth).collect();
                let n = recent.len() as f64;
                let m = recent.iter().map(|h| (h[0].max(1e-6) / qm).ln()).sum::<f64>() / n;
                let s = recent.iter().map(|h| (h[1].max(1e-6) / qs).ln()).sum::<f64>() / n;
                Some([m as f32, s as f32])
            })
            .collect()
    }

    /// Where the network puts the person now, once every link has a history.
    pub fn position(&self) -> Option<[f64; 2]> {
        let signals = self.signals()?;
        let w = &self.weights;
        let signal_part: Vec<Vec<f32>> = signals
            .iter()
            .map(|s| w.link0_w.iter().map(|row| row[4] * s[0] + row[5] * s[1]).collect())
            .collect();
        let scores: Vec<f32> = self
            .geometry_part
            .par_iter()
            .map(|links| {
                let mut pooled = vec![0.0f32; w.link2_b.len()];
                for (geo, signal) in links.iter().zip(&signal_part) {
                    let hidden: Vec<f32> = geo.iter().zip(signal).map(|(g, s)| (g + s).max(0.0)).collect();
                    let embedded = dense(&w.link2_w, &w.link2_b, &hidden, true);
                    pooled.iter_mut().zip(embedded).for_each(|(p, e)| *p += e / links.len() as f32);
                }
                let hidden = dense(&w.point0_w, &w.point0_b, &pooled, true);
                dense(&w.point2_w, &w.point2_b, &hidden, false)[0]
            })
            .collect();
        let top = scores.iter().copied().fold(f32::MIN, f32::max);
        let weights: Vec<f64> = scores.iter().map(|s| f64::from((s - top).exp())).collect();
        let total: f64 = weights.iter().sum();
        let x = self.points.iter().zip(&weights).map(|(p, w)| p[0] * w).sum::<f64>() / total;
        let y = self.points.iter().zip(&weights).map(|(p, w)| p[1] * w).sum::<f64>() / total;
        Some([x, y])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full first layer over [geometry, signal], as the network was trained.
    fn reference(tracker: &LearnedTracker, geometry: &[Vec<[f32; 4]>], signals: &[[f32; 2]]) -> [f64; 2] {
        let w = &tracker.weights;
        let scores: Vec<f32> = geometry
            .iter()
            .map(|links| {
                let mut pooled = vec![0.0f32; w.link2_b.len()];
                for (geo, signal) in links.iter().zip(signals) {
                    let input = [geo[0], geo[1], geo[2], geo[3], signal[0], signal[1]];
                    let embedded = dense(&w.link2_w, &w.link2_b, &dense(&w.link0_w, &w.link0_b, &input, true), true);
                    pooled.iter_mut().zip(embedded).for_each(|(p, e)| *p += e / links.len() as f32);
                }
                dense(&w.point2_w, &w.point2_b, &dense(&w.point0_w, &w.point0_b, &pooled, true), false)[0]
            })
            .collect();
        let top = scores.iter().copied().fold(f32::MIN, f32::max);
        let weights: Vec<f64> = scores.iter().map(|s| f64::from((s - top).exp())).collect();
        let total: f64 = weights.iter().sum();
        [
            tracker.points.iter().zip(&weights).map(|(p, w)| p[0] * w).sum::<f64>() / total,
            tracker.points.iter().zip(&weights).map(|(p, w)| p[1] * w).sum::<f64>() / total,
        ]
    }

    #[test]
    fn the_fast_path_matches_the_trained_network() {
        let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../model/models/learned_tracker.json");
        let links = [([0.2, 0.2], [3.8, 0.2]), ([0.2, 0.2], [0.2, 3.3]), ([3.8, 0.2], [0.2, 3.3])];
        let names: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
        let mut tracker = LearnedTracker::load(&model, &links, &names, 4.0, 3.5).unwrap();
        for second in 0..40 {
            let busy = second >= 30;
            tracker.add_second(&[(Some(if busy { 4.0 } else { 1.0 }), Some(1.0)), (Some(1.0), Some(1.0)), (Some(1.1), Some(1.0))]);
        }
        let signals = tracker.signals().unwrap();
        let geometry: Vec<Vec<[f32; 4]>> = tracker
            .points
            .iter()
            .map(|&[x, y]| {
                links
                    .iter()
                    .map(|&(a, b)| {
                        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                        let length = dx.hypot(dy);
                        let (rx, ry) = (x - a[0], y - a[1]);
                        [
                            (rx.hypot(ry) + (x - b[0]).hypot(y - b[1]) - length) as f32,
                            ((dx * ry - dy * rx).abs() / length) as f32,
                            ((rx * dx + ry * dy) / (length * length)).clamp(-0.5, 1.5) as f32,
                            length as f32,
                        ]
                    })
                    .collect()
            })
            .collect();
        let fast = tracker.position().unwrap();
        let full = reference(&tracker, &geometry, &signals);
        assert!((fast[0] - full[0]).abs() < 1e-3 && (fast[1] - full[1]).abs() < 1e-3, "{fast:?} vs {full:?}");
    }
}
