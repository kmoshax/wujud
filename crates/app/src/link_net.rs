//! Where a walking person is, from every link at once (research/tools/link_net.py; trained on simulated
//! rooms only: 0.53 m median error with four devices, 0.68 m with three, on rooms it never saw).
//!
//! Each pair of devices is one token: its disturbance and RSSI spread over its own quiet level, the same
//! relative to the other pairs, each subcarrier's spread over its quiet level and how the spectrum's shape
//! moved, over the last three seconds. A transformer lets the tokens see each other; every grid point then
//! attends over the pairs through how it sits relative to each, and is scored. The position is the
//! softmax-weighted mean of the points. Both directions of a pair are averaged into one token, as the
//! simulator measured one link per pair.

use std::collections::{HashMap, VecDeque};
use std::path::Path;

use rayon::prelude::*;
use serde::Deserialize;

const HISTORY_SECONDS: usize = 300;
const HISTORY_NEEDED: usize = 20;
const LAYER_NORM_EPS: f32 = 1e-5;
/// How far the belief spreads between two seconds of walking (research/tools/track_filter.py chose it on
/// simulated rooms: as accurate as each second alone, with half the jumps over a metre).
const WALK_SPREAD_M: f64 = 1.2;

#[derive(Deserialize)]
#[serde(untagged)]
enum Tensor {
    Vector(Vec<f32>),
    Matrix(Vec<Vec<f32>>),
}

#[derive(Deserialize)]
struct File {
    grid_m: f64,
    bins: usize,
    context_s: usize,
    shape_lag_s: usize,
    quiet_percentile: f64,
    heads: usize,
    activities: Vec<String>,
    weights: HashMap<String, Tensor>,
    #[cfg_attr(not(test), allow(dead_code))]
    check: Check,
}

#[derive(Deserialize)]
#[cfg_attr(not(test), allow(dead_code))]
struct Check {
    nodes: Vec<[f64; 2]>,
    links: Vec<[usize; 2]>,
    width: f64,
    depth: f64,
    tokens: Vec<Vec<f32>>,
    position: [f64; 2],
    activity: Vec<f32>,
}

struct Dense {
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
}

impl Dense {
    fn apply(&self, x: &[f32], relu: bool) -> Vec<f32> {
        self.w
            .iter()
            .zip(&self.b)
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

    /// The layer over only the columns `from..to` of its input, without the bias.
    fn part(&self, x: &[f32], from: usize, to: usize) -> Vec<f32> {
        self.w.iter().map(|row| row[from..to].iter().zip(x).map(|(a, b)| a * b).sum()).collect()
    }
}

struct Norm {
    gain: Vec<f32>,
    shift: Vec<f32>,
}

impl Norm {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let scale = 1.0 / (var + LAYER_NORM_EPS).sqrt();
        x.iter().zip(&self.gain).zip(&self.shift).map(|((v, g), s)| (v - mean) * scale * g + s).collect()
    }
}

/// One post-norm transformer encoder layer (PyTorch's default), dropout off.
struct Encoder {
    in_proj: Dense,
    out_proj: Dense,
    linear1: Dense,
    linear2: Dense,
    norm1: Norm,
    norm2: Norm,
}

impl Encoder {
    fn apply(&self, tokens: &[Vec<f32>], heads: usize) -> Vec<Vec<f32>> {
        let width = tokens[0].len();
        let head = width / heads;
        let qkv: Vec<Vec<f32>> = tokens.iter().map(|t| self.in_proj.apply(t, false)).collect();
        let scale = 1.0 / (head as f32).sqrt();
        let attended: Vec<Vec<f32>> = qkv
            .iter()
            .map(|query| {
                let mut out = vec![0.0f32; width];
                for h in 0..heads {
                    let q = &query[h * head..(h + 1) * head];
                    let scores: Vec<f32> = qkv
                        .iter()
                        .map(|key| q.iter().zip(&key[width + h * head..width + (h + 1) * head]).map(|(a, b)| a * b).sum::<f32>() * scale)
                        .collect();
                    for (value, weight) in qkv.iter().zip(softmax(&scores)) {
                        let v = &value[2 * width + h * head..2 * width + (h + 1) * head];
                        out[h * head..(h + 1) * head].iter_mut().zip(v).for_each(|(o, v)| *o += weight * v);
                    }
                }
                self.out_proj.apply(&out, false)
            })
            .collect();
        tokens
            .iter()
            .zip(attended)
            .map(|(t, a)| {
                let x = self.norm1.apply(&add(t, &a));
                let ff = self.linear2.apply(&self.linear1.apply(&x, true), false);
                self.norm2.apply(&add(&x, &ff))
            })
            .collect()
    }
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(a, b)| a + b).collect()
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let top = x.iter().copied().fold(f32::MIN, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - top).exp()).collect();
    let total: f32 = e.iter().sum();
    e.into_iter().map(|v| v / total).collect()
}

struct Weights(HashMap<String, Tensor>);

impl Weights {
    fn vector(&mut self, name: &str) -> anyhow::Result<Vec<f32>> {
        match self.0.remove(name) {
            Some(Tensor::Vector(v)) => Ok(v),
            _ => anyhow::bail!("link_net: no vector {name}"),
        }
    }

    fn matrix(&mut self, name: &str) -> anyhow::Result<Vec<Vec<f32>>> {
        match self.0.remove(name) {
            Some(Tensor::Matrix(m)) => Ok(m),
            _ => anyhow::bail!("link_net: no matrix {name}"),
        }
    }

    fn dense(&mut self, name: &str) -> anyhow::Result<Dense> {
        Ok(Dense { w: self.matrix(&format!("{name}.weight"))?, b: self.vector(&format!("{name}.bias"))? })
    }
}

struct Network {
    embed0: Dense,
    embed2: Dense,
    encoders: Vec<Encoder>,
    heads: usize,
    geo0: Dense,
    geo2: Dense,
    pair0: Dense,
    pair2: Dense,
    attend: Dense,
    score0: Dense,
    score2: Dense,
    activity0: Dense,
    activity2: Dense,
}

impl Network {
    fn from(weights: HashMap<String, Tensor>, heads: usize) -> anyhow::Result<Self> {
        let mut w = Weights(weights);
        let mut encoders = Vec::new();
        for layer in 0..2 {
            let at = |part: &str| format!("links.layers.{layer}.{part}");
            encoders.push(Encoder {
                in_proj: Dense { w: w.matrix(&at("self_attn.in_proj_weight"))?, b: w.vector(&at("self_attn.in_proj_bias"))? },
                out_proj: w.dense(&at("self_attn.out_proj"))?,
                linear1: w.dense(&at("linear1"))?,
                linear2: w.dense(&at("linear2"))?,
                norm1: Norm { gain: w.vector(&at("norm1.weight"))?, shift: w.vector(&at("norm1.bias"))? },
                norm2: Norm { gain: w.vector(&at("norm2.weight"))?, shift: w.vector(&at("norm2.bias"))? },
            });
        }
        Ok(Network {
            embed0: w.dense("embed.0")?,
            embed2: w.dense("embed.2")?,
            encoders,
            heads,
            geo0: w.dense("geo.0")?,
            geo2: w.dense("geo.2")?,
            pair0: w.dense("pair.0")?,
            pair2: w.dense("pair.2")?,
            attend: w.dense("attend")?,
            score0: w.dense("score.0")?,
            score2: w.dense("score.2")?,
            activity0: w.dense("activity.0")?,
            activity2: w.dense("activity.2")?,
        })
    }

    /// Each token after the transformer: what that pair says, knowing what the others say.
    fn read_links(&self, tokens: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut c: Vec<Vec<f32>> = tokens.iter().map(|t| self.embed2.apply(&self.embed0.apply(t, true), false)).collect();
        for encoder in &self.encoders {
            c = encoder.apply(&c, self.heads);
        }
        c
    }

    /// The first pair layer's geometry part (bias included) for one point and link, which never changes.
    fn geometry_part(&self, g: &[f32; 4]) -> Vec<f32> {
        let embedded = self.geo2.apply(&self.geo0.apply(g, true), false);
        let width = self.embed2.b.len();
        add(&self.pair0.part(&embedded, width, width + embedded.len()), &self.pair0.b)
    }

    fn score(&self, geometry_part: &[Vec<f32>], link_part: &[Vec<f32>]) -> f32 {
        let hidden: Vec<Vec<f32>> = geometry_part
            .iter()
            .zip(link_part)
            .map(|(g, l)| {
                let first: Vec<f32> = g.iter().zip(l).map(|(g, l)| (g + l).max(0.0)).collect();
                self.pair2.apply(&first, true)
            })
            .collect();
        let attention: Vec<f32> = hidden.iter().map(|h| self.attend.apply(h, false)[0]).collect();
        let mut pooled = vec![0.0f32; hidden[0].len()];
        for (h, w) in hidden.iter().zip(softmax(&attention)) {
            pooled.iter_mut().zip(h).for_each(|(p, h)| *p += w * h);
        }
        self.score2.apply(&self.score0.apply(&pooled, true), false)[0]
    }

    fn activity(&self, links: &[Vec<f32>]) -> Vec<f32> {
        let width = links[0].len();
        let mut pooled = vec![0.0f32; width * 2];
        for (i, slot) in pooled.iter_mut().take(width).enumerate() {
            *slot = links.iter().map(|l| l[i]).sum::<f32>() / links.len() as f32;
        }
        for i in 0..width {
            pooled[width + i] = links.iter().map(|l| l[i]).fold(f32::MIN, f32::max);
        }
        self.activity2.apply(&self.activity0.apply(&pooled, true), false)
    }
}

fn grid(step: f64, width: f64, depth: f64) -> Vec<[f64; 2]> {
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
    points
}

fn geometry(point: [f64; 2], a: [f64; 2], b: [f64; 2]) -> [f32; 4] {
    let [x, y] = point;
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let length = dx.hypot(dy);
    let (rx, ry) = (x - a[0], y - a[1]);
    let extra = rx.hypot(ry) + (x - b[0]).hypot(y - b[1]) - length;
    let cross = (dx * ry - dy * rx).abs() / length.max(1e-9);
    let along = ((rx * dx + ry * dy) / (length * length).max(1e-9)).clamp(-0.5, 1.5);
    [extra as f32, cross as f32, along as f32, length as f32]
}

/// `values` resampled linearly onto `bins` evenly spaced points, as numpy's interp in training.
pub fn to_bins(values: &[f32], bins: usize) -> Vec<f32> {
    let last = values.len().saturating_sub(1) as f32;
    (0..bins)
        .map(|b| {
            let position = b as f32 * last / (bins - 1) as f32;
            let (lo, hi) = (position.floor() as usize, (position.ceil() as usize).min(values.len() - 1));
            let w = position - lo as f32;
            values[lo] * (1.0 - w) + values[hi] * w
        })
        .collect()
}

fn percentile(values: impl Iterator<Item = f32>, p: f64) -> f32 {
    let mut v: Vec<f32> = values.collect();
    v.sort_by(f32::total_cmp);
    let position = p / 100.0 * (v.len() - 1) as f64;
    let (lo, hi) = (position.floor() as usize, position.ceil() as usize);
    let w = (position - lo as f64) as f32;
    v[lo] * (1.0 - w) + v[hi] * w
}

/// One link's second: disturbance, RSSI spread, and per subcarrier the spread of its dB level and its
/// mean level with the spectrum's mean removed (any number of subcarriers; resampled to the model's bins).
pub struct LinkSecond {
    pub motion: f64,
    pub shadow: f64,
    pub spread: Vec<f32>,
    pub shape: Vec<f32>,
}

#[derive(Clone)]
struct Second {
    motion: f32,
    shadow: f32,
    spread: Vec<f32>,
    shape: Vec<f32>,
}

pub struct LinkNetTracker {
    network: Network,
    bins: usize,
    context: usize,
    shape_lag: usize,
    quiet_percentile: f64,
    activities: Vec<String>,
    points: Vec<[f64; 2]>,
    /// points x pairs x hidden.
    geometry_part: Vec<Vec<Vec<f32>>>,
    /// Per pair of devices, the live links between them (one per direction heard).
    pairs: Vec<Vec<usize>>,
    names: Vec<Vec<String>>,
    history: Vec<VecDeque<Second>>,
    /// points x points: where someone at one point may be a second later.
    walk: Vec<Vec<f32>>,
    /// Over the points: where the person is, followed through the seconds they walk.
    belief: Vec<f32>,
}

impl LinkNetTracker {
    /// `links`: each live link's ends; links between the same two places become one pair.
    pub fn load(
        path: &Path,
        links: &[([f64; 2], [f64; 2])],
        names: &[String],
        width: f64,
        depth: f64,
    ) -> anyhow::Result<Self> {
        let file: File = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let network = Network::from(file.weights, file.heads)?;
        let mut pairs: Vec<Vec<usize>> = Vec::new();
        let mut ends: Vec<([f64; 2], [f64; 2])> = Vec::new();
        let same = |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).abs() < 1e-6 && (p[1] - q[1]).abs() < 1e-6;
        for (i, &(a, b)) in links.iter().enumerate() {
            match ends.iter().position(|&(c, d)| (same(a, c) && same(b, d)) || (same(a, d) && same(b, c))) {
                Some(p) => pairs[p].push(i),
                None => {
                    ends.push((a, b));
                    pairs.push(vec![i]);
                }
            }
        }
        let points = grid(file.grid_m, width, depth);
        let geometry_part = points
            .par_iter()
            .map(|&p| ends.iter().map(|&(a, b)| network.geometry_part(&geometry(p, a, b))).collect())
            .collect();
        let walk = walk_spread(&points);
        let belief = vec![1.0 / points.len() as f32; points.len()];
        Ok(LinkNetTracker {
            network,
            bins: file.bins,
            context: file.context_s,
            shape_lag: file.shape_lag_s,
            quiet_percentile: file.quiet_percentile,
            activities: file.activities,
            points,
            geometry_part,
            names: pairs.iter().map(|p| p.iter().map(|&i| names[i].clone()).collect()).collect(),
            history: pairs.iter().map(|_| VecDeque::new()).collect(),
            walk,
            belief,
            pairs,
        })
    }

    /// Keeps each pair's quiet level and recent seconds from the room as it was laid out before.
    pub fn carry_over(&mut self, previous: &LinkNetTracker) {
        for (names, history) in self.names.iter().zip(&mut self.history) {
            if let Some(i) = previous.names.iter().position(|old| old.iter().any(|n| names.contains(n))) {
                *history = previous.history[i].clone();
            }
        }
    }

    /// Once a second, every live link's reading (None for a link with no fresh packets). A pair heard
    /// in neither direction repeats its last second, so all pairs stay in step.
    pub fn add_second(&mut self, links: &[Option<LinkSecond>]) {
        for (pair, history) in self.pairs.iter().zip(&mut self.history) {
            let heard: Vec<&LinkSecond> = pair.iter().filter_map(|&i| links.get(i)?.as_ref()).collect();
            let second = if heard.is_empty() {
                match history.back() {
                    Some(last) => last.clone(),
                    None => continue,
                }
            } else {
                let n = heard.len() as f32;
                let mean_bins = |pick: fn(&LinkSecond) -> &Vec<f32>| {
                    let mut out = vec![0.0f32; self.bins];
                    for link in &heard {
                        out.iter_mut().zip(to_bins(pick(link), self.bins)).for_each(|(o, v)| *o += v / n);
                    }
                    out
                };
                Second {
                    motion: heard.iter().map(|l| l.motion as f32).sum::<f32>() / n,
                    shadow: heard.iter().map(|l| l.shadow as f32).sum::<f32>() / n,
                    spread: mean_bins(|l| &l.spread),
                    shape: mean_bins(|l| &l.shape),
                }
            };
            if history.len() == HISTORY_SECONDS {
                history.pop_front();
            }
            history.push_back(second);
        }
    }

    /// Per pair, `ago` seconds before the newest: log disturbance and spread over the quiet level,
    /// per-bin log spread over its quiet level, and how far each bin of the shape moved.
    fn own_features(&self, history: &VecDeque<Second>, ago: usize) -> (f32, f32, Vec<f32>) {
        let eps = 1e-6f32;
        let at = |back: usize| &history[history.len() - 1 - back.min(history.len() - 1)];
        let p = self.quiet_percentile;
        let quiet_m = percentile(history.iter().map(|s| s.motion), p).max(eps);
        let quiet_s = percentile(history.iter().map(|s| s.shadow), p).max(eps);
        let now = at(ago);
        let mut rest = Vec::with_capacity(self.bins * 2);
        for b in 0..self.bins {
            let quiet = percentile(history.iter().map(|s| s.spread[b]), p).max(eps);
            rest.push((now.spread[b].max(eps) / quiet).ln());
        }
        let lagged = at(ago + self.shape_lag);
        rest.extend(now.shape.iter().zip(&lagged.shape).map(|(a, b)| (a - b).abs()));
        ((now.motion.max(eps) / quiet_m).ln(), (now.shadow.max(eps) / quiet_s).ln(), rest)
    }

    fn tokens(&self) -> Option<Vec<Vec<f32>>> {
        if self.history.iter().any(|h| h.len() < HISTORY_NEEDED) {
            return None;
        }
        let mut tokens = vec![Vec::new(); self.history.len()];
        for ago in (0..self.context).rev() {
            let own: Vec<(f32, f32, Vec<f32>)> = self.history.iter().map(|h| self.own_features(h, ago)).collect();
            let n = own.len() as f32;
            let mean_m = own.iter().map(|o| o.0).sum::<f32>() / n;
            let mean_s = own.iter().map(|o| o.1).sum::<f32>() / n;
            for (token, (m, s, rest)) in tokens.iter_mut().zip(own) {
                token.extend([m, s, m - mean_m, s - mean_s]);
                token.extend(rest);
            }
        }
        for token in &mut tokens {
            token.iter_mut().filter(|v| !v.is_finite()).for_each(|v| *v = 0.0);
        }
        Some(tokens)
    }

    /// The network's distribution over the points, and over the activities.
    fn answer(&self, tokens: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
        let read = self.network.read_links(tokens);
        let width = read[0].len();
        let link_part: Vec<Vec<f32>> = read.iter().map(|c| self.network.pair0.part(c, 0, width)).collect();
        let scores: Vec<f32> = self.geometry_part.par_iter().map(|g| self.network.score(g, &link_part)).collect();
        (softmax(&scores), softmax(&self.network.activity(&read)))
    }

    fn mean_point(&self, weights: &[f32]) -> [f64; 2] {
        let mut xy = [0.0f64; 2];
        for (p, w) in self.points.iter().zip(weights) {
            xy[0] += p[0] * f64::from(*w);
            xy[1] += p[1] * f64::from(*w);
        }
        xy
    }

    /// Where the person is now and what the network thinks they do, once every pair has a history.
    /// While they walk, the belief moves: spread by a second of walking, then weighed by this second's
    /// answer; while they do not, it stays where they stopped.
    pub fn position(&mut self, walking: bool) -> Option<([f64; 2], &str)> {
        let (where_now, activity) = self.answer(&self.tokens()?);
        if walking {
            let spread: Vec<f32> = self.walk.par_iter().map(|row| row.iter().zip(&self.belief).map(|(m, b)| m * b).sum()).collect();
            let weighed: Vec<f32> = spread.iter().zip(&where_now).map(|(s, p)| s * p.max(1e-12)).collect();
            let total: f32 = weighed.iter().sum();
            if total > 0.0 && total.is_finite() {
                self.belief = weighed.into_iter().map(|w| w / total).collect();
            }
        }
        let best = (0..activity.len()).max_by(|&a, &b| activity[a].total_cmp(&activity[b]))?;
        Some((self.mean_point(&self.belief), &self.activities[best]))
    }
}

/// Row i: from each point, the share of walkers who reach point i a second later.
fn walk_spread(points: &[[f64; 2]]) -> Vec<Vec<f32>> {
    let kernel = |a: [f64; 2], b: [f64; 2]| (-((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)) / (2.0 * WALK_SPREAD_M * WALK_SPREAD_M)).exp();
    let from_totals: Vec<f64> = points.iter().map(|&from| points.iter().map(|&to| kernel(from, to)).sum()).collect();
    points
        .iter()
        .map(|&to| points.iter().zip(&from_totals).map(|(&from, total)| (kernel(from, to) / total) as f32).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_matches_the_trained_network_on_a_simulated_second() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../model/models/link_net.json");
        let check: Check = serde_json::from_str::<File>(&std::fs::read_to_string(&path).unwrap()).unwrap().check;
        let links: Vec<([f64; 2], [f64; 2])> = check.links.iter().map(|&[a, b]| (check.nodes[a], check.nodes[b])).collect();
        let names: Vec<String> = (0..links.len()).map(|i| i.to_string()).collect();
        let tracker = LinkNetTracker::load(&path, &links, &names, check.width, check.depth).unwrap();
        let (weights, activity) = tracker.answer(&check.tokens);
        let xy = tracker.mean_point(&weights);
        let expected = softmax(&check.activity);
        assert!((xy[0] - check.position[0]).abs() < 1e-3 && (xy[1] - check.position[1]).abs() < 1e-3, "{xy:?} vs {:?}", check.position);
        assert!(activity.iter().zip(&expected).all(|(a, e)| (a - e).abs() < 1e-3), "{activity:?} vs {expected:?}");
    }

    #[test]
    fn both_directions_of_a_pair_become_one_token() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../model/models/link_net.json");
        let (a, b, c) = ([0.2, 0.2], [3.8, 0.2], [0.2, 3.3]);
        let links = [(a, b), (b, a), (a, c), (b, c)];
        let names: Vec<String> = ["ab", "ba", "ac", "bc"].map(String::from).to_vec();
        let mut tracker = LinkNetTracker::load(&path, &links, &names, 4.0, 3.5).unwrap();
        assert_eq!(tracker.pairs, vec![vec![0, 1], vec![2], vec![3]]);
        for second in 0..30 {
            let busy = if second >= 25 { 4.0 } else { 1.0 };
            let reading = |motion: f64| Some(LinkSecond { motion, shadow: 1.0, spread: vec![1.0; 52], shape: vec![0.0; 52] });
            tracker.add_second(&[reading(busy), None, reading(1.0), reading(1.1)]);
        }
        let (xy, _) = tracker.position(true).unwrap();
        assert!(xy[0] > 0.0 && xy[0] < 4.0 && xy[1] > 0.0 && xy[1] < 3.5);
    }
}
