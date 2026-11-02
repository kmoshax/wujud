//! Which body movement is under way (still, walk, wave, bow, prostrate, sit down, stand up), from every
//! link's ten-per-second readings (research/tools/movement_net.py; trained on simulated rooms only, see
//! research/results/movements for how it does on rooms it never saw).
//!
//! Each 100 ms step of a link is its spectrum shape in 32 bins (dB, mean removed) and its mean RSSI, as
//! crates/app/examples/export_movements.rs computes them. Every link becomes inputs relative to itself
//! (shape change over 0.3 s and 1 s, departure from its 30 s mean, RSSI departures), divided by its own
//! typical 1 s change. A dilated 1-D CNN reads each link, the links are pooled (mean and max), and a
//! temporal head scores the movement. The answer is for the second that ended `delay_steps()` ago,
//! since the network looks as far ahead as it looks back.

#![cfg_attr(not(test), allow(dead_code))]

use std::collections::{HashMap, VecDeque};
use std::path::Path;

use serde::Deserialize;

pub const BINS: usize = 32;
pub const STEP_FEATURES: usize = BINS + 1;
const HZ: usize = 10;
const SHORT_LAG: usize = 3;
const MEAN_STEPS: usize = 300;
const TYPICAL_FLOOR: f32 = 0.05;
const INPUTS: usize = 3 * BINS + 2;

#[derive(Deserialize)]
#[serde(untagged)]
enum Tensor {
    Vector(Vec<f32>),
    Kernel(Vec<Vec<Vec<f32>>>),
}

#[derive(Deserialize)]
struct File {
    movements: Vec<String>,
    dilations: Vec<usize>,
    weights: HashMap<String, Tensor>,
    #[cfg_attr(not(test), allow(dead_code))]
    check: Check,
}

#[derive(Deserialize)]
#[cfg_attr(not(test), allow(dead_code))]
struct Check {
    /// steps x links x 33 from the start of a simulated room.
    raw: Vec<Vec<Vec<f32>>>,
    /// The network inputs of the last step, per link.
    last_inputs: Vec<Vec<f32>>,
    /// Logits (classes x 10) of the last whole second that has its full look-ahead in `raw`.
    logits: Vec<Vec<f32>>,
}

/// A 1-D convolution without padding: `out[o][t] = b[o] + Σ_i Σ_k w[o][i][k] x[i][t + k·dilation]`.
struct Conv {
    w: Vec<Vec<Vec<f32>>>,
    b: Vec<f32>,
    dilation: usize,
}

impl Conv {
    fn span(&self) -> usize {
        (self.w[0][0].len() - 1) * self.dilation
    }

    fn apply(&self, x: &[Vec<f32>], relu: bool) -> Vec<Vec<f32>> {
        let steps = x[0].len() - self.span();
        self.w
            .iter()
            .zip(&self.b)
            .map(|(kernels, &bias)| {
                let mut out = vec![bias; steps];
                for (row, kernel) in x.iter().zip(kernels) {
                    for (k, &w) in kernel.iter().enumerate() {
                        let shifted = &row[k * self.dilation..k * self.dilation + steps];
                        out.iter_mut().zip(shifted).for_each(|(o, v)| *o += w * v);
                    }
                }
                if relu {
                    out.iter_mut().for_each(|v| *v = v.max(0.0));
                }
                out
            })
            .collect()
    }
}

struct Network {
    encoder: Vec<Conv>,
    head: Conv,
    out: Conv,
}

impl Network {
    fn from_file(file: File) -> anyhow::Result<(Self, Vec<String>)> {
        let mut weights = file.weights;
        let mut conv = |name: &str, dilation: usize| -> anyhow::Result<Conv> {
            let w = match weights.remove(&format!("{name}.weight")) {
                Some(Tensor::Kernel(w)) => w,
                _ => anyhow::bail!("movement_net: no kernel {name}"),
            };
            let b = match weights.remove(&format!("{name}.bias")) {
                Some(Tensor::Vector(b)) => b,
                _ => anyhow::bail!("movement_net: no bias {name}"),
            };
            Ok(Conv { w, b, dilation })
        };
        let encoder = file
            .dilations
            .iter()
            .enumerate()
            .map(|(i, &d)| conv(&format!("encoder.{}", 2 * i), d))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let head = conv("head.0", 1)?;
        let out = conv("head.2", 1)?;
        Ok((Network { encoder, head, out }, file.movements))
    }

    /// How many steps before and after a step the network reads.
    fn half_span(&self) -> usize {
        (self.encoder.iter().map(Conv::span).sum::<usize>() + self.head.span()) / 2
    }

    /// links x (inputs x steps) -> logits (classes x (steps - 2·half_span)).
    fn logits(&self, links: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
        let encoded: Vec<Vec<Vec<f32>>> = links
            .iter()
            .map(|x| self.encoder.iter().fold(x.clone(), |h, layer| layer.apply(&h, true)))
            .collect();
        let n = encoded.len() as f32;
        let (width, steps) = (encoded[0].len(), encoded[0][0].len());
        let mut pooled = vec![vec![0.0f32; steps]; 2 * width];
        for c in 0..width {
            for t in 0..steps {
                let values = encoded.iter().map(|h| h[c][t]);
                pooled[c][t] = values.clone().sum::<f32>() / n;
                pooled[width + c][t] = values.fold(f32::NEG_INFINITY, f32::max);
            }
        }
        self.out.apply(&self.head.apply(&pooled, true), false)
    }
}

/// One link's raw steps and the network inputs computed from them.
#[derive(Default)]
struct LinkHistory {
    raw: VecDeque<[f32; STEP_FEATURES]>,
    first: Option<[f32; STEP_FEATURES]>,
    second_change: VecDeque<f32>,
    inputs: VecDeque<[f32; INPUTS]>,
    steps_seen: usize,
}

impl LinkHistory {
    fn lagged(&self, lag: usize) -> [f32; STEP_FEATURES] {
        let len = self.raw.len();
        if self.steps_seen > lag {
            self.raw[len - 1 - lag]
        } else {
            self.first.expect("a step was added")
        }
    }

    fn add(&mut self, step: [f32; STEP_FEATURES], keep: usize) {
        self.first.get_or_insert(step);
        self.raw.push_back(step);
        if self.raw.len() > MEAN_STEPS.max(HZ + 1) {
            self.raw.pop_front();
        }
        self.steps_seen += 1;
        let (short, second) = (self.lagged(SHORT_LAG), self.lagged(HZ));
        let mean_of = |i: usize| self.raw.iter().rev().take(MEAN_STEPS).map(|s| s[i]).sum::<f32>() / self.raw.len().min(MEAN_STEPS) as f32;
        let mut x = [0.0f32; INPUTS];
        for b in 0..BINS {
            x[b] = step[b] - short[b];
            x[BINS + b] = step[b] - second[b];
            x[2 * BINS + b] = step[b] - mean_of(b);
        }
        x[3 * BINS] = step[BINS] - mean_of(BINS);
        x[3 * BINS + 1] = step[BINS] - second[BINS];
        self.second_change.push_back(x[BINS..2 * BINS].iter().map(|v| v.abs()).sum::<f32>() / BINS as f32);
        if self.second_change.len() > MEAN_STEPS {
            self.second_change.pop_front();
        }
        let typical = self.second_change.iter().sum::<f32>() / self.second_change.len() as f32 + TYPICAL_FLOOR;
        x.iter_mut().for_each(|v| *v /= typical);
        self.inputs.push_back(x);
        if self.inputs.len() > keep {
            self.inputs.pop_front();
        }
    }
}

pub struct MovementReader {
    network: Network,
    movements: Vec<String>,
    links: Vec<LinkHistory>,
}

impl MovementReader {
    pub fn load(path: &Path, links: usize) -> anyhow::Result<Self> {
        let file: File = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let (network, movements) = Network::from_file(file)?;
        Ok(MovementReader {
            network,
            movements,
            links: (0..links).map(|_| LinkHistory::default()).collect(),
        })
    }

    pub fn movements(&self) -> &[String] {
        &self.movements
    }

    /// How many 100 ms steps after the end of a second its answer comes.
    pub fn delay_steps(&self) -> usize {
        self.network.half_span()
    }

    /// One 100 ms step of every link (spectrum shape and mean RSSI), in a fixed link order.
    pub fn add_step(&mut self, steps: &[[f32; STEP_FEATURES]]) {
        let keep = 2 * self.network.half_span() + HZ;
        for (link, &step) in self.links.iter_mut().zip(steps) {
            link.add(step, keep);
        }
    }

    /// Logits (classes x 10 steps) of the second that ended `delay_steps()` ago.
    fn second_logits(&self) -> Option<Vec<Vec<f32>>> {
        let keep = 2 * self.network.half_span() + HZ;
        if self.links.is_empty() || self.links.iter().any(|l| l.inputs.len() < keep) {
            return None;
        }
        let windows: Vec<Vec<Vec<f32>>> = self
            .links
            .iter()
            .map(|l| (0..INPUTS).map(|i| l.inputs.iter().map(|x| x[i]).collect()).collect())
            .collect();
        Some(self.network.logits(&windows))
    }

    /// The movement of the second that ended `delay_steps()` ago, with the class probabilities.
    pub fn movement(&self) -> Option<(&str, Vec<f32>)> {
        let logits = self.second_logits()?;
        let mean: Vec<f32> = logits.iter().map(|row| row.iter().sum::<f32>() / row.len() as f32).collect();
        let max = mean.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = mean.iter().map(|v| (v - max).exp()).collect();
        let total: f32 = exp.iter().sum();
        let probabilities: Vec<f32> = exp.iter().map(|v| v / total).collect();
        let best = (0..probabilities.len()).max_by(|&a, &b| probabilities[a].total_cmp(&probabilities[b]))?;
        Some((&self.movements[best], probabilities))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../model/models/movement_net.json")
    }

    fn reader_after_check() -> (MovementReader, Check) {
        let check: Check = serde_json::from_str::<File>(&std::fs::read_to_string(model_path()).unwrap()).unwrap().check;
        let mut reader = MovementReader::load(&model_path(), check.raw[0].len()).unwrap();
        for step in &check.raw {
            let steps: Vec<[f32; STEP_FEATURES]> = step.iter().map(|l| l.as_slice().try_into().unwrap()).collect();
            reader.add_step(&steps);
        }
        (reader, check)
    }

    #[test]
    fn inputs_match_the_python_features() {
        let (reader, check) = reader_after_check();
        for (link, expected) in reader.links.iter().zip(&check.last_inputs) {
            let got = link.inputs.back().unwrap();
            assert!(got.iter().zip(expected).all(|(a, e)| (a - e).abs() < 1e-3 * (1.0 + e.abs())), "{got:?} vs {expected:?}");
        }
    }

    #[test]
    fn rust_matches_the_trained_network_on_a_simulated_room() {
        let (reader, check) = reader_after_check();
        let logits = reader.second_logits().unwrap();
        for (row, expected) in logits.iter().zip(&check.logits) {
            assert!(row.iter().zip(expected).all(|(a, e)| (a - e).abs() < 2e-3 * (1.0 + e.abs())), "{row:?} vs {expected:?}");
        }
        let (said, probabilities) = reader.movement().unwrap();
        assert_eq!(probabilities.len(), reader.movements().len());
        assert!(reader.movements().iter().any(|m| m == said));
        assert!(reader.delay_steps() * 2 + 10 < check.raw.len());
    }
}
