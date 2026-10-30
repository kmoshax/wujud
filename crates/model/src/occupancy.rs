//! Occupancy latch: motion latches "someone is here"; a quiet room is then compared with a fresh empty reference.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::numeric::median;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Occupancy {
    Unknown,
    Empty,
    Moving,
    Still,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LatchConfig {
    /// Profile columns compared with the reference; `None` uses all.
    pub columns: Option<Vec<usize>>,
    pub threshold: f64,
    pub reference_len: usize,
    pub min_reference: usize,
    pub clear_after: usize,
    /// Windows after which the empty reference is no longer trusted.
    pub reference_max_age: i64,
    /// dB: integer I/Q and AGC make anything finer meaningless.
    pub sd_floor: f64,
    /// Without motion since the room was last empty, a profile change is more likely drift than a person.
    pub no_entry_factor: f64,
}

impl Default for LatchConfig {
    fn default() -> Self {
        LatchConfig {
            columns: None,
            threshold: 3.0,
            reference_len: 24,
            min_reference: 6,
            clear_after: 3,
            reference_max_age: 120,
            sd_floor: 0.2,
            no_entry_factor: 3.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OccupancyLatch {
    pub config: LatchConfig,
    reference: VecDeque<Vec<f64>>,
    state: Occupancy,
    windows_seen: i64,
    quiet_windows_near_reference: usize,
    reference_updated_at: i64,
    motion_since_empty: bool,
}

const MAD_TO_SIGMA: f64 = 1.4826;

impl OccupancyLatch {
    pub fn new(config: LatchConfig) -> Self {
        OccupancyLatch {
            reference: VecDeque::with_capacity(config.reference_len),
            config,
            state: Occupancy::Unknown,
            windows_seen: 0,
            quiet_windows_near_reference: 0,
            reference_updated_at: -1_000_000_000,
            motion_since_empty: false,
        }
    }

    pub fn state(&self) -> Occupancy {
        self.state
    }

    pub fn reference_len(&self) -> usize {
        self.reference.len()
    }

    pub fn clear_reference(&mut self) {
        self.reference.clear();
    }

    fn compared_columns(&self, width: usize) -> Vec<usize> {
        self.config
            .columns
            .clone()
            .unwrap_or_else(|| (0..width).collect())
    }

    /// Robust z-distance (RMS over columns) of a window's static profile from the empty reference.
    pub fn distance(&self, x: &[f64]) -> f64 {
        let columns = self.compared_columns(x.len());
        let squared_z: Vec<f64> = columns
            .iter()
            .map(|&c| {
                let column: Vec<f64> = self.reference.iter().map(|r| r[c]).collect();
                let center = median(&column);
                let deviations: Vec<f64> = column.iter().map(|v| (v - center).abs()).collect();
                let sd = (MAD_TO_SIGMA * median(&deviations)).max(self.config.sd_floor);
                ((x[c] - center) / sd).powi(2)
            })
            .collect();
        (squared_z.iter().sum::<f64>() / squared_z.len() as f64).sqrt()
    }

    fn remember_empty(&mut self, x: &[f64]) {
        if self.reference.len() == self.config.reference_len {
            self.reference.pop_front();
        }
        self.reference.push_back(x.to_vec());
        self.reference_updated_at = self.windows_seen;
    }

    fn reference_is_fresh(&self) -> bool {
        self.reference.len() >= self.config.min_reference
            && self.windows_seen - self.reference_updated_at <= self.config.reference_max_age
    }

    /// One window: its static profile `x`, the motion probability and (optionally) a still-vs-empty probability
    /// used only without a fresh reference. Returns the state and the distance, when one was computed.
    pub fn step(
        &mut self,
        x: &[f64],
        p_motion: f64,
        p_still: Option<f64>,
    ) -> (Occupancy, Option<f64>) {
        self.windows_seen += 1;
        if p_motion > 0.5 {
            self.state = Occupancy::Moving;
            self.quiet_windows_near_reference = 0;
            self.motion_since_empty = true;
            return (self.state, None);
        }
        if !self.reference_is_fresh() {
            self.state = self.state_without_reference(p_still);
            if self.state == Occupancy::Empty {
                self.remember_empty(x);
            }
            return (self.state, None);
        }
        let distance = self.distance(x);
        self.judge_quiet_window(distance);
        if self.state == Occupancy::Empty {
            self.remember_empty(x);
            self.motion_since_empty = false;
        }
        (self.state, Some(distance))
    }

    fn state_without_reference(&self, p_still: Option<f64>) -> Occupancy {
        match (p_still, self.state) {
            (Some(p), _) if p > 0.5 => Occupancy::Still,
            (Some(_), _) => Occupancy::Empty,
            (None, Occupancy::Moving | Occupancy::Still) => Occupancy::Still,
            (None, _) => Occupancy::Empty,
        }
    }

    fn judge_quiet_window(&mut self, distance: f64) {
        let threshold = if self.motion_since_empty || self.state == Occupancy::Still {
            self.config.threshold
        } else {
            self.config.threshold * self.config.no_entry_factor
        };
        if distance > threshold {
            self.state = Occupancy::Still;
            self.quiet_windows_near_reference = 0;
        } else {
            self.quiet_windows_near_reference += 1;
            if self.state != Occupancy::Still
                || self.quiet_windows_near_reference >= self.config.clear_after
            {
                self.state = Occupancy::Empty;
            }
        }
    }
}
