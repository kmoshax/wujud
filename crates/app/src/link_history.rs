use std::collections::VecDeque;

use wujud_model::motion_tracker;

use crate::sensor_feed::CsiFrame;

const KEPT_PACKETS: usize = 1500;
const RATE_WINDOW_SECONDS: f64 = 2.0;
const ESP32_BINS: usize = 64;
const MIN_POWER: f32 = 0.5;
/// A subcarrier whose mean power stays below this is guard band or DC, not signal.
const SIGNAL_POWER: f32 = 2.0;
const DISTURBANCE_PACKETS: usize = 50;
/// Below this a link counts as silent (a phone asleep, a board unplugged), not as quiet.
const ALIVE_PACKETS_PER_SECOND: f64 = 10.0;
const SILENT_AFTER: std::time::Duration = std::time::Duration::from_secs(1);

pub struct LinkHistory {
    pub name: String,
    amplitudes_db: VecDeque<Vec<f32>>,
    rssi: VecDeque<i8>,
    newest_iq: VecDeque<Vec<i8>>,
    arrivals: VecDeque<f64>,
    last_packet: Option<std::time::Instant>,
}

impl LinkHistory {
    pub fn new(name: String) -> Self {
        LinkHistory {
            name,
            amplitudes_db: VecDeque::new(),
            rssi: VecDeque::new(),
            newest_iq: VecDeque::new(),
            arrivals: VecDeque::new(),
            last_packet: None,
        }
    }

    pub fn push(&mut self, frame: &CsiFrame, arrived_at: f64) {
        let amplitudes = relative_amplitudes_db(&frame.iq);
        if self
            .amplitudes_db
            .back()
            .is_some_and(|last| last.len() != amplitudes.len())
        {
            self.amplitudes_db.clear();
        }
        push_bounded(&mut self.amplitudes_db, amplitudes, KEPT_PACKETS);
        push_bounded(&mut self.rssi, frame.rssi, KEPT_PACKETS);
        if self
            .newest_iq
            .back()
            .is_some_and(|last| last.len() != frame.iq.len())
        {
            self.newest_iq.clear();
        }
        push_bounded(&mut self.newest_iq, frame.iq.clone(), DISTURBANCE_PACKETS);
        self.arrivals.push_back(arrived_at);
        self.last_packet = Some(std::time::Instant::now());
        while self
            .arrivals
            .front()
            .is_some_and(|&t| t < arrived_at - RATE_WINDOW_SECONDS)
        {
            self.arrivals.pop_front();
        }
    }

    fn alive(&self) -> bool {
        self.last_packet.is_some_and(|t| t.elapsed() < SILENT_AFTER)
            && self.packets_per_second() >= ALIVE_PACKETS_PER_SECOND
    }

    pub fn packets_per_second(&self) -> f64 {
        self.arrivals.len() as f64 / RATE_WINDOW_SECONDS
    }

    pub fn latest_rssi(&self) -> Option<i8> {
        self.rssi.back().copied()
    }

    pub fn rssi_history(&self) -> impl Iterator<Item = i8> + '_ {
        self.rssi.iter().copied()
    }

    /// `motion_tracker::disturbance_db` over the newest packets (~0.5 s).
    pub fn disturbance_db(&self) -> Option<f64> {
        if !self.alive() {
            return None;
        }
        let width = self.newest_iq.front()?.len();
        if self.newest_iq.len() < DISTURBANCE_PACKETS / 2 {
            return None;
        }
        let bytes: Vec<i8> = self.newest_iq.iter().flatten().copied().collect();
        let iq = ndarray::ArrayView2::from_shape((self.newest_iq.len(), width), &bytes).ok()?;
        Some(motion_tracker::disturbance_db(iq))
    }

    /// `motion_tracker::rssi_spread_db` over the newest packets (~0.5 s).
    pub fn rssi_spread_db(&self) -> Option<f64> {
        if !self.alive() {
            return None;
        }
        if self.rssi.len() < DISTURBANCE_PACKETS / 2 {
            return None;
        }
        let newest: Vec<f64> = self
            .rssi
            .iter()
            .rev()
            .take(DISTURBANCE_PACKETS)
            .map(|&r| f64::from(r))
            .collect();
        Some(motion_tracker::rssi_spread_db(&newest))
    }

    /// Over the newest packets, per signal subcarrier (guard band and DC left out, centred order): the
    /// spread of its dB level, and its mean level with the spectrum's mean removed, as the simulator's
    /// training rows (crates/app/examples/export_tracking.rs).
    pub fn subcarrier_db(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        if !self.alive() || self.newest_iq.len() < DISTURBANCE_PACKETS / 2 {
            return None;
        }
        let bins = self.newest_iq.front()?.len() / 2;
        let n = self.newest_iq.len() as f32;
        let mut power = vec![0.0f32; bins];
        for packet in &self.newest_iq {
            for (p, [i, q]) in power.iter_mut().zip(packet.as_chunks::<2>().0) {
                *p += (f32::from(*i).powi(2) + f32::from(*q).powi(2)) / n;
            }
        }
        let mut order: Vec<usize> = (0..bins).collect();
        if bins == ESP32_BINS && is_fft_order(&power) {
            order.rotate_left(ESP32_BINS / 2);
        }
        let signal: Vec<usize> = order.into_iter().filter(|&k| power[k] > SIGNAL_POWER).collect();
        if signal.len() < 2 {
            return None;
        }
        let db = |packet: &[i8], k: usize| 20.0 * (f32::from(packet[2 * k]).hypot(f32::from(packet[2 * k + 1])) + 1.0).log10();
        let (mut spread, mut level) = (Vec::with_capacity(signal.len()), Vec::with_capacity(signal.len()));
        for &k in &signal {
            let values: Vec<f32> = self.newest_iq.iter().map(|packet| db(packet, k)).collect();
            let mean = values.iter().sum::<f32>() / n;
            spread.push((values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt());
            level.push(mean);
        }
        let overall = level.iter().sum::<f32>() / level.len() as f32;
        Some((spread, level.into_iter().map(|v| v - overall).collect()))
    }

    /// The newest packets, oldest first: each subcarrier in dB over the packet's mean power.
    pub fn newest_amplitudes_db(&self, packets: usize) -> Vec<Vec<f32>> {
        let skip = self.amplitudes_db.len().saturating_sub(packets);
        self.amplitudes_db.iter().skip(skip).cloned().collect()
    }

    /// Each subcarrier as dB above or below its own average over the kept packets, oldest first.
    pub fn amplitude_changes_db(&self) -> Vec<Vec<f32>> {
        let Some(first) = self.amplitudes_db.front() else {
            return vec![];
        };
        let n = self.amplitudes_db.len() as f32;
        let mut average = vec![0.0f32; first.len()];
        for packet in &self.amplitudes_db {
            average
                .iter_mut()
                .zip(packet)
                .for_each(|(a, v)| *a += v / n);
        }
        self.amplitudes_db
            .iter()
            .map(|packet| packet.iter().zip(&average).map(|(v, a)| v - a).collect())
            .collect()
    }
}

/// Per-packet AGC removal, as the host features do: the shape across subcarriers, not the gain.
fn relative_amplitudes_db(iq: &[i8]) -> Vec<f32> {
    let mut power: Vec<f32> = iq
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[i, q]| (f32::from(*i).powi(2) + f32::from(*q).powi(2)).max(MIN_POWER))
        .collect();
    if power.len() == ESP32_BINS && is_fft_order(&power) {
        power.rotate_left(ESP32_BINS / 2);
    }
    let mean = power.iter().sum::<f32>() / power.len().max(1) as f32;
    power.iter().map(|p| 10.0 * (p / mean).log10()).collect()
}

/// The ESP32 buffer comes in FFT order (bins 0..31 then -32..-1, guard band in the middle) or in
/// centred order (-32..31, guard band at the ends); index 30 is guard in the first, signal in the second.
fn is_fft_order(power: &[f32]) -> bool {
    power[30] <= MIN_POWER && power[16] > MIN_POWER
}

fn push_bounded<T>(queue: &mut VecDeque<T>, value: T, capacity: usize) {
    if queue.len() == capacity {
        queue.pop_front();
    }
    queue.push_back(value);
}
