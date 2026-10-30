//! What a hall node does with the CSI it hears: per-packet subcarrier-group levels, reduced to one
//! `ReportRecord` per source per second. Shared by the firmware and by the host when it plays a simulated hall.

use libm::{log10f, roundf, sqrtf};

use crate::{CsiBuffer, ReportRecord, L_LTF_DATA_SUBCARRIERS, N_GROUPS};
const SUBCARRIERS: usize = L_LTF_DATA_SUBCARRIERS.len();
const PER_GROUP: usize = SUBCARRIERS / N_GROUPS;
const MIN_POWER: f32 = 0.5;
const SUBCARRIER_PLUS_ONE: usize = 26;

pub fn subcarrier_powers(csi: &CsiBuffer, first_word_invalid: bool) -> [f32; SUBCARRIERS] {
    let mut powers = L_LTF_DATA_SUBCARRIERS.map(|k| {
        let sample = csi
            .subcarrier(k)
            .expect("L-LTF data subcarriers are inside the buffer");
        (sample.power() as f32).max(MIN_POWER)
    });
    // The driver may garble subcarriers 0 and +1; +2 stands in for +1.
    if first_word_invalid {
        powers[SUBCARRIER_PLUS_ONE] = powers[SUBCARRIER_PLUS_ONE + 1];
    }
    powers
}

pub fn group_levels_db(powers: &[f32; SUBCARRIERS], rssi_dbm: f32) -> [f32; N_GROUPS] {
    let mean = powers.iter().sum::<f32>() / SUBCARRIERS as f32;
    core::array::from_fn(|group| {
        let members = &powers[group * PER_GROUP..(group + 1) * PER_GROUP];
        let relative_db: f32 = members.iter().map(|&p| 10.0 * log10f(p / mean)).sum();
        relative_db / PER_GROUP as f32 + rssi_dbm
    })
}

pub const RSSI_OUTLIER_DB: f32 = 8.0;
const RSSI_EMA_ALPHA: f32 = 0.1;
pub const MIN_PACKETS_PER_REPORT: u16 = 3;

#[derive(Clone, Copy, Debug, Default)]
pub struct LinkWindow {
    level_sum: [f32; N_GROUPS],
    level_square_sum: [f32; N_GROUPS],
    packets: u16,
    rssi_ema: Option<f32>,
}

impl LinkWindow {
    /// Returns false when the packet's RSSI jumps too far from the running average (collisions, interference).
    pub fn add(&mut self, levels_db: &[f32; N_GROUPS], rssi_dbm: f32) -> bool {
        let ema = self.rssi_ema.get_or_insert(rssi_dbm);
        if (rssi_dbm - *ema).abs() > RSSI_OUTLIER_DB {
            return false;
        }
        *ema += RSSI_EMA_ALPHA * (rssi_dbm - *ema);
        for ((sum, square_sum), &level) in self
            .level_sum
            .iter_mut()
            .zip(&mut self.level_square_sum)
            .zip(levels_db)
        {
            *sum += level;
            *square_sum += level * level;
        }
        self.packets = self.packets.saturating_add(1);
        true
    }

    /// Closes the window: the record if enough packets arrived, and a fresh window that remembers the RSSI average.
    pub fn close(&mut self, source: u8) -> Option<ReportRecord> {
        let closed = core::mem::replace(
            self,
            LinkWindow {
                rssi_ema: self.rssi_ema,
                ..LinkWindow::default()
            },
        );
        closed.record(source)
    }

    fn record(&self, source: u8) -> Option<ReportRecord> {
        if self.packets < MIN_PACKETS_PER_REPORT {
            return None;
        }
        let n = f32::from(self.packets);
        let mut record = ReportRecord {
            source,
            packets: self.packets.min(255) as u8,
            mean_q4: [0; N_GROUPS],
            std_q20: [0; N_GROUPS],
        };
        for group in 0..N_GROUPS {
            let mean = self.level_sum[group] / n;
            let variance = self.level_square_sum[group] / n - mean * mean;
            let std = if variance > 0.0 { sqrtf(variance) } else { 0.0 };
            record.mean_q4[group] = roundf(mean * 4.0) as i16;
            record.std_q20[group] = roundf(std * 20.0).min(255.0) as u8;
        }
        Some(record)
    }
}

#[cfg(test)]
mod level_tests {
    use super::*;
    use crate::CSI_BUFFER_LEN;

    fn flat_buffer() -> CsiBuffer {
        let mut bytes = [0u8; CSI_BUFFER_LEN];
        bytes.iter_mut().for_each(|b| *b = 3);
        CsiBuffer::from_unsigned_bytes(bytes)
    }

    #[test]
    fn flat_channel_sits_at_the_rssi() {
        let levels = group_levels_db(&subcarrier_powers(&flat_buffer(), false), -60.0);
        assert!(levels.iter().all(|&l| (l + 60.0).abs() < 1e-4));
    }

    #[test]
    fn dead_subcarriers_are_floored_not_infinite() {
        let silent = CsiBuffer::from_unsigned_bytes([0; CSI_BUFFER_LEN]);
        let levels = group_levels_db(&subcarrier_powers(&silent, false), -70.0);
        assert!(levels.iter().all(|l| l.is_finite()));
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;

    #[test]
    fn needs_three_packets() {
        let mut window = LinkWindow::default();
        window.add(&[-60.0; N_GROUPS], -60.0);
        window.add(&[-60.0; N_GROUPS], -60.0);
        assert!(window.close(1).is_none());
    }

    #[test]
    fn quantizes_mean_and_spread() {
        let mut window = LinkWindow::default();
        for level in [-60.0, -61.0, -62.0] {
            window.add(&[level; N_GROUPS], -60.0);
        }
        let record = window.close(4).unwrap();
        assert_eq!((record.source, record.packets), (4, 3));
        assert_eq!(record.mean_q4[0], -244);
        assert_eq!(record.std_q20[0], 16);
    }

    #[test]
    fn drops_rssi_outliers_and_keeps_the_average_across_windows() {
        let mut window = LinkWindow::default();
        assert!(window.add(&[-60.0; N_GROUPS], -60.0));
        assert!(!window.add(&[-60.0; N_GROUPS], -40.0));
        window.close(0);
        assert!(!window.add(&[-60.0; N_GROUPS], -45.0));
    }
}
