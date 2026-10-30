//! Carrier, channel width, subcarrier layout, channel hopping and the frequency-dependent body/radio effects.

use serde::{Deserialize, Serialize};

use crate::constants::{CARRIER_HZ, SPEED_OF_LIGHT};
use crate::room::BODY_ABSORPTION_RANGE;

/// Which CSI the board reports: the long training field of an 802.11 frame and its channel width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubcarrierLayout {
    /// HT20 legacy L-LTF: indices ±1..±26 at 312.5 kHz (ESP32, `ESPectre` recordings, the hall firmware).
    LegacyLtf52,
    /// HT20 HT-LTF: indices ±1..±28 at 312.5 kHz (the room firmware, the simulator's default).
    HtLtf56,
    /// HT40 HT-LTF: indices ±2..±58 at 312.5 kHz.
    Ht40Ltf114,
    /// 802.11ax HE20 HE-LTF (RU 242): indices ±2..±122 at 78.125 kHz (ESP32-C5 / C6).
    He20Ltf242,
}

impl SubcarrierLayout {
    pub fn spacing_hz(self) -> f64 {
        match self {
            SubcarrierLayout::He20Ltf242 => 78.125e3,
            _ => 312.5e3,
        }
    }

    pub fn channel_width_hz(self) -> f64 {
        match self {
            SubcarrierLayout::Ht40Ltf114 => 40e6,
            _ => 20e6,
        }
    }

    /// Signed subcarrier indices around the carrier (DC and its neighbours excluded).
    pub fn indices(self) -> Vec<i32> {
        let (first, last) = match self {
            SubcarrierLayout::LegacyLtf52 => (1, 26),
            SubcarrierLayout::HtLtf56 => (1, 28),
            SubcarrierLayout::Ht40Ltf114 => (2, 58),
            SubcarrierLayout::He20Ltf242 => (2, 122),
        };
        (-last..=-first).chain(first..=last).collect()
    }

    pub fn count(self) -> usize {
        self.indices().len()
    }

    pub fn frequencies(self, carrier_hz: f64) -> Vec<f64> {
        let spacing = self.spacing_hz();
        self.indices()
            .into_iter()
            .map(|index| carrier_hz + f64::from(index) * spacing)
            .collect()
    }

    /// Per-subcarrier SNR change against thermal noise relative to 20 MHz: the same transmit power is spread
    /// over a wider band (HE20 has 4× the subcarriers but each sees 1/4 of the noise bandwidth: no change).
    pub fn thermal_snr_offset_db(self) -> f64 {
        -10.0 * (self.channel_width_hz() / 20e6).log10()
    }
}

/// The carrier of a symmetric subcarrier grid (exact for every layout above).
pub fn carrier_of(frequencies: &[f64]) -> f64 {
    match (frequencies.first(), frequencies.last()) {
        (Some(low), Some(high)) => 0.5 * (low + high),
        _ => CARRIER_HZ,
    }
}

pub fn wavelength_of(frequencies: &[f64]) -> f64 {
    SPEED_OF_LIGHT / carrier_of(frequencies)
}

pub fn wavelength_at(carrier_hz: f64) -> f64 {
    SPEED_OF_LIGHT / carrier_hz
}

/// Frequency-dependent properties of bodies and radios that the geometry alone does not give.
/// The default is the 2.4 GHz simulator as calibrated; the 5 GHz settings and their sources are in the README.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrequencyEffects {
    /// Range the body-box absorption `η` is mapped onto; `None` keeps the 2.4 GHz draw.
    pub body_absorption_range: Option<(f64, f64)>,
    /// Absorption cross-section of one person in the Sabine factor (m²).
    pub person_absorption_m2: f64,
    pub tx_power_offset_db: f64,
    pub evm_offset_db: f64,
}

impl Default for FrequencyEffects {
    fn default() -> Self {
        FrequencyEffects {
            body_absorption_range: None,
            person_absorption_m2: crate::radio::PERSON_ABSORPTION_M2,
            tx_power_offset_db: 0.0,
            evm_offset_db: 0.0,
        }
    }
}

impl FrequencyEffects {
    pub fn body_absorption(&self, drawn: f64) -> f64 {
        let Some((low, high)) = self.body_absorption_range else {
            return drawn;
        };
        let (drawn_low, drawn_high) = BODY_ABSORPTION_RANGE;
        low + (drawn - drawn_low) / (drawn_high - drawn_low) * (high - low)
    }
}

/// What the link is measured on: one or more channels visited round-robin (packet `t` uses channel
/// `t mod N`, so the total packet rate is shared), the subcarrier layout and the frequency effects.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BandPlan {
    pub carriers_hz: Vec<f64>,
    pub layout: SubcarrierLayout,
    pub effects: FrequencyEffects,
}

impl Default for BandPlan {
    fn default() -> Self {
        BandPlan::single(CARRIER_HZ, SubcarrierLayout::HtLtf56)
    }
}

impl BandPlan {
    pub fn single(carrier_hz: f64, layout: SubcarrierLayout) -> Self {
        BandPlan {
            carriers_hz: vec![carrier_hz],
            layout,
            effects: FrequencyEffects::default(),
        }
    }

    pub fn hopping(carriers_hz: Vec<f64>, layout: SubcarrierLayout) -> Self {
        BandPlan {
            carriers_hz,
            layout,
            effects: FrequencyEffects::default(),
        }
    }

    pub fn with_effects(mut self, effects: FrequencyEffects) -> Self {
        self.effects = effects;
        self
    }

    pub fn channel_count(&self) -> usize {
        self.carriers_hz.len()
    }

    pub fn channel_of_frame(&self, frame: usize) -> usize {
        frame % self.channel_count()
    }

    /// The subcarrier frequencies of every channel.
    pub fn channel_frequencies(&self) -> Vec<Vec<f64>> {
        self.carriers_hz
            .iter()
            .map(|&carrier| self.layout.frequencies(carrier))
            .collect()
    }
}

/// 2.4 GHz channel `n` (1..13): 2407 + 5n MHz.
pub fn channel_2g4_hz(channel: u32) -> f64 {
    (2407.0 + 5.0 * f64::from(channel)) * 1e6
}

/// 5 GHz channel `n`: 5000 + 5n MHz.
pub fn channel_5g_hz(channel: u32) -> f64 {
    (5000.0 + 5.0 * f64::from(channel)) * 1e6
}
