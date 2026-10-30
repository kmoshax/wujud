//! Radio constants shared by the simulator and the ESP32 firmware.

pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// Wi-Fi channel 6.
pub const CARRIER_HZ: f64 = 2.437e9;
pub const WAVELENGTH: f64 = SPEED_OF_LIGHT / CARRIER_HZ;
pub const SUBCARRIER_SPACING_HZ: f64 = 312.5e3;
pub const NOISE_FLOOR_DBM: f64 = -94.0;

/// The 52 L-LTF data subcarriers (indices -26..-1, 1..26) the hall firmware reports.
pub fn l_ltf_frequencies() -> Vec<f64> {
    subcarrier_frequencies(26)
}

/// The 56 HT-LTF data subcarriers (indices -28..-1, 1..28) the room firmware reports.
pub fn ht_ltf_frequencies() -> Vec<f64> {
    subcarrier_frequencies(28)
}

fn subcarrier_frequencies(half_width: i32) -> Vec<f64> {
    (-half_width..=half_width)
        .filter(|&index| index != 0)
        .map(|index| CARRIER_HZ + f64::from(index) * SUBCARRIER_SPACING_HZ)
        .collect()
}
