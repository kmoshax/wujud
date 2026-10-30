//! The ESP32 receive chain (sim3d.py::measure) with numpy's own random draws injected.

mod common;

use common::Npz;
use hallsense_sim::radio::{measure_with_noise, LinkRadio, MeasureNoise};

/// Python and Rust sum the per-packet power in a different order, so a value within ~1e-15 of a
/// rounding boundary may land one int8/dB step apart.
const MAX_OFF_BY_ONE_FRACTION: f64 = 1e-4;

fn check_case(name: &str, expect_agc_steps: bool) {
    let mut golden = Npz::open(name);
    let h = golden.complex_matrix("H");
    let radio_values = golden.vector("radio");
    let radio = LinkRadio {
        tx_power_dbm: radio_values[0],
        evm_db: radio_values[1],
        agc_target: radio_values[2],
        loss_probability: radio_values[3],
        interference_probability: radio_values[4],
    };
    let fs = golden.scalar("fs");
    let noise = MeasureNoise {
        noise_re: golden.matrix("noise_re"),
        noise_im: golden.matrix("noise_im"),
        interference_draw: golden.vector("interference_draw"),
        burst_db: golden.vector("burst_db"),
        burst_re: golden.matrix("burst_re"),
        burst_im: golden.matrix("burst_im"),
        agc_step_draw: golden.vector("agc_step_draw"),
        rssi_bias_levels: golden.vector("rssi_bias_levels"),
        rssi_jitter: golden.vector("rssi_jitter"),
        agc_gain_db: golden.vector("agc_gain_db"),
        loss_draw: golden.vector("loss_draw"),
    };
    assert_eq!(
        noise.rssi_bias_levels.len() > 1,
        expect_agc_steps,
        "{name}: AGC steps"
    );
    let measured = measure_with_noise(&h, &radio, fs, &noise);

    let expected_iq = golden.vector("iq");
    let actual_iq: Vec<f64> = measured.iq.iter().map(|&v| f64::from(v)).collect();
    assert_close_integers(name, "iq", &actual_iq, &expected_iq);
    let actual_rssi: Vec<f64> = measured.rssi.iter().map(|&v| f64::from(v)).collect();
    assert_close_integers(name, "rssi", &actual_rssi, &golden.vector("rssi"));
    let expected_kept: Vec<bool> = golden.vector("kept").iter().map(|&k| k != 0.0).collect();
    assert_eq!(measured.kept, expected_kept, "{name}: kept");
}

fn assert_close_integers(case: &str, what: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{case} {what}: length");
    let mut off_by_one = 0;
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= 1.0, "{case} {what}: {a} vs {e}");
        if a != e {
            off_by_one += 1;
        }
    }
    let fraction = off_by_one as f64 / actual.len() as f64;
    assert!(
        fraction <= MAX_OFF_BY_ONE_FRACTION,
        "{case} {what}: {off_by_one} values off by one"
    );
}

#[test]
fn typical_link_matches_numpy() {
    check_case("measure_typical.npz", false);
}

#[test]
fn harsh_link_with_interference_and_agc_steps_matches() {
    check_case("measure_harsh.npz", true);
}
