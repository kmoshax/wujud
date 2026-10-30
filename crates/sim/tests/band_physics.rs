//! Carrier, subcarrier layouts and channel hopping: wavelength enters every physical term consistently,
//! and the default band reproduces the simulator exactly.

use std::f64::consts::PI;

use hallsense_sim::band::{
    carrier_of, channel_2g4_hz, channel_5g_hz, wavelength_at, BandPlan, FrequencyEffects,
    SubcarrierLayout,
};
use hallsense_sim::channel::Link;
use hallsense_sim::constants::{
    ht_ltf_frequencies, l_ltf_frequencies, CARRIER_HZ, SPEED_OF_LIGHT, WAVELENGTH,
};
use hallsense_sim::fresnel::{
    fresnel_scale, fresnel_scale_at, fresnel_zone_radius, fresnel_zone_radius_at,
};
use hallsense_sim::paths::path_response;
use hallsense_sim::random::seeded;
use hallsense_sim::room::{random_room, Activity, PlanStep, RoomOptions};
use hallsense_sim::room_sim::{simulate_room, simulate_room_on_band, RoomSession, RoomSimOptions};
use hallsense_sim::scatter::{add_moving_scatterer, MovingScatterer};
use hallsense_sim::shadow::{path_shadow, ShadowCasters};
use ndarray::Array2;
use num_complex::Complex64;

const FIVE_GHZ: f64 = 5.18e9;

#[test]
fn subcarrier_grids_have_the_standard_size_spacing_and_span() {
    let cases = [
        (SubcarrierLayout::LegacyLtf52, 52, 312.5e3, 26),
        (SubcarrierLayout::HtLtf56, 56, 312.5e3, 28),
        (SubcarrierLayout::Ht40Ltf114, 114, 312.5e3, 58),
        (SubcarrierLayout::He20Ltf242, 242, 78.125e3, 122),
    ];
    for (layout, count, spacing, edge) in cases {
        let frequencies = layout.frequencies(CARRIER_HZ);
        assert_eq!(frequencies.len(), count, "{layout:?}");
        assert_eq!(layout.spacing_hz(), spacing);
        let highest = frequencies.last().unwrap() - CARRIER_HZ;
        assert!((highest - f64::from(edge) * spacing).abs() < 1e-3);
        assert!(frequencies
            .iter()
            .all(|f| (f - CARRIER_HZ).abs() >= spacing * 0.99));
        assert_eq!(carrier_of(&frequencies), CARRIER_HZ, "{layout:?}");
    }
    assert_eq!(
        SubcarrierLayout::LegacyLtf52.frequencies(CARRIER_HZ),
        l_ltf_frequencies()
    );
    assert_eq!(
        SubcarrierLayout::HtLtf56.frequencies(CARRIER_HZ),
        ht_ltf_frequencies()
    );
    let ht40_dc_gap = SubcarrierLayout::Ht40Ltf114.indices();
    assert!(!ht40_dc_gap.contains(&-1) && !ht40_dc_gap.contains(&1));
    assert!((SubcarrierLayout::Ht40Ltf114.thermal_snr_offset_db() + 3.0103).abs() < 1e-4);
    assert_eq!(SubcarrierLayout::He20Ltf242.thermal_snr_offset_db(), 0.0);
}

#[test]
fn channel_numbers_map_to_carriers() {
    assert_eq!(channel_2g4_hz(6), CARRIER_HZ);
    assert_eq!(channel_2g4_hz(1), 2.412e9);
    assert_eq!(channel_5g_hz(36), FIVE_GHZ);
}

#[test]
fn fresnel_zone_shrinks_with_the_square_root_of_wavelength() {
    let (short, long) = (wavelength_at(FIVE_GHZ), WAVELENGTH);
    for (d1, d2) in [(1.0, 3.0), (2.0, 2.0), (0.4, 4.6)] {
        let ratio =
            fresnel_zone_radius_at(short, 1, d1, d2) / fresnel_zone_radius_at(long, 1, d1, d2);
        assert!((ratio - (short / long).sqrt()).abs() < 1e-12);
        assert_eq!(
            fresnel_zone_radius_at(WAVELENGTH, 1, d1, d2),
            fresnel_zone_radius(1, d1, d2)
        );
        assert_eq!(
            fresnel_scale_at(WAVELENGTH, d1, d1 + d2),
            fresnel_scale(d1, d1 + d2)
        );
        let at_5ghz =
            fresnel_zone_radius_at(short, 1, d1, d2) * fresnel_scale_at(short, d1, d1 + d2);
        assert!((at_5ghz - 2f64.sqrt()).abs() < 1e-12);
    }
}

#[test]
fn path_amplitude_follows_friis_and_phase_follows_frequency() {
    let length = 4.3;
    let gain = Complex64::new(-0.4, 0.0);
    let low = path_response(length, gain, &l_ltf_frequencies());
    let frequencies = SubcarrierLayout::LegacyLtf52.frequencies(FIVE_GHZ);
    let high = path_response(length, gain, &frequencies);
    let expected_loss_db = 20.0 * (FIVE_GHZ / CARRIER_HZ).log10();
    let loss_db = 20.0 * (low[0].norm() / high[0].norm()).log10();
    assert!((loss_db - expected_loss_db).abs() < 1e-9, "{loss_db}");
    for (h, f) in high.iter().zip(&frequencies) {
        let expected = gain
            * (wavelength_at(FIVE_GHZ) / (4.0 * PI * length))
            * Complex64::from_polar(1.0, -2.0 * PI * f * length / SPEED_OF_LIGHT);
        assert!((h - expected).norm() < 1e-15);
    }
}

#[test]
fn breathing_phase_scales_with_carrier_frequency() {
    let (tx, rx) = ([0.0, 0.0, 1.0], [4.0, 0.0, 1.0]);
    let reference = [2.0, 1.5, 1.0];
    let chest_out = [2.0, 1.503, 1.0];
    let track = [reference, chest_out];
    let path_change = {
        let length = |p: [f64; 3]| {
            let d = |a: [f64; 3], b: [f64; 3]| {
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
            };
            d(p, tx) + d(p, rx)
        };
        length(chest_out) - length(reference)
    };
    for carrier in [CARRIER_HZ, FIVE_GHZ] {
        let frequencies = SubcarrierLayout::LegacyLtf52.frequencies(carrier);
        let mut channel = Array2::zeros((2, frequencies.len()));
        let scatterer = MovingScatterer {
            track: &track,
            reference,
            rcs: 0.5,
        };
        add_moving_scatterer(&mut channel, tx, rx, &scatterer, &frequencies, 1.0);
        let rotation = (channel[[1, 0]] / channel[[0, 0]]).arg();
        let expected = -2.0 * PI * carrier * path_change / SPEED_OF_LIGHT;
        assert!(
            (rotation - expected).abs() < 1e-9,
            "{carrier}: {rotation} vs {expected}"
        );
    }
}

#[test]
fn body_shadow_uses_the_link_wavelength() {
    let room = random_room(&mut seeded(3, 0), &RoomOptions::default());
    let options = RoomSimOptions {
        seconds: 2.0,
        fs: 20.0,
        plan: Some(vec![PlanStep {
            activity: Activity::Stand,
            seconds: 2.0,
        }]),
        ..RoomSimOptions::default()
    };
    let session = RoomSession::new(&room, &options, &mut seeded(3, 1));
    let bodies = session.bodies();
    let (a, b) = (room.nodes[0], room.nodes[1]);
    let default = path_shadow(&[a, b], &ShadowCasters::new(bodies), 1.0);
    let explicit = path_shadow(
        &[a, b],
        &ShadowCasters::with_wavelength(bodies, WAVELENGTH),
        1.0,
    );
    assert_eq!(default, explicit);
    let link = Link {
        tx: a,
        rx: b,
        frequencies: &l_ltf_frequencies(),
    };
    assert_eq!(link.wavelength(), WAVELENGTH);
}

#[test]
fn body_absorption_is_remapped_only_when_asked() {
    let neutral = FrequencyEffects::default();
    assert_eq!(neutral.body_absorption(0.873), 0.873);
    let opaque = FrequencyEffects {
        body_absorption_range: Some((0.9, 0.98)),
        ..FrequencyEffects::default()
    };
    assert!((opaque.body_absorption(0.8) - 0.9).abs() < 1e-12);
    assert!((opaque.body_absorption(0.95) - 0.98).abs() < 1e-12);
}

fn short_options() -> RoomSimOptions {
    RoomSimOptions {
        seconds: 6.0,
        fs: 30.0,
        calibration_seconds: 1.0,
        plan: Some(vec![
            PlanStep {
                activity: Activity::Sit,
                seconds: 3.0,
            },
            PlanStep {
                activity: Activity::Walk,
                seconds: 3.0,
            },
        ]),
        ..RoomSimOptions::default()
    }
}

#[test]
fn default_band_reproduces_the_simulator_exactly() {
    let room = random_room(&mut seeded(11, 0), &RoomOptions::default());
    let options = short_options();
    let plain = simulate_room(&room, &options, &mut seeded(11, 1));
    let banded = simulate_room_on_band(&room, &options, &BandPlan::default(), &mut seeded(11, 1));
    assert_eq!(plain, banded);
}

#[test]
fn hopping_over_one_repeated_channel_equals_staying_on_it() {
    let room = random_room(&mut seeded(12, 0), &RoomOptions::default());
    let options = short_options();
    let layout = SubcarrierLayout::LegacyLtf52;
    let single = BandPlan::single(CARRIER_HZ, layout);
    let repeated = BandPlan::hopping(vec![CARRIER_HZ; 3], layout);
    let a = simulate_room_on_band(&room, &options, &single, &mut seeded(12, 1));
    let b = simulate_room_on_band(&room, &options, &repeated, &mut seeded(12, 1));
    assert_eq!(a, b);
}

#[test]
fn hopping_measures_frame_t_on_channel_t_mod_n() {
    let room = random_room(&mut seeded(13, 0), &RoomOptions::default());
    let options = short_options();
    let layout = SubcarrierLayout::LegacyLtf52;
    let carriers = vec![channel_2g4_hz(1), channel_2g4_hz(6), channel_2g4_hz(11)];
    let hopping = RoomSession::with_band(
        &room,
        &options,
        &BandPlan::hopping(carriers.clone(), layout),
        &mut seeded(13, 1),
    )
    .channels(0..options.session_frames());
    for (channel, &carrier) in carriers.iter().enumerate() {
        let fixed = RoomSession::with_band(
            &room,
            &options,
            &BandPlan::single(carrier, layout),
            &mut seeded(13, 1),
        )
        .channels(0..options.session_frames());
        for t in (channel..options.session_frames()).step_by(carriers.len()) {
            let error = (&hopping[0].row(t) - &fixed[0].row(t))
                .iter()
                .map(|d| d.norm())
                .fold(0.0, f64::max);
            assert!(error < 1e-15, "frame {t}: {error}");
        }
    }
}

#[test]
fn the_same_seed_gives_the_same_person_on_every_band() {
    let room = random_room(&mut seeded(14, 0), &RoomOptions::default());
    let options = short_options();
    let low = simulate_room(&room, &options, &mut seeded(14, 1));
    let band = BandPlan::single(FIVE_GHZ, SubcarrierLayout::He20Ltf242);
    let high = simulate_room_on_band(&room, &options, &band, &mut seeded(14, 1));
    assert_eq!(low.labels, high.labels);
    assert_eq!(high.links[0].session.iq.dim().1, 242);
}
