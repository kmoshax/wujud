//! Exact comparisons with research/hallsense/physics3d.py (golden data: research/tools/export_golden_sim.py).

mod common;

use common::{json_complex, json_complexes, json_f64s, json_point, read_json, relative_error};
use hallsense_sim::constants::{ht_ltf_frequencies, WAVELENGTH};
use hallsense_sim::fresnel::{fresnel_integrals, fresnel_scale, fresnel_zone_radius};
use hallsense_sim::paths::{
    diffuse_paths, geometric_paths, DiffuseDraws, Enclosure, PropagationPath, RoomShape,
};
use hallsense_sim::radio::quantize_to_int8;
use hallsense_sim::scatter::{static_rest, Scatterer};
use num_complex::Complex64;

#[test]
fn wavelength_matches_config() {
    let golden = read_json("physics.json");
    assert_eq!(WAVELENGTH, golden["wavelength"].as_f64().unwrap());
}

#[test]
fn fresnel_integrals_match_scipy() {
    let golden = &read_json("physics.json")["fresnel"];
    let xs = json_f64s(&golden["x"]);
    let s = json_f64s(&golden["s"]);
    let c = json_f64s(&golden["c"]);
    for (k, &x) in xs.iter().enumerate() {
        let (rs, rc) = fresnel_integrals(x);
        assert!(
            (rs - s[k]).abs() < 1e-14 && (rc - c[k]).abs() < 1e-14,
            "x={x}: ({rs}, {rc}) vs ({}, {})",
            s[k],
            c[k]
        );
    }
}

#[test]
fn fresnel_scale_matches_segment_shadow() {
    let golden = &read_json("physics.json")["fresnel"];
    let total = golden["scale_total"].as_f64().unwrap();
    let expected = json_f64s(&golden["scale"]);
    for (d1, kf) in json_f64s(&golden["scale_from_tx"])
        .into_iter()
        .zip(expected)
    {
        assert!(
            (fresnel_scale(d1, total) - kf).abs() <= 1e-12 * kf,
            "d1={d1}"
        );
    }
}

#[test]
fn first_fresnel_radius_is_sqrt2_over_scale() {
    for (d1, total) in [(1.0, 4.0), (2.5, 5.0), (0.3, 12.0)] {
        let radius = fresnel_zone_radius(1, d1, total - d1);
        assert!((radius * fresnel_scale(d1, total) - 2f64.sqrt()).abs() < 1e-12);
    }
}

fn assert_paths_match(actual: &[PropagationPath], expected: &serde_json::Value) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len(), "path count");
    let frequencies = ht_ltf_frequencies();
    for (path, golden) in actual.iter().zip(expected) {
        let points: Vec<_> = golden["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(json_point)
            .collect();
        assert_eq!(path.points.len(), points.len());
        for (p, e) in path.points.iter().zip(&points) {
            assert!(
                (0..3).all(|i| (p[i] - e[i]).abs() < 1e-12),
                "{p:?} vs {e:?}"
            );
        }
        assert!((path.gain - json_complex(&golden["gain"])).norm() <= 1e-12 * path.gain.norm());
        assert_eq!(path.shadow_scale, golden["eta_scale"].as_f64().unwrap());
        assert_eq!(path.shadowable, golden["shadow"].as_bool().unwrap());
        let response = path.response(&frequencies);
        assert!(relative_error(&response, &json_complexes(&golden["response56"])) < 1e-12);
    }
}

#[test]
fn image_method_paths_match() {
    for case in read_json("physics.json")["geometric"].as_array().unwrap() {
        let wall_gains = json_f64s(&case["wall_gains"]);
        let enclosure = Enclosure {
            shape: RoomShape {
                width: case["width"].as_f64().unwrap(),
                depth: case["depth"].as_f64().unwrap(),
                height: case["height"].as_f64().unwrap(),
            },
            floor_reflects: case["floor"].as_bool().unwrap(),
            wall_gains: [wall_gains[0], wall_gains[1], wall_gains[2], wall_gains[3]],
        };
        let paths = geometric_paths(json_point(&case["a"]), json_point(&case["b"]), &enclosure);
        assert_paths_match(&paths, &case["paths"]);
    }
}

#[test]
fn diffuse_paths_match_with_injected_draws() {
    let case = &read_json("physics.json")["diffuse"];
    let draws = DiffuseDraws {
        via_points: case["via"]
            .as_array()
            .unwrap()
            .iter()
            .map(json_point)
            .collect(),
        rician_k_db: case["rician_k_db"].as_f64().unwrap(),
        delay_spread_s: case["delay_spread_s"].as_f64().unwrap(),
        phase_turns: json_f64s(&case["phase_turns"]),
    };
    let paths = diffuse_paths(json_point(&case["a"]), json_point(&case["b"]), &draws);
    assert_paths_match(&paths, &case["paths"]);
}

#[test]
fn static_rest_matches() {
    let case = &read_json("physics.json")["static_rest"];
    let clutter: Vec<Scatterer> = case["clutter"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let v = json_f64s(row);
            Scatterer {
                position: [v[0], v[1], v[2]],
                rcs: v[3],
            }
        })
        .collect();
    let rest = static_rest(
        json_point(&case["a"]),
        json_point(&case["b"]),
        &clutter,
        &ht_ltf_frequencies(),
    );
    assert!(relative_error(&rest, &json_complexes(&case["rest56"])) < 1e-12);
}

#[test]
fn quantization_rounds_half_to_even_and_clips() {
    let case = &read_json("physics.json")["quantize"];
    let expected: Vec<i64> = case["int8"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    for (value, want) in json_f64s(&case["values"]).into_iter().zip(expected) {
        assert_eq!(i64::from(quantize_to_int8(value)), want, "value {value}");
    }
}

#[test]
fn path_response_is_free_space_at_unit_gain() {
    let length = 3.0;
    let response =
        hallsense_sim::paths::path_response(length, Complex64::new(1.0, 0.0), &[2.437e9]);
    assert!(
        (response[0].norm() - WAVELENGTH / (4.0 * std::f64::consts::PI * length)).abs() < 1e-15
    );
}
