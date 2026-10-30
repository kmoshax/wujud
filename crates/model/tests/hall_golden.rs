//! Hall seat model, engine and windower against `research/wujud/model.py` and `research/live.py`.

mod common;

use common::*;
use ndarray::{Array2, Array3, Axis, Ix3};
use serde_json::Value;
use wujud_model::hall::{
    distance_bins, link_features, scene_vector, seat_descriptor, HallEngine, HallLayout,
    SceneEvidence, SeatModel, Windower, LINK_FEATURE_NAMES,
};

fn read_3d(name: &str) -> Array3<f32> {
    read_npy::<f32>(name).into_dimensionality::<Ix3>().unwrap()
}

fn points(value: &Value) -> Vec<[f64; 3]> {
    serde_json::from_value(value.clone()).unwrap()
}

fn golden_layout() -> HallLayout {
    let json = read_json("hall_layout");
    let nodes = points(&json["nodes"]);
    let links: Vec<[usize; 2]> = serde_json::from_value(json["links"].clone()).unwrap();
    HallLayout {
        width: json["W"].as_f64().unwrap(),
        depth: json["D"].as_f64().unwrap(),
        height: json["H"].as_f64().unwrap(),
        front: json["front"].as_f64().unwrap(),
        ids: (0..nodes.len() as i64).collect(),
        nodes,
        seats: points(&json["seats"]),
        links: links.iter().map(|l| (l[0], l[1])).collect(),
        tiers: Some(json["tiers"].clone()),
    }
}

fn hall_model() -> SeatModel {
    SeatModel::load(&model_path("hall3d.json")).unwrap()
}

fn to_f64(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| v as f64).collect()
}

/// Measured: <= 1.9e-7 relative (NumPy's float32 `log` is not correctly rounded); probabilities then bit-identical.
const FLOAT32_TOLERANCE: f64 = 5e-7;
const PROBABILITY_TOLERANCE: f64 = 1e-12;

#[test]
fn seat_model_matches_python() {
    let layout = golden_layout();
    let chests = layout.chest_points();
    let features = link_features(
        read_3d("hall_cal_mean").view(),
        read_3d("hall_cal_std").view(),
        read_3d("hall_ses_mean").view(),
        read_3d("hall_ses_std").view(),
    );
    let counts = read_json("hall_counts");
    for (k, name) in LINK_FEATURE_NAMES.iter().enumerate() {
        let expected: Vec<f64> = counts["link_features"][name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(f64::NAN))
            .collect();
        assert_close(name, &features.values[k], &expected, FLOAT32_TOLERANCE);
    }
    let bins = distance_bins(&layout.nodes, &layout.links, &chests);
    let descriptor = seat_descriptor(&features, &bins, &layout.nodes, &chests);
    let expected_descriptor = read_2d::<f32>("hall_seat_X");
    assert_close(
        "seat X",
        &to_f64(descriptor.as_slice().unwrap()),
        &to_f64(expected_descriptor.as_slice().unwrap()),
        FLOAT32_TOLERANCE,
    );
    let scene = scene_vector(
        &features,
        &layout.nodes,
        &layout.links,
        chests.len(),
        layout.width,
        layout.depth,
    );
    let expected_scene = read_1d::<f32>("hall_scene_vector");
    assert_close(
        "scene vector",
        &to_f64(&scene),
        &to_f64(expected_scene.as_slice().unwrap()),
        FLOAT32_TOLERANCE,
    );

    let evidence = SceneEvidence {
        link_features: features,
        seat_descriptor: descriptor,
        scene_vector: scene,
    };
    let prediction = hall_model().predict(&evidence, &bins, &chests).unwrap();
    let p1 = read_1d::<f64>("hall_p1");
    assert_close(
        "p1",
        &prediction.stage1,
        p1.as_slice().unwrap(),
        PROBABILITY_TOLERANCE,
    );
    let p = read_1d::<f64>("hall_p");
    assert_close(
        "p",
        &prediction.occupied,
        p.as_slice().unwrap(),
        PROBABILITY_TOLERANCE,
    );
    let count_sum = counts["count_sum"].as_f64().unwrap();
    assert_close(
        "count_sum",
        &[prediction.count_sum],
        &[count_sum],
        PROBABILITY_TOLERANCE,
    );
    let count_glob = counts["count_glob"].as_f64().unwrap();
    assert_close(
        "count_glob",
        &[prediction.count_global],
        &[count_glob],
        PROBABILITY_TOLERANCE,
    );
}

#[test]
fn hall_engine_publishes_python_states() {
    let (session_mean, session_std) = (read_3d("hall_ses_mean"), read_3d("hall_ses_std"));
    let truth = read_2d::<i64>("hall_truth");
    let expected = read_json("hall_engine_states");
    let mut engine = HallEngine::new(
        golden_layout(),
        hall_model(),
        read_3d("hall_cal_mean"),
        read_3d("hall_cal_std"),
        "REPLAY",
    );
    for w in 0..session_mean.len_of(Axis(1)) {
        let window = |x: &Array3<f32>| x.index_axis(Axis(1), w).to_owned();
        let truth_row = truth.row(w).to_vec();
        let state = engine
            .push(
                w as f64,
                window(&session_mean),
                window(&session_std),
                Some(&truth_row),
            )
            .unwrap();
        let mut json = serde_json::to_value(&state).unwrap();
        let object = json.as_object_mut().unwrap();
        for dropped in ["scene", "history", "t"] {
            object.remove(dropped);
        }
        assert_eq!(json, expected[w], "window {w}");
    }
}

#[test]
fn windower_matches_python() {
    let mut windower = Windower::new(&golden_layout());
    for line in read_json("hall_windower")["lines"].as_array().unwrap() {
        windower.add_line(line.as_str().unwrap());
    }
    let (mean, std) = windower.tick();
    let expected_mean: Array2<f32> = read_2d("hall_windower_mean");
    let expected_std: Array2<f32> = read_2d("hall_windower_std");
    let same = |a: &Array2<f32>, e: &Array2<f32>| {
        a.iter()
            .zip(e)
            .all(|(a, e)| a == e || (a.is_nan() && e.is_nan()))
    };
    assert!(same(&mean, &expected_mean));
    assert!(same(&std, &expected_std));
    assert!(windower.tick().0.iter().all(|v| v.is_nan()));
}

#[test]
fn grid_config_builds_seats_and_links() {
    let config = r#"{"W": 10, "D": 8, "front": 1.5, "nodes": {"3": [0, 0], "1": [10, 0, 2.0], "2": [5, 8]},
        "grid": {"rows": 2, "cols": 3, "x0": 1, "y0": 2, "seat_pitch": 0.5, "row_pitch": 1.0, "row_rise": 0.2,
                 "aisles_after_col": [0]}}"#;
    let hall = HallLayout::from_config_json(config).unwrap();
    assert_eq!(hall.ids, vec![1, 2, 3]);
    assert_eq!(hall.nodes[0], [10.0, 0.0, 2.0]);
    assert_eq!(hall.nodes[2], [0.0, 0.0, 1.1]);
    assert_eq!(hall.links, vec![(0, 1), (0, 2), (1, 2)]);
    assert_eq!(hall.seats[1], [2.5, 2.0, 0.0]);
    assert_eq!(hall.seats[5], [3.0, 3.0, 0.2]);
    assert!((hall.height - 3.7).abs() < 1e-12);
}
