#![allow(dead_code)]

use std::path::PathBuf;

use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2};
use ndarray_npy::{ReadNpyExt, ReadableElement};

pub fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

pub fn model_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name)
}

pub fn read_npy<T: ReadableElement>(name: &str) -> ArrayD<T> {
    let path = golden_path(&format!("{name}.npy"));
    let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    ArrayD::<T>::read_npy(file).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub fn read_1d<T: ReadableElement>(name: &str) -> Array1<T> {
    read_npy::<T>(name).into_dimensionality::<Ix1>().unwrap()
}

pub fn read_2d<T: ReadableElement>(name: &str) -> Array2<T> {
    read_npy::<T>(name).into_dimensionality::<Ix2>().unwrap()
}

pub fn read_2d_f32_as_f64(name: &str) -> Array2<f64> {
    read_2d::<f32>(name).mapv(f64::from)
}

pub fn read_json(name: &str) -> serde_json::Value {
    let path = golden_path(&format!("{name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

/// Largest `|actual - expected| / max(1, |expected|)`; NaN must meet NaN.
pub fn max_relative_error<'a>(
    actual: impl IntoIterator<Item = &'a f64>,
    expected: impl IntoIterator<Item = &'a f64>,
) -> f64 {
    let mut worst: f64 = 0.0;
    for (a, e) in actual.into_iter().zip(expected) {
        if a.is_nan() || e.is_nan() {
            assert!(a.is_nan() && e.is_nan(), "NaN mismatch: {a} vs {e}");
            continue;
        }
        worst = worst.max((a - e).abs() / e.abs().max(1.0));
    }
    worst
}

pub fn assert_close(label: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let error = max_relative_error(actual, expected);
    assert!(
        error <= tolerance,
        "{label}: max relative error {error:e} > {tolerance:e}"
    );
}

pub fn rows_to_array(rows: &[Vec<f64>], columns: usize) -> Array2<f64> {
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows.len(), columns), flat).unwrap()
}

pub use wujud_model::espectre::Recording;

pub const RECORDINGS: [&str; 4] = ["c3_empty", "c3_motion", "c3_still", "esp32_still"];

pub fn recording(name: &str) -> Recording {
    Recording {
        times: read_1d::<f64>(&format!("rec_{name}_t")).to_vec(),
        iq: read_2d::<i8>(&format!("rec_{name}_csi")),
        rssi: read_1d::<f64>(&format!("rec_{name}_rssi")).to_vec(),
    }
}

/// Streams recordings back to back (0.05 s apart, like `research/bench/occupancy_engine_replay.py`) and ticks the
/// engine once per second of stream time; `on_tick(segment, now, state)`.
pub fn replay_room(
    engine: &mut wujud_model::room::RoomEngine,
    segments: &[(&str, &Recording)],
    mut on_tick: impl FnMut(&str, f64, wujud_model::room::RoomState),
) {
    let mut offset = 0.0;
    for (label, recording) in segments {
        let mut next_tick = offset + 1.0;
        for (i, &t) in recording.times.iter().enumerate() {
            let time = t + offset;
            while time >= next_tick {
                on_tick(label, next_tick, engine.tick(next_tick).unwrap());
                next_tick += 1.0;
            }
            engine.add(
                0,
                time,
                recording.iq.row(i).as_slice().unwrap(),
                recording.rssi[i],
            );
        }
        offset += recording.times[recording.times.len() - 1] + 0.05;
    }
}

pub fn replay_engine() -> wujud_model::room::RoomEngine {
    use wujud_model::{gbdt::Gbdt, room, scene::Scene};
    let nodes = vec![[0.2, 1.7, 1.0], [3.8, 1.7, 0.8]];
    let scene = Scene::room(
        4.0,
        3.5,
        2.8,
        &nodes,
        vec![],
        vec!["router".into(), "esp".into()],
    );
    let model = Gbdt::load(&model_path(room::activity_model_file(1))).unwrap();
    room::RoomEngine::new(scene, vec![(0, 1)], "REPLAY", model, nodes)
}
