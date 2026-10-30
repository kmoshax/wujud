//! Needs the ESPectre dataset (`ESPECTRE_DATA`, default `~/Projects/ext/espectre/data`); skipped when absent.

mod common;

use std::path::PathBuf;

use common::*;
use hallsense_model::espectre::EspectreDataset;

pub fn espectre_root() -> Option<PathBuf> {
    let root = std::env::var_os("ESPECTRE_DATA")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Projects/ext/espectre/data"))
        })?;
    root.join("dataset_info.json").exists().then_some(root)
}

#[test]
fn loader_matches_python_excerpt() {
    let Some(root) = espectre_root() else {
        eprintln!("ESPectre data not found, skipped");
        return;
    };
    let dataset = EspectreDataset::open(&root).unwrap();
    let info = dataset
        .find("empty", "2026-08-10T19:31", Some("vacation_home"))
        .unwrap();
    let excerpt = dataset.load("empty", info).unwrap().first_seconds(75.0);
    let expected = recording("c3_empty");
    assert_eq!(excerpt.times, expected.times);
    assert_eq!(excerpt.iq, expected.iq);
    assert_eq!(excerpt.rssi, expected.rssi);
}

fn session(
    dataset: &EspectreDataset,
    label: &str,
    when: &str,
) -> hallsense_model::espectre::Recording {
    let info = dataset.find(label, when, Some("vacation_home")).unwrap();
    dataset.load(label, info).unwrap()
}

/// Per-second `state3` and latch distance of `research/results/occupancy_engine_replay.json`
/// (`research/bench/occupancy_engine_replay.py`). Slow in debug builds: run with `--release --ignored`.
#[test]
#[ignore]
fn room_engine_replay_agrees_with_python_bench() {
    let Some(root) = espectre_root() else {
        eprintln!("ESPectre data not found, skipped");
        return;
    };
    let dataset = EspectreDataset::open(&root).unwrap();
    let bench_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/results/occupancy_engine_replay.json");
    let bench: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(bench_path).unwrap()).unwrap();
    let days = [
        (
            "day1",
            ["2026-08-10T19:31", "2026-08-10T19:25", "2026-08-10T19:22"],
        ),
        (
            "day2",
            ["2026-08-11T12:48", "2026-08-11T12:42", "2026-08-11T12:39"],
        ),
    ];
    let (mut total, mut same_state, mut same_distance) = (0, 0, 0);
    for (day, [empty, walk, still]) in days {
        let recordings = [
            session(&dataset, "empty", empty),
            session(&dataset, "motion", walk),
            session(&dataset, "static_presence", still),
        ];
        for kind in ["realistic", "no entry"] {
            let mut segments = vec![("empty", &recordings[0])];
            if kind == "realistic" {
                segments.push(("walk", &recordings[1]));
            }
            segments.push(("still", &recordings[2]));
            let mut rows = Vec::new();
            let mut engine = replay_engine();
            replay_room(&mut engine, &segments, |truth, _, state| {
                let decision = state.room.decision.as_ref();
                rows.push((
                    truth.to_string(),
                    decision.map(|d| d.state3),
                    decision.and_then(|d| d.latch.distance),
                ));
            });
            let expected = bench[format!("{day} {kind}")].as_array().unwrap();
            assert_eq!(rows.len(), expected.len(), "{day} {kind}");
            let (mut run_same, run_total) = (0, rows.len());
            for ((truth, state3, distance), row) in rows.iter().zip(expected) {
                assert_eq!(truth, row[1].as_str().unwrap());
                let state_json = serde_json::to_value(state3).unwrap();
                let state_agrees = state_json == row[2];
                let distance_agrees = match (distance, row[3].as_f64()) {
                    (Some(a), Some(e)) => (a - e).abs() <= 0.0101,
                    (a, e) => a.is_none() && e.is_none(),
                };
                run_same += state_agrees as usize;
                same_state += state_agrees as usize;
                same_distance += distance_agrees as usize;
            }
            total += run_total;
            eprintln!("{day} {kind}: state3 agrees {run_same}/{run_total}");
        }
    }
    eprintln!(
        "room engine replay: state3 {same_state}/{total} ({:.2} %), latch distance {same_distance}/{total} ({:.2} %)",
        100.0 * same_state as f64 / total as f64,
        100.0 * same_distance as f64 / total as f64
    );
    assert!(same_state * 100 >= total * 99);
}

fn amplitude_windows(recording: &hallsense_model::espectre::Recording) -> Vec<Vec<f64>> {
    use hallsense_model::features::{amplitude_from_iq, windows, HOP_SECONDS, WINDOW_SECONDS};
    let amplitude = amplitude_from_iq(recording.iq.view()).mapv(f64::from);
    windows(
        &recording.times,
        amplitude.view(),
        Some(&recording.rssi),
        WINDOW_SECONDS,
        HOP_SECONDS,
    )
    .rows
}

fn real_split(
    dataset: &EspectreDataset,
    role: &str,
) -> Vec<hallsense_model::metrics::LabelledWindows> {
    use hallsense_model::espectre::room_class;
    use rayon::prelude::*;
    let jobs: Vec<(&String, &hallsense_model::espectre::RecordingInfo)> = dataset
        .files
        .iter()
        .flat_map(|(label, files)| files.iter().map(move |info| (label, info)))
        .filter(|(_, info)| info.dataset_role.as_deref() == Some(role))
        .collect();
    jobs.par_iter()
        .map(|(label, info)| hallsense_model::metrics::LabelledWindows {
            rows: amplitude_windows(&dataset.load(label, info).unwrap()),
            label: room_class(label).unwrap(),
        })
        .collect()
}

fn research_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../research")
        .join(relative)
}

fn print_metrics(
    name: &str,
    split: &str,
    m: &hallsense_model::metrics::RoomMetrics,
    python: &serde_json::Value,
) {
    eprintln!(
        "{name:16} {split:9} motion recall {:.3} (py {:.3})  false alarm {:.3} (py {:.3})  still-vs-empty AUC window {:.3} (py {:.3})  recording {:.3} (py {:.3})  windows {} (py {})",
        m.motion_recall, python["motion_recall"].as_f64().unwrap(),
        m.motion_false_alarm, python["motion_false_alarm"].as_f64().unwrap(),
        m.still_vs_empty_auc, python["still_vs_empty_auc"].as_f64().unwrap(),
        m.still_vs_empty_auc_recording, python["still_vs_empty_auc_recording"].as_f64().unwrap(),
        m.n_windows, python["n_windows"],
    );
}

/// Native trainer vs sklearn on the `research/bench/phase_exp.py` protocol, "amplitude (current)" variant:
/// train on the simulation (`research/cache/room_sim_amp_{X,y}.npy`, written by
/// `tools/export_golden_model.py --bench`), evaluate on ESPectre selection + holdout.
/// `HALLSENSE_SKLEARN_ROOM_MODEL` may point to `cache/phase_exp_amplitude_current.joblib` exported with
/// `tools/export_models.py`, to also score sklearn's own trees on the Rust features. Run with `--release --ignored`.
#[test]
#[ignore]
fn native_trainer_matches_sklearn_on_real_data() {
    use hallsense_model::gbdt::{train_classifier, Gbdt, TrainOptions};
    use hallsense_model::metrics::room_metrics;
    let (Some(root), true) = (
        espectre_root(),
        research_path("cache/room_sim_amp_X.npy").exists(),
    ) else {
        eprintln!("ESPectre data or research/cache/room_sim_amp_X.npy missing, skipped");
        return;
    };
    let x: ndarray::Array2<f64> =
        ndarray_npy::read_npy(research_path("cache/room_sim_amp_X.npy")).unwrap();
    let y: ndarray::Array1<i64> =
        ndarray_npy::read_npy(research_path("cache/room_sim_amp_y.npy")).unwrap();
    let options = TrainOptions {
        max_iter: 300,
        learning_rate: 0.05,
        min_samples_leaf: 40,
        balanced_class_weight: true,
        ..TrainOptions::default()
    };
    let started = std::time::Instant::now();
    let native = train_classifier(x.view(), y.as_slice().unwrap(), &options).unwrap();
    eprintln!(
        "native training: {} iterations in {:.1} s",
        native.iterations.len(),
        started.elapsed().as_secs_f64()
    );

    let dataset = EspectreDataset::open(&root).unwrap();
    let python: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(research_path("results/phase_exp.json")).unwrap(),
    )
    .unwrap();
    let columns: Vec<usize> = (0..17).collect();
    let sklearn = std::env::var_os("HALLSENSE_SKLEARN_ROOM_MODEL")
        .map(|p| Gbdt::load(std::path::Path::new(&p)).unwrap());
    for split in ["selection", "holdout"] {
        let recordings = real_split(&dataset, split);
        let reference = &python["amplitude (current)"][split];
        let ours = room_metrics(&native, &recordings, &columns).unwrap();
        print_metrics("native trainer", split, &ours, reference);
        if let Some(sklearn) = &sklearn {
            let theirs = room_metrics(sklearn, &recordings, &columns).unwrap();
            print_metrics("sklearn trees", split, &theirs, reference);
        }
        assert_eq!(
            ours.n_windows as u64,
            reference["n_windows"].as_u64().unwrap()
        );
        assert!((ours.motion_recall - reference["motion_recall"].as_f64().unwrap()).abs() <= 0.05);
        assert!(
            (ours.motion_false_alarm - reference["motion_false_alarm"].as_f64().unwrap()).abs()
                <= 0.05
        );
        assert!(
            (ours.still_vs_empty_auc - reference["still_vs_empty_auc"].as_f64().unwrap()).abs()
                <= 0.1
        );
    }
}
