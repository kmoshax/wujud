//! `BeaconEngine` against `research/apsense.py::Engine` on a real 5.5-minute laptop beacon recording.

mod common;

use common::golden_path;
use hallsense_model::beacon::{BeaconEngine, BeaconRoom, Calibration};
use serde_json::{json, Value};

struct Beacon {
    time: f64,
    bssid: String,
    chains: Vec<i64>,
}

fn beacons() -> Vec<Beacon> {
    std::fs::read_to_string(golden_path("beacon_rows.csv"))
        .unwrap()
        .lines()
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            Beacon {
                time: fields[0].parse().unwrap(),
                bssid: fields[2].into(),
                chains: fields[4..].iter().map(|v| v.parse().unwrap()).collect(),
            }
        })
        .collect()
}

/// Python's `json.dump` writes NaN, which strict JSON (and serde) spells `null`.
fn golden_ticks() -> Value {
    let text = std::fs::read_to_string(golden_path("beacon_ticks.json")).unwrap();
    serde_json::from_str(&text.replace("NaN", "null")).unwrap()
}

/// Rounded values agree exactly; unrounded link activity to the last bits.
fn agree(actual: &Value, expected: &Value) -> bool {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            (a.as_f64().unwrap() - e.as_f64().unwrap()).abs() <= 1e-12
        }
        (Value::Array(a), Value::Array(e)) => {
            a.len() == e.len() && a.iter().zip(e).all(|(a, e)| agree(a, e))
        }
        (Value::Object(a), Value::Object(e)) => {
            a.len() == e.len()
                && a.iter()
                    .all(|(k, v)| e.get(k).is_some_and(|ev| agree(v, ev)))
        }
        (a, e) => a == e,
    }
}

/// (seconds after the first beacon, label, spot) of the calibration walk scripted in the Python export.
const CALIBRATION_PLAN: [(f64, &str, Option<[f64; 2]>); 4] = [
    (30.0, "empty", None),
    (80.0, "desk", Some([1.0, 2.5])),
    (130.0, "empty", None),
    (180.0, "desk", Some([1.0, 2.5])),
];

fn replay(mode: &str, room: &BeaconRoom, beacons: &[Beacon]) -> (Vec<Value>, Calibration) {
    let mut engine = BeaconEngine::new(room.clone(), Calibration::default());
    let start = beacons[0].time;
    let mut next_tick = start + 1.0;
    let mut ticks = Vec::new();
    for beacon in beacons {
        while beacon.time >= next_tick {
            if mode == "calibrated" {
                for (at, label, xy) in CALIBRATION_PLAN {
                    if (next_tick - (start + at)).abs() < 0.5 {
                        engine.start_calibration(next_tick, label, xy, 15.0, 2.0);
                    }
                }
            }
            let state = engine.tick(next_tick);
            let last = engine.diagnostics();
            ticks.push(json!({
                "now": next_tick, "room": state.room, "link_activity": state.link_activity, "alive": state.alive,
                "t": state.t,
                "last": {"m": last.m, "mot": last.mot, "pending": last.pending, "cv": last.cv, "spots": last.spots},
            }));
            next_tick += 1.0;
        }
        engine.add(beacon.time, &beacon.bssid, &beacon.chains);
    }
    (ticks, engine.calibration().clone())
}

#[test]
fn beacon_engine_matches_python_every_second() {
    let golden = golden_ticks();
    let room: BeaconRoom = serde_json::from_value(golden["config"].clone()).unwrap();
    let beacons = beacons();
    for mode in ["auto", "calibrated"] {
        let (ticks, calibration) = replay(mode, &room, &beacons);
        let expected = golden[mode]["ticks"].as_array().unwrap();
        assert_eq!(ticks.len(), expected.len(), "{mode}");
        for (actual, expected) in ticks.iter().zip(expected) {
            assert!(agree(actual, expected), "{mode}:\n{actual}\n{expected}");
        }
        let calibration = serde_json::to_value(&calibration).unwrap();
        assert!(
            agree(&calibration, &golden[mode]["calibration"]),
            "{mode} calibration"
        );
    }
}
