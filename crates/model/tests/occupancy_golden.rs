//! `OccupancyLatch` against `research/wujud/occupancy.py` step by step.

mod common;

use common::read_json;
use serde::Deserialize;
use wujud_model::occupancy::{LatchConfig, Occupancy, OccupancyLatch};

/// Measured: last-bit differences (NumPy sums the mean pairwise).
const DISTANCE_TOLERANCE: f64 = 1e-12;

#[derive(Deserialize)]
struct Step {
    x: Vec<f64>,
    p_motion: f64,
    p_still: Option<f64>,
    clear: bool,
    state: Occupancy,
    distance: Option<f64>,
}

#[derive(Deserialize)]
struct Case {
    thr: f64,
    steps: Vec<Step>,
}

#[test]
fn latch_follows_python_step_by_step() {
    let cases: Vec<Case> = serde_json::from_value(read_json("latch_cases")).unwrap();
    assert_eq!(cases.len(), 3);
    for (case_index, case) in cases.iter().enumerate() {
        let mut latch = OccupancyLatch::new(LatchConfig {
            threshold: case.thr,
            ..LatchConfig::default()
        });
        for (i, step) in case.steps.iter().enumerate() {
            if step.clear {
                latch.clear_reference();
            }
            let (state, distance) = latch.step(&step.x, step.p_motion, step.p_still);
            assert_eq!(state, step.state, "case {case_index} step {i}");
            match (distance, step.distance) {
                (Some(a), Some(e)) => {
                    assert!(
                        (a - e).abs() <= DISTANCE_TOLERANCE * e.max(1.0),
                        "case {case_index} step {i}: {a} vs {e}"
                    )
                }
                (a, e) => assert_eq!(a.is_some(), e.is_some(), "case {case_index} step {i}"),
            }
        }
    }
}

fn quiet_profile(offset: f64) -> Vec<f64> {
    (0..8).map(|k| k as f64 * 0.1 + offset).collect()
}

#[test]
fn motion_then_unchanged_room_clears_after_three_windows() {
    let mut latch = OccupancyLatch::new(LatchConfig::default());
    for i in 0..6 {
        assert_eq!(
            latch
                .step(&quiet_profile(0.01 * (i % 2) as f64), 0.0, None)
                .0,
            Occupancy::Empty
        );
    }
    assert_eq!(
        latch.step(&quiet_profile(0.0), 0.9, None).0,
        Occupancy::Moving
    );
    assert_eq!(
        latch.step(&quiet_profile(5.0), 0.0, None).0,
        Occupancy::Still
    );
    assert_eq!(
        latch.step(&quiet_profile(0.0), 0.0, None).0,
        Occupancy::Still
    );
    assert_eq!(
        latch.step(&quiet_profile(0.0), 0.0, None).0,
        Occupancy::Still
    );
    assert_eq!(
        latch.step(&quiet_profile(0.0), 0.0, None).0,
        Occupancy::Empty
    );
}

#[test]
fn without_reference_motion_latches_still() {
    let mut latch = OccupancyLatch::new(LatchConfig::default());
    assert_eq!(
        latch.step(&quiet_profile(0.0), 0.9, None).0,
        Occupancy::Moving
    );
    assert_eq!(
        latch.step(&quiet_profile(0.0), 0.1, None),
        (Occupancy::Still, None)
    );
    assert_eq!(
        latch.step(&quiet_profile(0.0), 0.1, Some(0.2)),
        (Occupancy::Empty, None)
    );
    assert_eq!(latch.reference_len(), 1);
}

#[test]
fn drift_without_entry_needs_more_evidence() {
    let mut latch = OccupancyLatch::new(LatchConfig::default());
    for i in 0..6 {
        latch.step(&quiet_profile(0.01 * (i % 2) as f64), 0.0, None);
    }
    let (state, distance) = latch.step(&quiet_profile(0.8), 0.0, None);
    assert!(distance.unwrap() > 3.0 && distance.unwrap() < 9.0);
    assert_eq!(state, Occupancy::Empty);
}
