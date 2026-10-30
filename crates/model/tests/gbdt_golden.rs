//! Tree inference against sklearn `predict_proba` / `predict` on real windows and branch-edge inputs (incl. NaN).

mod common;

use common::*;
use hallsense_model::gbdt::{Gbdt, Loss, SeatModelTrees};

/// Measured: 0 (bit-identical); the margin is for libm `exp` differences across platforms.
const PROBABILITY_TOLERANCE: f64 = 1e-12;

fn assert_matches_sklearn(model: &Gbdt, case: &str) {
    let x = read_2d::<f64>(&format!("gbdt_{case}_X"));
    let expected = read_npy::<f64>(&format!("gbdt_{case}_out"));
    let actual = model.predict_proba_rows(x.view()).unwrap();
    assert_eq!(actual.len(), expected.len(), "{case}");
    assert_close(
        case,
        actual.as_slice().unwrap(),
        expected.as_slice().unwrap(),
        PROBABILITY_TOLERANCE,
    );
}

#[test]
fn classifiers_match_sklearn() {
    for name in [
        "room_sim_all",
        "room_sim_no-rssi",
        "room_activity_esp1_random",
        "room_activity_esp3_smart",
        "room_what_moved",
    ] {
        let model = Gbdt::load(&model_path(&format!("{name}.json"))).unwrap();
        assert_eq!(model.loss, Loss::Multinomial);
        assert_matches_sklearn(&model, name);
    }
}

#[test]
fn seat_model_stages_match_sklearn() {
    let hall = SeatModelTrees::load(&model_path("hall3d.json")).unwrap();
    assert_eq!(hall.stage1.loss, Loss::Binomial);
    assert_eq!(hall.count.loss, Loss::SquaredError);
    assert_matches_sklearn(&hall.stage1, "hall3d.m1");
    assert_matches_sklearn(&hall.stage2, "hall3d.m2");
    assert_matches_sklearn(&hall.count, "hall3d.mc");
}

#[test]
fn json_round_trip_is_exact() {
    let model = Gbdt::load(&model_path("room_sim_all.json")).unwrap();
    assert_eq!(Gbdt::from_json(&model.to_json()).unwrap(), model);
}

#[test]
fn wrong_feature_count_is_an_error() {
    let model = Gbdt::load(&model_path("room_sim_all.json")).unwrap();
    assert!(model.predict_proba(&[0.0; 3]).is_err());
}
