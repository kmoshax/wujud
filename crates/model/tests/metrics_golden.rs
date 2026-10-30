mod common;

use common::read_json;
use hallsense_model::metrics::roc_auc;

#[test]
fn auc_matches_sklearn_with_ties() {
    let case = read_json("auc_case");
    let labels: Vec<bool> = case["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64() == Some(1))
        .collect();
    let scores: Vec<f64> = case["score"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let expected = case["auc"].as_f64().unwrap();
    assert!((roc_auc(&labels, &scores) - expected).abs() < 1e-12);
}

#[test]
fn auc_needs_both_classes() {
    assert!(roc_auc(&[true, true], &[0.1, 0.2]).is_nan());
    assert_eq!(roc_auc(&[false, true], &[0.1, 0.2]), 1.0);
}
