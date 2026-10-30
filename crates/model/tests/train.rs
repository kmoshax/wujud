//! Native trainer on small synthetic problems (the sklearn comparison is the ignored test in espectre_data.rs).

use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use wujud_model::gbdt::{
    train_classifier, train_regressor, EarlyStopping, Gbdt, Loss, TrainOptions,
};

fn three_blobs(n: usize, seed: u64) -> (Array2<f64>, Vec<i64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut x = Array2::zeros((n, 3));
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let class = (i % 3) as i64 * 10;
        x[[i, 0]] = class as f64 / 10.0 + rng.gen_range(-0.3..0.3);
        x[[i, 1]] = rng.gen_range(-1.0..1.0);
        x[[i, 2]] = if class == 20 {
            f64::NAN
        } else {
            rng.gen_range(0.0..1.0)
        };
        y.push(class);
    }
    (x, y)
}

fn accuracy(model: &Gbdt, x: &Array2<f64>, y: &[i64]) -> f64 {
    let hits = x
        .rows()
        .into_iter()
        .zip(y)
        .filter(|(row, &label)| model.predict_class(&row.to_vec()).unwrap() == label);
    hits.count() as f64 / y.len() as f64
}

#[test]
fn multiclass_learns_blobs_and_missing_values() {
    let (x, y) = three_blobs(600, 1);
    let options = TrainOptions {
        max_iter: 30,
        ..TrainOptions::default()
    };
    let model = train_classifier(x.view(), &y, &options).unwrap();
    assert_eq!(model.loss, Loss::Multinomial);
    assert_eq!(model.classes, vec![0, 10, 20]);
    let (test_x, test_y) = three_blobs(300, 2);
    assert!(accuracy(&model, &test_x, &test_y) > 0.97);
    assert_eq!(Gbdt::from_json(&model.to_json()).unwrap(), model);
}

#[test]
fn balanced_weights_and_early_stopping_run() {
    let (x, y) = three_blobs(900, 3);
    let options = TrainOptions {
        max_iter: 200,
        balanced_class_weight: true,
        early_stopping: EarlyStopping::On,
        n_iter_no_change: 5,
        ..TrainOptions::default()
    };
    let model = train_classifier(x.view(), &y, &options).unwrap();
    assert!(model.iterations.len() < 200);
    let total: f64 = model
        .predict_proba(&[1.0, 0.0, f64::NAN])
        .unwrap()
        .iter()
        .sum();
    assert!((total - 1.0).abs() < 1e-12);
}

#[test]
fn binomial_and_regression() {
    let (x, y) = three_blobs(400, 4);
    let binary: Vec<i64> = y.iter().map(|&c| (c == 10) as i64).collect();
    let model = train_classifier(x.view(), &binary, &TrainOptions::default()).unwrap();
    assert_eq!(model.loss, Loss::Binomial);
    assert!(accuracy(&model, &x, &binary) > 0.97);

    let target: Vec<f64> = x.column(0).iter().map(|v| 3.0 * v + 1.0).collect();
    let regressor = train_regressor(x.view(), &target, &TrainOptions::default()).unwrap();
    let error = (regressor.predict_value(&[1.0, 0.0, 0.5]).unwrap() - 4.0).abs();
    assert!(error < 0.3, "{error}");
}

#[test]
fn one_class_is_rejected() {
    let x = Array2::zeros((5, 1));
    assert!(train_classifier(x.view(), &[1; 5], &TrainOptions::default()).is_err());
}
