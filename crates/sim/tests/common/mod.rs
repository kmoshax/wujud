#![allow(dead_code)]

use std::fs::File;
use std::path::PathBuf;

use hallsense_sim::bodies::Body;
use hallsense_sim::paths::PropagationPath;
use hallsense_sim::vec3::Vec3;
use ndarray::{Array2, ArrayD, Axis, Ix2, IxDyn};
use ndarray_npy::NpzReader;
use num_complex::Complex64;

pub fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

pub fn read_json(name: &str) -> serde_json::Value {
    let file = File::open(golden_path(name)).expect("golden json exists");
    serde_json::from_reader(file).expect("valid json")
}

pub struct Npz(NpzReader<File>);

impl Npz {
    pub fn open(name: &str) -> Self {
        let file = File::open(golden_path(name)).expect("golden npz exists");
        Npz(NpzReader::new(file).expect("valid npz"))
    }

    pub fn array(&mut self, name: &str) -> ArrayD<f64> {
        self.0
            .by_name::<ndarray::OwnedRepr<f64>, IxDyn>(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    pub fn integers(&mut self, name: &str) -> Vec<i64> {
        self.0
            .by_name::<ndarray::OwnedRepr<i64>, IxDyn>(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .into_iter()
            .collect()
    }

    pub fn scalar(&mut self, name: &str) -> f64 {
        self.array(name).into_iter().next().expect("one value")
    }

    pub fn vector(&mut self, name: &str) -> Vec<f64> {
        self.array(name).into_iter().collect()
    }

    pub fn matrix(&mut self, name: &str) -> Array2<f64> {
        self.array(name).into_dimensionality::<Ix2>().expect("2-D")
    }

    pub fn point(&mut self, name: &str) -> Vec3 {
        let v = self.vector(name);
        [v[0], v[1], v[2]]
    }

    /// A `[..., 2]` array of (re, im) pairs as complex values with the last axis removed.
    pub fn complex(&mut self, name: &str) -> ArrayD<Complex64> {
        let pairs = self.array(name);
        let last = pairs.ndim() - 1;
        pairs.map_axis(Axis(last), |p| Complex64::new(p[0], p[1]))
    }

    pub fn complex_vector(&mut self, name: &str) -> Vec<Complex64> {
        self.complex(name).into_iter().collect()
    }

    pub fn complex_matrix(&mut self, name: &str) -> Array2<Complex64> {
        self.complex(name)
            .into_dimensionality::<Ix2>()
            .expect("2-D")
    }

    pub fn bodies(&mut self) -> Vec<Body> {
        let centre = self.array("c");
        let z0 = self.matrix("z0");
        let z1 = self.matrix("z1");
        let half_a = self.matrix("half_a");
        let half_b = self.matrix("half_b");
        let yaw = self.matrix("yaw");
        let eta = self.vector("eta");
        let chest = self.array("chest");
        let limb = self.array("limb");
        let rcs = self.vector("rcs");
        let limb_rcs = self.vector("limb_rcs");
        let frames = z0.ncols();
        let row = |m: &Array2<f64>, q: usize| m.row(q).to_vec();
        let points2 = |q: usize| {
            (0..frames)
                .map(|t| [centre[[q, t, 0]], centre[[q, t, 1]]])
                .collect()
        };
        let points3 = |a: &ArrayD<f64>, q: usize| {
            (0..frames)
                .map(|t| [a[[q, t, 0]], a[[q, t, 1]], a[[q, t, 2]]])
                .collect()
        };
        (0..eta.len())
            .map(|q| Body {
                centre: points2(q),
                z_bottom: row(&z0, q),
                z_top: row(&z1, q),
                half_width: row(&half_a, q),
                half_depth: row(&half_b, q),
                yaw: row(&yaw, q),
                absorption: eta[q],
                chest: points3(&chest, q),
                limb: points3(&limb, q),
                chest_rcs: rcs[q],
                limb_rcs: limb_rcs[q],
            })
            .collect()
    }

    pub fn paths(&mut self) -> Vec<PropagationPath> {
        let points = self.array("path_points");
        let counts = self.integers("path_point_count");
        let gains = self.complex_vector("path_gain");
        let scales = self.vector("path_eta_scale");
        let shadowable = self.vector("path_shadow");
        (0..counts.len())
            .map(|p| PropagationPath {
                points: (0..counts[p] as usize)
                    .map(|m| [points[[p, m, 0]], points[[p, m, 1]], points[[p, m, 2]]])
                    .collect(),
                gain: gains[p],
                shadow_scale: scales[p],
                shadowable: shadowable[p] != 0.0,
            })
            .collect()
    }
}

pub fn json_f64s(value: &serde_json::Value) -> Vec<f64> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_f64().expect("number"))
        .collect()
}

pub fn json_point(value: &serde_json::Value) -> Vec3 {
    let v = json_f64s(value);
    [v[0], v[1], v[2]]
}

pub fn json_complex(value: &serde_json::Value) -> Complex64 {
    let v = json_f64s(value);
    Complex64::new(v[0], v[1])
}

pub fn json_complexes(value: &serde_json::Value) -> Vec<Complex64> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(json_complex)
        .collect()
}

pub fn max_abs_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "length");
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).norm())
        .fold(0.0, f64::max)
}

pub fn max_norm(values: &[Complex64]) -> f64 {
    values.iter().map(|v| v.norm()).fold(0.0, f64::max)
}

/// Max error relative to the largest expected magnitude (the scale complex64 rounding works at).
pub fn relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    max_abs_error(actual, expected) / max_norm(expected)
}
