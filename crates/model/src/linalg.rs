//! Small dense linear algebra: the principal component of a few columns.

use ndarray::{Array1, Array2, ArrayView2};

/// First principal component of the columns of `x` (already centred), as in `u[:, 0] * s[0]` of an SVD.
/// Returns the component scores and the share of variance it explains (`s0^2 / sum(s^2)`).
pub fn first_principal_component(x: ArrayView2<f64>) -> (Array1<f64>, f64) {
    let gram = x.t().dot(&x);
    let (eigenvalues, eigenvectors) = symmetric_eigen(gram);
    let top = (0..eigenvalues.len())
        .max_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]))
        .unwrap_or(0);
    let scores = x.dot(&eigenvectors.column(top));
    let total: f64 = x.iter().map(|v| v * v).sum();
    (scores, eigenvalues[top].max(0.0) / (total + 1e-12))
}

/// Cyclic Jacobi eigen-decomposition of a symmetric matrix: (eigenvalues, eigenvectors as columns).
pub fn symmetric_eigen(mut a: Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut v = Array2::eye(n);
    for _sweep in 0..100 {
        let off_diagonal: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[[i, j]] * a[[i, j]])
            .sum();
        if off_diagonal < 1e-30 * (1.0 + a.iter().map(|x| x * x).sum::<f64>()) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                rotate(&mut a, &mut v, p, q);
            }
        }
    }
    ((0..n).map(|i| a[[i, i]]).collect(), v)
}

fn rotate(a: &mut Array2<f64>, v: &mut Array2<f64>, p: usize, q: usize) {
    if a[[p, q]] == 0.0 {
        return;
    }
    let theta = (a[[q, q]] - a[[p, p]]) / (2.0 * a[[p, q]]);
    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
    let c = 1.0 / (t * t + 1.0).sqrt();
    let s = t * c;
    let n = a.nrows();
    for k in 0..n {
        let akp = a[[k, p]];
        let akq = a[[k, q]];
        a[[k, p]] = c * akp - s * akq;
        a[[k, q]] = s * akp + c * akq;
    }
    for k in 0..n {
        let apk = a[[p, k]];
        let aqk = a[[q, k]];
        a[[p, k]] = c * apk - s * aqk;
        a[[q, k]] = s * apk + c * aqk;
    }
    for k in 0..n {
        let vkp = v[[k, p]];
        let vkq = v[[k, q]];
        v[[k, p]] = c * vkp - s * vkq;
        v[[k, q]] = s * vkp + c * vkq;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn eigen_of_known_matrix() {
        let (values, vectors) = symmetric_eigen(array![[2.0, 1.0], [1.0, 2.0]]);
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        assert!((sorted[0] - 1.0).abs() < 1e-12 && (sorted[1] - 3.0).abs() < 1e-12);
        let top = if values[0] > values[1] { 0 } else { 1 };
        assert!((vectors[[0, top]].abs() - 0.5f64.sqrt()).abs() < 1e-12);
    }
}
