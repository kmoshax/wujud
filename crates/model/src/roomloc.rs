//! Physics-only position of one person: the grid point whose Fresnel-zone pattern best matches which links lit up.

use crate::numeric::median;

const GRID_STEP_M: f64 = 0.2;
const FRESNEL_SCALE_M: f64 = 0.12;

fn grid_axis(extent: f64) -> Vec<f64> {
    let count = ((extent - 0.1 - 0.2) / GRID_STEP_M).ceil().max(0.0) as usize;
    (0..count).map(|i| 0.2 + i as f64 * GRID_STEP_M).collect()
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// `link_features` are the window feature rows of each link (`[breath top, micro top]` at columns 1 and 3);
/// `links` index into `nodes` (x, y, z).
pub fn locate(
    link_features: &[Vec<f64>],
    links: &[(usize, usize)],
    nodes: &[[f64; 3]],
    width: f64,
    depth: f64,
) -> [f64; 2] {
    let raw: Vec<f64> = link_features.iter().map(|row| row[1].max(row[3])).collect();
    let center = median(&raw);
    let evidence: Vec<f64> = raw.iter().map(|e| e - center).collect();
    let mut best = ([0.0, 0.0], f64::NEG_INFINITY);
    for y in grid_axis(depth) {
        for x in grid_axis(width) {
            let score = match_score([x, y], &evidence, links, nodes);
            if score > best.1 {
                best = ([x, y], score);
            }
        }
    }
    best.0
}

fn match_score(
    point: [f64; 2],
    evidence: &[f64],
    links: &[(usize, usize)],
    nodes: &[[f64; 3]],
) -> f64 {
    let weights: Vec<f64> = links
        .iter()
        .map(|&(i, j)| {
            let (a, b) = ([nodes[i][0], nodes[i][1]], [nodes[j][0], nodes[j][1]]);
            let excess_path = distance(point, a) + distance(point, b) - distance(a, b);
            (-excess_path / FRESNEL_SCALE_M).exp()
        })
        .collect();
    let mean = weights.iter().sum::<f64>() / weights.len() as f64;
    let centered: Vec<f64> = weights.iter().map(|w| w - mean).collect();
    let norm = centered.iter().map(|w| w * w).sum::<f64>().sqrt();
    centered
        .iter()
        .zip(evidence)
        .map(|(w, e)| w * e)
        .sum::<f64>()
        / (norm + 1e-9)
}
