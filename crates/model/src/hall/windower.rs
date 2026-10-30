use std::collections::HashMap;

use ndarray::Array2;

use super::{HallLayout, SUBCARRIER_GROUPS};

/// Turns mesh report lines `R,rx,src,n,<13 means>,<13 stds>` into 1-s link windows `[links, 13]`.
pub struct Windower {
    node_index: HashMap<i64, usize>,
    link_index: HashMap<(usize, usize), usize>,
    mean_sum: Array2<f64>,
    std_sum: Array2<f64>,
    reports: Vec<f64>,
}

struct Report {
    link: usize,
    values: Vec<f64>,
}

impl Windower {
    pub fn new(hall: &HallLayout) -> Self {
        let links = hall.links.len();
        Windower {
            node_index: hall
                .ids
                .iter()
                .enumerate()
                .map(|(k, &id)| (id, k))
                .collect(),
            link_index: hall
                .links
                .iter()
                .enumerate()
                .map(|(n, &l)| (l, n))
                .collect(),
            mean_sum: Array2::zeros((links, SUBCARRIER_GROUPS)),
            std_sum: Array2::zeros((links, SUBCARRIER_GROUPS)),
            reports: vec![0.0; links],
        }
    }

    /// Malformed lines, unknown nodes and self-reports are ignored.
    pub fn add_line(&mut self, line: &str) {
        if let Some(report) = self.parse(line) {
            for g in 0..SUBCARRIER_GROUPS {
                self.mean_sum[[report.link, g]] += report.values[g];
                self.std_sum[[report.link, g]] += report.values[SUBCARRIER_GROUPS + g];
            }
            self.reports[report.link] += 1.0;
        }
    }

    fn parse(&self, line: &str) -> Option<Report> {
        let parts: Vec<&str> = line.trim().split(',').collect();
        if parts.len() != 4 + 2 * SUBCARRIER_GROUPS || parts[0] != "R" {
            return None;
        }
        let receiver = self.node_index.get(&parts[1].trim().parse().ok()?)?;
        let source = self.node_index.get(&parts[2].trim().parse().ok()?)?;
        if receiver == source {
            return None;
        }
        let link = self.link_index[&(*receiver.min(source), *receiver.max(source))];
        let values = parts[4..]
            .iter()
            .map(|v| v.trim().parse().ok())
            .collect::<Option<Vec<f64>>>()?;
        Some(Report { link, values })
    }

    /// Mean of the reports of the last second per link (NaN for a silent link), as float32; starts a new second.
    pub fn tick(&mut self) -> (Array2<f32>, Array2<f32>) {
        let average = |sums: &Array2<f64>, reports: &[f64]| {
            Array2::from_shape_fn(sums.dim(), |(l, g)| {
                if reports[l] == 0.0 {
                    f32::NAN
                } else {
                    (sums[[l, g]] / reports[l]) as f32
                }
            })
        };
        let windows = (
            average(&self.mean_sum, &self.reports),
            average(&self.std_sum, &self.reports),
        );
        self.mean_sum.fill(0.0);
        self.std_sum.fill(0.0);
        self.reports.fill(0.0);
        windows
    }
}
