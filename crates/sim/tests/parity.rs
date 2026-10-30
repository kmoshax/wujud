//! Statistical parity with the Python room simulator (research/tools/export_parity_sim.py writes
//! tests/golden/parity_room.json). Both sides simulate 100 random rooms with the same recipe but different
//! random streams, so the per-activity distributions are compared, not the numbers.

mod common;

use std::collections::BTreeMap;

use hallsense_sim::capture::Capture;
use hallsense_sim::random::{chance, uniform};
use hallsense_sim::room::{random_room, Activity, PlanStep, RoomOptions};
use hallsense_sim::room_sim::{simulate_room, RoomSimOptions};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const ROOMS: u64 = 100;
const SECONDS: f64 = 60.0;
const STATISTICS: [&str; 5] = [
    "amplitude_level",
    "temporal_std",
    "rssi_median",
    "rssi_std",
    "kept_fraction",
];
/// Two-sample Kolmogorov-Smirnov critical distance for n = m = 100 at α = 0.001 (1.95·sqrt(2/100)).
const KS_CRITICAL: f64 = 0.276;
/// Medians may differ by this fraction of the pooled inter-quartile range (measured max 0.23).
const MEDIAN_SHIFT_IN_IQR: f64 = 0.35;

type RoomStatistics = BTreeMap<&'static str, f64>;

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64]) -> f64 {
    let m = mean(values);
    (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}

/// numpy `median`.
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// numpy `quantile` with linear interpolation.
fn quantile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = q * (sorted.len() - 1) as f64;
    let below = position.floor() as usize;
    let above = (below + 1).min(sorted.len() - 1);
    sorted[below] + (position - below as f64) * (sorted[above] - sorted[below])
}

/// Same recipe as export_parity_sim.py::link_stats.
fn link_stats(capture: &Capture, fs: f64) -> RoomStatistics {
    let (packets, subcarriers, _) = capture.iq.dim();
    let amplitude: Vec<Vec<f64>> = (0..packets)
        .map(|n| {
            (0..subcarriers)
                .map(|k| f64::from(capture.iq[[n, k, 0]]).hypot(f64::from(capture.iq[[n, k, 1]])))
                .collect()
        })
        .collect();
    let all: Vec<f64> = amplitude.iter().flatten().copied().collect();
    let normalised: Vec<Vec<f64>> = amplitude
        .iter()
        .map(|row| row.iter().map(|a| a / mean(row).max(1e-6)).collect())
        .collect();
    let per_subcarrier_std: Vec<f64> = (0..subcarriers)
        .map(|k| std_dev(&normalised.iter().map(|row| row[k]).collect::<Vec<_>>()))
        .collect();
    let rssi: Vec<f64> = capture.rssi.iter().map(|&r| f64::from(r)).collect();
    BTreeMap::from([
        ("amplitude_level", mean(&all)),
        ("temporal_std", median(&per_subcarrier_std)),
        ("rssi_median", median(&rssi)),
        ("rssi_std", std_dev(&rssi)),
        ("kept_fraction", packets as f64 / (SECONDS * fs).floor()),
    ])
}

fn one_room(seed: u64) -> Vec<RoomStatistics> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let smart_placement = chance(&mut rng, 0.3);
    let room = random_room(
        &mut rng,
        &RoomOptions {
            smart_placement,
            ..RoomOptions::default()
        },
    );
    let fs = uniform(&mut rng, 50.0, 120.0);
    let harsh = uniform(&mut rng, 0.8, 1.6);
    Activity::ALL
        .iter()
        .enumerate()
        .map(|(k, &activity)| {
            let options = RoomSimOptions {
                seconds: SECONDS,
                fs,
                harsh,
                calibration_seconds: 0.0,
                plan: Some(vec![PlanStep {
                    activity,
                    seconds: SECONDS,
                }]),
                ..RoomSimOptions::default()
            };
            let mut activity_rng = ChaCha8Rng::seed_from_u64(seed * 16 + k as u64 + 1_000_000);
            let sim = simulate_room(&room, &options, &mut activity_rng);
            link_stats(&sim.links[0].session, fs)
        })
        .collect()
}

fn ks_distance(a: &[f64], b: &[f64]) -> f64 {
    let cdf = |values: &[f64], x: f64| {
        values.iter().filter(|&&v| v <= x).count() as f64 / values.len() as f64
    };
    a.iter()
        .chain(b)
        .map(|&x| (cdf(a, x) - cdf(b, x)).abs())
        .fold(0.0, f64::max)
}

#[test]
#[ignore = "slow in debug builds: cargo test -p hallsense-sim --release --test parity -- --ignored --nocapture"]
fn room_statistics_match_python_distributions() {
    let golden = common::read_json("parity_room.json");
    let rooms: Vec<_> = (0..ROOMS).into_par_iter().map(one_room).collect();
    let mut failures = Vec::new();
    for activity in Activity::ALL {
        for statistic in STATISTICS {
            let python = common::json_f64s(&golden["per_room"][activity.name()][statistic]);
            let rust: Vec<f64> = rooms
                .iter()
                .map(|r| r[activity.index()][statistic])
                .collect();
            let ks = ks_distance(&python, &rust);
            let pooled: Vec<f64> = python.iter().chain(&rust).copied().collect();
            let iqr = quantile(&pooled, 0.75) - quantile(&pooled, 0.25);
            let shift = (median(&rust) - median(&python)).abs() / iqr.max(1e-9);
            println!(
                "{:5} {:15} python median {:9.4} rust median {:9.4}  KS {:.3}  shift/IQR {:.2}",
                activity.name(),
                statistic,
                median(&python),
                median(&rust),
                ks,
                shift
            );
            if ks > KS_CRITICAL || shift > MEDIAN_SHIFT_IN_IQR {
                failures.push(format!(
                    "{} {statistic}: KS {ks:.3}, shift {shift:.2} IQR",
                    activity.name()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
