//! `limits_study`: which physical change breaks the single-link limit of Wi-Fi presence sensing?
//! See research/results/limits/README.md for the protocol and the results.

mod configs;
mod evaluate;
mod scenes;
mod windows;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context};
use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;

use configs::{all_configs, StudyConfig};
use evaluate::{
    latch_day, latch_report, session_records, train, window_report, LatchReport, ScoredWindows,
    WindowReport,
};
use scenes::{study_room, StudyRoom, CALIBRATED_KNOBS};

const TRAIN_SEEDS_FROM: u64 = 1_000_000;
const TEST_SEEDS_FROM: u64 = 2_000_000;

#[derive(Parser, Debug)]
struct Args {
    /// Comma-separated configuration names, or "all".
    #[arg(long, default_value = "all")]
    configs: String,
    #[arg(long, default_value_t = 300)]
    train_rooms: u64,
    #[arg(long, default_value_t = 200)]
    test_rooms: u64,
    /// Days (taken from the training rooms) on which the latch threshold is tuned.
    #[arg(long, default_value_t = 60)]
    tune_days: u64,
    #[arg(long, default_value_t = 300)]
    bootstrap: usize,
    #[arg(long, default_value = "research/results/limits")]
    out: PathBuf,
    /// Recompute configurations whose result file already exists.
    #[arg(long)]
    force: bool,
}

#[derive(Serialize)]
struct ConfigResult {
    config: StudyConfig,
    links: usize,
    knobs: hallsense_sim::room::RealismKnobs,
    train_rooms: u64,
    test_rooms: u64,
    tune_days: u64,
    train_windows: usize,
    feature_columns: usize,
    windows: WindowReport,
    latch: LatchReport,
    seconds: f64,
}

fn simulate_rooms(
    config: &StudyConfig,
    seeds: std::ops::Range<u64>,
    days_below: u64,
) -> Vec<StudyRoom> {
    seeds
        .into_par_iter()
        .map(|seed| study_room(config, seed, seed < days_below))
        .collect()
}

fn run(config: &StudyConfig, args: &Args) -> anyhow::Result<ConfigResult> {
    let started = Instant::now();
    let train_rooms = simulate_rooms(
        config,
        TRAIN_SEEDS_FROM..TRAIN_SEEDS_FROM + args.train_rooms,
        TRAIN_SEEDS_FROM + args.tune_days,
    );
    let train_records = session_records(&train_rooms);
    let model = train(&train_records)?;
    let tune_days = train_rooms
        .iter()
        .filter_map(|room| room.day.as_ref())
        .map(|day| latch_day(day, &model))
        .collect::<anyhow::Result<Vec<_>>>()?;
    drop(train_rooms);
    let test_rooms = simulate_rooms(
        config,
        TEST_SEEDS_FROM..TEST_SEEDS_FROM + args.test_rooms,
        u64::MAX,
    );
    let scored = ScoredWindows::new(session_records(&test_rooms), &model)?;
    let test_days = test_rooms
        .iter()
        .filter_map(|room| room.day.as_ref())
        .map(|day| latch_day(day, &model))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(ConfigResult {
        config: config.clone(),
        links: config.links(),
        knobs: CALIBRATED_KNOBS,
        train_rooms: args.train_rooms,
        test_rooms: args.test_rooms,
        tune_days: args.tune_days,
        train_windows: train_records.len(),
        feature_columns: train_records[0].features.len(),
        windows: window_report(&scored, args.bootstrap),
        latch: latch_report(&tune_days, &test_days, args.bootstrap),
        seconds: started.elapsed().as_secs_f64(),
    })
}

fn chosen_configs(names: &str) -> anyhow::Result<Vec<StudyConfig>> {
    let all = all_configs();
    if names == "all" {
        return Ok(all);
    }
    names
        .split(',')
        .map(|name| {
            all.iter()
                .find(|c| c.name == name)
                .cloned()
                .with_context(|| format!("unknown configuration {name}"))
        })
        .collect()
}

fn result_path(out: &Path, config: &StudyConfig) -> PathBuf {
    out.join(format!("config_{}.json", config.name))
}

fn summary_line(result: &ConfigResult) -> String {
    let w = &result.windows;
    format!(
        "{:20} still AUC {:.3}±{:.3}  latch AUC {:.3}±{:.3}  breathing visible {:.1}%±{:.1}  rate error {:.2} bpm  motion {:.3}/{:.3}  ({:.0} s)",
        result.config.name,
        w.still_vs_empty_auc.value,
        w.still_vs_empty_auc.se,
        result.latch.distance_auc_still_vs_empty.value,
        result.latch.distance_auc_still_vs_empty.se,
        100.0 * w.breathing_visible.value,
        100.0 * w.breathing_visible.se,
        w.rate_error_median_bpm.value,
        w.motion_recall.value,
        w.motion_false_alarm.value,
        result.seconds,
    )
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.tune_days > args.train_rooms {
        bail!("tune days come from the training rooms");
    }
    std::fs::create_dir_all(&args.out)?;
    for config in chosen_configs(&args.configs)? {
        let path = result_path(&args.out, &config);
        if path.exists() && !args.force {
            println!("{:20} already in {}", config.name, path.display());
            continue;
        }
        let result = run(&config, &args)?;
        println!("{}", summary_line(&result));
        std::fs::write(&path, serde_json::to_string_pretty(&result)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hallsense_model::features::amplitude_from_iq;
    use hallsense_model::features::{
        profile_windows, windows as model_windows, HOP_SECONDS, WINDOW_SECONDS,
    };

    #[test]
    fn single_channel_windows_equal_the_model_features() {
        let config = all_configs().remove(0);
        let room = study_room(&config, 5, false);
        let simulation = {
            use hallsense_sim::random::seeded;
            use hallsense_sim::room::{random_room, Activity, PlanStep, RoomOptions};
            use hallsense_sim::room_sim::{simulate_room_on_band, RoomSimOptions};
            let room = random_room(&mut seeded(5, 0), &RoomOptions::default());
            let options = RoomSimOptions {
                seconds: 40.0,
                fs: 50.0,
                calibration_seconds: 0.0,
                plan: Some(vec![PlanStep {
                    activity: Activity::Sit,
                    seconds: 40.0,
                }]),
                knobs: CALIBRATED_KNOBS,
                ..RoomSimOptions::default()
            };
            simulate_room_on_band(&room, &options, &config.band, &mut seeded(5, 1))
        };
        assert_eq!(room.sessions.len(), 5);
        let capture = &simulation.links[0].session;
        let ours = windows::link_windows(capture, 1);
        let (packets, subcarriers, _) = capture.iq.dim();
        let flat = capture
            .iq
            .clone()
            .into_shape_with_order((packets, 2 * subcarriers))
            .unwrap();
        let amplitude = amplitude_from_iq(flat.view()).mapv(f64::from);
        let rssi: Vec<f64> = capture.rssi.iter().map(|&r| f64::from(r)).collect();
        let reference = model_windows(
            &capture.t,
            amplitude.view(),
            Some(&rssi),
            WINDOW_SECONDS,
            HOP_SECONDS,
        );
        assert_eq!(ours.rows, reference.rows);
        assert_eq!(ours.breathing_rates, reference.breathing_rates);
        let profiles = profile_windows(&capture.t, amplitude.view(), WINDOW_SECONDS, HOP_SECONDS);
        assert_eq!(ours.profiles, profiles);
    }
}
