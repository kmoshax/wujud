//! Plays a simulated lecture hall through exactly what the real hall nodes and host do:
//! each node reduces the CSI it hears to one report per neighbour per second, the host averages the reports per link.

use std::sync::mpsc::{channel, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use ndarray::{Array2, Array3, Axis};
use wujud_model::hall::{HallLayout, Windower};
use wujud_proto::node_report::{group_levels_db, LinkWindow};
use wujud_proto::N_GROUPS;
use wujud_sim::capture::Capture;
use wujud_sim::hall::Hall;
use wujud_sim::stream::HallStream;

const DATA_SUBCARRIERS: usize = 52;

pub struct HallSecond {
    pub second: usize,
    pub mean: Array2<f32>,
    pub std: Array2<f32>,
    pub present: Vec<bool>,
}

pub struct SimulatedHall {
    pub layout: HallLayout,
    pub calibration_mean: Array3<f32>,
    pub calibration_std: Array3<f32>,
    pub seconds: Receiver<HallSecond>,
}

pub fn simulate(mut stream: HallStream) -> anyhow::Result<SimulatedHall> {
    let layout = layout_of(stream.hall())?;
    let directions = stream.directions();
    let calibration = calibration_windows(&layout, &directions, &stream.calibration());
    let (sender, seconds) = channel();
    let worker_layout = layout_of(stream.hall())?;
    thread::spawn(move || {
        let started = Instant::now();
        let mut windower = Windower::new(&worker_layout);
        for second in stream.by_ref() {
            for line in node_reports(&directions, &second.links, |_| true) {
                windower.add_line(&line);
            }
            let (mean, std) = windower.tick();
            thread::sleep(
                Duration::from_secs(second.second as u64).saturating_sub(started.elapsed()),
            );
            if sender
                .send(HallSecond {
                    second: second.second,
                    mean,
                    std,
                    present: second.present,
                })
                .is_err()
            {
                return;
            }
        }
    });
    Ok(SimulatedHall {
        layout,
        calibration_mean: calibration.0,
        calibration_std: calibration.1,
        seconds,
    })
}

/// The hall as the host's engine sees it: node ids 0.., seats where the simulator put them.
fn layout_of(hall: &Hall) -> anyhow::Result<HallLayout> {
    let shape = hall.shape();
    let nodes: serde_json::Map<String, serde_json::Value> = hall
        .nodes
        .iter()
        .enumerate()
        .map(|(id, node)| (id.to_string(), serde_json::json!(node)))
        .collect();
    let config = serde_json::json!({
        "W": shape.width, "D": shape.depth, "H": shape.height, "front": hall.tiers.front,
        "nodes": nodes, "seats": hall.seats,
    });
    Ok(HallLayout::from_config_json(&config.to_string())?)
}

/// Empty-hall calibration: one averaged window per second, stacked to `[links, seconds, 13]`.
fn calibration_windows(
    layout: &HallLayout,
    directions: &[(usize, usize)],
    captures: &[Capture],
) -> (Array3<f32>, Array3<f32>) {
    let last_second = captures
        .iter()
        .flat_map(|c| c.t.last().copied())
        .fold(0.0, f64::max) as usize;
    let mut windower = Windower::new(layout);
    let mut means = Vec::new();
    let mut stds = Vec::new();
    for second in 0..=last_second {
        for line in node_reports(directions, captures, |t| t as usize == second) {
            windower.add_line(&line);
        }
        let (mean, std) = windower.tick();
        means.push(mean);
        stds.push(std);
    }
    let stack = |windows: &[Array2<f32>]| {
        let views: Vec<_> = windows.iter().map(|w| w.view()).collect();
        ndarray::stack(Axis(1), &views).expect("every window has the same links")
    };
    (stack(&means), stack(&stds))
}

/// What each receiving node would print for one second: `R,<rx>,<src>,...` for every direction with enough packets.
fn node_reports(
    directions: &[(usize, usize)],
    captures: &[Capture],
    in_second: impl Fn(f64) -> bool,
) -> Vec<String> {
    directions
        .iter()
        .zip(captures)
        .filter_map(|(&(transmitter, receiver), capture)| {
            let mut window = LinkWindow::default();
            for packet in (0..capture.t.len()).filter(|&p| in_second(capture.t[p])) {
                let rssi = f32::from(capture.rssi[packet]);
                window.add(
                    &group_levels_db(&subcarrier_powers(capture, packet), rssi),
                    rssi,
                );
            }
            let record = window.close(transmitter as u8)?;
            Some(record.as_link_report(receiver as u8).to_string())
        })
        .collect()
}

fn subcarrier_powers(capture: &Capture, packet: usize) -> [f32; DATA_SUBCARRIERS] {
    let iq = capture.iq.index_axis(Axis(0), packet);
    core::array::from_fn(|k| {
        let (a, b) = (f32::from(iq[[k, 0]]), f32::from(iq[[k, 1]]));
        (a * a + b * b).max(0.5)
    })
}

const _: () = assert!(DATA_SUBCARRIERS == 4 * N_GROUPS);
