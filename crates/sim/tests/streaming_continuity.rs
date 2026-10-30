//! Streaming second by second must give exactly the session simulated at once, with no seams.

use hallsense_sim::capture::Capture;
use hallsense_sim::distractors::DistractorKind;
use hallsense_sim::hall::{random_hall, HallOptions};
use hallsense_sim::hall_sim::{ArrivalOptions, HallSession};
use hallsense_sim::random::seeded;
use hallsense_sim::room::{random_room, Activity, PlanStep, RoomOptions};
use hallsense_sim::room_sim::{simulate_room, RoomSession, RoomSimOptions};
use hallsense_sim::stream::{HallStream, RoomStream};
use ndarray::{concatenate, Array2, Axis};
use num_complex::Complex64;

const SEED: u64 = 7;

fn room_options() -> RoomSimOptions {
    let plan = [
        (Activity::Walk, 4.0),
        (Activity::Sit, 4.0),
        (Activity::Lie, 3.0),
        (Activity::Walk, 3.0),
    ]
    .map(|(activity, seconds)| PlanStep { activity, seconds })
    .to_vec();
    RoomSimOptions {
        seconds: 14.0,
        fs: 50.0,
        calibration_seconds: 2.0,
        harsh: 1.4,
        plan: Some(plan),
        distractors: vec![DistractorKind::Fan, DistractorKind::Door],
        ..RoomSimOptions::default()
    }
}

fn test_room() -> hallsense_sim::room::Room {
    random_room(
        &mut seeded(SEED, 0),
        &RoomOptions {
            boards: 2,
            mesh: true,
            smart_placement: true,
        },
    )
}

fn concatenated(pieces: &[Capture]) -> Capture {
    let views: Vec<_> = pieces.iter().map(|c| c.iq.view()).collect();
    Capture {
        frame: pieces.iter().flat_map(|c| c.frame.clone()).collect(),
        t: pieces.iter().flat_map(|c| c.t.clone()).collect(),
        iq: concatenate(Axis(0), &views).expect("same subcarrier count"),
        rssi: pieces.iter().flat_map(|c| c.rssi.clone()).collect(),
    }
}

fn per_link<T: Clone>(seconds: &[Vec<T>], link: usize) -> Vec<T> {
    seconds.iter().map(|links| links[link].clone()).collect()
}

#[test]
fn room_stream_equals_batch_capture_exactly() {
    let room = test_room();
    let options = room_options();
    let batch = simulate_room(&room, &options, &mut seeded(SEED, 1));
    let mut stream = RoomStream::new(&room, &options, SEED);
    let streamed: Vec<Vec<Capture>> = stream.by_ref().map(|second| second.links).collect();
    assert_eq!(streamed.len(), 14);
    for (link, batch_link) in batch.links.iter().enumerate() {
        assert_eq!(
            concatenated(&per_link(&streamed, link)),
            batch_link.session,
            "link {link}"
        );
    }
}

#[test]
fn room_channel_slices_concatenate_to_the_whole_session() {
    let session = RoomSession::new(&test_room(), &room_options(), &mut seeded(SEED, 1));
    let whole = session.channels(0..session.frames());
    let seconds: Vec<Vec<Array2<Complex64>>> = (0..14)
        .map(|s| session.channels(s * 50..(s + 1) * 50))
        .collect();
    for (link, expected) in whole.iter().enumerate() {
        let pieces: Vec<_> = seconds.iter().map(|links| links[link].view()).collect();
        let joined = concatenate(Axis(0), &pieces).unwrap();
        let scale = expected.iter().map(|h| h.norm()).fold(0.0, f64::max);
        let error = joined
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(error <= 1e-12 * scale, "link {link}: {error}");
    }
}

/// Median per-subcarrier amplitude change between consecutive frames, for pairs that
/// straddle a one-second boundary and pairs inside a second (the median ignores the few
/// boundaries where the plan switches activity and the person jumps to a new place).
fn boundary_and_interior_change(amplitude: &[(u32, Vec<f64>)], fs: usize) -> (f64, f64) {
    let (mut boundary, mut interior) = (Vec::new(), Vec::new());
    for pair in amplitude.windows(2) {
        let ((f0, a0), (f1, a1)) = (&pair[0], &pair[1]);
        if *f1 != f0 + 1 {
            continue;
        }
        let change = a0.iter().zip(a1).map(|(x, y)| (x - y).abs()).sum::<f64>() / a0.len() as f64;
        if (*f1 as usize).is_multiple_of(fs) {
            boundary.push(change);
        } else {
            interior.push(change);
        }
    }
    (median(boundary), median(interior))
}

/// RSSI changes are mostly 0 dB, so its seam check compares means instead of medians.
fn mean_boundary_and_interior_change(values: &[(u32, Vec<f64>)], fs: usize) -> (f64, f64) {
    let (mut boundary, mut interior) = (Vec::new(), Vec::new());
    for pair in values.windows(2) {
        let ((f0, a0), (f1, a1)) = (&pair[0], &pair[1]);
        if *f1 == f0 + 1 {
            let target = if (*f1 as usize).is_multiple_of(fs) {
                &mut boundary
            } else {
                &mut interior
            };
            target.push((a1[0] - a0[0]).abs());
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    (mean(&boundary), mean(&interior))
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
fn demo_stream_has_no_seams_at_second_boundaries() {
    let mut stream = RoomStream::demo(SEED, 1);
    let mut packet_amplitude = Vec::new();
    let mut packet_rssi = Vec::new();
    for second in stream.by_ref() {
        let capture = &second.links[0];
        for (n, &frame) in capture.frame.iter().enumerate() {
            packet_rssi.push((frame, vec![f64::from(capture.rssi[n])]));
            let row = capture.iq.index_axis(Axis(0), n);
            let amplitude = row
                .outer_iter()
                .map(|iq| f64::from(iq[0]).hypot(f64::from(iq[1])))
                .collect();
            packet_amplitude.push((frame, amplitude));
        }
    }
    let (rssi_boundary, rssi_interior) = mean_boundary_and_interior_change(&packet_rssi, 50);
    eprintln!("demo rssi: boundary {rssi_boundary:.4} interior {rssi_interior:.4}");
    assert!(
        rssi_boundary <= 1.3 * rssi_interior,
        "rssi boundary {rssi_boundary:.4} vs interior {rssi_interior:.4}"
    );
    let (boundary, interior) = boundary_and_interior_change(&packet_amplitude, 50);
    eprintln!("demo: boundary {boundary:.4} interior {interior:.4}");
    assert!(
        boundary <= 1.3 * interior,
        "boundary {boundary:.4} vs interior {interior:.4}"
    );
}

#[test]
fn noiseless_channel_is_smooth_across_seconds() {
    let session = RoomSession::new(&test_room(), &room_options(), &mut seeded(SEED, 1));
    let fs = 50;
    let mut amplitude = Vec::new();
    for s in 0..14 {
        let h = &session.channels(s * fs..(s + 1) * fs)[0];
        for (t, row) in h.outer_iter().enumerate() {
            amplitude.push(((s * fs + t) as u32, row.iter().map(|v| v.norm()).collect()));
        }
    }
    let (boundary, interior) = boundary_and_interior_change(&amplitude, fs);
    eprintln!("noiseless: boundary {boundary:.3e} interior {interior:.3e}");
    assert!(
        boundary <= 1.5 * interior,
        "boundary {boundary:.3e} vs interior {interior:.3e}"
    );
}

#[test]
fn hall_stream_equals_batch_capture_exactly() {
    let hall = random_hall(
        &mut seeded(SEED, 0),
        &HallOptions {
            nodes: 3,
            width: Some(9.0),
            depth: Some(8.0),
            ..HallOptions::default()
        },
    );
    let options = ArrivalOptions {
        minutes: 0.2,
        calibration_seconds: 1.0,
        ..ArrivalOptions::default()
    };
    let batch = HallSession::arrivals(&hall, &options, &mut seeded(SEED, 1)).simulate();
    let streamed: Vec<_> = HallStream::new(&hall, &options, SEED).collect();
    assert_eq!(streamed.len(), 12);
    let links: Vec<Vec<Capture>> = streamed.iter().map(|s| s.links.clone()).collect();
    for (direction, batch_direction) in batch.links.iter().enumerate() {
        assert_eq!(
            concatenated(&per_link(&links, direction)),
            batch_direction.session,
            "direction {direction}"
        );
    }
}
