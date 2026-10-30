//! Body shadowing, body scatter and the full link channel against physics3d.py, for fixed body trajectories
//! (a room with a person + fan + door, a tiered hall with 40 students + lecturer).

mod common;

use common::{relative_error, Npz};
use hallsense_sim::channel::{link_channel, Link};
use hallsense_sim::scatter::{add_moving_scatterer, mean_present_position, MovingScatterer};
use hallsense_sim::shadow::{apply_segment_shadow, Segment, ShadowCasters};
use ndarray::Array2;
use num_complex::Complex64;

const SCENES: [&str; 2] = ["room_channel.npz", "hall_channel.npz"];

fn row_point(matrix: &Array2<f64>, row: usize) -> [f64; 3] {
    [matrix[[row, 0]], matrix[[row, 1]], matrix[[row, 2]]]
}

#[test]
fn segment_shadow_matches_fresnel_kirchhoff_reference() {
    for scene in SCENES {
        let mut golden = Npz::open(scene);
        let bodies = golden.bodies();
        let casters = ShadowCasters::new(&bodies);
        let starts = golden.matrix("seg_start");
        let ends = golden.matrix("seg_end");
        let offsets = golden.vector("seg_offset");
        let totals = golden.vector("seg_total");
        let scales = golden.vector("seg_eta_scale");
        let expected = golden.complex_matrix("seg_shadow");
        let mut blocked_segments = 0;
        for s in 0..offsets.len() {
            let segment = Segment {
                start: row_point(&starts, s),
                end: row_point(&ends, s),
                offset: offsets[s],
                path_length: totals[s],
            };
            let mut field = vec![Complex64::new(1.0, 0.0); bodies[0].frames()];
            apply_segment_shadow(&mut field, &segment, &casters, scales[s]);
            let want = expected.row(s).to_vec();
            let error = field
                .iter()
                .zip(&want)
                .map(|(a, e)| (a - e).norm())
                .fold(0.0, f64::max);
            assert!(error < 1e-12, "{scene} segment {s}: error {error}");
            if want.iter().any(|v| (v - 1.0).norm() > 0.05) {
                blocked_segments += 1;
            }
        }
        assert!(
            blocked_segments > 0,
            "{scene}: golden data must include a blocked segment"
        );
    }
}

fn scatter_of(golden: &mut Npz, use_limb: bool) -> Vec<Complex64> {
    let bodies = golden.bodies();
    let tx = golden.point("tx");
    let rx = golden.point("rx");
    let frequencies = golden.vector("freqs");
    let mut channel = Array2::zeros((bodies[0].frames(), frequencies.len()));
    for body in &bodies {
        let (track, rcs) = if use_limb {
            (&body.limb, body.limb_rcs)
        } else {
            (&body.chest, body.chest_rcs)
        };
        if let Some(reference) = mean_present_position(track) {
            let scatterer = MovingScatterer {
                track,
                reference,
                rcs,
            };
            add_moving_scatterer(&mut channel, tx, rx, &scatterer, &frequencies, 1.0);
        }
    }
    channel.into_iter().collect()
}

#[test]
fn moving_scatter_matches_scatter_series() {
    for scene in SCENES {
        let mut golden = Npz::open(scene);
        for (use_limb, name) in [(false, "chest_scatter"), (true, "limb_scatter")] {
            let actual = scatter_of(&mut golden, use_limb);
            let expected: Vec<Complex64> = golden.complex(name).into_iter().collect();
            // Python evaluates the scatter sum in complex64.
            let error = relative_error(&actual, &expected);
            assert!(error < 1e-6, "{scene} {name}: relative error {error}");
        }
    }
}

fn channel_of(golden: &mut Npz, absorb: f64) -> Vec<Complex64> {
    let bodies = golden.bodies();
    let paths = golden.paths();
    let rest = golden.complex_vector("rest");
    let frequencies = golden.vector("freqs");
    let link = Link {
        tx: golden.point("tx"),
        rx: golden.point("rx"),
        frequencies: &frequencies,
    };
    link_channel(link, &paths, &rest, &bodies, bodies[0].frames(), absorb)
        .into_iter()
        .collect()
}

#[test]
fn link_channel_matches_for_fixed_trajectories() {
    let mut room = Npz::open("room_channel.npz");
    let expected: Vec<Complex64> = room.complex("H").into_iter().collect();
    let error = relative_error(&channel_of(&mut room, 1.0), &expected);
    assert!(error < 1e-6, "room: relative error {error}");
    for scene in SCENES {
        let mut golden = Npz::open(scene);
        let absorb = golden.scalar("absorb");
        let expected: Vec<Complex64> = golden.complex("H_absorb").into_iter().collect();
        let error = relative_error(&channel_of(&mut golden, absorb), &expected);
        assert!(
            error < 1e-6,
            "{scene} absorb {absorb}: relative error {error}"
        );
    }
}
