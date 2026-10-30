//! Shape, reproducibility and sanity checks of the room and hall session pipelines.

use hallsense_sim::distractors::DistractorKind;
use hallsense_sim::hall::{random_hall, HallOptions};
use hallsense_sim::hall_sim::{ArrivalOptions, HallSession, HallSimOptions};
use hallsense_sim::room::{random_room, Activity, PlanStep, RoomOptions};
use hallsense_sim::room_sim::{simulate_room, RoomSimOptions};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn short_room_options() -> RoomSimOptions {
    RoomSimOptions {
        seconds: 6.0,
        fs: 50.0,
        calibration_seconds: 2.0,
        plan: Some(vec![
            PlanStep {
                activity: Activity::Walk,
                seconds: 3.0,
            },
            PlanStep {
                activity: Activity::Sit,
                seconds: 3.0,
            },
        ]),
        ..RoomSimOptions::default()
    }
}

#[test]
fn room_simulation_has_one_capture_per_link_with_labels_per_frame() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let room = random_room(
        &mut rng,
        &RoomOptions {
            boards: 2,
            mesh: true,
            smart_placement: true,
        },
    );
    let options = RoomSimOptions {
        distractors: vec![DistractorKind::Fan],
        wall_db: 6.0,
        ..short_room_options()
    };
    let sim = simulate_room(&room, &options, &mut rng);
    assert_eq!(sim.links.len(), 3);
    assert_eq!(sim.labels.activity.len(), 300);
    for link in &sim.links {
        let kept = link.session.len();
        assert!((200..=300).contains(&kept), "kept {kept}");
        assert_eq!(link.session.iq.dim(), (kept, 56, 2));
        assert!(link.session.frame.windows(2).all(|w| w[0] < w[1]));
        assert!(link.calibration.len() <= 100);
        assert!(
            link.session.rssi.iter().all(|&r| (-95..0).contains(&r)),
            "{:?}",
            &link.session.rssi[..5]
        );
    }
}

#[test]
fn room_simulation_is_reproducible_from_the_seed() {
    let run = || {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let room = random_room(&mut rng, &RoomOptions::default());
        simulate_room(&room, &short_room_options(), &mut rng)
    };
    assert_eq!(run(), run());
}

#[test]
fn hall_lecture_captures_both_directions_of_every_pair() {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let hall = random_hall(
        &mut rng,
        &HallOptions {
            nodes: 4,
            width: Some(10.0),
            depth: Some(9.0),
            ..HallOptions::default()
        },
    );
    let occupied: Vec<bool> = (0..hall.seats.len()).map(|s| s % 3 == 0).collect();
    let options = HallSimOptions {
        seconds: 4.0,
        calibration_seconds: 2.0,
        ..HallSimOptions::default()
    };
    let sim = HallSession::lecture(&hall, &occupied, &options, &mut rng).simulate();
    assert_eq!(sim.links.len(), 2 * 6);
    assert_eq!(
        (sim.links[0].tx, sim.links[0].rx),
        (sim.links[1].rx, sim.links[1].tx)
    );
    for link in &sim.links {
        assert_eq!(link.session.iq.dim().1, 52);
        assert!(link.session.len() > 60);
    }
}

#[test]
fn hall_arrivals_fill_seats_over_time() {
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let hall = random_hall(
        &mut rng,
        &HallOptions {
            nodes: 3,
            width: Some(10.0),
            depth: Some(9.0),
            ..HallOptions::default()
        },
    );
    let options = ArrivalOptions {
        minutes: 0.5,
        calibration_seconds: 1.0,
        ..ArrivalOptions::default()
    };
    let mut session = HallSession::arrivals(&hall, &options, &mut rng);
    let count = |frame| session.present_at(frame).iter().filter(|&&p| p).count();
    let taken = session.occupied().iter().filter(|&&o| o).count();
    assert_eq!(count(0), 0);
    assert_eq!(count(session.frames() - 1), taken);
    assert!(taken > hall.seats.len() / 2);
    let first_second = session.capture(0..20);
    assert_eq!(first_second.len(), 2 * 3);
}
