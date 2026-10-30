//! The rayon dataset generators and the second-by-second demo streams.

use std::sync::atomic::{AtomicUsize, Ordering};

use wujud_sim::dataset::{
    for_each_room_scene, hall_dataset, hall_scene, room_dataset, room_scene, HallDatasetConfig,
    RoomDatasetConfig,
};
use wujud_sim::hall_sim::HallSimOptions;
use wujud_sim::room::{Activity, PlanStep};
use wujud_sim::stream::{HallStream, RoomStream};

fn small_room_config() -> RoomDatasetConfig {
    RoomDatasetConfig {
        seconds: 3.0,
        calibration_seconds: 1.0,
        boards: 2,
        mesh: true,
        ..RoomDatasetConfig::default()
    }
}

#[test]
fn room_dataset_is_ordered_and_independent_of_threads() {
    let config = small_room_config();
    let parallel = room_dataset(&config, 10..14);
    assert_eq!(
        parallel.iter().map(|s| s.seed).collect::<Vec<_>>(),
        vec![10, 11, 12, 13]
    );
    let sequential = room_scene(&config, 12);
    assert_eq!(parallel[2].simulation.links, sequential.simulation.links);
    assert_eq!(
        parallel[2].simulation.labels.activity,
        sequential.simulation.labels.activity
    );
    let scene = &parallel[0];
    assert_eq!(scene.simulation.links.len(), 3);
    let frames = scene.simulation.labels.activity.len();
    assert_eq!(frames, (3.0 * scene.simulation.fs) as usize);
    assert!(scene.simulation.links[0]
        .session
        .frame
        .iter()
        .all(|&f| (f as usize) < frames));
}

#[test]
fn for_each_room_scene_visits_every_seed() {
    let visited = AtomicUsize::new(0);
    for_each_room_scene(&small_room_config(), 0..5, |_| {
        visited.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(visited.into_inner(), 5);
}

#[test]
fn hall_dataset_shares_the_hall_between_scenes() {
    let config = HallDatasetConfig {
        node_counts: vec![3],
        scenes_per_hall: 2,
        tiered: Some(true),
        simulation: HallSimOptions {
            seconds: 2.0,
            calibration_seconds: 1.0,
            ..HallSimOptions::default()
        },
    };
    let scenes = hall_dataset(&config, 5..6);
    assert_eq!(scenes.len(), 2);
    assert_eq!(scenes[0].hall, scenes[1].hall);
    assert_eq!(scenes[1], hall_scene(&config, 5, 1));
    assert_eq!(scenes[0].simulation.links.len(), 2 * 3);
    assert_eq!(
        scenes[0].simulation.occupied.len(),
        scenes[0].hall.seats.len()
    );
}

#[test]
fn room_stream_yields_every_second_with_truth() {
    let mut stream = RoomStream::demo(7, 2);
    assert_eq!(stream.seconds(), 217);
    assert_eq!(stream.calibration().len(), 2);
    let first = stream.next().expect("first second");
    assert_eq!(first.links.len(), 2);
    assert!(first.links.iter().all(|c| (35..=50).contains(&c.len())));
    assert_eq!(first.truth.activity, Activity::Away);
    assert_eq!(first.truth.xy, None);
    assert!(first.links[0].t.iter().all(|&t| (-0.01..1.01).contains(&t)));

    let room = stream.room().clone();
    let options = wujud_sim::room_sim::RoomSimOptions {
        seconds: 4.0,
        fs: 30.0,
        plan: Some(vec![PlanStep {
            activity: Activity::Lie,
            seconds: 4.0,
        }]),
        ..Default::default()
    };
    let seconds: Vec<_> = RoomStream::new(&room, &options, 3).collect();
    assert_eq!(seconds.len(), 4);
    assert!(seconds
        .iter()
        .all(|s| s.truth.activity == Activity::Lie && s.truth.breath_bpm.is_some()));
}

#[test]
fn hall_stream_starts_empty_with_both_directions() {
    let mut stream = HallStream::demo(3);
    assert_eq!(stream.seconds(), 240);
    assert_eq!(stream.directions().len(), 2 * 66);
    let first = stream.next().expect("first second");
    assert_eq!(first.links.len(), 2 * 66);
    assert!(first.present.iter().all(|&p| !p));
    assert!(first.links.iter().all(|c| c.len() >= 14));
}
