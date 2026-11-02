use wujud_sim::random::seeded;
use wujud_sim::room::{
    random_movement_plan, random_plan, random_room, room_person, Activity, Movement, PlanStep, RealismKnobs,
    RoomOptions,
};

const FS: f64 = 20.0;

fn person(plan: &[PlanStep]) -> (wujud_sim::bodies::Body, wujud_sim::room::PersonLabels) {
    let room = random_room(&mut seeded(3, 0), &RoomOptions { boards: 2, mesh: true, smart_placement: false });
    let frames = (plan.iter().map(|s| s.seconds).sum::<f64>() * FS) as usize;
    room_person(&mut seeded(3, 1), &room, frames, FS, Some(plan), &RealismKnobs::default())
}

fn prayer() -> Vec<PlanStep> {
    [
        (Activity::Walk, 5.0),
        (Activity::Stand, 6.0),
        (Activity::Bow, 4.0),
        (Activity::Stand, 3.0),
        (Activity::Prostrate, 4.0),
        (Activity::Kneel, 3.0),
        (Activity::Prostrate, 4.0),
        (Activity::Stand, 5.0),
    ]
    .map(|(activity, seconds)| PlanStep { activity, seconds })
    .to_vec()
}

fn frame_at(seconds: f64) -> usize {
    (seconds * FS) as usize
}

#[test]
fn prayer_lowers_the_body_in_place() {
    let (body, labels) = person(&prayer());
    let standing = frame_at(10.0);
    let bowing = frame_at(14.5);
    let prostrate = frame_at(21.0);
    assert!(body.z_top[bowing] < 1.1 && body.z_top[standing] > 1.6);
    assert!(body.z_top[prostrate] < 0.6);
    assert!(body.head[prostrate][2] < 0.2 && body.head[standing][2] > 1.5);
    let moved = (body.centre[standing][0] - body.centre[frame_at(32.0)][0])
        .hypot(body.centre[standing][1] - body.centre[frame_at(32.0)][1]);
    assert!(moved < 0.2, "stood up {moved} m away from where the prayer started");
    assert_eq!(labels.movement_at(bowing), Movement::Bow);
    assert_eq!(labels.movement_at(prostrate), Movement::Prostrate);
    assert_eq!(labels.movement_at(frame_at(15.5)), Movement::Bow, "rising from the bow is part of it");
    assert_eq!(labels.movement_at(standing), Movement::Still);
}

#[test]
fn a_wave_moves_the_hand_above_the_head() {
    let plan = [(Activity::Walk, 5.0), (Activity::Sit, 6.0), (Activity::Wave, 4.0), (Activity::Sit, 4.0)]
        .map(|(activity, seconds)| PlanStep { activity, seconds });
    let (body, labels) = person(&plan);
    let waving: Vec<usize> = (frame_at(13.0)..frame_at(15.0)).collect();
    assert!(waving.iter().all(|&t| body.limb[t][2] > body.z_top[t]));
    assert!(waving.iter().all(|&t| (body.z_top[t] - 1.3).abs() < 0.05), "waves seated");
    assert!(waving.iter().all(|&t| labels.movement_at(t) == Movement::Wave));
    let sitting_down = (frame_at(5.0)..frame_at(11.0)).find(|&t| labels.movement_at(t) == Movement::SitDown).unwrap();
    assert!(body.z_top[sitting_down] > 1.6, "walks to the chair standing, then sits down");
    assert!((frame_at(5.0)..sitting_down).all(|t| labels.movement_at(t) == Movement::Walk));
}

#[test]
fn legacy_plans_have_no_head_and_no_new_moves() {
    let plan = random_plan(&mut seeded(5, 0), 300.0);
    let (body, labels) = person(&plan);
    assert!(body.head.is_empty() && body.head_rcs == 0.0);
    assert!((0..body.frames()).all(|t| !matches!(labels.movement_at(t), Movement::Wave | Movement::Bow | Movement::Prostrate)));
}

#[test]
fn movement_plans_hold_every_move() {
    let plan = random_movement_plan(&mut seeded(9, 0), 600.0);
    let (_, labels) = person(&plan);
    let seen: std::collections::HashSet<Movement> = (0..labels.activity.len()).map(|t| labels.movement_at(t)).collect();
    for movement in Movement::ALL {
        assert!(seen.contains(&movement), "{} never happens", movement.name());
    }
}
