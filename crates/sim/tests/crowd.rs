//! Several people in one room: an empty crowd changes nothing, and the people keep their distance.

use wujud_sim::band::BandPlan;
use wujud_sim::crowd::{Crowd, PERSONAL_SPACE_M};
use wujud_sim::random::seeded;
use wujud_sim::room::{random_room, Activity, PlanStep, Room, RoomOptions};
use wujud_sim::room_sim::{RoomSession, RoomSimOptions};
use wujud_sim::stream::RoomStream;

fn room() -> Room {
    random_room(&mut seeded(3, 0), &RoomOptions { boards: 3, mesh: true, smart_placement: false })
}

fn walking(seconds: f64) -> Vec<PlanStep> {
    vec![PlanStep { activity: Activity::Walk, seconds }]
}

fn options() -> RoomSimOptions {
    RoomSimOptions { seconds: 30.0, fs: 50.0, calibration_seconds: 1.0, plan: Some(walking(30.0)), ..RoomSimOptions::default() }
}

#[test]
fn empty_crowd_is_the_single_person_session() {
    let (room, options) = (room(), options());
    let mut alone = RoomSession::new(&room, &options, &mut seeded(9, 1));
    let mut crowd = RoomSession::with_crowd(&room, &options, &BandPlan::default(), &Crowd::default(), &mut seeded(9, 1));
    assert_eq!(alone.capture(0..500), crowd.capture(0..500));
    assert_eq!(alone.labels(), crowd.labels());
}

#[test]
fn walkers_keep_their_distance_and_each_has_truth() {
    let (room, options) = (room(), options());
    let crowd = Crowd { plans: vec![Some(walking(30.0)), Some(walking(30.0)), None] };
    let mut stream = RoomStream::with_crowd(&room, &options, &crowd, 4);
    let mut pairs = 0;
    let mut close = 0;
    while let Some(second) = stream.next() {
        let people = stream.people_at(second.second);
        assert_eq!(people.len(), 4);
        assert_eq!(people[0], second.truth);
        let spots: Vec<[f64; 2]> = people.iter().filter_map(|p| p.xy).collect();
        for (i, a) in spots.iter().enumerate() {
            for b in &spots[..i] {
                pairs += 1;
                close += usize::from((a[0] - b[0]).hypot(a[1] - b[1]) < PERSONAL_SPACE_M * 0.75);
            }
        }
    }
    assert!(pairs >= 80, "{pairs} pairs");
    assert!(close * 20 < pairs, "{close} of {pairs} pairs too close");
}
