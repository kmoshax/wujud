//! More than one person in a room: every extra person follows their own plan, and is nudged away from
//! the people already placed so that no two bodies stand in the same spot.

use crate::bodies::Body;
use crate::room::{room_person, PersonLabels, PlanStep, RealismKnobs, Room};
use crate::random::SimRng;

/// Closest two torso centres may come (two shoulder widths and a little room).
pub const PERSONAL_SPACE_M: f64 = 0.6;
const WALL_MARGIN_M: f64 = 0.3;
const NUDGE_PASSES: usize = 4;

/// The people after the first: one entry per person, `None` for a random plan.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Crowd {
    pub plans: Vec<Option<Vec<PlanStep>>>,
}

impl Crowd {
    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }
}

/// The crowd's bodies and labels, each kept `PERSONAL_SPACE_M` from `placed` and from each other.
pub fn crowd_people(
    rng: &mut SimRng,
    room: &Room,
    crowd: &Crowd,
    placed: &[Body],
    frames: usize,
    fs: f64,
    knobs: &RealismKnobs,
) -> Vec<(Body, PersonLabels)> {
    let mut others: Vec<Body> = placed.to_vec();
    let mut people = Vec::new();
    for plan in &crowd.plans {
        let (mut body, mut labels) = room_person(rng, room, frames, fs, plan.as_deref(), knobs);
        keep_apart(&mut body, &others, room, fs);
        labels.xy = body.centre.clone();
        others.push(body.clone());
        people.push((body, labels));
    }
    people
}

/// Moves `body` (centre, chest and hands together) out of everyone's personal space, smoothly over
/// about a second so that the nudge looks like stepping aside rather than a jump.
fn keep_apart(body: &mut Body, others: &[Body], room: &Room, fs: f64) {
    let shape = room.shape();
    let window = (fs as usize).max(1);
    for _ in 0..NUDGE_PASSES {
        let push: Vec<[f64; 2]> = (0..body.frames()).map(|t| push_away(body, others, t)).collect();
        if push.iter().all(|p| p[0] == 0.0 && p[1] == 0.0) {
            return;
        }
        let smooth = moving_average(&push, window);
        for (t, nudge) in smooth.iter().enumerate() {
            if !body.is_present(t) {
                continue;
            }
            let [x, y] = body.centre[t];
            let moved = [
                (x + 1.5 * nudge[0]).clamp(WALL_MARGIN_M, shape.width - WALL_MARGIN_M),
                (y + 1.5 * nudge[1]).clamp(WALL_MARGIN_M, shape.depth - WALL_MARGIN_M),
            ];
            let shift = [moved[0] - x, moved[1] - y];
            body.centre[t] = moved;
            for point in [&mut body.chest[t], &mut body.limb[t]] {
                point[0] += shift[0];
                point[1] += shift[1];
            }
        }
    }
}

fn push_away(body: &Body, others: &[Body], t: usize) -> [f64; 2] {
    if !body.is_present(t) {
        return [0.0, 0.0];
    }
    let [x, y] = body.centre[t];
    let mut push = [0.0, 0.0];
    for other in others.iter().filter(|o| o.is_present(t)) {
        let [ox, oy] = other.centre[t];
        let (dx, dy) = (x - ox, y - oy);
        let gap = dx.hypot(dy);
        if gap < PERSONAL_SPACE_M {
            let (ux, uy) = if gap > 1e-6 { (dx / gap, dy / gap) } else { (1.0, 0.0) };
            push[0] += ux * (PERSONAL_SPACE_M - gap);
            push[1] += uy * (PERSONAL_SPACE_M - gap);
        }
    }
    push
}

fn moving_average(values: &[[f64; 2]], window: usize) -> Vec<[f64; 2]> {
    let mut prefix = vec![[0.0, 0.0]; values.len() + 1];
    for (i, v) in values.iter().enumerate() {
        prefix[i + 1] = [prefix[i][0] + v[0], prefix[i][1] + v[1]];
    }
    let half = window / 2;
    (0..values.len())
        .map(|i| {
            let (lo, hi) = (i.saturating_sub(half), (i + half + 1).min(values.len()));
            let n = (hi - lo) as f64;
            [(prefix[hi][0] - prefix[lo][0]) / n, (prefix[hi][1] - prefix[lo][1]) / n]
        })
        .collect()
}
