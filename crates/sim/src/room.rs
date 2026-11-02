//! A bedroom/study with a router, ESP32 boards, furniture and one person following a plan.

use serde::{Deserialize, Serialize};
use std::f64::consts::{FRAC_PI_2, PI};

use crate::bodies::Body;
use crate::motion::{standing_body, wander, Gait, TargetArea};
use crate::paths::{Enclosure, RoomShape};
use crate::random::{
    chance, integer, normal, ornstein_uhlenbeck_2d, poisson, uniform, weighted_index, SimRng,
};
use crate::scatter::Scatterer;
use crate::vec3::{Vec3, ABSENT};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Activity {
    Away,
    Lie,
    Sit,
    Stand,
    Walk,
    /// Raises a hand and waves (or holds it up), keeping the posture it had.
    Wave,
    /// Ruku': torso pitched forward to horizontal, hands on the knees.
    Bow,
    /// Sujud: kneeling with forehead and hands on the floor.
    Prostrate,
    /// Sitting on the floor (between prostrations, or tashahhud).
    Kneel,
}

impl Activity {
    /// In label order: the index is the class id used by the room models.
    pub const ALL: [Activity; 5] = [
        Activity::Away,
        Activity::Lie,
        Activity::Sit,
        Activity::Stand,
        Activity::Walk,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        ["away", "lie", "sit", "stand", "walk", "wave", "bow", "prostrate", "kneel"][self.index()]
    }

    fn is_body_movement(self) -> bool {
        matches!(self, Activity::Wave | Activity::Bow | Activity::Prostrate | Activity::Kneel)
    }

    fn is_seated(self) -> bool {
        matches!(self, Activity::Sit | Activity::Kneel)
    }

    fn is_upright(self) -> bool {
        matches!(self, Activity::Stand | Activity::Walk)
    }
}

/// What the body is doing at a moment, the classes of the movement model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Movement {
    /// Away, or holding a posture (standing, sitting, kneeling, lying).
    Still,
    /// Walking, or any other whole-body change of place.
    Walk,
    Wave,
    Bow,
    Prostrate,
    SitDown,
    StandUp,
}

impl Movement {
    pub const ALL: [Movement; 7] = [
        Movement::Still,
        Movement::Walk,
        Movement::Wave,
        Movement::Bow,
        Movement::Prostrate,
        Movement::SitDown,
        Movement::StandUp,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        ["still", "walk", "wave", "bow", "prostrate", "sit_down", "stand_up"][self.index()]
    }

    /// Going into or out of a wave, bow or prostration belongs to it; a seated-upright change is
    /// sitting down or standing up; any other change moves the whole body.
    fn of_change(from: Activity, to: Activity) -> Movement {
        let movement_of = |activity| match activity {
            Activity::Wave => Some(Movement::Wave),
            Activity::Bow => Some(Movement::Bow),
            Activity::Prostrate => Some(Movement::Prostrate),
            _ => None,
        };
        if let Some(movement) = movement_of(from).or(movement_of(to)) {
            return movement;
        }
        if from.is_seated() && to.is_upright() {
            Movement::StandUp
        } else if from.is_upright() && to.is_seated() {
            Movement::SitDown
        } else {
            Movement::Walk
        }
    }

    fn of_activity(activity: Activity) -> Movement {
        match activity {
            Activity::Walk => Movement::Walk,
            Activity::Wave => Movement::Wave,
            Activity::Bow => Movement::Bow,
            Activity::Prostrate => Movement::Prostrate,
            _ => Movement::Still,
        }
    }
}

/// How much of the field a person's body box absorbs at 2.4 GHz (`η` of the diffraction model).
pub const BODY_ABSORPTION_RANGE: (f64, f64) = (0.8, 0.95);

/// Random plans mix activities with these weights (away, lie, sit, stand, walk).
const PLAN_WEIGHTS: [f64; 5] = [0.15, 0.25, 0.25, 0.15, 0.2];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanStep {
    pub activity: Activity,
    pub seconds: f64,
}

/// A furniture box for the 3-D view (centre x, y; base z; size w × d × h; yaw).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Furniture {
    pub kind: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
    pub d: f64,
    pub h: f64,
    pub yaw: f64,
}

impl Furniture {
    fn on_floor(kind: &str, x: f64, y: f64, w: f64, d: f64, h: f64) -> Self {
        Furniture {
            kind: kind.to_string(),
            x,
            y,
            z: 0.0,
            w,
            d,
            h,
            yaw: 0.0,
        }
    }
}

/// Node 0 is the router (transmit-only); links run router -> each board (plus board pairs in a mesh).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Room {
    pub enclosure: Enclosure,
    pub nodes: Vec<Vec3>,
    pub links: Vec<(usize, usize)>,
    pub clutter: Vec<Scatterer>,
    pub bed: Furniture,
    pub desk: Furniture,
    pub wardrobe: Furniture,
    pub chair: [f64; 2],
    pub smart_placement: bool,
}

impl Room {
    pub fn shape(&self) -> RoomShape {
        self.enclosure.shape
    }

    pub fn furniture(&self) -> Vec<Furniture> {
        vec![self.bed.clone(), self.desk.clone(), self.wardrobe.clone()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoomOptions {
    pub boards: usize,
    /// Also simulate board <-> board links.
    pub mesh: bool,
    /// Router and first board on opposite walls level with the bed, so the link crosses it.
    pub smart_placement: bool,
}

impl Default for RoomOptions {
    fn default() -> Self {
        RoomOptions {
            boards: 1,
            mesh: false,
            smart_placement: false,
        }
    }
}

/// Port of scenes3d.py::random_room.
pub fn random_room(rng: &mut SimRng, options: &RoomOptions) -> Room {
    let shape = RoomShape {
        width: uniform(rng, 3.0, 5.5),
        depth: uniform(rng, 3.0, 5.0),
        height: uniform(rng, 2.7, 3.2),
    };
    let (w, d) = (shape.width, shape.depth);
    let bed_width = uniform(rng, 0.9, 1.6);
    let bed_length = uniform(rng, 1.9, 2.1);
    let bed_x = if chance(rng, 0.5) {
        bed_width / 2.0 + 0.02
    } else {
        w - bed_width / 2.0 - 0.02
    };
    let bed = Furniture::on_floor(
        "bed",
        bed_x,
        d - bed_length / 2.0 - 0.02,
        bed_width,
        bed_length,
        0.5,
    );
    let bed_on_left = bed.x < w / 2.0;
    let desk_x = if bed_on_left { w - 0.6 } else { 0.6 };
    let desk = Furniture::on_floor("desk", desk_x, uniform(rng, 0.6, d * 0.45), 0.6, 1.2, 0.75);
    let chair = [desk_x + if desk_x > w / 2.0 { -0.55 } else { 0.55 }, desk.y];
    let wardrobe = Furniture::on_floor("wardrobe", uniform(rng, 1.0, w - 1.0), 0.32, 1.0, 0.6, 2.0);

    let mut nodes = if options.smart_placement {
        smart_router_and_board(rng, &shape, &bed)
    } else {
        let router = wall_node(rng, &shape, (0.8, 2.0));
        let board = wall_node(rng, &shape, (0.5, 1.5));
        vec![router, board]
    };
    for _ in 1..options.boards {
        nodes.push(wall_node(rng, &shape, (0.5, 1.5)));
    }
    let clutter = room_clutter(rng, &shape, &bed, &desk, &wardrobe);
    let wall_gains = [0; 4].map(|_| uniform(rng, -0.6, -0.25));
    Room {
        enclosure: Enclosure {
            shape,
            floor_reflects: true,
            wall_gains,
        },
        links: room_links(nodes.len(), options.mesh),
        nodes,
        clutter,
        bed,
        desk,
        wardrobe,
        chair,
        smart_placement: options.smart_placement,
    }
}

fn smart_router_and_board(rng: &mut SimRng, shape: &RoomShape, bed: &Furniture) -> Vec<Vec3> {
    let w = shape.width;
    let bed_on_left = bed.x < w / 2.0;
    let router_x = if bed_on_left { 0.08 } else { w - 0.08 };
    let board_x = if bed_on_left { w - 0.08 } else { 0.08 };
    let board_y = bed.y + normal(rng, 0.0, 0.3);
    let router_z = uniform(rng, 0.7, 1.0);
    let board_z = uniform(rng, 0.7, 1.0);
    vec![[router_x, bed.y, router_z], [board_x, board_y, board_z]]
}

fn wall_node(rng: &mut SimRng, shape: &RoomShape, height_range: (f64, f64)) -> Vec3 {
    let [x, y] = wall_point(rng, shape);
    [x, y, uniform(rng, height_range.0, height_range.1)]
}

fn wall_point(rng: &mut SimRng, shape: &RoomShape) -> [f64; 2] {
    const INSET: f64 = 0.08;
    let (w, d) = (shape.width, shape.depth);
    match integer(rng, 0, 4) {
        0 => [uniform(rng, 0.3, w - 0.3), INSET],
        1 => [w - INSET, uniform(rng, 0.3, d - 0.3)],
        2 => [uniform(rng, 0.3, w - 0.3), d - INSET],
        _ => [INSET, uniform(rng, 0.3, d - 0.3)],
    }
}

fn room_links(nodes: usize, mesh: bool) -> Vec<(usize, usize)> {
    let mut links: Vec<(usize, usize)> = (1..nodes).map(|k| (0, k)).collect();
    if mesh {
        links.extend((1..nodes).flat_map(|i| (i + 1..nodes).map(move |j| (i, j))));
    }
    links
}

/// Metal bed frame, monitor, wardrobe and 3-9 small random objects.
fn room_clutter(
    rng: &mut SimRng,
    shape: &RoomShape,
    bed: &Furniture,
    desk: &Furniture,
    wardrobe: &Furniture,
) -> Vec<Scatterer> {
    let mut clutter = vec![
        Scatterer {
            position: [bed.x, bed.y, 0.4],
            rcs: uniform(rng, 0.1, 0.5),
        },
        Scatterer {
            position: [desk.x, desk.y, 0.9],
            rcs: uniform(rng, 0.1, 0.6),
        },
        Scatterer {
            position: [wardrobe.x, wardrobe.y, 1.0],
            rcs: uniform(rng, 0.3, 1.5),
        },
    ];
    for _ in 0..integer(rng, 3, 10) {
        let position = [
            uniform(rng, 0.2, shape.width - 0.2),
            uniform(rng, 0.2, shape.depth - 0.2),
            uniform(rng, 0.0, 2.0),
        ];
        clutter.push(Scatterer {
            position,
            rcs: uniform(rng, 0.01, 0.2),
        });
    }
    clutter
}

/// Scales applied to the person: torso RCS and chest excursion (see bench/sim_calibrate.py).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealismKnobs {
    pub rcs_scale: f64,
    pub breath_scale: f64,
    pub evm_shift_db: f64,
}

impl Default for RealismKnobs {
    fn default() -> Self {
        RealismKnobs {
            rcs_scale: 1.0,
            breath_scale: 1.0,
            evm_shift_db: 0.0,
        }
    }
}

/// Frames `first..end` over which the body went from one activity's pose into the next.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PostureChange {
    pub from: Activity,
    pub to: Activity,
    pub first: usize,
    pub end: usize,
}

/// Ground truth per frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersonLabels {
    pub activity: Vec<Activity>,
    /// Torso centre, NaN while away.
    pub xy: Vec<[f64; 2]>,
    pub breath_hz: f64,
    pub plan: Vec<PlanStep>,
    pub changes: Vec<PostureChange>,
}

impl PersonLabels {
    pub fn movement_at(&self, frame: usize) -> Movement {
        let change = self.changes.iter().find(|c| (c.first..c.end).contains(&frame));
        match change {
            Some(change) => Movement::of_change(change.from, change.to),
            None => Movement::of_activity(self.activity[frame]),
        }
    }
}

pub fn random_plan(rng: &mut SimRng, seconds: f64) -> Vec<PlanStep> {
    let mut plan = Vec::new();
    let mut total = 0.0;
    while total < seconds {
        let activity = Activity::ALL[weighted_index(rng, &PLAN_WEIGHTS)];
        let step = uniform(rng, 20.0, 90.0);
        plan.push(PlanStep {
            activity,
            seconds: step,
        });
        total += step;
    }
    plan
}

fn step(activity: Activity, seconds: f64) -> PlanStep {
    PlanStep { activity, seconds }
}

/// Prayer cycles, hand waves while standing or sitting, sitting down and standing up, mixed with
/// walking, lying and leaving, each move lasting a few seconds.
pub fn random_movement_plan(rng: &mut SimRng, seconds: f64) -> Vec<PlanStep> {
    let mut plan = vec![step(Activity::Walk, uniform(rng, 3.0, 8.0))];
    while plan.iter().map(|s| s.seconds).sum::<f64>() < seconds {
        match weighted_index(rng, &[0.35, 0.2, 0.15, 0.2, 0.1]) {
            0 => prayer(rng, &mut plan),
            1 => {
                plan.push(step(Activity::Stand, uniform(rng, 3.0, 10.0)));
                for _ in 0..integer(rng, 1, 3) {
                    plan.push(step(Activity::Wave, uniform(rng, 2.0, 6.0)));
                    plan.push(step(Activity::Stand, uniform(rng, 2.0, 8.0)));
                }
            }
            2 => {
                plan.push(step(Activity::Sit, uniform(rng, 5.0, 15.0)));
                for _ in 0..integer(rng, 1, 3) {
                    plan.push(step(Activity::Wave, uniform(rng, 2.0, 6.0)));
                    plan.push(step(Activity::Sit, uniform(rng, 3.0, 10.0)));
                }
            }
            3 => {
                for _ in 0..integer(rng, 1, 4) {
                    plan.push(step(Activity::Sit, uniform(rng, 4.0, 15.0)));
                    plan.push(step(Activity::Stand, uniform(rng, 3.0, 10.0)));
                }
            }
            _ => {
                let rest = [Activity::Lie, Activity::Away][integer(rng, 0, 2)];
                plan.push(step(rest, uniform(rng, 8.0, 25.0)));
            }
        }
        plan.push(step(Activity::Walk, uniform(rng, 3.0, 10.0)));
    }
    plan
}

/// Stand, bow, stand, prostrate, sit, prostrate, stand; one to four times, sometimes ending seated.
fn prayer(rng: &mut SimRng, plan: &mut Vec<PlanStep>) {
    plan.push(step(Activity::Stand, uniform(rng, 3.0, 10.0)));
    let cycles = integer(rng, 1, 5);
    for cycle in 0..cycles {
        plan.push(step(Activity::Bow, uniform(rng, 2.5, 6.0)));
        plan.push(step(Activity::Stand, uniform(rng, 2.0, 4.5)));
        plan.push(step(Activity::Prostrate, uniform(rng, 2.5, 6.0)));
        plan.push(step(Activity::Kneel, uniform(rng, 2.0, 5.0)));
        plan.push(step(Activity::Prostrate, uniform(rng, 2.5, 6.0)));
        if cycle + 1 == cycles && chance(rng, 0.5) {
            plan.push(step(Activity::Kneel, uniform(rng, 6.0, 20.0)));
        }
        plan.push(step(Activity::Stand, uniform(rng, 3.0, 8.0)));
    }
}

struct Breathing {
    hz: f64,
    excursion: Vec<f64>,
}

fn breathing(rng: &mut SimRng, frames: usize, fs: f64, breath_scale: f64) -> Breathing {
    let hz = uniform(rng, 0.2, 0.33);
    let amplitude = uniform(rng, 0.004, 0.008) * breath_scale;
    let phase = uniform(rng, 0.0, 6.3);
    let excursion = (0..frames)
        .map(|t| amplitude * (2.0 * PI * hz * t as f64 / fs + phase).sin())
        .collect();
    Breathing { hz, excursion }
}

fn absent_person(frames: usize) -> Body {
    Body {
        centre: vec![[f64::NAN; 2]; frames],
        z_bottom: vec![0.0; frames],
        z_top: vec![0.0; frames],
        half_width: vec![0.21; frames],
        half_depth: vec![0.13; frames],
        yaw: vec![0.0; frames],
        absorption: 0.0,
        chest: vec![ABSENT; frames],
        limb: vec![ABSENT; frames],
        head: Vec::new(),
        chest_rcs: 0.0,
        limb_rcs: 0.08,
        head_rcs: 0.0,
    }
}

/// One person following `plan` (random if `None`). Port of scenes3d.py::room_person.
pub fn room_person(
    rng: &mut SimRng,
    room: &Room,
    frames: usize,
    fs: f64,
    plan: Option<&[PlanStep]>,
    knobs: &RealismKnobs,
) -> (Body, PersonLabels) {
    let plan = match plan {
        Some(plan) => plan.to_vec(),
        None => random_plan(rng, frames as f64 / fs),
    };
    let breath = breathing(rng, frames, fs, knobs.breath_scale);
    let mut body = absent_person(frames);
    let movement_session = plan.iter().any(|step| step.activity.is_body_movement());
    if movement_session {
        body.head = vec![ABSENT; frames];
    }
    let mut activity = vec![Activity::Away; frames];
    let mut last_position = [room.shape().width / 2.0, room.shape().depth / 2.0];
    let mut last_yaw = 0.0;
    let mut previous = Activity::Away;
    let mut start = 0;
    for step in &plan {
        let n = (frames - start).min((step.seconds * fs) as usize);
        if n == 0 {
            break;
        }
        let span = start..start + n;
        activity[span.clone()].fill(step.activity);
        let segment = PlanSegment {
            span,
            fs,
            breath: &breath.excursion,
        };
        let sway = ornstein_uhlenbeck_2d(rng, n, 0.003, 3.0, fs);
        let here = Place {
            spot: last_position,
            yaw: last_yaw,
        };
        let stays_in_place = movement_session
            && previous != Activity::Away
            && (previous.is_body_movement()
                || (step.activity == Activity::Stand && previous == Activity::Sit && chance(rng, 0.6)));
        match step.activity {
            Activity::Away => {}
            Activity::Lie => last_position = lie(rng, room, &mut body, &segment, &sway),
            Activity::Sit | Activity::Stand if stays_in_place => {
                let seated = step.activity == Activity::Sit;
                hold_posture(rng, &mut body, &segment, &sway, &here, seated);
                add_hand_fidgets(rng, &mut body, &segment);
            }
            Activity::Sit | Activity::Stand => {
                last_position = sit_or_stand(rng, room, &mut body, &segment, &sway, step.activity)
            }
            Activity::Walk => last_position = walk(rng, room, &mut body, &segment, last_position),
            Activity::Wave => {
                let place = if matches!(previous, Activity::Away | Activity::Walk) {
                    Place {
                        spot: last_position,
                        yaw: uniform(rng, 0.0, 2.0 * PI),
                    }
                } else {
                    here
                };
                wave(rng, &mut body, &segment, &sway, &place, previous.is_seated());
            }
            Activity::Bow => pose_in_place(&mut body, &segment, &sway, &here, &BOW),
            Activity::Prostrate => pose_in_place(&mut body, &segment, &sway, &here, &PROSTRATION),
            Activity::Kneel => pose_in_place(&mut body, &segment, &sway, &here, &KNEELING),
        }
        if body.is_present(start + n - 1) {
            last_yaw = body.yaw[start + n - 1];
        }
        previous = step.activity;
        start += n;
    }
    if movement_session {
        place_head_over_body(&mut body);
    }
    let changes = move_between_activities(&mut body, &plan, fs, frames);
    body.absorption = uniform(rng, BODY_ABSORPTION_RANGE.0, BODY_ABSORPTION_RANGE.1);
    body.chest_rcs = uniform(rng, 0.3, 0.8) * knobs.rcs_scale;
    if movement_session {
        body.head_rcs = uniform(rng, 0.02, 0.08) * knobs.rcs_scale;
        body.limb_rcs = uniform(rng, 0.04, 0.15);
    }
    let labels = PersonLabels {
        activity,
        xy: body.centre.clone(),
        breath_hz: breath.hz,
        plan,
        changes,
    };
    (body, labels)
}

/// Where a person stands and which way they face; prayer and waving stay there.
struct Place {
    spot: [f64; 2],
    yaw: f64,
}

impl Place {
    fn forward(&self) -> [f64; 2] {
        [(self.yaw + FRAC_PI_2).cos(), (self.yaw + FRAC_PI_2).sin()]
    }

    fn right(&self) -> [f64; 2] {
        [self.yaw.cos(), self.yaw.sin()]
    }

    /// The point `forward` m ahead and `right` m to the right of `anchor`, at height `z`.
    fn offset(&self, anchor: [f64; 2], forward: f64, right: f64, z: f64) -> Vec3 {
        let (f, r) = (self.forward(), self.right());
        [
            anchor[0] + f[0] * forward + r[0] * right,
            anchor[1] + f[1] * forward + r[1] * right,
            z,
        ]
    }
}

/// A held posture relative to where the person stands: the body box (its centre ahead of the feet,
/// height, half depth along the facing direction) and the chest, hand and head scatterers.
struct Pose {
    box_ahead: f64,
    z_top: f64,
    half_width: f64,
    half_depth: f64,
    chest: (f64, f64),
    hand: (f64, f64, f64),
    head: (f64, f64),
}

/// Torso horizontal at hip height, head down ahead, hands on the knees.
const BOW: Pose = Pose {
    box_ahead: 0.22,
    z_top: 1.05,
    half_width: 0.21,
    half_depth: 0.4,
    chest: (0.35, 0.98),
    hand: (0.2, 0.18, 0.55),
    head: (0.7, 0.95),
};

/// Kneeling folded forward: forehead and hands on the floor ahead of the knees.
const PROSTRATION: Pose = Pose {
    box_ahead: 0.3,
    z_top: 0.5,
    half_width: 0.24,
    half_depth: 0.5,
    chest: (0.35, 0.4),
    hand: (0.75, 0.2, 0.04),
    head: (0.8, 0.12),
};

/// Sitting on the heels, upright, hands on the thighs.
const KNEELING: Pose = Pose {
    box_ahead: -0.05,
    z_top: 0.92,
    half_width: 0.22,
    half_depth: 0.3,
    chest: (0.02, 0.65),
    hand: (0.25, 0.15, 0.42),
    head: (0.0, 0.8),
};

fn pose_in_place(body: &mut Body, segment: &PlanSegment, sway: &[[f64; 2]], place: &Place, pose: &Pose) {
    for (k, t) in segment.span.clone().enumerate() {
        let breath = segment.breath[t];
        let anchor = [place.spot[0] + sway[k][0], place.spot[1] + sway[k][1]];
        let centre = place.offset(anchor, pose.box_ahead, 0.0, 0.0);
        body.centre[t] = [centre[0], centre[1]];
        body.z_bottom[t] = 0.0;
        body.z_top[t] = pose.z_top + breath;
        body.half_width[t] = pose.half_width;
        body.half_depth[t] = pose.half_depth;
        body.yaw[t] = place.yaw;
        body.chest[t] = place.offset(anchor, pose.chest.0, 0.0, pose.chest.1 + breath);
        body.limb[t] = place.offset(anchor, pose.hand.0, pose.hand.1, pose.hand.2);
        body.head[t] = place.offset(anchor, pose.head.0, 0.0, pose.head.1);
    }
}

/// Standing, or seated on a chair, at `place` (the poses of `sit_or_stand`).
fn hold_posture(
    rng: &mut SimRng,
    body: &mut Body,
    segment: &PlanSegment,
    sway: &[[f64; 2]],
    place: &Place,
    seated: bool,
) {
    let (z_bottom, z_top, chest_z) = if seated { (0.4, 1.3, 1.0) } else { (0.0, 1.7, 1.3) };
    let forward = place.forward();
    let hand_jitter = ornstein_uhlenbeck_2d(rng, segment.span.len(), 0.01, 0.5, segment.fs);
    for (k, t) in segment.span.clone().enumerate() {
        let breath = segment.breath[t];
        let [x, y] = [place.spot[0] + sway[k][0], place.spot[1] + sway[k][1]];
        body.centre[t] = [x, y];
        body.z_bottom[t] = z_bottom;
        body.z_top[t] = z_top;
        body.half_width[t] = 0.21;
        body.half_depth[t] = 0.13 + breath;
        body.yaw[t] = place.yaw;
        body.chest[t] = [x + forward[0] * breath, y + forward[1] * breath, chest_z];
        body.limb[t] = [
            x + forward[0] * 0.3 + hand_jitter[k][0],
            y + forward[1] * 0.3 + hand_jitter[k][1],
            chest_z - 0.2,
        ];
    }
}

/// A hand raised above the head that waves side to side (or, a quarter of the time, is just held up).
fn wave(
    rng: &mut SimRng,
    body: &mut Body,
    segment: &PlanSegment,
    sway: &[[f64; 2]],
    place: &Place,
    seated: bool,
) {
    hold_posture(rng, body, segment, sway, place, seated);
    let side = if chance(rng, 0.5) { 1.0 } else { -1.0 };
    let (amplitude, hz) = if chance(rng, 0.25) {
        (uniform(rng, 0.01, 0.04), uniform(rng, 0.2, 0.5))
    } else {
        (uniform(rng, 0.08, 0.25), uniform(rng, 1.2, 3.0))
    };
    let reach = uniform(rng, 0.4, 0.6);
    let phase = uniform(rng, 0.0, 2.0 * PI);
    for (k, t) in segment.span.clone().enumerate() {
        let swing = amplitude * (2.0 * PI * hz * k as f64 / segment.fs + phase).sin();
        let anchor = body.centre[t];
        let shoulder_z = body.z_top[t] - 0.25;
        body.limb[t] = place.offset(anchor, 0.1, side * (0.3 + swing), shoulder_z + reach - swing.abs() * 0.3);
    }
}

/// A head over every present frame that no pose placed one at: above the box, or at the pillow end
/// of a lying body.
fn place_head_over_body(body: &mut Body) {
    for t in 0..body.frames() {
        if !body.is_present(t) || body.head[t][0].is_finite() {
            continue;
        }
        let [x, y] = body.centre[t];
        let lying = body.half_width[t] > 0.5;
        body.head[t] = if lying {
            let along = body.half_width[t] - 0.12;
            [x + body.yaw[t].cos() * along, y + body.yaw[t].sin() * along, body.z_top[t] - 0.07]
        } else {
            [x, y, body.z_top[t] - 0.12]
        };
    }
}

/// Seconds a body takes to go from one posture into another (sitting down, lying down, getting up).
const POSTURE_CHANGE_S: f64 = 1.8;
/// Walking speed to the next spot, and the longest a change may take.
const WALK_TO_SPOT_M_S: f64 = 0.8;
const LONGEST_CHANGE_S: f64 = 6.0;

/// Without this the body jumps from one activity's pose and place to the next in one frame. Each change
/// between two activities of someone in the room becomes movement over the start of the new one: the
/// body walks to the new spot and goes down onto the chair or bed (or gets up), eased in and out, so the
/// links see sitting down and lying down as they happen.
fn move_between_activities(body: &mut Body, plan: &[PlanStep], fs: f64, frames: usize) -> Vec<PostureChange> {
    let mut changes = Vec::new();
    let mut start = 0usize;
    let mut previous: Option<Activity> = None;
    for step in plan {
        let n = (frames - start).min((step.seconds * fs) as usize);
        if n == 0 {
            break;
        }
        if previous.is_some_and(|p| p != Activity::Away) && step.activity != Activity::Away && start > 0 {
            let from = start - 1;
            let walk_m = {
                let target = body.centre[(start + n / 2).min(start + n - 1)];
                (target[0] - body.centre[from][0]).hypot(target[1] - body.centre[from][1])
            };
            let posture = previous != Some(step.activity);
            let seconds = (walk_m / WALK_TO_SPOT_M_S + if posture { POSTURE_CHANGE_S } else { 0.0 })
                .min(LONGEST_CHANGE_S)
                .min(step.seconds / 2.0);
            let length = ((seconds * fs) as usize).min(n.saturating_sub(1));
            if length > 1 {
                ease(body, from, start + length, length);
                changes.push(PostureChange {
                    from: previous.expect("checked above"),
                    to: step.activity,
                    first: start,
                    end: start + length,
                });
            }
        }
        previous = Some(step.activity);
        start += n;
    }
    changes
}

/// Frames `to - length .. to` move from the pose at `from` to the pose at `to`.
fn ease(body: &mut Body, from: usize, to: usize, length: usize) {
    let lerp = |a: f64, b: f64, w: f64| a + (b - a) * w;
    let (c0, c1) = (body.centre[from], body.centre[to]);
    let (zb0, zb1, zt0, zt1) = (body.z_bottom[from], body.z_bottom[to], body.z_top[from], body.z_top[to]);
    let (hw0, hw1, hd0, hd1) = (body.half_width[from], body.half_width[to], body.half_depth[from], body.half_depth[to]);
    let (ch0, ch1, li0, li1) = (body.chest[from], body.chest[to], body.limb[from], body.limb[to]);
    let head_ends = (!body.head.is_empty()).then(|| [body.head[from], body.head[to]]);
    let yaw0 = body.yaw[from];
    let turn = (body.yaw[to] - yaw0 + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
    for k in 0..length {
        let t = to - length + k;
        let x = (k + 1) as f64 / length as f64;
        let w = x * x * (3.0 - 2.0 * x);
        body.centre[t] = [lerp(c0[0], c1[0], w), lerp(c0[1], c1[1], w)];
        body.z_bottom[t] = lerp(zb0, zb1, w);
        body.z_top[t] = lerp(zt0, zt1, w);
        body.half_width[t] = lerp(hw0, hw1, w);
        body.half_depth[t] = lerp(hd0, hd1, w);
        body.yaw[t] = yaw0 + turn * w;
        body.chest[t] = [lerp(ch0[0], ch1[0], w), lerp(ch0[1], ch1[1], w), lerp(ch0[2], ch1[2], w)];
        body.limb[t] = [lerp(li0[0], li1[0], w), lerp(li0[1], li1[1], w), lerp(li0[2], li1[2], w)];
        if let Some(&[h0, h1]) = head_ends.as_ref() {
            body.head[t] = [lerp(h0[0], h1[0], w), lerp(h0[1], h1[1], w), lerp(h0[2], h1[2], w)];
        }
    }
}

struct PlanSegment<'a> {
    span: std::ops::Range<usize>,
    fs: f64,
    breath: &'a [f64],
}

fn lie(
    rng: &mut SimRng,
    room: &Room,
    body: &mut Body,
    segment: &PlanSegment,
    sway: &[[f64; 2]],
) -> [f64; 2] {
    let bed = &room.bed;
    let spot = [
        bed.x + normal(rng, 0.0, bed.w * 0.15),
        bed.y + normal(rng, 0.0, 0.1),
    ];
    for (k, t) in segment.span.clone().enumerate() {
        let breath = segment.breath[t];
        let [x, y] = [spot[0] + sway[k][0], spot[1] + sway[k][1]];
        body.centre[t] = [x, y];
        body.z_bottom[t] = 0.5;
        body.z_top[t] = 0.75 + breath;
        body.half_width[t] = 0.9;
        body.half_depth[t] = 0.22;
        body.yaw[t] = FRAC_PI_2;
        body.chest[t] = [x, y, 0.7 + breath];
        body.limb[t] = [x + 0.25, y + 0.3, 0.6];
    }
    spot
}

fn sit_or_stand(
    rng: &mut SimRng,
    room: &Room,
    body: &mut Body,
    segment: &PlanSegment,
    sway: &[[f64; 2]],
    activity: Activity,
) -> [f64; 2] {
    let shape = room.shape();
    let (spot, z_bottom, z_top, chest_z) = if activity == Activity::Sit {
        let spot = if chance(rng, 0.7) {
            room.chair
        } else {
            [room.bed.x, room.bed.y - room.bed.d / 2.0 + 0.3]
        };
        (spot, 0.4, 1.3, 1.0)
    } else {
        let spot = [
            uniform(rng, 0.6, shape.width - 0.6),
            uniform(rng, 0.6, shape.depth - 0.6),
        ];
        (spot, 0.0, 1.7, 1.3)
    };
    let yaw = uniform(rng, 0.0, 2.0 * PI);
    let forward = [(yaw + FRAC_PI_2).cos(), (yaw + FRAC_PI_2).sin()];
    let n = segment.span.len();
    let hand_jitter = ornstein_uhlenbeck_2d(rng, n, 0.01, 0.5, segment.fs);
    for (k, t) in segment.span.clone().enumerate() {
        let breath = segment.breath[t];
        let [x, y] = [spot[0] + sway[k][0], spot[1] + sway[k][1]];
        body.centre[t] = [x, y];
        body.z_bottom[t] = z_bottom;
        body.z_top[t] = z_top;
        body.half_depth[t] = 0.13 + breath;
        body.yaw[t] = yaw;
        body.chest[t] = [x + forward[0] * breath, y + forward[1] * breath, chest_z];
        body.limb[t] = [
            x + forward[0] * 0.3 + hand_jitter[k][0],
            y + forward[1] * 0.3 + hand_jitter[k][1],
            chest_z - 0.2,
        ];
    }
    add_hand_fidgets(rng, body, segment);
    spot
}

/// Typing / phone: every ~20 s the hand jumps by ~10 cm for 1-3 s.
fn add_hand_fidgets(rng: &mut SimRng, body: &mut Body, segment: &PlanSegment) {
    let n = segment.span.len();
    for _ in 0..poisson(rng, n as f64 / segment.fs / 20.0) {
        let first = integer(rng, 0, n);
        let length = (n - first).min((uniform(rng, 1.0, 3.0) * segment.fs) as usize);
        let shift = [normal(rng, 0.0, 0.1), normal(rng, 0.0, 0.1)];
        let from = segment.span.start + first;
        for limb in &mut body.limb[from..from + length] {
            limb[0] += shift[0];
            limb[1] += shift[1];
        }
    }
}

fn walk(
    rng: &mut SimRng,
    room: &Room,
    body: &mut Body,
    segment: &PlanSegment,
    last_position: [f64; 2],
) -> [f64; 2] {
    let shape = room.shape();
    let area = TargetArea {
        x_range: (0.5, shape.width - 0.5),
        y_range: (0.5, shape.depth - 0.5),
    };
    let gait = Gait {
        speed_m_s: uniform(rng, 0.5, 1.2),
        arrive_within_m: 0.15,
        mean_pause_s: 0.0,
    };
    let first_target = area.sample(rng);
    let start = if last_position[0].is_finite() {
        last_position
    } else {
        first_target
    };
    let track = wander(
        rng,
        start,
        first_target,
        &area,
        &gait,
        segment.span.len(),
        segment.fs,
    );
    let walker = standing_body(rng, &track, segment.fs, 0.0);
    for (k, t) in segment.span.clone().enumerate() {
        body.centre[t] = track[k];
        body.z_bottom[t] = 0.0;
        body.z_top[t] = 1.7;
        body.yaw[t] = walker.yaw[k];
        body.chest[t] = walker.chest[k];
        body.limb[t] = walker.limb[k];
        body.half_depth[t] = walker.half_depth[k];
    }
    *track.last().unwrap_or(&start)
}
