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
        ["away", "lie", "sit", "stand", "walk"][self.index()]
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

/// Ground truth per frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersonLabels {
    pub activity: Vec<Activity>,
    /// Torso centre, NaN while away.
    pub xy: Vec<[f64; 2]>,
    pub breath_hz: f64,
    pub plan: Vec<PlanStep>,
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
        chest_rcs: 0.0,
        limb_rcs: 0.08,
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
    let mut activity = vec![Activity::Away; frames];
    let mut last_position = [room.shape().width / 2.0, room.shape().depth / 2.0];
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
        match step.activity {
            Activity::Away => {}
            Activity::Lie => last_position = lie(rng, room, &mut body, &segment, &sway),
            Activity::Sit | Activity::Stand => {
                last_position = sit_or_stand(rng, room, &mut body, &segment, &sway, step.activity)
            }
            Activity::Walk => last_position = walk(rng, room, &mut body, &segment, last_position),
        }
        start += n;
    }
    body.absorption = uniform(rng, BODY_ABSORPTION_RANGE.0, BODY_ABSORPTION_RANGE.1);
    body.chest_rcs = uniform(rng, 0.3, 0.8) * knobs.rcs_scale;
    let labels = PersonLabels {
        activity,
        xy: body.centre.clone(),
        breath_hz: breath.hz,
        plan,
    };
    (body, labels)
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
