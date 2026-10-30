//! Reproducible training data: many random rooms / halls simulated in parallel (rayon).
//! Every scene depends only on its seed, so results do not depend on the thread count.

use std::ops::Range;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::distractors::DistractorKind;
use crate::hall::{random_hall, sample_occupancy, Hall, HallOptions, OccupancyPattern};
use crate::hall_sim::{HallSession, HallSimOptions, HallSimulation};
use crate::random::{chance, pick, seeded, uniform};
use crate::room::{random_room, PlanStep, RealismKnobs, Room, RoomOptions};
use crate::room_sim::{simulate_room, RoomSimOptions, RoomSimulation};

/// How random room scenes are drawn (defaults: the recipe of research/bench/room_sim2real.py).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoomDatasetConfig {
    pub seconds: f64,
    pub calibration_seconds: f64,
    pub fs_range: (f64, f64),
    pub harsh_range: (f64, f64),
    pub smart_placement_probability: f64,
    pub boards: usize,
    pub mesh: bool,
    /// `None`: a random activity plan per scene.
    pub plan: Option<Vec<PlanStep>>,
    pub distractors: Vec<DistractorKind>,
    pub wall_db: f64,
    pub outside: bool,
    pub knobs: RealismKnobs,
}

impl Default for RoomDatasetConfig {
    fn default() -> Self {
        RoomDatasetConfig {
            seconds: 60.0,
            calibration_seconds: 20.0,
            fs_range: (50.0, 120.0),
            harsh_range: (0.8, 1.6),
            smart_placement_probability: 0.3,
            boards: 1,
            mesh: false,
            plan: None,
            distractors: Vec::new(),
            wall_db: 0.0,
            outside: false,
            knobs: RealismKnobs::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoomScene {
    pub seed: u64,
    pub room: Room,
    pub harsh: f64,
    /// Packets per link with labels per frame (`simulation.labels`, indexed by `Capture::frame`).
    pub simulation: RoomSimulation,
}

/// One random room scene, a pure function of `seed`.
pub fn room_scene(config: &RoomDatasetConfig, seed: u64) -> RoomScene {
    let mut rng = seeded(seed, 0);
    let smart_placement = chance(&mut rng, config.smart_placement_probability);
    let room = random_room(
        &mut rng,
        &RoomOptions {
            boards: config.boards,
            mesh: config.mesh,
            smart_placement,
        },
    );
    let fs = uniform(&mut rng, config.fs_range.0, config.fs_range.1);
    let harsh = uniform(&mut rng, config.harsh_range.0, config.harsh_range.1);
    let options = RoomSimOptions {
        seconds: config.seconds,
        fs,
        calibration_seconds: config.calibration_seconds,
        harsh,
        plan: config.plan.clone(),
        distractors: config.distractors.clone(),
        wall_db: config.wall_db,
        outside: config.outside,
        knobs: config.knobs,
    };
    let simulation = simulate_room(&room, &options, &mut seeded(seed, 1));
    RoomScene {
        seed,
        room,
        harsh,
        simulation,
    }
}

/// All scenes of `seeds`, in seed order.
pub fn room_dataset(config: &RoomDatasetConfig, seeds: Range<u64>) -> Vec<RoomScene> {
    seeds
        .into_par_iter()
        .map(|seed| room_scene(config, seed))
        .collect()
}

/// Hands every scene to `sink` as soon as it is ready (any order), without holding the dataset in memory.
pub fn for_each_room_scene(
    config: &RoomDatasetConfig,
    seeds: Range<u64>,
    sink: impl Fn(RoomScene) + Sync + Send,
) {
    seeds
        .into_par_iter()
        .for_each(|seed| sink(room_scene(config, seed)));
}

/// How random hall scenes are drawn (defaults: research/hallsense/dataset3d.py).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HallDatasetConfig {
    /// Each hall gets one of these node counts at random.
    pub node_counts: Vec<usize>,
    /// Occupancy scenes simulated per hall.
    pub scenes_per_hall: usize,
    pub tiered: Option<bool>,
    pub simulation: HallSimOptions,
}

impl Default for HallDatasetConfig {
    fn default() -> Self {
        HallDatasetConfig {
            node_counts: vec![6, 8, 10, 12, 16],
            scenes_per_hall: 2,
            tiered: None,
            simulation: HallSimOptions::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HallScene {
    pub hall_seed: u64,
    pub scene: usize,
    pub hall: Hall,
    pub pattern: OccupancyPattern,
    /// Packets per direction plus the seat labels (`occupied`).
    pub simulation: HallSimulation,
}

/// The hall of `hall_seed` (the same for every scene of that seed).
pub fn dataset_hall(config: &HallDatasetConfig, hall_seed: u64) -> Hall {
    let mut rng = seeded(hall_seed, 0);
    let nodes = pick(&mut rng, &config.node_counts);
    random_hall(
        &mut rng,
        &HallOptions {
            nodes,
            tiered: config.tiered,
            ..HallOptions::default()
        },
    )
}

/// Scene `scene` of hall `hall_seed`: a random occupancy and its simulation.
pub fn hall_scene(config: &HallDatasetConfig, hall_seed: u64, scene: usize) -> HallScene {
    let hall = dataset_hall(config, hall_seed);
    let mut rng = seeded(hall_seed, 1 + scene as u64);
    let (occupied, pattern) = sample_occupancy(&mut rng, &hall, None);
    let simulation =
        HallSession::lecture(&hall, &occupied, &config.simulation, &mut rng).simulate();
    HallScene {
        hall_seed,
        scene,
        hall,
        pattern,
        simulation,
    }
}

fn hall_jobs(config: &HallDatasetConfig, hall_seeds: Range<u64>) -> Vec<(u64, usize)> {
    hall_seeds
        .flat_map(|seed| (0..config.scenes_per_hall).map(move |scene| (seed, scene)))
        .collect()
}

/// All scenes of all halls, ordered by (hall seed, scene).
pub fn hall_dataset(config: &HallDatasetConfig, hall_seeds: Range<u64>) -> Vec<HallScene> {
    hall_jobs(config, hall_seeds)
        .into_par_iter()
        .map(|(seed, scene)| hall_scene(config, seed, scene))
        .collect()
}

/// Hands every hall scene to `sink` as soon as it is ready (any order).
pub fn for_each_hall_scene(
    config: &HallDatasetConfig,
    hall_seeds: Range<u64>,
    sink: impl Fn(HallScene) + Sync + Send,
) {
    hall_jobs(config, hall_seeds)
        .into_par_iter()
        .for_each(|(seed, scene)| sink(hall_scene(config, seed, scene)));
}
