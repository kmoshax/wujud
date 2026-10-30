//! Lecture-hall seat occupancy from an ESP32 link mesh (`research/hallsense/model.py`, `research/live.py`).

mod engine;
mod layout;
mod seat_features;
mod seat_model;
mod windower;

pub use engine::{HallEngine, HallHistoryPoint, HallReport, HallState};
pub use layout::HallLayout;
pub use seat_features::{
    context_features, distance_bins, excess_path, link_features, scene_vector, seat_descriptor,
    LinkFeatures, SceneEvidence, DISTANCE_BIN_EDGES_M, LINK_FEATURE_NAMES,
};
pub use seat_model::{SeatModel, SeatPrediction};
pub use windower::Windower;

/// Subcarrier groups each mesh node reports per source (52 L-LTF subcarriers in groups of 4).
pub const SUBCARRIER_GROUPS: usize = 13;
