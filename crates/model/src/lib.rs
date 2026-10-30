//! wujud model: CSI features, gradient-boosted trees, the occupancy latch and the room and hall engines.

pub mod error;
pub mod espectre;
pub mod features;
pub mod gbdt;
pub mod hall;
pub mod linalg;
pub mod metrics;
pub mod motion_tracker;
pub mod numeric;
pub mod occupancy;
pub mod room;
pub mod roomloc;
pub mod scene;
pub mod spectrum;

pub use error::{ModelError, Result};
