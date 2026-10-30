//! wujud 3-D Wi-Fi CSI physics simulator (port of `research/wujud/{physics3d,scenes3d,sim3d}.py`).

pub mod band;
pub mod bodies;
pub mod capture;
pub mod channel;
pub mod constants;
pub mod dataset;
pub mod distractors;
pub mod fresnel;
mod fresnel_tables;
pub mod hall;
pub mod hall_sim;
pub mod motion;
pub mod paths;
pub mod radio;
pub mod random;
pub mod room;
pub mod room_sim;
pub mod scatter;
pub mod shadow;
pub mod stream;
pub mod vec3;
