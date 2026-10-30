//! Board logic that does not touch the radio, so it builds and is tested on any machine.

pub mod beacon_clock;
pub mod node_table;

pub const REPORT_INTERVAL_MS: u32 = 1000;
pub const ROOM_PING_HZ: u32 = 50;
pub const HALL_BEACON_HZ: u32 = 20;
pub const HALL_CHANNEL: u8 = 6;
