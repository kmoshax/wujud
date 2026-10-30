//! Wire format between HallSense ESP32 boards and the host. The format is described in README.md.
#![no_std]

#[cfg(feature = "std")]
extern crate std;

mod csi;
mod espnow;
mod hex;
mod line;
mod line_buffer;
mod mac;
pub mod node_report;
mod rx_clock;

pub use csi::{
    bin_of_subcarrier, CsiBuffer, CsiSample, CSI_BINS, CSI_BUFFER_LEN, L_LTF_DATA_SUBCARRIERS,
};
pub use espnow::{
    BeaconFrame, Frame, FrameError, ReportFrame, ReportRecord, BEACON_FRAME_LEN, FRAME_MAGIC,
    MAX_REPORT_FRAME_LEN, RECORDS_PER_REPORT_FRAME, REPORT_HEADER_LEN, REPORT_RECORD_LEN,
};
pub use line::{
    CsiPacket, Info, Line, LineError, LineKind, LinkReport, NodeBanner, NodeSeen, RoomBanner,
};
pub use line_buffer::{LineBuffer, MAX_LINE_LEN};
pub use mac::{MacAddress, MacCase};
pub use rx_clock::RxClockUnwrapper;

#[cfg(feature = "std")]
pub use line::to_wire_line;

/// Subcarrier groups per report: 52 L-LTF data subcarriers in groups of 4 (research/hallsense/config.py).
pub const N_GROUPS: usize = 13;

/// Node ids are 0..MAX_NODES (hallsense_node.ino).
pub const MAX_NODES: usize = 32;
