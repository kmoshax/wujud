//! The text lines a board prints on its USB serial port, one message per `\n`-terminated line.

use core::fmt;
use core::net::Ipv4Addr;
use core::str::FromStr;

use crate::csi::{CsiBuffer, CSI_BUFFER_LEN};
use crate::hex::{decode_hex, write_hex};
use crate::mac::{MacAddress, MacCase};
use crate::N_GROUPS;

pub const ROOM_FIRMWARE: &str = "hallsense-room";
pub const NODE_FIRMWARE: &str = "hallsense";

const CSI_HEX_LEN: usize = 2 * CSI_BUFFER_LEN;
const CSI_FIELDS: usize = 4;
const NODE_SEEN_FIELDS: usize = 3;
const REPORT_FIELDS: usize = 4 + 2 * N_GROUPS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    Info,
    Csi,
    NodeSeen,
    Report,
}

impl LineKind {
    pub const fn tag(self) -> &'static str {
        match self {
            LineKind::Info => "I",
            LineKind::Csi => "P",
            LineKind::NodeSeen => "N",
            LineKind::Report => "R",
        }
    }

    fn from_tag(tag: &str) -> Option<Self> {
        [
            LineKind::Info,
            LineKind::Csi,
            LineKind::NodeSeen,
            LineKind::Report,
        ]
        .into_iter()
        .find(|kind| kind.tag() == tag)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineError {
    Empty,
    UnknownKind,
    FieldCount {
        kind: LineKind,
        expected: usize,
        found: usize,
    },
    InvalidField {
        kind: LineKind,
        field: &'static str,
    },
    CsiHexTooShort {
        found: usize,
    },
    WrongFirmware,
}

impl fmt::Display for LineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LineError::Empty => write!(f, "empty line"),
            LineError::UnknownKind => write!(f, "line does not start with I, P, N or R"),
            LineError::FieldCount {
                kind,
                expected,
                found,
            } => write!(
                f,
                "{} line has {found} fields, expected {expected}",
                kind.tag()
            ),
            LineError::InvalidField { kind, field } => {
                write!(f, "{} line: invalid {field}", kind.tag())
            }
            LineError::CsiHexTooShort { found } => write!(
                f,
                "P line: CSI hex has {found} characters, need {CSI_HEX_LEN}"
            ),
            LineError::WrongFirmware => write!(f, "I line comes from a different firmware"),
        }
    }
}

impl core::error::Error for LineError {}

/// Any line a HallSense board prints. `Display` gives the line without its trailing `\n`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Line<'a> {
    Info(Info<'a>),
    Csi(CsiPacket),
    NodeSeen(NodeSeen),
    Report(LinkReport),
}

impl<'a> Line<'a> {
    /// Surrounding whitespace (the `\n`, a `\r`) is ignored, like Python's `line.strip()`.
    pub fn parse(text: &'a str) -> Result<Self, LineError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(LineError::Empty);
        }
        let tag = text.split(',').next().unwrap_or_default();
        match LineKind::from_tag(tag).ok_or(LineError::UnknownKind)? {
            LineKind::Info => Info::parse(text).map(Line::Info),
            LineKind::Csi => CsiPacket::parse(text).map(Line::Csi),
            LineKind::NodeSeen => NodeSeen::parse(text).map(Line::NodeSeen),
            LineKind::Report => LinkReport::parse(text).map(Line::Report),
        }
    }

    pub fn kind(&self) -> LineKind {
        match self {
            Line::Info(_) => LineKind::Info,
            Line::Csi(_) => LineKind::Csi,
            Line::NodeSeen(_) => LineKind::NodeSeen,
            Line::Report(_) => LineKind::Report,
        }
    }
}

impl fmt::Display for Line<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Line::Info(info) => info.fmt(f),
            Line::Csi(packet) => packet.fmt(f),
            Line::NodeSeen(seen) => seen.fmt(f),
            Line::Report(report) => report.fmt(f),
        }
    }
}

/// The whole wire line including its `\n`.
#[cfg(feature = "std")]
pub fn to_wire_line(message: &impl fmt::Display) -> std::string::String {
    std::format!("{message}\n")
}

/// `I,<firmware>,<key=value,...>`: a free-form banner the host only prints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Info<'a> {
    pub firmware: &'a str,
    pub fields: &'a str,
}

impl<'a> Info<'a> {
    fn parse(text: &'a str) -> Result<Self, LineError> {
        let after_tag = text.strip_prefix("I,").ok_or(LineError::FieldCount {
            kind: LineKind::Info,
            expected: 2,
            found: 1,
        })?;
        let (firmware, fields) = after_tag.split_once(',').unwrap_or((after_tag, ""));
        Ok(Info { firmware, fields })
    }

    pub fn field(&self, key: &str) -> Option<&'a str> {
        self.fields
            .split(',')
            .filter_map(|pair| pair.split_once('='))
            .find(|(name, _)| *name == key)
            .map(|(_, value)| value)
    }

    fn parsed_field<T: FromStr>(&self, key: &'static str) -> Result<T, LineError> {
        let invalid = LineError::InvalidField {
            kind: LineKind::Info,
            field: key,
        };
        self.field(key).ok_or(invalid)?.parse().map_err(|_| invalid)
    }

    fn mac_field(&self) -> Result<MacAddress, LineError> {
        self.field("mac")
            .and_then(MacAddress::parse)
            .ok_or(LineError::InvalidField {
                kind: LineKind::Info,
                field: "mac",
            })
    }
}

impl fmt::Display for Info<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I,{}", self.firmware)?;
        if !self.fields.is_empty() {
            write!(f, ",{}", self.fields)?;
        }
        Ok(())
    }
}

/// The room firmware's banner: `I,hallsense-room,ch=6,gw=192.168.1.1,mac=24:0A:C4:00:BE:EF`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoomBanner {
    pub channel: u8,
    pub gateway: Ipv4Addr,
    pub mac: MacAddress,
}

impl RoomBanner {
    pub fn from_info(info: &Info<'_>) -> Result<Self, LineError> {
        if info.firmware != ROOM_FIRMWARE {
            return Err(LineError::WrongFirmware);
        }
        Ok(RoomBanner {
            channel: info.parsed_field("ch")?,
            gateway: info.parsed_field("gw")?,
            mac: info.mac_field()?,
        })
    }
}

impl fmt::Display for RoomBanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "I,{ROOM_FIRMWARE},ch={},gw={},mac={}",
            self.channel,
            self.gateway,
            self.mac.display(MacCase::Upper)
        )
    }
}

/// The hall node's banner: `I,hallsense,node=3,ch=6,mac=24:0A:C4:00:BE:EF`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeBanner {
    pub node: u8,
    pub channel: u8,
    pub mac: MacAddress,
}

impl NodeBanner {
    pub fn from_info(info: &Info<'_>) -> Result<Self, LineError> {
        if info.firmware != NODE_FIRMWARE {
            return Err(LineError::WrongFirmware);
        }
        Ok(NodeBanner {
            node: info.parsed_field("node")?,
            channel: info.parsed_field("ch")?,
            mac: info.mac_field()?,
        })
    }
}

impl fmt::Display for NodeBanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "I,{NODE_FIRMWARE},node={},ch={},mac={}",
            self.node,
            self.channel,
            self.mac.display(MacCase::Upper)
        )
    }
}

/// `P,<rssi dBm>,<rx timestamp us>,<256 hex digits>`: one router reply heard by a room board.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsiPacket {
    pub rssi: i8,
    pub rx_timestamp_us: u32,
    pub csi: CsiBuffer,
}

impl CsiPacket {
    /// Port of research/live.py::parse_p, including its leniency: hex beyond 256 digits is ignored.
    fn parse(text: &str) -> Result<Self, LineError> {
        let [_, rssi, timestamp, hex] = split_exact::<CSI_FIELDS>(text, LineKind::Csi)?;
        if hex.len() < CSI_HEX_LEN {
            return Err(LineError::CsiHexTooShort { found: hex.len() });
        }
        let invalid_hex = LineError::InvalidField {
            kind: LineKind::Csi,
            field: "csi hex",
        };
        let csi_hex = hex.get(..CSI_HEX_LEN).ok_or(invalid_hex)?;
        let bytes = decode_hex::<CSI_BUFFER_LEN>(csi_hex).ok_or(invalid_hex)?;
        Ok(CsiPacket {
            rssi: parse_field(rssi, LineKind::Csi, "rssi")?,
            rx_timestamp_us: parse_field(timestamp, LineKind::Csi, "rx timestamp")?,
            csi: CsiBuffer::from_unsigned_bytes(bytes),
        })
    }
}

impl fmt::Display for CsiPacket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P,{},{},", self.rssi, self.rx_timestamp_us)?;
        write_hex(&self.csi.as_unsigned_bytes(), f)
    }
}

/// `N,<node id>,<mac>`: a node learned another node's MAC from its first beacon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeSeen {
    pub node: u8,
    pub mac: MacAddress,
}

impl NodeSeen {
    fn parse(text: &str) -> Result<Self, LineError> {
        let [_, node, mac] = split_exact::<NODE_SEEN_FIELDS>(text, LineKind::NodeSeen)?;
        Ok(NodeSeen {
            node: parse_field(node, LineKind::NodeSeen, "node")?,
            mac: MacAddress::parse(mac).ok_or(LineError::InvalidField {
                kind: LineKind::NodeSeen,
                field: "mac",
            })?,
        })
    }
}

impl fmt::Display for NodeSeen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "N,{},{}", self.node, self.mac.display(MacCase::Lower))
    }
}

/// `R,<rx>,<src>,<n>,<13 means>,<13 stds>`: one second of one link, in dB with two decimals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkReport {
    pub receiver: u8,
    pub source: u8,
    pub packets: u8,
    pub mean_db: [f32; N_GROUPS],
    pub std_db: [f32; N_GROUPS],
}

impl LinkReport {
    /// Port of research/live.py::Windower.add_line (which ignores `n`; it is kept here).
    fn parse(text: &str) -> Result<Self, LineError> {
        let fields = split_exact::<REPORT_FIELDS>(text, LineKind::Report)?;
        let level = |i: usize| parse_field::<f32>(fields[i], LineKind::Report, "level");
        let mut report = LinkReport {
            receiver: parse_field(fields[1], LineKind::Report, "receiver")?,
            source: parse_field(fields[2], LineKind::Report, "source")?,
            packets: parse_field(fields[3], LineKind::Report, "packets")?,
            mean_db: [0.0; N_GROUPS],
            std_db: [0.0; N_GROUPS],
        };
        for group in 0..N_GROUPS {
            report.mean_db[group] = level(4 + group)?;
            report.std_db[group] = level(4 + N_GROUPS + group)?;
        }
        Ok(report)
    }
}

impl fmt::Display for LinkReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "R,{},{},{}", self.receiver, self.source, self.packets)?;
        for value in self.mean_db.iter().chain(&self.std_db) {
            write!(f, ",{value:.2}")?;
        }
        Ok(())
    }
}

fn split_exact<const N: usize>(text: &str, kind: LineKind) -> Result<[&str; N], LineError> {
    let mut fields = [""; N];
    let mut found = 0;
    for field in text.split(',') {
        if let Some(slot) = fields.get_mut(found) {
            *slot = field;
        }
        found += 1;
    }
    if found == N {
        Ok(fields)
    } else {
        Err(LineError::FieldCount {
            kind,
            expected: N,
            found,
        })
    }
}

fn parse_field<T: FromStr>(
    text: &str,
    kind: LineKind,
    field: &'static str,
) -> Result<T, LineError> {
    text.parse()
        .map_err(|_| LineError::InvalidField { kind, field })
}
