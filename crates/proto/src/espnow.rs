//! The two ESP-NOW frames hall nodes broadcast to each other (packed little-endian structs of wujud_node.ino).

use crate::line::LinkReport;
use crate::N_GROUPS;

pub const FRAME_MAGIC: u8 = b'H';
pub const BEACON_FRAME_LEN: usize = 5;
pub const REPORT_HEADER_LEN: usize = 6;
pub const REPORT_RECORD_LEN: usize = 2 + 2 * N_GROUPS + N_GROUPS;
pub const RECORDS_PER_REPORT_FRAME: usize = 5;
pub const MAX_REPORT_FRAME_LEN: usize =
    REPORT_HEADER_LEN + RECORDS_PER_REPORT_FRAME * REPORT_RECORD_LEN;

const BEACON_TYPE: u8 = 1;
const REPORT_TYPE: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    TooShort { needed: usize, found: usize },
    NotWujud,
    UnknownType(u8),
    ReportFull,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeaconFrame {
    pub node_id: u8,
    pub seq: u16,
}

impl BeaconFrame {
    pub fn encode(&self) -> [u8; BEACON_FRAME_LEN] {
        let seq = self.seq.to_le_bytes();
        [FRAME_MAGIC, BEACON_TYPE, self.node_id, seq[0], seq[1]]
    }

    fn decode(bytes: &[u8]) -> Result<Self, FrameError> {
        require_len(bytes, BEACON_FRAME_LEN)?;
        Ok(BeaconFrame {
            node_id: bytes[2],
            seq: read_u16(bytes, 3),
        })
    }
}

/// One second of one link as heard by the receiver: per subcarrier group, mean in dB*4 and std in dB*20.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReportRecord {
    pub source: u8,
    pub packets: u8,
    pub mean_q4: [i16; N_GROUPS],
    pub std_q20: [u8; N_GROUPS],
}

impl ReportRecord {
    pub fn mean_db(&self) -> [f32; N_GROUPS] {
        self.mean_q4.map(|q| f32::from(q) / 4.0)
    }

    pub fn std_db(&self) -> [f32; N_GROUPS] {
        self.std_q20.map(|q| f32::from(q) / 20.0)
    }

    pub fn as_link_report(&self, receiver: u8) -> LinkReport {
        LinkReport {
            receiver,
            source: self.source,
            packets: self.packets,
            mean_db: self.mean_db(),
            std_db: self.std_db(),
        }
    }

    fn encode_into(&self, out: &mut [u8]) {
        out[0] = self.source;
        out[1] = self.packets;
        for (group, mean) in self.mean_q4.iter().enumerate() {
            out[2 + 2 * group..4 + 2 * group].copy_from_slice(&mean.to_le_bytes());
        }
        out[2 + 2 * N_GROUPS..REPORT_RECORD_LEN].copy_from_slice(&self.std_q20);
    }

    fn decode(bytes: &[u8]) -> Self {
        let mut record = ReportRecord {
            source: bytes[0],
            packets: bytes[1],
            mean_q4: [0; N_GROUPS],
            std_q20: [0; N_GROUPS],
        };
        for group in 0..N_GROUPS {
            record.mean_q4[group] = read_u16(bytes, 2 + 2 * group) as i16;
        }
        record
            .std_q20
            .copy_from_slice(&bytes[2 + 2 * N_GROUPS..REPORT_RECORD_LEN]);
        record
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReportFrame {
    pub receiver: u8,
    pub seq: u16,
    records: [Option<ReportRecord>; RECORDS_PER_REPORT_FRAME],
}

impl ReportFrame {
    pub fn new(receiver: u8, seq: u16) -> Self {
        ReportFrame {
            receiver,
            seq,
            records: [None; RECORDS_PER_REPORT_FRAME],
        }
    }

    pub fn push(&mut self, record: ReportRecord) -> Result<(), FrameError> {
        let slot = self
            .records
            .iter_mut()
            .find(|slot| slot.is_none())
            .ok_or(FrameError::ReportFull)?;
        *slot = Some(record);
        Ok(())
    }

    pub fn is_full(&self) -> bool {
        self.records.iter().all(Option::is_some)
    }

    pub fn records(&self) -> impl Iterator<Item = &ReportRecord> {
        self.records.iter().flatten()
    }

    pub fn len(&self) -> usize {
        self.records().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of bytes written; the firmware sends only the filled records.
    pub fn encode(&self, out: &mut [u8; MAX_REPORT_FRAME_LEN]) -> usize {
        let seq = self.seq.to_le_bytes();
        out[..REPORT_HEADER_LEN].copy_from_slice(&[
            FRAME_MAGIC,
            REPORT_TYPE,
            self.receiver,
            self.len() as u8,
            seq[0],
            seq[1],
        ]);
        for (index, record) in self.records().enumerate() {
            let start = REPORT_HEADER_LEN + index * REPORT_RECORD_LEN;
            record.encode_into(&mut out[start..start + REPORT_RECORD_LEN]);
        }
        REPORT_HEADER_LEN + self.len() * REPORT_RECORD_LEN
    }

    fn decode(bytes: &[u8]) -> Result<Self, FrameError> {
        require_len(bytes, REPORT_HEADER_LEN)?;
        let mut frame = ReportFrame::new(bytes[2], read_u16(bytes, 4));
        // on_recv reads at most five records whatever nrec claims.
        let count = usize::from(bytes[3]).min(RECORDS_PER_REPORT_FRAME);
        require_len(bytes, REPORT_HEADER_LEN + count * REPORT_RECORD_LEN)?;
        for index in 0..count {
            let start = REPORT_HEADER_LEN + index * REPORT_RECORD_LEN;
            frame.push(ReportRecord::decode(
                &bytes[start..start + REPORT_RECORD_LEN],
            ))?;
        }
        Ok(frame)
    }
}

// No heap on the board, so the report stays inline instead of boxed.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frame {
    Beacon(BeaconFrame),
    Report(ReportFrame),
}

impl Frame {
    pub fn decode(bytes: &[u8]) -> Result<Self, FrameError> {
        require_len(bytes, 3)?;
        if bytes[0] != FRAME_MAGIC {
            return Err(FrameError::NotWujud);
        }
        match bytes[1] {
            BEACON_TYPE => BeaconFrame::decode(bytes).map(Frame::Beacon),
            REPORT_TYPE => ReportFrame::decode(bytes).map(Frame::Report),
            other => Err(FrameError::UnknownType(other)),
        }
    }
}

fn require_len(bytes: &[u8], needed: usize) -> Result<(), FrameError> {
    if bytes.len() < needed {
        return Err(FrameError::TooShort {
            needed,
            found: bytes.len(),
        });
    }
    Ok(())
}

fn read_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(source: u8) -> ReportRecord {
        ReportRecord {
            source,
            packets: 101,
            mean_q4: core::array::from_fn(|g| -250 - g as i16),
            std_q20: core::array::from_fn(|g| 17 + g as u8),
        }
    }

    #[test]
    fn report_frame_round_trips() {
        let mut frame = ReportFrame::new(3, 513);
        frame.push(record(1)).unwrap();
        frame.push(record(7)).unwrap();
        let mut bytes = [0; MAX_REPORT_FRAME_LEN];
        let len = frame.encode(&mut bytes);
        assert_eq!(len, REPORT_HEADER_LEN + 2 * REPORT_RECORD_LEN);
        assert_eq!(Frame::decode(&bytes[..len]), Ok(Frame::Report(frame)));
    }

    #[test]
    fn beacon_round_trips() {
        let beacon = BeaconFrame {
            node_id: 9,
            seq: 0xBEEF,
        };
        assert_eq!(Frame::decode(&beacon.encode()), Ok(Frame::Beacon(beacon)));
    }

    #[test]
    fn report_holds_five_records() {
        let mut frame = ReportFrame::new(0, 0);
        for source in 0..5 {
            frame.push(record(source)).unwrap();
        }
        assert!(frame.is_full());
        assert_eq!(frame.push(record(5)), Err(FrameError::ReportFull));
    }

    #[test]
    fn rejects_foreign_and_truncated_frames() {
        assert_eq!(
            Frame::decode(&[0x00, 1, 2, 3, 4]),
            Err(FrameError::NotWujud)
        );
        assert_eq!(
            Frame::decode(&[FRAME_MAGIC, 9, 0]),
            Err(FrameError::UnknownType(9))
        );
        assert_eq!(
            Frame::decode(&[FRAME_MAGIC, REPORT_TYPE, 0, 1, 0, 0]),
            Err(FrameError::TooShort {
                needed: REPORT_HEADER_LEN + REPORT_RECORD_LEN,
                found: REPORT_HEADER_LEN
            })
        );
    }
}
