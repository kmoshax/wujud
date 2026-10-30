//! Real serial lines and frames against what the Python host decodes from them (research/tools/export_golden_proto.py).

use hallsense_proto::{to_wire_line, Frame, Line, ReportRecord, RxClockUnwrapper, N_GROUPS};
use serde::Deserialize;

#[derive(Deserialize)]
struct RoomGolden {
    lines: Vec<String>,
    expected: Vec<RoomPacket>,
}

#[derive(Deserialize)]
struct RoomPacket {
    rssi: i8,
    rx_timestamp_us: u32,
    csi: Vec<i8>,
    seconds: f64,
}

#[derive(Deserialize)]
struct HallGolden {
    receiver: u8,
    records: Vec<GoldenRecord>,
    lines: Vec<String>,
    expected: Vec<HallLine>,
    report_frames: Vec<GoldenFrame>,
}

#[derive(Deserialize, Clone)]
struct GoldenRecord {
    source: u8,
    packets: u8,
    mean_q4: Vec<i16>,
    std_q20: Vec<u8>,
}

#[derive(Deserialize)]
struct HallLine {
    receiver: u8,
    source: u8,
    packets: u8,
    mean_db: Vec<f32>,
    std_db: Vec<f32>,
}

#[derive(Deserialize)]
struct GoldenFrame {
    seq: u16,
    hex: String,
    records: Vec<GoldenRecord>,
}

fn load<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = format!("{}/tests/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn as_record(golden: &GoldenRecord) -> ReportRecord {
    ReportRecord {
        source: golden.source,
        packets: golden.packets,
        mean_q4: golden.mean_q4.clone().try_into().unwrap(),
        std_q20: golden.std_q20.clone().try_into().unwrap(),
    }
}

fn bytes_of_hex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn room_lines_decode_like_the_python_host() {
    let golden: RoomGolden = load("room_lines.json");
    let mut clock = RxClockUnwrapper::new();
    for (text, want) in golden.lines.iter().zip(&golden.expected) {
        let Line::Csi(packet) = Line::parse(text).unwrap() else {
            panic!("not a CSI line: {text}")
        };
        assert_eq!(packet.rssi, want.rssi);
        assert_eq!(packet.rx_timestamp_us, want.rx_timestamp_us);
        assert_eq!(packet.csi.as_bytes().as_slice(), want.csi.as_slice());
        assert!((clock.seconds(packet.rx_timestamp_us) - want.seconds).abs() < 1e-9);
    }
}

#[test]
fn room_lines_round_trip() {
    let golden: RoomGolden = load("room_lines.json");
    for text in &golden.lines {
        let Line::Csi(packet) = Line::parse(text).unwrap() else {
            unreachable!()
        };
        let Line::Csi(again) = Line::parse(&to_wire_line(&packet)).unwrap() else {
            unreachable!()
        };
        assert_eq!(again, packet);
    }
}

#[test]
fn report_lines_decode_like_the_python_windower() {
    let golden: HallGolden = load("hall_reports.json");
    for (text, want) in golden.lines.iter().zip(&golden.expected) {
        let Line::Report(report) = Line::parse(text).unwrap() else {
            panic!("not a report line: {text}")
        };
        assert_eq!(
            (report.receiver, report.source, report.packets),
            (want.receiver, want.source, want.packets)
        );
        assert_eq!(report.mean_db.as_slice(), want.mean_db.as_slice());
        assert_eq!(report.std_db.as_slice(), want.std_db.as_slice());
    }
}

#[test]
fn records_print_the_same_line_as_the_node_firmware() {
    let golden: HallGolden = load("hall_reports.json");
    for (record, line) in golden.records.iter().zip(&golden.lines) {
        let printed = to_wire_line(&as_record(record).as_link_report(golden.receiver));
        assert_eq!(printed.trim_end(), line);
    }
}

#[test]
fn espnow_report_frames_decode_to_the_firmware_records() {
    let golden: HallGolden = load("hall_reports.json");
    assert!(!golden.report_frames.is_empty());
    for frame in &golden.report_frames {
        let Ok(Frame::Report(report)) = Frame::decode(&bytes_of_hex(&frame.hex)) else {
            panic!("bad frame {}", frame.seq)
        };
        assert_eq!((report.receiver, report.seq), (golden.receiver, frame.seq));
        let decoded: Vec<_> = report.records().copied().collect();
        let expected: Vec<_> = frame.records.iter().map(as_record).collect();
        assert_eq!(decoded, expected);
        assert!(decoded.iter().all(|r| r.mean_q4.len() == N_GROUPS));
    }
}
