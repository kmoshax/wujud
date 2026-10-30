//! The firmware's per-link reports on a real ESP32 recording, against the Python emulation of hallsense_node.ino
//! (research/tools/export_golden_proto.py -> crates/proto/tests/golden/hall_reports.json).

use hallsense_proto::node_report::{group_levels_db, subcarrier_powers, LinkWindow};
use hallsense_proto::{CsiBuffer, ReportRecord, RxClockUnwrapper, CSI_BUFFER_LEN};
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use serde::Deserialize;
use std::fs::File;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Golden {
    source: String,
    records: Vec<GoldenRecord>,
}

#[derive(Deserialize)]
struct GoldenRecord {
    source: u8,
    packets: u8,
    mean_q4: Vec<i16>,
    std_q20: Vec<u8>,
}

struct Packet {
    seconds: f64,
    rssi: f32,
    csi: CsiBuffer,
}

fn espectre_recording(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("HOME").ok()?)
        .join("Projects/ext/espectre/data/motion")
        .join(name);
    path.exists().then_some(path)
}

fn packets_in_time_order(path: &PathBuf) -> Vec<Packet> {
    let mut npz = NpzReader::new(File::open(path).unwrap()).unwrap();
    let csi: Array2<i8> = npz.by_name("csi_data").unwrap();
    let rssi: Array1<i16> = npz.by_name("rssi_dbm").unwrap();
    let timestamps: Array1<u32> = npz.by_name("wifi_rx_ts_us").unwrap();
    let mut clock = RxClockUnwrapper::new();
    let mut packets: Vec<Packet> = (0..csi.nrows())
        .map(|row| {
            let bytes: [u8; CSI_BUFFER_LEN] = core::array::from_fn(|i| csi[[row, i]] as u8);
            Packet {
                seconds: clock.seconds(timestamps[row]),
                rssi: f32::from(rssi[row]),
                csi: CsiBuffer::from_unsigned_bytes(bytes),
            }
        })
        .collect();
    packets.sort_by(|a, b| a.seconds.total_cmp(&b.seconds));
    packets
}

fn one_second_records(packets: &[Packet], source: u8) -> Vec<ReportRecord> {
    let start = packets[0].seconds;
    let mut window = LinkWindow::default();
    let mut current_second = 0;
    let mut records = Vec::new();
    for packet in packets {
        let second = (packet.seconds - start).floor() as i64;
        while current_second < second {
            records.extend(window.close(source));
            current_second += 1;
        }
        window.add(
            &group_levels_db(&subcarrier_powers(&packet.csi, false), packet.rssi),
            packet.rssi,
        );
    }
    records.extend(window.close(source));
    records
}

#[test]
fn reports_match_the_node_firmware_on_real_csi() {
    let golden: Golden = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../crates/proto/tests/golden/hall_reports.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let Some(path) = espectre_recording(&golden.source) else {
        eprintln!(
            "skipped: {} not found (clone github.com/francescopace/espectre into ~/Projects/ext)",
            golden.source
        );
        return;
    };
    let ours = one_second_records(&packets_in_time_order(&path), golden.records[0].source);
    assert_eq!(ours.len(), golden.records.len());
    let mut exact = 0;
    let mut values = 0;
    for (mine, theirs) in ours.iter().zip(&golden.records) {
        assert_eq!((mine.source, mine.packets), (theirs.source, theirs.packets));
        let pairs = mine
            .mean_q4
            .iter()
            .map(|&q| i32::from(q))
            .zip(theirs.mean_q4.iter().map(|&q| i32::from(q)))
            .chain(
                mine.std_q20
                    .iter()
                    .map(|&q| i32::from(q))
                    .zip(theirs.std_q20.iter().map(|&q| i32::from(q))),
            );
        for (a, b) in pairs {
            // The board sums in f32, the Python emulation in f64: one quantization step apart at most.
            assert!((a - b).abs() <= 1, "{a} vs {b}");
            exact += usize::from(a == b);
            values += 1;
        }
    }
    eprintln!("{exact}/{values} quantized values identical, the rest one step apart");
}
