//! Where packets come from: real boards on serial ports, or the simulator playing a room in real time.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use wujud_proto::{Line, RxClockUnwrapper};
use wujud_sim::stream::{RoomSecond, RoomStream};

use crate::serial;

pub struct CsiFrame {
    pub link: usize,
    pub board_seconds: f64,
    /// Interleaved I/Q bytes exactly as the board (or the simulated board) reports them.
    pub iq: Vec<i8>,
    pub rssi: i8,
}

pub struct SimulatedTruth {
    pub activity: String,
    pub breath_bpm: Option<f64>,
    pub xy: Option<[f64; 2]>,
}

pub enum FeedEvent {
    Frame(CsiFrame),
    Banner { link: usize, text: String },
    Truth(SimulatedTruth),
}

pub fn from_serial_ports(
    ports: &[String],
    record: Option<&str>,
) -> anyhow::Result<Receiver<FeedEvent>> {
    let lines = serial::read_ports(ports, record)?;
    let (events, receiver) = channel();
    let mut clocks: Vec<RxClockUnwrapper> = ports.iter().map(|_| RxClockUnwrapper::new()).collect();
    thread::spawn(move || {
        for line in lines {
            let event = match Line::parse(&line.text) {
                Ok(Line::Csi(packet)) => FeedEvent::Frame(CsiFrame {
                    link: line.port,
                    board_seconds: clocks[line.port].seconds(packet.rx_timestamp_us),
                    iq: packet.csi.as_bytes().to_vec(),
                    rssi: packet.rssi,
                }),
                Ok(Line::Info(info)) => FeedEvent::Banner {
                    link: line.port,
                    text: info.to_string(),
                },
                _ => continue,
            };
            if events.send(event).is_err() {
                return;
            }
        }
    });
    Ok(receiver)
}

/// Plays the simulated room second by second at wall-clock speed, so it looks like boards are attached.
pub fn from_simulator(mut stream: RoomStream) -> Receiver<FeedEvent> {
    let (events, receiver) = channel();
    thread::spawn(move || {
        let started = Instant::now();
        for second in stream.by_ref() {
            let due = Duration::from_secs(second.second as u64);
            thread::sleep(due.saturating_sub(started.elapsed()));
            if send_second(&events, second).is_err() {
                return;
            }
        }
    });
    receiver
}

fn send_second(
    events: &Sender<FeedEvent>,
    second: RoomSecond,
) -> Result<(), std::sync::mpsc::SendError<FeedEvent>> {
    for (link, capture) in second.links.iter().enumerate() {
        for packet in 0..capture.t.len() {
            let iq = capture
                .iq
                .index_axis(ndarray::Axis(0), packet)
                .iter()
                .copied()
                .collect();
            events.send(FeedEvent::Frame(CsiFrame {
                link,
                board_seconds: capture.t[packet],
                iq,
                rssi: capture.rssi[packet].clamp(i16::from(i8::MIN), i16::from(i8::MAX)) as i8,
            }))?;
        }
    }
    events.send(FeedEvent::Truth(SimulatedTruth {
        activity: format!("{:?}", second.truth.activity).to_lowercase(),
        breath_bpm: second.truth.breath_bpm,
        xy: second.truth.xy,
    }))
}
