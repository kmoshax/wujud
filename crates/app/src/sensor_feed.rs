//! Where packets come from: real boards on serial ports, or the simulator playing a room in real time.

use std::collections::{BTreeSet, HashMap};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
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
    /// A board started measuring a device the window does not read yet (a board switched on later).
    NewLink(String),
    /// Signal strength on a link the window does not read (a phone): one in `HEARD_EVERY` packets.
    Heard { link: String, rssi: i8 },
}

const HEARD_EVERY: u32 = 10;

/// Lines a new device must send before it counts as a link, so a phone passing by does not.
const NEW_LINK_LINES: u32 = 300;

/// Every board counts from its own boot, so the hub's links and each node's links run on different
/// clocks. Each link keeps its board's timing but is anchored to the laptop's clock at its first packet,
/// and anchored again when it drifts off by more than `REANCHOR_SECONDS` (the board rebooted).
#[derive(Default)]
struct LinkClock {
    board: RxClockUnwrapper,
    offset: Option<f64>,
}

const REANCHOR_SECONDS: f64 = 5.0;

impl LinkClock {
    fn seconds(&mut self, rx_timestamp_us: u32, wall_seconds: f64) -> f64 {
        let board_seconds = self.board.seconds(rx_timestamp_us);
        let anchored = self
            .offset
            .map(|offset| board_seconds + offset)
            .filter(|t| (t - wall_seconds).abs() <= REANCHOR_SECONDS);
        anchored.unwrap_or_else(|| {
            self.offset = Some(wall_seconds - board_seconds);
            wall_seconds
        })
    }
}

/// `reading` is the links the window reads, in its order; the window changes it when a board joins.
pub fn from_serial_ports(
    ports: &[String],
    reading: Arc<Mutex<Vec<String>>>,
    record: Option<&str>,
) -> anyhow::Result<Receiver<FeedEvent>> {
    let lines = serial::read_ports(ports, record)?;
    let (events, receiver) = channel();
    let mut clocks: HashMap<String, LinkClock> = HashMap::new();
    let mut unknown: HashMap<String, u32> = HashMap::new();
    let mut announced = BTreeSet::new();
    thread::spawn(move || {
        for line in lines {
            let index = reading.lock().unwrap().iter().position(|l| *l == line.link);
            let Some(link) = index else {
                let count = unknown.entry(line.link.clone()).or_default();
                *count += 1;
                let mut event = None;
                if *count >= NEW_LINK_LINES && announced.insert(line.link.clone()) {
                    event = Some(FeedEvent::NewLink(line.link.clone()));
                } else if count.is_multiple_of(HEARD_EVERY) {
                    if let Ok(Line::Csi(packet)) = Line::parse(&line.text) {
                        event = Some(FeedEvent::Heard {
                            link: line.link,
                            rssi: packet.rssi,
                        });
                    }
                }
                if event.is_some_and(|e| events.send(e).is_err()) {
                    return;
                }
                continue;
            };
            let event = match Line::parse(&line.text) {
                Ok(Line::Csi(packet)) => FeedEvent::Frame(CsiFrame {
                    link,
                    board_seconds: clocks
                        .entry(line.link)
                        .or_default()
                        .seconds(packet.rx_timestamp_us, line.wall_seconds),
                    iq: packet.csi.as_bytes().to_vec(),
                    rssi: packet.rssi,
                }),
                Ok(Line::Info(info)) => FeedEvent::Banner {
                    link,
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
