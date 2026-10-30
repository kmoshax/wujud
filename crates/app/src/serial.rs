//! Lines from the boards (USB serial ports, or Wi-Fi as `udp:<ip>`), surviving unplug/reset,
//! optionally recorded to a file.
//!
//! The room firmware tags every line with the device it measured (`<mac>|P,...`); each device is
//! its own link, named `<board>@<mac>`. A link named after the board alone takes every line.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::sync::mpsc::{channel, Receiver};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::udp::UdpReader;

pub const BAUD: u32 = 921_600;
const READ_TIMEOUT: Duration = Duration::from_millis(200);
const REOPEN_DELAY: Duration = Duration::from_secs(1);
const DISCOVERY_TIME: Duration = Duration::from_secs(3);

pub struct PortLine {
    pub port: usize,
    pub wall_seconds: f64,
    pub text: String,
}

/// One reader thread per board; its lines go out on one channel tagged with the link's index.
pub fn read_ports(ports: &[String], record: Option<&str>) -> anyhow::Result<Receiver<PortLine>> {
    let (sender, receiver) = channel();
    let recorder = record.map(Recorder::append_to).transpose()?;
    let mut links_by_board: BTreeMap<&str, BTreeMap<Option<String>, usize>> = BTreeMap::new();
    for (index, name) in ports.iter().enumerate() {
        let (board, device) = split_link_name(name);
        links_by_board
            .entry(board)
            .or_default()
            .insert(device.map(str::to_owned), index);
    }
    for (board, links) in links_by_board {
        let board = board.to_owned();
        let sender = sender.clone();
        let mut recorder = recorder.as_ref().map(Recorder::try_clone).transpose()?;
        thread::spawn(move || {
            for text in open_board(&board) {
                let (device, text) = split_device_tag(&text);
                let Some(&port) = device
                    .and_then(|d| links.get(&Some(d.to_owned())))
                    .or_else(|| links.get(&None))
                else {
                    continue;
                };
                let line = PortLine {
                    port,
                    wall_seconds: now_seconds(),
                    text: text.to_owned(),
                };
                if let Some(recorder) = recorder.as_mut() {
                    recorder.write(&line, &link_name(&board, device));
                }
                if sender.send(line).is_err() {
                    return;
                }
            }
        });
    }
    Ok(receiver)
}

/// Turns each board without a device into one link per device it currently measures.
pub fn one_link_per_device(ports: &[String]) -> Vec<String> {
    ports
        .iter()
        .flat_map(|name| {
            if split_link_name(name).1.is_some() {
                return vec![name.clone()];
            }
            let devices = devices_seen(name);
            if devices.is_empty() {
                vec![name.clone()]
            } else {
                devices.iter().map(|d| link_name(name, Some(d))).collect()
            }
        })
        .collect()
}

fn devices_seen(board: &str) -> Vec<String> {
    let started = Instant::now();
    let mut devices = std::collections::BTreeSet::new();
    for text in open_board(board) {
        if let (Some(device), _) = split_device_tag(&text) {
            devices.insert(device.to_owned());
        }
        if started.elapsed() > DISCOVERY_TIME {
            break;
        }
    }
    devices.into_iter().collect()
}

fn open_board(board: &str) -> Box<dyn Iterator<Item = String>> {
    match board.strip_prefix("udp:") {
        Some(board_ip) => Box::new(UdpReader::new(board_ip)),
        None => Box::new(PortReader::new(board.to_owned())),
    }
}

pub fn split_link_name(name: &str) -> (&str, Option<&str>) {
    match name.split_once('@') {
        Some((board, device)) => (board, Some(device)),
        None => (name, None),
    }
}

fn link_name(board: &str, device: Option<&str>) -> String {
    match device {
        Some(device) => format!("{board}@{device}"),
        None => board.to_owned(),
    }
}

fn split_device_tag(text: &str) -> (Option<&str>, &str) {
    match text.split_once('|') {
        Some((device, rest)) => (Some(device), rest),
        None => (None, text),
    }
}

struct PortReader {
    name: String,
    reader: Option<BufReader<Box<dyn serialport::SerialPort>>>,
}

impl PortReader {
    fn new(name: String) -> Self {
        PortReader { name, reader: None }
    }

    fn open(&mut self) -> Option<&mut BufReader<Box<dyn serialport::SerialPort>>> {
        if self.reader.is_none() {
            match serialport::new(&self.name, BAUD)
                .timeout(READ_TIMEOUT)
                .open()
            {
                Ok(port) => self.reader = Some(BufReader::new(port)),
                Err(error) => {
                    eprintln!("{}: {error} - retrying", self.name);
                    thread::sleep(REOPEN_DELAY);
                }
            }
        }
        self.reader.as_mut()
    }
}

impl Iterator for PortReader {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        loop {
            let Some(reader) = self.open() else { continue };
            let mut text = String::new();
            match reader.read_line(&mut text) {
                Ok(0) => continue,
                Ok(_) => return Some(text),
                Err(error) if error.kind() == ErrorKind::TimedOut => continue,
                Err(error) => {
                    eprintln!("{}: {error} - reconnecting", self.name);
                    self.reader = None;
                    thread::sleep(REOPEN_DELAY);
                }
            }
        }
    }
}

/// `<unix seconds>,<port>,<line>` per line, the format research/live.py records and replays.
struct Recorder {
    file: File,
}

impl Recorder {
    fn append_to(path: &str) -> anyhow::Result<Self> {
        Ok(Recorder {
            file: OpenOptions::new().create(true).append(true).open(path)?,
        })
    }

    fn try_clone(&self) -> anyhow::Result<Self> {
        Ok(Recorder {
            file: self.file.try_clone()?,
        })
    }

    fn write(&mut self, line: &PortLine, port_name: &str) {
        let _ = writeln!(
            self.file,
            "{:.3},{port_name},{}",
            line.wall_seconds,
            line.text.trim_end()
        );
    }
}

fn now_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}
