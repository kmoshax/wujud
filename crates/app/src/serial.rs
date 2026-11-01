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
    /// `<board>@<device>`, or the board alone for untagged lines.
    pub link: String,
    pub wall_seconds: f64,
    pub text: String,
}

/// One reader thread per board; its lines go out on one channel tagged with their link's name,
/// including devices the board starts measuring later.
pub fn read_ports(ports: &[String], record: Option<&str>) -> anyhow::Result<Receiver<PortLine>> {
    let (sender, receiver) = channel();
    let recorder = record.map(Recorder::append_to).transpose()?;
    let mut takes_untagged: BTreeMap<&str, bool> = BTreeMap::new();
    for name in ports {
        let (board, device) = split_link_name(name);
        *takes_untagged.entry(board).or_default() |= device.is_none();
    }
    for (board, takes_untagged) in takes_untagged {
        let board = board.to_owned();
        let sender = sender.clone();
        let mut recorder = recorder.as_ref().map(Recorder::try_clone).transpose()?;
        thread::spawn(move || {
            for text in open_board(&board) {
                let (device, text) = split_device_tag(&text);
                // Untagged lines (status) only go to a link named after the board alone.
                if device.is_none() && !takes_untagged {
                    continue;
                }
                let line = PortLine {
                    link: link_name(&board, device),
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
    discover(ports).0
}

/// The links, and each link's median signal strength over the discovery seconds (dBm): how far apart
/// the boards are, for laying them out without being told.
pub fn discover(ports: &[String]) -> (Vec<String>, BTreeMap<String, f64>) {
    let mut links = Vec::new();
    let mut rssi = BTreeMap::new();
    for name in ports {
        if split_link_name(name).1.is_some() {
            links.push(name.clone());
            continue;
        }
        let devices = devices_seen(name);
        if devices.is_empty() {
            links.push(name.clone());
        }
        for (device, mut heard) in devices {
            let link = link_name(name, Some(&device));
            heard.sort_by(f64::total_cmp);
            if let Some(&middle) = heard.get(heard.len() / 2) {
                rssi.insert(link.clone(), middle);
            }
            links.push(link);
        }
    }
    (links, rssi)
}

fn devices_seen(board: &str) -> BTreeMap<String, Vec<f64>> {
    let started = Instant::now();
    let mut devices: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for text in open_board(board) {
        if let (Some(device), rest) = split_device_tag(&text) {
            let heard = devices.entry(device.to_owned()).or_default();
            if let Some(rssi) = rest.strip_prefix("P,").and_then(|r| r.split(',').next()?.parse().ok()) {
                heard.push(rssi);
            }
        }
        if started.elapsed() > DISCOVERY_TIME {
            break;
        }
    }
    devices
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

/// A line whose tag is not `<mac>`, `<mac>>hub` or `<mac>><mac>` was garbled on the way (a USB
/// hiccup glues two lines together): it comes back empty, so nobody reads it as a new device.
fn split_device_tag(text: &str) -> (Option<&str>, &str) {
    match text.split_once('|') {
        Some((device, rest)) if is_device_tag(device) => (Some(device), rest),
        Some(_) => (None, ""),
        None => (None, text),
    }
}

fn is_device_tag(tag: &str) -> bool {
    let is_mac = |part: &str| {
        part.len() == 17
            && part.split(':').count() == 6
            && part.split(':').all(|b| b.len() == 2 && b.chars().all(|c| c.is_ascii_hexdigit()))
    };
    match tag.split_once('>') {
        Some((node, from)) => is_mac(node) && (from == "hub" || is_mac(from)),
        None => is_mac(tag),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn garbled_tags_are_dropped() {
        assert_eq!(split_device_tag("ec:e3:34:22:74:ec|P,1"), (Some("ec:e3:34:22:74:ec"), "P,1"));
        assert_eq!(split_device_tag("ec:e3:34:22:74:ec>hub|P,1").0, Some("ec:e3:34:22:74:ec>hub"));
        assert_eq!(
            split_device_tag("9f009f00af10afec:e3:34:22:74:ec>hub|P,1"),
            (None, "")
        );
        assert_eq!(split_device_tag("I,wujud-room,hub"), (None, "I,wujud-room,hub"));
    }
}
