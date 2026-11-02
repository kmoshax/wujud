//! Lines from a board on Wi-Fi: the room firmware streams every CSI line to whoever asked last.
//! A datagram is one line from the hub itself, or a batch of whole lines a node passed through it.

use std::collections::VecDeque;
use std::net::UdpSocket;
use std::time::{Duration, Instant};

pub const CSI_PORT: u16 = 5005;
const ASK_EVERY: Duration = Duration::from_secs(1);
const READ_TIMEOUT: Duration = Duration::from_millis(200);
const MAX_DATAGRAM: usize = 2048;
/// A few seconds of every link: a moment the app is busy no longer drops datagrams.
const RECEIVE_BUFFER: usize = 4 << 20;

/// Asking again every second keeps the stream coming after the board resets or the laptop reconnects.
pub struct UdpReader {
    board: String,
    socket: Option<UdpSocket>,
    last_ask: Option<Instant>,
    pending: VecDeque<String>,
}

impl UdpReader {
    pub fn new(board_ip: &str) -> Self {
        UdpReader {
            board: format!("{board_ip}:{CSI_PORT}"),
            socket: None,
            last_ask: None,
            pending: VecDeque::new(),
        }
    }

    fn socket(&mut self) -> Option<&UdpSocket> {
        if self.socket.is_none() {
            match bind() {
                Ok(socket) => {
                    let _ = socket.set_read_timeout(Some(READ_TIMEOUT));
                    self.socket = Some(socket);
                }
                Err(error) => {
                    eprintln!("udp: {error} - retrying");
                    std::thread::sleep(ASK_EVERY);
                }
            }
        }
        self.socket.as_ref()
    }

    fn ask_if_due(&mut self) {
        if self.last_ask.is_some_and(|t| t.elapsed() < ASK_EVERY) {
            return;
        }
        self.last_ask = Some(Instant::now());
        let board = self.board.clone();
        if let Some(socket) = self.socket() {
            let _ = socket.send_to(b"wujud?", &board);
        }
    }
}

impl Iterator for UdpReader {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        let mut buffer = [0u8; MAX_DATAGRAM];
        loop {
            if let Some(line) = self.pending.pop_front() {
                return Some(line);
            }
            self.ask_if_due();
            let Some(socket) = self.socket.as_ref() else {
                continue;
            };
            if let Ok(length) = socket.recv(&mut buffer) {
                let text = String::from_utf8_lossy(&buffer[..length]);
                self.pending
                    .extend(text.split_inclusive('\n').map(str::to_owned));
            }
        }
    }
}

fn bind() -> std::io::Result<UdpSocket> {
    let socket = socket2::Socket::new(socket2::Domain::IPV4, socket2::Type::DGRAM, Some(socket2::Protocol::UDP))?;
    let _ = socket.set_recv_buffer_size(RECEIVE_BUFFER);
    socket.bind(&std::net::SocketAddr::from(([0, 0, 0, 0], 0)).into())?;
    Ok(socket.into())
}
