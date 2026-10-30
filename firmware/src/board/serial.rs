use core::fmt::Display;
use std::io::Write;

/// The console UART is the wire to the host: one protocol message per line, nothing else.
pub fn send_line(message: &impl Display) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{message}");
    let _ = out.flush();
}
