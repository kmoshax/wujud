//! A fixed-size text buffer so the firmware can format a whole line before writing it to the serial port.

use core::fmt;

/// Same size as the .ino's `static char line[300]`; the longest line (a P line) is 275 bytes.
pub const MAX_LINE_LEN: usize = 300;

#[derive(Clone)]
pub struct LineBuffer {
    bytes: [u8; MAX_LINE_LEN],
    len: usize,
}

impl LineBuffer {
    pub const fn new() -> Self {
        Self {
            bytes: [0; MAX_LINE_LEN],
            len: 0,
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    pub fn as_str(&self) -> &str {
        core::str::from_utf8(self.as_bytes()).expect("only whole &str values are ever written")
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for LineBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Write for LineBuffer {
    /// Fails, leaving the buffer unchanged, when the text does not fit.
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let end = self.len + text.len();
        let free = self.bytes.get_mut(self.len..end).ok_or(fmt::Error)?;
        free.copy_from_slice(text.as_bytes());
        self.len = end;
        Ok(())
    }
}

impl fmt::Debug for LineBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::fmt::Write;

    #[test]
    fn collects_writes() {
        let mut buf = LineBuffer::new();
        write!(buf, "P,{},{}", -60, 12).unwrap();
        assert_eq!(buf.as_str(), "P,-60,12");
        assert_eq!(buf.len(), 8);
    }

    #[test]
    fn overflow_is_an_error_and_keeps_contents() {
        let mut buf = LineBuffer::new();
        buf.write_str(&"x".repeat(MAX_LINE_LEN - 1)).unwrap();
        assert!(buf.write_str("yz").is_err());
        assert_eq!(buf.len(), MAX_LINE_LEN - 1);
        buf.write_str("y").unwrap();
        assert_eq!(buf.len(), MAX_LINE_LEN);
    }
}
