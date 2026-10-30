//! MAC addresses as the firmware prints them: `aa:bb:cc:dd:ee:ff`.

use core::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MacAddress(pub [u8; 6]);

/// Arduino's `WiFi.macAddress()` prints uppercase (I lines); the node's `%02x` prints lowercase (N lines).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacCase {
    Lower,
    Upper,
}

impl MacAddress {
    pub const BROADCAST: MacAddress = MacAddress([0xff; 6]);

    pub fn parse(text: &str) -> Option<Self> {
        let mut bytes = [0u8; 6];
        let mut parts = text.split(':');
        for byte in &mut bytes {
            let part = parts.next()?;
            if part.len() != 2 || !part.bytes().all(|c| c.is_ascii_hexdigit()) {
                return None;
            }
            *byte = u8::from_str_radix(part, 16).ok()?;
        }
        parts.next().is_none().then_some(MacAddress(bytes))
    }

    pub fn display(self, case: MacCase) -> impl fmt::Display {
        MacText { mac: self, case }
    }
}

struct MacText {
    mac: MacAddress,
    case: MacCase,
}

impl fmt::Display for MacText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, byte) in self.mac.0.iter().enumerate() {
            if i > 0 {
                f.write_str(":")?;
            }
            match self.case {
                MacCase::Lower => write!(f, "{byte:02x}")?,
                MacCase::Upper => write!(f, "{byte:02X}")?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LineBuffer;
    use core::fmt::Write;

    const MAC: MacAddress = MacAddress([0x24, 0x0a, 0xc4, 0x00, 0xbe, 0xef]);

    #[test]
    fn prints_both_cases() {
        let mut out = LineBuffer::new();
        write!(
            out,
            "{} {}",
            MAC.display(MacCase::Lower),
            MAC.display(MacCase::Upper)
        )
        .unwrap();
        assert_eq!(out.as_str(), "24:0a:c4:00:be:ef 24:0A:C4:00:BE:EF");
    }

    #[test]
    fn parses_either_case() {
        assert_eq!(MacAddress::parse("24:0a:c4:00:be:ef"), Some(MAC));
        assert_eq!(MacAddress::parse("24:0A:C4:00:BE:EF"), Some(MAC));
    }

    #[test]
    fn rejects_malformed() {
        for bad in [
            "",
            "replay",
            "24:0a:c4:00:be",
            "24:0a:c4:00:be:ef:00",
            "24:0a:c4:00:be:eg",
            "240a:c4:00:be:ef:0",
            "+4:0a:c4:00:be:ef",
        ] {
            assert_eq!(MacAddress::parse(bad), None, "{bad}");
        }
    }
}
