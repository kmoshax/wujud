//! Lowercase hex encoding of byte buffers, without allocation.

use core::fmt;

const DIGITS: &[u8; 16] = b"0123456789abcdef";

pub fn write_hex(bytes: &[u8], out: &mut impl fmt::Write) -> fmt::Result {
    for &byte in bytes {
        out.write_char(DIGITS[usize::from(byte >> 4)] as char)?;
        out.write_char(DIGITS[usize::from(byte & 0x0f)] as char)?;
    }
    Ok(())
}

/// Decodes `2 * N` hex digits (either case). `None` on a wrong length or a non-hex character.
pub fn decode_hex<const N: usize>(text: &str) -> Option<[u8; N]> {
    let digits = text.as_bytes();
    if digits.len() != 2 * N {
        return None;
    }
    let mut out = [0u8; N];
    for (byte, &[high, low]) in out.iter_mut().zip(digits.as_chunks::<2>().0) {
        *byte = (nibble(high)? << 4) | nibble(low)?;
    }
    Some(out)
}

fn nibble(digit: u8) -> Option<u8> {
    match digit {
        b'0'..=b'9' => Some(digit - b'0'),
        b'a'..=b'f' => Some(digit - b'a' + 10),
        b'A'..=b'F' => Some(digit - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LineBuffer;

    #[test]
    fn encodes_lowercase() {
        let mut out = LineBuffer::new();
        write_hex(&[0x00, 0x7f, 0x80, 0xff, 0xa5], &mut out).unwrap();
        assert_eq!(out.as_str(), "007f80ffa5");
    }

    #[test]
    fn decodes_both_cases() {
        assert_eq!(decode_hex::<3>("00fFa5"), Some([0x00, 0xff, 0xa5]));
    }

    #[test]
    fn rejects_bad_digits_and_lengths() {
        assert_eq!(decode_hex::<2>("0g12"), None);
        assert_eq!(decode_hex::<2>("012"), None);
        assert_eq!(decode_hex::<2>("01234"), None);
        assert_eq!(decode_hex::<1>("+1"), None);
    }

    #[test]
    fn round_trips_every_byte_value() {
        let all: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut text = LineBuffer::new();
        write_hex(&all[..128], &mut text).unwrap();
        assert_eq!(decode_hex::<128>(text.as_str()).unwrap()[..], all[..128]);
        text.clear();
        write_hex(&all[128..], &mut text).unwrap();
        assert_eq!(decode_hex::<128>(text.as_str()).unwrap()[..], all[128..]);
    }
}
