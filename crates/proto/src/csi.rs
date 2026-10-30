//! The 128-byte CSI buffer exactly as ESP-IDF's `wifi_csi_info_t::buf` hands it over.

pub const CSI_BUFFER_LEN: usize = 128;
pub const CSI_BINS: usize = CSI_BUFFER_LEN / 2;

/// L-LTF data subcarriers -26..-1, 1..26, in the order of research/wujud/config.py::SC_IDX.
pub const L_LTF_DATA_SUBCARRIERS: [i8; 52] = l_ltf_data_subcarriers();

const fn l_ltf_data_subcarriers() -> [i8; 52] {
    let mut out = [0i8; 52];
    let mut i = 0;
    let mut k: i8 = -26;
    while k <= 26 {
        if k != 0 {
            out[i] = k;
            i += 1;
        }
        k += 1;
    }
    out
}

/// One complex channel estimate; ESP-IDF stores the imaginary byte first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsiSample {
    pub imag: i8,
    pub real: i8,
}

impl CsiSample {
    pub fn power(self) -> u32 {
        let imag = i32::from(self.imag);
        let real = i32::from(self.real);
        (imag * imag + real * real) as u32
    }
}

/// Bins 0..=31 hold subcarriers 0..=31 and bins 32..=63 hold -32..=-1 (FFT order).
pub fn bin_of_subcarrier(subcarrier: i8) -> Option<usize> {
    match subcarrier {
        0..=31 => Some(subcarrier as usize),
        -32..=-1 => Some((CSI_BINS as i16 + i16::from(subcarrier)) as usize),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsiBuffer {
    bytes: [i8; CSI_BUFFER_LEN],
}

impl CsiBuffer {
    pub const fn from_bytes(bytes: [i8; CSI_BUFFER_LEN]) -> Self {
        Self { bytes }
    }

    /// Keeps the first 128 bytes; `None` when the driver delivered fewer (the .ino drops those packets).
    pub fn from_driver_buffer(buf: &[i8]) -> Option<Self> {
        let first: &[i8; CSI_BUFFER_LEN] = buf.get(..CSI_BUFFER_LEN)?.try_into().ok()?;
        Some(Self::from_bytes(*first))
    }

    pub fn as_bytes(&self) -> &[i8; CSI_BUFFER_LEN] {
        &self.bytes
    }

    pub fn as_unsigned_bytes(&self) -> [u8; CSI_BUFFER_LEN] {
        self.bytes.map(|b| b as u8)
    }

    pub fn from_unsigned_bytes(bytes: [u8; CSI_BUFFER_LEN]) -> Self {
        Self::from_bytes(bytes.map(|b| b as i8))
    }

    /// Panics if `bin >= 64`.
    pub fn bin(&self, bin: usize) -> CsiSample {
        CsiSample {
            imag: self.bytes[2 * bin],
            real: self.bytes[2 * bin + 1],
        }
    }

    pub fn subcarrier(&self, subcarrier: i8) -> Option<CsiSample> {
        bin_of_subcarrier(subcarrier).map(|bin| self.bin(bin))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_subcarriers_skip_dc() {
        assert_eq!(L_LTF_DATA_SUBCARRIERS[0], -26);
        assert_eq!(L_LTF_DATA_SUBCARRIERS[25], -1);
        assert_eq!(L_LTF_DATA_SUBCARRIERS[26], 1);
        assert_eq!(L_LTF_DATA_SUBCARRIERS[51], 26);
    }

    #[test]
    fn subcarrier_to_bin_follows_fft_order() {
        assert_eq!(bin_of_subcarrier(0), Some(0));
        assert_eq!(bin_of_subcarrier(26), Some(26));
        assert_eq!(bin_of_subcarrier(-1), Some(63));
        assert_eq!(bin_of_subcarrier(-26), Some(38));
        assert_eq!(bin_of_subcarrier(-32), Some(32));
        assert_eq!(bin_of_subcarrier(32), None);
        assert_eq!(bin_of_subcarrier(-33), None);
    }

    #[test]
    fn imaginary_byte_comes_first() {
        let mut bytes = [0i8; CSI_BUFFER_LEN];
        bytes[2 * 5] = -3;
        bytes[2 * 5 + 1] = 4;
        let csi = CsiBuffer::from_bytes(bytes);
        assert_eq!(csi.subcarrier(5), Some(CsiSample { imag: -3, real: 4 }));
        assert_eq!(csi.bin(5).power(), 25);
    }

    #[test]
    fn power_of_most_negative_sample_does_not_overflow() {
        let sample = CsiSample {
            imag: i8::MIN,
            real: i8::MIN,
        };
        assert_eq!(sample.power(), 2 * 128 * 128);
    }

    #[test]
    fn short_driver_buffers_are_rejected_and_long_ones_truncated() {
        assert!(CsiBuffer::from_driver_buffer(&[0; 127]).is_none());
        let mut long = [7i8; 384];
        long[0] = 1;
        let csi = CsiBuffer::from_driver_buffer(&long).unwrap();
        assert_eq!(csi.as_bytes()[0], 1);
        assert_eq!(csi.as_bytes()[127], 7);
    }

    #[test]
    fn unsigned_round_trip() {
        let csi = CsiBuffer::from_bytes(core::array::from_fn(|i| (i as i32 - 64) as i8));
        assert_eq!(CsiBuffer::from_unsigned_bytes(csi.as_unsigned_bytes()), csi);
    }
}
