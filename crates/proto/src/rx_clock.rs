//! Turns the board's 32-bit microsecond receive timestamps into a monotonic time in seconds.

/// Port of the room reader loop in research/live.py (`wraps += 1` whenever the timestamp goes backwards).
/// Like the Python, a board reset also counts as one wrap.
#[derive(Clone, Copy, Debug, Default)]
pub struct RxClockUnwrapper {
    previous_us: Option<u32>,
    wraps: u64,
}

impl RxClockUnwrapper {
    pub const fn new() -> Self {
        Self {
            previous_us: None,
            wraps: 0,
        }
    }

    pub fn seconds(&mut self, rx_timestamp_us: u32) -> f64 {
        if self
            .previous_us
            .is_some_and(|previous| rx_timestamp_us < previous)
        {
            self.wraps += 1;
        }
        self.previous_us = Some(rx_timestamp_us);
        self.unwrapped_seconds(rx_timestamp_us)
    }

    pub fn last_seconds(&self) -> Option<f64> {
        self.previous_us.map(|us| self.unwrapped_seconds(us))
    }

    fn unwrapped_seconds(&self, rx_timestamp_us: u32) -> f64 {
        (u64::from(rx_timestamp_us) + (self.wraps << 32)) as f64 / 1e6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continues_across_the_wrap() {
        let mut clock = RxClockUnwrapper::new();
        assert_eq!(clock.seconds(u32::MAX - 999_999), 4293.967296);
        assert_eq!(clock.seconds(500_000), 4295.467296);
        assert_eq!(clock.seconds(600_000), 4295.567296);
    }

    #[test]
    fn equal_timestamps_are_not_a_wrap() {
        let mut clock = RxClockUnwrapper::new();
        assert_eq!(clock.seconds(10), clock.seconds(10));
    }
}
