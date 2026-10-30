//! When a hall node sends its next beacon and its next report (wujud_node.ino loop).

use crate::REPORT_INTERVAL_MS;

pub const BEACON_JITTER_MS: i32 = 5;

#[derive(Clone, Copy, Debug)]
pub struct BeaconClock {
    beacon_period_ms: u32,
    next_beacon_ms: u32,
    next_report_ms: u32,
}

impl BeaconClock {
    pub fn new(beacons_per_second: u32) -> Self {
        BeaconClock {
            beacon_period_ms: 1000 / beacons_per_second,
            next_beacon_ms: 0,
            next_report_ms: REPORT_INTERVAL_MS,
        }
    }

    /// `jitter_ms` in -5..=5 keeps nodes from transmitting in lock-step and colliding forever.
    pub fn beacon_due(&mut self, now_ms: u32, jitter_ms: i32) -> bool {
        if !reached(now_ms, self.next_beacon_ms) {
            return false;
        }
        let jitter = jitter_ms.clamp(-BEACON_JITTER_MS, BEACON_JITTER_MS);
        self.next_beacon_ms = now_ms
            .wrapping_add(self.beacon_period_ms)
            .wrapping_add_signed(jitter);
        true
    }

    pub fn report_due(&mut self, now_ms: u32) -> bool {
        if !reached(now_ms, self.next_report_ms) {
            return false;
        }
        self.next_report_ms = self.next_report_ms.wrapping_add(REPORT_INTERVAL_MS);
        true
    }
}

fn reached(now_ms: u32, deadline_ms: u32) -> bool {
    now_ms.wrapping_sub(deadline_ms) as i32 >= 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beacons_at_the_configured_rate() {
        let mut clock = BeaconClock::new(20);
        let sent = (0..1000).filter(|&ms| clock.beacon_due(ms, 0)).count();
        assert_eq!(sent, 20);
    }

    #[test]
    fn reports_once_a_second_without_drift() {
        let mut clock = BeaconClock::new(20);
        let due: Vec<u32> = (0..3500).filter(|&ms| clock.report_due(ms)).collect();
        assert_eq!(due, vec![1000, 2000, 3000]);
    }

    #[test]
    fn survives_the_millisecond_counter_wrap() {
        let mut clock = BeaconClock::new(20);
        clock.next_beacon_ms = u32::MAX - 10;
        assert!(!clock.beacon_due(u32::MAX - 20, 0));
        assert!(clock.beacon_due(5, 0));
    }
}
