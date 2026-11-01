//! A phone on the hub's network answers the hub's pings, and every board hears those answers: the
//! hub directly, each node from the air. Signal strength falls with distance, so each board's
//! reading says roughly how far the phone, and the person carrying it, is from that board.
//! Two boards measuring each other cannot tell this: their path is the same in both directions.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::room_setup;

/// What a board hears from a phone one metre away, and how fast that falls off indoors (log-distance).
const RSSI_AT_1M: f64 = -50.0;
const PATH_LOSS_EXPONENT: f64 = 2.2;
const SMOOTHING: f64 = 0.15;
const FORGET_AFTER: Duration = Duration::from_secs(5);

struct Heard {
    rssi: f64,
    at: Instant,
}

#[derive(Default)]
pub struct PhoneRange {
    heard: BTreeMap<(String, String), Heard>,
}

pub struct PhoneDistance {
    pub phone: String,
    /// Index into the room's nodes, and metres to that board.
    pub boards: Vec<(usize, f64)>,
}

impl PhoneRange {
    /// `node_ids` are the room's boards; a link whose sender is not one of them is a phone heard by its receiver.
    pub fn hear(&mut self, link: &str, rssi: f64, node_ids: &[String]) {
        let (sender, receiver) = room_setup::link_ends(link);
        if node_ids.contains(&sender) || !node_ids.contains(&receiver) {
            return;
        }
        let heard = self.heard.entry((sender, receiver)).or_insert(Heard {
            rssi,
            at: Instant::now(),
        });
        heard.rssi += SMOOTHING * (rssi - heard.rssi);
        heard.at = Instant::now();
    }

    pub fn distances(&self, node_ids: &[String]) -> Vec<PhoneDistance> {
        let mut phones: BTreeMap<&str, Vec<(usize, f64)>> = BTreeMap::new();
        for ((phone, board), heard) in &self.heard {
            if heard.at.elapsed() > FORGET_AFTER {
                continue;
            }
            if let Some(index) = node_ids.iter().position(|id| id == board) {
                phones.entry(phone).or_default().push((index, metres(heard.rssi)));
            }
        }
        phones
            .into_iter()
            .map(|(phone, boards)| PhoneDistance {
                phone: room_setup::mac_tail(phone).to_owned(),
                boards,
            })
            .collect()
    }
}

fn metres(rssi: f64) -> f64 {
    10f64
        .powf((RSSI_AT_1M - rssi) / (10.0 * PATH_LOSS_EXPONENT))
        .clamp(0.2, 8.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weaker_is_further() {
        assert!((metres(RSSI_AT_1M) - 1.0).abs() < 1e-9);
        assert!(metres(-70.0) > metres(-55.0));
    }

    #[test]
    fn a_phone_heard_by_hub_and_node() {
        let ids = vec!["router".to_owned(), "aa:aa:aa:aa:aa:01".to_owned()];
        let mut range = PhoneRange::default();
        range.hear("udp:192.168.4.1@bb:bb:bb:bb:bb:02", -50.0, &ids);
        range.hear("udp:192.168.4.1@aa:aa:aa:aa:aa:01>bb:bb:bb:bb:bb:02", -65.0, &ids);
        range.hear("udp:192.168.4.1@aa:aa:aa:aa:aa:01", -40.0, &ids);
        let phones = range.distances(&ids);
        assert_eq!(phones.len(), 1);
        let to = |board: usize| phones[0].boards.iter().find(|b| b.0 == board).unwrap().1;
        assert_eq!(phones[0].boards.len(), 2);
        assert!(to(0) < to(1));
    }
}
