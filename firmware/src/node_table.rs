//! Which MAC address belongs to which hall node, learned from the nodes' beacons.

use hallsense_proto::{MacAddress, MAX_NODES};

#[derive(Clone, Debug, Default)]
pub struct NodeTable {
    macs: [Option<MacAddress>; MAX_NODES],
}

impl NodeTable {
    /// True the first time a node is heard, which is when the sink prints its `N` line.
    pub fn learn(&mut self, node_id: u8, mac: MacAddress) -> bool {
        match self.macs.get_mut(usize::from(node_id)) {
            Some(slot @ None) => {
                *slot = Some(mac);
                true
            }
            _ => false,
        }
    }

    pub fn node_of(&self, mac: MacAddress) -> Option<u8> {
        self.macs
            .iter()
            .position(|known| *known == Some(mac))
            .map(|id| id as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_each_node_once() {
        let mut table = NodeTable::default();
        let mac = MacAddress([1, 2, 3, 4, 5, 6]);
        assert!(table.learn(7, mac));
        assert!(!table.learn(7, MacAddress([9; 6])));
        assert_eq!(table.node_of(mac), Some(7));
        assert!(!table.learn(MAX_NODES as u8, mac));
    }
}
