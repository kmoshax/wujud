use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use esp_idf_svc::espnow::{EspNow, PeerInfo};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::sys::{
    esp, esp_random, esp_wifi_config_espnow_rate, esp_wifi_set_channel, esp_wifi_set_ps,
    wifi_interface_t_WIFI_IF_STA, wifi_phy_rate_t_WIFI_PHY_RATE_MCS0_LGI,
    wifi_ps_type_t_WIFI_PS_NONE, wifi_second_chan_t_WIFI_SECOND_CHAN_NONE,
};
use esp_idf_svc::wifi::{BlockingWifi, ClientConfiguration, Configuration, EspWifi, WifiDeviceId};
use wujud_firmware::beacon_clock::{BeaconClock, BEACON_JITTER_MS};
use wujud_proto::node_report::{group_levels_db, subcarrier_powers, LinkWindow};
use wujud_firmware::node_table::NodeTable;
use wujud_firmware::{HALL_BEACON_HZ, HALL_CHANNEL};
use wujud_proto::{
    BeaconFrame, Frame, MacAddress, NodeBanner, NodeSeen, ReportFrame, MAX_NODES,
    MAX_REPORT_FRAME_LEN,
};

use super::csi::{self, CsiEvent};
use super::serial::send_line;

const NODE_ID: u8 = parse_node_id(env!("WUJUD_NODE_ID"));

pub fn run() -> anyhow::Result<()> {
    let peripherals = Peripherals::take()?;
    let event_loop = EspSystemEventLoop::take()?;
    let mut wifi = BlockingWifi::wrap(
        EspWifi::new(
            peripherals.modem,
            event_loop.clone(),
            Some(EspDefaultNvsPartition::take()?),
        )?,
        event_loop,
    )?;
    wifi.set_configuration(&Configuration::Client(ClientConfiguration::default()))?;
    wifi.start()?;
    esp!(unsafe { esp_wifi_set_ps(wifi_ps_type_t_WIFI_PS_NONE) })?;
    esp!(unsafe { esp_wifi_set_channel(HALL_CHANNEL, wifi_second_chan_t_WIFI_SECOND_CHAN_NONE) })?;

    let espnow = EspNow::take()?;
    espnow.add_peer(PeerInfo {
        peer_addr: MacAddress::BROADCAST.0,
        channel: HALL_CHANNEL,
        ifidx: wifi_interface_t_WIFI_IF_STA,
        ..Default::default()
    })?;
    // HT20 MCS0 is OFDM, so receivers get CSI; the default 1 Mbps DSSS has none.
    esp!(unsafe {
        esp_wifi_config_espnow_rate(
            wifi_interface_t_WIFI_IF_STA,
            wifi_phy_rate_t_WIFI_PHY_RATE_MCS0_LGI,
        )
    })?;

    let nodes = Arc::new(Mutex::new(NodeTable::default()));
    let heard_nodes = Arc::clone(&nodes);
    espnow.register_recv_cb(move |info, data| {
        on_espnow(&heard_nodes, MacAddress(*info.src_addr), data)
    })?;
    let csi_events = csi::start()?;
    send_line(&NodeBanner {
        node: NODE_ID,
        channel: HALL_CHANNEL,
        mac: MacAddress(wifi.wifi().get_mac(WifiDeviceId::Sta)?),
    });

    let mut windows = [LinkWindow::default(); MAX_NODES];
    let mut clock = BeaconClock::new(HALL_BEACON_HZ);
    let mut seq: u16 = 0;
    let started = Instant::now();
    loop {
        for event in csi_events.try_iter() {
            add_to_window(&nodes, &mut windows, &event);
        }
        let now_ms = started.elapsed().as_millis() as u32;
        if clock.beacon_due(now_ms, random_jitter_ms()) {
            espnow.send(
                MacAddress::BROADCAST.0,
                &BeaconFrame {
                    node_id: NODE_ID,
                    seq,
                }
                .encode(),
            )?;
            seq = seq.wrapping_add(1);
        }
        if clock.report_due(now_ms) {
            send_reports(&espnow, &mut windows, seq)?;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn add_to_window(
    nodes: &Mutex<NodeTable>,
    windows: &mut [LinkWindow; MAX_NODES],
    event: &CsiEvent,
) {
    let Some(source) = nodes.lock().unwrap().node_of(event.sender) else {
        return;
    };
    if source == NODE_ID {
        return;
    }
    let rssi = f32::from(event.rssi);
    let levels = group_levels_db(
        &subcarrier_powers(&event.csi, event.first_word_invalid),
        rssi,
    );
    windows[usize::from(source)].add(&levels, rssi);
}

fn send_reports(
    espnow: &EspNow,
    windows: &mut [LinkWindow; MAX_NODES],
    seq: u16,
) -> anyhow::Result<()> {
    let mut frame = ReportFrame::new(NODE_ID, seq);
    for (source, window) in windows.iter_mut().enumerate() {
        let Some(record) = window.close(source as u8) else {
            continue;
        };
        send_line(&record.as_link_report(NODE_ID));
        frame
            .push(record)
            .expect("frame is flushed before it overflows");
        if frame.is_full() {
            broadcast(espnow, &frame)?;
            frame = ReportFrame::new(NODE_ID, seq);
        }
    }
    if !frame.is_empty() {
        broadcast(espnow, &frame)?;
    }
    Ok(())
}

fn broadcast(espnow: &EspNow, frame: &ReportFrame) -> anyhow::Result<()> {
    let mut bytes = [0u8; MAX_REPORT_FRAME_LEN];
    let len = frame.encode(&mut bytes);
    espnow.send(MacAddress::BROADCAST.0, &bytes[..len])?;
    Ok(())
}

fn on_espnow(nodes: &Mutex<NodeTable>, sender: MacAddress, data: &[u8]) {
    match Frame::decode(data) {
        Ok(Frame::Beacon(beacon)) => {
            if nodes.lock().unwrap().learn(beacon.node_id, sender) {
                send_line(&NodeSeen {
                    node: beacon.node_id,
                    mac: sender,
                });
            }
        }
        Ok(Frame::Report(report)) => {
            for record in report.records() {
                send_line(&record.as_link_report(report.receiver));
            }
        }
        Err(_) => {}
    }
}

fn random_jitter_ms() -> i32 {
    (unsafe { esp_random() } % (2 * BEACON_JITTER_MS as u32 + 1)) as i32 - BEACON_JITTER_MS
}

const fn parse_node_id(text: &str) -> u8 {
    let bytes = text.as_bytes();
    let mut value = 0u32;
    let mut i = 0;
    while i < bytes.len() {
        assert!(
            bytes[i].is_ascii_digit(),
            "WUJUD_NODE_ID must be a number"
        );
        value = value * 10 + (bytes[i] - b'0') as u32;
        i += 1;
    }
    assert!(
        value < MAX_NODES as u32,
        "WUJUD_NODE_ID must be below MAX_NODES"
    );
    value as u8
}
