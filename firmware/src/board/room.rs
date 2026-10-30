use std::time::Duration;

use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::ping::{self, EspPing};
use esp_idf_svc::sys::{esp, esp_wifi_set_ps, wifi_ps_type_t_WIFI_PS_NONE};
use esp_idf_svc::wifi::{
    AuthMethod, BlockingWifi, ClientConfiguration, Configuration, EspWifi, WifiDeviceId,
};
use wujud_firmware::ROOM_PING_HZ;
use wujud_proto::{CsiPacket, MacAddress, RoomBanner};

use super::csi;
use super::serial::send_line;

const WIFI_SSID: &str = env!("WIFI_SSID");
const WIFI_PASS: &str = env!("WIFI_PASS");
// esp_ping treats a count of 0 as "forever".
const PING_FOREVER: u32 = 0;
const PING_THREAD_STACK: usize = 6 * 1024;

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
    wifi.set_configuration(&Configuration::Client(ClientConfiguration {
        ssid: WIFI_SSID
            .try_into()
            .map_err(|_| anyhow::anyhow!("WIFI_SSID too long"))?,
        password: WIFI_PASS
            .try_into()
            .map_err(|_| anyhow::anyhow!("WIFI_PASS too long"))?,
        auth_method: AuthMethod::WPA2Personal,
        ..Default::default()
    }))?;
    wifi.start()?;
    wifi.connect()?;
    wifi.wait_netif_up()?;
    // Power save would make the board miss most router replies.
    esp!(unsafe { esp_wifi_set_ps(wifi_ps_type_t_WIFI_PS_NONE) })?;

    let router = wifi.wifi().driver().get_ap_info()?;
    let gateway = wifi.wifi().sta_netif().get_ip_info()?.subnet.gateway;
    let csi_events = csi::start()?;
    send_line(&RoomBanner {
        channel: router.channel,
        gateway,
        mac: MacAddress(wifi.wifi().get_mac(WifiDeviceId::Sta)?),
    });
    start_pinging(gateway, wifi.wifi().sta_netif().get_index())?;

    let router_mac = MacAddress(router.bssid);
    for event in csi_events {
        if event.sender == router_mac {
            send_line(&CsiPacket {
                rssi: event.rssi,
                rx_timestamp_us: event.rx_timestamp_us,
                csi: event.csi,
            });
        }
    }
    Ok(())
}

/// Every reply from the router is a packet the board measures CSI on.
fn start_pinging(gateway: std::net::Ipv4Addr, interface: u32) -> anyhow::Result<()> {
    let config = ping::Configuration {
        count: PING_FOREVER,
        interval: Duration::from_millis(u64::from(1000 / ROOM_PING_HZ)),
        timeout: Duration::from_secs(1),
        data_size: 1,
        tos: 0,
    };
    std::thread::Builder::new()
        .stack_size(PING_THREAD_STACK)
        .spawn(move || {
            let _ = EspPing::new(interface).ping(gateway, &config);
        })?;
    Ok(())
}
