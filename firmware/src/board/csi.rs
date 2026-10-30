use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::OnceLock;

use esp_idf_svc::sys::{
    esp, esp_wifi_set_csi, esp_wifi_set_csi_config, esp_wifi_set_csi_rx_cb, wifi_csi_config_t,
    wifi_csi_info_t, EspError,
};
use wujud_proto::{CsiBuffer, MacAddress, CSI_BUFFER_LEN};

#[cfg(esp32c6)]
compile_error!(
    "ESP32-C6 uses wifi_csi_acquire_config_t; see firmware/README.md (not written yet)."
);

pub struct CsiEvent {
    pub sender: MacAddress,
    pub rssi: i8,
    pub rx_timestamp_us: u32,
    pub first_word_invalid: bool,
    pub csi: CsiBuffer,
}

const QUEUE_DEPTH: usize = 64;

static EVENTS: OnceLock<SyncSender<CsiEvent>> = OnceLock::new();

/// L-LTF only, raw: the same CSI the Arduino firmware and the ESPectre recordings use.
pub fn start() -> Result<Receiver<CsiEvent>, EspError> {
    let (sender, receiver) = sync_channel(QUEUE_DEPTH);
    let _ = EVENTS.set(sender);
    let config = wifi_csi_config_t {
        lltf_en: true,
        htltf_en: false,
        stbc_htltf2_en: false,
        ltf_merge_en: false,
        channel_filter_en: false,
        manu_scale: false,
        ..Default::default()
    };
    esp!(unsafe { esp_wifi_set_csi_config(&config) })?;
    esp!(unsafe { esp_wifi_set_csi_rx_cb(Some(on_csi), core::ptr::null_mut()) })?;
    esp!(unsafe { esp_wifi_set_csi(true) })?;
    Ok(receiver)
}

// Runs inside the Wi-Fi task: copy and hand off, never block.
unsafe extern "C" fn on_csi(_context: *mut core::ffi::c_void, info: *mut wifi_csi_info_t) {
    let Some(info) = info.as_ref() else { return };
    if info.buf.is_null() || usize::from(info.len) < CSI_BUFFER_LEN {
        return;
    }
    let bytes = core::slice::from_raw_parts(info.buf, CSI_BUFFER_LEN);
    let Some(csi) = CsiBuffer::from_driver_buffer(bytes) else {
        return;
    };
    let event = CsiEvent {
        sender: MacAddress(info.mac),
        rssi: info.rx_ctrl.rssi() as i8,
        rx_timestamp_us: info.rx_ctrl.timestamp(),
        first_word_invalid: info.first_word_invalid,
        csi,
    };
    if let Some(events) = EVENTS.get() {
        let _ = events.try_send(event);
    }
}
