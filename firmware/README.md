# hallsense-firmware

Rust firmware for the ESP32 boards, a port of `research/firmware/*.ino`. Two modes from one codebase:

| Feature | What the board does | Serial output |
|---|---|---|
| `room` (default) | joins your Wi-Fi, pings the router 50×/s, sends the CSI of every router reply | `I,hallsense-room,...` then `P,...` lines |
| `hall-node` | no router: broadcasts ESP-NOW beacons 20×/s on channel 6, measures CSI of every other node, sends one report per source per second | `I,hallsense,...`, `N,...`, `R,...` lines |

The wire format is `crates/proto`; the host side is `crates/app`.

## Layout
| Path | Builds on | Tested |
|---|---|---|
| `src/csi_levels.rs` | anywhere | yes: unit tests + real CSI |
| `src/link_window.rs` | anywhere | yes: unit tests + real CSI |
| `src/node_table.rs`, `src/beacon_clock.rs` | anywhere | yes: unit tests |
| `src/board/*` | ESP32 only (`target_os = "espidf"`) | **no** (see below) |

`tests/hall_golden.rs` runs the board's report logic on a real ESP32 recording (ESPectre `motion_c3_…`) and compares with the Python emulation of `hallsense_node.ino`: **2329 of 2340 quantized values identical, the other 11 one step apart** (the board sums in f32, the emulation in f64).

## Why esp-idf-svc
CSI needs `esp_wifi_set_csi_config` / `esp_wifi_set_csi_rx_cb` / `esp_wifi_set_csi`. With `esp-idf-svc` 0.53 (on `esp-idf-sys` 0.38) these come straight from the ESP-IDF bindings, next to safe wrappers for Wi-Fi (`EspWifi`, `BlockingWifi`), ESP-NOW (`EspNow::take`, `add_peer`, `register_recv_cb`, `send`) and ping (`EspPing::new(..).ping(..)`), all checked against the 0.53.0 source. It is the same ESP-IDF the Arduino firmware runs on, so the CSI is the same. The `no_std` stack (`esp-hal` + `esp-radio`) was not chosen because the room mode needs ICMP ping through lwIP, which comes for free with ESP-IDF.

## Build and flash (not done yet on this machine)
Needs the Espressif toolchain, which is not installed here:
```bash
cargo install espup espflash ldproxy
espup install
```
Then, from `firmware/`, for an ESP32-C3:
```bash
WIFI_SSID="your-wifi" WIFI_PASS="your-password" cargo +nightly build --release --target riscv32imc-esp-espidf -Zbuild-std=std,panic_abort
```
```bash
espflash flash --monitor --baud 921600 target/riscv32imc-esp-espidf/release/hallsense-firmware
```
Hall node (one id per board, 0–31):
```bash
HALLSENSE_NODE_ID=3 cargo +nightly build --release --no-default-features --features hall-node --target riscv32imc-esp-espidf -Zbuild-std=std,panic_abort
```
Other chips: `xtensa-esp32-espidf` (ESP32), `xtensa-esp32s3-espidf` (ESP32-S3), using the `esp` toolchain from `espup` instead of `nightly`.

## Not verified yet
- `src/board/*` has never been compiled for an ESP32 (no toolchain here) nor run on hardware. The API names were checked against the esp-idf-svc 0.53.0 source; the raw ESP-IDF names (`wifi_csi_config_t` fields, `rx_ctrl.rssi()` / `timestamp()` bitfield accessors, `wifi_ps_type_t_WIFI_PS_NONE`, …) follow bindgen's naming of the ESP-IDF 5.3 headers and may need small fixes on the first build.
- ESP32-C6 uses a different CSI config struct (`wifi_csi_acquire_config_t`); building for it stops with a clear error until that path is written.
- `esp_ping` with count 0 runs forever (ESP-IDF's `ESP_PING_COUNT_INFINITE`); the first hardware run should confirm ~50 `P` lines per second.

First check on hardware: the `I,` banner appears, then `P` lines at ~50 Hz (room) or `R` lines once a second per neighbour (hall). Then feed them to `crates/app` and compare with `research/tools/replay_espectre.py`.
