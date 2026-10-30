# wujud-proto

Everything that travels between a wujud ESP32 and the host, as types. `no_std`, no allocation; the `std` feature (default) only adds `to_wire_line`. The firmware in `firmware/` and the host in `crates/app` both use this crate, so the two sides cannot drift apart.

## Serial lines (board → host, one message per `\n` line)

| Line | Type | Sent by | Meaning |
|---|---|---|---|
| `I,wujud-room,ch=<n>,gw=<ip>,mac=<AA:BB:..>` | `Info` → `RoomBanner` | room firmware | boot banner |
| `I,wujud,node=<id>,ch=<n>,mac=<aa:bb:..>` | `Info` → `NodeBanner` | hall node | boot banner |
| `P,<rssi>,<rx timestamp µs>,<256 hex digits>` | `CsiPacket` | room firmware | one packet's raw CSI buffer |
| `N,<node id>,<mac>` | `NodeSeen` | hall sink | first beacon heard from a node |
| `R,<rx>,<src>,<packets>,<13 means dB>,<13 stds dB>` | `LinkReport` | hall sink | one second of one link |

Parse any of them with `Line::parse(text)`; print any of them with `Display` (or `to_wire_line`, which adds the `\n`).

### The CSI buffer
`CsiBuffer` is the 128 bytes of ESP-IDF's `wifi_csi_info_t::buf`, passed through untouched: 64 bins of `[imag, real]` int8, bins 0..31 = subcarriers 0..31, bins 32..63 = subcarriers −32..−1 (FFT order). `CsiBuffer::subcarrier(k)` hides the ordering. The host features use amplitude only, so the byte order inside a pair would not change them; the type still names it correctly.

### Timestamps
`rx timestamp` is the radio's 32-bit microsecond counter and wraps every ~71.6 minutes. `RxClockUnwrapper::seconds` turns it into a continuous time, exactly like `research/live.py`'s room reader.

## ESP-NOW frames (node ↔ node, hall mode)

Packed little-endian structs from `wujud_node.ino`, all starting with magic `'H'`:

| Frame | Layout | Size |
|---|---|---|
| `BeaconFrame` | magic, type=1, node id, seq u16 | 5 B |
| `ReportFrame` | magic, type=2, receiver, record count, seq u16, then up to 5 `ReportRecord` | 6 + 41·n B |
| `ReportRecord` | source, packets, 13 × mean (dB·4, i16), 13 × std (dB·20, u8) | 41 B |

`Frame::decode(bytes)` reads either kind. `ReportRecord::as_link_report(receiver)` gives the `R` line the sink prints for it.

## Mapping to the Python reference

| Rust | Python / firmware |
|---|---|
| `CsiPacket` parsing | `research/live.py::parse_p` (same leniency: hex past 256 digits is ignored) |
| `LinkReport` parsing | `research/live.py::Windower.add_line` |
| `RxClockUnwrapper` | timestamp unwrap in `research/live.py` room reader |
| frames, `ReportRecord` quantization | `research/firmware/wujud_node/wujud_node.ino` (`beacon_t`, `rec_t`, `report_t`, `print_rec`) |

## Tests
- Unit tests for every message, MAC parsing, hex, the clock wrap and frame round-trips.
- `tests/golden.rs` on real data (`research/tools/export_golden_proto.py`): 300 real `P` lines from an ESPectre recording with a forced clock wrap decode exactly as `live.py` does; the hall `R` lines and ESP-NOW report frames emulated from the node firmware on the same real packets decode to the same values, and the records print character-for-character the same lines as the firmware.

```bash
cargo test -p wujud-proto
cargo build -p wujud-proto --no-default-features
```
