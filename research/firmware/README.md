# Firmware (Arduino)

## `wujud_room`: one ESP32, no router

The board opens an open Wi-Fi network called `wujud`. Every laptop or phone that joins it becomes one sensing link, up to 4. The blue LED lights up while at least one device is connected.

- Flash it once over USB with `board.sh flash`. After that, it updates over Wi-Fi.
- Put the board on any phone charger, where the links cross the places people sit, sleep and walk.
- Watch it with:
  ```bash
  cargo run --release -p wujud -- live --ports udp:192.168.4.1 --room research/rec/room.json
  ```
  The app finds the devices by itself; every device's MAC becomes one link.

### Talking to the board: `research/firmware/board.sh`

| command | what it does |
|---|---|
| `board.sh flash` | builds the sketch and sends it, over USB if plugged in, otherwise over Wi-Fi |
| `board.sh status` | uptime, channel, packets/s, and CSI/s per device, as JSON |
| `board.sh set ping_hz=200 channel=6` | changes settings, which are kept across reboots, then the board reboots |
| `board.sh reboot` | restarts the board |

Everything except `status` needs the key in `wujud_room/ota_key.h`:

- `board.sh` creates it the first time.
- git ignores it.
- A board flashed with one key only accepts that key, so keep the file.

### How it works

- The board pings every connected device `ping_hz` times a second.
- Every reply is OFDM, so it carries CSI.
- The board sends each CSI line, tagged with the device's MAC, both over USB and over UDP port 5005. Over UDP it goes to whoever asked in the last 5 s.
- Asking first lets the reply through the laptop's firewall.

### Why not the home router

On the D-Link here, the router answers the ESP32 with 802.11b frames at 2 Mbps. Those frames are DSSS, so they have no subcarriers and no CSI. The classic ESP32 cannot refuse 802.11b.

With the laptop as the other end, each device gives 100–110 CSI packets/s at `ping_hz=100`, and about 185 at `ping_hz=150`. Measured on 2026-10-30.

## `wujud_node`

Firmware for a hall node, meant for many boards. It has not been run on hardware yet.
