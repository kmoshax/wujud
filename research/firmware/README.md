# Firmware (Arduino)

## `wujud_room`: ESP32s, no router

Every board runs the same firmware and chooses its role when it starts:

- **hub**: when no network called `wujud` is on the air, the board opens it. Every laptop, phone or other board that joins becomes one sensing link, up to 8.
- **node**: when `wujud` is already there, the board joins it. The hub measures the node's replies, and the node measures the hub's pings and every frame the other boards and phones send the hub. So three boards give six board-to-board links, not two.

Nothing is set per board:

- Flash every board the same way, once over USB, with `board.sh flash`. After that, `board.sh flash` over Wi-Fi updates the nodes and then the hub.
- Switch them on in any order. If two open `wujud` at once, the one with the higher MAC reboots as a node within 30 s.
- A node that loses the hub for 15 s reboots and chooses again, so pulling the hub's plug makes one of the nodes the new hub.
- The blue LED is on while a board has a link: for the hub, a device joined; for a node, joined to the hub.

Put the boards on chargers around the room, about 1 m up, then start the live view:

```bash
cargo run --release -p wujud -- live --ports udp:192.168.4.1
```

- The app finds every link by itself, from the hub and from each node.
- Without `--room`, it places the hub on the left wall of a 4 x 3.5 m room and the nodes in the far corners. It prints where each board goes, by the end of its MAC. Put them there, or describe the real room in a `--room` file, keyed by MAC.
- With nodes, only the links between boards are used, because a phone's place is unknown. A phone link counts only when the room file gives it a place.

### Talking to the board: `research/firmware/board.sh`

| command | what it does |
|---|---|
| `board.sh flash` | builds the sketch and sends it, over USB if plugged in, otherwise over Wi-Fi |
| `board.sh status` | the hub's uptime, channel, packets/s and CSI/s per device, then each node's, as JSON |
| `board.sh set ping_hz=200 channel=6` | changes settings, which are kept across reboots, then the board reboots |
| `board.sh reboot` | restarts the hub |
| `BOARD=192.168.4.x board.sh ...` | talks to that board instead of the hub; `status` lists the nodes' addresses |

Everything except `status` needs the key in `wujud_room/ota_key.h`:

- `board.sh` creates it the first time.
- git ignores it.
- A board flashed with one key only accepts that key, so keep the file.

### How it works

- The hub pings every connected device `ping_hz` times a second.
- Every reply is OFDM, so it carries CSI.
- The hub sends each CSI line, tagged with the device's MAC, both over USB and over UDP port 5005. Over UDP it goes to whoever asked in the last 5 s.
- Asking first lets the reply through the laptop's firewall.
- A node tags its lines `<node mac>><sender mac or hub>`, batches them, and sends them to the hub on UDP port 5006. The hub passes them on unchanged with its own.
- A node copies the hub's `ping_len` when it joins, so both ends keep the same frames.
- Each board counts time from its own boot. The app anchors each link to the laptop's clock, and anchors it again after a board reboots.

### Why not the home router

On the D-Link here, the router answers the ESP32 with 802.11b frames at 2 Mbps. Those frames are DSSS, so they have no subcarriers and no CSI. The classic ESP32 cannot refuse 802.11b.

With the laptop as the other end, each device gives 100–110 CSI packets/s at `ping_hz=100`, and about 185 at `ping_hz=150`. Measured on 2026-10-30.

## `wujud_node`

Firmware for a hall node, meant for many boards. It has not been run on hardware yet.
