# HallSense

HallSense senses people with Wi-Fi channel state information (CSI) from a $5 ESP32. It tells:

- an empty room;
- a person holding still;
- a person moving.

Every model is trained on physics simulation. None of them has seen a labelled real sample.

- **No router needed.** The ESP32 opens its own network. Every phone or laptop that joins becomes one sensing link, and the board measures the CSI of its replies.
- **Native app.** A Rust and egui 3D view of the room, fed by the board over Wi-Fi or USB, or by the simulator.
- **Measured, not claimed.** Each result has a protocol and a number, including the ones that did not work. See `docs/FINDINGS.md` and `research/results/`.

## Layout

| path | what |
|---|---|
| `crates/proto` | the line protocol between board and host (`no_std`, shared with the firmware) |
| `crates/sim` | physics simulator: rooms, lecture halls, people, walls, ESP32 impairments |
| `crates/model` | features, gradient-boosted models (bit-exact with sklearn), occupancy latch, motion tracker |
| `crates/app` | the `hallsense` program: live view, simulator demos, monitor, recording and replay |
| `crates/study` | limits study: which change in the hardware breaks the single-link limit |
| `research/firmware` | the ESP32 room firmware (Arduino) and `board.sh` to flash and control it over Wi-Fi |
| `research/rt` | GPU ray-traced CSI dataset (Sionna RT) |
| `research/` | the original Python implementation, benchmarks and results |
| `firmware/` | a Rust port of the firmware, not yet run on hardware |
| `docs/` | architecture guide, code conventions, findings |

## Run

You need a Rust toolchain. To see the simulator, no hardware required:

```bash
cargo run --release -p hallsense -- demo --boards 3
```

To run with a real board:

1. Install the Arduino CLI and the ESP32 core.
2. Flash the board once over USB:
   ```bash
   research/firmware/board.sh flash
   ```
3. Join the laptop and a phone to the Wi-Fi network `hallsense`.
4. Start the live view:
   ```bash
   cargo run --release -p hallsense -- live --ports udp:192.168.4.1
   ```

Run the tests, which include golden comparisons against the Python reference:

```bash
cargo test --workspace
```

## What we found

- **Motion is easy and presence is fine.** A person holding still versus an empty room is the hard case. On public ESP32 data (ESPectre), every single-link method we tried reaches an AUC of 0.55–0.65, and so do 31 deep networks.
- **Physical information breaks the limit, not bigger models.** In simulation, a 40 MHz channel, 100+ packets per second and a second link take the AUC from 0.64 to 0.97 (`research/results/limits`).
- **Two-antenna laptop Wi-Fi cards corrupt the CSI.** Their STBC frames make the CSI jump about 4.5 dB from packet to packet at a single-antenna ESP32. Phones measured 0.36 dB. The firmware keeps only non-STBC 802.11n frames.
- **Cheap routers answer the ESP32 in 802.11b.** Those frames have no CSI, which is why the firmware uses phones instead of the router.
- **The public dataset is confounded.** In ESPectre's holdout split, the receiver noise level alone separates still from empty with an AUC of 0.84, because the still-person recordings were made at weak signal.
