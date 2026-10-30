# Which physical change breaks the single-link limit of Wi-Fi presence sensing?

A simulation study with the HallSense Rust simulator (`crates/sim`) and the native trainer (`crates/model`).
Driver: `crates/study` (`cargo run --release -p hallsense-study --bin limits_study -- --configs all`),
figures and table: `research/tools/limits_figures.py`. Everything here is simulation. Nothing in this folder
has been measured on hardware.

## Question

On real ESP32 recordings (ESPectre: one link, 2.4 GHz, HT20 L-LTF 52 subcarriers, ~50–100 packets/s) motion is
solved, but a still person vs. an empty room reaches only AUC ≈ 0.55–0.65 per 20-s window
(`../phase_exp.json`, `../sim_calibrate.json`, `../room_sim2real.json`), and breathing is visible in ~11 % of real still
windows (`../breath_visibility.json`). Which change in the physics of the measurement (band, bandwidth, hopping, packet
rate, number of links, placement) raises that ceiling, and at what hardware cost?

## Protocol

* **Realism:** the calibrated, pessimistic knobs of `../sim_calibrate.json` (`rcs_scale 0.1`, `breath_scale 0.5`,
  `evm_shift +6 dB`) in every configuration.
* **Rooms:** random bedrooms (`random_room`), board harshness ~ U(0.8, 1.6). Training rooms are seeds 1 000 000 + i
  (300 rooms), test rooms are seeds 2 000 000 + i (200 rooms), so they never overlap. Each configuration uses the
  same seeds, so the comparison is paired (same rooms, same people, same breathing; the random draws do not depend
  on the band).
* **Per room:** one 60-s session of each activity (away, lie, sit, stand, walk; classes empty / still / motion), and
  for the latch a ~10-min "day" (away, walk in, sit/lie/stand 60–180 s, walk out, away, …,
  `bench/occupancy_sim.py::plan`).
* **Features:** the 17 amplitude window features of `crates/model/src/features` (10 core, RSSI wobble, 6 extra incl.
  the breathing GLRT), 20-s windows every 5 s on the 20 Hz grid. With several links the classifier reads
  `[mean, max, min]` of every feature over the links (as `RoomEngine`); the breathing detector uses the link with the
  largest GLRT.
* **Classifier:** `train_classifier` (300 iterations, learning rate 0.05, 40 samples per leaf, balanced classes), one
  per configuration, trained on the training rooms only.
* **Metrics on test rooms:**
  * still-vs-empty AUC per window with the score `p_still / (p_empty + p_still)`;
  * latch distance AUC: `OccupancyLatch` (robust z-distance of the static profile to the fresh empty reference)
    over the test days, quiet windows with a fresh reference; the latch threshold is chosen for accuracy on
    60 tuning days drawn from the training rooms (`bench/occupancy_sim.py` protocol);
  * breathing visible: share of still windows whose GLRT exceeds the 95th percentile of the empty windows of the same
    configuration (5 % CFAR, `bench/breath_visibility.py`); breathing-rate error: median |estimated − true| bpm of the
    visible windows;
  * motion recall / false alarm (argmax = motion) as a sanity check.
* **Errors:** ± values are standard errors from 300 bootstrap resamples of whole test rooms (days for the latch).

### Anchor check (done first)

The baseline (2.4 GHz ch 6, HT20 L-LTF 52, 50 packets/s, router → 1 ESP32, random placement) with the calibrated
realism gives still-vs-empty AUC **0.643 ± 0.014** and breathing visible in **19.7 ± 1.9 %** of still windows.
Both are inside the real-data ranges (AUC 0.55–0.65; visibility 11 % on all real recordings, 18.5 % on the ESPectre
train split used for calibration). The simulator is therefore anchored for this study, at the top of the real AUC
range. The latch AUC is 0.802 ± 0.018 (`../occupancy_sim.json`, Python, different classifier: 0.828).

## Results

`limits_table.md` (generated), `limits_summary.json`, one `config_<name>.json` per configuration.
Figures: `fig_limits_detection.png` (all metrics, 95 % intervals), `fig_limits_cost.png` (gain vs. cost),
`fig_limits_packet_rate.png`.

| configuration | links | extra cost (USD) | still-vs-empty AUC | latch distance AUC | breathing visible | rate error (bpm) | motion recall / false alarm |
|---|---|---|---|---|---|---|---|
| baseline | 1 | 0 | 0.643 ± 0.014 | 0.802 ± 0.018 | 19.7 ± 1.9 % | 4.28 ± 0.31 | 0.941 / 0.034 |
| 5 GHz conservative | 1 | 5 | 0.670 ± 0.014 | 0.727 ± 0.017 | 20.3 ± 1.7 % | 4.78 ± 0.24 | 0.917 / 0.056 |
| 5 GHz optimistic | 1 | 5 | 0.688 ± 0.013 | 0.742 ± 0.017 | 23.5 ± 1.9 % | 4.45 ± 0.29 | 0.944 / 0.054 |
| HT40, 114 sc | 1 | 0 | 0.738 ± 0.011 | 0.896 ± 0.014 | 34.9 ± 1.9 % | 3.77 ± 0.24 | 0.948 / 0.021 |
| HE20, 242 sc (ESP32-C6) | 1 | 1 | 0.692 ± 0.013 | 0.821 ± 0.017 | 26.8 ± 1.6 % | 3.96 ± 0.23 | 0.943 / 0.029 |
| hop 3 ch. 2.4 GHz | 1 | 5 | 0.641 ± 0.013 | 0.848 ± 0.017 | 22.3 ± 1.7 % | 3.85 ± 0.22 | 0.904 / 0.028 |
| hop 4 ch. 5 GHz | 1 | 15 | 0.649 ± 0.013 | 0.794 ± 0.019 | 21.6 ± 1.8 % | 3.94 ± 0.26 | 0.938 / 0.032 |
| 25 pkt/s | 1 | 0 | 0.597 ± 0.012 | 0.782 ± 0.019 | 14.7 ± 1.3 % | 4.85 ± 0.20 | 0.939 / 0.035 |
| 100 pkt/s | 1 | 0 | 0.692 ± 0.013 | 0.825 ± 0.017 | 25.8 ± 1.7 % | 3.73 ± 0.27 | 0.933 / 0.023 |
| 200 pkt/s | 1 | 0 | 0.742 ± 0.013 | 0.823 ± 0.016 | 31.3 ± 1.8 % | 3.79 ± 0.19 | 0.927 / 0.028 |
| 2 boards | 2 | 5 | 0.748 ± 0.013 | 0.926 ± 0.011 | 31.9 ± 1.7 % | 4.10 ± 0.22 | 0.978 / 0.010 |
| 3 boards | 3 | 10 | 0.777 ± 0.010 | 0.960 ± 0.008 | 39.7 ± 1.8 % | 3.67 ± 0.19 | 0.989 / 0.006 |
| 2 boards + mesh | 3 | 5 | 0.781 ± 0.011 | 0.961 ± 0.008 | 39.5 ± 1.8 % | 3.86 ± 0.18 | 0.987 / 0.004 |
| smart placement | 1 | 0 | 0.664 ± 0.012 | 0.840 ± 0.013 | 16.0 ± 1.4 % | 4.23 ± 0.29 | 0.955 / 0.018 |
| smart + 100 pkt/s | 1 | 0 | 0.712 ± 0.012 | 0.862 ± 0.014 | 26.2 ± 1.6 % | 3.45 ± 0.31 | 0.971 / 0.016 |
| smart + 2-board mesh | 3 | 5 | 0.813 ± 0.010 | 0.978 ± 0.005 | 41.1 ± 2.0 % | 3.37 ± 0.21 | 0.991 / 0.003 |
| 5 GHz + smart | 1 | 5 | 0.751 ± 0.012 | 0.822 ± 0.016 | 21.7 ± 1.7 % | 3.61 ± 0.31 | 0.951 / 0.029 |
| 5 GHz HE20 (ESP32-C5) | 1 | 5 | 0.706 ± 0.014 | 0.750 ± 0.017 | 28.7 ± 1.8 % | 3.84 ± 0.21 | 0.949 / 0.050 |
| 5 GHz HT40 + 100 pkt/s | 1 | 5 | 0.802 ± 0.011 | 0.883 ± 0.014 | 46.0 ± 2.1 % | 3.44 ± 0.19 | 0.923 / 0.038 |
| 2.4 GHz HT40 + 100 pkt/s + smart + mesh | 3 | 5 | 0.965 ± 0.004 | 0.999 ± 0.001 | 82.3 ± 1.3 % | 2.89 ± 0.12 | 0.999 / 0.002 |
| same at 5 GHz, conservative | 3 | 15 | 0.974 ± 0.003 | 0.998 ± 0.001 | 83.9 ± 1.3 % | 2.57 ± 0.11 | 0.997 / 0.001 |
| same at 5 GHz, optimistic | 3 | 15 | 0.985 ± 0.002 | 0.999 ± 0.001 | 89.7 ± 1.0 % | 2.54 ± 0.11 | 0.996 / 0.000 |

### What the numbers say

1. **More independent looks at the room beat a shorter wavelength.** The single changes that move still-vs-empty AUC
   most are more links (+0.10 for one extra board, +0.14 for two boards with the board↔board link, which gives
   3 links for the price of one board) and a wider channel (HT40: +0.10 for free if the router runs 40 MHz). Packet
   rate helps steadily (25 → 200 pkt/s: 0.60 → 0.74), because the 20 Hz grid averages more packets per bin. Extra
   links also raise the latch AUC most (0.80 → 0.96).
2. **5 GHz alone is a small gain** (+0.03 conservative, +0.05 optimistic) and it *lowers* the latch AUC (0.80 → 0.73):
   likely because the doubled phase sensitivity makes the static profile fluctuate more (sway, AGC and EVM noise with
   the conservative radio penalty), widening the empty reference (not separately tested). Breathing visibility barely moves (20 → 20–24 %): the GLRT
   compares the breathing band with the 1–3 Hz band of the same subcarrier, and body sway and hand jitter grow with
   1/λ exactly like breathing does (our reading of the detector; not separately tested). On top of the best 2.4 GHz combination, 5 GHz adds only +0.009 (conservative).
3. **Hopping does not pay at a fixed total packet rate:** each channel gets 50/N packets/s, so the per-channel noise
   rises as fast as the diversity gain; it helps only the latch (0.85 at 2.4 GHz). HE20's 242 subcarriers in the same
   20 MHz add little frequency diversity (+0.05).
4. **Smart placement alone is small (+0.02) but multiplies with everything else.** The factors are strongly
   super-additive: HT40 + 100 pkt/s + smart placement + one extra board in a mesh reaches AUC 0.965 ± 0.004, latch
   0.999, breathing visible in 82 % of still windows with a 2.9 bpm median rate error, for $5 of extra hardware.

**Largest gain per unit hardware cost:** the free changes first (router at HT40, 100–200 packets/s, link across the
bed), then **one extra ESP32 in a mesh** (+0.14 AUC alone, +0.32 together with the free changes, per $5). The band
change to 5 GHz is the worst value: +$5 per board for +0.03 AUC alone and +0.01 in the combination.

## Physics assumptions and sources

The carrier is a parameter of every wavelength-dependent term (`crates/sim/src/band.rs`; tests in
`crates/sim/tests/band_physics.rs`): path spreading `λ/(4πL)` (Friis, −6.6 dB at 5.18 vs 2.437 GHz), path phase
`2πfL/c` per subcarrier, the Fresnel scale `k_F = sqrt(2L/(λ d1 d2))` of body diffraction (zone radius ∝ √λ), the
bistatic scatter amplitude `λ√σ/((4π)^1.5 d1 d2)` and the breathing/fidget phase `2π f_c ΔL / c`. The geometry
(paths, image method, diffuse points), the person and the radios are identical on every band for a given seed.

Subcarrier grids: HT20 L-LTF ±1..26 at 312.5 kHz (52); HT40 HT-LTF ±2..58 at 312.5 kHz (114); HE20 RU-242 ±2..122 at
78.125 kHz (242) (IEEE 802.11-2020 HT, 802.11ax-2021 HE). The per-subcarrier SNR against thermal noise
drops 3 dB at 40 MHz (same Tx power over twice the band) and is unchanged for HE20 (¼ of the power, ¼ of the noise
bandwidth). The receiver's EVM floor (−29…−23 dB + 6 dB calibrated) dominates thermal noise indoors, so this is minor.

| effect | 2.4 GHz (calibrated simulator) | 5 GHz conservative | 5 GHz optimistic | source / reasoning |
|---|---|---|---|---|
| free-space loss | Friis | Friis (automatic) | Friis | – |
| wall/floor/ceiling reflection gain | U(−0.6, −0.25), −0.5, −0.35 | unchanged | unchanged | ITU-R P.2040-3 Table 3: concrete ε′ = 5.24, σ = 0.0462 f^0.7822 S/m gives normal-incidence |Γ| ≈ 0.39 at both 2.4 and 5.2 GHz (σ/ωε0 = 0.69 vs 0.58) Other materials were not checked; thin-slab resonances are not modelled. |
| wall transmission loss | not used (boards in the room) | – | – | ITU-R P.1238 / P.2040: penetration loss grows with frequency; would matter only through walls |
| body box absorption η | U(0.80, 0.95) | unchanged | U(0.90, 0.98) | tissue penetration depth ~2 cm at 2.45 GHz vs ~1 cm at 5.8 GHz (Gabriel et al., Phys. Med. Biol. 41, 1996): the body is more opaque at 5 GHz |
| person absorption cross-section (Sabine) | 0.4 m² | 0.4 m² | 0.55 m² | Flintoft et al., Phys. Med. Biol. 59 (2014): average ACS of 60 adults 0.15–0.4 m² over 1–12 GHz, rising with frequency; Bamba et al., IEEE TEMC (2012) at 2.3 GHz |
| torso RCS | U(0.3, 0.8) m² × 0.1 | unchanged | unchanged | human RCS is roughly flat in the optical region 2–6 GHz; the 0.1 knob is a real-world loss calibrated at 2.4 GHz and assumed to transfer |
| Tx power | U(4, 16) dBm | −3 dB | unchanged | ESP32-C5 5 GHz maximum output is a few dB below its 2.4 GHz output (Espressif datasheet; exact value depends on rate) |
| EVM / estimation noise | U(−29, −23) dB + 6 dB | +2 dB | unchanged | higher carrier: more phase noise; no published ESP32-C5 CSI EVM, so bracketed |
| room absorption α, diffuse K-factor, delay spread | U(0.1,0.3), U(−3,6) dB, 15–40 ns | unchanged | unchanged | measured reverberation at 5 GHz is somewhat shorter (Andersen et al., IEEE AP Mag. 2007); not modelled |

Hardware prices are approximate 2026 retail for dev boards (ESP32 $5, ESP32-C6 $6, ESP32-C5 $10). Hopping needs both
ends to hop, so it needs an ESP32 transmitter instead of the router (+1 board). HT40 is free on the ESP32 but needs
the router to run 40 MHz at 2.4 GHz (often off by default, and against good neighbour practice in dense 2.4 GHz
bands); higher packet rates are free in hardware but cost airtime (not quantified here).

## Adapted features

* **Multiple channels (hopping):** each channel's packets are converted to amplitude and gridded separately; the grids
  are placed side by side as N × 52 columns (aligned to the shorter one, ≤ 1 bin offset). RSSI is centred per
  channel before its wobble is measured, otherwise fading differences between channels would look like motion. At
  N = 3–4 and 50 packets/s each channel is sampled at 12.5–17 packets/s, below the 20 Hz grid, so its noise band
  (4–9.5 Hz) is filled by interpolation; the classifier is retrained per configuration, so this changes the feature
  distribution, not the comparison's fairness, but it is part of why hopping does not help.
* **More subcarriers:** no change needed: the "top" subcarriers are `max(3, K/8)`, the GLRT is normalised by K and
  `signal_subcarriers` drops nulls of any layout. The static profile for the latch grows to K (× channels × links)
  columns; the latch's RMS z-distance is normalised by the column count.
* **Multiple links:** `[mean, max, min]` over links (51 columns), as the room engine does; breathing from the link with
  the largest GLRT, its CFAR threshold set on the same maximum over empty windows.

## What is simulation-only (read before citing)

* Only the 2.4 GHz, 1-link, HT20 baseline is anchored to real data. Every gain above is the simulator's prediction;
  none has been measured on hardware.
* The simulator has no slow environmental drift (temperature, doors, furniture moved, neighbours' Wi-Fi). On real data
  day-to-day drift exceeds a still person's effect; the latch numbers here (fresh reference minutes earlier) avoid
  most of that by design, but the window classifier numbers do not include it.
* The calibration knobs (RCS × 0.1, breathing × 0.5, EVM + 6 dB) were fitted at 2.4 GHz on one link. That they hold at
  5 GHz, at 40 MHz, and for board↔board links is an assumption. In particular, if the "missing" real-world signal at
  2.4 GHz is incoherent scattering that grows with frequency, 5 GHz would be worse than the conservative setting.
* The strong super-additivity of the best combinations (0.81 + HT40 + 100 pkt/s → 0.965) is the most likely result to
  shrink on real hardware: it relies on independent noise across links and subcarriers, and real ESP32 links share
  interference and AGC behaviour.
* Room geometry is a single box with first-order reflections and 24 diffuse paths; HT40's gain depends on how fast
  the simulated channel decorrelates across 40 MHz (delay spread 15–40 ns), which is plausible for bedrooms but not
  validated.
* Body absorption and ACS at 5 GHz are bracketed (conservative/optimistic); the conclusion "5 GHz is a small gain"
  holds at both ends of the bracket.

## Reproduce

```
cargo run --release -p hallsense-study --bin limits_study -- --configs all       # ~1 h on 12 threads
cd research && .venv/bin/python tools/limits_figures.py
```
Existing `config_*.json` are skipped (use `--force`).
