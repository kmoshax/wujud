# hallsense-sim

The 3-D Wi-Fi CSI physics simulator of HallSense, ported from the Python research code
(`research/hallsense/{physics3d,scenes3d,sim3d}.py`). It builds random bedrooms and lecture halls,
moves people (and pets, fans, doors) through them, computes the complex channel of every ESP32 link
and turns it into what the board reports: int8 I/Q per subcarrier, RSSI, lost packets.

The math is explained step by step (in Arabic, with figures) in `research/explainer/template.html`,
**Appendix A** (`#math`). Each section below points to the matching subsection.

## Module map

| Module | What it does | Python reference |
|---|---|---|
| `band` | carrier as a parameter: subcarrier layouts (HT20 L-LTF 52, HT-LTF 56, HT40 114, HE20 242), channel hopping (`BandPlan`, packet `t` on channel `t mod N`), frequency-dependent body/radio effects (`FrequencyEffects`) | – (new; study in `research/results/limits/`) |
| `constants` | carrier, wavelength, subcarrier grids (52 L-LTF for halls, 56 HT-LTF for rooms) | `config.py`, `sim3d.py::FREQS52/56` |
| `vec3` | `[f64; 3]` arithmetic | – |
| `random` | numpy-style distributions on a seeded `ChaCha8Rng`, OU processes, `seeded(seed, stream)` | `scenes3d.py::_ou` |
| `paths` | line of sight, image-method wall/floor/ceiling reflections, diffuse paths, path response | `physics3d.py::link_paths3, diffuse_paths, path_response` |
| `fresnel` (+ generated `fresnel_tables`) | Fresnel integrals, strip integral, Fresnel scale/radius | `scipy.special.fresnel` |
| `shadow` | Fresnel-Kirchhoff diffraction of path segments by body boxes | `physics3d.py::segment_shadow` |
| `scatter` | bistatic point scatterers: static clutter and moving chest/limb | `physics3d.py::static_rest, scatter_series` |
| `bodies` | `Body`: per-frame box + scatter points of one moving thing | `physics3d.py::Bodies` |
| `channel` | `H[t, k]` of one link | `physics3d.py::link_channel` |
| `radio` | board radios, ESP32 receive chain (`measure`), Sabine absorption, wall loss | `sim3d.py::measure, _radio, _absorb` |
| `motion` | random walks with pauses, standing/walking body | `scenes3d.py::_standing`, walk loops |
| `room` | random bedroom, activity plans, the person (`room_person`) | `scenes3d.py::random_room, room_person` |
| `distractors` | fan, pet, robot vacuum, door | `scenes3d.py::distractor` |
| `hall` | random flat/tiered hall, occupancy, seated students, lecturer | `scenes3d.py::random_hall3, seated_bodies, lecturer`, `sim.py::sample_occupancy` |
| `capture` | `Capture` (kept packets of one direction), `LinkCapture` | the `out` dicts of `simulate_room` |
| `room_sim` | a room session: walls/flash, distractors, realism knobs; batch `simulate_room` | `sim3d.py::simulate_room` |
| `hall_sim` | hall sessions: a lecture with fixed occupancy, or students arriving over time | `sim3d.py::simulate_hall3, simulate_hall_timeline3` |
| `dataset` | reproducible rayon dataset generation (room and hall) | `bench/room_sim2real.py`, `hallsense/dataset3d.py` |
| `stream` | second-by-second streams for the live demos | `live.py demo-room / demo-hall` |

## Physics summary

* **Channel** (A.1 `#m-csi`): `H[t,k] = absorb·rest[k] + Σ_paths F_p(t)·g_p·λ/(4πL_p)·e^{-j2πf_k L_p/c} + absorb·(chest + limb scatter)`.
* **Reflections** (A.2 `#m-image`): first-order image method on the four walls, the floor (gain −0.5, shadowed at 0.7 strength;
  absent in tiered halls) and the ceiling (−0.35). Wall gains are random per scene.
* **Diffuse field** (A.3 `#m-diffuse`): 24 (room) / 8 (hall) paths `a → random point → b` with an exponential power-delay
  profile (15–40 ns) whose total is `10^{-K/10}` of the line-of-sight power, K ~ U(−3, 6) dB. Because each one has a geometry,
  a person anywhere in the room shadows some of them.
* **Bodies block** (A.4 `#m-fresnel`): every path segment is multiplied per frame by
  `1 − η·(j/2)·I(u1,u2)·I(v1,v2)`, `I(α,β) = ∫_α^β e^{−jπt²/2}dt`, the exact diffraction of a rectangular absorber in
  Fresnel units `k_F = sqrt(2L/(λ d1 d2))`; bodies more than 3.5 Fresnel units away are skipped.
* **Breathing and fidgets** (A.5 `#m-breath`): chest and limb are bistatic point scatterers
  `λ√σ/((4π)^{1.5} d1 d2)`; their motion changes the carrier phase and 1/L amplitude around the track's mean position.
* **People absorb** (A.6 `#m-sabine`): the reverberant clutter `rest` is scaled by `sqrt(A0/(A0 + 0.4 m²·N))`,
  `A0 = α·S`, α ~ U(0.1, 0.3).
* **Through walls** (A.7 `#m-wall`): everything is attenuated by `wall_db` except a static "flash" of six echoes off the
  outside of the wall.
* **ESP32** (A.8 `#m-esp`): Tx power, thermal + EVM noise, interference bursts, RSSI with AGC-state bias and jitter,
  AGC scaling to a target, int8 quantisation (round half to even, clip ±127), packet loss.

## Public API (main entry points, exact signatures)

Scenes and people:

```rust
pub fn random_room(rng: &mut SimRng, options: &RoomOptions) -> Room
pub fn room_person(rng: &mut SimRng, room: &Room, frames: usize, fs: f64, plan: Option<&[PlanStep]>, knobs: &RealismKnobs) -> (Body, PersonLabels)
pub fn distractor(rng: &mut SimRng, kind: DistractorKind, shape: &RoomShape, frames: usize, fs: f64) -> Body
pub fn random_hall(rng: &mut SimRng, options: &HallOptions) -> Hall
pub fn sample_occupancy(rng: &mut SimRng, hall: &Hall, fill_fraction: Option<f64>) -> (Vec<bool>, OccupancyPattern)
pub fn seated_students(rng: &mut SimRng, seats: &[Vec3], frames: usize, fs: f64, restless: f64) -> Vec<Body>
pub fn lecturer(rng: &mut SimRng, hall: &Hall, frames: usize, fs: f64) -> Option<Body>
```

Physics:

```rust
pub fn geometric_paths(a: Vec3, b: Vec3, enclosure: &Enclosure) -> Vec<PropagationPath>
pub fn diffuse_paths(a: Vec3, b: Vec3, draws: &DiffuseDraws) -> Vec<PropagationPath>   // DiffuseDraws::sample(rng, &shape, count)
pub fn static_rest(a: Vec3, b: Vec3, clutter: &[Scatterer], frequencies: &[f64]) -> Vec<Complex64>
pub fn link_channel(link: Link, paths: &[PropagationPath], rest: &[Complex64], bodies: &[Body], frames: usize, absorb: f64) -> Array2<Complex64>
pub fn link_channel_with_references(link: Link, paths: &[PropagationPath], rest: &[Complex64], bodies: &[Body], references: &[ScatterReference], frames: usize, absorb: f64) -> Array2<Complex64>
impl ReceiverNoise { pub fn new(rng: SimRng) -> Self; pub fn draw(&mut self, frames: usize, subcarriers: usize, radio: &LinkRadio, fs: f64) -> (MeasureNoise, Vec<f64>) }
pub fn measure(h: &Array2<Complex64>, radio: &LinkRadio, fs: f64, rng: &mut SimRng) -> Measurement
pub fn measure_with_noise(h: &Array2<Complex64>, radio: &LinkRadio, fs: f64, noise: &MeasureNoise) -> Measurement
pub fn fresnel_integrals(x: f64) -> (f64, f64)            // (S, C), like scipy
pub fn fresnel_zone_radius(zone: u32, from_tx: f64, to_rx: f64) -> f64
```

Sessions (batch or any frame range):

```rust
pub fn simulate_room(room: &Room, options: &RoomSimOptions, rng: &mut SimRng) -> RoomSimulation
pub fn simulate_room_on_band(room: &Room, options: &RoomSimOptions, band: &BandPlan, rng: &mut SimRng) -> RoomSimulation
impl RoomSession {
    pub fn new(room: &Room, options: &RoomSimOptions, rng: &mut SimRng) -> Self              // default band
    pub fn with_band(room: &Room, options: &RoomSimOptions, band: &BandPlan, rng: &mut SimRng) -> Self
    pub fn channels(&self, frames: Range<usize>) -> Vec<Array2<Complex64>>        // noiseless H per link
    pub fn capture(&mut self, frames: Range<usize>) -> Vec<Capture>                 // one per room.links
    pub fn calibrate(&mut self) -> Vec<Capture>
}
impl HallSession {
    pub fn lecture(hall: &Hall, occupied: &[bool], options: &HallSimOptions, rng: &mut SimRng) -> Self   // simulate_hall3
    pub fn arrivals(hall: &Hall, options: &ArrivalOptions, rng: &mut SimRng) -> Self                    // simulate_hall_timeline3
    pub fn channels(&self, frames: Range<usize>) -> Vec<Array2<Complex64>>        // noiseless H per pair
    pub fn capture(&mut self, frames: Range<usize>) -> Vec<Capture>                 // both directions of every pair
    pub fn calibrate(&mut self) -> Vec<Capture>
    pub fn simulate(&mut self) -> HallSimulation
    pub fn directions(&self) -> Vec<(usize, usize)>
    pub fn present_at(&self, frame: usize) -> Vec<bool>
}
```

Sessions are continuous: `capture` is meant to be called with consecutive frame ranges, and any split
(e.g. one call per second when streaming) gives exactly the packets of one call over the whole session. Body scatter is
evaluated against the whole session's mean positions (`ScatterReference`, `link_channel_with_references`), and every
receiving direction owns a `ReceiverNoise` (its own ChaCha stream, drawn frame by frame, with the RSSI gain-estimate bias
carried over between calls). `tests/streaming_continuity.rs` checks streamed == batch bit for bit (room and hall),
channel slices == whole-session channel (≤ 1e-12), and that per-subcarrier amplitude and RSSI changes across second
boundaries are no larger than inside a second.

`Capture { frame: Vec<u32>, t: Vec<f64>, iq: Array3<i8> /* [packets, K, 2] */, rssi: Vec<i16> }` holds only the packets
that were not lost; `frame` indexes the per-frame labels (`PersonLabels { activity, xy, breath_hz, plan }`).
`LinkCapture { tx, rx, calibration, session }`.

Dataset generation (rayon; each scene is a pure function of its seed, so output does not depend on the thread count):

```rust
pub fn room_scene(config: &RoomDatasetConfig, seed: u64) -> RoomScene
pub fn room_dataset(config: &RoomDatasetConfig, seeds: Range<u64>) -> Vec<RoomScene>
pub fn for_each_room_scene(config: &RoomDatasetConfig, seeds: Range<u64>, sink: impl Fn(RoomScene) + Sync + Send)
pub fn dataset_hall(config: &HallDatasetConfig, hall_seed: u64) -> Hall
pub fn hall_scene(config: &HallDatasetConfig, hall_seed: u64, scene: usize) -> HallScene
pub fn hall_dataset(config: &HallDatasetConfig, hall_seeds: Range<u64>) -> Vec<HallScene>
pub fn for_each_hall_scene(config: &HallDatasetConfig, hall_seeds: Range<u64>, sink: impl Fn(HallScene) + Sync + Send)
```

Live-demo streams (`Iterator`, one item per simulated second):

```rust
impl RoomStream {   // Item = RoomSecond { second, links: Vec<Capture>, truth: RoomTruth { activity, breath_bpm, xy } }
    pub fn new(room: &Room, options: &RoomSimOptions, seed: u64) -> Self
    pub fn demo(seed: u64, boards: usize) -> Self   // live.py demo-room (plan: demo_room_plan(), fs 50)
    pub fn calibration(&mut self) -> Vec<Capture>
    pub fn seconds(&self) -> usize
}
impl HallStream {   // Item = HallSecond { second, links: Vec<Capture>, present: Vec<bool> }
    pub fn new(hall: &Hall, options: &ArrivalOptions, seed: u64) -> Self
    pub fn demo(seed: u64) -> Self                  // live.py demo-hall: 13 × 12 m tiered, 12 boards, 4 min, 75 % full
    pub fn directions(&self) -> Vec<(usize, usize)>
    pub fn calibration(&mut self) -> Vec<Capture>
    pub fn seconds(&self) -> usize
}
```

Examples: `cargo run -p hallsense-sim --release --example stream_demo -- room|hall [seconds]`,
`cargo run -p hallsense-sim --release --example bench_room [rooms]`.

## Parity with the Python reference

### Exact (golden) tests

`research/tools/export_golden_sim.py` writes `tests/golden/{physics.json, room_channel.npz, hall_channel.npz, measure_*.npz}`
(~3 MB). Random numbers are recorded from numpy (a duck-typed `RecordingRng`) and injected, so the deterministic code is
compared value by value:

| Test | What | Tolerance | Measured |
|---|---|---|---|
| `golden_physics` | Fresnel S, C vs scipy (−6…6, 10, 50, 300) | 1e-14 abs | passes |
| | image-method points, gains, path response; diffuse paths from injected draws; `static_rest` | 1e-12 rel | passes |
| | Fresnel scale `k_F`, first-zone radius `= √2/k_F` | 1e-12 | passes |
| | int8 quantisation (half to even, clipping) | exact | exact |
| `golden_channel` | segment shadow, every segment of every path of 5 hall links (40 students + lecturer) and 3 room links (person + fan + door) | 1e-12 abs | passes |
| | chest/limb scatter series; full `link_channel` (absorb 1, 0.93, 0.81) | 1e-6 rel to max |H| (Python is complex64) | 1.6e-7 / 4.8e-7 |
| `golden_measure` | `measure()` with numpy's draws injected (typical, and harsh with 179 interference bursts and AGC steps) | ≤ 1 step in ≤ 1e-4 of values | identical iq, rssi, kept |

### Statistical parity of random scenes

Rust uses `ChaCha8Rng`, not numpy's PCG64, so random scenes can only match in distribution.
`research/tools/export_parity_sim.py --rooms 100` writes `tests/golden/parity_room.json`; `tests/parity.rs` runs the same
recipe in Rust (random room, smart placement 30 %, fs ~ U(50, 120), harsh ~ U(0.8, 1.6), 60 s of one activity, router →
board link) over 100 rooms and compares the per-room statistics per activity. Tolerances: two-sample Kolmogorov-Smirnov
distance ≤ 0.276 (critical value at α = 0.001 for 100 vs 100) and |median difference| ≤ 0.35 × pooled IQR.
It is `#[ignore]`d because it is slow in debug builds; run it with
`cargo test -p hallsense-sim --release --test parity -- --ignored --nocapture` (≈ 5 s).

| activity | statistic | Python median | Rust median | KS | shift / IQR |
|---|---|---|---|---|---|
| away | amplitude level (int8 units) | 17.16 | 18.15 | 0.12 | 0.15 |
| away | temporal std per subcarrier | 0.0631 | 0.0635 | 0.10 | 0.03 |
| away | RSSI median (dBm) | −37.0 | −36.0 | 0.10 | 0.11 |
| away | RSSI std (dB) | 0.742 | 0.716 | 0.18 | 0.15 |
| away | kept fraction | 0.948 | 0.954 | 0.10 | 0.15 |
| lie | amplitude level | 17.73 | 18.28 | 0.08 | 0.09 |
| lie | temporal std | 0.0633 | 0.0645 | 0.15 | 0.08 |
| lie | RSSI median | −37.5 | −37.0 | 0.08 | 0.05 |
| lie | RSSI std | 0.762 | 0.754 | 0.07 | 0.04 |
| sit | amplitude level | 19.23 | 17.80 | 0.15 | 0.23 |
| sit | temporal std | 0.0630 | 0.0659 | 0.13 | 0.19 |
| sit | RSSI median | −38.0 | −37.0 | 0.10 | 0.10 |
| sit | RSSI std | 0.762 | 0.795 | 0.15 | 0.17 |
| stand | amplitude level | 18.59 | 17.20 | 0.15 | 0.23 |
| stand | temporal std | 0.0644 | 0.0658 | 0.13 | 0.10 |
| stand | RSSI std | 0.761 | 0.765 | 0.14 | 0.02 |
| walk | amplitude level | 18.05 | 18.25 | 0.06 | 0.04 |
| walk | temporal std | 0.0793 | 0.0784 | 0.11 | 0.02 |
| walk | RSSI median | −38.5 | −37.0 | 0.11 | 0.17 |
| walk | RSSI std | 1.873 | 1.726 | 0.09 | 0.11 |

(kept fraction for lie/sit/stand/walk and RSSI median for stand: KS ≤ 0.20, shift ≤ 0.10.)
Largest KS 0.20, largest median shift 0.23 IQR. With only 30 rooms per side `away / rssi_std` once reached KS 0.47; 200 rooms
per side gave medians 0.711 (Python) vs 0.717 (Rust), so that was sampling noise.

## Speed

Seconds of CPU per simulated room-minute (the parity recipe: 60 s + 20 s calibration, fs ~ U(50,120), one link),
12-thread machine:

| | single thread | 12 workers |
|---|---|---|
| Python (`export_parity_sim.py`, multiprocessing Pool) | 0.118 | 0.031 |
| Rust (`examples/bench_room.rs`, rayon) | 0.031 | 0.006 |

The live streams run far faster than real time in release builds: the room demo (3 links, 50 packets/s) takes ~1.4 ms per
simulated second, the hall demo (132 directions, 20 packets/s) 5 ms per second while empty and ~150 ms with 122 students.

Where the speed comes from (all exact, the golden tests are unchanged): `link_channel` sums the unblocked paths once and adds
`(F−1)·h` only at frames a body actually changes; `ShadowCasters` does the per-frame trigonometry once per link and skips a
body's whole track for a segment when a provable lower bound on its distance already exceeds the 3.5-Fresnel-unit prune;
the Fresnel integrals use the power series below 1.5, degree-14 Chebyshev fits of the auxiliary functions f, g on
[1.5, 6] (`research/tools/fit_fresnel_aux.py` → `src/fresnel_tables.rs`, ≤ 1e-15 from scipy) and the asymptotic series above.

## Carrier as a parameter

Every wavelength-dependent term takes the carrier of the subcarrier grid it is evaluated on (`band::wavelength_of`,
exact for the symmetric grids): path spreading, the Fresnel scale of body diffraction (`fresnel_scale_at`,
`ShadowCasters::with_wavelength`), bistatic scatter amplitude and the breathing/fidget carrier phase, the wall flash.
Path geometry and gains are frequency-independent, so one set of paths serves every channel of a hopping plan.
`RoomSession::with_band` maps the body absorption, the person's Sabine cross-section and the radios' Tx/EVM through
`FrequencyEffects`; the default `BandPlan` (2.437 GHz, HT-LTF 56, neutral effects) reproduces the simulator bit for bit
(`tests/band_physics.rs`, which also checks grids, √λ Fresnel radius, Friis loss, carrier-scaled breathing phase and
the per-frame channel of a hopping link). The physics choices for 5 GHz are documented in `research/results/limits/README.md`.

## Deliberate deviations from the Python

* Random streams: `ChaCha8Rng` instead of PCG64, and some draws happen in a different order (e.g. seated students are
  drawn one by one, numpy draws all at once; hall placement uses a coin flip instead of `rng.choice`). Scenes match in
  distribution, not value by value.
* Precision: the channel is computed in `f64`; Python accumulates `H` and the scatter series in `complex64` (≤ 5e-7 relative).
* Hall sessions return raw packets (both directions, calibration and session), not 1-s window features: the features live
  in `crates/model`.
* Receiver noise is drawn frame by frame from one stream per direction (numpy draws each kind of noise for the whole
  session at once), so that streaming and batch runs agree exactly; the distributions are unchanged.
* Calibration length 0 works (numpy's `steps.max()` fails on an empty array).
* Dataset seeding: `random::seeded(seed, stream)` (ChaCha stream) plays the role of `default_rng([seed, k])`.
