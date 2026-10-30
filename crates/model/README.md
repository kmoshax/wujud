# hallsense-model

Pure logic for HallSense: CSI/RSSI window features, gradient-boosted trees (inference of the exported sklearn
models and native training), the occupancy latch, and the room, hall and laptop-beacon engines that produce the
dashboard state every second. No I/O apart from reading model, config and dataset files.

The Python code in `research/` is the reference. Every module below names the Python function it ports, and the
tests in `tests/` compare the two on real recordings (see "Measured parity").

## Module map

| Module | What it does | Python reference |
|---|---|---|
| `features` | Per-packet amplitude and sanitized phase from I/Q, 20 Hz grid, 20-s window features (10 core + RSSI wobble + 6 "what moved" + breathing rate), static profile, RSSI-only features | `hallsense/roomfeat.py`, `hallsense/rssifeat.py` |
| `spectrum`, `linalg`, `numeric` | NumPy-equivalent FFT, Hann window, PCA, statistics (including NumPy's float32 pairwise sums and median rules) | NumPy |
| `gbdt` | `Gbdt` inference of exported `HistGradientBoosting*` models, `SeatModelTrees`, model JSON reader/writer, native trainer | sklearn `HistGradientBoostingClassifier/Regressor`, `tools/export_models.py` |
| `occupancy` | `OccupancyLatch`: motion latches "someone is here", a quiet room is compared with a fresh empty reference | `hallsense/occupancy.py` |
| `room` | `RoomEngine`: router -> board CSI links in, one `RoomState` per second out (activity model, smoothing, breathing override, latch, pose, breathing rate) | `live.py::RoomEngine` |
| `roomloc` | Physics-only position from which links light up (off by default) | `hallsense/roomloc.py` |
| `scene` | The `scene` object the dashboard draws | `live.py::room_scene_json`, `hall_scene_json` |
| `hall` | `HallLayout` (hall config), `Windower` (mesh report lines -> 1-s windows), link features, seat descriptors, `SeatModel`, `HallEngine` | `live.py::load_hall`, `Windower`, `HallEngine`; `hallsense/model.py` |
| `beacon` | `BeaconEngine`: laptop monitor-mode beacon RSSI -> motion, self-calibrating quiet/moving, or spot classification after a calibration walk | `apsense.py::Engine` |
| `espectre` | Reader for the ESPectre real ESP32 dataset (`dataset_info.json` + compressed `.npz`) | `bench/occupancy_engine_replay.py::stream`, `bench/phase_exp.py::_real` |
| `metrics` | ROC AUC (ties count half) and the room benchmark metrics | sklearn `roc_auc_score`, `bench/phase_exp.py::metrics` |

## Public API

All fallible functions return `hallsense_model::Result<T>` with `ModelError` (`Io`, `Json`, `Npz`,
`InvalidModel`, `FeatureCount`, `InvalidTrainingData`).

### features
```rust
pub fn amplitude_from_iq(iq: ArrayView2<i8>) -> Array2<f32>          // iq: [packets, 2 * subcarriers], interleaved
pub fn phase_from_iq(iq: ArrayView2<i8>) -> Array2<f64>
pub fn signal_subcarriers(magnitudes: &Array2<f32>) -> Vec<usize>
pub fn to_grid(times: &[f64], values: ArrayView2<f64>, rate_hz: f64) -> Array2<f64>
pub fn windows(times: &[f64], amplitude: ArrayView2<f64>, rssi: Option<&[f64]>,
               window_seconds: f64, hop_seconds: f64) -> WindowSet   // WindowSet { rows, breathing_rates }
pub fn phase_windows(times: &[f64], phase: ArrayView2<f64>, window_seconds: f64, hop_seconds: f64) -> Vec<Vec<f64>>
pub fn profile_windows(times: &[f64], amplitude: ArrayView2<f64>, window_seconds: f64, hop_seconds: f64) -> Vec<Vec<f64>>
pub fn window_features(linear_amplitude: ArrayView2<f64>, rate_hz: f64) -> (Vec<f64>, BreathingRate)
pub fn spectral_features(series: ArrayView2<f64>, rate_hz: f64) -> (Vec<f64>, BreathingRate)
pub fn extra_features(linear_amplitude: ArrayView2<f64>, rate_hz: f64) -> Vec<f64>
pub fn rssi_window(grid: ArrayView2<f64>, rate_hz: f64) -> (Vec<f64>, BreathingRate)
pub fn rssi_windows(times: &[f64], rssi: ArrayView2<f64>, window_seconds: f64, hop_seconds: f64)
    -> (Vec<Vec<f64>>, Vec<BreathingRate>)
// constants: GRID_RATE_HZ, WINDOW_SECONDS, HOP_SECONDS, BREATH_BAND_HZ, MICRO_BAND_HZ, NOISE_BAND_HZ,
// CORE_FEATURE_NAMES, RSSI_COLUMN_NAME, EXTRA_FEATURE_NAMES, PHASE_FEATURE_NAMES, RSSI_FEATURE_NAMES, GLRT_COLUMN_WITH_RSSI
```

### gbdt
```rust
pub enum Loss { Multinomial, Binomial, SquaredError }
pub struct Tree { feature, threshold, missing_goes_left, left, right, leaf_value }
impl Tree { pub fn single_leaf(value: f64) -> Self; pub fn predict(&self, x: &[f64]) -> f64 }
pub struct Gbdt { pub loss: Loss, pub n_features: usize, pub classes: Vec<i64>, pub baseline: Vec<f64>, pub iterations: Vec<Vec<Tree>> }
impl Gbdt {
    pub fn load(path: &Path) -> Result<Self>
    pub fn from_json(text: &str) -> Result<Self>
    pub fn to_json(&self) -> String
    pub fn trees_per_iteration(&self) -> usize
    pub fn raw_scores(&self, x: &[f64]) -> Result<Vec<f64>>
    pub fn predict_proba(&self, x: &[f64]) -> Result<Vec<f64>>       // a regressor returns [value]
    pub fn predict_proba_rows(&self, rows: ArrayView2<f64>) -> Result<Array2<f64>>
    pub fn predict_value(&self, x: &[f64]) -> Result<f64>
    pub fn predict_class(&self, x: &[f64]) -> Result<i64>
}
pub fn softmax(raw: &[f64]) -> Vec<f64>
pub fn sigmoid(raw: f64) -> f64
pub struct SeatModelTrees { pub stage1: Gbdt, pub stage2: Gbdt, pub count: Gbdt }
impl SeatModelTrees { pub fn load(path: &Path) -> Result<Self> }
pub struct TrainOptions { max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization,
                          balanced_class_weight, early_stopping, validation_fraction, n_iter_no_change, tol, seed }
pub enum EarlyStopping { Auto, On, Off }
pub fn train_classifier(x: ArrayView2<f64>, y: &[i64], options: &TrainOptions) -> Result<Gbdt>
pub fn train_regressor(x: ArrayView2<f64>, y: &[f64], options: &TrainOptions) -> Result<Gbdt>
// constants: FORMAT = "hallsense-gbdt-v2", SEAT_MODEL_FORMAT = "hallsense-seat-model-v1"
```
`TrainOptions::default()` has sklearn's defaults (100 iterations, learning rate 0.1, 31 leaves, 20 samples per
leaf, early stopping above 10 000 samples, 10 % validation, 10 rounds without change, tol 1e-7).

### occupancy
```rust
pub enum Occupancy { Unknown, Empty, Moving, Still }                // serialized lowercase
pub struct LatchConfig { columns, threshold, reference_len, min_reference, clear_after,
                         reference_max_age, sd_floor, no_entry_factor }   // Default = Python defaults (thr 3.0)
impl OccupancyLatch {
    pub fn new(config: LatchConfig) -> Self
    pub fn step(&mut self, x: &[f64], p_motion: f64, p_still: Option<f64>) -> (Occupancy, Option<f64>)
    pub fn distance(&self, x: &[f64]) -> f64
    pub fn clear_reference(&mut self)
    pub fn state(&self) -> Occupancy
    pub fn reference_len(&self) -> usize
}
```

### room
```rust
pub fn activity_model_file(links: usize) -> &'static str          // "room_activity_esp{1,3}_random.json"
impl RoomEngine {
    pub fn new(scene: Scene, links: Vec<(usize, usize)>, mode: &str, model: Gbdt, nodes: Vec<[f64; 3]>) -> Self
    pub fn add(&mut self, link: usize, t_seconds: f64, iq: &[i8], rssi: f64)
    pub fn tick(&mut self, now: f64) -> Result<RoomState>
    pub fn tick_with_truth(&mut self, now: f64, truth: Option<RoomTruth>) -> Result<RoomState>
    pub locate_person: bool                                         // off, like live.py
}
pub struct RoomState { mode, t, scene, room: RoomReport, link_activity, alive, history: Vec<(f64, Option<f64>, Option<f64>)> }
pub struct RoomReport { decision: Option<RoomDecision>, truth: Option<RoomTruth> }   // both flattened; {} when empty
pub struct RoomDecision { state: Pose, state3: RoomState3, probs, latch: LatchReport, breath_bpm, xy }
pub fn locate(link_features: &[Vec<f64>], links: &[(usize, usize)], nodes: &[[f64; 3]], width: f64, depth: f64) -> [f64; 2]
impl Scene { pub fn room(width: f64, depth: f64, height: f64, nodes: &[[f64; 3]],
                         furniture: Vec<serde_json::Value>, roles: Vec<String>) -> Self }
```
`serde_json::to_value(&RoomState)` is the JSON `live.py::publish()` sends (fields read by
`research/realtime/dashboard3d.html`: `room.state`, `state3`, `breath_bpm`, `xy`, `truth*`, `link_activity`, `alive`, `history`).

### hall
```rust
impl HallLayout {
    pub fn load(path: &Path) -> Result<Self>                        // hall.json with "seats" or "grid"
    pub fn from_config_json(text: &str) -> Result<Self>
    pub fn chest_points(&self) -> Vec<[f64; 3]>
    pub fn scene(&self) -> Scene
}
impl Windower { pub fn new(hall: &HallLayout) -> Self; pub fn add_line(&mut self, line: &str);
                pub fn tick(&mut self) -> (Array2<f32>, Array2<f32>) }
pub fn link_features(cal_mean: ArrayView3<f32>, cal_std: ArrayView3<f32>, ses_mean: ArrayView3<f32>, ses_std: ArrayView3<f32>) -> LinkFeatures
pub fn excess_path(a: [f64; 3], b: [f64; 3], p: [f64; 3]) -> f64
pub fn distance_bins(nodes: &[[f64; 3]], links: &[(usize, usize)], chests: &[[f64; 3]]) -> Array2<Option<usize>>
pub fn seat_descriptor(features: &LinkFeatures, bins: &Array2<Option<usize>>, nodes: &[[f64; 3]], chests: &[[f64; 3]]) -> Array2<f32>
pub fn context_features(p1: &[f64], bins: &Array2<Option<usize>>, chests: &[[f64; 3]]) -> Array2<f32>
pub fn scene_vector(features: &LinkFeatures, nodes: &[[f64; 3]], links: &[(usize, usize)], seats: usize, width: f64, depth: f64) -> Vec<f32>
impl SeatModel { pub fn load(path: &Path) -> Result<Self>;
                 pub fn predict(&self, evidence: &SceneEvidence, bins: &Array2<Option<usize>>, chests: &[[f64; 3]]) -> Result<SeatPrediction> }
impl HallEngine {
    pub fn new(layout: HallLayout, model: SeatModel, calibration_mean: Array3<f32>, calibration_std: Array3<f32>, mode: &str) -> Self
    pub fn push(&mut self, t_seconds: f64, mean: Array2<f32>, std: Array2<f32>, truth: Option<&[i64]>) -> Result<HallState>
    pub fn evidence(&self) -> Option<SceneEvidence>
}
```

### beacon
```rust
pub struct BeaconRoom { width, depth, height, laptop, aps: Vec<AccessPoint>, furniture }   // research/rec/myroom.json
impl BeaconEngine {
    pub fn new(room: BeaconRoom, calibration: Calibration) -> Self
    pub fn add(&mut self, t: f64, bssid: &str, chains: &[i64])
    pub fn start_calibration(&mut self, now: f64, label: &str, xy: Option<[f64; 2]>, seconds: f64, delay: f64)
    pub fn tick(&mut self, now: f64) -> BeaconState
    pub fn diagnostics(&self) -> &BeaconDiagnostics                 // Engine.last
    pub fn calibration(&self) -> &Calibration                       // serialize to save cal.json
    pub fn spots(&self) -> Vec<SpotSummary>
    pub fn delete_spot(&mut self, label: &str)
}
```

### espectre, metrics
```rust
impl EspectreDataset {
    pub fn open(root: &Path) -> Result<Self>
    pub fn find(&self, label: &str, collected_at_prefix: &str, environment: Option<&str>) -> Option<&RecordingInfo>
    pub fn load(&self, label: &str, info: &RecordingInfo) -> Result<Recording>
}
pub fn load_recording(path: &Path) -> Result<Recording>            // Recording { times, iq, rssi }, times from 0 s
impl Recording { pub fn first_seconds(&self, seconds: f64) -> Recording }
pub fn room_class(label: &str) -> Option<i64>                       // empty 0, static_presence 1, motion 2
pub fn roc_auc(positive: &[bool], scores: &[f64]) -> f64
pub fn room_metrics(model: &Gbdt, recordings: &[LabelledWindows], columns: &[usize]) -> Result<RoomMetrics>
```

## Model JSON format

`research/tools/export_models.py` writes every `research/models/*.joblib` to `models/*.json`:

```json
{"format": "hallsense-gbdt-v2", "source": "models/room_sim_all.joblib",
 "loss": "multinomial" | "binomial" | "squared_error",
 "n_features": 17, "classes": [0, 1, 2] | null, "baseline": [b0, b1, b2], "trees_per_iteration": 3,
 "iterations": [[tree, tree, tree], ...]}
```
A tree keeps its split nodes and its leaves in separate arrays:
```json
{"feature": [16, 13, ...], "threshold": [0.7614890336990356, ..., null], "missing_left": [1, 0, ...],
 "left": [1, -1, ...], "right": [2, -2, ...], "leaf_value": [0.013, ...]}
```
Split node 0 is the root (a tree with no split nodes is its single leaf). A child `c >= 0` is split node `c`,
`c < 0` is leaf `-1 - c`. A sample goes left when `x <= threshold`, or when `x` is NaN and `missing_left` is 1.
`null` is a threshold of `+inf` (sklearn's "split on NaN": every value left, missing values right), because
JSON has no infinity. Numbers are written with Python's shortest round-trip repr and read with
`serde_json`'s `float_roundtrip`, so thresholds are bit-exact (without it, values on a threshold were routed
wrongly). Raw score of tree slot `k` = `baseline[k]` + the sum of its leaves over all iterations; `multinomial`
applies softmax, `binomial` the logistic function (`[1 - p, p]`), `squared_error` returns the raw value.

Compared with the first layout (one entry per node with `-1` features and zero thresholds on leaves) this is
23 % smaller: the nine shipped models went from 19 MB to 14 MB with identical predictions.

The hall model (`hall3d.json`) wraps three of these:
`{"format": "hallsense-seat-model-v1", "source": ..., "stage1": gbdt, "stage2": gbdt, "count": gbdt}`.

## Measured parity

Golden data: `research/tools/export_golden_model.py` (4.84 MB in `tests/golden/`, recordings from the ESPectre
dataset and `research/rec/`). Run `cargo test -p hallsense-model`; the two ESPectre tests need the dataset
(`ESPECTRE_DATA`, default `~/Projects/ext/espectre/data`) and run with
`cargo test --release -p hallsense-model --test espectre_data -- --ignored`.

| What | Compared with | Result |
|---|---|---|
| `amplitude_from_iq` | `roomfeat.amplitude_from_iq`, 4 recordings | bit-identical |
| `to_grid`, `profile_windows` | `roomfeat` | <= 2.2e-15 relative |
| `phase_from_iq` | `roomfeat.phase_from_iq` | <= 2.9e-14 |
| `windows` (17 columns), `phase_windows`, `rssi_windows` (1 and 2 chains), odd window lengths | `roomfeat`, `rssifeat` | float32 features identical, RSSI column <= 1.2e-15, breathing rates identical |
| `Gbdt::predict_proba` | sklearn `predict_proba` / `predict`, 9 models, real windows + branch-edge and NaN inputs | bit-identical |
| `OccupancyLatch` | `occupancy.py`, 3 x 160 steps | states identical, distance within 1 ulp |
| `RoomEngine` | `live.py::RoomEngine` on the empty -> walk -> still excerpts | 162 / 162 published states identical |
| `RoomEngine` replay | `results/occupancy_engine_replay.json`, 4 full sessions (1375 s) | `state3` 1375 / 1375 (100 %), latch distance 1375 / 1375 |
| hall `link_features`, seat descriptor, scene vector | `model.py` | <= 1.9e-7 relative (NumPy's float32 `log` is not correctly rounded) |
| `SeatModel::predict` | `SeatModel.predict` | stage-1 / stage-2 probabilities and both counts bit-identical |
| `HallEngine` | `live.py::HallEngine`, 12 windows | published states identical |
| `Windower` | `live.py::Windower` | identical |
| `BeaconEngine` | `apsense.py::Engine`, 329 s real recording, self-calibrating and calibrated modes | every tick and the calibration agree (rounded fields exact, link activity <= 1e-12) |
| `roc_auc` | sklearn `roc_auc_score` with ties | < 1e-12 |

Native trainer vs sklearn, `bench/phase_exp.py` protocol, "amplitude (current)" (train on the simulation in
`research/cache/room_sim_amp_{X,y}.npy`, 22 486 windows; test on ESPectre):

| Split | Metric | sklearn (`results/phase_exp.json`) | native |
|---|---|---|---|
| selection (1011 windows) | motion recall / false alarm | 1.000 / 0.000 | 1.000 / 0.000 |
| | still-vs-empty AUC, window / recording | 0.546 / 0.661 | 0.555 / 0.661 |
| holdout (1044 windows) | motion recall / false alarm | 1.000 / 0.000 | 1.000 / 0.000 |
| | still-vs-empty AUC, window / recording | 0.651 / 0.688 | 0.656 / 0.688 |

The native model stopped early after 95 iterations (sklearn: 104) and trains in 0.9 s (release build). sklearn's
own trees evaluated on the Rust features reproduce the sklearn column exactly, so the remaining difference is
the trainer, not the features.

## Deliberate differences from the Python reference

- `amplitude_from_iq` divides by the float32 mean summed subcarrier by subcarrier: `a[:, keep]` is not
  C-contiguous, so NumPy does not use its pairwise sum there.
- `RoomEngine::add` clears a link's buffer when the packet length changes (Python's `np.stack` would raise).
- `HallEngine::push` and `BeaconEngine` take the time from the caller instead of `time.time()`; saving the
  beacon calibration (`cal.json`) is left to the caller (`calibration()` is serializable).
- The ESPectre loader sorts packets with a stable sort (`np.argsort` default is not stable).
- Trainer: thresholds use unweighted quantiles (sklearn 1.9 weights them by the class weights), gradients are
  f64 (sklearn: f32), and the early-stopping validation split is drawn from `ChaCha8Rng(seed)`, not sklearn's RNG.
  Only multinomial, binomial and squared-error losses, no categorical features.
