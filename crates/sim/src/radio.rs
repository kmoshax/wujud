//! The ESP32 receive chain: Tx power, thermal + estimation noise, interference bursts, RSSI,
//! AGC, int8 quantisation and packet loss.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::constants::NOISE_FLOOR_DBM;
use crate::random::{normal, standard_normals, uniform, unit, SimRng};

/// The radio of every board in a scene, drawn once per simulation (sim3d.py::_radio).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoardRadios {
    pub tx_power_dbm: Vec<f64>,
    pub evm_db: Vec<f64>,
    pub agc_target: Vec<f64>,
    pub loss_probability: f64,
    pub interference_probability: f64,
}

impl BoardRadios {
    /// `harsh > 1` means noisier, lossier boards; `evm_shift_db` raises the estimation noise.
    pub fn sample(rng: &mut SimRng, boards: usize, harsh: f64, evm_shift_db: f64) -> Self {
        let tx_power_dbm = (0..boards).map(|_| uniform(rng, 4.0, 16.0)).collect();
        let evm_db = (0..boards)
            .map(|_| uniform(rng, -29.0, -23.0) + 2.0 * (harsh - 1.0) + evm_shift_db)
            .collect();
        let agc_target = (0..boards).map(|_| uniform(rng, 12.0, 24.0)).collect();
        BoardRadios {
            tx_power_dbm,
            evm_db,
            agc_target,
            loss_probability: uniform(rng, 0.01, 0.08) * harsh,
            interference_probability: uniform(rng, 0.005, 0.03) * harsh,
        }
    }

    pub fn link(&self, tx: usize, rx: usize, tx_drift_db: f64) -> LinkRadio {
        LinkRadio {
            tx_power_dbm: self.tx_power_dbm[tx] + tx_drift_db,
            evm_db: self.evm_db[rx],
            agc_target: self.agc_target[rx],
            loss_probability: self.loss_probability,
            interference_probability: self.interference_probability,
        }
    }
}

/// Radio parameters of one transmitter -> receiver direction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkRadio {
    pub tx_power_dbm: f64,
    pub evm_db: f64,
    pub agc_target: f64,
    pub loss_probability: f64,
    pub interference_probability: f64,
}

/// What the board reports per packet. `kept[t] = false` means the packet was lost.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    pub iq: Array3<i8>,
    pub rssi: Vec<i16>,
    pub kept: Vec<bool>,
}

/// Every random number `measure` uses, drawn in sim3d.py::measure's order. Golden tests fill it
/// with the numbers numpy drew, so the deterministic part is checked bit for bit.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasureNoise {
    pub noise_re: Array2<f64>,
    pub noise_im: Array2<f64>,
    pub interference_draw: Vec<f64>,
    /// One entry per packet hit by interference, in packet order.
    pub burst_db: Vec<f64>,
    pub burst_re: Array2<f64>,
    pub burst_im: Array2<f64>,
    pub agc_step_draw: Vec<f64>,
    pub rssi_bias_levels: Vec<f64>,
    pub rssi_jitter: Vec<f64>,
    pub agc_gain_db: Vec<f64>,
    pub loss_draw: Vec<f64>,
}

/// The RSSI gain-estimate bias changes about three times a minute (sim3d.py).
const AGC_STEPS_PER_MINUTE: f64 = 3.0;

impl MeasureNoise {
    pub fn sample(
        rng: &mut SimRng,
        frames: usize,
        subcarriers: usize,
        radio: &LinkRadio,
        fs: f64,
    ) -> Self {
        let noise_re = normal_matrix(rng, frames, subcarriers);
        let noise_im = normal_matrix(rng, frames, subcarriers);
        let interference_draw: Vec<f64> = (0..frames).map(|_| unit(rng)).collect();
        let bursts = interference_draw
            .iter()
            .filter(|&&u| u < radio.interference_probability)
            .count();
        let burst_db = (0..bursts).map(|_| uniform(rng, -12.0, 0.0)).collect();
        let burst_re = normal_matrix(rng, bursts, subcarriers);
        let burst_im = normal_matrix(rng, bursts, subcarriers);
        let agc_step_draw: Vec<f64> = (0..frames).map(|_| unit(rng)).collect();
        let levels = agc_step_index(&agc_step_draw, fs).last().map_or(0, |&s| s) + 1;
        MeasureNoise {
            noise_re,
            noise_im,
            interference_draw,
            burst_db,
            burst_re,
            burst_im,
            agc_step_draw,
            rssi_bias_levels: (0..levels)
                .map(|_| normal(rng, 0.0, RSSI_BIAS_STD_DB))
                .collect(),
            rssi_jitter: (0..frames).map(|_| normal(rng, 0.0, 0.5)).collect(),
            agc_gain_db: (0..frames).map(|_| normal(rng, 0.0, 1.0)).collect(),
            loss_draw: (0..frames).map(|_| unit(rng)).collect(),
        }
    }
}

/// The random state of one receiving direction over a whole session. Noise is drawn frame by frame
/// in a fixed order and the RSSI gain-estimate bias carries over between calls, so splitting the
/// session into any number of `draw` calls yields exactly the numbers of a single call.
#[derive(Clone, Debug)]
pub struct ReceiverNoise {
    rng: SimRng,
    rssi_bias_db: f64,
}

struct Burst {
    level_db: f64,
    re: Vec<f64>,
    im: Vec<f64>,
}

/// One frame's worth of receiver noise, in draw order.
struct FrameNoise {
    noise_re: Vec<f64>,
    noise_im: Vec<f64>,
    interference_draw: f64,
    burst: Option<Burst>,
    agc_step_draw: f64,
    new_rssi_bias: Option<f64>,
    rssi_jitter: f64,
    agc_gain_db: f64,
    loss_draw: f64,
    time_jitter: f64,
}

impl ReceiverNoise {
    pub fn new(mut rng: SimRng) -> Self {
        let rssi_bias_db = normal(&mut rng, 0.0, RSSI_BIAS_STD_DB);
        ReceiverNoise { rng, rssi_bias_db }
    }

    /// Noise for the next `frames` packets, plus one standard-normal draw per packet for the
    /// host's receive-time jitter.
    pub fn draw(
        &mut self,
        frames: usize,
        subcarriers: usize,
        radio: &LinkRadio,
        fs: f64,
    ) -> (MeasureNoise, Vec<f64>) {
        let jump_probability = AGC_STEPS_PER_MINUTE / (60.0 * fs);
        let per_frame: Vec<FrameNoise> = (0..frames)
            .map(|_| self.frame(subcarriers, radio, jump_probability))
            .collect();
        let mut rssi_bias_levels = vec![self.rssi_bias_db];
        rssi_bias_levels.extend(per_frame.iter().filter_map(|f| f.new_rssi_bias));
        self.rssi_bias_db = *rssi_bias_levels
            .last()
            .expect("at least the starting level");
        let rows = |pick: fn(&FrameNoise) -> &[f64]| {
            let values: Vec<f64> = per_frame
                .iter()
                .flat_map(|f| pick(f).iter().copied())
                .collect();
            Array2::from_shape_vec((frames, subcarriers), values).expect("frames × subcarriers")
        };
        let bursts: Vec<&Burst> = per_frame.iter().filter_map(|f| f.burst.as_ref()).collect();
        let burst_rows = |pick: fn(&Burst) -> &[f64]| {
            let values: Vec<f64> = bursts
                .iter()
                .flat_map(|b| pick(b).iter().copied())
                .collect();
            Array2::from_shape_vec((bursts.len(), subcarriers), values)
                .expect("bursts × subcarriers")
        };
        let noise = MeasureNoise {
            noise_re: rows(|f| &f.noise_re),
            noise_im: rows(|f| &f.noise_im),
            interference_draw: per_frame.iter().map(|f| f.interference_draw).collect(),
            burst_db: bursts.iter().map(|b| b.level_db).collect(),
            burst_re: burst_rows(|b| &b.re),
            burst_im: burst_rows(|b| &b.im),
            agc_step_draw: per_frame.iter().map(|f| f.agc_step_draw).collect(),
            rssi_bias_levels,
            rssi_jitter: per_frame.iter().map(|f| f.rssi_jitter).collect(),
            agc_gain_db: per_frame.iter().map(|f| f.agc_gain_db).collect(),
            loss_draw: per_frame.iter().map(|f| f.loss_draw).collect(),
        };
        (noise, per_frame.iter().map(|f| f.time_jitter).collect())
    }

    fn frame(
        &mut self,
        subcarriers: usize,
        radio: &LinkRadio,
        jump_probability: f64,
    ) -> FrameNoise {
        let rng = &mut self.rng;
        let noise_re = standard_normals(rng, subcarriers);
        let noise_im = standard_normals(rng, subcarriers);
        let interference_draw = unit(rng);
        let burst = (interference_draw < radio.interference_probability).then(|| {
            let level_db = uniform(rng, -12.0, 0.0);
            let re = standard_normals(rng, subcarriers);
            Burst {
                level_db,
                re,
                im: standard_normals(rng, subcarriers),
            }
        });
        let agc_step_draw = unit(rng);
        let new_rssi_bias =
            (agc_step_draw < jump_probability).then(|| normal(rng, 0.0, RSSI_BIAS_STD_DB));
        FrameNoise {
            noise_re,
            noise_im,
            interference_draw,
            burst,
            agc_step_draw,
            new_rssi_bias,
            rssi_jitter: normal(rng, 0.0, 0.5),
            agc_gain_db: normal(rng, 0.0, 1.0),
            loss_draw: unit(rng),
            time_jitter: normal(rng, 0.0, 1.0),
        }
    }
}

const RSSI_BIAS_STD_DB: f64 = 0.6;

fn normal_matrix(rng: &mut SimRng, rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_vec((rows, cols), standard_normals(rng, rows * cols))
        .expect("rows × cols values")
}

/// Index of the current AGC state for each packet (a counting process with ~3 jumps a minute).
fn agc_step_index(step_draw: &[f64], fs: f64) -> Vec<usize> {
    let jump_probability = AGC_STEPS_PER_MINUTE / (60.0 * fs);
    let mut index = 0;
    step_draw
        .iter()
        .map(|&u| {
            if u < jump_probability {
                index += 1;
            }
            index
        })
        .collect()
}

/// Simulates the ESP32 measuring the complex channel `h[t, k]` at `fs` packets per second.
pub fn measure(h: &Array2<Complex64>, radio: &LinkRadio, fs: f64, rng: &mut SimRng) -> Measurement {
    let (frames, subcarriers) = h.dim();
    let noise = MeasureNoise::sample(rng, frames, subcarriers, radio, fs);
    measure_with_noise(h, radio, fs, &noise)
}

pub fn measure_with_noise(
    h: &Array2<Complex64>,
    radio: &LinkRadio,
    fs: f64,
    noise: &MeasureNoise,
) -> Measurement {
    let (frames, subcarriers) = h.dim();
    let received = h * Complex64::new(db_to_amplitude(radio.tx_power_dbm), 0.0);
    let signal_power: Vec<f64> = received
        .rows()
        .into_iter()
        .map(|r| mean_power(r.iter()))
        .collect();
    let mut y = add_receiver_noise(&received, &signal_power, radio.evm_db, noise);
    add_interference_bursts(&mut y, &signal_power, radio.interference_probability, noise);
    let received_power: Vec<f64> = y.rows().into_iter().map(|r| mean_power(r.iter())).collect();
    let steps = agc_step_index(&noise.agc_step_draw, fs);
    let rssi = (0..frames)
        .map(|t| {
            let dbm = 10.0 * received_power[t].log10()
                + noise.rssi_bias_levels[steps[t]]
                + noise.rssi_jitter[t];
            dbm.round_ties_even() as i16
        })
        .collect();
    let mut iq = Array3::zeros((frames, subcarriers, 2));
    for t in 0..frames {
        let agc_gain =
            radio.agc_target * db_to_amplitude(noise.agc_gain_db[t]) / received_power[t].sqrt();
        for k in 0..subcarriers {
            let scaled = y[[t, k]] * agc_gain;
            iq[[t, k, 0]] = quantize_to_int8(scaled.re);
            iq[[t, k, 1]] = quantize_to_int8(scaled.im);
        }
    }
    let kept = noise
        .loss_draw
        .iter()
        .map(|&u| u > radio.loss_probability)
        .collect();
    Measurement { iq, rssi, kept }
}

/// `σ² = N0 + P_sig·10^(EVM/10)` per packet, split equally over I and Q.
fn add_receiver_noise(
    received: &Array2<Complex64>,
    signal_power: &[f64],
    evm_db: f64,
    noise: &MeasureNoise,
) -> Array2<Complex64> {
    let thermal = 10f64.powf(NOISE_FLOOR_DBM / 10.0);
    let estimation = 10f64.powf(evm_db / 10.0);
    Array2::from_shape_fn(received.dim(), |(t, k)| {
        let sigma = ((thermal + signal_power[t] * estimation) / 2.0).sqrt();
        received[[t, k]] + sigma * Complex64::new(noise.noise_re[[t, k]], noise.noise_im[[t, k]])
    })
}

fn add_interference_bursts(
    y: &mut Array2<Complex64>,
    signal_power: &[f64],
    probability: f64,
    noise: &MeasureNoise,
) {
    let hit = noise
        .interference_draw
        .iter()
        .enumerate()
        .filter(|(_, &u)| u < probability)
        .map(|(t, _)| t);
    for (burst, t) in hit.enumerate() {
        let sigma = (signal_power[t] * 10f64.powf(noise.burst_db[burst] / 10.0) / 2.0).sqrt();
        for k in 0..y.ncols() {
            y[[t, k]] +=
                sigma * Complex64::new(noise.burst_re[[burst, k]], noise.burst_im[[burst, k]]);
        }
    }
}

fn mean_power<'a>(values: impl ExactSizeIterator<Item = &'a Complex64>) -> f64 {
    let n = values.len() as f64;
    values.map(|v| v.norm_sqr()).sum::<f64>() / n
}

pub fn db_to_amplitude(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

/// numpy `clip(round(x), -127, 127).astype(int8)`; numpy rounds half to even.
pub fn quantize_to_int8(value: f64) -> i8 {
    value.round_ties_even().clamp(-127.0, 127.0) as i8
}

/// Amplitude factor of the reverberant field when `absorbers` people (0.4 m² each) join a room
/// with absorption area `A0 = α·S`: `sqrt(A0 / (A0 + 0.4 N))` (Sabine, steady state).
pub fn sabine_amplitude(surface_area: f64, absorption_coefficient: f64, absorbers: f64) -> f64 {
    let room_absorption = absorption_coefficient * surface_area;
    (room_absorption / (room_absorption + PERSON_ABSORPTION_M2 * absorbers)).sqrt()
}

pub const PERSON_ABSORPTION_M2: f64 = 0.4;

/// Draws the room's absorption coefficient α ~ U(0.1, 0.3) (sim3d.py::_absorb).
pub fn sample_absorption_coefficient(rng: &mut SimRng) -> f64 {
    uniform(rng, 0.1, 0.3)
}

/// One-way amplitude factor of a wall with `wall_db` of loss.
pub fn wall_amplitude(wall_db: f64) -> f64 {
    db_to_amplitude(-wall_db)
}
