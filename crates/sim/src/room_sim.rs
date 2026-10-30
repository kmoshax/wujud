//! A simulated bedroom session: one person (plus optional distractors and walls) seen by every link.
//! Port of sim3d.py::simulate_room; `RoomSession::capture` is shared by the batch and streaming APIs.

use std::f64::consts::PI;
use std::ops::Range;

use ndarray::Array2;
use num_complex::Complex64;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::band::{wavelength_of, BandPlan};
use crate::bodies::Body;
use crate::capture::{receive, Capture, LinkCapture};
use crate::channel::{link_channel, link_channel_with_references, Link, ScatterReference};
use crate::constants::SPEED_OF_LIGHT;
use crate::distractors::{distractor, DistractorKind};
use crate::paths::{diffuse_paths, geometric_paths, DiffuseDraws, PropagationPath};
use crate::radio::{
    sabine_amplitude, sample_absorption_coefficient, wall_amplitude, BoardRadios, LinkRadio,
    ReceiverNoise, PERSON_ABSORPTION_M2,
};
use crate::random::{seeded, standard_normals, uniform, SimRng};
use crate::room::{room_person, PersonLabels, PlanStep, RealismKnobs, Room};
use crate::vec3::distance;

pub const ROOM_DIFFUSE_PATHS: usize = 24;
/// Std of the host receive-time jitter (s).
const TIME_JITTER_S: f64 = 0.002;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoomSimOptions {
    pub seconds: f64,
    /// Packets per second on every link.
    pub fs: f64,
    pub calibration_seconds: f64,
    /// `> 1` means noisier, lossier boards.
    pub harsh: f64,
    pub plan: Option<Vec<PlanStep>>,
    pub distractors: Vec<DistractorKind>,
    /// One-way loss of the walls between the boards and the room (through-wall scenario).
    pub wall_db: f64,
    /// Boards outside the room: adds the static "flash" even when `wall_db` is 0.
    pub outside: bool,
    pub knobs: RealismKnobs,
}

impl Default for RoomSimOptions {
    fn default() -> Self {
        RoomSimOptions {
            seconds: 120.0,
            fs: 100.0,
            calibration_seconds: 20.0,
            harsh: 1.0,
            plan: None,
            distractors: Vec::new(),
            wall_db: 0.0,
            outside: false,
            knobs: RealismKnobs::default(),
        }
    }
}

impl RoomSimOptions {
    pub fn session_frames(&self) -> usize {
        (self.seconds * self.fs) as usize
    }

    pub fn calibration_frames(&self) -> usize {
        (self.calibration_seconds * self.fs) as usize
    }

    fn behind_walls(&self) -> bool {
        self.wall_db > 0.0 || self.outside
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoomSimulation {
    /// One entry per `room.links`, router (or lower index) transmitting.
    pub links: Vec<LinkCapture>,
    pub labels: PersonLabels,
    pub fs: f64,
}

/// The static part of one link: its paths, and per channel the furniture clutter and, behind walls,
/// the leakage "flash".
struct RoomLinkSetup {
    tx: usize,
    rx: usize,
    paths: Vec<PropagationPath>,
    rest: Vec<Vec<Complex64>>,
    flash: Option<Vec<Vec<Complex64>>>,
    radio: LinkRadio,
}

/// Everything drawn once per simulated room session; captures consecutive frame ranges on demand.
/// Capturing a session in pieces gives exactly the packets of capturing it at once.
pub struct RoomSession {
    room: Room,
    options: RoomSimOptions,
    band: BandPlan,
    channel_frequencies: Vec<Vec<f64>>,
    bodies: Vec<Body>,
    scatter_references: Vec<ScatterReference>,
    labels: PersonLabels,
    /// Sabine factor of the reverberant field per frame (< 1 while the person is in the room).
    rest_absorb: Vec<f64>,
    links: Vec<RoomLinkSetup>,
    session_receivers: Vec<ReceiverNoise>,
    calibration_receivers: Vec<ReceiverNoise>,
}

impl RoomSession {
    /// A session on the default band (2.4 GHz channel 6, HT-LTF 56).
    pub fn new(room: &Room, options: &RoomSimOptions, rng: &mut SimRng) -> Self {
        RoomSession::with_band(room, options, &BandPlan::default(), rng)
    }

    /// A session measured on `band`. The random draws do not depend on the band, so the same seed gives
    /// the same person, radios and room on every band.
    pub fn with_band(
        room: &Room,
        options: &RoomSimOptions,
        band: &BandPlan,
        rng: &mut SimRng,
    ) -> Self {
        let frames = options.session_frames();
        let (mut person, labels) = room_person(
            rng,
            room,
            frames,
            options.fs,
            options.plan.as_deref(),
            &options.knobs,
        );
        person.absorption = band.effects.body_absorption(person.absorption);
        let person_present: Vec<bool> = (0..frames).map(|t| person.is_present(t)).collect();
        let mut bodies = vec![person];
        for &kind in &options.distractors {
            bodies.push(distractor(rng, kind, &room.shape(), frames, options.fs));
        }
        let radios = band_radios(
            BoardRadios::sample(
                rng,
                room.nodes.len(),
                options.harsh,
                options.knobs.evm_shift_db,
            ),
            band,
        );
        let absorb = sabine_amplitude(
            room.shape().surface_area(),
            sample_absorption_coefficient(rng),
            band.effects.person_absorption_m2 / PERSON_ABSORPTION_M2,
        );
        let rest_absorb = person_present
            .iter()
            .map(|&p| if p { absorb } else { 1.0 })
            .collect();
        let channel_frequencies = band.channel_frequencies();
        let links: Vec<RoomLinkSetup> = room
            .links
            .iter()
            .map(|&(tx, rx)| {
                room_link_setup(rng, room, options, &channel_frequencies, &radios, tx, rx)
            })
            .collect();
        let noise_seed: u64 = rng.gen();
        let receivers = |offset: u64| -> Vec<ReceiverNoise> {
            (0..links.len() as u64)
                .map(|l| ReceiverNoise::new(seeded(noise_seed, 2 * l + offset)))
                .collect()
        };
        RoomSession {
            room: room.clone(),
            options: options.clone(),
            band: band.clone(),
            channel_frequencies,
            scatter_references: bodies.iter().map(ScatterReference::of).collect(),
            bodies,
            labels,
            rest_absorb,
            session_receivers: receivers(0),
            calibration_receivers: receivers(1),
            links,
        }
    }

    pub fn labels(&self) -> &PersonLabels {
        &self.labels
    }

    pub fn bodies(&self) -> &[Body] {
        &self.bodies
    }

    pub fn room(&self) -> &Room {
        &self.room
    }

    pub fn frames(&self) -> usize {
        self.options.session_frames()
    }

    pub fn band(&self) -> &BandPlan {
        &self.band
    }

    /// The noiseless channel `H[t, k]` of every link for `frames`; frame `t` is measured on
    /// channel `t mod N` of the band.
    pub fn channels(&self, frames: Range<usize>) -> Vec<Array2<Complex64>> {
        let by_channel = self.frames_by_channel(frames.clone());
        self.links
            .iter()
            .map(|setup| {
                let per_channel: Vec<Array2<Complex64>> = by_channel
                    .iter()
                    .enumerate()
                    .map(|(channel, selected)| self.session_channel(setup, channel, selected))
                    .collect();
                interleave(per_channel, &by_channel, frames.start, frames.len())
            })
            .collect()
    }

    fn frames_by_channel(&self, frames: Range<usize>) -> Vec<Vec<usize>> {
        let mut by_channel = vec![Vec::new(); self.band.channel_count()];
        for t in frames {
            by_channel[self.band.channel_of_frame(t)].push(t);
        }
        by_channel
    }

    fn session_channel(
        &self,
        setup: &RoomLinkSetup,
        channel: usize,
        frames: &[usize],
    ) -> Array2<Complex64> {
        let bodies: Vec<Body> = self
            .bodies
            .iter()
            .map(|b| b.select_frames(frames))
            .collect();
        let absorb: Vec<f64> = frames.iter().map(|&t| self.rest_absorb[t]).collect();
        let rest = &setup.rest[channel];
        let mut h = link_channel_with_references(
            self.link(setup, channel),
            &setup.paths,
            rest,
            &bodies,
            &self.scatter_references,
            frames.len(),
            1.0,
        );
        adjust_rest_absorption(&mut h, rest, 1.0, &absorb);
        self.apply_walls(&mut h, setup, channel);
        h
    }

    /// The session packets of every link for `frames` (one entry per link). Call it with
    /// consecutive ranges: the receivers' noise state continues from the previous call.
    pub fn capture(&mut self, frames: Range<usize>) -> Vec<Capture> {
        let channels = self.channels(frames.clone());
        let fs = self.options.fs;
        channels
            .iter()
            .zip(&self.links)
            .zip(&mut self.session_receivers)
            .map(|((h, setup), receiver)| {
                receive(h, &setup.radio, receiver, frames.start, fs, TIME_JITTER_S)
            })
            .collect()
    }

    /// The empty-room calibration packets of every link.
    pub fn calibrate(&mut self) -> Vec<Capture> {
        let frames = self.options.calibration_frames();
        let fs = self.options.fs;
        let by_channel = self.frames_by_channel(0..frames);
        let channels: Vec<Array2<Complex64>> = self
            .links
            .iter()
            .map(|setup| {
                let per_channel: Vec<Array2<Complex64>> = by_channel
                    .iter()
                    .enumerate()
                    .map(|(channel, selected)| {
                        let mut h = link_channel(
                            self.link(setup, channel),
                            &setup.paths,
                            &setup.rest[channel],
                            &[],
                            selected.len(),
                            1.0,
                        );
                        self.apply_walls(&mut h, setup, channel);
                        h
                    })
                    .collect();
                interleave(per_channel, &by_channel, 0, frames)
            })
            .collect();
        channels
            .iter()
            .zip(&self.links)
            .zip(&mut self.calibration_receivers)
            .map(|((h, setup), receiver)| receive(h, &setup.radio, receiver, 0, fs, 0.0))
            .collect()
    }

    fn link(&self, setup: &RoomLinkSetup, channel: usize) -> Link<'_> {
        Link {
            tx: self.room.nodes[setup.tx],
            rx: self.room.nodes[setup.rx],
            frequencies: &self.channel_frequencies[channel],
        }
    }

    fn apply_walls(&self, h: &mut Array2<Complex64>, setup: &RoomLinkSetup, channel: usize) {
        let Some(flash) = setup.flash.as_ref().map(|f| &f[channel]) else {
            return;
        };
        let through_wall = wall_amplitude(self.options.wall_db);
        for mut row in h.rows_mut() {
            for (value, leak) in row.iter_mut().zip(flash) {
                *value = *value * through_wall + leak;
            }
        }
    }
}

/// Rows of each channel's `H` (taken at `frames_by_channel[c]`) put back in frame order.
fn interleave(
    per_channel: Vec<Array2<Complex64>>,
    frames_by_channel: &[Vec<usize>],
    first_frame: usize,
    frames: usize,
) -> Array2<Complex64> {
    if per_channel.len() == 1 {
        return per_channel.into_iter().next().expect("one channel");
    }
    let subcarriers = per_channel.iter().map(|h| h.ncols()).max().unwrap_or(0);
    let mut h = Array2::zeros((frames, subcarriers));
    for (channel_h, selected) in per_channel.iter().zip(frames_by_channel) {
        for (row, &t) in channel_h.rows().into_iter().zip(selected) {
            h.row_mut(t - first_frame).assign(&row);
        }
    }
    h
}

/// The board radios with the band's transmit-power and estimation-noise offsets.
fn band_radios(mut radios: BoardRadios, band: &BandPlan) -> BoardRadios {
    let tx_offset = band.effects.tx_power_offset_db + band.layout.thermal_snr_offset_db();
    if tx_offset != 0.0 {
        radios.tx_power_dbm.iter_mut().for_each(|p| *p += tx_offset);
    }
    if band.effects.evm_offset_db != 0.0 {
        let offset = band.effects.evm_offset_db;
        radios.evm_db.iter_mut().for_each(|e| *e += offset);
    }
    radios
}

fn room_link_setup(
    rng: &mut SimRng,
    room: &Room,
    options: &RoomSimOptions,
    channel_frequencies: &[Vec<f64>],
    radios: &BoardRadios,
    tx: usize,
    rx: usize,
) -> RoomLinkSetup {
    let (a, b) = (room.nodes[tx], room.nodes[rx]);
    let mut paths = geometric_paths(a, b, &room.enclosure);
    let draws = DiffuseDraws::sample(rng, &room.shape(), ROOM_DIFFUSE_PATHS);
    paths.extend(diffuse_paths(a, b, &draws));
    let rest = channel_frequencies
        .iter()
        .map(|frequencies| crate::scatter::static_rest(a, b, &room.clutter, frequencies))
        .collect();
    let flash = options.behind_walls().then(|| {
        let echoes = WallEchoes::sample(rng);
        channel_frequencies
            .iter()
            .map(|frequencies| echoes.spectrum(distance(a, b), frequencies))
            .collect()
    });
    RoomLinkSetup {
        tx,
        rx,
        paths,
        rest,
        flash,
        radio: radios.link(tx, rx, 0.0),
    }
}

/// Rescales the reverberant field `rest` in `h` from the `applied` Sabine factor to `wanted[t]`.
pub fn adjust_rest_absorption(
    h: &mut Array2<Complex64>,
    rest: &[Complex64],
    applied: f64,
    wanted: &[f64],
) {
    for (mut row, &factor) in h.rows_mut().into_iter().zip(wanted) {
        for (value, r) in row.iter_mut().zip(rest) {
            *value += r * (factor - applied);
        }
    }
}

/// Six static echoes off the outside of the walls (5-60 ns late) that the walls do not attenuate.
struct WallEchoes {
    delays: Vec<f64>,
    re: Vec<f64>,
    im: Vec<f64>,
}

impl WallEchoes {
    const COUNT: usize = 6;

    fn sample(rng: &mut SimRng) -> Self {
        let delays = (0..Self::COUNT)
            .map(|_| uniform(rng, 5e-9, 60e-9))
            .collect();
        let re = standard_normals(rng, Self::COUNT);
        let im = standard_normals(rng, Self::COUNT);
        WallEchoes { delays, re, im }
    }

    fn spectrum(&self, link_length: f64, frequencies: &[f64]) -> Vec<Complex64> {
        let scale = 0.35 * wavelength_of(frequencies) / (4.0 * PI * link_length) / 12f64.sqrt();
        frequencies
            .iter()
            .map(|&f| {
                (0..Self::COUNT)
                    .map(|e| {
                        let gain = Complex64::new(self.re[e], self.im[e]) * scale;
                        let delay = link_length / SPEED_OF_LIGHT + self.delays[e];
                        gain * Complex64::from_polar(1.0, -2.0 * PI * f * delay)
                    })
                    .sum()
            })
            .collect()
    }
}

/// The whole session at once (sim3d.py::simulate_room).
pub fn simulate_room(room: &Room, options: &RoomSimOptions, rng: &mut SimRng) -> RoomSimulation {
    simulate_room_on_band(room, options, &BandPlan::default(), rng)
}

/// `simulate_room` measured on `band` (packet `n` of a capture used channel `frame[n] mod N`).
pub fn simulate_room_on_band(
    room: &Room,
    options: &RoomSimOptions,
    band: &BandPlan,
    rng: &mut SimRng,
) -> RoomSimulation {
    let mut session = RoomSession::with_band(room, options, band, rng);
    let captures = session.capture(0..session.frames());
    let calibrations = session.calibrate();
    let links = session
        .links
        .iter()
        .zip(captures.into_iter().zip(calibrations))
        .map(|(setup, (session, calibration))| LinkCapture {
            tx: setup.tx,
            rx: setup.rx,
            calibration,
            session,
        })
        .collect();
    RoomSimulation {
        links,
        labels: session.labels,
        fs: options.fs,
    }
}
