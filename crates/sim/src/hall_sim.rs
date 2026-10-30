//! Simulated lecture-hall sessions seen by every board pair, in both directions.
//! Ports of sim3d.py::simulate_hall3 (`HallSession::lecture`) and ::simulate_hall_timeline3
//! (`HallSession::arrivals`), producing raw packets instead of window features.

use std::ops::Range;

use ndarray::Array2;
use num_complex::Complex64;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::bodies::Body;
use crate::capture::{receive, Capture, LinkCapture};
use crate::channel::{link_channel, link_channel_with_references, Link, ScatterReference};
use crate::constants::l_ltf_frequencies;
use crate::hall::{lecturer, sample_occupancy, seated_students, Hall};
use crate::paths::{diffuse_paths, geometric_paths, DiffuseDraws, PropagationPath};
use crate::radio::{
    sabine_amplitude, sample_absorption_coefficient, BoardRadios, ReceiverNoise,
    PERSON_ABSORPTION_M2,
};
use crate::random::{chance, normal, seeded, uniform, SimRng};
use crate::room_sim::adjust_rest_absorption;
use crate::scatter::{static_rest, Scatterer};

pub const HALL_DIFFUSE_PATHS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HallSimOptions {
    pub seconds: f64,
    pub calibration_seconds: f64,
    pub fs: f64,
    pub harsh: f64,
    /// Scales every student's fidget rate.
    pub restless: f64,
    pub with_lecturer: bool,
}

impl Default for HallSimOptions {
    fn default() -> Self {
        HallSimOptions {
            seconds: 60.0,
            calibration_seconds: 30.0,
            fs: 20.0,
            harsh: 1.0,
            restless: 1.0,
            with_lecturer: true,
        }
    }
}

impl HallSimOptions {
    pub fn session_frames(&self) -> usize {
        (self.seconds * self.fs) as usize
    }

    pub fn calibration_frames(&self) -> usize {
        (self.calibration_seconds * self.fs) as usize
    }
}

/// A lecture where students arrive over time (the live demo's hall).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArrivalOptions {
    pub minutes: f64,
    /// Fraction of seats taken at the end.
    pub final_fraction: f64,
    pub calibration_seconds: f64,
    pub fs: f64,
}

impl Default for ArrivalOptions {
    fn default() -> Self {
        ArrivalOptions {
            minutes: 4.0,
            final_fraction: 0.75,
            calibration_seconds: 30.0,
            fs: 20.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HallSimulation {
    /// Two entries per `hall.links` pair `(i, j)`: `i -> j` then `j -> i`.
    pub links: Vec<LinkCapture>,
    pub occupied: Vec<bool>,
    /// Per seat, the frame the student sits down (`Some(0)` = from the start, `None` = stays empty).
    pub arrival_frame: Vec<Option<usize>>,
    pub fs: f64,
}

struct HallLinkSetup {
    tx: usize,
    rx: usize,
    paths: Vec<PropagationPath>,
    calibration_rest: Vec<Complex64>,
    session_rest: Vec<Complex64>,
}

/// Everything drawn once per hall session; captures consecutive frame ranges of every direction on
/// demand, with the same result as capturing the whole session at once.
pub struct HallSession {
    hall: Hall,
    fs: f64,
    session_frames: usize,
    calibration_frames: usize,
    frequencies: Vec<f64>,
    bodies: Vec<Body>,
    occupied: Vec<bool>,
    arrival_frame: Vec<Option<usize>>,
    radios: BoardRadios,
    session_drift_db: Vec<f64>,
    /// Applied to the body scatter and the reverberant field (simulate_hall3) ...
    scatter_absorb: f64,
    /// ... or to the reverberant field only, per frame, as people arrive (timeline).
    rest_absorb: Vec<f64>,
    links: Vec<HallLinkSetup>,
    scatter_references: Vec<ScatterReference>,
    session_receivers: Vec<ReceiverNoise>,
    calibration_receivers: Vec<ReceiverNoise>,
}

impl HallSession {
    /// Students in `occupied` seats for the whole session, chairs pulled out and bags dropped.
    pub fn lecture(
        hall: &Hall,
        occupied: &[bool],
        options: &HallSimOptions,
        rng: &mut SimRng,
    ) -> Self {
        let frames = options.session_frames();
        let seats: Vec<_> = seats_where(hall, occupied);
        let mut bodies = seated_students(rng, &seats, frames, options.fs, options.restless);
        if options.with_lecturer {
            bodies.extend(lecturer(rng, hall, frames, options.fs));
        }
        let radios = BoardRadios::sample(rng, hall.nodes.len(), options.harsh, 0.0);
        let session_drift_db = (0..hall.nodes.len())
            .map(|_| normal(rng, 0.0, 0.25 * options.harsh))
            .collect();
        let people = occupied.iter().filter(|&&o| o).count() as f64;
        let absorb = sabine_amplitude(
            hall.shape().surface_area(),
            sample_absorption_coefficient(rng),
            people,
        );
        let session_clutter = lecture_clutter(rng, hall, occupied);
        let frequencies = l_ltf_frequencies();
        let links = hall
            .links
            .iter()
            .map(|&(i, j)| hall_link_setup(rng, hall, &frequencies, &session_clutter, i, j))
            .collect::<Vec<_>>();
        let noise_seed: u64 = rng.gen();
        HallSession {
            hall: hall.clone(),
            scatter_references: bodies.iter().map(ScatterReference::of).collect(),
            session_receivers: direction_receivers(noise_seed, links.len(), 0),
            calibration_receivers: direction_receivers(noise_seed, links.len(), 1),
            fs: options.fs,
            session_frames: frames,
            calibration_frames: options.calibration_frames(),
            frequencies,
            bodies,
            occupied: occupied.to_vec(),
            arrival_frame: occupied.iter().map(|&o| o.then_some(0)).collect(),
            radios,
            session_drift_db,
            scatter_absorb: absorb,
            rest_absorb: vec![absorb; frames],
            links,
        }
    }

    /// Students arrive one by one between 15 % and 70 % of the session.
    pub fn arrivals(hall: &Hall, options: &ArrivalOptions, rng: &mut SimRng) -> Self {
        let frames = (options.minutes * 60.0 * options.fs) as usize;
        let (occupied, _) = sample_occupancy(rng, hall, Some(options.final_fraction));
        let seat_index: Vec<usize> = (0..occupied.len()).filter(|&s| occupied[s]).collect();
        let arrivals = arrival_frames(rng, seat_index.len(), frames);
        let seats = seats_where(hall, &occupied);
        let mut bodies = seated_students(rng, &seats, frames, options.fs, 1.0);
        for (body, &arrival) in bodies.iter_mut().zip(&arrivals) {
            body.hide(0..arrival);
        }
        bodies.extend(lecturer(rng, hall, frames, options.fs));
        let radios = BoardRadios::sample(rng, hall.nodes.len(), 1.0, 0.0);
        let rest_absorb = arrival_absorption(rng, hall, &arrivals, frames);
        let mut arrival_frame = vec![None; occupied.len()];
        for (&seat, &arrival) in seat_index.iter().zip(&arrivals) {
            arrival_frame[seat] = Some(arrival);
        }
        let frequencies = l_ltf_frequencies();
        let links = hall
            .links
            .iter()
            .map(|&(i, j)| hall_link_setup(rng, hall, &frequencies, &hall.clutter, i, j))
            .collect::<Vec<_>>();
        let noise_seed: u64 = rng.gen();
        HallSession {
            hall: hall.clone(),
            scatter_references: bodies.iter().map(ScatterReference::of).collect(),
            session_receivers: direction_receivers(noise_seed, links.len(), 0),
            calibration_receivers: direction_receivers(noise_seed, links.len(), 1),
            fs: options.fs,
            session_frames: frames,
            calibration_frames: (options.calibration_seconds * options.fs) as usize,
            frequencies,
            bodies,
            occupied,
            arrival_frame,
            session_drift_db: vec![0.0; hall.nodes.len()],
            radios,
            scatter_absorb: 1.0,
            rest_absorb,
            links,
        }
    }

    pub fn hall(&self) -> &Hall {
        &self.hall
    }

    pub fn bodies(&self) -> &[Body] {
        &self.bodies
    }

    pub fn frames(&self) -> usize {
        self.session_frames
    }

    pub fn fs(&self) -> f64 {
        self.fs
    }

    pub fn occupied(&self) -> &[bool] {
        &self.occupied
    }

    pub fn arrival_frame(&self) -> &[Option<usize>] {
        &self.arrival_frame
    }

    /// Which seats hold a student at `frame`.
    pub fn present_at(&self, frame: usize) -> Vec<bool> {
        self.arrival_frame
            .iter()
            .map(|a| a.is_some_and(|a| a <= frame))
            .collect()
    }

    /// `(tx, rx)` of each capture, in the order `capture` and `calibrate` return them.
    pub fn directions(&self) -> Vec<(usize, usize)> {
        self.links
            .iter()
            .flat_map(|s| [(s.tx, s.rx), (s.rx, s.tx)])
            .collect()
    }

    /// The noiseless session channel of every link pair for `frames`.
    pub fn channels(&self, frames: Range<usize>) -> Vec<Array2<Complex64>> {
        let bodies: Vec<Body> = self
            .bodies
            .iter()
            .map(|b| b.slice_frames(frames.clone()))
            .collect();
        let rest_absorb = &self.rest_absorb[frames.clone()];
        self.links
            .iter()
            .map(|setup| {
                let mut h = link_channel_with_references(
                    self.link(setup),
                    &setup.paths,
                    &setup.session_rest,
                    &bodies,
                    &self.scatter_references,
                    frames.len(),
                    self.scatter_absorb,
                );
                adjust_rest_absorption(
                    &mut h,
                    &setup.session_rest,
                    self.scatter_absorb,
                    rest_absorb,
                );
                h
            })
            .collect()
    }

    /// Session packets of every direction for `frames`. Call it with consecutive ranges: the
    /// receivers' noise state continues from the previous call.
    pub fn capture(&mut self, frames: Range<usize>) -> Vec<Capture> {
        let channels = self.channels(frames.clone());
        self.receive_all(&channels, frames.start, true)
    }

    /// Empty-hall calibration packets of every direction.
    pub fn calibrate(&mut self) -> Vec<Capture> {
        let frames = self.calibration_frames;
        let channels: Vec<Array2<Complex64>> = self
            .links
            .iter()
            .map(|setup| {
                link_channel(
                    self.link(setup),
                    &setup.paths,
                    &setup.calibration_rest,
                    &[],
                    frames,
                    1.0,
                )
            })
            .collect();
        self.receive_all(&channels, 0, false)
    }

    fn receive_all(
        &mut self,
        channels: &[Array2<Complex64>],
        first_frame: usize,
        session: bool,
    ) -> Vec<Capture> {
        let receivers = if session {
            &mut self.session_receivers
        } else {
            &mut self.calibration_receivers
        };
        let directions = self
            .links
            .iter()
            .zip(channels)
            .flat_map(|(setup, h)| [(setup.tx, setup.rx, h), (setup.rx, setup.tx, h)]);
        directions
            .zip(receivers.iter_mut())
            .map(|((tx, rx, h), receiver)| {
                let drift = if session {
                    self.session_drift_db[tx]
                } else {
                    0.0
                };
                let radio = self.radios.link(tx, rx, drift);
                receive(h, &radio, receiver, first_frame, self.fs, 0.0)
            })
            .collect()
    }

    fn link<'a>(&'a self, setup: &HallLinkSetup) -> Link<'a> {
        Link {
            tx: self.hall.nodes[setup.tx],
            rx: self.hall.nodes[setup.rx],
            frequencies: &self.frequencies,
        }
    }

    /// The whole session at once.
    pub fn simulate(&mut self) -> HallSimulation {
        let sessions = self.capture(0..self.session_frames);
        let calibrations = self.calibrate();
        let links = self
            .directions()
            .into_iter()
            .zip(sessions.into_iter().zip(calibrations))
            .map(|((tx, rx), (session, calibration))| LinkCapture {
                tx,
                rx,
                calibration,
                session,
            })
            .collect();
        HallSimulation {
            links,
            occupied: self.occupied.clone(),
            arrival_frame: self.arrival_frame.clone(),
            fs: self.fs,
        }
    }
}

/// Two directions per link; `offset` 0 = session, 1 = calibration.
fn direction_receivers(noise_seed: u64, links: usize, offset: u64) -> Vec<ReceiverNoise> {
    (0..2 * links as u64)
        .map(|d| ReceiverNoise::new(seeded(noise_seed, 2 * d + offset)))
        .collect()
}

fn seats_where(hall: &Hall, occupied: &[bool]) -> Vec<crate::vec3::Vec3> {
    hall.seats
        .iter()
        .zip(occupied)
        .filter(|(_, &o)| o)
        .map(|(&s, _)| s)
        .collect()
}

fn hall_link_setup(
    rng: &mut SimRng,
    hall: &Hall,
    frequencies: &[f64],
    session_clutter: &[Scatterer],
    tx: usize,
    rx: usize,
) -> HallLinkSetup {
    let (a, b) = (hall.nodes[tx], hall.nodes[rx]);
    let mut paths = geometric_paths(a, b, &hall.enclosure);
    let draws = DiffuseDraws::sample(rng, &hall.shape(), HALL_DIFFUSE_PATHS);
    paths.extend(diffuse_paths(a, b, &draws));
    HallLinkSetup {
        tx,
        rx,
        paths,
        calibration_rest: static_rest(a, b, &hall.clutter, frequencies),
        session_rest: static_rest(a, b, session_clutter, frequencies),
    }
}

/// During the lecture the chairs of occupied seats are pulled out (~8 cm) and 60 % of students drop a bag.
fn lecture_clutter(rng: &mut SimRng, hall: &Hall, occupied: &[bool]) -> Vec<Scatterer> {
    let mut clutter = hall.clutter.clone();
    for (chair, _) in clutter.iter_mut().zip(occupied).filter(|(_, &o)| o) {
        chair.position[0] += normal(rng, 0.0, 0.08);
        chair.position[1] += normal(rng, 0.0, 0.08);
    }
    let has_bag: Vec<bool> = occupied.iter().map(|&o| chance(rng, 0.6) && o).collect();
    for (seat, _) in hall.seats.iter().zip(&has_bag).filter(|(_, &b)| b) {
        let position = [
            seat[0] + normal(rng, 0.0, 0.2),
            seat[1] + normal(rng, 0.0, 0.2),
            seat[2] + 0.2,
        ];
        clutter.push(Scatterer {
            position,
            rcs: uniform(rng, 0.01, 0.08),
        });
    }
    clutter
}

/// Sorted arrival times between 15 % and 70 % of the session, dealt to the students in random order.
fn arrival_frames(rng: &mut SimRng, students: usize, frames: usize) -> Vec<usize> {
    let mut fractions: Vec<f64> = (0..students).map(|_| uniform(rng, 0.15, 0.7)).collect();
    fractions.sort_by(f64::total_cmp);
    let mut arrivals: Vec<usize> = fractions
        .iter()
        .map(|f| (f * frames as f64) as usize)
        .collect();
    arrivals.shuffle(rng);
    arrivals
}

fn arrival_absorption(
    rng: &mut SimRng,
    hall: &Hall,
    arrivals: &[usize],
    frames: usize,
) -> Vec<f64> {
    let room_absorption = sample_absorption_coefficient(rng) * hall.shape().surface_area();
    let mut present = vec![0usize; frames];
    for &arrival in arrivals {
        for count in &mut present[arrival.min(frames)..] {
            *count += 1;
        }
    }
    present
        .iter()
        .map(|&n| (room_absorption / (room_absorption + PERSON_ABSORPTION_M2 * n as f64)).sqrt())
        .collect()
}
