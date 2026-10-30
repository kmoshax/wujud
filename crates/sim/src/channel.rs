//! The complex channel `H[t, k]` of one link: shadowed paths + static rest + body scatter.

use ndarray::Array2;
use num_complex::Complex64;

use crate::band::wavelength_of;
use crate::bodies::Body;
use crate::paths::PropagationPath;
use crate::scatter::{add_moving_scatterer, mean_present_position, MovingScatterer};
use crate::shadow::{path_shadow, ShadowCasters};
use crate::vec3::Vec3;

/// The link endpoints and the frequencies it is observed at.
#[derive(Clone, Copy, Debug)]
pub struct Link<'a> {
    pub tx: Vec3,
    pub rx: Vec3,
    pub frequencies: &'a [f64],
}

impl Link<'_> {
    /// The wavelength of the carrier the subcarrier grid is centred on.
    pub fn wavelength(&self) -> f64 {
        wavelength_of(self.frequencies)
    }
}

/// Where the chest and limb spectra of one body are evaluated (`None` = never present).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScatterReference {
    pub chest: Option<Vec3>,
    pub limb: Option<Vec3>,
}

impl ScatterReference {
    /// The mean present positions over the body's whole track.
    pub fn of(body: &Body) -> Self {
        ScatterReference {
            chest: mean_present_position(&body.chest),
            limb: mean_present_position(&body.limb),
        }
    }
}

/// `H[t,k] = absorb·rest[k] + Σ_paths F_p(t) h_p[k] + absorb·(chest + limb scatter)`.
///
/// `absorb` is the Sabine factor by which bodies weaken the reverberant field (1 = no bodies).
pub fn link_channel(
    link: Link,
    paths: &[PropagationPath],
    rest: &[Complex64],
    bodies: &[Body],
    frames: usize,
    absorb: f64,
) -> Array2<Complex64> {
    let references: Vec<ScatterReference> = bodies.iter().map(ScatterReference::of).collect();
    link_channel_with_references(link, paths, rest, bodies, &references, frames, absorb)
}

/// `link_channel` for a time slice of a session: `references` come from the bodies' whole tracks,
/// so slices concatenate to exactly the channel of the whole session.
pub fn link_channel_with_references(
    link: Link,
    paths: &[PropagationPath],
    rest: &[Complex64],
    bodies: &[Body],
    references: &[ScatterReference],
    frames: usize,
    absorb: f64,
) -> Array2<Complex64> {
    let responses: Vec<Vec<Complex64>> =
        paths.iter().map(|p| p.response(link.frequencies)).collect();
    let mut unblocked: Vec<Complex64> = rest.iter().map(|r| r * absorb).collect();
    for response in &responses {
        for (value, h) in unblocked.iter_mut().zip(response) {
            *value += h;
        }
    }
    let mut channel = Array2::from_shape_fn((frames, unblocked.len()), |(_, k)| unblocked[k]);
    let casters = ShadowCasters::with_wavelength(bodies, link.wavelength());
    if !casters.is_empty() {
        for (path, response) in paths.iter().zip(&responses).filter(|(p, _)| p.shadowable) {
            add_shadow_change(&mut channel, path, response, &casters);
        }
    }
    for (body, reference) in bodies.iter().zip(references) {
        let parts = [
            (&body.chest, reference.chest, body.chest_rcs),
            (&body.limb, reference.limb, body.limb_rcs),
        ];
        for (track, reference, rcs) in parts {
            let Some(reference) = reference else {
                continue;
            };
            let scatterer = MovingScatterer {
                track,
                reference,
                rcs,
            };
            add_moving_scatterer(
                &mut channel,
                link.tx,
                link.rx,
                &scatterer,
                link.frequencies,
                absorb,
            );
        }
    }
    channel
}

/// `H[t] += (F(t) - 1) h`, only at frames where a body changes the path (most frames it does not).
fn add_shadow_change(
    channel: &mut Array2<Complex64>,
    path: &PropagationPath,
    response: &[Complex64],
    casters: &ShadowCasters,
) {
    let shadow = path_shadow(&path.points, casters, path.shadow_scale);
    for (t, factor) in shadow.into_iter().enumerate() {
        let change = factor - 1.0;
        if change == Complex64::new(0.0, 0.0) {
            continue;
        }
        for (value, h) in channel.row_mut(t).iter_mut().zip(response) {
            *value += change * h;
        }
    }
}

/// The channel with nobody present, a single spectrum `[k]`.
pub fn empty_channel(link: Link, paths: &[PropagationPath], rest: &[Complex64]) -> Vec<Complex64> {
    let mut spectrum = rest.to_vec();
    for path in paths {
        for (value, h) in spectrum.iter_mut().zip(path.response(link.frequencies)) {
            *value += h;
        }
    }
    spectrum
}
