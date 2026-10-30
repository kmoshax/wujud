//! Fresnel-Kirchhoff diffraction of a path segment by rectangular absorbing bodies.

use num_complex::Complex64;

use crate::bodies::Body;
use crate::constants::WAVELENGTH;
use crate::fresnel::{fresnel_scale_at, fresnel_strip_integral};
use crate::vec3::{cross, dot, norm, scale, sub, Vec3};

/// Bodies farther than this many Fresnel units from a segment are skipped (physics3d.py prune).
pub const PRUNE_FRESNEL_UNITS: f64 = 3.5;
/// A body closer than this to either end of a segment is treated as not on it (m).
const END_MARGIN: f64 = 0.05;

/// A straight piece of a path: `start -> end`, beginning `offset` metres after the transmitter
/// on a path of `path_length` metres.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub start: Vec3,
    pub end: Vec3,
    pub offset: f64,
    pub path_length: f64,
}

struct SegmentFrame {
    length: f64,
    along: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

fn segment_frame(segment: &Segment) -> Option<SegmentFrame> {
    let direction = sub(segment.end, segment.start);
    let length = norm(direction);
    if length < 1e-6 {
        return None;
    }
    let along = scale(direction, 1.0 / length);
    let horizontal = cross(along, [0.0, 0.0, 1.0]);
    let horizontal = scale(horizontal, 1.0 / (norm(horizontal) + 1e-12));
    let vertical = cross(horizontal, along);
    Some(SegmentFrame {
        length,
        along,
        horizontal,
        vertical,
    })
}

/// One frame of a body box: centre (NaN when absent), heading and half sizes.
struct CasterFrame {
    centre: Vec3,
    cos_yaw: f64,
    sin_yaw: f64,
    half_width: f64,
    half_depth: f64,
    half_height: f64,
}

impl CasterFrame {
    /// Upper bound of the box's half extent seen from any direction (|a| + |b| + half height).
    fn reach(&self) -> f64 {
        self.half_width.abs() + self.half_depth.abs() + self.half_height.abs()
    }
}

/// A sphere containing every present frame's box centre, plus the largest box reach.
struct TrackBounds {
    centre: Vec3,
    centre_spread: f64,
    box_reach: f64,
}

struct CasterTrack {
    frames: Vec<CasterFrame>,
    absorption: f64,
    bounds: Option<TrackBounds>,
}

/// The bodies of a scene prepared for shadowing many path segments (per-frame trigonometry done once).
pub struct ShadowCasters {
    tracks: Vec<CasterTrack>,
    frames: usize,
    wavelength: f64,
}

impl ShadowCasters {
    /// Bodies seen at the default 2.4 GHz carrier.
    pub fn new(bodies: &[Body]) -> Self {
        ShadowCasters::with_wavelength(bodies, WAVELENGTH)
    }

    pub fn with_wavelength(bodies: &[Body], wavelength: f64) -> Self {
        let frames = bodies.first().map_or(0, Body::frames);
        ShadowCasters {
            tracks: bodies.iter().map(caster_track).collect(),
            frames,
            wavelength,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }
}

fn caster_track(body: &Body) -> CasterTrack {
    let frames: Vec<CasterFrame> = (0..body.frames())
        .map(|t| {
            let [x, y] = body.centre[t];
            let (sin_yaw, cos_yaw) = body.yaw[t].sin_cos();
            CasterFrame {
                centre: [x, y, 0.5 * (body.z_bottom[t] + body.z_top[t])],
                cos_yaw,
                sin_yaw,
                half_width: body.half_width[t],
                half_depth: body.half_depth[t],
                half_height: 0.5 * (body.z_top[t] - body.z_bottom[t]),
            }
        })
        .collect();
    let bounds = track_bounds(&frames);
    CasterTrack {
        frames,
        absorption: body.absorption,
        bounds,
    }
}

fn track_bounds(frames: &[CasterFrame]) -> Option<TrackBounds> {
    let present: Vec<&CasterFrame> = frames.iter().filter(|f| f.centre[0].is_finite()).collect();
    if present.is_empty() {
        return None;
    }
    let sum = present.iter().fold([0.0; 3], |s, f| {
        [s[0] + f.centre[0], s[1] + f.centre[1], s[2] + f.centre[2]]
    });
    let centre = scale(sum, 1.0 / present.len() as f64);
    Some(TrackBounds {
        centre,
        centre_spread: present
            .iter()
            .map(|f| norm(sub(f.centre, centre)))
            .fold(0.0, f64::max),
        box_reach: present.iter().map(|f| f.reach()).fold(0.0, f64::max),
    })
}

/// True when no frame of the track can come within `PRUNE_FRESNEL_UNITS` of the segment, so the
/// per-frame test would skip every frame anyway: `gap ≥ d⊥/√2 - reach` and `k_F ≥ sqrt(8 / (λ L))`.
fn always_pruned(
    bounds: &TrackBounds,
    segment: &Segment,
    frame: &SegmentFrame,
    wavelength: f64,
) -> bool {
    let not_vertical = frame.horizontal[0].hypot(frame.horizontal[1]) > 0.5;
    if !not_vertical {
        return false;
    }
    let relative = sub(bounds.centre, segment.start);
    let across = sub(relative, scale(frame.along, dot(relative, frame.along)));
    let closest = norm(across) - bounds.centre_spread;
    let gap_lower_bound = closest / std::f64::consts::SQRT_2 - bounds.box_reach;
    let smallest_scale = (8.0 / (wavelength * segment.path_length)).sqrt();
    gap_lower_bound * smallest_scale >= PRUNE_FRESNEL_UNITS + 1e-9
}

/// Where a body box sits relative to a segment: distance along it, offsets across it, and the
/// half sizes of the box projected onto the plane perpendicular to it.
struct BoxProjection {
    along: f64,
    horizontal_offset: f64,
    vertical_offset: f64,
    half_horizontal: f64,
    half_vertical: f64,
}

fn project_box(body: &CasterFrame, start: Vec3, frame: &SegmentFrame) -> BoxProjection {
    let relative = sub(body.centre, start);
    let body_x = scale([body.cos_yaw, body.sin_yaw, 0.0], body.half_width);
    let body_y = scale([-body.sin_yaw, body.cos_yaw, 0.0], body.half_depth);
    let footprint = |axis: Vec3| dot(body_x, axis).hypot(dot(body_y, axis));
    BoxProjection {
        along: dot(relative, frame.along),
        horizontal_offset: dot(relative, frame.horizontal),
        vertical_offset: dot(relative, frame.vertical),
        half_horizontal: footprint(frame.horizontal)
            + (body.half_height * frame.horizontal[2]).abs(),
        half_vertical: (body.half_height * frame.vertical[2]).abs() + footprint(frame.vertical),
    }
}

/// `E/E0 = 1 - η (j/2) I(u1,u2) I(v1,v2)` for a box seen through the segment, or `None` when the
/// body is absent, off the segment, or pruned.
fn box_diffraction(
    projection: &BoxProjection,
    segment: &Segment,
    segment_length: f64,
    absorption: f64,
    wavelength: f64,
) -> Option<Complex64> {
    let p = projection;
    let on_segment =
        p.along.is_finite() && p.along > END_MARGIN && p.along < segment_length - END_MARGIN;
    if !on_segment {
        return None;
    }
    let k = fresnel_scale_at(wavelength, segment.offset + p.along, segment.path_length);
    let gap = (p.horizontal_offset.abs() - p.half_horizontal)
        .max(p.vertical_offset.abs() - p.half_vertical)
        * k;
    if gap >= PRUNE_FRESNEL_UNITS {
        return None;
    }
    let across = fresnel_strip_integral(
        (p.horizontal_offset - p.half_horizontal) * k,
        (p.horizontal_offset + p.half_horizontal) * k,
    );
    let up = fresnel_strip_integral(
        (p.vertical_offset - p.half_vertical) * k,
        (p.vertical_offset + p.half_vertical) * k,
    );
    Some(Complex64::new(1.0, 0.0) - Complex64::new(0.0, 0.5 * absorption) * across * up)
}

/// Multiplies `field[t]` by the diffraction factor of every body on this segment.
pub fn apply_segment_shadow(
    field: &mut [Complex64],
    segment: &Segment,
    casters: &ShadowCasters,
    shadow_scale: f64,
) {
    let Some(frame) = segment_frame(segment) else {
        return;
    };
    for track in &casters.tracks {
        let Some(bounds) = &track.bounds else {
            continue;
        };
        if always_pruned(bounds, segment, &frame, casters.wavelength) {
            continue;
        }
        let absorption = track.absorption * shadow_scale;
        for (value, body) in field.iter_mut().zip(&track.frames) {
            let projection = project_box(body, segment.start, &frame);
            let factor = box_diffraction(
                &projection,
                segment,
                frame.length,
                absorption,
                casters.wavelength,
            );
            if let Some(factor) = factor {
                *value *= factor;
            }
        }
    }
}

/// The product of all segment factors of a path, per frame (1 where nothing blocks).
pub fn path_shadow(points: &[Vec3], casters: &ShadowCasters, shadow_scale: f64) -> Vec<Complex64> {
    let mut field = vec![Complex64::new(1.0, 0.0); casters.frames];
    let path_length: f64 = points.windows(2).map(|s| norm(sub(s[1], s[0]))).sum();
    let mut offset = 0.0;
    for pair in points.windows(2) {
        let segment = Segment {
            start: pair[0],
            end: pair[1],
            offset,
            path_length,
        };
        apply_segment_shadow(&mut field, &segment, casters, shadow_scale);
        offset += norm(sub(pair[1], pair[0]));
    }
    field
}
