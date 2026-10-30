//! Moving bodies (people, pets, fans, doors) as boxes that block paths plus two point scatterers.

use crate::vec3::Vec3;

/// One body over `frames()` time steps. A NaN `centre` means the body is absent at that frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    pub centre: Vec<[f64; 2]>,
    pub z_bottom: Vec<f64>,
    pub z_top: Vec<f64>,
    /// Half size along the body's own x axis (shoulders, or half the length when lying).
    pub half_width: Vec<f64>,
    /// Half size along the body's own y axis (chest depth, modulated by breathing).
    pub half_depth: Vec<f64>,
    pub yaw: Vec<f64>,
    /// 0..1, how much of the field the box absorbs.
    pub absorption: f64,
    pub chest: Vec<Vec3>,
    pub limb: Vec<Vec3>,
    /// Radar cross-sections in m².
    pub chest_rcs: f64,
    pub limb_rcs: f64,
}

impl Body {
    pub fn frames(&self) -> usize {
        self.centre.len()
    }

    pub fn is_present(&self, frame: usize) -> bool {
        self.centre[frame][0].is_finite()
    }

    /// The same body restricted to `frames` (for second-by-second streaming).
    pub fn slice_frames(&self, frames: std::ops::Range<usize>) -> Body {
        Body {
            centre: self.centre[frames.clone()].to_vec(),
            z_bottom: self.z_bottom[frames.clone()].to_vec(),
            z_top: self.z_top[frames.clone()].to_vec(),
            half_width: self.half_width[frames.clone()].to_vec(),
            half_depth: self.half_depth[frames.clone()].to_vec(),
            yaw: self.yaw[frames.clone()].to_vec(),
            absorption: self.absorption,
            chest: self.chest[frames.clone()].to_vec(),
            limb: self.limb[frames].to_vec(),
            chest_rcs: self.chest_rcs,
            limb_rcs: self.limb_rcs,
        }
    }

    /// The same body at the listed frames only (e.g. the packets sent on one hopping channel).
    pub fn select_frames(&self, frames: &[usize]) -> Body {
        let pick = |values: &[f64]| frames.iter().map(|&t| values[t]).collect::<Vec<f64>>();
        Body {
            centre: frames.iter().map(|&t| self.centre[t]).collect(),
            z_bottom: pick(&self.z_bottom),
            z_top: pick(&self.z_top),
            half_width: pick(&self.half_width),
            half_depth: pick(&self.half_depth),
            yaw: pick(&self.yaw),
            absorption: self.absorption,
            chest: frames.iter().map(|&t| self.chest[t]).collect(),
            limb: frames.iter().map(|&t| self.limb[t]).collect(),
            chest_rcs: self.chest_rcs,
            limb_rcs: self.limb_rcs,
        }
    }

    /// Marks the body absent (NaN centre, chest and limb) for `frames`.
    pub fn hide(&mut self, frames: std::ops::Range<usize>) {
        for t in frames {
            self.centre[t] = [f64::NAN; 2];
            self.chest[t] = [f64::NAN; 3];
            self.limb[t] = [f64::NAN; 3];
        }
    }
}
