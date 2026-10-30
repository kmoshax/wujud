//! Fresnel integrals and the Fresnel-zone scale used by body diffraction.

use num_complex::Complex64;
use std::f64::consts::{FRAC_PI_2, PI};

use crate::constants::WAVELENGTH;
use crate::fresnel_tables::{AuxiliaryTable, TABLES};

const SERIES_LIMIT: f64 = 1.5;
const ASYMPTOTIC_FROM: f64 = 6.0;
const MAX_ITERATIONS: usize = 200;
const SERIES_TOLERANCE: f64 = 1e-16;
const TINY: f64 = 1e-300;

/// `(S(x), C(x))` with `S(x) = ∫₀ˣ sin(πt²/2) dt`, `C(x) = ∫₀ˣ cos(πt²/2) dt` (scipy's `fresnel`).
pub fn fresnel_integrals(x: f64) -> (f64, f64) {
    let magnitude = x.abs();
    let (s, c) = if magnitude < TINY.sqrt() {
        (0.0, magnitude)
    } else if magnitude <= SERIES_LIMIT {
        power_series(magnitude)
    } else {
        from_auxiliary(magnitude, auxiliary_functions(magnitude))
    };
    if x < 0.0 {
        (-s, -c)
    } else {
        (s, c)
    }
}

// Numerical Recipes `frenel`: the alternating series for small x.
fn power_series(x: f64) -> (f64, f64) {
    let factor = FRAC_PI_2 * x * x;
    let mut sum_sin = 0.0;
    let mut sum_cos = x;
    let mut sum = 0.0;
    let mut sign = 1.0;
    let mut term = x;
    let mut odd = true;
    let mut denominator = 3.0;
    for k in 1..=MAX_ITERATIONS {
        term *= factor / k as f64;
        sum += sign * term / denominator;
        let test = sum.abs() * SERIES_TOLERANCE;
        if odd {
            sign = -sign;
            sum_sin = sum;
            sum = sum_cos;
        } else {
            sum_cos = sum;
            sum = sum_sin;
        }
        if term < test {
            break;
        }
        odd = !odd;
        denominator += 2.0;
    }
    (sum_sin, sum_cos)
}

/// `C = 1/2 + f sin θ - g cos θ`, `S = 1/2 - f cos θ - g sin θ` with `θ = πx²/2`.
fn from_auxiliary(x: f64, (f, g): (f64, f64)) -> (f64, f64) {
    let (sin, cos) = (FRAC_PI_2 * x * x).sin_cos();
    (0.5 - f * cos - g * sin, 0.5 + f * sin - g * cos)
}

fn auxiliary_functions(x: f64) -> (f64, f64) {
    match TABLES.iter().find(|table| x <= table.to) {
        Some(table) => chebyshev_auxiliary(table, x),
        None => asymptotic_auxiliary(x),
    }
}

fn chebyshev_auxiliary(table: &AuxiliaryTable, x: f64) -> (f64, f64) {
    let u = 2.0 * (x - table.from) / (table.to - table.from) - 1.0;
    (clenshaw(&table.f, u), clenshaw(&table.g, u))
}

/// `Σ c_n T_n(u)`.
fn clenshaw(coefficients: &[f64], u: f64) -> f64 {
    let (mut b1, mut b2) = (0.0, 0.0);
    for &c in coefficients[1..].iter().rev() {
        (b1, b2) = (2.0 * u * b1 - b2 + c, b1);
    }
    coefficients[0] + u * b1 - b2
}

/// `f ~ 1/(πx) Σ (-1)ⁿ (4n-1)!!/(πx²)²ⁿ`, `g ~ 1/(π²x³) Σ (-1)ⁿ (4n+1)!!/(πx²)²ⁿ`, summed while terms shrink.
fn asymptotic_auxiliary(x: f64) -> (f64, f64) {
    debug_assert!(x >= ASYMPTOTIC_FROM);
    let inverse_square = 1.0 / (PI * x * x).powi(2);
    let (mut f_term, mut g_term) = (1.0_f64, 1.0_f64);
    let (mut f_sum, mut g_sum) = (1.0_f64, 1.0_f64);
    for n in 1..MAX_ITERATIONS {
        let m = 4.0 * n as f64;
        let next_f = -f_term * (m - 3.0) * (m - 1.0) * inverse_square;
        let next_g = -g_term * (m - 1.0) * (m + 1.0) * inverse_square;
        if next_g.abs() >= g_term.abs() || next_g.abs() < 1e-17 * g_sum.abs() {
            break;
        }
        (f_term, g_term) = (next_f, next_g);
        f_sum += f_term;
        g_sum += g_term;
    }
    (f_sum / (PI * x), g_sum / (PI * PI * x * x * x))
}

/// `I(α, β) = ∫_α^β exp(-jπt²/2) dt`.
pub fn fresnel_strip_integral(from: f64, to: f64) -> Complex64 {
    let (s1, c1) = fresnel_integrals(from);
    let (s2, c2) = fresnel_integrals(to);
    Complex64::new(c2 - c1, -(s2 - s1))
}

/// Converts a transverse offset (m) into Fresnel units at distance `from_tx` along a path of
/// length `total`: `k_F = sqrt(2L / (λ d1 d2))`, i.e. `sqrt(2) / r1`.
pub fn fresnel_scale(from_tx: f64, total: f64) -> f64 {
    fresnel_scale_at(WAVELENGTH, from_tx, total)
}

pub fn fresnel_scale_at(wavelength: f64, from_tx: f64, total: f64) -> f64 {
    let to_rx = (total - from_tx).max(1e-3);
    (2.0 * total / (wavelength * from_tx * to_rx)).sqrt()
}

/// Radius of the n-th Fresnel zone, `sqrt(n λ d1 d2 / (d1 + d2))`.
pub fn fresnel_zone_radius(zone: u32, from_tx: f64, to_rx: f64) -> f64 {
    fresnel_zone_radius_at(WAVELENGTH, zone, from_tx, to_rx)
}

pub fn fresnel_zone_radius_at(wavelength: f64, zone: u32, from_tx: f64, to_rx: f64) -> f64 {
    (f64::from(zone) * wavelength * from_tx * to_rx / (from_tx + to_rx)).sqrt()
}
