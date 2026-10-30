//! Thin wrappers that give numpy's distribution signatures to a `ChaCha8Rng`.

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Exp1, Poisson, StandardNormal, WeightedIndex};

pub type SimRng = ChaCha8Rng;

pub fn uniform(rng: &mut SimRng, low: f64, high: f64) -> f64 {
    low + (high - low) * rng.gen::<f64>()
}

pub fn unit(rng: &mut SimRng) -> f64 {
    rng.gen::<f64>()
}

pub fn chance(rng: &mut SimRng, probability: f64) -> bool {
    rng.gen::<f64>() < probability
}

pub fn normal(rng: &mut SimRng, mean: f64, std_dev: f64) -> f64 {
    let z: f64 = StandardNormal.sample(rng);
    mean + std_dev * z
}

pub fn standard_normals(rng: &mut SimRng, count: usize) -> Vec<f64> {
    (0..count).map(|_| StandardNormal.sample(rng)).collect()
}

pub fn exponential(rng: &mut SimRng, mean: f64) -> f64 {
    let e: f64 = Exp1.sample(rng);
    mean * e
}

pub fn poisson(rng: &mut SimRng, mean: f64) -> usize {
    if mean <= 0.0 {
        return 0;
    }
    let count: f64 = Poisson::new(mean)
        .expect("positive finite mean")
        .sample(rng);
    count as usize
}

/// Integer in `low..high` (numpy `integers`, high exclusive).
pub fn integer(rng: &mut SimRng, low: usize, high: usize) -> usize {
    rng.gen_range(low..high)
}

pub fn weighted_index(rng: &mut SimRng, weights: &[f64]) -> usize {
    WeightedIndex::new(weights)
        .expect("non-negative weights")
        .sample(rng)
}

pub fn pick<T: Copy>(rng: &mut SimRng, items: &[T]) -> T {
    items[rng.gen_range(0..items.len())]
}

/// Ornstein-Uhlenbeck (AR(1)) series with stationary std `sigma` and correlation time `tau_s`.
pub fn ornstein_uhlenbeck(
    rng: &mut SimRng,
    samples: usize,
    sigma: f64,
    tau_s: f64,
    fs: f64,
) -> Vec<f64> {
    let rho = (-1.0 / (tau_s * fs)).exp();
    let innovation_std = sigma * (1.0 - rho * rho).sqrt();
    let mut state = 0.0;
    (0..samples)
        .map(|_| {
            state = rho * state + normal(rng, 0.0, innovation_std);
            state
        })
        .collect()
}

/// Two independent OU series, as `[x, y]` per sample.
pub fn ornstein_uhlenbeck_2d(
    rng: &mut SimRng,
    samples: usize,
    sigma: f64,
    tau_s: f64,
    fs: f64,
) -> Vec<[f64; 2]> {
    let x = ornstein_uhlenbeck(rng, samples, sigma, tau_s, fs);
    let y = ornstein_uhlenbeck(rng, samples, sigma, tau_s, fs);
    x.into_iter().zip(y).map(|(x, y)| [x, y]).collect()
}

/// The generator for `(seed, stream)`, like numpy's `default_rng([seed, stream])`: independent streams of one seed.
pub fn seeded(seed: u64, stream: u64) -> SimRng {
    let mut rng = <SimRng as rand::SeedableRng>::seed_from_u64(seed);
    rng.set_stream(stream);
    rng
}
