use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use num_complex::Complex64;

use super::{round_like_float32, BREATH_BAND_HZ, MICRO_BAND_HZ, NOISE_BAND_HZ};
use crate::linalg::first_principal_component;
use crate::numeric::{argmax, argsort_descending, mean, median, percentile, std};
use crate::spectrum::{band_mask, hann_power_spectrum, hanning, irfft, rfft, rfft_frequencies};

/// Breathing-rate guess (breaths per minute) from the strongest line of the breathing band.
pub type BreathingRate = f64;

/// Per-subcarrier spectrum of one window: centred series, Hann-windowed FFT and power, frequencies.
struct WindowSpectrum {
    centered: Array2<f64>,
    fft: Array2<Complex64>,
    power: Array2<f64>,
    freqs: Vec<f64>,
}

impl WindowSpectrum {
    fn new(series: ArrayView2<f64>, rate_hz: f64) -> Self {
        let n = series.nrows();
        let centered = &series - &series.mean_axis(Axis(0)).unwrap();
        let window = Array1::from(hanning(n));
        let bins = n / 2 + 1;
        let mut fft = Array2::zeros((bins, series.ncols()));
        for (k, column) in centered.columns().into_iter().enumerate() {
            let windowed: Vec<f64> = column.iter().zip(&window).map(|(v, w)| v * w).collect();
            for (f, value) in rfft(&windowed, n).into_iter().enumerate() {
                fft[[f, k]] = value;
            }
        }
        let power = fft.mapv(|c: Complex64| c.norm_sqr());
        Self {
            centered,
            fft,
            power,
            freqs: rfft_frequencies(n, rate_hz),
        }
    }

    /// Mean power over the frequencies in `[lo, hi)`, per subcarrier.
    fn band_power(&self, (lo, hi): (f64, f64)) -> Vec<f64> {
        let mask = band_mask(&self.freqs, lo, hi);
        self.mean_over_rows(&mask)
    }

    fn mean_over_rows(&self, mask: &[bool]) -> Vec<f64> {
        let rows: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
        (0..self.power.ncols())
            .map(|k| rows.iter().map(|&r| self.power[[r, k]]).sum::<f64>() / rows.len() as f64)
            .collect()
    }

    fn subcarriers(&self) -> usize {
        self.power.ncols()
    }
}

fn strongest(count: usize, scores: &[f64]) -> Vec<usize> {
    argsort_descending(scores).into_iter().take(count).collect()
}

/// The best `max(3, K/8)` subcarriers carry the person; the others sit in fades or far from the body.
fn top_count(subcarriers: usize) -> usize {
    (subcarriers / 8).max(3)
}

fn log_ratio(numerator: &[f64], denominator: &[f64]) -> Vec<f64> {
    numerator
        .iter()
        .zip(denominator)
        .map(|(n, d)| (n / d).log10())
        .collect()
}

fn fraction_above(values: &[f64], threshold: f64) -> f64 {
    values.iter().filter(|&&v| v > threshold).count() as f64 / values.len() as f64
}

/// Ten band-power / periodicity features of a `[samples, subcarriers]` window on the 20 Hz grid
/// (amplitude in dB or sanitized phase in rad), plus the breathing-rate guess.
pub fn spectral_features(series: ArrayView2<f64>, rate_hz: f64) -> (Vec<f64>, BreathingRate) {
    let spectrum = WindowSpectrum::new(series, rate_hz);
    let noise: Vec<f64> = spectrum
        .band_power(NOISE_BAND_HZ)
        .into_iter()
        .map(|p| p.max(1e-9))
        .collect();
    let breath_ratio = log_ratio(&spectrum.band_power(BREATH_BAND_HZ), &noise);
    let micro_ratio = log_ratio(&spectrum.band_power(MICRO_BAND_HZ), &noise);
    let top_breath = strongest(top_count(breath_ratio.len()), &breath_ratio);
    let top_micro = strongest(top_count(micro_ratio.len()), &micro_ratio);
    let periodicity = breathing_periodicity(&spectrum, &top_breath, rate_hz);
    let spread: Vec<f64> = spectrum
        .centered
        .columns()
        .into_iter()
        .map(|column| std(&column.to_vec()))
        .collect();
    let mut features = vec![
        median(&breath_ratio),
        mean(&select(&breath_ratio, &top_breath)),
        median(&micro_ratio),
        mean(&select(&micro_ratio, &top_micro)),
        (median(&spread) + 1e-6).log10(),
        (percentile(&spread, 90.0) + 1e-6).log10(),
        periodicity.peak_share,
        periodicity.component_share,
        fraction_above(&breath_ratio, 0.5),
        fraction_above(&micro_ratio, 0.5),
    ];
    round_like_float32(&mut features);
    (features, periodicity.rate_bpm)
}

fn select(values: &[f64], indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&i| values[i]).collect()
}

struct BreathingPeriodicity {
    peak_share: f64,
    component_share: f64,
    rate_bpm: f64,
}

/// Breathing-band-filtered top subcarriers -> first principal component -> how concentrated its spectrum is.
fn breathing_periodicity(
    spectrum: &WindowSpectrum,
    top: &[usize],
    rate_hz: f64,
) -> BreathingPeriodicity {
    let n = spectrum.centered.nrows();
    let in_band = band_mask(&spectrum.freqs, BREATH_BAND_HZ.0, BREATH_BAND_HZ.1);
    let mut filtered = Array2::zeros((n, top.len()));
    for (j, &k) in top.iter().enumerate() {
        let half: Vec<Complex64> = (0..spectrum.fft.nrows())
            .map(|f| {
                if in_band[f] {
                    spectrum.fft[[f, k]]
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();
        filtered
            .column_mut(j)
            .assign(&Array1::from(irfft(&half, n)));
    }
    let centered = &filtered - &filtered.mean_axis(Axis(0)).unwrap();
    let (component, component_share) = first_principal_component(centered.view());
    let power = hann_power_spectrum(component.as_slice().unwrap(), 4 * n);
    let freqs = rfft_frequencies(4 * n, rate_hz);
    let band: Vec<usize> = (0..freqs.len())
        .filter(|&i| freqs[i] >= BREATH_BAND_HZ.0 && freqs[i] < BREATH_BAND_HZ.1)
        .collect();
    let band_power: Vec<f64> = band.iter().map(|&i| power[i]).collect();
    let strongest = argmax(&band_power);
    BreathingPeriodicity {
        peak_share: band_power[strongest] / (band_power.iter().sum::<f64>() + 1e-12),
        component_share,
        rate_bpm: freqs[band[strongest]] * 60.0,
    }
}

/// Amplitude window features: `spectral_features` of the amplitude in dB.
pub fn window_features(
    linear_amplitude: ArrayView2<f64>,
    rate_hz: f64,
) -> (Vec<f64>, BreathingRate) {
    spectral_features(to_decibels(linear_amplitude).view(), rate_hz)
}

fn to_decibels(linear_amplitude: ArrayView2<f64>) -> Array2<f64> {
    linear_amplitude.mapv(|a| 20.0 * a.max(1e-3).log10())
}

/// What kind of thing moved: fans are spectral lines, doors are steps, people are broadband bursts
/// plus breathing, pets and vacuums are weak and touch few subcarriers.
pub fn extra_features(linear_amplitude: ArrayView2<f64>, rate_hz: f64) -> Vec<f64> {
    let spectrum = WindowSpectrum::new(to_decibels(linear_amplitude).view(), rate_hz);
    let noise: Vec<f64> = spectrum
        .band_power(NOISE_BAND_HZ)
        .iter()
        .map(|p| p + 1e-12)
        .collect();
    let activity: Vec<f64> = spectrum
        .band_power((BREATH_BAND_HZ.0, MICRO_BAND_HZ.1))
        .iter()
        .zip(&noise)
        .map(|(p, n)| p / n)
        .collect();
    let active = strongest(top_count(spectrum.subcarriers()), &activity);
    let active_spectrum = mean_normalised_spectrum(&spectrum, &active, &noise);
    let (line, flatness) = line_and_flatness(&spectrum.freqs, &active_spectrum);
    let mut features = vec![
        line,
        flatness,
        step_size(&spectrum.centered),
        burstiness(&spectrum.centered, &active, rate_hz),
        fraction_above(&activity, 3.0),
        breathing_glrt(&spectrum),
    ];
    round_like_float32(&mut features);
    features
}

fn mean_normalised_spectrum(
    spectrum: &WindowSpectrum,
    active: &[usize],
    noise: &[f64],
) -> Vec<f64> {
    spectrum
        .power
        .rows()
        .into_iter()
        .map(|row| active.iter().map(|&k| row[k] / noise[k]).sum::<f64>() / active.len() as f64)
        .collect()
}

fn line_and_flatness(freqs: &[f64], active_spectrum: &[f64]) -> (f64, f64) {
    let band: Vec<f64> = freqs
        .iter()
        .zip(active_spectrum)
        .filter(|(&f, _)| (0.5..9.8).contains(&f))
        .map(|(_, &p)| p)
        .collect();
    let peak = band.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let line = (peak / median(&band)).log10();
    let log_mean = mean(&band.iter().map(|p| (p + 1e-12).ln()).collect::<Vec<_>>());
    let flatness = log_mean.exp() / (mean(&band) + 1e-12);
    (line, flatness)
}

/// Median over subcarriers of the jump between the window's two halves, in units of their spread.
fn step_size(centered: &Array2<f64>) -> f64 {
    let half = centered.nrows() / 2;
    let steps: Vec<f64> = centered
        .columns()
        .into_iter()
        .map(|column| {
            let first = column.slice(s![..half]).to_vec();
            let second = column.slice(s![half..]).to_vec();
            (mean(&first) - mean(&second)).abs() / (0.5 * (std(&first) + std(&second)) + 1e-6)
        })
        .collect();
    median(&steps)
}

/// Peak-to-median ratio of the 1-s smoothed sample-to-sample energy on the active subcarriers.
fn burstiness(centered: &Array2<f64>, active: &[usize], rate_hz: f64) -> f64 {
    let energy: Vec<f64> = (1..centered.nrows())
        .map(|i| {
            active
                .iter()
                .map(|&k| (centered[[i, k]] - centered[[i - 1, k]]).powi(2))
                .sum::<f64>()
                / active.len() as f64
        })
        .collect();
    let smoothed = moving_average_same(&energy, rate_hz as usize, rate_hz);
    (percentile(&smoothed, 95.0) / (median(&smoothed) + 1e-9) + 1e-9).log10()
}

/// `np.convolve(x, np.ones(taps) / scale, "same")`.
fn moving_average_same(x: &[f64], taps: usize, scale: f64) -> Vec<f64> {
    let n = x.len();
    let out_len = n.max(taps);
    let offset = (taps - 1) / 2;
    (0..out_len)
        .map(|i| {
            let full_index = i + offset;
            (0..taps)
                .filter_map(|j| full_index.checked_sub(j).and_then(|src| x.get(src)))
                .map(|v| v / scale)
                .sum()
        })
        .collect()
}

/// Generalised likelihood ratio for one breathing line shared by all subcarriers (Patwari-style ML combining).
fn breathing_glrt(spectrum: &WindowSpectrum) -> f64 {
    let reference: Vec<f64> = (0..spectrum.subcarriers())
        .map(|k| {
            let micro: Vec<f64> = (0..spectrum.freqs.len())
                .filter(|&f| spectrum.freqs[f] >= 1.0 && spectrum.freqs[f] < 3.0)
                .map(|f| spectrum.power[[f, k]])
                .collect();
            median(&micro) + 1e-12
        })
        .collect();
    let combined: Vec<f64> = (0..spectrum.freqs.len())
        .filter(|&f| spectrum.freqs[f] >= 0.13 && spectrum.freqs[f] <= 0.6)
        .map(|f| {
            (0..spectrum.subcarriers())
                .map(|k| spectrum.power[[f, k]] / reference[k])
                .sum()
        })
        .collect();
    let peak = combined.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (peak / spectrum.subcarriers() as f64 + 1e-9).ln()
}
