//! Real FFT helpers with NumPy's conventions (`np.fft.rfft`, `irfft`, `rfftfreq`, `np.hanning`).

use num_complex::Complex64;
use rustfft::FftPlanner;

/// `np.hanning(n)`: symmetric Hann window.
pub fn hanning(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|k| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / (n - 1) as f64).cos())
        .collect()
}

/// `np.fft.rfftfreq(n, 1 / sample_rate)`.
pub fn rfft_frequencies(n: usize, sample_rate: f64) -> Vec<f64> {
    let spacing = 1.0 / (n as f64 * (1.0 / sample_rate));
    (0..n / 2 + 1).map(|k| k as f64 * spacing).collect()
}

/// `np.fft.rfft(x, n=padded_len)`: zero-pads (or truncates) to `padded_len`.
pub fn rfft(x: &[f64], padded_len: usize) -> Vec<Complex64> {
    let mut buffer: Vec<Complex64> = (0..padded_len)
        .map(|i| Complex64::new(x.get(i).copied().unwrap_or(0.0), 0.0))
        .collect();
    FftPlanner::new()
        .plan_fft_forward(padded_len)
        .process(&mut buffer);
    buffer.truncate(padded_len / 2 + 1);
    buffer
}

/// `np.fft.irfft(spectrum, n)`: the real signal of length `n` whose half spectrum is given.
pub fn irfft(half_spectrum: &[Complex64], n: usize) -> Vec<f64> {
    let mut buffer = vec![Complex64::new(0.0, 0.0); n];
    for (k, value) in half_spectrum.iter().enumerate().take(n / 2 + 1) {
        buffer[k] = *value;
        if k > 0 && n - k != k {
            buffer[n - k] = value.conj();
        }
    }
    buffer[0].im = 0.0;
    if n.is_multiple_of(2) {
        buffer[n / 2].im = 0.0;
    }
    FftPlanner::new().plan_fft_inverse(n).process(&mut buffer);
    buffer.iter().map(|c| c.re / n as f64).collect()
}

/// `|rfft(x * hanning, n=padded_len)|^2`.
pub fn hann_power_spectrum(x: &[f64], padded_len: usize) -> Vec<f64> {
    let window = hanning(x.len());
    let windowed: Vec<f64> = x.iter().zip(&window).map(|(v, w)| v * w).collect();
    rfft(&windowed, padded_len)
        .iter()
        .map(|c| c.norm_sqr())
        .collect()
}

/// Boolean mask `lo <= f < hi`.
pub fn band_mask(freqs: &[f64], lo: f64, hi: f64) -> Vec<bool> {
    freqs.iter().map(|&f| f >= lo && f < hi).collect()
}
