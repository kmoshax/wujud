//! NumPy-equivalent statistics, so ports reproduce the Python numbers.

pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Population standard deviation (NumPy's default `ddof=0`).
pub fn std(values: &[f64]) -> f64 {
    let m = mean(values);
    let squared: Vec<f64> = values.iter().map(|v| (v - m) * (v - m)).collect();
    mean(&squared).sqrt()
}

/// `np.median`: the two middle values are averaged as `(a + b) / 2`, not interpolated.
pub fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    match n {
        0 => f64::NAN,
        _ if n % 2 == 1 => sorted[n / 2],
        _ => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
    }
}

/// `np.percentile` with the default linear interpolation; `q` in 0..=100.
pub fn percentile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    percentile_of_sorted(&sorted, q)
}

pub fn percentile_of_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let position = q / 100.0 * (n - 1) as f64;
    let below = position.floor() as usize;
    let above = (below + 1).min(n - 1);
    let fraction = position - below as f64;
    lerp(sorted[below], sorted[above], fraction)
}

/// NumPy's `_lerp`: exact at both ends.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

pub fn finite_values(values: &[f64]) -> Vec<f64> {
    values.iter().copied().filter(|v| !v.is_nan()).collect()
}

/// `np.nanmean`: NaN when every value is NaN.
pub fn nan_mean(values: &[f64]) -> f64 {
    let finite = finite_values(values);
    if finite.is_empty() {
        f64::NAN
    } else {
        mean(&finite)
    }
}

pub fn nan_std(values: &[f64]) -> f64 {
    let finite = finite_values(values);
    if finite.is_empty() {
        f64::NAN
    } else {
        std(&finite)
    }
}

pub fn nan_median(values: &[f64]) -> f64 {
    median(&finite_values(values))
}

pub fn nan_percentile(values: &[f64], q: f64) -> f64 {
    percentile(&finite_values(values), q)
}

pub fn nan_min(values: &[f64]) -> f64 {
    finite_values(values).into_iter().fold(f64::NAN, f64::min)
}

pub fn nan_max(values: &[f64]) -> f64 {
    finite_values(values).into_iter().fold(f64::NAN, f64::max)
}

/// `np.nansum`: NaN counts as zero.
pub fn nan_sum(values: &[f64]) -> f64 {
    finite_values(values).iter().sum()
}

/// Index of the first maximum, like `np.argmax`; NaN wins like in NumPy.
pub fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            return i;
        }
        if v > values[best] {
            best = i;
        }
    }
    best
}

/// `np.nanargmax`: first maximum ignoring NaN (0 if all NaN).
pub fn nan_argmax(values: &[f64]) -> usize {
    let mut best: Option<usize> = None;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_nan() && best.is_none_or(|b| v > values[b]) {
            best = Some(i);
        }
    }
    best.unwrap_or(0)
}

/// Indices that sort `values` from largest to smallest, ties in index order (`np.argsort(-x)`).
pub fn argsort_descending(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    order
}

/// `np.interp` at integer positions `0..len`, through the known `(position, value)` points.
pub fn interp_at_integers(len: usize, known_positions: &[f64], known_values: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    let mut j = 0;
    for i in 0..len {
        let x = i as f64;
        while j + 1 < known_positions.len() && known_positions[j + 1] <= x {
            j += 1;
        }
        out.push(interp_segment(x, j, known_positions, known_values));
    }
    out
}

fn interp_segment(x: f64, j: usize, xp: &[f64], fp: &[f64]) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    if j + 1 >= xp.len() {
        return fp[xp.len() - 1];
    }
    if xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// Python's `round(x, digits)` (half to even) for display values.
pub fn round_to(x: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    let scaled = x * scale;
    let rounded = scaled.round();
    let is_tie = (scaled - scaled.trunc()).abs() == 0.5;
    let even = if is_tie && rounded % 2.0 != 0.0 {
        rounded - scaled.signum()
    } else {
        rounded
    };
    even / scale
}

/// Unwraps a 32-bit microsecond counter that rolls over, like `np.unwrap(ts, period=2**32)`.
pub fn unwrap_u32_microseconds(timestamps: &[u32]) -> Vec<f64> {
    const PERIOD: f64 = 4_294_967_296.0;
    let mut out = Vec::with_capacity(timestamps.len());
    let mut offset = 0.0;
    for (i, &ts) in timestamps.iter().enumerate() {
        if i > 0 {
            let step = ts as f64 - timestamps[i - 1] as f64;
            if step.abs() > PERIOD / 2.0 {
                offset -= step.signum() * PERIOD;
            }
        }
        out.push(ts as f64 + offset);
    }
    out
}

fn nan_as_zero_f32(values: &[f32]) -> (Vec<f32>, usize) {
    let finite = values.iter().filter(|v| !v.is_nan()).count();
    (
        values
            .iter()
            .map(|&v| if v.is_nan() { 0.0 } else { v })
            .collect(),
        finite,
    )
}

/// `np.nanmean` of a contiguous float32 row.
pub fn nan_mean_f32(values: &[f32]) -> f32 {
    let (zeroed, count) = nan_as_zero_f32(values);
    (pairwise_sum(&zeroed) as f64 / count as f64) as f32
}

/// `np.nanstd` (ddof 0) of a contiguous float32 row.
pub fn nan_std_f32(values: &[f32]) -> f32 {
    let (zeroed, count) = nan_as_zero_f32(values);
    let mean = (pairwise_sum(&zeroed) as f64 / count as f64) as f32;
    let squared: Vec<f32> = values
        .iter()
        .map(|&v| {
            if v.is_nan() {
                0.0
            } else {
                (v - mean) * (v - mean)
            }
        })
        .collect();
    ((pairwise_sum(&squared) as f64 / count as f64) as f32).sqrt()
}

/// `np.nanmedian` of float32 values: the two middle values are averaged in float32.
pub fn nan_median_f32(values: &[f32]) -> f32 {
    let mut finite: Vec<f32> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if finite.is_empty() {
        return f32::NAN;
    }
    finite.sort_by(f32::total_cmp);
    let n = finite.len();
    if n % 2 == 1 {
        finite[n / 2]
    } else {
        (finite[n / 2 - 1] + finite[n / 2]) / 2.0
    }
}

/// NumPy's `add.reduce` of a contiguous array: pairwise, eight interleaved partial sums per block of <= 128.
pub fn pairwise_sum<T: Copy + Default + std::ops::Add<Output = T>>(values: &[T]) -> T {
    if values.len() < 8 {
        return values.iter().fold(T::default(), |acc, &v| acc + v);
    }
    if values.len() > 128 {
        let half = values.len() / 2 / 8 * 8;
        return pairwise_sum(&values[..half]) + pairwise_sum(&values[half..]);
    }
    let mut partial = [T::default(); 8];
    partial.copy_from_slice(&values[..8]);
    let full_blocks = values.len() / 8 * 8;
    for block in values[8..full_blocks].as_chunks::<8>().0 {
        for (p, &v) in partial.iter_mut().zip(block) {
            *p = *p + v;
        }
    }
    let mut sum = ((partial[0] + partial[1]) + (partial[2] + partial[3]))
        + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
    for &v in &values[full_blocks..] {
        sum = sum + v;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy_linear() {
        assert!((percentile(&[1.0, 2.0, 3.0, 4.0], 90.0) - 3.7).abs() < 1e-12);
        assert_eq!(median(&[3.0, 1.0, 2.0, 10.0]), 2.5);
    }

    #[test]
    fn unwrap_handles_rollover() {
        let t = unwrap_u32_microseconds(&[u32::MAX - 10, 5, 20]);
        assert_eq!(t[1] - t[0], 16.0);
        assert_eq!(t[2] - t[1], 15.0);
    }

    #[test]
    fn python_round_half_even() {
        assert_eq!(round_to(2.5, 0), 2.0);
        assert_eq!(round_to(3.5, 0), 4.0);
        assert_eq!(round_to(-2.5, 0), -2.0);
        assert_eq!(round_to(1.234, 2), 1.23);
    }
}
