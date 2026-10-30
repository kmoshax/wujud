use ndarray::{Array2, ArrayView2};

/// Subcarriers that carry signal in more than 90 % of packets (drops guard bands and nulls of any layout).
/// `magnitudes` is `[packets, subcarriers]`.
pub fn signal_subcarriers(magnitudes: &Array2<f32>) -> Vec<usize> {
    let packets = magnitudes.nrows() as f64;
    (0..magnitudes.ncols())
        .filter(|&k| {
            let nonzero = magnitudes.column(k).iter().filter(|&&a| a > 0.0).count();
            nonzero as f64 / packets > 0.9
        })
        .collect()
}

fn magnitudes(iq: ArrayView2<i8>) -> Array2<f32> {
    let subcarriers = iq.ncols() / 2;
    Array2::from_shape_fn((iq.nrows(), subcarriers), |(n, k)| {
        let i = iq[[n, 2 * k]] as f32;
        let q = iq[[n, 2 * k + 1]] as f32;
        (i * i + q * q).sqrt()
    })
}

/// Linear amplitude of the signal-carrying subcarriers, divided by each packet's mean amplitude.
/// `iq` is `[packets, 2 * subcarriers]` with interleaved bytes `[a0, b0, a1, b1, ...]` as the ESP32 sends them.
pub fn amplitude_from_iq(iq: ArrayView2<i8>) -> Array2<f32> {
    let all = magnitudes(iq);
    let keep = signal_subcarriers(&all);
    let mut amplitude =
        Array2::from_shape_fn((all.nrows(), keep.len()), |(n, j)| all[[n, keep[j]]]);
    for mut packet in amplitude.rows_mut() {
        // per-packet AGC removal: the gain jumps from packet to packet, the shape across subcarriers does not
        let packet_mean = numpy_float32_sum(packet.as_slice().unwrap()) / packet.len() as f32;
        let divisor = packet_mean.max(1e-6);
        packet.mapv_inplace(|a| a / divisor);
    }
    amplitude
}

/// `a[:, keep]` is not C-contiguous, so NumPy's `a.mean(1)` adds subcarriers one by one in float32 (no pairwise sum).
fn numpy_float32_sum(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |acc, v| acc + v)
}

/// Sanitized CSI phase (rad) of the signal-carrying subcarriers, `[packets, kept subcarriers]`.
/// Per packet, the receiver's timing offset (slope over subcarriers), frequency offset (constant) and a
/// half-band shift are fitted and removed; what remains is what the room did to each subcarrier.
pub fn phase_from_iq(iq: ArrayView2<i8>) -> Array2<f64> {
    let keep = signal_subcarriers(&magnitudes(iq));
    let centered_index = centered_indices(&keep);
    let mut residual = Array2::zeros((iq.nrows(), keep.len()));
    for n in 0..iq.nrows() {
        let raw: Vec<f32> = keep
            .iter()
            .map(|&k| (iq[[n, 2 * k + 1]] as f32).atan2(iq[[n, 2 * k]] as f32))
            .collect();
        let unwrapped = unwrap_float32(&raw);
        let fitted = fit_clock_offsets(&centered_index, &unwrapped);
        for (j, (&phase, &model)) in unwrapped.iter().zip(&fitted).enumerate() {
            let r = phase as f64 - model;
            residual[[n, j]] = r.sin().atan2(r.cos());
        }
    }
    residual
}

fn centered_indices(keep: &[usize]) -> Vec<f64> {
    let mean = keep.iter().sum::<usize>() as f64 / keep.len() as f64;
    keep.iter().map(|&k| k as f64 - mean).collect()
}

/// `np.unwrap` on a float32 array: all arithmetic stays in float32 like NumPy's.
fn unwrap_float32(phase: &[f32]) -> Vec<f32> {
    let period = std::f64::consts::TAU as f32;
    let half = std::f64::consts::PI as f32;
    let mut out = Vec::with_capacity(phase.len());
    let mut correction = 0.0f32;
    for (i, &p) in phase.iter().enumerate() {
        if i > 0 {
            let step = p - phase[i - 1];
            let mut wrapped = python_mod(step + half, period) - half;
            if wrapped == -half && step > 0.0 {
                wrapped = half;
            }
            if step.abs() >= half {
                correction += wrapped - step;
            }
        }
        out.push(if i == 0 { p } else { p + correction });
    }
    out
}

fn python_mod(a: f32, b: f32) -> f32 {
    let m = a % b;
    if m != 0.0 && (m < 0.0) != (b < 0.0) {
        m + b
    } else {
        m
    }
}

/// Least-squares fit of `slope * k + offset + jump * [k > 0]`, evaluated at every kept subcarrier.
fn fit_clock_offsets(centered_index: &[f64], phase: &[f32]) -> Vec<f64> {
    let basis = |k: f64| [k, 1.0, if k > 0.0 { 1.0 } else { 0.0 }];
    let mut normal = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    for (&k, &p) in centered_index.iter().zip(phase) {
        let b = basis(k);
        for r in 0..3 {
            rhs[r] += b[r] * p as f64;
            for c in 0..3 {
                normal[r][c] += b[r] * b[c];
            }
        }
    }
    let beta = solve_3x3(normal, rhs);
    centered_index
        .iter()
        .map(|&k| {
            let b = basis(k);
            b[0] * beta[0] + b[1] * beta[1] + b[2] * beta[2]
        })
        .collect()
}

fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> [f64; 3] {
    for col in 0..3 {
        let pivot = (col..3)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        let pivot_row = a[col];
        for row in col + 1..3 {
            let factor = a[row][col] / pivot_row[col];
            for (value, pivot) in a[row].iter_mut().zip(pivot_row).skip(col) {
                *value -= factor * pivot;
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let known: f64 = (row + 1..3).map(|c| a[row][c] * x[c]).sum();
        x[row] = (b[row] - known) / a[row][row];
    }
    x
}
