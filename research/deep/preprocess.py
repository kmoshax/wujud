"""Identical preprocessing of simulated and real ESP32 CSI for the deep models.

Per packet: the 52 L-LTF data subcarriers (k = -26..-1, 1..26), amplitude divided by its per-packet mean
(AGC removal) and sanitized phase (per-packet a*k + b + c*[k>0] removed); then everything goes onto a 20 Hz
grid (bin means, gaps linearly filled). Amplitude is returned in dB.
"""
import numpy as np

FS_GRID = 20.0
WINDOW = 400
HOP = 100
SUBCARRIER_INDEX = np.r_[-26:0, 1:27]
SIM_HT_LTF_POSITION = np.where(SUBCARRIER_INDEX < 0, SUBCARRIER_INDEX + 28, SUBCARRIER_INDEX + 27)


def real_buffer_bins(csi_int8):
    """ESP32 128-byte buffers come in two layouts: FFT order (bins 29..35 empty) or centred (DC at bin 32)."""
    amplitude = np.hypot(*np.moveaxis(csi_int8.reshape(len(csi_int8), 64, 2)[:2000].astype(np.float32), -1, 0))
    fft_order = amplitude[:, 29:36].mean() < 0.05 * amplitude.mean()
    if fft_order:
        return np.where(SUBCARRIER_INDEX > 0, SUBCARRIER_INDEX, 64 + SUBCARRIER_INDEX)
    return 32 + SUBCARRIER_INDEX


def real_complex(csi_int8):
    """raw buffer [n,128] of [imag, real] pairs -> complex CSI [n, 52]"""
    pairs = csi_int8.reshape(len(csi_int8), 64, 2).astype(np.float32)
    bins = real_buffer_bins(csi_int8)
    return pairs[:, bins, 1] + 1j * pairs[:, bins, 0]


def sim_complex(iq_int8):
    """simulator I/Q [n, 56, 2] (real, imag) on HT-LTF -> complex CSI [n, 52]"""
    pairs = iq_int8[:, SIM_HT_LTF_POSITION].astype(np.float32)
    return pairs[..., 0] + 1j * pairs[..., 1]


def fill_dead_subcarriers(amplitude):
    """a subcarrier that is zero in most packets (e.g. k=+1 on the original ESP32) takes its neighbours' mean"""
    dead = np.mean(amplitude > 0, 0) < 0.9
    if not dead.any() or dead.all():
        return amplitude
    alive = np.nonzero(~dead)[0]
    filled = amplitude.copy()
    for k in np.nonzero(dead)[0]:
        nearest = alive[np.argsort(np.abs(alive - k))[:2]]
        filled[:, k] = amplitude[:, nearest].mean(1)
    return filled


def normalised_amplitude(csi):
    amplitude = fill_dead_subcarriers(np.abs(csi))
    return amplitude / np.maximum(amplitude.mean(1, keepdims=True), 1e-6)


def sanitized_phase_phasor(csi):
    phase = np.unwrap(np.angle(csi), axis=1)
    k = SUBCARRIER_INDEX.astype(float)
    design = np.c_[k - k.mean(), np.ones_like(k), (k > 0).astype(float)]
    beta = np.linalg.lstsq(design, phase.T, rcond=None)[0]
    return np.exp(1j * (phase - (design @ beta).T))


def to_grid(t, values):
    """t [n] s, values [n, C] (real or complex) -> [m, C] on the 20 Hz grid"""
    t = t - t[0]
    m = int(t[-1] * FS_GRID) + 1
    index = np.minimum((t * FS_GRID).astype(int), m - 1)
    sums = np.zeros((m, values.shape[1]), values.dtype)
    counts = np.zeros(m)
    np.add.at(sums, index, values)
    np.add.at(counts, index, 1)
    seen = counts > 0
    grid = sums[seen] / counts[seen, None]
    x, xs = np.arange(m), np.nonzero(seen)[0]
    if np.iscomplexobj(grid):
        return np.stack([np.interp(x, xs, grid[:, c].real) + 1j * np.interp(x, xs, grid[:, c].imag)
                         for c in range(values.shape[1])], 1)
    return np.stack([np.interp(x, xs, grid[:, c]) for c in range(values.shape[1])], 1)


def gridded_streams(t, csi, rssi):
    """-> amplitude dB [m,52], phase rad [m,52], rssi dBm [m] (float16-ready)"""
    amplitude = to_grid(t, normalised_amplitude(csi))
    phase = np.angle(to_grid(t, sanitized_phase_phasor(csi)))
    rssi_grid = to_grid(t, rssi.astype(np.float64)[:, None])[:, 0]
    return 20 * np.log10(np.maximum(amplitude, 1e-3)), phase, rssi_grid


def window_starts(length):
    return np.arange(0, length - WINDOW + 1, HOP)
