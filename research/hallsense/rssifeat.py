"""RSSI-only sensing (any laptop Wi-Fi card, no CSI): features from one or more RSSI streams.

Works with 1 dB integer RSSI at any rate >= ~10 Hz (resampled to a 20 Hz grid).
Same bands as the CSI features: breathing 0.12-0.6 Hz, micro-motion 0.6-3 Hz, noise 4-9.5 Hz.
"""
import numpy as np
from .roomfeat import to_grid, FS_GRID

NAMES = ["std", "p90_absdiff", "br", "mi", "hi", "br_mi", "peak", "glrt", "range", "slow"]


def rssi_window(g, fs=FS_GRID):
    """g [m, C] RSSI (dBm) on the grid, C = antenna chains (1 or 2) -> features"""
    g = g - g.mean(0)
    n = len(g)
    P = np.abs(np.fft.rfft(g * np.hanning(n)[:, None], n=4 * n, axis=0)) ** 2
    f = np.fft.rfftfreq(4 * n, 1 / fs)
    b = lambda lo, hi: P[(f >= lo) & (f < hi)].mean(0)
    br, mi, hi = b(0.12, 0.6), b(0.6, 3.0), b(4.0, 9.5) + 1e-6
    mb = (f >= 0.12) & (f < 0.6)
    s = P[mb].sum(1)
    peak = s.max() / (s.sum() + 1e-9)
    glrt = np.log(s.max() / (np.median(P[(f >= 1.0) & (f < 3.0)].sum(1)) + 1e-9) + 1e-9)
    d = np.abs(np.diff(g, axis=0))
    half = n // 2
    slow = np.abs(g[:half].mean(0) - g[half:].mean(0)).mean()
    return np.array([np.log10(g.std(0).mean() + 1e-3), np.log10(np.percentile(d, 90) + 1e-3),
                     np.log10(br.mean() / hi.mean()), np.log10(mi.mean() / hi.mean()), np.log10(hi.mean()),
                     np.log10(br.mean() / (mi.mean() + 1e-6)), peak, glrt, np.ptp(g), slow], np.float32), \
        f[mb][np.argmax(s)] * 60


def rssi_windows(t, rssi, win_s=20.0, hop_s=5.0):
    """t [n] s, rssi [n] or [n, C] -> X [W, F], breathing-rate guesses [W]"""
    r = np.asarray(rssi, float)
    if r.ndim == 1:
        r = r[:, None]
    G = to_grid(np.asarray(t, float), r)
    w, h = int(win_s * FS_GRID), int(hop_s * FS_GRID)
    X, R = [], []
    for s in range(0, len(G) - w + 1, h):
        x, rate = rssi_window(G[s:s + w]); X.append(x); R.append(rate)
    return (np.array(X), np.array(R)) if X else (np.zeros((0, len(NAMES))), np.zeros(0))
