"""Single-link (router -> one ESP32) features from the per-packet CSI stream.

Works on any packet rate (resampled onto a 20 Hz grid) and any subcarrier layout
(L-LTF 52, HT-LTF 56, raw 64 with nulls): unused subcarriers are dropped automatically.
Everything is relative (band-power ratios, normalised spectra), so it does not depend
on the board, the AGC, or how far the router is.

Bands (per subcarrier amplitude in dB, 20 Hz grid):
  breath  0.12 - 0.6 Hz   chest motion of a still person
  micro   0.6  - 3   Hz   fidgets, typing, limbs
  noise   4    - 9.5 Hz   receiver noise floor reference (people don't move there)
"""
import numpy as np

FS_GRID = 20.0
WIN_S = 20.0
HOP_S = 5.0
BANDS = {"breath": (0.12, 0.6), "micro": (0.6, 3.0), "noise": (4.0, 9.5)}
FEAT_NAMES = []


def to_grid(t, amp, fs=FS_GRID):
    """t [n] seconds, amp [n,K] linear amplitude -> uniform grid [m,K] (bin means, gaps linearly filled)"""
    if len(t) < 2:
        return np.zeros((0, amp.shape[1]))
    t = t - t[0]
    m = int(t[-1] * fs) + 1
    idx = np.minimum((t * fs).astype(int), m - 1)
    s = np.zeros((m, amp.shape[1])); c = np.zeros(m)
    np.add.at(s, idx, amp); np.add.at(c, idx, 1)
    ok = c > 0
    g = np.full((m, amp.shape[1]), np.nan)
    g[ok] = s[ok] / c[ok, None]
    x = np.arange(m)
    for k in range(amp.shape[1]):
        g[:, k] = np.interp(x, x[ok], g[ok, k])
    return g


def amplitude_from_iq(iq):
    """iq int8 [n,K,2] -> linear amplitude [n,K'] keeping only subcarriers that carry signal"""
    a = np.sqrt(iq[..., 0].astype(np.float32) ** 2 + iq[..., 1].astype(np.float32) ** 2)
    keep = np.mean(a > 0, 0) > 0.9
    a = a[:, keep]
    # per-packet AGC removal: the receiver gain jumps from packet to packet, the shape across subcarriers doesn't
    return a / np.maximum(a.mean(1, keepdims=True), 1e-6)


def _band(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return P[m].mean(0)


def window_features(G, fs=FS_GRID):
    """G [m,K] linear amplitude on the grid for one window -> feature vector"""
    return spectral_features(20 * np.log10(np.maximum(G, 1e-3)), fs)


def spectral_features(A, fs=FS_GRID):
    """A [m,K] per-subcarrier series on the grid (amplitude in dB, or sanitized phase in rad) -> feature vector"""
    A = A - A.mean(0)
    n = len(A)
    win = np.hanning(n)[:, None]
    F = np.fft.rfft(A * win, axis=0)
    P = np.abs(F) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    br, mi, no = (_band(f, P, *BANDS[b]) for b in ("breath", "micro", "noise"))
    no = np.maximum(no, 1e-9)
    r_br = np.log10(br / no); r_mi = np.log10(mi / no)
    # best subcarriers carry the person (others sit in fades / far from the body)
    top = np.argsort(-r_br)[: max(3, len(r_br) // 8)]
    topm = np.argsort(-r_mi)[: max(3, len(r_mi) // 8)]
    # breathing periodicity: PCA of breath-band-filtered subcarriers, spectral concentration of PC1
    mb = (f >= BANDS["breath"][0]) & (f < BANDS["breath"][1])
    Fb = np.zeros_like(F); Fb[mb] = F[mb]
    xb = np.fft.irfft(Fb, n=n, axis=0)
    xs = xb[:, top]
    u, s, vt = np.linalg.svd(xs - xs.mean(0), full_matrices=False)
    pc = u[:, 0] * s[0]
    spc = np.abs(np.fft.rfft(pc * np.hanning(n), n=4 * n)) ** 2
    fz = np.fft.rfftfreq(4 * n, 1 / fs)
    mz = (fz >= BANDS["breath"][0]) & (fz < BANDS["breath"][1])
    pk = spc[mz].max() / (spc[mz].sum() + 1e-12)
    rate = fz[mz][np.argmax(spc[mz])] * 60.0
    pc_share = s[0] ** 2 / (np.sum(s ** 2) + 1e-12)
    # temporal structure: how bursty is the micro band (fidgets are bursts, noise is flat)
    tot = np.std(A, 0)
    feats = [np.median(r_br), r_br[top].mean(), np.median(r_mi), r_mi[topm].mean(),
             np.log10(np.median(tot) + 1e-6), np.log10(np.percentile(tot, 90) + 1e-6),
             pk, pc_share, np.mean(r_br > 0.5), np.mean(r_mi > 0.5)]
    return np.array(feats, np.float32), rate


def phase_from_iq(iq):
    """iq int8 [n,K,2] -> sanitized phase [n,K'] (rad) on the subcarriers that carry signal.
    The receiver's clock offsets add, per packet, a phase that is linear in the subcarrier index (timing offset)
    plus a constant (frequency offset); some chips also shift one half of the band. Fitting and removing
    a*k + b + c*[upper half] per packet leaves what the room did to each subcarrier; chest motion of a few mm
    moves that phase by 2*pi*d/lambda, far more visibly than it moves the amplitude.
    Byte order ([I,Q] or [Q,I]) only flips the sign, which the power-based features don't see."""
    a = np.hypot(iq[..., 0].astype(np.float32), iq[..., 1].astype(np.float32))
    keep = np.mean(a > 0, 0) > 0.9
    ph = np.unwrap(np.arctan2(iq[..., 1].astype(np.float32), iq[..., 0].astype(np.float32))[:, keep], axis=1)
    k = np.nonzero(keep)[0].astype(float); k = k - k.mean()
    X = np.c_[k, np.ones_like(k), (k > 0).astype(float)]
    beta = np.linalg.lstsq(X, ph.T, rcond=None)[0]
    r = ph - (X @ beta).T
    return np.angle(np.exp(1j * r))                                    # residuals live in (-pi, pi]


def phase_windows(t, ph, win_s=WIN_S, hop_s=HOP_S):
    """sanitized phase [n,K] -> X [W, len(PHASE_NAMES)] on the same 20 s / 5 s windows as windows()"""
    G = to_grid(t, ph)
    w, h = int(win_s * FS_GRID), int(hop_s * FS_GRID)
    out = [spectral_features(G[s:s + w])[0] for s in range(0, len(G) - w + 1, h)]
    return np.array(out) if out else np.zeros((0, len(PHASE_NAMES)))


def profile_windows(t, amp, win_s=WIN_S, hop_s=HOP_S):
    """the room's STATIC fingerprint per window: mean amplitude (dB) of every subcarrier, [W, K'].
    A body standing or sitting somewhere reshapes the multipath, so this profile moves when someone stays in the
    room, even if they don't move at all. It also drifts over hours and days, so it is only compared with a
    reference taken minutes earlier (hallsense/occupancy.py), never with a stored calibration."""
    G = to_grid(t, amp)
    w, h = int(win_s * FS_GRID), int(hop_s * FS_GRID)
    out = [20 * np.log10(np.maximum(G[s:s + w], 1e-3)).mean(0) for s in range(0, len(G) - w + 1, h)]
    return np.array(out) if out else np.zeros((0, amp.shape[1]))


def extra_features(G, fs=FS_GRID):
    """what kind of thing moved: fans are spectral lines, doors are steps, people are broadband
    bursts plus breathing, pets/vacuums are weak and low (few subcarriers react)."""
    A = 20 * np.log10(np.maximum(G, 1e-3)); A = A - A.mean(0); n = len(A)
    P = np.abs(np.fft.rfft(A * np.hanning(n)[:, None], axis=0)) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    no = P[(f >= 4) & (f < 9.5)].mean(0) + 1e-12
    act = np.argsort(-(P[(f >= 0.12) & (f < 3)].mean(0) / no))[: max(3, A.shape[1] // 8)]
    sp = (P[:, act] / no[act]).mean(1)
    hb = (f >= 0.5) & (f < 9.8)
    line = np.log10(sp[hb].max() / np.median(sp[hb]))
    flat = np.exp(np.mean(np.log(sp[hb] + 1e-12))) / (np.mean(sp[hb]) + 1e-12)
    h = n // 2
    step = np.median(np.abs(A[:h].mean(0) - A[h:].mean(0)) / (0.5 * (A[:h].std(0) + A[h:].std(0)) + 1e-6))
    e = np.convolve((np.diff(A[:, act], axis=0) ** 2).mean(1), np.ones(int(fs)) / fs, "same")
    burst = np.log10(np.percentile(e, 95) / (np.median(e) + 1e-9) + 1e-9)
    spread = np.mean((P[(f >= 0.12) & (f < 3)].mean(0) / no) > 3.0)       # share of subcarriers that react
    # breathing GLRT across subcarriers (Patwari-style ML combining)
    Z = P[(f >= 0.13) & (f <= 0.6)] / (np.median(P[(f >= 1.0) & (f < 3.0)], 0) + 1e-12)
    S = Z.sum(1); glrt = np.log(S.max() / Z.shape[1] + 1e-9)
    return np.array([line, flat, step, burst, spread, glrt], np.float32)


EXTRA_NAMES = ["line", "flat", "step", "burst", "spread", "glrt"]


PHASE_NAMES = ["ph_br_med", "ph_br_top", "ph_mi_med", "ph_mi_top", "ph_std_med", "ph_std_p90", "ph_br_peak",
               "ph_pc_share", "ph_br_frac", "ph_mi_frac"]

FEAT_NAMES[:] = ["br_med", "br_top", "mi_med", "mi_top", "std_med", "std_p90", "br_peak", "pc_share",
                 "br_frac", "mi_frac"]


def windows(t, amp, rssi=None, win_s=WIN_S, hop_s=HOP_S):
    G = to_grid(t, amp)
    w, h = int(win_s * FS_GRID), int(hop_s * FS_GRID)
    out, rates = [], []
    for s in range(0, len(G) - w + 1, h):
        x, r = window_features(G[s:s + w])
        if rssi is not None:
            g = to_grid(t, rssi[:, None].astype(float))[s:s + w, 0]
            x = np.r_[x, np.log10(np.std(g) + 0.1)]
        x = np.r_[x, extra_features(G[s:s + w])]
        out.append(x); rates.append(r)
    return (np.array(out), np.array(rates)) if out else (np.zeros((0, len(FEAT_NAMES) + (rssi is not None) + 6)), np.zeros(0))
