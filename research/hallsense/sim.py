"""Scene simulator: people in seats -> per-link CSI over time -> ESP32 measurements.

Physics that matters for seated people (and that most WiFi-sensing sims skip):
  1. Body SHADOWING of every propagation path (LOS, wall bounces, floor bounce),
     computed with exact Fresnel-Kirchhoff strip diffraction: the body is an
     absorbing strip whose projected width depends on the link angle.
  2. Bistatic SCATTERING from torso and hands.
  3. MICRO-MOTION: breathing, postural sway, and Poisson "fidget" events
     (the crowd-fidget effect from Mostofi lab's stationary-crowd counting work).
  4. Calibration drift: chairs moved, bags placed, Tx-power drift between
     the empty-hall calibration and the lecture.
  5. ESP32 measurement chain: thermal + estimation noise, imperfect AGC,
     int8 quantisation, integer RSSI, packet loss, interference bursts.
Features are amplitude-only (phase is unusable on ESP32 without heavy correction).
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import fresnel
from scipy.signal import lfilter

from .config import LAM, FC, C, FREQS, K, FS, WIN_S, CALIB_S, SESSION_S, NOISE_FLOOR_DBM
from .geometry import Hall, link_paths, path_response, scatter_response
from .features import window_features


# ---------------------------------------------------------------------------
# occupancy + people
# ---------------------------------------------------------------------------
def sample_occupancy(hall: Hall, rng, frac=None):
    S = len(hall.seats)
    if frac is None:
        bucket = rng.choice(4, p=[0.08, 0.27, 0.35, 0.30])
        frac = [0.0, rng.uniform(0.02, 0.3), rng.uniform(0.3, 0.7), rng.uniform(0.7, 1.0)][bucket]
    n = int(round(frac * S))
    rows = hall.seat_rc[:, 0] / max(1, hall.seat_rc[:, 0].max())
    pattern = rng.choice(["uniform", "front", "back", "clusters"])
    score = rng.random(S)
    if pattern == "front":
        score -= 0.8 * rows
    elif pattern == "back":
        score += 0.8 * rows
    elif pattern == "clusters":
        for _ in range(rng.integers(2, 6)):
            c = hall.seats[rng.integers(S)]
            score += 1.5 * np.exp(-np.sum((hall.seats - c) ** 2, 1) / (2 * rng.uniform(1, 3) ** 2))
    occ = np.zeros(S, bool)
    if n:
        occ[np.argsort(-score)[:n]] = True
    return occ, pattern


@dataclass
class People:
    pos: np.ndarray      # [Q,T,2] torso centre
    breath: np.ndarray   # [Q,T] chest radial expansion (m)
    hand: np.ndarray     # [Q,T,2]
    half_a: np.ndarray   # [Q] shoulder half-width (x)
    half_b: np.ndarray   # [Q] chest half-depth (y)
    eta: np.ndarray      # [Q] blocking strength
    rcs: np.ndarray      # [Q] torso RCS
    hand_rcs: np.ndarray  # [Q]


def seated_people(base, T, rng, fs=FS, restless=1.0):
    Q = len(base)
    t = np.arange(T) / fs
    base = base + rng.normal(0, 0.04, (Q, 2))
    # postural sway: AR(1), sigma 4 mm, tau 3 s
    rho = np.exp(-1 / (3.0 * fs))
    e = rng.normal(0, 0.004 * np.sqrt(1 - rho ** 2), (Q, T, 2))
    sway = lfilter([1], [1, -rho], e, axis=1)
    pos = base[:, None, :] + sway
    fb = rng.uniform(0.2, 0.33, Q)
    breath = rng.uniform(0.003, 0.006, Q)[:, None] * np.sin(2 * np.pi * fb[:, None] * t + rng.uniform(0, 6.3, (Q, 1)))
    hand_base = base + np.stack([rng.uniform(-0.25, 0.25, Q), rng.uniform(0.15, 0.35, Q)], 1)
    hand = np.repeat(hand_base[:, None, :], T, 1) + rng.normal(0, 0.003, (Q, T, 2))
    # fidget events (Poisson); median one every ~20 s, spread per person
    rate = restless * np.exp(rng.normal(np.log(1 / 20), 0.7, Q))
    for q in range(Q):
        for _ in range(rng.poisson(rate[q] * T / fs)):
            t0 = rng.integers(0, T)
            dur = int(rng.uniform(0.8, 4.0) * fs)
            idx = np.arange(t0, min(T, t0 + dur))
            if len(idx) < 2:
                continue
            ph = (idx - t0) / dur
            ang = rng.uniform(0, 2 * np.pi)
            mag = rng.uniform(0.02, 0.15)
            vec = mag * np.array([np.cos(ang), np.sin(ang)])
            if rng.random() < 0.6:          # shift and come back
                prof = np.sin(np.pi * ph) ** 2
                pos[q, idx] += prof[:, None] * vec
            else:                            # shift and stay
                prof = 0.5 - 0.5 * np.cos(np.pi * ph)
                pos[q, idx] += prof[:, None] * vec
                pos[q, idx[-1] + 1:] += vec
            ha = rng.uniform(0.05, 0.3)
            fh = rng.uniform(0.5, 2.0)
            osc = ha * np.sin(2 * np.pi * fh * (idx - t0) / fs) * np.sin(np.pi * ph)
            ang2 = rng.uniform(0, 2 * np.pi)
            hand[q, idx] += osc[:, None] * np.array([np.cos(ang2), np.sin(ang2)])
    return People(pos, breath, hand,
                  rng.uniform(0.19, 0.25, Q), rng.uniform(0.11, 0.15, Q),
                  rng.uniform(0.6, 0.97, Q), rng.uniform(0.1, 0.5, Q), rng.uniform(0.02, 0.1, Q))


def walkers(hall: Hall, T, rng, fs=FS):
    """lecturer in the front zone + maybe a late student walking down an aisle.
    Returns People with NaN positions when absent. Not counted as seated."""
    tracks = []
    if rng.random() < 0.75:
        p = np.array([rng.uniform(1, hall.W - 1), rng.uniform(0.6, hall.front - 0.3)])
        tgt = p.copy()
        v = rng.uniform(0.3, 0.9)
        tr = np.zeros((T, 2))
        pause = 0
        for i in range(T):
            if pause > 0:
                pause -= 1
            else:
                d = tgt - p
                n = np.linalg.norm(d)
                if n < 0.1:
                    tgt = np.array([rng.uniform(1, hall.W - 1), rng.uniform(0.6, hall.front - 0.3)])
                    pause = int(rng.exponential(4.0) * fs)
                else:
                    p = p + d / n * min(n, v / fs)
            tr[i] = p
        tracks.append(tr)
    if rng.random() < 0.25 and hall.aisles_x:
        ax = rng.choice(hall.aisles_x)
        T0 = rng.integers(0, T // 2)
        dur = int((hall.D - hall.front) / rng.uniform(0.7, 1.2) * fs)
        tr = np.full((T, 2), np.nan)
        idx = np.arange(T0, min(T, T0 + dur))
        y = np.linspace(hall.D - 0.5, hall.front + 0.5, dur)[: len(idx)]
        if rng.random() < 0.5:
            y = y[::-1]
        tr[idx] = np.stack([np.full(len(idx), ax), y], 1)
        tracks.append(tr)
    if not tracks:
        return None
    Q = len(tracks)
    pos = np.stack(tracks)
    return People(pos, np.zeros((Q, T)), pos.copy(), np.full(Q, 0.22), np.full(Q, 0.14),
                  np.full(Q, 0.92), np.full(Q, 0.8), np.full(Q, 0.1))


# ---------------------------------------------------------------------------
# propagation
# ---------------------------------------------------------------------------
def _strip(P0, P1, cum0, Ltot, pos, half_a, half_b, breath, eta, prune=None, prune_t=0):
    """Fresnel strip-diffraction factor of bodies on one path segment.
    pos [Q,T,2], breath [Q,T]; returns [T] complex (product over bodies)."""
    seg = P1 - P0
    Ls = np.linalg.norm(seg)
    if Ls < 1e-6:
        return None
    u = seg / Ls
    nrm = np.array([-u[1], u[0]])
    halfw = np.sqrt((half_a * nrm[0]) ** 2 + (half_b * nrm[1]) ** 2)       # [Q]
    rel = pos - P0
    tt = rel @ u
    hh = rel @ nrm
    if prune is not None:
        tb, hb = tt[:, prune_t], hh[:, prune_t]
        ok = (tb > 0.05) & (tb < Ls - 0.05)
        d1 = cum0 + np.clip(tb, 0.05, Ls - 0.05)
        kf = np.sqrt(2 * Ltot / (LAM * d1 * (Ltot - d1)))
        ok &= (np.abs(hb) - halfw) * kf < prune
        if not ok.any():
            return None
        tt, hh, halfw, breath, eta = tt[ok], hh[ok], halfw[ok], breath[ok], eta[ok]
    valid = (tt > 0.05) & (tt < Ls - 0.05) & np.isfinite(tt)
    d1 = cum0 + np.where(valid, tt, Ls / 2)
    d2 = Ltot - d1
    kf = np.sqrt(2 * Ltot / (LAM * d1 * d2))
    w = halfw[:, None] + breath
    hh = np.where(valid, hh, 1e3)
    S1, C1 = fresnel((hh - w) * kf)
    S2, C2 = fresnel((hh + w) * kf)
    F = 1 - eta[:, None] * (1 + 1j) / 2 * ((C2 - C1) - 1j * (S2 - S1))
    F = np.where(valid, F, 1.0)
    return np.prod(F, axis=0)


def _scatter_series(a, b, pos, rcs, extra_len=None):
    """sum of point-scatterer returns over time, [T,K]. pos [Q,T,2] (NaN -> absent)."""
    Q, T, _ = pos.shape
    present = np.isfinite(pos[..., 0])
    p0 = np.where(present[..., None], pos, 0.0)
    if not present.any():
        return 0.0
    cnt = np.maximum(present.sum(1), 1)[:, None]
    base = p0.sum(1) / cnt
    G, L0 = scatter_response(a, b, base, rcs)                      # [Q,K]
    L = np.linalg.norm(p0 - a, axis=-1) + np.linalg.norm(p0 - b, axis=-1)
    if extra_len is not None:
        L = L - extra_len
    # distance-amplitude change relative to base (first order) + carrier phase; small in-band delay change ignored
    E = np.exp(-2j * np.pi * FC * (L - L0[:, None]) / C) * (L0[:, None] / np.maximum(L, 0.3)) * present
    return E.T.astype(np.complex64) @ G.astype(np.complex64)   # [T,K]


@dataclass
class LinkStatic:
    paths: list
    Hp: np.ndarray          # [P,K] free-space response per geometric path
    rest_calib: np.ndarray  # [K] clutter + diffuse during calibration
    rest_sess: np.ndarray   # [K] clutter + diffuse during the lecture


def build_static(hall, i, j, clutter_c, clutter_s, rng, absorb=1.0):
    a, b = hall.nodes[i], hall.nodes[j]
    paths = link_paths(hall, i, j)
    Hp = np.stack([path_response(p.length, p.amp) for p in paths])
    d = paths[0].length
    # diffuse multipath (unmodelled furniture, tiers, doors): exponential PDP, Rician-K random
    Kdb = rng.uniform(-2, 8)
    m = 30
    tau = rng.exponential(rng.uniform(15e-9, 40e-9), m)
    pw = np.exp(-tau / 30e-9)
    pw = pw / pw.sum() * (LAM / (4 * np.pi * d)) ** 2 * 10 ** (-Kdb / 10)
    alpha = np.sqrt(pw / 2) * (rng.normal(size=m) + 1j * rng.normal(size=m))
    diff = (alpha[:, None] * np.exp(-2j * np.pi * FREQS[None] * (d / C + tau[:, None]))).sum(0)
    Gc, _ = scatter_response(a, b, clutter_c[:, :2], clutter_c[:, 2])
    Gs, _ = scatter_response(a, b, clutter_s[:, :2], clutter_s[:, 2])
    # room electromagnetics: bodies absorb the reverberant field -> diffuse + clutter power drops
    return LinkStatic(paths, Hp, Gc.sum(0) + diff, (Gs.sum(0) + diff) * absorb)


def link_session_channel(st: LinkStatic, a, b, people: People, walk: People, T, absorb=1.0, prune_t=0):
    """H(t,k) during the lecture, [T,K] complex64"""
    H = np.zeros((T, K), np.complex64) + st.rest_sess[None].astype(np.complex64)
    for p, hp in zip(st.paths, st.Hp):
        F = np.ones(T, np.complex128)
        for (P0, P1, cum0) in p.segments:
            if people is not None:
                f = _strip(P0, P1, cum0, p.length, people.pos, people.half_a, people.half_b,
                           people.breath, people.eta * p.shadow_eta, prune=3.5, prune_t=prune_t)
                if f is not None:
                    F *= f
            if walk is not None:
                f = _strip(P0, P1, cum0, p.length, walk.pos, walk.half_a, walk.half_b,
                           walk.breath, walk.eta * p.shadow_eta)
                if f is not None:
                    F *= f
        H += (F[:, None] * hp[None]).astype(np.complex64)
    if people is not None:
        H += absorb * _scatter_series(a, b, people.pos, people.rcs, extra_len=2 * people.breath)
        H += absorb * _scatter_series(a, b, people.hand, people.hand_rcs)
    if walk is not None:
        H += _scatter_series(a, b, walk.pos, walk.rcs)
    return H


# ---------------------------------------------------------------------------
# ESP32 measurement chain
# ---------------------------------------------------------------------------
@dataclass
class RadioParams:
    ptx: np.ndarray        # [N] dBm (Tx power + antenna/enclosure loss)
    drift: np.ndarray      # [N] dB change between calibration and lecture
    agc_target: np.ndarray  # [N] rms LSB the receiver AGC aims at
    evm_db: np.ndarray     # [N] channel-estimation noise relative to signal
    loss: float
    interf: float


def random_radio(n, rng, harsh=1.0):
    return RadioParams(rng.uniform(4, 14, n), rng.normal(0, 0.25 * harsh, n), rng.uniform(12, 24, n),
                       rng.uniform(-29, -23, n) + 2 * (harsh - 1), rng.uniform(0.02, 0.10) * harsh,
                       rng.uniform(0.005, 0.03) * harsh)


def measure(H, tx, rx, radio: RadioParams, rng, session=True):
    """complex channel [T,K] -> (iq int8 [T,K,2], rssi int [T], received bool [T])"""
    T = H.shape[0]
    ptx = radio.ptx[tx] + (radio.drift[tx] if session else 0.0)
    g = 10 ** (ptx / 20)
    Hs = H * g                                           # sqrt(mW) units
    p_sig = np.mean(np.abs(Hs) ** 2, axis=1)             # mW
    n_th = 10 ** (NOISE_FLOOR_DBM / 10)
    n_est = p_sig * 10 ** (radio.evm_db[rx] / 10)
    sig = np.sqrt((n_th + n_est) / 2)[:, None]
    Y = Hs + sig * (rng.normal(size=(T, K)) + 1j * rng.normal(size=(T, K)))
    # interference bursts: phones/APs in the hall collide with some packets
    bad = rng.random(T) < radio.interf
    if bad.any():
        nb = bad.sum()
        Y[bad] += np.sqrt(p_sig[bad] * 10 ** (rng.uniform(-12, 0, nb) / 10) / 2)[:, None] * \
            (rng.normal(size=(nb, K)) + 1j * rng.normal(size=(nb, K)))
    p_rx = np.mean(np.abs(Y) ** 2, axis=1)
    rssi_true = 10 * np.log10(p_rx)
    # AGC state: piecewise-constant gain-estimate bias, changes a few times a minute
    steps = np.cumsum(rng.random(T) < 3 / (60 * FS))
    bias = rng.normal(0, 0.6, steps.max() + 1)[steps]
    rssi = np.round(rssi_true + bias + rng.normal(0, 0.5, T)).astype(np.int16)
    tgt = radio.agc_target[rx] * 10 ** (rng.normal(0, 1.0, T) / 20)
    scale = tgt / np.sqrt(p_rx)
    Yq = Y * scale[:, None]
    iq = np.clip(np.round(np.stack([Yq.real, Yq.imag], -1)), -127, 127).astype(np.int8)
    got = rng.random(T) > radio.loss
    return iq, rssi, got


# ---------------------------------------------------------------------------
# full scene
# ---------------------------------------------------------------------------
def make_clutter(hall, occ, rng, env_rng=None):
    S = len(hall.seats)
    e = rng if env_rng is None else env_rng       # the empty hall's furniture is a property of the hall
    chairs = np.c_[hall.seats + e.normal(0, 0.03, (S, 2)), e.uniform(0.005, 0.03, S)]
    nr = e.integers(15, 40)
    objs = np.c_[e.uniform(0.3, hall.W - 0.3, nr), e.uniform(0.3, hall.D - 0.3, nr), e.uniform(0.02, 0.4, nr)]
    calib = np.vstack([chairs, objs])
    # lecture time: occupied chairs pulled out, bags/laptops appear, some objects moved
    ch2 = chairs.copy()
    ch2[occ, :2] += rng.normal(0, 0.08, (occ.sum(), 2))
    ob2 = objs.copy()
    mv = rng.random(nr) < 0.2
    ob2[mv, :2] += rng.normal(0, 0.2, (mv.sum(), 2))
    bag = occ & (rng.random(S) < 0.6)
    bags = np.c_[hall.seats[bag] + rng.normal(0, 0.2, (bag.sum(), 2)), rng.uniform(0.01, 0.08, bag.sum())]
    sess = np.vstack([ch2, ob2, bags])
    return calib, sess


def simulate_scene(hall: Hall, rng, occ=None, harsh=1.0, restless=1.0, with_walkers=True,
                   calib_s=CALIB_S, session_s=SESSION_S, keep_series=False, env_seed=None):
    """Returns dict with per-link window features for calibration and session.
    feats arrays: [L, W, 13] mean-dB and std-dB, averaged over both link directions.
    env_seed: if given, everything that belongs to the HALL (furniture, diffuse multipath,
    wall absorption, each board's Tx power/AGC/noise) is fixed by it, so several scenes
    with the same env_seed are different lectures in the same deployed hall."""
    env = None if env_seed is None else np.random.default_rng([env_seed, 4242])
    if occ is None:
        occ, pattern = sample_occupancy(hall, rng)
    else:
        pattern = "given"
    Tc, Ts = int(calib_s * FS), int(session_s * FS)
    people = seated_people(hall.seats[occ], Ts, rng, restless=restless) if occ.any() else None
    walk = walkers(hall, Ts, rng) if with_walkers else None
    radio = random_radio(len(hall.nodes), rng if env is None else env, harsh)
    if env is not None:
        radio.drift = rng.normal(0, 0.25 * harsh, len(hall.nodes))
    cl_c, cl_s = make_clutter(hall, occ, rng, env)
    # Sabine-style absorption: A0 = alpha*surface (room), each seated person adds a_p
    surface = 2 * hall.W * hall.D + 2 * (hall.W + hall.D) * hall.Hc
    A0 = (rng if env is None else env).uniform(0.1, 0.3) * surface
    # each body absorbs 0.3-0.5 m^2 (size, clothing); summed per person so it averages over a crowd
    absorb = float(np.sqrt(A0 / (A0 + rng.uniform(0.3, 0.5, occ.sum()).sum())))
    links = hall.links
    out = {k: [] for k in ("cal_mean", "cal_std", "ses_mean", "ses_std", "cal_n", "ses_n")}
    for (i, j) in links:
        a, b = hall.nodes[i], hall.nodes[j]
        lrng = rng if env_seed is None else np.random.default_rng([env_seed, 4243, i, j])
        st = build_static(hall, i, j, cl_c, cl_s, lrng, absorb)
        Hc = np.repeat((st.Hp.sum(0) + st.rest_calib)[None].astype(np.complex64), Tc, 0)
        Hs = link_session_channel(st, a, b, people, walk, Ts, absorb)
        acc = {k: [] for k in out}
        for (tx, rx) in ((i, j), (j, i)):
            for tag, H, ses in (("cal", Hc, False), ("ses", Hs, True)):
                iq, rssi, got = measure(H, tx, rx, radio, rng, session=ses)
                m, s, n = window_features(iq[got], rssi[got], np.nonzero(got)[0] / FS, WIN_S, H.shape[0] / FS)
                acc[f"{tag}_mean"].append(m)
                acc[f"{tag}_std"].append(s)
                acc[f"{tag}_n"].append(n)
        for k in out:
            out[k].append(np.nanmean(np.stack(acc[k]), 0) if not k.endswith("_n") else np.sum(acc[k], 0))
    res = {k: np.stack(v).astype(np.float32) for k, v in out.items()}
    res["occ"] = occ
    res["pattern"] = pattern
    res["n_walkers"] = 0 if walk is None else len(walk.pos)
    res["absorb_db"] = 20 * np.log10(absorb)
    return res
