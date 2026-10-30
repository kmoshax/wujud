"""3-D scene simulation -> ESP32 measurements.

simulate_room():  per-packet CSI stream of each link (router -> ESP32, optionally ESP32 <-> ESP32),
                  plus per-frame ground truth (activity, position, breathing rate).
simulate_hall3(): per-link 1-s window features (same format as the mesh firmware reports).
"""
import numpy as np

from .config import FC, DF, SC_IDX, NOISE_FLOOR_DBM, WIN_S, LAM, C
from .physics3d import link_paths3, link_channel, static_rest, diffuse_paths, concat
from .scenes3d import seated_bodies, lecturer, room_person, Scene
from .features import window_features

FREQS52 = FC + SC_IDX * DF
FREQS56 = FC + np.r_[-28:0, 1:29] * DF


def measure(H, rng, ptx, evm_db, agc_target, loss, interf, fs):
    """complex channel [T,K] -> iq int8 [T,K,2], rssi int [T], kept bool [T]"""
    T, K = H.shape
    Hs = H * 10 ** (ptx / 20)
    p_sig = np.mean(np.abs(Hs) ** 2, 1)
    sig = np.sqrt((10 ** (NOISE_FLOOR_DBM / 10) + p_sig * 10 ** (evm_db / 10)) / 2)[:, None]
    Y = Hs + sig * (rng.normal(size=(T, K)) + 1j * rng.normal(size=(T, K)))
    bad = rng.random(T) < interf
    if bad.any():
        nb = bad.sum()
        Y[bad] += np.sqrt(p_sig[bad] * 10 ** (rng.uniform(-12, 0, nb) / 10) / 2)[:, None] * (rng.normal(size=(nb, K)) + 1j * rng.normal(size=(nb, K)))
    p_rx = np.mean(np.abs(Y) ** 2, 1)
    steps = np.cumsum(rng.random(T) < 3 / (60 * fs))
    bias = rng.normal(0, 0.6, steps.max() + 1)[steps]
    rssi = np.round(10 * np.log10(p_rx) + bias + rng.normal(0, 0.5, T)).astype(np.int16)
    tgt = agc_target * 10 ** (rng.normal(0, 1.0, T) / 20)
    Yq = Y * (tgt / np.sqrt(p_rx))[:, None]
    iq = np.clip(np.round(np.stack([Yq.real, Yq.imag], -1)), -127, 127).astype(np.int8)
    return iq, rssi, rng.random(T) > loss


def _radio(n, rng, harsh=1.0, evm_shift=0.0):
    return dict(ptx=rng.uniform(4, 16, n), evm=rng.uniform(-29, -23, n) + 2 * (harsh - 1) + evm_shift, agc=rng.uniform(12, 24, n),
                loss=rng.uniform(0.01, 0.08) * harsh, interf=rng.uniform(0.005, 0.03) * harsh)


def _absorb(sc, n_people, rng):
    surface = 2 * sc.W * sc.D + 2 * (sc.W + sc.D) * sc.H
    A0 = rng.uniform(0.1, 0.3) * surface
    return np.sqrt(A0 / (A0 + n_people * 0.4))


def simulate_room(sc: Scene, rng, seconds=120, fs=100.0, plan=None, calib_s=20, harsh=1.0,
                  distractors=(), wall_db=0.0, outside=False, rcs_scale=1.0, breath_scale=1.0, evm_shift=0.0):
    """wall_db: total one-way loss of the walls between the boards and the room (through-wall /
    rubble scenario). Everything that passes through the room is attenuated by it, while a static
    'flash' (reflections off the outside of the walls, leakage around them) is not - so the
    person's modulation sinks towards the noise as the walls get thicker."""
    T = int(seconds * fs); Tc = int(calib_s * fs)
    B, lab = room_person(sc, T, rng, fs, plan, rcs_scale, breath_scale)
    from .scenes3d import distractor
    if distractors:
        B = concat([B] + [distractor(k, sc, T, rng, fs) for k in distractors])
    att = 10 ** (-wall_db / 20)
    radio = _radio(len(sc.nodes), rng, harsh, evm_shift)
    present = np.isfinite(B.c[0, :, 0])
    absorb_t = np.where(present, _absorb(sc, 1, rng), 1.0)[:, None]
    out = []
    for (i, j) in sc.links:
        a, b = sc.nodes[i], sc.nodes[j]
        paths = link_paths3(a, b, sc.W, sc.D, sc.H, sc.floor, sc.gam_walls) + diffuse_paths(a, b, sc.W, sc.D, sc.H, rng, 24)
        rest = static_rest(a, b, sc.clutter, FREQS56, rng, np.linalg.norm(b - a))
        H = link_channel(a, b, paths, rest, B, T, FREQS56, 1.0)
        H += (rest[None] * (absorb_t - 1)).astype(np.complex64)
        Hc = link_channel(a, b, paths, rest, None, Tc, FREQS56)
        if wall_db > 0 or outside:
            d = np.linalg.norm(b - a)
            tau = rng.uniform(5e-9, 60e-9, 6)
            g = 0.35 * LAM / (4 * np.pi * d) * (rng.normal(size=6) + 1j * rng.normal(size=6)) / np.sqrt(12)
            flash = (g[:, None] * np.exp(-2j * np.pi * FREQS56[None] * (d / C + tau[:, None]))).sum(0).astype(np.complex64)
            H = H * att + flash[None]; Hc = Hc * att + flash[None]
        tx, rx = i, j    # router (or lower index) transmits, ESP32 receives
        iq, rssi, kept = measure(H, rng, radio["ptx"][tx], radio["evm"][rx], radio["agc"][rx], radio["loss"], radio["interf"], fs)
        iqc, rssic, keptc = measure(Hc, rng, radio["ptx"][tx], radio["evm"][rx], radio["agc"][rx], radio["loss"], radio["interf"], fs)
        tt = np.arange(T) / fs + rng.normal(0, 0.002, T)
        out.append(dict(link=(i, j), t=tt[kept], iq=iq[kept], rssi=rssi[kept],
                        t_cal=(np.arange(Tc) / fs)[keptc], iq_cal=iqc[keptc], rssi_cal=rssic[keptc]))
    lab["fs"] = fs
    return out, lab, B


def simulate_hall3(sc: Scene, rng, occ, seconds=60, calib_s=30, fs=20.0, harsh=1.0, restless=1.0, with_lecturer=True):
    T = int(seconds * fs); Tc = int(calib_s * fs)
    B = seated_bodies(sc.seats[occ], T, rng, fs, restless) if occ.any() else None
    L = lecturer(sc, T, rng, fs) if with_lecturer else None
    BB = concat([B, L])
    radio = _radio(len(sc.nodes), rng, harsh)
    drift = rng.normal(0, 0.25 * harsh, len(sc.nodes))
    absorb = _absorb(sc, occ.sum(), rng)
    # lecture-time clutter: chairs pulled out, bags
    cl_s = sc.clutter.copy()
    S = len(sc.seats)
    cl_s[:S][occ, :2] += rng.normal(0, 0.08, (occ.sum(), 2))
    bag = occ & (rng.random(S) < 0.6)
    bags = np.c_[sc.seats[bag, :2] + rng.normal(0, 0.2, (bag.sum(), 2)), sc.seats[bag, 2] + 0.2, rng.uniform(0.01, 0.08, bag.sum())]
    cl_s = np.vstack([cl_s, bags])
    res = {k: [] for k in ("cal_mean", "cal_std", "ses_mean", "ses_std")}
    for (i, j) in sc.links:
        a, b = sc.nodes[i], sc.nodes[j]
        paths = link_paths3(a, b, sc.W, sc.D, sc.H, sc.floor, sc.gam_walls) + diffuse_paths(a, b, sc.W, sc.D, sc.H, rng, 8)
        rest_c = static_rest(a, b, sc.clutter, FREQS52, rng, 0)
        rest_s = static_rest(a, b, cl_s, FREQS52, rng, 0)     # chairs pulled out, bags placed
        Hs = link_channel(a, b, paths, rest_s, BB, T, FREQS52, absorb)
        Hc = link_channel(a, b, paths, rest_c, None, Tc, FREQS52)
        acc = {k: [] for k in res}
        for (tx, rx) in ((i, j), (j, i)):
            for tag, H, ses in (("cal", Hc, False), ("ses", Hs, True)):
                ptx = radio["ptx"][tx] + (drift[tx] if ses else 0)
                iq, rssi, kept = measure(H, rng, ptx, radio["evm"][rx], radio["agc"][rx], radio["loss"], radio["interf"], fs)
                m, s, _ = window_features(iq[kept], rssi[kept], np.nonzero(kept)[0] / fs, WIN_S, H.shape[0] / fs)
                acc[f"{tag}_mean"].append(m); acc[f"{tag}_std"].append(s)
        for k in res:
            with np.errstate(all="ignore"):
                res[k].append(np.nanmean(np.stack(acc[k]), 0))
    out = {k: np.stack(v).astype(np.float32) for k, v in res.items()}
    out["occ"] = occ
    return out


def simulate_hall_timeline3(sc: Scene, rng, minutes=4.0, final_frac=0.75, calib_s=30, fs=20.0):
    """lecture where students arrive over time (for the live demo); same physics as simulate_hall3"""
    from .sim import sample_occupancy
    T = int(minutes * 60 * fs); Tc = int(calib_s * fs)

    class _V:
        seats = sc.seats[:, :2]; seat_rc = sc.seat_rc
    occ, _ = sample_occupancy(_V, rng, frac=final_frac)
    idx = np.nonzero(occ)[0]
    arrival = (np.sort(rng.uniform(0.15, 0.7, len(idx))) * T).astype(int); rng.shuffle(arrival)
    B = seated_bodies(sc.seats[idx], T, rng, fs)
    for q, t0 in enumerate(arrival):
        B.c[q, :t0] = np.nan; B.chest[q, :t0] = np.nan; B.limb[q, :t0] = np.nan
    BB = concat([B, lecturer(sc, T, rng, fs)])
    radio = _radio(len(sc.nodes), rng)
    n_present = (np.arange(T)[None] >= arrival[:, None]).sum(0)
    surface = 2 * sc.W * sc.D + 2 * (sc.W + sc.D) * sc.H
    A0 = rng.uniform(0.1, 0.3) * surface
    absorb_t = np.sqrt(A0 / (A0 + n_present * 0.4))[:, None]
    res = {k: [] for k in ("cal_mean", "cal_std", "ses_mean", "ses_std")}
    for (i, j) in sc.links:
        a, b = sc.nodes[i], sc.nodes[j]
        paths = link_paths3(a, b, sc.W, sc.D, sc.H, sc.floor, sc.gam_walls) + diffuse_paths(a, b, sc.W, sc.D, sc.H, rng, 8)
        rest = static_rest(a, b, sc.clutter, FREQS52, rng, 0)
        Hs = link_channel(a, b, paths, rest, BB, T, FREQS52, 1.0)
        Hs += (rest[None] * (absorb_t - 1)).astype(np.complex64)
        Hc = link_channel(a, b, paths, rest, None, Tc, FREQS52)
        acc = {k: [] for k in res}
        for (tx, rx) in ((i, j), (j, i)):
            for tag, H in (("cal", Hc), ("ses", Hs)):
                iq, rssi, kept = measure(H, rng, radio["ptx"][tx], radio["evm"][rx], radio["agc"][rx], radio["loss"], radio["interf"], fs)
                m, s, _ = window_features(iq[kept], rssi[kept], np.nonzero(kept)[0] / fs, WIN_S, H.shape[0] / fs)
                acc[f"{tag}_mean"].append(m); acc[f"{tag}_std"].append(s)
        for k in res:
            with np.errstate(all="ignore"):
                res[k].append(np.nanmean(np.stack(acc[k]), 0))
    out = {k: np.stack(v).astype(np.float32) for k, v in res.items()}
    W = out["ses_mean"].shape[1]
    present = np.zeros((W, len(sc.seats)), bool)
    for q, t0 in enumerate(arrival):
        present[np.arange(W) * fs * WIN_S >= t0, idx[q]] = True
    out["present"] = present
    return out
