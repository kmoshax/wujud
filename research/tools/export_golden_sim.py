"""Golden data for crates/sim: the deterministic physics of the Python simulator, exactly.

Random draws are recorded with RecordingRng (duck-types numpy's Generator) so the Rust tests can inject the
very numbers numpy drew into the deterministic part (diffuse_paths, measure).

Run from research/:  .venv/bin/python tools/export_golden_sim.py
Writes crates/sim/tests/golden/{physics.json, room_channel.npz, hall_channel.npz, measure_*.npz}.
"""
import json, os, sys
import numpy as np
from scipy.special import fresnel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from wujud.config import LAM
from wujud.physics3d import (link_paths3, path_response, segment_shadow, scatter_series, link_channel,
                                 static_rest, diffuse_paths, concat)
from wujud.scenes3d import random_room, random_hall3, room_person, seated_bodies, lecturer, distractor
from wujud.sim3d import measure, FREQS52, FREQS56

OUT = os.path.join(HERE, "..", "crates", "sim", "tests", "golden")


class RecordingRng:
    def __init__(self, rng):
        self.rng, self.calls = rng, []

    def _record(self, name, value):
        self.calls.append((name, np.asarray(value, float)))
        return value

    def normal(self, *args, **kwargs):
        return self._record("normal", self.rng.normal(*args, **kwargs))

    def uniform(self, *args, **kwargs):
        return self._record("uniform", self.rng.uniform(*args, **kwargs))

    def random(self, *args, **kwargs):
        return self._record("random", self.rng.random(*args, **kwargs))


def cplx(z):
    z = np.asarray(z, np.complex128)
    return np.stack([z.real, z.imag], -1)


def path_json(p):
    return dict(points=p.pts.tolist(), gain=[p.gain.real, p.gain.imag], eta_scale=p.eta_scale, shadow=bool(p.shadow),
                response56=cplx(path_response(p.length, p.gain, FREQS56)).tolist())


def fresnel_cases():
    x = np.r_[np.linspace(-6, 6, 241), 0.0, 1e-10, 1.4999, 1.5, 1.5001, 10.0, 50.0, 300.0, -0.7]
    s, c = fresnel(x)
    d1 = np.array([0.1, 1.0, 2.5, 4.9, 5.0]); total = 5.0
    kf = np.sqrt(2 * total / (LAM * d1 * np.maximum(total - d1, 1e-3)))
    return dict(x=x.tolist(), s=s.tolist(), c=c.tolist(), scale_from_tx=d1.tolist(), scale_total=total, scale=kf.tolist())


def geometric_cases():
    cases = []
    for a, b, W, D, H, floor, gam in (([0.08, 1.2, 0.9], [4.1, 3.3, 1.1], 4.2, 4.0, 2.8, True, (-0.4, -0.3, -0.5, -0.6)),
                                      ([3.0, 0.1, 1.2], [12.9, 9.0, 2.6], 13.0, 12.0, 6.1, False, (-0.25, -0.5, -0.3, -0.45))):
        a, b = np.array(a), np.array(b)
        cases.append(dict(a=a.tolist(), b=b.tolist(), width=W, depth=D, height=H, floor=floor, wall_gains=list(gam),
                          paths=[path_json(p) for p in link_paths3(a, b, W, D, H, floor, gam)]))
    return cases


def diffuse_case(seed=11):
    a, b, W, D, H = np.array([0.08, 1.2, 0.9]), np.array([4.1, 3.3, 1.1]), 4.2, 4.0, 2.8
    rec = RecordingRng(np.random.default_rng(seed))
    paths = diffuse_paths(a, b, W, D, H, rec, 24)
    draws = [v for _, v in rec.calls]
    return dict(a=a.tolist(), b=b.tolist(), via=np.c_[draws[0], draws[1], draws[2]].tolist(),
                rician_k_db=float(draws[3]), delay_spread_s=float(draws[4]), phase_turns=draws[5].tolist(),
                paths=[path_json(p) for p in paths])


def static_rest_case(seed=5):
    sc = random_room(np.random.default_rng(seed))
    a, b = sc.nodes[0], sc.nodes[1]
    return dict(a=a.tolist(), b=b.tolist(), clutter=sc.clutter.tolist(),
                rest56=cplx(static_rest(a, b, sc.clutter, FREQS56, None, 0)).tolist())


def quantize_case():
    v = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49999, 126.5, 127.4, 127.6, 300.0, -126.5, -127.5, -300.0, 3.7, -3.7])
    return dict(values=v.tolist(), int8=np.clip(np.round(v), -127, 127).astype(np.int8).tolist())


def body_arrays(B):
    return dict(c=B.c, z0=B.z0, z1=B.z1, half_a=B.a, half_b=B.b, yaw=B.yaw, eta=B.eta, chest=B.chest, limb=B.limb,
                rcs=B.rcs, limb_rcs=B.limb_rcs)


def path_arrays(paths):
    pts = np.full((len(paths), 3, 3), np.nan)
    for k, p in enumerate(paths):
        pts[k, :len(p.pts)] = p.pts
    return dict(path_points=pts, path_point_count=np.array([len(p.pts) for p in paths]),
                path_gain=cplx([p.gain for p in paths]), path_eta_scale=np.array([p.eta_scale for p in paths]),
                path_shadow=np.array([p.shadow for p in paths], np.float64))


def segment_cases(B, nodes, links, paths_of):
    """every segment of every path of the given links, with the [T] diffraction product (1 when pruned)"""
    starts, ends, offsets, totals, etas, shadows = [], [], [], [], [], []
    for (i, j) in links:
        for p in paths_of(i, j):
            cum = 0.0
            for s in range(len(p.pts) - 1):
                f = segment_shadow(p.pts[s], p.pts[s + 1], cum, p.length, B, p.eta_scale)
                starts.append(p.pts[s]); ends.append(p.pts[s + 1]); offsets.append(cum); totals.append(p.length)
                etas.append(p.eta_scale); shadows.append(np.ones(B.c.shape[1], complex) if f is None else f)
                cum += np.linalg.norm(p.pts[s + 1] - p.pts[s])
    return dict(seg_start=np.array(starts), seg_end=np.array(ends), seg_offset=np.array(offsets),
                seg_total=np.array(totals), seg_eta_scale=np.array(etas), seg_shadow=cplx(np.array(shadows)))


def room_channel_case(seed=21, fs=50.0, seconds=8):
    rng = np.random.default_rng(seed)
    sc = random_room(rng, n_esp=2, mesh=True, smart=True)
    T = int(seconds * fs)
    B, _ = room_person(sc, T, rng, fs, plan=[("walk", 3), ("sit", 2), ("lie", 2), ("away", 1)])
    B = concat([B, distractor("fan", sc, T, rng, fs), distractor("door", sc, T, rng, fs)])
    i, j = sc.links[0]
    a, b = sc.nodes[i], sc.nodes[j]
    paths = link_paths3(a, b, sc.W, sc.D, sc.H, sc.floor, sc.gam_walls) + diffuse_paths(a, b, sc.W, sc.D, sc.H, rng, 24)
    rest = static_rest(a, b, sc.clutter, FREQS56, rng, 0)
    H = link_channel(a, b, paths, rest, B, T, FREQS56, 1.0)
    absorb = 0.93
    H_absorb = link_channel(a, b, paths, rest, B, T, FREQS56, absorb)
    seg = segment_cases(B, sc.nodes, sc.links,
                        lambda i, j: link_paths3(sc.nodes[i], sc.nodes[j], sc.W, sc.D, sc.H, sc.floor, sc.gam_walls))
    np.savez_compressed(os.path.join(OUT, "room_channel.npz"), tx=a, rx=b, freqs=FREQS56, rest=cplx(rest), absorb=absorb,
                        H=cplx(H), H_absorb=cplx(H_absorb), chest_scatter=cplx(scatter_series(a, b, B.chest, B.rcs, FREQS56)),
                        limb_scatter=cplx(scatter_series(a, b, B.limb, B.limb_rcs, FREQS56)),
                        **body_arrays(B), **path_arrays(paths), **seg)
    return H


def hall_channel_case(seed=3, fs=20.0, seconds=5, students=40):
    rng = np.random.default_rng(seed)
    sc = random_hall3(rng, n_nodes=6, tiered=True, placement="walls+high", W=13.0, D=12.0)
    T = int(seconds * fs)
    seats = sc.seats[rng.choice(len(sc.seats), students, replace=False)]
    L = None
    while L is None:
        L = lecturer(sc, T, rng, fs)
    B = concat([seated_bodies(seats, T, rng, fs, restless=4.0), L])
    B.c[0, :T // 2] = np.nan; B.chest[0, :T // 2] = np.nan; B.limb[0, :T // 2] = np.nan
    i, j = sc.links[1]
    a, b = sc.nodes[i], sc.nodes[j]
    paths = link_paths3(a, b, sc.W, sc.D, sc.H, sc.floor, sc.gam_walls) + diffuse_paths(a, b, sc.W, sc.D, sc.H, rng, 8)
    rest = static_rest(a, b, sc.clutter, FREQS52, rng, 0)
    absorb = 0.81
    H = link_channel(a, b, paths, rest, B, T, FREQS52, absorb)
    seg = segment_cases(B, sc.nodes, sc.links[:5],
                        lambda i, j: link_paths3(sc.nodes[i], sc.nodes[j], sc.W, sc.D, sc.H, sc.floor, sc.gam_walls))
    np.savez_compressed(os.path.join(OUT, "hall_channel.npz"), tx=a, rx=b, freqs=FREQS52, rest=cplx(rest), absorb=absorb,
                        H_absorb=cplx(H), chest_scatter=cplx(scatter_series(a, b, B.chest, B.rcs, FREQS52)),
                        limb_scatter=cplx(scatter_series(a, b, B.limb, B.limb_rcs, FREQS52)),
                        **body_arrays(B), **path_arrays(paths), **seg)


def measure_case(name, H, seed, ptx, evm, agc, loss, interf, fs):
    rec = RecordingRng(np.random.default_rng(seed))
    iq, rssi, kept = measure(H, rec, ptx, evm, agc, loss, interf, fs)
    calls = [v for _, v in rec.calls]
    noise_re, noise_im, interference_draw = calls[:3]
    bursts = int((interference_draw < interf).sum())
    if bursts:
        burst_db, burst_re, burst_im = calls[3:6]; rest = calls[6:]
    else:
        burst_db, burst_re, burst_im = np.zeros(0), np.zeros((0, H.shape[1])), np.zeros((0, H.shape[1])); rest = calls[3:]
    agc_step_draw, rssi_bias_levels, rssi_jitter, agc_gain_db, loss_draw = rest
    np.savez_compressed(os.path.join(OUT, f"measure_{name}.npz"), H=cplx(H), radio=np.array([ptx, evm, agc, loss, interf]), fs=fs,
                        noise_re=noise_re, noise_im=noise_im, interference_draw=interference_draw, burst_db=burst_db,
                        burst_re=burst_re, burst_im=burst_im, agc_step_draw=agc_step_draw, rssi_bias_levels=rssi_bias_levels,
                        rssi_jitter=rssi_jitter, agc_gain_db=agc_gain_db, loss_draw=loss_draw,
                        iq=iq.astype(np.float64), rssi=rssi.astype(np.float64), kept=kept.astype(np.float64))
    return bursts


def main():
    os.makedirs(OUT, exist_ok=True)
    physics = dict(fresnel=fresnel_cases(), geometric=geometric_cases(), diffuse=diffuse_case(),
                   static_rest=static_rest_case(), quantize=quantize_case(), wavelength=LAM)
    json.dump(physics, open(os.path.join(OUT, "physics.json"), "w"))
    H = room_channel_case()
    hall_channel_case()
    long_H = np.tile(H[:200], (3, 1))
    n1 = measure_case("typical", H, 1, 9.5, -26.0, 18.0, 0.04, 0.02, 50.0)
    n2 = measure_case("harsh", long_H, 2, 4.2, -21.0, 13.0, 0.2, 0.3, 2.0)
    sizes = {f: os.path.getsize(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT))}
    print(f"bursts typical={n1} harsh={n2}", sizes, f"total {sum(sizes.values()) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
