"""3-D propagation with human bodies.

Paths:   LOS, 4 wall reflections (image method), floor and ceiling reflections, all as 3-D polylines.
Bodies:  boxes (shoulder half-width a, half-depth b, vertical extent z0..z1, yaw) that can sit,
         stand, walk or lie down. Shadowing of every path segment uses exact Fresnel-Kirchhoff
         diffraction by a RECTANGULAR absorber (separable in the two transverse directions):
             E/E0 = 1 - eta * (j/2) * I(u1,u2) * I(v1,v2),   I(a,b) = int_a^b exp(-j pi t^2 / 2) dt
         so a link that passes over a seated student's head is attenuated far less than one through
         the torso, which is what makes tiered lecture halls (and nodes at different heights) work.
Scatter: torso and limb point scatterers (bistatic radar equation) with breathing and fidget motion.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import fresnel

from .config import C, FC, LAM

C_LIGHT = C


@dataclass
class Bodies:
    c: np.ndarray       # [Q,T,2] horizontal centre (NaN = absent)
    z0: np.ndarray      # [Q,T] bottom of the blocking box (m)
    z1: np.ndarray      # [Q,T] top
    a: np.ndarray       # [Q,T] half-size along body x (shoulders, or half body length when lying)
    b: np.ndarray       # [Q,T] half-size along body y (chest depth; breathing modulates it)
    yaw: np.ndarray     # [Q,T]
    eta: np.ndarray     # [Q] absorption strength (0..1)
    chest: np.ndarray   # [Q,T,3] torso scatter point (breathing displaces it)
    limb: np.ndarray    # [Q,T,3] hand/limb scatter point
    rcs: np.ndarray     # [Q]
    limb_rcs: np.ndarray  # [Q]

    @property
    def Q(self):
        return self.c.shape[0]


def concat(bs):
    bs = [b for b in bs if b is not None and b.Q]
    if not bs:
        return None
    return Bodies(*[np.concatenate([getattr(b, f) for b in bs], 0) for f in Bodies.__dataclass_fields__])


@dataclass
class Path3:
    pts: np.ndarray     # [m,3] polyline a -> ... -> b
    gain: complex
    eta_scale: float = 1.0
    shadow: bool = True

    @property
    def length(self):
        return float(np.sum(np.linalg.norm(np.diff(self.pts, axis=0), axis=1)))


def link_paths3(a, b, W, D, H, floor=True, gam_walls=(-0.4, -0.4, -0.4, -0.4), gam_floor=-0.5, gam_ceil=-0.35):
    paths = [Path3(np.stack([a, b]), 1.0 + 0j)]
    for w, (ax, pos) in enumerate([(0, 0.0), (0, W), (1, 0.0), (1, D)]):
        img = a.copy(); img[ax] = 2 * pos - a[ax]
        t = (pos - img[ax]) / (b[ax] - img[ax] + 1e-12)
        P = img + t * (b - img); P[ax] = pos
        paths.append(Path3(np.stack([a, P, b]), complex(gam_walls[w])))
    for z, g, sh, es in (((0.0, gam_floor, True, 0.7) if floor else (None, None, None, None)), (H, gam_ceil, True, 1.0)):
        if z is None:
            continue
        img = a.copy(); img[2] = 2 * z - a[2]
        t = (z - img[2]) / (b[2] - img[2] + 1e-12)
        P = img + t * (b - img); P[2] = z
        paths.append(Path3(np.stack([a, P, b]), complex(g), eta_scale=es, shadow=sh))
    return paths


def path_response(L, gain, freqs):
    return gain * LAM / (4 * np.pi * L) * np.exp(-2j * np.pi * freqs * L / C_LIGHT)


def _I(n1, n2):
    S1, C1 = fresnel(n1); S2, C2 = fresnel(n2)
    return (C2 - C1) - 1j * (S2 - S1)


def segment_shadow(P0, P1, cum0, Ltot, B: Bodies, eta_scale=1.0, prune=3.5):
    """product over bodies of the rectangular-absorber diffraction factor on one segment -> [T] complex"""
    seg = P1 - P0
    Ls = np.linalg.norm(seg)
    if Ls < 1e-6:
        return None
    u = seg / Ls
    nh = np.cross(u, [0, 0, 1.0]); nh /= np.linalg.norm(nh) + 1e-12
    nv = np.cross(nh, u)
    zc = 0.5 * (B.z0 + B.z1)
    rel = np.stack([B.c[..., 0] - P0[0], B.c[..., 1] - P0[1], zc - P0[2]], -1)   # [Q,T,3]
    tt = rel @ u; hh = rel @ nh; vv = rel @ nv
    ex = np.stack([np.cos(B.yaw), np.sin(B.yaw), 0 * B.yaw], -1) * B.a[..., None]
    ey = np.stack([-np.sin(B.yaw), np.cos(B.yaw), 0 * B.yaw], -1) * B.b[..., None]
    hz = 0.5 * (B.z1 - B.z0)
    wh = np.sqrt((ex @ nh) ** 2 + (ey @ nh) ** 2) + np.abs(hz * nh[2])
    wv = np.abs(hz * nv[2]) + np.sqrt((ex @ nv) ** 2 + (ey @ nv) ** 2)
    valid = np.isfinite(tt) & (tt > 0.05) & (tt < Ls - 0.05)
    d1 = cum0 + np.where(valid, tt, Ls / 2)
    kf = np.sqrt(2 * Ltot / (LAM * d1 * np.maximum(Ltot - d1, 1e-3)))
    gap = np.maximum(np.abs(hh) - wh, np.abs(vv) - wv) * kf
    near = valid & (gap < prune)
    keep = near.any(1)
    if not keep.any():
        return None
    hh, vv, wh, wv, kf, near = hh[keep], vv[keep], wh[keep], wv[keep], kf[keep], near[keep]
    eta = (B.eta[keep] * eta_scale)[:, None]
    hh = np.where(near, hh, 1e3); vv = np.where(near, vv, 1e3)
    F = 1 - eta * 0.5j * _I((hh - wh) * kf, (hh + wh) * kf) * _I((vv - wv) * kf, (vv + wv) * kf)
    F = np.where(near, F, 1.0)
    return np.prod(F, 0)


def scatter_series(a, b, P, rcs, freqs):
    """point scatterers P [Q,T,3] -> [T,K]"""
    present = np.isfinite(P[..., 0])
    if not present.any():
        return 0.0
    P0 = np.where(present[..., None], P, 0.0)
    cnt = np.maximum(present.sum(1), 1)[:, None]
    base = P0.sum(1) / cnt
    d1 = np.linalg.norm(base - a, axis=-1); d2 = np.linalg.norm(base - b, axis=-1)
    L0 = d1 + d2
    G = (LAM * np.sqrt(rcs) / ((4 * np.pi) ** 1.5 * d1 * d2))[:, None] * np.exp(-2j * np.pi * freqs[None] * L0[:, None] / C_LIGHT)
    L = np.linalg.norm(P0 - a, axis=-1) + np.linalg.norm(P0 - b, axis=-1)
    E = np.exp(-2j * np.pi * FC * (L - L0[:, None]) / C_LIGHT) * (L0[:, None] / np.maximum(L, 0.3)) * present
    return E.T.astype(np.complex64) @ G.astype(np.complex64)


def link_channel(a, b, paths, rest, B: Bodies, T, freqs, absorb=1.0):
    """H(t,k) for one link: shadowed geometric paths + (absorbed) static rest + body scatter"""
    K = len(freqs)
    H = np.zeros((T, K), np.complex64) + (rest * absorb)[None].astype(np.complex64)
    for p in paths:
        hp = path_response(p.length, p.gain, freqs)
        F = np.ones(T, np.complex128)
        if B is not None and p.shadow:
            cum = 0.0
            for s in range(len(p.pts) - 1):
                f = segment_shadow(p.pts[s], p.pts[s + 1], cum, p.length, B, p.eta_scale)
                if f is not None:
                    F *= f
                cum += np.linalg.norm(p.pts[s + 1] - p.pts[s])
        H += (F[:, None] * hp[None]).astype(np.complex64)
    if B is not None:
        H += absorb * scatter_series(a, b, B.chest, B.rcs, freqs)
        H += absorb * scatter_series(a, b, B.limb, B.limb_rcs, freqs)
    return H


def static_rest(a, b, clutter, freqs, rng, d_los):
    """furniture point scatterers (static) -> [K]"""
    out = np.zeros(len(freqs), np.complex128)
    if clutter is not None and len(clutter):
        p = clutter[:, :3]; r = clutter[:, 3]
        d1 = np.linalg.norm(p - a, axis=1); d2 = np.linalg.norm(p - b, axis=1)
        out += ((LAM * np.sqrt(r) / ((4 * np.pi) ** 1.5 * d1 * d2))[:, None] *
                np.exp(-2j * np.pi * freqs[None] * (d1 + d2)[:, None] / C_LIGHT)).sum(0)
    return out


def diffuse_paths(a, b, W, D, H, rng, n=24):
    """the reverberant field as n paths a -> random point in the room -> b.
    Their power follows an exponential delay profile (Rician K random) and, unlike a
    stochastic channel model, each one has a geometry, so a person walking ANYWHERE in the
    room shadows some of them. That is why real single-link sensors see motion room-wide."""
    d = np.linalg.norm(b - a)
    via = np.c_[rng.uniform(0.1, W - 0.1, n), rng.uniform(0.1, D - 0.1, n), rng.uniform(0.2, H - 0.2, n)]
    L = np.linalg.norm(via - a, axis=1) + np.linalg.norm(via - b, axis=1)
    Kdb = rng.uniform(-3, 6)
    pw = np.exp(-(L - d) / C_LIGHT / rng.uniform(15e-9, 40e-9))
    pw = pw / pw.sum() * (LAM / (4 * np.pi * d)) ** 2 * 10 ** (-Kdb / 10)
    amp = np.sqrt(pw) * np.exp(2j * np.pi * rng.random(n))     # complex gain relative to free-space at L
    out = []
    for k in range(n):
        g = amp[k] / (LAM / (4 * np.pi * L[k]))
        out.append(Path3(np.stack([a, via[k], b]), complex(g), eta_scale=1.0))
    return out
