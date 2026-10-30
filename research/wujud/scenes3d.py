"""3-D scenes: a bedroom (1 person, router + 1..N ESP32) and lecture halls (flat or tiered).

Everything is generated at random within realistic ranges so the models never see
the same room twice, and the test rooms are disjoint from the training rooms.
"""
from dataclasses import dataclass, field
import numpy as np

from .physics3d import Bodies

ACTIVITIES = ["away", "lie", "sit", "stand", "walk"]


@dataclass
class Scene:
    kind: str                     # "room" | "hall"
    W: float
    D: float
    H: float
    nodes: np.ndarray             # [N,3]
    links: list                   # [(i,j)]
    clutter: np.ndarray           # [M,4] x,y,z,rcs
    furniture: list = field(default_factory=list)   # boxes for the 3-D view: dict(x,y,z,w,d,h,yaw,kind)
    seats: np.ndarray = None      # [S,3] seat surface points (hall)
    seat_rc: np.ndarray = None
    tiers: dict = None            # hall: {"y0":..,"rise":..,"pitch":..,"front":..}
    floor: bool = True            # flat floor reflection exists
    gam_walls: tuple = (-0.4, -0.4, -0.4, -0.4)
    router: int = None            # room: index of the router node (transmit-only)
    meta: dict = field(default_factory=dict)

    def floor_z(self, y):
        if not self.tiers or self.tiers["rise"] <= 0:
            return np.zeros_like(np.asarray(y, float))
        t = self.tiers
        r = np.floor((np.asarray(y, float) - t["y0"]) / t["pitch"] + 0.5)
        return np.clip(r, 0, None) * t["rise"]


def _wall_point(W, D, rng, side=None, inset=0.08):
    side = rng.integers(4) if side is None else side
    if side == 0:
        return np.array([rng.uniform(0.3, W - 0.3), inset])
    if side == 1:
        return np.array([W - inset, rng.uniform(0.3, D - 0.3)])
    if side == 2:
        return np.array([rng.uniform(0.3, W - 0.3), D - inset])
    return np.array([inset, rng.uniform(0.3, D - 0.3)])


def random_room(rng, n_esp=1, mesh=False, smart=False):
    """bedroom/study. n_esp ESP32 boards + a router. smart=True: placement rule
    (router and first board on opposite sides of the bed, so the link crosses it)."""
    W, D, H = rng.uniform(3.0, 5.5), rng.uniform(3.0, 5.0), rng.uniform(2.7, 3.2)
    bw, bl = rng.uniform(0.9, 1.6), rng.uniform(1.9, 2.1)
    # bed against a side wall, head to the back wall
    bx = bw / 2 + 0.02 if rng.random() < 0.5 else W - bw / 2 - 0.02
    by = D - bl / 2 - 0.02
    bed = dict(kind="bed", x=bx, y=by, z=0.0, w=bw, d=bl, h=0.5, yaw=0.0)
    dx = W - 0.6 if bx < W / 2 else 0.6
    desk = dict(kind="desk", x=dx, y=rng.uniform(0.6, D * 0.45), z=0.0, w=0.6, d=1.2, h=0.75, yaw=0.0)
    chair = np.array([dx + (-0.55 if dx > W / 2 else 0.55), desk["y"]])
    ward = dict(kind="wardrobe", x=rng.uniform(1.0, W - 1.0), y=0.32, z=0.0, w=1.0, d=0.6, h=2.0, yaw=0.0)
    furn = [bed, desk, ward]
    if smart:
        # router on the wall beside the bed, ESP32 on the opposite wall, both just above mattress height
        rpos = np.array([0.08 if bx < W / 2 else W - 0.08, by])
        epos = np.array([W - 0.08 if bx < W / 2 else 0.08, by + rng.normal(0, 0.3)])
        router = np.r_[rpos, rng.uniform(0.7, 1.0)]
        esp = [np.r_[epos, rng.uniform(0.7, 1.0)]]
    else:
        router = np.r_[_wall_point(W, D, rng), rng.uniform(0.8, 2.0)]
        esp = [np.r_[_wall_point(W, D, rng), rng.uniform(0.5, 1.5)]]
    for _ in range(n_esp - 1):
        esp.append(np.r_[_wall_point(W, D, rng), rng.uniform(0.5, 1.5)])
    nodes = np.vstack([router] + esp)
    links = [(0, k) for k in range(1, len(nodes))]
    if mesh:
        links += [(i, j) for i in range(1, len(nodes)) for j in range(i + 1, len(nodes))]
    # furniture as scatterers (metal frames, monitor, wardrobe)
    cl = [[bx, by, 0.4, rng.uniform(0.1, 0.5)], [dx, desk["y"], 0.9, rng.uniform(0.1, 0.6)],
          [ward["x"], ward["y"], 1.0, rng.uniform(0.3, 1.5)]]
    for _ in range(rng.integers(3, 10)):
        cl.append([rng.uniform(0.2, W - 0.2), rng.uniform(0.2, D - 0.2), rng.uniform(0, 2), rng.uniform(0.01, 0.2)])
    gam = tuple(rng.uniform(-0.6, -0.25, 4))
    return Scene("room", W, D, H, nodes, links, np.array(cl), furn, floor=True, gam_walls=gam, router=0,
                 meta=dict(bed=bed, desk=desk, chair=chair, smart=smart))


def random_hall3(rng, n_nodes=12, tiered=None, placement=None, W=None, D=None):
    W = W or rng.uniform(9.0, 18.0); D = D or rng.uniform(8.0, 16.0); H = rng.uniform(3.2, 5.0)
    tiered = (rng.random() < 0.6) if tiered is None else tiered
    front = rng.uniform(2.5, 3.5)
    pitch = rng.uniform(0.85, 1.10); seat_pitch = rng.uniform(0.55, 0.70)
    rise = rng.uniform(0.15, 0.40) if tiered else 0.0
    y0 = front + 0.9
    n_aisles = int(rng.integers(1, 3)) if W > 11 else 1
    x0, x1 = 0.9, W - 0.9
    cuts = np.sort(rng.uniform(0.3, 0.7, n_aisles)) if n_aisles == 1 else np.array([0.33, 0.67]) + rng.normal(0, 0.03, 2)
    aisles = list(x0 + cuts * (x1 - x0))
    seats, rc = [], []
    ys = np.arange(y0, D - 0.6, pitch)
    for r, y in enumerate(ys):
        c = 0
        for x in np.arange(x0, x1 + 1e-6, seat_pitch):
            if any(abs(x - a) < 0.5 for a in aisles):
                continue
            seats.append((x, y, r * rise)); rc.append((r, c)); c += 1
    seats = np.array(seats); rc = np.array(rc)
    sc = Scene("hall", W, D, H + len(ys) * rise, None, None, None, seats=seats, seat_rc=rc,
               tiers=dict(y0=y0, rise=rise, pitch=pitch, front=front), floor=not tiered,
               gam_walls=tuple(rng.uniform(-0.5, -0.25, 4)), meta=dict(aisles=aisles, tiered=tiered))
    placement = placement or rng.choice(["walls", "walls+high"])
    # nodes on the 4 walls (perimeter), at seated-chest height above the local floor,
    # or mixed with some high mounts (projector shelf / upper wall) when "walls+high"
    P = 2 * (W + D)
    s = (np.arange(n_nodes) + rng.uniform(0.2, 0.8)) / n_nodes * P
    s = (s + rng.normal(0, 0.12 * P / n_nodes, n_nodes)) % P
    pts = []
    for v in s:
        if v < W: p = (v, 0.1)
        elif v < W + D: p = (W - 0.1, v - W)
        elif v < 2 * W + D: p = (W - (v - W - D), D - 0.1)
        else: p = (0.1, D - (v - 2 * W - D))
        pts.append(p)
    pts = np.array(pts)
    zf = sc.floor_z(pts[:, 1])
    z = zf + rng.uniform(1.0, 1.3, n_nodes)
    if placement == "walls+high":
        hi = rng.random(n_nodes) < 0.35
        z[hi] = zf[hi] + rng.uniform(2.2, 2.8, hi.sum())
    sc.nodes = np.c_[pts, z]
    sc.links = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    S = len(seats)
    chairs = np.c_[seats[:, 0], seats[:, 1], seats[:, 2] + 0.45, rng.uniform(0.005, 0.03, S)]
    nr = rng.integers(15, 40)
    objs = np.c_[rng.uniform(0.3, W - 0.3, nr), rng.uniform(0.3, D - 0.3, nr), rng.uniform(0, 2.5, nr), rng.uniform(0.02, 0.4, nr)]
    sc.clutter = np.vstack([chairs, objs])
    sc.meta["placement"] = placement
    return sc


# ---------------------------------------------------------------------------
# people
# ---------------------------------------------------------------------------
def _ou(rng, shape, sigma, tau, fs):
    from scipy.signal import lfilter
    rho = np.exp(-1 / (tau * fs))
    e = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), shape)
    return lfilter([1], [1, -rho], e, axis=-1)


def seated_bodies(seats_xyz, T, rng, fs, restless=1.0):
    """students sitting in hall seats (facing the front, -y)"""
    Q = len(seats_xyz)
    t = np.arange(T) / fs
    c = seats_xyz[:, None, :2] + rng.normal(0, 0.04, (Q, 1, 2)) + np.moveaxis(_ou(rng, (Q, 2, T), 0.004, 3.0, fs), 1, 2)
    zf = seats_xyz[:, 2][:, None] * np.ones((1, T))
    fb = rng.uniform(0.2, 0.33, Q)[:, None]
    br = rng.uniform(0.003, 0.006, Q)[:, None] * np.sin(2 * np.pi * fb * t + rng.uniform(0, 6.3, (Q, 1)))
    limb = np.concatenate([c + np.stack([rng.uniform(-0.25, 0.25, Q), -rng.uniform(0.15, 0.35, Q)], 1)[:, None], (zf + 0.8)[..., None]], -1)
    rate = restless * np.exp(rng.normal(np.log(1 / 20), 0.7, Q))
    for q in range(Q):
        for _ in range(rng.poisson(rate[q] * T / fs)):
            t0 = rng.integers(0, T); dur = int(rng.uniform(0.8, 4.0) * fs)
            idx = np.arange(t0, min(T, t0 + dur))
            if len(idx) < 2:
                continue
            ph = (idx - t0) / dur
            ang = rng.uniform(0, 2 * np.pi); vec = rng.uniform(0.02, 0.15) * np.array([np.cos(ang), np.sin(ang)])
            c[q, idx] += (np.sin(np.pi * ph) ** 2)[:, None] * vec
            osc = rng.uniform(0.05, 0.3) * np.sin(2 * np.pi * rng.uniform(0.5, 2) * (idx - t0) / fs) * np.sin(np.pi * ph)
            a2 = rng.uniform(0, 2 * np.pi)
            limb[q, idx, 0] += osc * np.cos(a2); limb[q, idx, 1] += osc * np.sin(a2)
    chest = np.concatenate([c - np.stack([0 * br, br], -1), (zf + 1.0)[..., None]], -1)
    return Bodies(c, zf + 0.40, zf + 1.30, np.full((Q, T), 0.21) + 0 * br, 0.13 + br, np.zeros((Q, T)),
                  rng.uniform(0.75, 0.97, Q), chest, limb, rng.uniform(0.1, 0.5, Q), rng.uniform(0.02, 0.1, Q))


def lecturer(sc: Scene, T, rng, fs):
    if rng.random() > 0.75:
        return None
    front = sc.tiers["front"]
    p = np.array([rng.uniform(1, sc.W - 1), rng.uniform(0.6, front - 0.3)]); tgt = p.copy()
    v = rng.uniform(0.3, 0.9); tr = np.zeros((T, 2)); pause = 0
    for i in range(T):
        if pause > 0:
            pause -= 1
        else:
            d = tgt - p; n = np.linalg.norm(d)
            if n < 0.1:
                tgt = np.array([rng.uniform(1, sc.W - 1), rng.uniform(0.6, front - 0.3)]); pause = int(rng.exponential(4.0) * fs)
            else:
                p = p + d / n * min(n, v / fs)
        tr[i] = p
    return _standing(tr[None], rng, fs, walking=True)


def _standing(c, rng, fs, walking=False, zf=None):
    Q, T, _ = c.shape
    t = np.arange(T) / fs
    zf = np.zeros((Q, T)) if zf is None else zf
    br = rng.uniform(0.003, 0.006, (Q, 1)) * np.sin(2 * np.pi * rng.uniform(0.2, 0.33, (Q, 1)) * t)
    vel = np.zeros((Q, T, 2)); vel[:, 1:] = np.diff(c, axis=1) * fs
    yaw = np.arctan2(vel[..., 1], vel[..., 0]) - np.pi / 2
    sp = np.linalg.norm(vel, axis=-1)
    swing = 0.25 * np.sin(2 * np.pi * np.maximum(sp, 0.3) / 0.8 * t) * (sp > 0.1)
    fwd = np.stack([np.cos(yaw + np.pi / 2), np.sin(yaw + np.pi / 2)], -1)
    limb = np.concatenate([c + fwd * swing[..., None], (zf + 0.9)[..., None]], -1)
    chest = np.concatenate([c + fwd * br[..., None], (zf + 1.3)[..., None]], -1)
    return Bodies(c, zf + 0.0, zf + 1.70, np.full((Q, T), 0.21), 0.13 + br, yaw, np.full(Q, 0.9), chest, limb,
                  np.full(Q, 0.6), np.full(Q, 0.1))


def room_person(sc: Scene, T, rng, fs, plan=None, rcs_scale=1.0, breath_scale=1.0):
    """one person following an activity plan [(activity, seconds), ...] -> Bodies [1,T], labels.
    rcs_scale / breath_scale: realism knobs (torso reflection strength, chest excursion), see bench/sim_calibrate.py"""
    if plan is None:
        plan, tt = [], 0
        while tt < T / fs:
            a = rng.choice(ACTIVITIES, p=[0.15, 0.25, 0.25, 0.15, 0.2])
            d = rng.uniform(20, 90); plan.append((a, d)); tt += d
    W, D = sc.W, sc.D
    bed, chair = sc.meta["bed"], sc.meta["chair"]
    c = np.full((T, 2), np.nan); z0 = np.zeros(T); z1 = np.zeros(T); a = np.full(T, 0.21); b = np.full(T, 0.13)
    yaw = np.zeros(T); chest = np.full((T, 3), np.nan); limb = np.full((T, 3), np.nan)
    act = np.zeros(T, int); t = np.arange(T) / fs
    f_b = rng.uniform(0.2, 0.33); A_b = rng.uniform(0.004, 0.008) * breath_scale
    br = A_b * np.sin(2 * np.pi * f_b * t + rng.uniform(0, 6.3))
    pos = np.array([W / 2, D / 2])
    i = 0
    for (name, dur) in plan:
        n = min(T - i, int(dur * fs))
        if n <= 0:
            break
        s = slice(i, i + n); act[s] = ACTIVITIES.index(name)
        sway = np.moveaxis(_ou(rng, (2, n), 0.003, 3.0, fs), 0, 1)
        if name == "away":
            pass
        elif name == "lie":
            p = np.array([bed["x"] + rng.normal(0, bed["w"] * 0.15), bed["y"] + rng.normal(0, 0.1)])
            c[s] = p + sway; z0[s] = 0.5; z1[s] = 0.75 + br[s]; a[s] = 0.9; b[s] = 0.22; yaw[s] = np.pi / 2
            chest[s] = np.c_[c[s], 0.7 + br[s]]; limb[s] = np.c_[c[s] + [0.25, 0.3], np.full(n, 0.6)]
            pos = p
        elif name in ("sit", "stand"):
            if name == "sit":
                p = chair.copy() if rng.random() < 0.7 else np.array([bed["x"], bed["y"] - bed["d"] / 2 + 0.3])
                z0[s] = 0.4; z1[s] = 1.3; cz = 1.0
            else:
                p = np.array([rng.uniform(0.6, W - 0.6), rng.uniform(0.6, D - 0.6)]); z0[s] = 0; z1[s] = 1.7; cz = 1.3
            c[s] = p + sway; b[s] = 0.13 + br[s]; yaw[s] = rng.uniform(0, 2 * np.pi)
            fw = np.array([np.cos(yaw[i] + np.pi / 2), np.sin(yaw[i] + np.pi / 2)])
            chest[s] = np.c_[c[s] + fw * br[s, None], np.full(n, cz)]
            limb[s] = np.c_[c[s] + fw * 0.3, np.full(n, cz - 0.2)]
            # typing / phone: small hand jitter and occasional fidgets
            limb[s, :2] += np.moveaxis(_ou(rng, (2, n), 0.01, 0.5, fs), 0, 1)
            for _ in range(rng.poisson(n / fs / 20)):
                k0 = rng.integers(0, n); dd = min(n - k0, int(rng.uniform(1, 3) * fs))
                limb[i + k0:i + k0 + dd, :2] += rng.normal(0, 0.1, 2)
            pos = p
        elif name == "walk":
            v = rng.uniform(0.5, 1.2); tgt = np.array([rng.uniform(0.5, W - 0.5), rng.uniform(0.5, D - 0.5)])
            p = pos.copy() if np.all(np.isfinite(pos)) else tgt.copy()
            for k in range(n):
                d = tgt - p; L = np.linalg.norm(d)
                if L < 0.15:
                    tgt = np.array([rng.uniform(0.5, W - 0.5), rng.uniform(0.5, D - 0.5)])
                else:
                    p = p + d / L * min(L, v / fs)
                c[i + k] = p
            z0[s] = 0; z1[s] = 1.7
            sb = _standing(c[s][None], rng, fs, walking=True)
            yaw[s] = sb.yaw[0]; chest[s] = sb.chest[0]; limb[s] = sb.limb[0]; b[s] = sb.b[0]
            pos = p
        i += n
    B = Bodies(c[None], z0[None], z1[None], a[None], b[None], yaw[None], np.array([rng.uniform(0.8, 0.95)]),
               chest[None], limb[None], np.array([rng.uniform(0.3, 0.8) * rcs_scale]), np.array([0.08]))
    return B, dict(activity=act, xy=c, breath_hz=f_b, plan=plan)


# ---------------------------------------------------------------------------
# non-human things that move (the classic false alarms)
# ---------------------------------------------------------------------------
def distractor(kind, sc: Scene, T, rng, fs):
    t = np.arange(T) / fs
    W, D = sc.W, sc.D
    z = lambda v: np.full(T, v, float)
    if kind == "fan":            # standing / ceiling fan: blades are a rotating scatterer, the body barely blocks
        p = np.array([rng.uniform(0.5, W - 0.5), rng.uniform(0.5, D - 0.5)])
        hz = rng.choice([1.1, 2.6]) if rng.random() < 0.3 else rng.uniform(1.0, 1.4)
        f = rng.uniform(12, 22) * 3                     # 3 blades
        r = rng.uniform(0.15, 0.25)
        c = np.repeat(p[None], T, 0)
        limb = np.c_[p[0] + r * np.cos(2 * np.pi * f * t), np.full(T, p[1]), hz + r * np.sin(2 * np.pi * f * t)]
        chest = np.c_[c, z(hz)]
        return Bodies(c[None], z(hz - 0.25)[None], z(hz + 0.25)[None], z(0.25)[None], z(0.05)[None], z(0)[None],
                      np.array([0.15]), chest[None], limb[None], np.array([0.02]), np.array([rng.uniform(0.03, 0.12)]))
    if kind in ("pet", "vacuum"):
        v = rng.uniform(0.2, 0.9) if kind == "pet" else rng.uniform(0.2, 0.35)
        p = np.array([rng.uniform(0.4, W - 0.4), rng.uniform(0.4, D - 0.4)]); tgt = p.copy()
        c = np.zeros((T, 2)); pause = 0
        for i in range(T):
            if pause > 0:
                pause -= 1
            else:
                d = tgt - p; L = np.linalg.norm(d)
                if L < 0.1:
                    tgt = np.array([rng.uniform(0.4, W - 0.4), rng.uniform(0.4, D - 0.4)])
                    pause = int(rng.exponential(6.0 if kind == "pet" else 0.5) * fs)
                else:
                    p = p + d / L * min(L, v / fs)
            c[i] = p
        vel = np.zeros((T, 2)); vel[1:] = np.diff(c, axis=0) * fs
        yaw = np.arctan2(vel[:, 1], vel[:, 0])
        if kind == "pet":        # cat / small dog: low, breathes fast, rcs much smaller than a person
            h = rng.uniform(0.2, 0.45)
            br = 0.0015 * np.sin(2 * np.pi * rng.uniform(0.4, 0.8) * t)
            chest = np.c_[c, z(h * 0.6) + br]
            limb = np.c_[c + 0.1 * np.stack([np.cos(yaw), np.sin(yaw)], 1) * np.sin(2 * np.pi * 2.5 * t)[:, None], z(0.1)]
            return Bodies(c[None], z(0.05)[None], z(h)[None], z(rng.uniform(0.18, 0.3))[None], z(0.08)[None], yaw[None],
                          np.array([0.6]), chest[None], limb[None], np.array([rng.uniform(0.02, 0.08)]), np.array([0.01]))
        chest = np.c_[c, z(0.06)]
        return Bodies(c[None], z(0.0)[None], z(0.1)[None], z(0.17)[None], z(0.17)[None], yaw[None],
                      np.array([0.9]), chest[None], chest[None], np.array([rng.uniform(0.05, 0.15)]), np.array([0.0]))
    if kind == "door":           # a door swinging open/closed every 15-40 s
        hinge = np.array([rng.uniform(0.5, W - 1.5), 0.05]); wdt = 0.85
        ang = np.zeros(T); k = 0
        while k < T:
            k += int(rng.uniform(15, 40) * fs)
            dur = int(rng.uniform(1.0, 2.5) * fs); target = rng.uniform(0.3, 1.5) if ang[min(k, T - 1)] < 0.1 else 0.0
            s = ang[min(k, T - 1)]
            for i in range(k, min(T, k + dur)):
                ang[i] = s + (target - s) * (0.5 - 0.5 * np.cos(np.pi * (i - k) / dur))
            if k + dur < T:
                ang[k + dur:] = target
            k += dur
        c = hinge + 0.5 * wdt * np.stack([np.cos(ang), np.sin(ang)], 1)
        chest = np.c_[c, z(1.0)]
        return Bodies(c[None], z(0.0)[None], z(2.0)[None], z(wdt / 2)[None], z(0.02)[None], ang[None],
                      np.array([0.4]), chest[None], chest[None], np.array([0.3]), np.array([0.0]))
    raise ValueError(kind)
