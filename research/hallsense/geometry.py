"""Lecture-hall geometry: walls, seat grid, ESP32 node placement, static paths.

2.5-D model: everything lives in the floor plane, except that node heights
and the ceiling height set the floor/ceiling bounce lengths. Links run at
seated-torso height, so a seated body is modelled as a vertical absorbing
strip whose blocking strength (eta) absorbs the height mismatch.
"""
from dataclasses import dataclass, field
import numpy as np

from .config import LAM, FREQS, C

# relative permittivity at 2.4 GHz (ITU-R P.2040 ballpark values)
MATERIALS = {"concrete": 5.3, "brick": 3.9, "drywall": 2.8, "glass": 6.3, "wood": 2.0}


def normal_reflection(eps):
    return (1 - np.sqrt(eps)) / (1 + np.sqrt(eps))


@dataclass
class Hall:
    W: float
    D: float
    Hc: float
    nodes: np.ndarray          # [N,2]
    node_h: np.ndarray         # [N]
    seats: np.ndarray          # [S,2]
    seat_rc: np.ndarray        # [S,2] (row, col) indices
    wall_gamma: np.ndarray     # [4] x=0, x=W, y=0, y=D
    front: float               # lecturer zone is y in [0, front]
    aisles_x: list = field(default_factory=list)
    placement: str = "perimeter"

    @property
    def links(self):
        n = len(self.nodes)
        return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _perimeter_points(W, D, n, rng, inset=0.1):
    """n points evenly spaced on the rectangle perimeter, jittered."""
    P = 2 * (W + D)
    s = (np.arange(n) + rng.uniform(0.2, 0.8)) / n * P
    s = (s + rng.normal(0, 0.15 * P / n, n)) % P
    pts = []
    for v in s:
        if v < W:
            pts.append((v, inset))
        elif v < W + D:
            pts.append((W - inset, v - W))
        elif v < 2 * W + D:
            pts.append((W - (v - W - D), D - inset))
        else:
            pts.append((inset, D - (v - 2 * W - D)))
    p = np.array(pts)
    p[:, 0] = np.clip(p[:, 0], inset, W - inset)
    p[:, 1] = np.clip(p[:, 1], inset, D - inset)
    return p


def random_hall(rng, n_nodes=12, placement=None, W=None, D=None):
    W = W or rng.uniform(9.0, 18.0)
    D = D or rng.uniform(8.0, 16.0)
    Hc = rng.uniform(3.0, 4.5)
    front = rng.uniform(2.5, 3.5)
    row_pitch = rng.uniform(0.85, 1.10)
    seat_pitch = rng.uniform(0.55, 0.70)
    placement = placement or rng.choice(["perimeter", "perimeter+aisle"])

    # aisles split the seating width into blocks
    n_aisles = int(rng.integers(1, 3)) if W > 11 else 1
    x0, x1 = 0.9, W - 0.9
    aisle_w = 1.0
    cuts = np.sort(rng.uniform(0.3, 0.7, n_aisles)) if n_aisles == 1 else np.array([0.33, 0.67]) + rng.normal(0, 0.03, 2)
    aisles_x = list(x0 + cuts * (x1 - x0))

    seats, rc = [], []
    ys = np.arange(front + 0.9, D - 0.6, row_pitch)
    for r, y in enumerate(ys):
        xs = np.arange(x0, x1 + 1e-6, seat_pitch)
        c = 0
        for x in xs:
            if any(abs(x - a) < aisle_w / 2 for a in aisles_x):
                continue
            seats.append((x, y))
            rc.append((r, c))
            c += 1
    seats = np.array(seats)
    rc = np.array(rc)

    if placement == "perimeter":
        nodes = _perimeter_points(W, D, n_nodes, rng)
    elif placement == "rows":
        # row-end placement: pairs of nodes on the two side walls, level with a row
        # (on the end desks at chest height) -> some links run straight along a row,
        # the rest cut diagonally across rows. 2 nodes front corners if n is odd.
        n_pairs = n_nodes // 2
        pick = np.round(np.linspace(0, len(ys) - 1, n_pairs)).astype(int)
        nodes = []
        for r in pick:
            y = ys[r] + rng.normal(0, 0.05)
            nodes += [(0.1, y), (W - 0.1, y + rng.normal(0, 0.05))]
        if n_nodes % 2:
            nodes.append((W / 2, 0.1))
        nodes = np.array(nodes)
    else:
        # ~1/3 of the nodes sit at aisle row-ends (on a desk edge), rest on the perimeter
        n_in = max(2, n_nodes // 3)
        nodes_p = _perimeter_points(W, D, n_nodes - n_in, rng)
        row_y = rng.choice(ys, n_in, replace=len(ys) < n_in)
        ax = rng.choice(aisles_x, n_in)
        nodes_i = np.stack([ax + rng.normal(0, 0.1, n_in), row_y + rng.uniform(-0.4, 0.4, n_in)], 1)
        nodes = np.vstack([nodes_p, nodes_i])
    node_h = rng.uniform(0.95, 1.25, len(nodes))

    mats = rng.choice(list(MATERIALS), 4)
    gam = np.array([normal_reflection(MATERIALS[m]) for m in mats]) * rng.uniform(0.8, 1.2, 4)
    return Hall(W, D, Hc, nodes, node_h, seats, rc, gam, front, aisles_x, placement)


# ---------------------------------------------------------------------------
# static propagation paths for one link
# ---------------------------------------------------------------------------
@dataclass
class Path:
    length: float
    amp: complex               # complex gain (reflection coefficients included)
    segments: list             # list of (P0, P1, cum_start) for body-shadowing; [] = not shadowable
    shadow_eta: float = 1.0    # how strongly a body on this path blocks it (floor bounce < 1)


def link_paths(hall: Hall, i: int, j: int):
    a, b = hall.nodes[i], hall.nodes[j]
    ha, hb = hall.node_h[i], hall.node_h[j]
    paths = []
    d = np.linalg.norm(b - a)
    paths.append(Path(d, 1.0 + 0j, [(a, b, 0.0)]))
    # first-order wall reflections (image method; always valid for a convex rectangle)
    for w, (axis, pos) in enumerate([(0, 0.0), (0, hall.W), (1, 0.0), (1, hall.D)]):
        img = a.copy()
        img[axis] = 2 * pos - a[axis]
        L = np.linalg.norm(b - img)
        t = (pos - img[axis]) / (b[axis] - img[axis] + 1e-12)
        P = img + t * (b - img)
        P[axis] = pos
        L1 = np.linalg.norm(P - a)
        paths.append(Path(L, complex(hall.wall_gamma[w]), [(a, P, 0.0), (P, b, L1)]))
    # floor and ceiling bounces (2.5-D lengths); floor bounce is partly blocked by people
    Lf = np.hypot(d, ha + hb)
    paths.append(Path(Lf, complex(-0.45), [(a, b, 0.0)], shadow_eta=0.5))
    Lc = np.hypot(d, 2 * hall.Hc - ha - hb)
    paths.append(Path(Lc, complex(-0.35), []))
    return paths


def path_response(L, amp):
    """free-space complex response of a path of length L at every subcarrier [K]"""
    return amp * LAM / (4 * np.pi * L) * np.exp(-2j * np.pi * FREQS * L / C)


def scatter_response(a, b, p, rcs):
    """bistatic point scatterer at p (vectorised over p: [P,2]) -> [P,K]"""
    d1 = np.linalg.norm(p - a, axis=-1)
    d2 = np.linalg.norm(p - b, axis=-1)
    A = LAM * np.sqrt(rcs) / ((4 * np.pi) ** 1.5 * d1 * d2)
    return A[:, None] * np.exp(-2j * np.pi * FREQS[None, :] * (d1 + d2)[:, None] / C), d1 + d2


def excess_path(a, b, p):
    """|a-p| + |p-b| - |a-b| for points p [S,2]"""
    return np.linalg.norm(p - a, axis=-1) + np.linalg.norm(p - b, axis=-1) - np.linalg.norm(b - a)
