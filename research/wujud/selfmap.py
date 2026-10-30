"""The mesh maps itself: node positions from the empty-hall signal strengths between nodes.

Borrowed from sensor-network localisation (MDS-MAP, Shang et al. 2003): every pair of boards
already measures received power during calibration. A log-distance path-loss model turns
power into distance, classical MDS turns the distance matrix into coordinates, and a stress
refinement fits the path-loss exponent and positions jointly. One known distance (e.g. the
hall width, or two boards placed a tape-length apart) fixes the scale; rotation/mirroring are
fixed by naming which wall is the front.
"""
import numpy as np
from scipy.optimize import least_squares


def classical_mds(Dm, dim=2):
    n = len(Dm)
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (Dm ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:dim]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 1e-9))


def self_map(P, links, n, dim=2, n0=2.0):
    """P [L] received power in dB per link (both directions averaged). Returns coordinates [n,dim]
    in units where the fitted 1-m power is P0 (scale is arbitrary until one distance is known)."""
    links = np.asarray(links)
    ok = np.isfinite(P)
    P0 = np.percentile(P[ok], 95) + 10 * n0 * np.log10(1.0)
    d = 10 ** ((P0 - P) / (10 * n0))
    Dm = np.zeros((n, n)); cnt = np.zeros((n, n))
    for (i, j), dd, o in zip(links, d, ok):
        if o:
            Dm[i, j] = Dm[j, i] = dd; cnt[i, j] = cnt[j, i] = 1
    # fill missing pairs by shortest paths
    Dfill = np.where(cnt > 0, Dm, np.inf); np.fill_diagonal(Dfill, 0)
    for k in range(n):
        Dfill = np.minimum(Dfill, Dfill[:, k:k + 1] + Dfill[k:k + 1, :])
    X0 = classical_mds(Dfill, dim)

    # unknowns: positions, P0, path-loss exponent, and a gain offset per board
    # (boards differ by several dB in Tx power / antenna / enclosure; ignoring that ruins the map)
    li, lj = links[ok, 0], links[ok, 1]

    def resid(v):
        X = v[:n * dim].reshape(n, dim); g = v[n * dim:n * dim + n]; p0, ne = v[-2], v[-1]
        dd = np.linalg.norm(X[li] - X[lj], axis=1)
        r = (P[ok] - (p0 + g[li] + g[lj] - 10 * ne * np.log10(np.maximum(dd, 1e-3)))) / 3.0
        return np.r_[r, g / 4.0, (ne - 2.2) / 0.5]
    v0 = np.r_[X0.ravel(), np.zeros(n), P0, n0]
    r = least_squares(resid, v0, loss="soft_l1", f_scale=1.0, max_nfev=4000)
    return r.x[:n * dim].reshape(n, dim), r.x[-1]


def align(X, Y, scale=True):
    """similarity Procrustes of X onto Y (rotation/reflection, translation, optional scale)"""
    mx, my = X.mean(0), Y.mean(0)
    A, B = X - mx, Y - my
    U, S, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    s = S.sum() / (A ** 2).sum() if scale else 1.0
    return s * A @ R + my


def perim_xy(s, W, D):
    s = np.mod(s, 2 * (W + D))
    x = np.where(s < W, s, np.where(s < W + D, W, np.where(s < 2 * W + D, W - (s - W - D), 0.0)))
    y = np.where(s < W, 0.0, np.where(s < W + D, s - W, np.where(s < 2 * W + D, D, D - (s - 2 * W - D))))
    return np.stack([x, y], -1)


def self_map_walls(P, links, n, W, D, restarts=40, seed=0):
    """boards are on the walls of a W x D hall (the user types W and D, or measures one wall):
    each board has a single unknown, its position along the perimeter. Fitted jointly with
    per-board gain offsets and the path-loss law, with random restarts."""
    links = np.asarray(links); ok = np.isfinite(P); li, lj = links[ok, 0], links[ok, 1]
    rng = np.random.default_rng(seed); per = 2 * (W + D)

    def resid(v):
        s = v[:n]; g = v[n:2 * n]; p0, ne = v[-2], v[-1]
        X = perim_xy(s, W, D)
        dd = np.linalg.norm(X[li] - X[lj], axis=1)
        return np.r_[(P[ok] - (p0 + g[li] + g[lj] - 10 * ne * np.log10(np.maximum(dd, 0.3)))) / 3.0, g / 4.0, (ne - 2.0) / 0.7]
    best = None
    for _ in range(restarts):
        v0 = np.r_[np.sort(rng.uniform(0, per, n))[rng.permutation(n)] if False else rng.uniform(0, per, n), np.zeros(n),
                   np.percentile(P[ok], 90), 2.0]
        r = least_squares(resid, v0, loss="soft_l1", f_scale=1.0, max_nfev=3000)
        if best is None or r.cost < best.cost:
            best = r
    return perim_xy(best.x[:n], W, D)
