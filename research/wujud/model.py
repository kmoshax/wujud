"""Seat-occupancy inference from an ESP32 link mesh.

Key idea (geometry-invariant learning): a model that maps "link i, link j, ..."
to seats only works in the hall it was trained in. Instead, every seat is
described by the evidence of the links that pass near it, binned by how far
the seat sits from each link's line in units of Fresnel excess path length.
That descriptor means the same thing in every hall and for any number or
placement of nodes, so one model trained on thousands of simulated halls
transfers to a hall it has never seen (incl. the real one).

Stage 1: seat descriptor -> P(occupied)                (gradient-boosted trees)
Stage 2: + neighbour context and "explaining away"   (who else could have
          caused the attenuation on this seat's links?)
Count:   sum of seat probabilities, and a scene-level regressor on
          crowd-absorption features; reported separately and blended.
"""
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .geometry import excess_path

BINS = np.array([0.0, 0.015, 0.035, 0.07, 0.14, 0.28])   # excess path (m); 1st Fresnel zone = lambda/2 = 0.061
LINK_FEATS = ["sh_mean", "sh_min", "sh_abs", "sh_spread", "mo_mean", "mo_p90", "mo_frac", "slow"]
BIN_FEATS = ["sh_mean", "sh_min", "sh_abs", "mo_mean", "mo_frac", "slow"]


# ---------------------------------------------------------------------------
def link_features(cal_mean, cal_std, ses_mean, ses_std, last=None):
    """window arrays [L,W,13] -> dict of [L] link features. `last` = use only last N session windows."""
    if last:
        ses_mean, ses_std = ses_mean[:, -last:], ses_std[:, -last:]
    with np.errstate(all="ignore"):
        cm = np.nanmedian(cal_mean, 1)
        cs = np.nanmedian(cal_std, 1)
        sm = np.nanmedian(ses_mean, 1)
        d = sm - cm
        base = np.nanmean(cs, 1)
        mw = np.nanmean(ses_std, 2) / np.maximum(base[:, None], 1e-3)
        lr = np.log(np.maximum(mw, 1e-3))
        slow_s = np.nanstd(np.nanmean(ses_mean, 2), 1)
        slow_c = np.nanstd(np.nanmean(cal_mean, 2), 1)
        f = {
            "sh_mean": np.nanmean(d, 1),
            "sh_min": np.nanmin(d, 1),
            "sh_abs": np.nanmean(np.abs(d), 1),
            "sh_spread": np.nanstd(d, 1),
            "mo_mean": np.nanmean(lr, 1),
            "mo_p90": np.nanpercentile(lr, 90, axis=1),
            "mo_frac": np.nanmean(mw > 1.5, 1),
            "slow": np.log(np.maximum(slow_s, 1e-3) / np.maximum(slow_c, 1e-3)),
        }
    return f


def geometry_cache(nodes, links, seats):
    """excess path [L,S], bin index [L,S] (-1 = outside), link lengths"""
    E = np.stack([excess_path(nodes[i], nodes[j], seats) for i, j in links])
    B = np.digitize(E, BINS) - 1
    B[E >= BINS[-1]] = -1
    return E, B


def seat_descriptor(f, B, nodes, seats, n_seats_total=None):
    L, S = B.shape
    nb = len(BINS) - 1
    cols, names = [], []
    for b in range(nb):
        M = (B == b).astype(np.float32)               # [L,S]
        cnt = M.sum(0)
        cols.append(cnt); names.append(f"n_b{b}")
        for k in BIN_FEATS:
            v = np.nan_to_num(f[k], nan=0.0)
            with np.errstate(all="ignore"):
                mean = (M * v[:, None]).sum(0) / cnt
            cols.append(np.where(cnt > 0, mean, np.nan)); names.append(f"{k}_b{b}")
        if b <= 1:
            for k, fn, fill in (("sh_mean", np.min, np.inf), ("mo_mean", np.max, -np.inf)):
                v = np.nan_to_num(f[k], nan=0.0)
                arr = np.where(M > 0, v[:, None], fill)
                r = fn(arr, 0)
                cols.append(np.where(np.isfinite(r), r, np.nan)); names.append(f"{k}_{fn.__name__}_b{b}")
    # scene context: crowd-level evidence over all links
    for k in LINK_FEATS:
        cols.append(np.full(S, np.nanmedian(f[k]))); names.append(f"g_{k}")
    dn = np.min(np.linalg.norm(seats[:, None] - nodes[None], axis=-1), 1)
    cols.append(dn); names.append("d_node")
    cols.append(np.full(S, len(nodes))); names.append("n_nodes")
    return np.stack(cols, 1).astype(np.float32), names


def context_features(p1, B, seats, radius=1.0):
    """stage-2 features from stage-1 probabilities p1 [S]"""
    S = len(seats)
    D = np.linalg.norm(seats[:, None] - seats[None], axis=-1)
    nbr = (D < radius) & (D > 0)
    with np.errstate(all="ignore"):
        m_nbr = (nbr * p1[None]).sum(1) / nbr.sum(1)
    near = (B >= 0) & (B <= 1)                      # [L,S] links through this seat's Fresnel zone
    load = (near * p1[None]).sum(1)                 # [L] expected #people on each link
    with np.errstate(all="ignore"):
        others = (near * (load[:, None] - p1[None])).sum(0) / near.sum(0)
    return np.stack([p1, np.nan_to_num(m_nbr, nan=0.0), others, p1 - np.nan_to_num(m_nbr, nan=0.0)], 1).astype(np.float32)


def scene_vector(f, nodes, links, n_seats, W, D, Hc=3.7):
    """hall-level features for the head-count regressor"""
    v = []
    for k in LINK_FEATS:
        x = f[k]
        v += [np.nanmedian(x), np.nanmean(x), np.nanpercentile(x, 10), np.nanpercentile(x, 90)]
    area = W * D; surf = 2 * area + 2 * (W + D) * Hc
    L = np.array([np.linalg.norm(nodes[i] - nodes[j]) for i, j in links])
    sh = f["sh_mean"]; ok = np.isfinite(sh)
    slope = np.polyfit(L[ok], sh[ok], 1)[0] if ok.sum() > 3 else 0.0
    # Sabine-style inversion: a power drop of d dB in the reverberant field ~ surface * (10^(d/10) - 1) absorbers
    v += [len(nodes), n_seats, area, surf, L.mean(),
          surf * (10 ** (-np.nanmedian(sh) / 10) - 1), surf * (10 ** (-np.nanmean(sh) / 10) - 1),
          np.nanmedian(f["mo_mean"]) * n_seats, slope]
    return np.array(v, np.float32)


# ---------------------------------------------------------------------------
def load_scene(path, last=None):
    z = np.load(path)
    f = link_features(z["cal_mean"].astype(np.float32), z["cal_std"].astype(np.float32),
                      z["ses_mean"].astype(np.float32), z["ses_std"].astype(np.float32), last)
    nodes, seats, links = z["nodes"], z["seats"], [tuple(l) for l in z["links"]]
    E, B = geometry_cache(nodes, links, seats)
    X, names = seat_descriptor(f, B, nodes, seats)
    meta = json.loads(str(z["meta"]))
    return dict(path=path, f=f, nodes=nodes, seats=seats, links=links, E=E, B=B, X=X, names=names,
                y=z["occ"].astype(int), meta=meta, sv=scene_vector(f, nodes, links, len(seats), float(z["W"]), float(z["D"])),
                W=float(z["W"]), D=float(z["D"]), front=float(z["front"]), rc=z["seat_rc"])


GB = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0)


class SeatModel:
    def __init__(self, feat_mask=None):
        self.feat_mask = feat_mask
        self.m1 = HistGradientBoostingClassifier(**GB)
        self.m2 = HistGradientBoostingClassifier(**GB)
        self.mc = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, min_samples_leaf=8)

    def _X(self, s):
        return s["X"] if self.feat_mask is None else s["X"][:, self.feat_mask]

    def fit(self, scenes, seed=0):
        X = np.vstack([self._X(s) for s in scenes]); y = np.concatenate([s["y"] for s in scenes])
        g = np.concatenate([[s["meta"]["hall_seed"]] * len(s["y"]) for s in scenes])
        # out-of-fold stage-1 probabilities so stage 2 learns from honest inputs
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            m = HistGradientBoostingClassifier(**GB, random_state=seed).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        self.m1.set_params(random_state=seed).fit(X, y)
        X2, o = [], 0
        for s in scenes:
            n = len(s["y"]); X2.append(np.hstack([self._X(s), context_features(oof[o:o + n], s["B"], s["seats"])])); o += n
        self.m2.set_params(random_state=seed).fit(np.vstack(X2), y)
        SV = np.nan_to_num(np.stack([s["sv"] for s in scenes])); cnt = np.array([s["y"].sum() for s in scenes])
        self.mc.fit(SV, cnt)
        return self

    def predict(self, s):
        p1 = self.m1.predict_proba(self._X(s))[:, 1]
        p2 = self.m2.predict_proba(np.hstack([self._X(s), context_features(p1, s["B"], s["seats"])]))[:, 1]
        n_seats = len(s["seats"])
        count_glob = float(np.clip(self.mc.predict(np.nan_to_num(s["sv"][None]))[0], 0, n_seats))
        count_sum = float(p2.sum())
        return dict(p1=p1, p=p2, count_sum=count_sum, count_glob=count_glob,
                    count=0.5 * (count_sum + count_glob))


# ---------------------------------------------------------------------------
# classic baseline: seat-grid radio tomographic imaging (Wilson & Patwari style)
# ---------------------------------------------------------------------------
class RTIBaseline:
    def __init__(self, lam_e=0.04, alpha=1.0, mode="both"):
        self.lam_e, self.alpha, self.mode, self.thr = lam_e, alpha, mode, 0.5

    def image(self, s):
        Wm = (s["E"] < self.lam_e) / np.sqrt(np.array([np.linalg.norm(s["nodes"][i] - s["nodes"][j]) for i, j in s["links"]]))[:, None]
        sh = -np.nan_to_num(s["f"]["sh_mean"]); mo = np.nan_to_num(s["f"]["mo_mean"])
        y = {"shadow": sh, "motion": mo, "both": sh / (np.std(sh) + 1e-6) + mo / (np.std(mo) + 1e-6)}[self.mode]
        A = Wm.T @ Wm + self.alpha * np.eye(Wm.shape[1])
        return np.linalg.solve(A, Wm.T @ y)

    def fit(self, scenes):
        best = (-1, None)
        for lam in (0.02, 0.04, 0.07):
            for al in (0.3, 1, 3, 10):
                self.lam_e, self.alpha = lam, al
                ims = [self.image(s) for s in scenes]
                for thr in np.quantile(np.concatenate(ims), np.linspace(0.05, 0.95, 30)):
                    tp = sum(((im > thr) & (s["y"] == 1)).sum() for im, s in zip(ims, scenes))
                    fp = sum(((im > thr) & (s["y"] == 0)).sum() for im, s in zip(ims, scenes))
                    fn = sum(((im <= thr) & (s["y"] == 1)).sum() for im, s in zip(ims, scenes))
                    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
                    if f1 > best[0]:
                        best = (f1, (lam, al, thr))
        self.lam_e, self.alpha, self.thr = best[1]
        return self

    def predict(self, s):
        im = self.image(s)
        return dict(p=im, occ=im > self.thr, count=float((im > self.thr).sum()))


def scene_from_arrays(cal_mean, cal_std, ses_mean, ses_std, nodes, links, seats, geo=None, W=None, D=None):
    """build the dict SeatModel.predict needs from raw window arrays (live use)"""
    f = link_features(cal_mean, cal_std, ses_mean, ses_std)
    E, B = geo if geo is not None else geometry_cache(nodes, links, seats)
    X, names = seat_descriptor(f, B, nodes, seats)
    return dict(f=f, nodes=nodes, seats=seats, links=links, E=E, B=B, X=X, names=names,
                sv=scene_vector(f, nodes, links, len(seats), W, D))
