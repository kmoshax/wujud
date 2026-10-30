import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score


def seat_f1(pred, true):
    tp = (pred & true).sum(); fp = (pred & ~true).sum(); fn = (~pred & true).sum()
    return 2 * tp / max(1, 2 * tp + fp + fn)


def tolerant_f1(pred, true, seats, r=1.0):
    """F1 where a predicted seat matches a true person sitting within r metres
    (one-to-one assignment). Measures 'found the student, maybe one seat off'."""
    P, T = np.nonzero(pred)[0], np.nonzero(true)[0]
    if len(P) == 0 and len(T) == 0:
        return 1.0
    if len(P) == 0 or len(T) == 0:
        return 0.0
    D = np.linalg.norm(seats[P][:, None] - seats[T][None], axis=-1)
    cost = np.where(D <= r, D, 1e6)
    ri, ci = linear_sum_assignment(cost)
    tp = (cost[ri, ci] < 1e6).sum()
    return 2 * tp / (len(P) + len(T))


def auc(p, y):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def block_corr(p, y, rc, size=3):
    """correlation of predicted vs true occupancy on size x size seat blocks (density map quality)"""
    key = (rc[:, 0] // size) * 10000 + rc[:, 1] // size
    u = np.unique(key)
    a = np.array([p[key == k].mean() for k in u]); b = np.array([y[key == k].mean() for k in u])
    return np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
