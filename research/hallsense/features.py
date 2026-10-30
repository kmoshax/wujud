"""Per-packet and per-window CSI features. The firmware (firmware/hallsense_node.ino)
implements exactly the same arithmetic so simulated and real data look identical.

per packet:
    p_k   = I_k^2 + Q_k^2                       (52 L-LTF subcarriers)
    a_k   = 10 log10(p_k / mean(p)) + RSSI       (AGC removed via RSSI -> absolute dB)
    g_m   = mean of a_k over group m (4 subcarriers), m = 0..12
    drop packet if |RSSI - EMA(RSSI)| > OUTLIER_DB   (interference bursts)
per window (1 s) and per source node:
    mean_m, std_m of g_m over accepted packets, n packets
"""
import numpy as np
from .config import N_GROUPS, OUTLIER_DB


def packet_groups(iq, rssi):
    p = iq[..., 0].astype(np.float32) ** 2 + iq[..., 1].astype(np.float32) ** 2
    p = np.maximum(p, 0.5)
    a = 10 * np.log10(p / p.mean(axis=-1, keepdims=True)) + rssi[:, None]
    return a.reshape(len(a), N_GROUPS, -1).mean(-1)


def ema_reject(rssi, alpha=0.1, thr=OUTLIER_DB):
    keep = np.ones(len(rssi), bool)
    if len(rssi) == 0:
        return keep
    ema = float(np.median(rssi[: min(20, len(rssi))]))
    for n, r in enumerate(rssi):
        if abs(r - ema) > thr:
            keep[n] = False
        else:
            ema += alpha * (r - ema)
    return keep


def window_features(iq, rssi, t, win, total):
    nw = int(round(total / win))
    mean = np.full((nw, N_GROUPS), np.nan, np.float32)
    std = np.full((nw, N_GROUPS), np.nan, np.float32)
    cnt = np.zeros(nw, np.float32)
    if len(rssi) == 0:
        return mean, std, cnt
    g = packet_groups(iq, rssi.astype(np.float32))
    keep = ema_reject(rssi)
    g, t = g[keep], t[keep]
    w = np.minimum((t / win).astype(int), nw - 1)
    for k in range(nw):
        sel = g[w == k]
        cnt[k] = len(sel)
        if len(sel) >= 3:
            mean[k] = sel.mean(0)
            std[k] = sel.std(0)
    return mean, std, cnt
