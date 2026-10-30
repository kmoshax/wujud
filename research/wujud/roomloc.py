"""Locate one person in a room from several links, by physics alone (no training):
a person disturbs most the links whose Fresnel ellipse they stand in, so the position is the
grid point whose 'which links should light up' pattern best matches which links did."""
import numpy as np


def locate(link_feats, links, nodes, W, D, step=0.2, scale=0.12):
    """link_feats [L,F] (roomfeat features of one window per link) -> (x, y)"""
    ev = np.maximum(link_feats[:, 1], link_feats[:, 3])           # breath / micro-motion energy above noise
    ev = ev - np.median(ev)
    gx, gy = np.meshgrid(np.arange(0.2, W - 0.1, step), np.arange(0.2, D - 0.1, step))
    G = np.c_[gx.ravel(), gy.ravel()]
    Wt = []
    for (i, j) in links:
        a, b = nodes[i, :2], nodes[j, :2]
        e = np.linalg.norm(G - a, axis=1) + np.linalg.norm(G - b, axis=1) - np.linalg.norm(b - a)
        Wt.append(np.exp(-e / scale))
    Wt = np.array(Wt)                                               # [L,G]
    Wn = Wt - Wt.mean(0, keepdims=True)
    score = (Wn * ev[:, None]).sum(0) / (np.linalg.norm(Wn, axis=0) + 1e-9)
    return G[np.argmax(score)]
