"""How much the links' geometry alone allows to tell people apart, per number of devices.

A point is "seen" by a link when its extra path over the direct line is under ZONE_M (a body there
changes that link). Points seen by the same set of links form one cell: by link intensity alone, two
people in one cell look like one person, and a person in a cell seen by no link is invisible to motion.

    .venv/bin/python -I tools/crowd_geometry.py cache/crowd/*.jsonl
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from link_net import geometry, grid  # noqa: E402

ZONE_M = 0.25


def rooms(paths):
    for path in paths:
        seen = set()
        for line in open(path):
            row = json.loads(line)
            if row["room"] in seen:
                continue
            seen.add(row["room"])
            yield row


def main():
    stats = {}
    rng = np.random.default_rng(0)
    for row in rooms(sys.argv[1:]):
        points = grid(row["width"], row["depth"])
        inside = (points[:, 0] > 0.5) & (points[:, 0] < row["width"] - 0.5) & (points[:, 1] > 0.5) & (points[:, 1] < row["depth"] - 0.5)
        points = points[inside]
        seen = geometry(points, row["nodes"], row["links"])[..., 0] < ZONE_M
        codes = seen @ (1 << np.arange(seen.shape[1], dtype=np.int64))
        s = stats.setdefault(len(row["nodes"]), {"links": len(row["links"]), "cells": [], "blind": [], "same": []})
        s["cells"].append(len(np.unique(codes)))
        s["blind"].append(np.mean(codes == 0))
        a, b = rng.integers(len(points), size=(2, 2000))
        far = np.linalg.norm(points[a] - points[b], axis=1) > 0.6
        s["same"].append(np.mean(codes[a][far] == codes[b][far]))
    for n, s in sorted(stats.items()):
        print(f"{n} devices, {s['links']} links: {np.median(s['cells']):.0f} distinguishable cells (median), "
              f"{np.mean(s['blind']) * 100:.0f}% of the floor seen by no link, "
              f"two people >0.6 m apart share a cell {np.mean(s['same']) * 100:.0f}% of the time")


if __name__ == "__main__":
    main()
