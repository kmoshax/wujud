"""How noisy each recording's CSI amplitude is from one packet to the next, per link.

Run from research/: .venv/bin/python tools/ltf_noise.py rec/a.csv rec/b.csv
The room should be still: what is left is the board's measurement noise. Amplitudes are divided by
each packet's mean first (the receiver's gain moves them all together).
"""
import collections
import sys

import numpy as np

DATA = np.r_[1:27, 38:64]


def noise_per_link(path):
    links = collections.defaultdict(list)
    for line in open(path):
        f = line.rstrip().split(',')
        if len(f) < 6 or f[2] != 'P' or len(f[5]) < 256:
            continue
        b = np.frombuffer(bytes.fromhex(f[5][:256]), dtype=np.int8).astype(float)
        amp = np.hypot(b[0::2], b[1::2])[DATA]
        if amp.mean() < 1:
            continue
        links[f[1].split('@', 1)[1]].append(amp / amp.mean())
    out = {}
    for tag, rows in links.items():
        a = np.array(rows)
        if len(a) < 200:
            continue
        step = np.abs(np.diff(a, axis=0))
        out[tag] = (np.median(step), len(a), np.mean(a[:, :8].mean(1) < 0.05))
    return out


for path in sys.argv[1:]:
    noise = noise_per_link(path)
    values = [v[0] for v in noise.values()]
    print(f"{path.split('/')[-1]:28s} links {len(values):2d}  median packet-to-packet change {np.median(values):.4f}"
          f"  (min {min(values):.4f}, max {max(values):.4f})")
