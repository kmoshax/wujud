"""Each board pair's CSI phase slope in nanoseconds, both directions averaged, per 20 s window.

Run from research/: .venv/bin/python tools/csi_delay_pairs.py rec/<recording>.csv
Starred pairs are the diagonals of the square measured on 2026-11-05 (docs/FINDINGS.md, section 6).
"""
import sys, numpy as np, collections
DF = 312.5e3
def delays(path, window=None):
    links = collections.defaultdict(list)
    t0 = None
    for line in open(path):
        f = line.rstrip().split(',')
        if len(f) < 6 or f[2] != 'P' or len(f[5]) < 256: continue
        t = float(f[0]); t0 = t0 or t
        if window and not (window[0] <= t - t0 < window[1]): continue
        b = np.frombuffer(bytes.fromhex(f[5][:256]), dtype=np.int8).astype(float)
        h = b[1::2] + 1j * b[0::2]
        k = np.r_[np.arange(0, 32), np.arange(-32, 0)]
        pos = dict(zip(k.tolist(), h))
        d = lambda ks: -np.angle(sum(pos[a + 1] * np.conj(pos[a]) for a in ks)) / (2 * np.pi * DF) * 1e9
        links[f[1].split('@', 1)[1]].append((d(range(1, 26)) + d(range(-26, -1))) / 2)
    pairs = collections.defaultdict(list)
    short = lambda m: 'hub' if m == 'hub' else m[-5:]
    for tag, v in links.items():
        rx, tx = (tag.split('>') if '>' in tag else ('hub', tag))
        pairs[tuple(sorted((short(rx), short(tx))))].append(np.median(v))
    return {p: np.mean(v) for p, v in pairs.items()}
diag = {('d9:bc', 'hub'), ('74:ec', 'b5:14')}
for path in sys.argv[1:]:
    print(path.split('/')[-1])
    for w in [None, (0, 20), (20, 40), (40, 60)]:
        r = delays(path, w)
        s = '  '.join(f"{a}-{b}{'*' if (a,b) in diag else ' '} {v:5.2f}" for (a, b), v in sorted(r.items(), key=lambda x: -x[1]))
        print(f"  {str(w):9} {s}")
