"""End-to-end test without hardware: simulate sessions in a hall.json room and write them out
exactly as the sink's serial stream would look (R,rx,src,n,means...,stds...) with timestamps,
plus a labels.csv. Then:  python real_eval.py eval --rec fake_rec.csv --labels fake_labels.csv ...

python tools/fake_serial.py --hall hall.example.json --out fake
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live import load_hall
from hallsense.geometry import Hall
from hallsense.sim import simulate_scene


def lines_for(m, s, links, ids, t0, rng):
    out = []
    for w in range(m.shape[1]):
        for l, (a, b) in enumerate(links):
            if not np.isfinite(m[l, w, 0]):
                continue
            for rx, src in ((ids[a], ids[b]), (ids[b], ids[a])):
                vals = ",".join(f"{v:.2f}" for v in np.r_[m[l, w], s[l, w]])
                out.append((t0 + w + 0.5 + rng.uniform(-0.3, 0.3), f"R,{rx},{src},20,{vals}"))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", default="hall.example.json"); ap.add_argument("--out", default="fake")
    ap.add_argument("--counts", default="0,1,2,3,5,7,10,14,20")
    a = ap.parse_args()
    h = load_hall(a.hall)
    hall = Hall(h["W"], h["D"], 3.2, h["nodes"][:, :2], np.full(len(h["nodes"]), 1.1), h["seats"][:, :2],
                np.c_[np.zeros(len(h["seats"]), int), np.arange(len(h["seats"]))], np.array([-0.4] * 4), h["front"], [], "custom")
    rng = np.random.default_rng(0)
    t = 1_800_000_000.0
    rec, lab = [], ["start,end,seats"]
    cal = None
    for k, n in enumerate(int(x) for x in a.counts.split(",")):
        occ = np.zeros(len(h["seats"]), bool)
        occ[rng.choice(len(occ), n, replace=False)] = True
        r = simulate_scene(hall, np.random.default_rng([1, k]), occ=occ, with_walkers=False, env_seed=99)
        if cal is None:
            cal = r
            rec += lines_for(r["cal_mean"], r["cal_std"], h["links"], h["ids"], t, rng); t += r["cal_mean"].shape[1] + 5
        rec += lines_for(r["ses_mean"], r["ses_std"], h["links"], h["ids"], t, rng)
        lab.append(f"{t:.0f},{t + r['ses_mean'].shape[1]:.0f},{';'.join(map(str, np.nonzero(occ)[0]))}")
        t += r["ses_mean"].shape[1] + 20
    with open(a.out + "_rec.csv", "w") as f:
        for tt, l in rec:
            f.write(f"{tt:.3f},{l}\n")
    open(a.out + "_labels.csv", "w").write("\n".join(lab) + "\n")
    np.savez(a.out + "_calib.npz", cal_mean=cal["cal_mean"], cal_std=cal["cal_std"])
    print("wrote", a.out + "_rec.csv", a.out + "_labels.csv")


if __name__ == "__main__":
    main()
