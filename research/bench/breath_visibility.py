"""How often is a still person's breathing visible on ONE link? Simulator prediction vs real ESP32 recordings.

"Visible" is defined identically in both worlds (CFAR style): take the breathing detector score (the
Patwari-style GLRT feature 'glrt' and the breathing-band ratio 'br_top' from hallsense/roomfeat.py), set its
threshold so that only 5% of EMPTY-room windows of that same world exceed it, and count the share of
still-person windows above it. Each world is calibrated on its own empty rooms, nothing is tuned across.

Needs cache/room_sim_phase.joblib (made by bench/phase_exp.py).
python bench/breath_visibility.py
"""
import json, os, sys
import numpy as np
import joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hallsense.roomfeat import FEAT_NAMES, EXTRA_NAMES
from phase_exp import _real, ROOT, NA

NAMES = FEAT_NAMES + ["rssi"] + EXTRA_NAMES
FA = 0.05


def visible_share(empty, still, col):
    thr = np.quantile(empty[:, col], 1 - FA)
    return float((still[:, col] > thr).mean())


if __name__ == "__main__":
    from multiprocessing import Pool
    sim = joblib.load("cache/room_sim_phase.joblib")
    se = np.vstack([r["X"][:, :NA] for r in sim if r["y"] == 0]); ss = np.vstack([r["X"][:, :NA] for r in sim if r["y"] == 1])
    info = json.load(open(os.path.join(ROOT, "dataset_info.json")))
    jobs = [(lab, m) for lab, fs in info["files"].items() if lab != "motion" for m in fs if m.get("dataset_role") != "exclude"]
    with Pool(os.cpu_count()) as p:
        real = p.map(_real, jobs)
    re_ = np.vstack([r["X"][:, :NA] for r in real if r["y"] == 0]); rs = np.vstack([r["X"][:, :NA] for r in real if r["y"] == 1])
    res = {"false_alarm_on_empty": FA, "sim": {"empty_windows": len(se), "still_windows": len(ss)},
           "real": {"empty_windows": len(re_), "still_windows": len(rs), "recordings": len(real)}}
    for name in ("glrt", "br_top"):
        c = NAMES.index(name)
        res["sim"][name] = visible_share(se, ss, c); res["real"][name] = visible_share(re_, rs, c)
        print(f"{name:7s} breathing visible in still-person windows: simulation {res['sim'][name]:.1%}   real ESP32 {res['real'][name]:.1%}")
    json.dump(res, open("results/breath_visibility.json", "w"), indent=1)
