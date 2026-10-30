"""Does training on ray-traced rooms (research/rt) work better on REAL ESP32 data than the old simulator?

Same features, classifier and ESPectre protocol as phase_exp.py: choose on `selection`, report
`holdout` once. Amplitude features only (the ray-traced CSI's phase is random per packet, as real).

python bench/rt_sim2real.py cache/rt_pilot
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.phase_exp import CLASS, NA, ROOT, _real, metrics  # noqa: E402
from wujud.roomfeat import amplitude_from_iq, windows  # noqa: E402


def ray_traced_session(path):
    z = np.load(path)
    activity = str(z["activity"])
    room = int(os.path.basename(path)[4:9])
    rows = []
    for link in range(z["iq"].shape[0]):
        X, _ = windows(z["t"], amplitude_from_iq(z["iq"][link]), z["rssi"][link].astype(float))
        if len(X):
            rows.append(dict(X=X, y=CLASS[activity], room=room))
    return rows


def train(rows, cols):
    X = np.vstack([r["X"][:, cols] for r in rows])
    y = np.concatenate([[r["y"]] * len(r["X"]) for r in rows])
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, min_samples_leaf=40, class_weight="balanced", random_state=0
    ).fit(X, y)


def main():
    rt_dir = sys.argv[1]
    with Pool(max(1, os.cpu_count() // 2)) as pool:
        ray = [r for rows in pool.map(ray_traced_session, sorted(glob.glob(f"{rt_dir}/room*.npz"))) for r in rows]
        info = json.load(open(os.path.join(ROOT, "dataset_info.json")))
        jobs = [(lab, m) for lab, fs in info["files"].items() for m in fs if m.get("dataset_role") in ("selection", "holdout")]
        real = pool.map(_real, jobs)
    old = [dict(X=r["X"][:, :NA], y=r["y"]) for r in joblib.load("cache/room_sim_phase.joblib")]
    for r in real:
        r["X"] = r["X"][:, :NA]
    sel = [r for r in real if r["role"] == "selection"]
    hold = [r for r in real if r["role"] == "holdout"]
    print(f"ray-traced: {len({r['room'] for r in ray})} rooms, {sum(len(r['X']) for r in ray)} windows; "
          f"old simulator: {sum(len(r['X']) for r in old)} windows")

    cols = list(range(NA))
    variants = {"old simulator": old, "ray-traced": ray, "both": old + ray}
    results = {}
    for name, rows in variants.items():
        model = train(rows, cols)
        results[name] = dict(selection=metrics(model, sel, cols), holdout=metrics(model, hold, cols))
        s = results[name]["selection"]
        print(f"{name:16s} SELECTION motion {s['motion_recall']:.3f}/{s['motion_false_alarm']:.3f}  "
              f"still-vs-empty AUC window {s['still_vs_empty_auc']:.3f} recording {s['still_vs_empty_auc_recording']:.3f}", flush=True)
    ok = {k: v for k, v in results.items() if v["selection"]["motion_false_alarm"] <= 0.02} or results
    best = max(ok, key=lambda k: ok[k]["selection"]["still_vs_empty_auc"])
    print(f"\nchosen on selection: {best}")
    for name in variants:
        h = results[name]["holdout"]
        print(f"{name:16s} HOLDOUT   motion {h['motion_recall']:.3f}/{h['motion_false_alarm']:.3f}  "
              f"still-vs-empty AUC window {h['still_vs_empty_auc']:.3f} recording {h['still_vs_empty_auc_recording']:.3f}")
    results["chosen_on_selection"] = best
    json.dump(results, open(f"results/rt_sim2real_{os.path.basename(rt_dir.rstrip('/'))}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
