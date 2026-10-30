"""Calibrate the simulator's realism knobs so that breathing is as (in)visible as on real ESP32 links.

bench/breath_visibility.py showed the simulator over-promises: a still person's breathing clears a 5%-false-alarm
detector in ~50% of simulated windows but ~11% of real ones. Here a small grid over torso RCS, chest excursion and
receiver noise is scored against that number measured on ESPectre's TRAIN split only, then the closest setting is
used to rebuild the training set and the room model is re-scored with the same selection/holdout rule as
bench/phase_exp.py (choose on selection; holdout is reported, never used to choose).

python bench/sim_calibrate.py
"""
import json, os, sys, itertools
from multiprocessing import Pool
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hallsense.scenes3d import random_room
from hallsense.sim3d import simulate_room
from hallsense.roomfeat import windows, amplitude_from_iq
from phase_exp import _real, metrics, ROOT, NA, RSSI_COL
from breath_visibility import NAMES, FA

CLASS = {"away": 0, "lie": 1, "sit": 1, "stand": 1, "walk": 2}
GRID = dict(rcs_scale=[1.0, 0.3, 0.1], breath_scale=[1.0, 0.5], evm_shift=[0.0, 6.0])


def _room(args):
    seed, knobs, acts = args
    rng = np.random.default_rng(seed)
    sc = random_room(rng, smart=rng.random() < 0.3)
    fs = rng.uniform(50, 120); harsh = rng.uniform(0.8, 1.6)
    rows = []
    for act in acts:
        out, _, _ = simulate_room(sc, np.random.default_rng([seed, hash(act) % 1000]), seconds=60, fs=fs, plan=[(act, 60)],
                                  harsh=harsh, **knobs)
        o = out[0]
        X, _ = windows(o["t"], amplitude_from_iq(o["iq"]), o["rssi"].astype(float))
        rows.append(dict(X=X, y=CLASS[act]))
    return rows


def sim_rows(knobs, seeds, acts, pool):
    return [r for rr in pool.map(_room, [(s, knobs, acts) for s in seeds]) for r in rr]


def visibility(rows, col):
    e = np.vstack([r["X"] for r in rows if r["y"] == 0]); s = np.vstack([r["X"] for r in rows if r["y"] == 1])
    return float((s[:, col] > np.quantile(e[:, col], 1 - FA)).mean())


if __name__ == "__main__":
    col = NAMES.index("glrt")
    info = json.load(open(os.path.join(ROOT, "dataset_info.json")))
    with Pool(os.cpu_count()) as pool:
        real = pool.map(_real, [(lab, m) for lab, fs in info["files"].items() for m in fs if m.get("dataset_role") != "exclude"])
        train = [dict(X=r["X"][:, :NA], y=r["y"]) for r in real if r["role"] == "train" and r["y"] < 2]
        target = visibility(train, col)
        print(f"real TRAIN split: breathing visible in {target:.1%} of still windows (5% false alarm)", flush=True)
        scores = []
        for vals in itertools.product(*GRID.values()):
            knobs = dict(zip(GRID, vals))
            v = visibility(sim_rows(knobs, range(20_000, 20_120), ("away", "lie", "sit", "stand"), pool), col)
            scores.append((abs(v - target), knobs, v)); print(f"  {knobs} -> sim {v:.1%}", flush=True)
        _, best, vbest = min(scores, key=lambda s: s[0])
        print(f"closest: {best} (sim {vbest:.1%} vs real {target:.1%})", flush=True)
        sim = sim_rows(best, range(10_000, 10_500), ("away", "lie", "sit", "stand", "walk"), pool)
    joblib.dump(dict(knobs=best, rows=sim), "cache/room_sim_calibrated.joblib")
    sel = [dict(X=r["X"][:, :NA], y=r["y"]) for r in real if r["role"] == "selection"]
    hold = [dict(X=r["X"][:, :NA], y=r["y"]) for r in real if r["role"] == "holdout"]
    base = json.load(open("results/phase_exp.json"))["amplitude (current)"]
    Xs = np.vstack([r["X"] for r in sim]); ys = np.concatenate([[r["y"]] * len(r["X"]) for r in sim])
    res = dict(target_visibility_train=target, grid=[dict(knobs=k, sim_visibility=v) for _, k, v in scores], chosen_knobs=best,
               baseline=base)
    for name, cols in (("calibrated sim, amplitude", list(range(NA))), ("calibrated sim, amplitude no RSSI", [c for c in range(NA) if c != RSSI_COL])):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=40, class_weight="balanced",
                                           random_state=0).fit(Xs[:, cols], ys)
        res[name] = dict(selection=metrics(m, sel, cols), holdout=metrics(m, hold, cols))
        joblib.dump(m, f"cache/room_sim_calibrated_{'all' if len(cols) == NA else 'no-rssi'}.joblib")
    for name in ("calibrated sim, amplitude", "calibrated sim, amplitude no RSSI"):
        for sp in ("selection", "holdout"):
            h = res[name][sp]
            print(f"{name:34s} {sp:9s} motion {h['motion_recall']:.3f}/{h['motion_false_alarm']:.3f}  still-vs-empty AUC win {h['still_vs_empty_auc']:.3f} rec {h['still_vs_empty_auc_recording']:.3f}")
    for sp in ("selection", "holdout"):
        h = base[sp]
        print(f"{'baseline (current model)':34s} {sp:9s} motion {h['motion_recall']:.3f}/{h['motion_false_alarm']:.3f}  still-vs-empty AUC win {h['still_vs_empty_auc']:.3f} rec {h['still_vs_empty_auc_recording']:.3f}")
    json.dump(res, open("results/sim_calibrate.json", "w"), indent=1)
