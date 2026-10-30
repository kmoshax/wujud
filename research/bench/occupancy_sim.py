"""Occupancy latch vs per-window classifier on simulated DAYS IN A ROOM (enter, stay still, leave, ...).

Simulator realism knobs set to the calibrated values from bench/sim_calibrate.py (breathing about as invisible as on
real links), so the still person is hard here too. Threshold tuned on rooms 30000-30059, reported on 31000-31099.
Scored on windows that contain no walking (still person vs empty room), plus the walking windows themselves.

python bench/occupancy_sim.py
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
import joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hallsense.scenes3d import random_room, ACTIVITIES
from hallsense.sim3d import simulate_room
from hallsense.roomfeat import windows, amplitude_from_iq, profile_windows, WIN_S, HOP_S, FS_GRID
from hallsense.occupancy import OccupancyLatch

KNOBS = json.load(open("results/sim_calibrate.json"))["chosen_knobs"]
NA = 17


def plan(rng):
    p = [("away", rng.uniform(90, 150))]
    while sum(d for _, d in p) < 560:
        p += [("walk", rng.uniform(8, 15)), (rng.choice(["sit", "lie", "stand"]), rng.uniform(60, 180)),
              ("walk", rng.uniform(8, 15)), ("away", rng.uniform(60, 150))]
    return p


def _day(seed):
    rng = np.random.default_rng(seed)
    sc = random_room(rng, smart=rng.random() < 0.3)
    fs = rng.uniform(50, 120)
    pl = plan(rng)
    out, lab, _ = simulate_room(sc, np.random.default_rng(seed + 7), seconds=sum(d for _, d in pl), fs=fs, plan=pl,
                                harsh=rng.uniform(0.8, 1.6), **KNOBS)
    o = out[0]
    amp = amplitude_from_iq(o["iq"])
    X, _ = windows(o["t"], amp, o["rssi"].astype(float)); Pf = profile_windows(o["t"], amp)[:len(X)]
    act = np.asarray(lab["activity"]); ft = np.arange(len(act)) / fs
    y, walk = [], []
    for k in range(len(X)):
        a = act[(ft >= k * HOP_S) & (ft < k * HOP_S + WIN_S)]
        names = [ACTIVITIES[i] for i in a]
        walk.append("walk" in names)
        y.append(0 if names.count("away") > len(names) / 2 else 1)
    return dict(X=X, P=Pf, y=np.array(y), walk=np.array(walk))


def run(days, model, thr):
    res = dict(cls=[], latch=[], y=[], walk=[], move=[], d=[])
    for d in days:
        pr = model.predict_proba(d["X"]); L = OccupancyLatch(thr=thr)
        for x, p, y, w in zip(d["P"], pr, d["y"], d["walk"]):
            st, dist = L.step(x, p[2], p[1] / (p[0] + p[1] + 1e-9)); res["d"].append(np.nan if dist is None else dist)
            res["latch"].append(0 if st == "empty" else 1); res["move"].append(st == "moving")
            res["cls"].append(0 if np.argmax(p) == 0 else 1); res["y"].append(y); res["walk"].append(w)
    r = {k: np.array(v) for k, v in res.items()}
    q = ~r["walk"]
    score = lambda pred: dict(accuracy=float((pred[q] == r["y"][q]).mean()), still_detected=float(pred[q & (r["y"] == 1)].mean()),
                              false_present_when_empty=float(pred[q & (r["y"] == 0)].mean()))
    from sklearn.metrics import roc_auc_score
    m = q & np.isfinite(r["d"])
    return dict(distance_auc_still_vs_empty=float(roc_auc_score(r["y"][m], r["d"][m])), windows_with_fresh_reference=int(m.sum()),
                per_window_classifier=score(r["cls"]), occupancy_latch=score(r["latch"]),
                walking_flagged_as_motion=float(r["move"][r["walk"]].mean()), quiet_windows=int(q.sum()))


if __name__ == "__main__":
    model = joblib.load("models/room_sim_all.joblib")
    with Pool(os.cpu_count()) as p:
        tune = p.map(_day, range(30_000, 30_060)); test = p.map(_day, range(31_000, 31_100))
    grid = {t: run(tune, model, t)["occupancy_latch"]["accuracy"] for t in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0)}
    thr = max(grid, key=grid.get)
    res = dict(knobs=KNOBS, tune_grid=grid, thr=thr, test=run(test, model, thr))
    print("tune:", {k: round(v, 3) for k, v in grid.items()}, "-> thr", thr)
    print(json.dumps(res["test"], indent=1))
    json.dump(res, open("results/occupancy_sim.json", "w"), indent=1)
