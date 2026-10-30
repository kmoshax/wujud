"""Does sanitized CSI phase help the simulation-trained room model on REAL ESP32 data?

Same protocol as room_sim2real.py (train on simulation only), but with ESPectre's splits used properly:
  selection -> the only split used to choose a variant
  holdout   -> reported once, for the chosen variant (and the others, for transparency)
Variants: amplitude (current), amplitude without RSSI, + sanitized phase, phase only.

python bench/phase_exp.py
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wujud.scenes3d import random_room
from wujud.sim3d import simulate_room
from wujud.roomfeat import windows, amplitude_from_iq, phase_from_iq, phase_windows, FEAT_NAMES

ROOT = os.environ.get("ESPECTRE_DATA", os.path.expanduser("~/Projects/ext/espectre/data"))
CLASS = {"away": 0, "lie": 1, "sit": 1, "stand": 1, "walk": 2}
LAB = {"empty": 0, "static_presence": 1, "motion": 2}
NA = len(FEAT_NAMES) + 1 + 6                      # amplitude block: 10 core + rssi + 6 extra
RSSI_COL = len(FEAT_NAMES)


def _sim_room(seed):
    rng = np.random.default_rng(seed)
    sc = random_room(rng, smart=rng.random() < 0.3)
    fs = rng.uniform(50, 120); harsh = rng.uniform(0.8, 1.6)
    rows = []
    for act in ("away", "lie", "sit", "stand", "walk"):
        out, lab, _ = simulate_room(sc, np.random.default_rng([seed, hash(act) % 1000]), seconds=60, fs=fs, plan=[(act, 60)], harsh=harsh)
        o = out[0]
        X, _ = windows(o["t"], amplitude_from_iq(o["iq"]), o["rssi"].astype(float))
        P = phase_windows(o["t"], phase_from_iq(o["iq"]))
        n = min(len(X), len(P))
        rows.append(dict(X=np.c_[X[:n], P[:n]], y=CLASS[act], seed=seed))
    return rows


def _real(arg):
    lab, m = arg
    z = np.load(os.path.join(ROOT, lab, m["filename"]))
    t = np.unwrap(z["wifi_rx_ts_us"].astype(np.float64), period=2 ** 32) / 1e6
    o = np.argsort(t); t = t[o]; iq = z["csi_data"][o].reshape(len(o), -1, 2)
    X, _ = windows(t, amplitude_from_iq(iq), z["rssi_dbm"][o].astype(float))
    P = phase_windows(t, phase_from_iq(iq))
    n = min(len(X), len(P))
    return dict(X=np.c_[X[:n], P[:n]], y=LAB[lab], role=m["dataset_role"], file=m["filename"], chip=m["chip"])


def metrics(m, rows, cols):
    X = np.vstack([r["X"][:, cols] for r in rows]); y = np.concatenate([[r["y"]] * len(r["X"]) for r in rows])
    p = m.predict(X); pr = m.predict_proba(X); s = y < 2
    still = pr[:, 1] / (pr[:, 0] + pr[:, 1] + 1e-9)
    ys, ps = [], []
    for r in rows:                                 # recording level: first ~minute
        if r["y"] < 2:
            k = m.predict_proba(r["X"][:9][:, cols]); ys.append(r["y"]); ps.append(np.mean(k[:, 1] / (k[:, 0] + k[:, 1] + 1e-9)))
    return dict(motion_recall=float((p[y == 2] == 2).mean()), motion_false_alarm=float((p[y < 2] == 2).mean()),
                still_vs_empty_auc=float(roc_auc_score(y[s], still[s])), still_vs_empty_auc_recording=float(roc_auc_score(ys, ps)),
                n_windows=int(len(y)), n_recordings=len(rows))


if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    cp = "cache/room_sim_phase.joblib"
    if os.path.exists(cp):
        sim = joblib.load(cp)
    else:
        with Pool(os.cpu_count()) as pool:
            sim = [r for rr in pool.map(_sim_room, range(10_000, 10_500)) for r in rr]
        joblib.dump(sim, cp)
    info = json.load(open(os.path.join(ROOT, "dataset_info.json")))
    jobs = [(lab, m) for lab, fs in info["files"].items() for m in fs if m.get("dataset_role") in ("selection", "holdout")]
    with Pool(os.cpu_count()) as pool:
        real = pool.map(_real, jobs)
    sel = [r for r in real if r["role"] == "selection"]; hold = [r for r in real if r["role"] == "holdout"]
    NP = sim[0]["X"].shape[1] - NA
    amp = list(range(NA)); amp_nr = [c for c in amp if c != RSSI_COL]; ph = list(range(NA, NA + NP))
    variants = {"amplitude (current)": amp, "amplitude, no RSSI": amp_nr, "amplitude + phase": amp + ph,
                "amplitude no RSSI + phase": amp_nr + ph, "phase only": ph}
    Xs = np.vstack([r["X"] for r in sim]); ys = np.concatenate([[r["y"]] * len(r["X"]) for r in sim])
    res = {}
    for name, cols in variants.items():
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=40, class_weight="balanced",
                                           random_state=0).fit(Xs[:, cols], ys)
        res[name] = dict(selection=metrics(m, sel, cols), holdout=metrics(m, hold, cols))
        joblib.dump(dict(model=m, cols=cols), f"cache/phase_exp_{name.replace(' ', '_').replace(',', '').replace('+', 'plus').replace('(', '').replace(')', '')}.joblib")
        s = res[name]["selection"]
        print(f"{name:28s} SELECTION motion {s['motion_recall']:.3f}/{s['motion_false_alarm']:.3f}  still-vs-empty AUC win {s['still_vs_empty_auc']:.3f} rec {s['still_vs_empty_auc_recording']:.3f}", flush=True)
    # choose on selection only: still-vs-empty window AUC, with motion false alarms kept <= 2%
    ok = {k: v for k, v in res.items() if v["selection"]["motion_false_alarm"] <= 0.02}
    best = max(ok, key=lambda k: ok[k]["selection"]["still_vs_empty_auc"])
    res["chosen_on_selection"] = best
    print("\nchosen on selection:", best)
    for name in variants:
        h = res[name]["holdout"]
        print(f"{name:28s} HOLDOUT   motion {h['motion_recall']:.3f}/{h['motion_false_alarm']:.3f}  still-vs-empty AUC win {h['still_vs_empty_auc']:.3f} rec {h['still_vs_empty_auc_recording']:.3f}")
    json.dump(res, open("results/phase_exp.json", "w"), indent=1)
