"""Laptop mode (RSSI only, no CSI): train on simulated RSSI, test on the REAL ESPectre recordings' RSSI.
python bench/rssi_sim2real.py
"""
import os, sys, json, time
from multiprocessing import Pool
import numpy as np, joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hallsense.scenes3d import random_room
from hallsense.sim3d import simulate_room
from hallsense.rssifeat import rssi_windows
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

CLASS = {"away": 0, "lie": 1, "sit": 1, "stand": 1, "walk": 2}
ROOT = os.environ.get("ESPECTRE_DATA", "/home/claude/ext/espectre/data")


def _room(seed):
    rng = np.random.default_rng(seed)
    sc = random_room(rng, smart=rng.random() < 0.3)
    fs = rng.uniform(10, 100)
    rows = []
    for k, act in enumerate(CLASS):
        out, lab, _ = simulate_room(sc, np.random.default_rng([seed, k]), seconds=60, fs=fs, plan=[(act, 60)], harsh=rng.uniform(0.8, 1.6))
        o = out[0]
        X, _ = rssi_windows(o["t"], o["rssi"])
        rows.append(dict(X=X, y=CLASS[act]))
    return rows


def real():
    info = json.load(open(os.path.join(ROOT, "dataset_info.json")))
    rows = []
    for lab, files in info["files"].items():
        for m in files:
            if m.get("dataset_role") == "exclude":
                continue
            z = np.load(os.path.join(ROOT, lab, m["filename"]))
            t = np.unwrap(z["wifi_rx_ts_us"].astype(float), period=2 ** 32) / 1e6
            o = np.argsort(t)
            X, _ = rssi_windows(t[o] - t[o][0], z["rssi_dbm"][o])
            rows.append(dict(X=X, y={"empty": 0, "static_presence": 1, "motion": 2}[lab], role=m["dataset_role"]))
    return rows


def stack(rows):
    return np.vstack([r["X"] for r in rows]), np.concatenate([[r["y"]] * len(r["X"]) for r in rows])


def report(name, m, rows):
    X, y = stack(rows); p = m.predict(X); pr = m.predict_proba(X); s = y < 2
    out = dict(acc=float((p == y).mean()), motion_recall=float((p[y == 2] == 2).mean()), motion_fp=float((p[y < 2] == 2).mean()),
               auc_empty_still=float(roc_auc_score(y[s], pr[s, 1] / (pr[s, 0] + pr[s, 1] + 1e-9))),
               auc_motion=float(roc_auc_score(y == 2, pr[:, 2])), cm=confusion_matrix(y, p, labels=[0, 1, 2]).tolist())
    print(name, json.dumps({k: round(v, 3) for k, v in out.items() if k != "cm"}), "\n", np.array(out["cm"]), flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    cf = "cache/rssi_sim.joblib"
    if os.path.exists(cf):
        sim = joblib.load(cf)
    else:
        with Pool(os.cpu_count()) as p:
            sim = [x for r in p.map(_room, range(30_000, 30_400)) for x in r]
        joblib.dump(sim, cf)
    rr = real(); test = [r for r in rr if r["role"] in ("selection", "holdout")]; tr = [r for r in rr if r["role"] == "train"]
    GB = dict(max_iter=300, learning_rate=0.05, min_samples_leaf=40, class_weight="balanced", random_state=0)
    m_sim = HistGradientBoostingClassifier(**GB).fit(*stack(sim))
    m_real = HistGradientBoostingClassifier(**GB).fit(*stack(tr))
    res = dict(sim_only=report("RSSI, trained on SIMULATION only -> real", m_sim, test),
               real_trained=report("RSSI, trained on real train split -> real", m_real, test))
    joblib.dump(m_sim, "models/laptop_rssi.joblib")
    json.dump(res, open("results/rssi_sim2real.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s")
