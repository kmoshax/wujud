"""The sim-to-real test for room mode (1 router + 1 ESP32).

Train ONLY on simulated rooms, test on REAL ESP32 recordings from the ESPectre project
(5 chip families, several real rooms, labels empty / static_presence / motion).
Also trains the same classifier on ESPectre's own real 'train' split, as a reference.

python bench/room_sim2real.py
"""
import os, sys, json, time
from multiprocessing import Pool
import numpy as np
import joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wujud.scenes3d import random_room
from wujud.sim3d import simulate_room
from wujud.roomfeat import windows, amplitude_from_iq, FEAT_NAMES
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

CLASS = {"away": 0, "lie": 1, "sit": 1, "stand": 1, "walk": 2}
NAMES = ["empty", "still person", "moving"]


def _sim_room(seed):
    rng = np.random.default_rng(seed)
    sc = random_room(rng, smart=rng.random() < 0.3)
    fs = rng.uniform(50, 120)
    harsh = rng.uniform(0.8, 1.6)
    rows = []
    for act in ("away", "lie", "sit", "stand", "walk"):
        out, lab, _ = simulate_room(sc, np.random.default_rng([seed, hash(act) % 1000]), seconds=60, fs=fs, plan=[(act, 60)], harsh=harsh)
        o = out[0]
        X, r = windows(o["t"], amplitude_from_iq(o["iq"]), o["rssi"].astype(float))
        rows.append(dict(X=X, y=CLASS[act], act=act, rates=r, breath=lab["breath_hz"] * 60, smart=sc.meta["smart"], seed=seed))
    return rows


def sim_set(n, seed0, workers=os.cpu_count()):
    with Pool(workers) as p:
        res = p.map(_sim_room, [seed0 + k for k in range(n)])
    return [r for rr in res for r in rr]


def stack(rows, cols):
    X = np.vstack([r["X"][:, cols] for r in rows]); y = np.concatenate([[r.get("y", r.get("label"))] * len(r["X"]) for r in rows])
    return X, y


def report(name, m, rows, cols):
    X, y = stack(rows, cols)
    p = m.predict(X); pr = m.predict_proba(X)
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    mot_recall = (p[y == 2] == 2).mean(); mot_fp = (p[y < 2] == 2).mean()
    s = y < 2
    auc_es = roc_auc_score(y[s], pr[s, 1] / (pr[s, 0] + pr[s, 1] + 1e-9))
    acc = (p == y).mean()
    print(f"\n== {name}: accuracy {acc:.3f} | motion recall {mot_recall:.3f} FP {mot_fp:.3f} | empty-vs-still AUC {auc_es:.3f}")
    print("   confusion (rows true empty/still/moving):\n", cm)
    return dict(acc=float(acc), motion_recall=float(mot_recall), motion_fp=float(mot_fp), auc_empty_still=float(auc_es), cm=cm.tolist())


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs("cache", exist_ok=True)
    if os.path.exists("cache/room_sim.joblib"):
        sim = joblib.load("cache/room_sim.joblib")
    else:
        sim = sim_set(500, 10_000)
        joblib.dump(sim, "cache/room_sim.joblib")
    print(f"sim windows: {sum(len(r['X']) for r in sim)} ({time.time() - t0:.0f}s)")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from espectre_real import load_real
    if os.path.exists("cache/espectre_feats_all.joblib") and os.path.getmtime("cache/espectre_feats_all.joblib") > os.path.getmtime("wujud/roomfeat.py"):
        real = joblib.load("cache/espectre_feats_all.joblib")
    else:
        real = load_real(skip_long=False); joblib.dump(real, "cache/espectre_feats_all.joblib")
    test = [r for r in real if r["role"] in ("selection", "holdout")]
    train_real = [r for r in real if r["role"] == "train"]
    res = {}
    nf = sim[0]["X"].shape[1]
    for feat_set, cols in (("all", list(range(nf))), ("no-rssi", [c for c in range(nf) if c != len(FEAT_NAMES)])):
        GB = dict(max_iter=300, learning_rate=0.05, min_samples_leaf=40, class_weight="balanced", random_state=0)
        Xs, ys = stack(sim, cols)
        m_sim = HistGradientBoostingClassifier(**GB).fit(Xs, ys)
        Xr, yr = stack(train_real, cols)
        m_real = HistGradientBoostingClassifier(**GB).fit(Xr, yr)
        res[feat_set] = dict(sim_only=report(f"[{feat_set}] trained on SIMULATION only -> real ESP32 test", m_sim, test, cols),
                             real_trained=report(f"[{feat_set}] trained on real ESPectre train split -> same test", m_real, test, cols))
        joblib.dump(m_sim, f"models/room_sim_{feat_set}.joblib")
        # recording-level: average over the first minute of each recording (empty vs still person)
        ys, ps = [], []
        for r in test:
            if r["label"] == 2:
                continue
            k = m_sim.predict_proba(r["X"][:9][:, cols]); ys.append(r["label"]); ps.append(np.mean(k[:, 1] / (k[:, 0] + k[:, 1] + 1e-9)))
        res[feat_set]["sim_only"]["auc_empty_still_recording"] = float(roc_auc_score(ys, ps))
        print(f"   recording-level empty-vs-still AUC (sim-trained): {res[feat_set]['sim_only']['auc_empty_still_recording']:.3f}")
    # physics consistency: how often is breathing clearly visible for a still person?
    json.dump(res, open("results/room_sim2real.json", "w"), indent=1)
    print(f"\ndone {time.time() - t0:.0f}s")
