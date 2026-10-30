"""Per-installation self-calibration: does knowing THIS room's empty baseline separate 'still person' from 'empty'?

Deployment story: after installing the board you leave the room for 2 minutes; afterwards every 20 s window is
scored by how far it sits from that room's own empty statistics (robust z-scores, one-class; no labels of people).

Honest protocol on ESPectre (same board + same room only):
  calibrate on the first 120 s of empty session A
  test on   empty session B (a DIFFERENT recording)  vs  every static_presence recording of that board+room
Pairs with only one empty session are reported separately as 'same-session' (optimistic upper bound: the test
empties then come from the rest of the calibration recording). Empty recordings in ESPectre are often long night
runs while still-person runs are short daytime runs, so 'cross-session' is the number to believe.
Also reports the cross-room sim-trained model on exactly the same test windows, and accumulation over time.

python bench/self_calib.py
"""
import json, os, sys, collections
from multiprocessing import Pool
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_exp import _real, ROOT, NA, RSSI_COL

CAL_WIN = 21                          # 120 s of 20 s windows at 5 s hop


def score_fn(cal, cols):
    mu = np.median(cal[:, cols], 0)
    sd = np.maximum(1.4826 * np.median(np.abs(cal[:, cols] - mu), 0), 1e-3)
    return lambda X: np.sqrt(np.mean(((X[:, cols] - mu) / sd) ** 2, 1))


def accumulate(scores, k):
    """mean score over the last k windows (k*5 s of evidence + 15 s)"""
    c = np.cumsum(np.r_[0, scores]); return (c[k:] - c[:-k]) / k


if __name__ == "__main__":
    info = json.load(open(os.path.join(ROOT, "dataset_info.json")))
    metas = [(lab, m) for lab, fs in info["files"].items() if lab != "motion" for m in fs]
    with Pool(os.cpu_count()) as p:
        recs = p.map(_real, metas)
    for r, (lab, m) in zip(recs, metas):
        r.update(device=m.get("device_id"), env=m.get("environment"), t=m["collected_at"])
    sim = joblib.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "room_sim_all.joblib"))
    groups = collections.defaultdict(list)
    for r in recs:
        groups[(r["device"], r["env"])].append(r)
    feats = {"all features": list(range(NA)), "no RSSI": [c for c in range(NA) if c != RSSI_COL]}
    out = {"cross_session": [], "same_session": []}
    for (dev, env), rs in sorted(groups.items()):
        emp = sorted([r for r in rs if r["y"] == 0], key=lambda r: r["t"]); sta = [r for r in rs if r["y"] == 1]
        if not emp or not sta:
            continue
        cases = []
        if len(emp) >= 2:
            for a in emp:
                for b in emp:
                    if a is not b and len(a["X"]) >= CAL_WIN:
                        cases.append(("cross_session", a["X"][:CAL_WIN], b["X"]))
        elif len(emp[0]["X"]) >= CAL_WIN + 6:
            cases.append(("same_session", emp[0]["X"][:CAL_WIN], emp[0]["X"][CAL_WIN:]))
        for kind, cal, et in cases:
            S = np.vstack([r["X"] for r in sta])
            res = dict(device=dev, env=env, n_empty_test=len(et), n_still=len(S), n_still_recordings=len(sta))
            y = np.r_[np.zeros(len(et)), np.ones(len(S))]
            for name, cols in feats.items():
                f = score_fn(cal, cols)
                res[name] = float(roc_auc_score(y, np.r_[f(et), f(S)]))
                # one minute of evidence: average 9 consecutive windows inside each recording
                ae = accumulate(f(et), 9); as_ = np.concatenate([accumulate(f(r["X"]), 9) for r in sta if len(r["X"]) >= 9])
                res[name + ", 1 min"] = float(roc_auc_score(np.r_[np.zeros(len(ae)), np.ones(len(as_))], np.r_[ae, as_]))
                # the decision a deployment would make: threshold = 99th percentile of the calibration scores
                thr = np.quantile(f(cal), 0.99)
                res[name + ", says present when empty"] = float((f(et) > thr).mean())
                res[name + ", says present when still"] = float((f(S) > thr).mean())
            pr = lambda X: (lambda q: q[:, 1] / (q[:, 0] + q[:, 1] + 1e-9))(sim.predict_proba(X[:, :NA]))
            res["sim-trained model, no calibration"] = float(roc_auc_score(y, np.r_[pr(et), pr(S)]))
            out[kind].append(res)
    summary = {}
    for kind, rows in out.items():
        if not rows:
            continue
        keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
        summary[kind] = dict(cases=len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in keys})
        print(f"\n== {kind}: {len(rows)} calibration cases")
        for r in rows:
            print(f"  {str(r['device'])[-8:]} {r['env']:13s} empty-test {r['n_empty_test']:3d} still {r['n_still']:3d}  "
                  f"self-calib AUC {r['all features']:.2f} (1 min {r['all features, 1 min']:.2f})  no-RSSI {r['no RSSI']:.2f}  "
                  f"sim model {r['sim-trained model, no calibration']:.2f}")
        s = summary[kind]
        print(f"  MEAN self-calib {s['all features']:.3f} | 1 min {s['all features, 1 min']:.3f} | no RSSI {s['no RSSI']:.3f} | sim model {s['sim-trained model, no calibration']:.3f}"
              f" | at the 99% calib threshold: present when empty {s['all features, says present when empty']:.2f}, when still {s['all features, says present when still']:.2f}")
    json.dump(dict(cases=out, summary=summary), open("results/self_calib.json", "w"), indent=1)
