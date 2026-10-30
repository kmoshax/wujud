"""Room mode in simulation, on rooms never used for training:
  - activity (away / lie / sit / stand / walk) with 1 board vs 3 boards, random vs smart placement
  - breathing-rate error when still
  - location error with 3 boards (physics-based Fresnel matching, no training)
python bench/room_eval.py
"""
import os, sys, json, time
from multiprocessing import Pool
import numpy as np, joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wujud.scenes3d import random_room, ACTIVITIES
from wujud.sim3d import simulate_room
from wujud.roomfeat import windows, amplitude_from_iq, FS_GRID, WIN_S, HOP_S
from wujud.roomloc import locate
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix


def agg(per_link):
    """[L, W, F] -> [W, 3F] link-count-invariant (mean, max, min over links)"""
    A = np.stack(per_link)
    return np.concatenate([A.mean(0), A.max(0), A.min(0)], 1)


def _room(args):
    seed, n_esp, smart = args
    rng = np.random.default_rng(seed)
    sc = random_room(rng, n_esp=n_esp, mesh=False, smart=smart)   # router -> each board, as the room firmware does
    rows = []
    for act in ACTIVITIES:
        out, lab, B = simulate_room(sc, np.random.default_rng([seed, ACTIVITIES.index(act)]), seconds=60, fs=50, plan=[(act, 60)])
        per, rates, peaks = [], [], []
        for o in out:
            X, r = windows(o["t"], amplitude_from_iq(o["iq"]), o["rssi"].astype(float))
            per.append(X); rates.append(r); peaks.append(X[:, 6])
        nW = min(len(x) for x in per)
        per = [x[:nW] for x in per]
        # true position per window
        xy = lab["xy"]; fs = lab["fs"]
        wpos = []
        for w in range(nW):
            s0 = int(w * HOP_S * fs); s1 = int((w * HOP_S + WIN_S) * fs)
            wpos.append(np.nanmean(xy[s0:s1], 0) if np.isfinite(xy[s0:s1, 0]).any() else [np.nan, np.nan])
        rates = np.stack([r[:nW] for r in rates]); peaks = np.stack([p[:nW] for p in peaks])
        best = peaks.argmax(0)
        est_rate = rates[best, np.arange(nW)]
        rows.append(dict(seed=seed, n_esp=n_esp, smart=smart, act=act, y=ACTIVITIES.index(act), X=agg(per),
                         per_link=np.stack(per), links=sc.links, nodes=sc.nodes, W=sc.W, D=sc.D,
                         xy=np.array(wpos), rate_true=lab["breath_hz"] * 60, rate_est=est_rate, peak=peaks.max(0)))
    return rows


def gen(n, seed0, n_esp, smart):
    with Pool(os.cpu_count()) as p:
        r = p.map(_room, [(seed0 + k, n_esp, smart) for k in range(n)])
    return [x for rr in r for x in rr]


if __name__ == "__main__":
    t0 = time.time(); os.makedirs("cache", exist_ok=True)
    res = {}
    for n_esp in (1, 3):
        for smart in (False, True):
            key = f"esp{n_esp}_{'smart' if smart else 'random'}"
            cf = f"cache/room_eval_{key}.joblib"
            if os.path.exists(cf):
                tr, te = joblib.load(cf)
            else:
                tr = gen(120, 50_000 + 1000 * n_esp + 500 * smart, n_esp, smart)
                te = gen(40, 90_000 + 1000 * n_esp + 500 * smart, n_esp, smart)
                joblib.dump((tr, te), cf)
            X = np.vstack([r["X"] for r in tr]); y = np.concatenate([[r["y"]] * len(r["X"]) for r in tr])
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=20, class_weight="balanced", random_state=0).fit(X, y)
            Xt = np.vstack([r["X"] for r in te]); yt = np.concatenate([[r["y"]] * len(r["X"]) for r in te])
            p = m.predict(Xt)
            cm = confusion_matrix(yt, p, labels=range(5))
            # 3-state view
            m3 = np.array([0, 1, 1, 1, 2])
            acc3 = (m3[p] == m3[yt]).mean()
            # breathing (still activities, confident windows)
            still = [r for r in te if r["act"] in ("lie", "sit", "stand")]
            err = np.concatenate([np.abs(r["rate_est"] - r["rate_true"]) for r in still])
            pk = np.concatenate([r["peak"] for r in still])
            conf = pk > np.percentile(pk, 50)
            out = dict(acc5=float((p == yt).mean()), acc3=float(acc3), cm=cm.tolist(),
                       breath_mae_all=float(np.median(err)), breath_within2_all=float(np.mean(err <= 2)),
                       breath_within2_confident=float(np.mean(err[conf] <= 2)), breath_median_err_confident=float(np.median(err[conf])))
            if n_esp >= 3:
                errs = []
                for r in te:
                    if r["act"] == "away":
                        continue
                    for w in range(len(r["xy"])):
                        if not np.isfinite(r["xy"][w, 0]):
                            continue
                        e = locate(r["per_link"][:, w], r["links"], r["nodes"], r["W"], r["D"])
                        errs.append(np.linalg.norm(e - r["xy"][w]))
                errs = np.array(errs)
                out.update(loc_median_m=float(np.median(errs)), loc_p80_m=float(np.percentile(errs, 80)))
            res[key] = out
            joblib.dump(m, f"models/room_activity_{key}.joblib")
            print(key, json.dumps({k: v for k, v in out.items() if k != "cm"}), flush=True)
            print(cm)
    json.dump(res, open("results/room_eval.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s")
