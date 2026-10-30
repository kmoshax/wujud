"""Two experiments in simulation (rooms never used for training):

1. "What moved?"  nobody / a human / something else (pet, robot vacuum, door, fan),
   also with a human present together with a pet or a fan.
2. Through walls  boards outside the room, every wall loss from glass (3 dB) to reinforced
   concrete (40 dB, one way, Wi-Vi Table 1). Can we still see a breathing, motionless person?

python bench/room_world.py
"""
import os, sys, json, time
from multiprocessing import Pool
import numpy as np, joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wujud.scenes3d import random_room
from wujud.sim3d import simulate_room
from wujud.roomfeat import windows, amplitude_from_iq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

CASES = [("nothing", (), "away", 0), ("fan", ("fan",), "away", 2), ("pet", ("pet",), "away", 2), ("door", ("door",), "away", 2),
         ("vacuum", ("vacuum",), "away", 2), ("human still", (), "still", 1), ("human moving", (), "walk", 1),
         ("human still + fan", ("fan",), "still", 1), ("human still + pet", ("pet",), "still", 1), ("human moving + pet", ("pet",), "walk", 1)]
WALLS = [0, 3, 6, 9, 12, 18, 25, 30, 40]


def feats(out):
    o = out[0]
    X, r = windows(o["t"], amplitude_from_iq(o["iq"]), o["rssi"].astype(float))
    return X, r


def _world(seed):
    rng = np.random.default_rng(seed)
    sc = random_room(rng, smart=rng.random() < 0.3)
    rows = []
    for k, (name, dis, act, y) in enumerate(CASES):
        a = rng.choice(["lie", "sit", "stand"]) if act == "still" else act
        out, lab, _ = simulate_room(sc, np.random.default_rng([seed, k]), seconds=40, fs=50, plan=[(a, 40)], distractors=dis)
        X, _ = feats(out)
        rows.append(dict(seed=seed, case=name, y=y, X=X))
    return rows


def _wall(args):
    seed, wall = args
    rng = np.random.default_rng(seed)
    sc = random_room(rng, smart=True)
    # boards just outside the two side walls (rescuers put them around the space)
    sc.nodes[0, 0] = -0.3 if sc.nodes[0, 0] < sc.W / 2 else sc.W + 0.3
    sc.nodes[1, 0] = -0.3 if sc.nodes[1, 0] < sc.W / 2 else sc.W + 0.3
    rows = []
    for k, act in enumerate(["away", "lie", "sit", "walk"]):
        out, lab, _ = simulate_room(sc, np.random.default_rng([seed, k, wall]), seconds=40, fs=50, plan=[(act, 40)], wall_db=wall, outside=True)
        X, r = feats(out)
        rows.append(dict(seed=seed, wall=wall, act=act, X=X, rate=r, true=lab["breath_hz"] * 60))
    return rows


if __name__ == "__main__":
    t0 = time.time(); res = {}
    cf = "cache/room_world.joblib"
    if os.path.exists(cf):
        tr, te, wl = joblib.load(cf)
        if os.environ.get("REDO_WALLS"):
            with Pool(os.cpu_count()) as p:
                wl = [x for r in p.map(_wall, [(s, w) for s in range(60_000, 60_040) for w in WALLS]) for x in r]
            joblib.dump((tr, te, wl), cf)
    else:
        with Pool(os.cpu_count()) as p:
            tr = [x for r in p.map(_world, range(70_000, 70_160)) for x in r]
            te = [x for r in p.map(_world, range(80_000, 80_050)) for x in r]
            wl = [x for r in p.map(_wall, [(s, w) for s in range(60_000, 60_040) for w in WALLS]) for x in r]
        joblib.dump((tr, te, wl), cf)
    print(f"simulated ({time.time() - t0:.0f}s)", flush=True)

    X = np.vstack([r["X"] for r in tr]); y = np.concatenate([[r["y"]] * len(r["X"]) for r in tr])
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, min_samples_leaf=20, class_weight="balanced", random_state=0).fit(X, y)
    joblib.dump(m, "models/room_what_moved.joblib")
    per_case = {}
    for name, *_ in CASES:
        rr = [r for r in te if r["case"] == name]
        Xt = np.vstack([r["X"] for r in rr]); pr = m.predict(Xt)
        per_case[name] = {"nothing": float(np.mean(pr == 0)), "human": float(np.mean(pr == 1)), "other": float(np.mean(pr == 2))}
    Xt = np.vstack([r["X"] for r in te]); yt = np.concatenate([[r["y"]] * len(r["X"]) for r in te])
    pt = m.predict(Xt)
    hum = (yt == 1); pred_h = (pt == 1)
    res["what_moved"] = dict(acc3=float(np.mean(pt == yt)), human_recall=float(pred_h[hum].mean()),
                             human_false_alarm_from_objects=float(pred_h[yt == 2].mean()),
                             human_false_alarm_empty=float(pred_h[yt == 0].mean()), per_case=per_case,
                             cm=confusion_matrix(yt, pt, labels=[0, 1, 2]).tolist())
    print(json.dumps(res["what_moved"], indent=1), flush=True)

    curve = []
    GL, MI = 16, 3
    for w in WALLS:
        rr = [r for r in wl if r["wall"] == w]
        e = [r["X"][:, GL].mean() for r in rr if r["act"] == "away"]
        s = [r["X"][:, GL].mean() for r in rr if r["act"] in ("lie", "sit")]
        mv = [r["X"][:, MI].mean() for r in rr if r["act"] == "walk"]
        em = [r["X"][:, MI].mean() for r in rr if r["act"] == "away"]
        errs = [np.median(np.abs(r["rate"] - r["true"])) for r in rr if r["act"] == "lie"]
        curve.append(dict(wall_db=w, still_auc=float(roc_auc_score([0] * len(e) + [1] * len(s), e + s)),
                          motion_auc=float(roc_auc_score([0] * len(em) + [1] * len(mv), em + mv)),
                          breath_err_lying_median=float(np.median(errs)), breath_within2=float(np.mean(np.array(errs) <= 2))))
        print(curve[-1], flush=True)
    res["through_wall"] = curve
    json.dump(res, open("results/room_world.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s")
