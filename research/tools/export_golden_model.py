"""Golden reference data for crates/model (tests in crates/model/tests/).

Writes small .npy/.json files to crates/model/tests/golden/ from REAL recordings (ESPectre ESP32 data, the laptop
beacon log in rec/) and from the shipped models, run through the Python reference code.

    cd research && .venv/bin/python tools/export_golden_model.py
    cd research && .venv/bin/python tools/export_golden_model.py --bench   # + big benchmark arrays in research/cache/
"""
import argparse, collections, json, os, sys, types
import numpy as np
import joblib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)
from hallsense.roomfeat import (to_grid, amplitude_from_iq, phase_from_iq, windows, phase_windows, profile_windows,
                                FS_GRID)
from hallsense.rssifeat import rssi_windows
from hallsense.occupancy import OccupancyLatch

OUT = os.path.join(HERE, "..", "crates", "model", "tests", "golden")
ROOT = os.environ.get("ESPECTRE_DATA", os.path.expanduser("~/Projects/ext/espectre/data"))
INFO = json.load(open(os.path.join(ROOT, "dataset_info.json")))

# (name, label folder, collected_at prefix, environment or None, seconds kept from the start)
RECORDINGS = [
    ("c3_empty", "empty", "2026-08-10T19:31", "vacation_home", 75),
    ("c3_motion", "motion", "2026-08-10T19:25", "vacation_home", 30),
    ("c3_still", "static_presence", "2026-08-10T19:22", "vacation_home", 60),
    ("esp32_still", "static_presence", "2026-10-03T19:33", None, 30),
]


HEAD_ROWS = 60                                   # per-packet arrays: the first rows are enough, windows cover the rest


def save(name, arr):
    np.save(os.path.join(OUT, name + ".npy"), np.ascontiguousarray(arr))


def save_json(name, obj):
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, separators=(",", ":"))


def pick(lab, when, env):
    return next(m for m in INFO["files"][lab] if m["collected_at"].startswith(when)
                and (env is None or m.get("environment") == env))


def load_excerpt(lab, when, env, seconds):
    """same loading as bench/occupancy_engine_replay.py::stream, cut to the first `seconds`"""
    z = np.load(os.path.join(ROOT, lab, pick(lab, when, env)["filename"]))
    t = np.unwrap(z["wifi_rx_ts_us"].astype(np.float64), period=2 ** 32) / 1e6
    o = np.argsort(t, kind="stable")
    t = t[o] - t[o][0]
    keep = t < seconds
    return t[keep], z["csi_data"][o][keep], z["rssi_dbm"][o][keep].astype(float)


def export_features(rec):
    for name, (t, csi, rssi) in rec.items():
        iq = csi.reshape(len(csi), -1, 2)
        save(f"rec_{name}_t", t); save(f"rec_{name}_csi", csi.astype(np.int8)); save(f"rec_{name}_rssi", rssi)
        amp = amplitude_from_iq(iq)
        save(f"feat_{name}_amp_head", amp[:HEAD_ROWS])
        save(f"feat_{name}_grid_head", to_grid(t, amp)[:HEAD_ROWS])
        X, r = windows(t, amp, rssi)
        save(f"feat_{name}_windows", X); save(f"feat_{name}_rates", r)
        Xo, ro = windows(t, amp, rssi, win_s=min(20, t[-1] - t[0] - 0.5) - 0.37, hop_s=100)
        save(f"feat_{name}_windows_odd", Xo); save(f"feat_{name}_rates_odd", ro)
        Xn, _ = windows(t, amp, None)
        save(f"feat_{name}_windows_no_rssi", Xn)
        ph = phase_from_iq(iq)
        save(f"feat_{name}_phase_head", ph[:HEAD_ROWS])
        save(f"feat_{name}_phase_windows", phase_windows(t, ph))
        save(f"feat_{name}_profile_windows", profile_windows(t, amp))
        Xr, rr = rssi_windows(t, rssi)
        save(f"feat_{name}_rssi_windows", Xr); save(f"feat_{name}_rssi_rates", rr)
        two = np.c_[rssi, np.roll(rssi, 7) + 2.0]
        Xr2, rr2 = rssi_windows(t, two)
        save(f"feat_{name}_rssi2_windows", Xr2); save(f"feat_{name}_rssi2_rates", rr2)


def random_inputs(m, rng, n=300):
    """inputs that hit every kind of branch: exact thresholds, values next to them, NaN"""
    th = collections.defaultdict(list)
    for it in m._predictors:
        for p in it:
            for nd in p.nodes[~p.nodes["is_leaf"].astype(bool)]:
                th[int(nd["feature_idx"])].append(float(nd["num_threshold"]))
    X = np.zeros((n, m.n_features_in_))
    for j in range(m.n_features_in_):
        v = np.array(th.get(j, [0.0]))
        base = rng.choice(v, n)
        mode = rng.integers(0, 4, n)
        X[:, j] = np.where(mode == 0, base, np.where(mode == 1, np.nextafter(base, np.inf),
                                                      base + rng.normal(0, 0.1 + np.abs(base) * 0.2, n)))
        X[rng.random(n) < 0.08, j] = np.nan
    return X


def export_gbdt(rec):
    rng = np.random.default_rng(0)
    real17 = np.vstack([windows(t, amplitude_from_iq(c.reshape(len(c), -1, 2)), r)[0] for t, c, r in rec.values()])
    cases = {}
    for f in ("room_sim_all", "room_sim_no-rssi", "room_activity_esp1_random", "room_activity_esp3_smart",
              "laptop_rssi", "room_what_moved"):
        m = joblib.load(f"models/{f}.joblib")
        X = random_inputs(m, rng)
        if m.n_features_in_ == 17:
            X = np.vstack([real17, X])
        if m.n_features_in_ == 51:
            X = np.vstack([np.c_[real17, real17, real17], X])
        cases[f] = (m, X)
    hall = joblib.load("models/hall3d.joblib")
    for part in ("m1", "m2", "mc"):
        m = getattr(hall, part)
        cases[f"hall3d.{part}"] = (m, random_inputs(m, rng, 200))
    for name, (m, X) in cases.items():
        save(f"gbdt_{name}_X", X)
        save(f"gbdt_{name}_out", m.predict_proba(X) if hasattr(m, "predict_proba") else m.predict(X))


def export_latch():
    rng = np.random.default_rng(1)
    cases = []
    for case in range(3):
        latch = OccupancyLatch(thr=0.4) if case < 2 else OccupancyLatch()
        dim = 52
        base = rng.normal(0, 3, dim)
        steps = []
        for i in range(160):
            phase = (i // 20) % 4                            # empty, motion, still, empty ...
            drift = 0.02 * i
            x = np.round(base + drift + rng.normal(0, 0.3, dim) + (rng.normal(0, 1.5, dim) if phase == 2 else 0), 2)
            p_motion = 0.9 if phase == 1 and i % 7 else float(rng.uniform(0, 0.4))
            p_still = None if case == 0 else float(rng.uniform(0, 1))
            if case == 1 and i % 50 == 49:
                latch.ref.clear()
            st, d = latch.step(x, p_motion, p_still)
            steps.append(dict(x=x.tolist(), p_motion=p_motion, p_still=p_still, clear=bool(case == 1 and i % 50 == 49),
                              state=st, distance=d))
        cases.append(dict(thr=latch.thr, steps=steps))
    save_json("latch_cases", cases)


def export_room_engine(rec):
    import live
    published = []
    live.publish = lambda st: published.append(st)
    m1, m3 = live.models(types.SimpleNamespace(model=None))
    nodes = np.array([[0.2, 1.7, 1.0], [3.8, 1.7, 0.8]])
    sj = live.room_scene_json(4, 3.5, 2.8, nodes, [], ["router", "esp"])
    eng = live.RoomEngine(sj, 1, [(0, 1)], "REPLAY", m1, m3, nodes)
    t_off, ticks = 0.0, []
    for name in ("c3_empty", "c3_motion", "c3_still"):
        t, csi, rs = rec[name]
        iq = csi.reshape(len(csi), -1, 2)
        nxt = t_off + 1.0
        for i in range(len(t)):
            ti = t[i] + t_off
            while ti >= nxt:
                published.clear(); eng.tick(nxt)
                st = dict(published[-1]); st["history"] = st["history"][-1:]
                ticks.append(dict(now=nxt, segment=name, state=st)); nxt += 1.0
            eng.add(0, ti, iq[i], rs[i])
        t_off = t[-1] + t_off + 0.05
    save_json("room_engine_ticks", dict(segments=["c3_empty", "c3_motion", "c3_still"], gap_s=0.05, ticks=ticks))


THREE_LINK_PLAN = [("away", 25), ("walk", 15), ("sit", 40)]


def export_room_engine_three_links(seed=7, fs=25):
    """live.py demo-room with 3 boards (simulated, short and at 25 packets/s to stay small), ticked at sec + 1"""
    import live
    from hallsense.scenes3d import random_room
    from hallsense.sim3d import simulate_room
    published = []
    live.publish = lambda st: published.append(st)
    m1, m3 = live.models(types.SimpleNamespace(model=None))
    seconds = sum(d for _, d in THREE_LINK_PLAN)
    sc = random_room(np.random.default_rng(seed), n_esp=3, smart=True)
    out, _, _ = simulate_room(sc, np.random.default_rng([seed, 1]), seconds=seconds, fs=fs, plan=THREE_LINK_PLAN)
    sj = live.room_scene_json(sc.W, sc.D, sc.H, sc.nodes, [], ["router"] + ["esp"] * 3)
    eng = live.RoomEngine(sj, len(out), [tuple(l) for l in sc.links], "REPLAY", m1, m3, sc.nodes)
    ks, ticks = [0] * len(out), []
    for sec in range(seconds):
        for k, o in enumerate(out):
            while ks[k] < len(o["t"]) and o["t"][ks[k]] < sec + 1:
                eng.add(k, o["t"][ks[k]], o["iq"][ks[k]], o["rssi"][ks[k]]); ks[k] += 1
        published.clear(); eng.tick(sec + 1)
        st = dict(published[-1]); st["history"] = st["history"][-1:]
        ticks.append(dict(now=sec + 1, state=st))
    for k, o in enumerate(out):
        save(f"room3_link{k}_t", o["t"]); save(f"room3_link{k}_iq", o["iq"].reshape(len(o["t"]), -1))
        save(f"room3_link{k}_rssi", o["rssi"].astype(np.float64))
    save_json("room_engine_three_links", dict(W=sc.W, D=sc.D, H=sc.H, nodes=sc.nodes.tolist(), links=sc.links, ticks=ticks))


def export_hall():
    import live
    from hallsense.model import scene_from_arrays, geometry_cache
    z = np.load("cache/demo_hall_3.npz")
    W = 12
    cal_mean, cal_std = z["cal_mean"].astype(np.float32), z["cal_std"].astype(np.float32)
    ses_mean, ses_std = z["ses_mean"][:, 60:60 + W].astype(np.float32), z["ses_std"][:, 60:60 + W].astype(np.float32)
    ses_mean[3, 2] = np.nan; ses_std[3, 2] = np.nan                      # a dead link-second
    save("hall_cal_mean", cal_mean); save("hall_cal_std", cal_std)
    save("hall_ses_mean", ses_mean); save("hall_ses_std", ses_std)
    from hallsense.scenes3d import random_hall3
    sc = random_hall3(np.random.default_rng(3), n_nodes=12, tiered=True, placement="walls", W=13.0, D=12.0)
    hall = dict(W=sc.W, D=sc.D, H=sc.H, front=sc.tiers["front"], ids=list(range(len(sc.nodes))), nodes=sc.nodes,
                seats=sc.seats, chest=sc.seats + [0, 0, 1.0], links=[list(map(int, l)) for l in sc.links],
                tiers={k: float(v) for k, v in sc.tiers.items()})
    save_json("hall_layout", dict(W=float(hall["W"]), D=float(hall["D"]), H=float(hall["H"]), front=float(hall["front"]),
                                  nodes=hall["nodes"].tolist(), seats=hall["seats"].tolist(), links=hall["links"],
                                  tiers=hall["tiers"]))
    model = joblib.load("models/hall3d.joblib")
    sc_full = scene_from_arrays(cal_mean, cal_std, ses_mean, ses_std, hall["nodes"], [tuple(l) for l in hall["links"]],
                                hall["chest"], None, hall["W"], hall["D"])
    r = model.predict(sc_full)
    save("hall_seat_X", sc_full["X"]); save("hall_scene_vector", sc_full["sv"])
    save("hall_p1", r["p1"]); save("hall_p", r["p"])
    save_json("hall_counts", dict(count_sum=r["count_sum"], count_glob=r["count_glob"], count=r["count"],
                                  feature_names=sc_full["names"],
                                  link_features={k: [None if not np.isfinite(v) else float(v) for v in a]
                                                 for k, a in sc_full["f"].items()}))
    published = []
    live.publish = lambda st: published.append(st)
    hall["links"] = [tuple(l) for l in hall["links"]]
    eng = live.HallEngine(hall, model, cal_mean, cal_std, "REPLAY")
    states = []
    for w in range(W):
        eng.push(ses_mean[:, w], ses_std[:, w], truth=z["present"][60 + w])
        st = dict(published[-1]); st.pop("scene"); st.pop("history"); st.pop("t"); states.append(st)
    save("hall_truth", z["present"][60:60 + W].astype(np.int64))
    save_json("hall_engine_states", states)
    # Windower: mesh report lines -> 1 s windows
    ids = hall["ids"]; rng = np.random.default_rng(5); lines = []
    for _ in range(40):
        rx, src = rng.choice(ids, 2, replace=False)
        v = np.round(rng.normal(-50, 5, 26), 2)
        lines.append("R,%d,%d,%d," % (rx, src, 20) + ",".join(f"{x:.2f}" for x in v))
    lines += ["R,1,1,20," + ",".join(["1"] * 26), "garbage", "R,99,1,20," + ",".join(["1"] * 26)]
    win = live.Windower(hall)
    for l in lines:
        win.add_line(l)
    m, s = win.tick()
    save_json("hall_windower", dict(lines=lines))
    save("hall_windower_mean", m); save("hall_windower_std", s)


def export_beacon():
    import apsense
    rows = []
    for line in open("rec/ap_live.csv"):
        q = line.rstrip("\n").split(",")
        rows.append((float(q[0]), int(q[1]), q[2], q[3], [int(v) for v in q[4:]]))
        if rows[-1][0] - rows[0][0] > 330:
            break
    rows.pop()
    with open(os.path.join(OUT, "beacon_rows.csv"), "w") as f:
        for t, c, b, s, ch in rows:
            f.write(f"{t:.3f},{c},{b},{s}," + ",".join(map(str, ch)) + "\n")
    cfg = json.load(open("rec/myroom.json"))
    out = {}
    for mode in ("auto", "calibrated"):
        cal_path = os.path.join(OUT, "_beacon_cal_tmp.json")
        if os.path.exists(cal_path):
            os.remove(cal_path)
        eng = apsense.Engine(cfg, cal_path)
        clock = [rows[0][0]]
        apsense.time = types.SimpleNamespace(time=lambda: clock[0])
        ticks, nxt = [], rows[0][0] + 1
        plan = [(30, "empty", None, 1), (80, "desk", [1.0, 2.5], 2), (130, "empty", None, 3), (180, "desk", [1.0, 2.5], 4)]
        for t, c, b, s, ch in rows:
            while t >= nxt:
                clock[0] = nxt
                if mode == "calibrated":
                    for at, label, xy, _ in plan:
                        if abs(nxt - (rows[0][0] + at)) < 0.5:
                            eng.start_cal(label, xy, 15, 2)
                st = eng.tick(nxt)
                ticks.append(dict(now=nxt, room=st["room"], link_activity=st["link_activity"], alive=st["alive"],
                                  t=st["t"], last=dict(m=eng.last["m"], mot=eng.last["mot"], cv=eng.last["cv"],
                                                       pending=eng.last["pending"], spots=eng.last["spots"])))
                nxt += 1
            eng.add(t, c, b, ch)
        out[mode] = dict(ticks=ticks, calibration=eng.cal)
        if os.path.exists(cal_path):
            os.remove(cal_path)
    save_json("beacon_ticks", dict(config=cfg, **out))


def export_metrics():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400); s = np.round(rng.normal(y * 0.7, 1.0), 1)
    save_json("auc_case", dict(y=y.tolist(), score=s.tolist(), auc=float(roc_auc_score(y, s))))


def export_bench_arrays():
    """big arrays for the #[ignore] benchmark test: research/cache/, never into crates/"""
    sim = joblib.load("cache/room_sim_phase.joblib")
    X = np.vstack([r["X"][:, :17] for r in sim]); y = np.concatenate([[r["y"]] * len(r["X"]) for r in sim])
    np.save("cache/room_sim_amp_X.npy", X); np.save("cache/room_sim_amp_y.npy", y.astype(np.int64))
    print("cache/room_sim_amp_X.npy", X.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rec = {name: load_excerpt(lab, when, env, secs) for name, lab, when, env, secs in RECORDINGS}
    export_features(rec)
    export_gbdt(rec)
    export_latch()
    export_room_engine(rec)
    export_room_engine_three_links()
    export_hall()
    export_beacon()
    export_metrics()
    if a.bench:
        export_bench_arrays()
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"golden: {len(os.listdir(OUT))} files, {total / 1e6:.2f} MB in {os.path.relpath(OUT, HERE)}")
