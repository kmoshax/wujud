"""Evaluate the sim-trained model on a REAL recording (the weekend experiment).

labels.csv (one line per session, times from your phone/laptop clock in unix seconds,
or use `python real_eval.py mark` while recording to print them):
    start,end,seats
    1730200000,1730200070,
    1730200100,1730200170,3;4;12
    ...
seats = seat indices from hall.json (row-major order of the grid), empty = nobody.

python real_eval.py eval --rec rec.csv --labels labels.csv --hall hall.json --calib calib.npz
Prints per-session true vs estimated count + seat AUC, plus a leave-one-session-out
linear recalibration of the count (how much a few real sessions close the sim-to-real gap).
"""
import argparse, sys, time
import numpy as np
import joblib

from live import load_hall, Windower
from hallsense.model import scene_from_arrays, geometry_cache
from hallsense.metrics import auc, tolerant_f1


def windows_from_recording(path, hall):
    win = Windower(hall)
    out = []
    next_t = None
    for l in open(path):
        t, line = l.split(",", 1)
        t = float(t)
        if next_t is None:
            next_t = t + 1.0
        while t >= next_t:
            m, s = win.tick(); out.append((next_t, m, s)); next_t += 1.0
        win.add_line(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "mark"])
    ap.add_argument("--rec"); ap.add_argument("--labels"); ap.add_argument("--hall", default="hall.json")
    ap.add_argument("--calib", default="calib.npz"); ap.add_argument("--model", default="models/hallsense.joblib")
    a = ap.parse_args()
    if a.cmd == "mark":
        print("press Enter at the START and END of each session; type seat ids before the END Enter")
        while True:
            input("start> "); t0 = time.time()
            seats = input("seats (e.g. 3;4;12) then Enter to END> ").strip(); t1 = time.time()
            print(f"{t0:.0f},{t1:.0f},{seats}")
    hall = load_hall(a.hall); model = joblib.load(a.model); c = np.load(a.calib)
    geo = geometry_cache(hall["nodes"], hall["links"], hall["chest"])
    W = windows_from_recording(a.rec, hall)
    ts = np.array([w[0] for w in W])
    rows = []
    for l in open(a.labels).read().strip().splitlines()[1:]:
        st, en, seats = (l.split(",") + [""])[:3]
        st, en = float(st) + 10, float(en)          # skip 10 s of people settling
        sel = [k for k, t in enumerate(ts) if st <= t <= en][-60:]
        if len(sel) < 10:
            print("too few windows for session", l); continue
        sm = np.stack([W[k][1] for k in sel], 1); ss = np.stack([W[k][2] for k in sel], 1)
        sc = scene_from_arrays(c["cal_mean"], c["cal_std"], sm, ss, hall["nodes"], hall["links"], hall["chest"], geo, hall["W"], hall["D"])
        r = model.predict(sc)
        y = np.zeros(len(hall["seats"]), int)
        if seats.strip():
            y[[int(x) for x in seats.split(";")]] = 1
        rows.append((y.sum(), r["count"], auc(r["p"], y), tolerant_f1(r["p"] > 0.5, y.astype(bool), hall["seats"])))
        print(f"true {y.sum():3d}  est {r['count']:6.1f}  seatAUC {rows[-1][2]:.2f}  tolF1 {rows[-1][3]:.2f}")
    R = np.array(rows, float)
    print(f"\nraw count MAE: {np.mean(np.abs(R[:, 1] - R[:, 0])):.2f} people over {len(R)} sessions")
    if len(R) >= 4:
        pred = []
        for k in range(len(R)):
            m = np.arange(len(R)) != k
            A = np.c_[R[m, 1], np.ones(m.sum())]
            coef = np.linalg.lstsq(A, R[m, 0], rcond=None)[0]
            pred.append(max(0, coef[0] * R[k, 1] + coef[1]))
        print(f"count MAE after leave-one-out linear recalibration on real sessions: {np.mean(np.abs(np.array(pred) - R[:, 0])):.2f}")
    print(f"mean seat AUC: {np.nanmean(R[:, 2]):.2f}")


if __name__ == "__main__":
    main()
