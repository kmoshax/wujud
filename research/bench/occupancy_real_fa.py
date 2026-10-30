"""Real-data check of the occupancy latch's weak side: does a REAL empty room stay close to its own reference from
a few minutes earlier? Runs the latch over every ESPectre empty recording (no motion, no people) and counts how
often it would wrongly say 'someone is here', at several thresholds. Also: the same distance between a still-person
recording and the empty recording of the same board+room made minutes later (vacation_home, same evening).

python bench/occupancy_real_fa.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hallsense.roomfeat import amplitude_from_iq, profile_windows
from hallsense.occupancy import OccupancyLatch

ROOT = os.environ.get("ESPECTRE_DATA", os.path.expanduser("~/Projects/ext/espectre/data"))
THRS = (0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0)


def load(lab, m):
    z = np.load(os.path.join(ROOT, lab, m["filename"]))
    t = np.unwrap(z["wifi_rx_ts_us"].astype(np.float64), period=2 ** 32) / 1e6
    o = np.argsort(t)
    return profile_windows(t[o], amplitude_from_iq(z["csi_data"][o].reshape(len(o), -1, 2)))


if __name__ == "__main__":
    info = json.load(open(os.path.join(ROOT, "dataset_info.json")))
    emp = [(m, load("empty", m)) for m in info["files"]["empty"]]
    res = {"empty_recordings": len(emp), "false_present_rate": {}, "per_recording_minutes": [round(len(P) * 5 / 60, 1) for _, P in emp]}
    for thr in THRS:
        flags = []
        for _, P in emp:
            L = OccupancyLatch(thr=thr)
            st = [L.step(x, 0.0)[0] for x in P]
            flags += [s == "still" for s in st[L.min_ref:]]
        res["false_present_rate"][thr] = float(np.mean(flags))
        print(f"thr {thr:4.1f}: real empty room flagged as 'someone here' {np.mean(flags):6.1%} of {len(flags)} windows")
    # same evening, same board and room: still person at 19:22 -> empty at 19:31 (and 12:39 -> 12:48 next day)
    sta = {m["collected_at"][:13]: (m, load("static_presence", m)) for m in info["files"]["static_presence"] if m.get("environment") == "vacation_home"}
    pairs = []
    for m, P in emp:
        if m.get("environment") == "vacation_home" and m["collected_at"][:13] in sta:
            S = sta[m["collected_at"][:13]][1]
            L = OccupancyLatch(thr=1e9); [L.step(x, 0.0) for x in P[:12]]        # reference: first minute of the EMPTY room
            d_s = [L.distance(x) for x in S]; d_e = [L.distance(x) for x in P[12:]]
            pairs.append(dict(when=m["collected_at"], still_person_distance_median=float(np.median(d_s)),
                              empty_later_distance_median=float(np.median(d_e))))
            print(f"vacation_home {m['collected_at'][:16]}: distance to the empty reference: still person (9 min earlier) {np.median(d_s):.2f}  vs  same empty room {np.median(d_e):.2f}")
    res["vacation_home_pairs"] = pairs
    json.dump(res, open("results/occupancy_real_fa.json", "w"), indent=1)
