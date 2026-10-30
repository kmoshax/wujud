"""End-to-end: real ESP32 recordings streamed through live.py's RoomEngine (the code the dashboard runs), as one
continuous stream, same board and room, same session (vacation_home):
  realistic : [empty room] -> [person walking] -> [still person]
  no entry  : [empty room] -> [still person]   (impossible in reality: nobody walked in; the latch must stay cautious)
Prints the state the engine publishes every 10 s.

python bench/occupancy_engine_replay.py
"""
import json, os, sys, types
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import live

ROOT = os.environ.get("ESPECTRE_DATA", os.path.expanduser("~/Projects/ext/espectre/data"))


def stream(lab, m):
    z = np.load(os.path.join(ROOT, lab, m["filename"]))
    t = np.unwrap(z["wifi_rx_ts_us"].astype(np.float64), period=2 ** 32) / 1e6; o = np.argsort(t)
    return t[o] - t[o][0], z["csi_data"][o].reshape(len(o), -1, 2), z["rssi_dbm"][o].astype(float)


if __name__ == "__main__":
    info = json.load(open(os.path.join(ROOT, "dataset_info.json")))
    pick = lambda lab, when: next(m for m in info["files"][lab] if m.get("environment") == "vacation_home" and m["collected_at"].startswith(when))
    published = []
    live.publish = lambda st: published.append(st)
    m1, m3 = live.models(types.SimpleNamespace(**{k: None for k in ("model", "model1", "model3")})) if hasattr(live, "models") else (None, None)
    out = {}
    sessions = {"day1": ("2026-08-10T19:31", "2026-08-10T19:25", "2026-08-10T19:22"), "day2": ("2026-08-11T12:48", "2026-08-11T12:42", "2026-08-11T12:39")}
    for (day, (e_when, m_when, s_when)), kind in [(d, k) for d in sessions.items() for k in ("realistic", "no entry")]:
        sj = live.room_scene_json(4, 3.5, 2.8, np.array([[0.2, 1.7, 1.0], [3.8, 1.7, 0.8]]), [], ["router", "esp"])
        eng = live.RoomEngine(sj, 1, [(0, 1)], "REPLAY", m1, m3, np.array([[0.2, 1.7, 1.0], [3.8, 1.7, 0.8]]))
        if os.environ.get("NO_LATCH"):                 # baseline: the engine exactly as before the latch
            eng.latch.step = lambda *a: ("unknown", None)
        segs = [("empty", stream("empty", pick("empty", e_when)))]
        if kind == "realistic":
            segs.append(("walk", stream("motion", pick("motion", m_when))))
        segs.append(("still", stream("static_presence", pick("static_presence", s_when))))
        t_off, rows = 0.0, []
        for truth, (t, iq, rs) in segs:
            nxt = t_off + 1.0
            for i in range(len(t)):
                ti = t[i] + t_off
                while ti >= nxt:
                    published.clear(); eng.tick(nxt); r = published[-1]["room"]
                    rows.append((round(nxt), truth, r.get("state3"), (r.get("latch") or {}).get("distance"))); nxt += 1.0
                eng.add(0, ti, iq[i], rs[i])
            t_off = t[-1] + t_off + 0.05
        out[f"{day} {kind}"] = rows
        print(f"\n== {day}, {kind}: " + " -> ".join(s[0] for s in segs))
        for truth in ("empty", "walk", "still"):
            st = [r[2] for r in rows if r[1] == truth and r[0] > 25]
            if st: print(f"  {truth:5s}: engine said {dict(zip(*np.unique(st, return_counts=True)))}")
    json.dump(out, open("results/occupancy_engine_replay" + ("_no_latch" if os.environ.get("NO_LATCH") else "") + ".json", "w"))
