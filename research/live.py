"""HallSense live system (3-D dashboard at http://localhost:8000).

No hardware:
  python live.py demo-room                 # simulated bedroom: away -> walk -> sit -> lie (breathing) -> ...
  python live.py demo-hall                 # simulated tiered lecture hall filling up
Room (router + 1..N ESP32 running firmware/hallsense_room, one USB port per board):
  python live.py room --room room.json --ports COM5 [COM6 COM7]
Hall (ESP32 mesh running firmware/hallsense_node, one board on USB as sink):
  python live.py calibrate --hall hall.json --port COM5      # 30 s, hall EMPTY
  python live.py hall --hall hall.json --port COM5 --record rec.csv
  python live.py replay --hall hall.json --file rec.csv
"""
import argparse, collections, json, os, sys, threading, time
# one row per second: OpenMP thread start-up in sklearn's predict costs more than the prediction, and on a busy
# laptop it made one room tick take ~20 s instead of 0.1 s (the dashboard froze). Must be set before sklearn loads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import joblib

from hallsense.config import N_GROUPS
from hallsense.model import scene_from_arrays, geometry_cache
from hallsense.roomfeat import windows, amplitude_from_iq, profile_windows
from hallsense.occupancy import OccupancyLatch
from hallsense.roomloc import locate

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = {"mode": "starting"}
LOCK = threading.Lock()
ACT = ["away", "lie", "sit", "stand", "walk"]


def publish(st):
    with LOCK:
        STATE.clear(); STATE.update(st)


# ---------------------------------------------------------------------------
# hall
# ---------------------------------------------------------------------------
def load_hall(path):
    h = json.load(open(path))
    ids = sorted(int(k) for k in h["nodes"])
    nodes = np.array([(list(h["nodes"][str(i)]) + [1.1])[:3] for i in ids], float)
    rise = 0.0
    if "seats" in h:
        seats = np.array([(list(s) + [0.0])[:3] for s in h["seats"]], float)
    else:
        g = h["grid"]; rise = g.get("row_rise", 0.0)
        seats = []
        for r in range(g["rows"]):
            for c in range(g["cols"]):
                x = g["x0"] + c * g["seat_pitch"] + sum(g.get("aisle_w", 1.0) for a in g.get("aisles_after_col", []) if c > a)
                seats.append((x, g["y0"] + r * g["row_pitch"], r * rise))
        seats = np.array(seats)
    links = [(a, b) for a in range(len(ids)) for b in range(a + 1, len(ids))]
    tiers = None
    if "grid" in h:
        g = h["grid"]; tiers = dict(y0=g["y0"], rise=rise, pitch=g["row_pitch"], front=h.get("front", 0))
    return dict(W=h["W"], D=h["D"], H=h.get("H", 3.5 + seats[:, 2].max()), front=h.get("front", 0), ids=ids, nodes=nodes,
                seats=seats, chest=seats + [0, 0, 1.0], links=links, tiers=tiers)


class Windower:
    """turns mesh report lines (R,rx,src,n,13 means,13 stds) into 1-s link windows [L,13]"""
    def __init__(self, hall):
        self.hall = hall; self.idx = {nid: k for k, nid in enumerate(hall["ids"])}
        self.lix = {l: n for n, l in enumerate(hall["links"])}; self.reset()

    def reset(self):
        L = len(self.hall["links"])
        self.sm = np.zeros((L, N_GROUPS)); self.ss = np.zeros((L, N_GROUPS)); self.n = np.zeros(L)

    def add_line(self, line):
        p = line.strip().split(",")
        if len(p) != 4 + 2 * N_GROUPS or p[0] != "R":
            return
        rx, src = int(p[1]), int(p[2])
        if rx not in self.idx or src not in self.idx or rx == src:
            return
        a, b = sorted((self.idx[rx], self.idx[src]))
        l = self.lix[(a, b)]; v = np.array(p[4:], float)
        self.sm[l] += v[:N_GROUPS]; self.ss[l] += v[N_GROUPS:]; self.n[l] += 1

    def tick(self):
        with np.errstate(all="ignore"):
            m = self.sm / self.n[:, None]; s = self.ss / self.n[:, None]
        m[self.n == 0] = np.nan; s[self.n == 0] = np.nan
        self.reset()
        return m.astype(np.float32), s.astype(np.float32)


def hall_scene_json(h):
    return dict(kind="hall", W=h["W"], D=h["D"], H=h["H"], tiers=h["tiers"], seats=np.round(h["seats"], 3).tolist(),
                nodes=np.round(h["nodes"], 3).tolist(), links=h["links"], furniture=[])


class HallEngine:
    def __init__(self, hall, model, cal_mean, cal_std, mode, max_w=60):
        self.h, self.model, self.mode = hall, model, mode
        self.cm, self.cs = cal_mean, cal_std
        self.geo = geometry_cache(hall["nodes"], hall["links"], hall["chest"])
        self.bm = collections.deque(maxlen=max_w); self.bs = collections.deque(maxlen=max_w)
        self.hist = collections.deque(maxlen=900); self.t0 = time.time(); self.ema = None
        self.sj = hall_scene_json(hall)

    def push(self, m, s, truth=None):
        self.bm.append(m); self.bs.append(s)
        h = {}
        act = np.zeros(len(m))
        if len(self.bm) >= 10:
            sc = scene_from_arrays(self.cm, self.cs, np.stack(self.bm, 1), np.stack(self.bs, 1), self.h["nodes"],
                                   self.h["links"], self.h["chest"], self.geo, self.h["W"], self.h["D"])
            r = self.model.predict(sc)
            self.ema = r["count"] if self.ema is None else 0.7 * self.ema + 0.3 * r["count"]
            h = dict(count=round(self.ema, 1), p=np.round(r["p"], 3).tolist())
            act = np.clip(np.nan_to_num(-sc["f"]["sh_mean"]) / 6 + np.nan_to_num(sc["f"]["mo_mean"]) / 1.5, 0, 1)
        if truth is not None:
            h["truth"] = truth.astype(int).tolist(); h["true_count"] = int(truth.sum())
        t = round(time.time() - self.t0, 1)
        self.hist.append([t, h.get("count"), h.get("true_count")])
        publish(dict(mode=self.mode, t=t, scene=self.sj, hall=h, link_activity=np.round(act, 2).tolist(),
                     alive=int(np.isfinite(m[:, 0]).sum()), history=list(self.hist)))


# ---------------------------------------------------------------------------
# room
# ---------------------------------------------------------------------------
class RoomEngine:
    """per-packet CSI from router -> board links; every second: activity, breathing, position"""
    def __init__(self, scene_json, n_links, link_nodes, mode, model1, model3, nodes):
        self.sj, self.mode, self.L = scene_json, mode, n_links
        self.links, self.nodes = link_nodes, nodes
        self.model = model1 if n_links < 3 else model3
        self.buf = [collections.deque() for _ in range(n_links)]
        self.hist = collections.deque(maxlen=900); self.t0 = None; self.last = None
        self.smooth = None
        self.locate = False
        # still person vs empty: compare the static multipath profile with the empty room of a few MINUTES ago
        # (day-to-day drift is bigger than a still person, minute-to-minute drift is not; see hallsense/occupancy.py)
        self.latch = OccupancyLatch(thr=0.4); self.latch_state = None; self.latch_d = None; self.prof_dim = None
        self.ticks = 0

    def add(self, k, t, iq, rssi):
        b = self.buf[k]; b.append((t, iq, rssi))
        while b and b[0][0] < t - 32:
            b.popleft()

    def tick(self, now, truth=None):
        """3-state decision every second; the detailed pose only when confident; breathing only
        after 25 s of stillness, from a 30-s window, and only if the breathing GLRT clears the bar."""
        self.t0 = now if self.t0 is None else self.t0
        per, Pb, rb, prof = [], [], [], []
        for b in self.buf:
            if len(b) < 50 or b[-1][0] - b[0][0] < 18:
                continue
            t = np.array([x[0] for x in b]); iq = np.stack([x[1] for x in b]); rs = np.array([x[2] for x in b], float)
            amp = amplitude_from_iq(iq)
            X, _ = windows(t - t[0], amp, rs, win_s=min(20, t[-1] - t[0] - 0.5), hop_s=100)
            if len(X):
                per.append(X[-1])
            if t[-1] - t[0] >= 20.5:
                pw = profile_windows(t - t[0], amp, win_s=20, hop_s=100)
                if len(pw):
                    prof.append(pw[-1])
            if t[-1] - t[0] >= 29:
                Xb, r = windows(t - t[0], amp, rs, win_s=min(30, t[-1] - t[0] - 0.5), hop_s=100)
                if len(Xb):
                    Pb.append(Xb[-1, 16]); rb.append(r[-1])
        room = {}
        act = [0.0] * self.L
        if per:
            P = np.stack(per)
            xa = np.concatenate([P.mean(0), P.max(0), P.min(0)])[None]
            pr = self.model.predict_proba(xa)[0]
            self.smooth = pr if self.smooth is None else 0.6 * self.smooth + 0.4 * pr
            p3 = np.array([self.smooth[0], self.smooth[1:4].sum(), self.smooth[4]])
            s3 = ["empty", "still", "moving"][int(np.argmax(p3))]
            # physics override: a clear breathing line for 10 s in a row means someone is there even if
            # the classifier says empty (GLRT > 1.8 fires on ~8% of real empty-room 20-s windows in the ESPectre data,
            # so we also require 10 s in a row and 25 s without motion)
            self.br_run = (getattr(self, "br_run", 0) + 1) if (Pb and max(Pb) > 1.8) else 0
            self.calm_for = (getattr(self, "calm_for", 0) + 1) if s3 != "moving" else 0
            if s3 == "empty" and self.br_run >= 10 and self.calm_for >= 25:
                s3 = "still"
            # occupancy latch, every 5 s (its reference and timings are in 5-s windows)
            self.ticks += 1
            if prof and len(prof) == len(per):
                v = np.concatenate(prof)
                if v.shape != self.prof_dim:                 # a link came or went: start a new reference
                    self.latch = OccupancyLatch(thr=0.4); self.prof_dim = v.shape
                if self.ticks % 5 == 0:
                    self.latch_state, self.latch_d = self.latch.step(v, p3[2], p3[1] / max(p3[0] + p3[1], 1e-9))
            if s3 != "moving" and self.latch_d is not None:   # fresh empty reference: the latch decides
                s3 = self.latch_state if self.latch_state in ("still", "empty") else s3
            self.still_for = (getattr(self, "still_for", 0) + 1) if s3 == "still" else 0
            state = s3
            if s3 == "still":
                k = 1 + int(np.argmax(self.smooth[1:4]))
                if self.smooth[k] / max(p3[1], 1e-9) >= 0.55:
                    state = ACT[k]
            room = dict(state=state, state3=s3, probs=np.round(self.smooth, 3).tolist(),
                        latch=dict(distance=None if self.latch_d is None else round(self.latch_d, 2), thr=self.latch.thr,
                                   reference_windows=len(self.latch.ref)))
            if s3 == "still" and self.calm_for >= 25 and Pb:
                j = int(np.argmax(Pb))
                if Pb[j] > 1.4 and 6 <= rb[j] <= 36:
                    room["breath_bpm"] = round(float(rb[j]), 1)
            # position: tested and NOT reliable without an empty-room calibration (1.25 m vs 1.14 m for
            # guessing the centre, see README), so it is only shown when explicitly enabled
            if self.locate and len(per) >= 3 and s3 != "empty":
                room["xy"] = locate(P, self.links, self.nodes, self.sj["W"], self.sj["D"]).round(2).tolist()
            act = np.clip(np.maximum(P[:, 3], P[:, 1]) / 1.5, 0, 1).round(2).tolist()
        if truth:
            room.update(truth)
        t = round(now - self.t0, 1)
        self.hist.append([t, room.get("breath_bpm"), room.get("truth_bpm")])
        publish(dict(mode=self.mode, t=t, scene=self.sj, room=room, link_activity=act, alive=len(per), history=list(self.hist)))


def room_scene_json(W, D, H, nodes, furniture, roles):
    return dict(kind="room", W=W, D=D, H=H, tiers=None, seats=[], nodes=np.round(nodes, 3).tolist(), node_roles=roles,
                links=[[0, k] for k in range(1, len(nodes))], furniture=furniture)


# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p.startswith("/state"):
            with LOCK:
                body = json.dumps(STATE).encode()
            ctype = "application/json"
        elif p.startswith("/vendor/") and ".." not in p:
            body = open(os.path.join(HERE, "realtime", p.lstrip("/")), "rb").read(); ctype = "text/javascript"
        else:
            body = open(os.path.join(HERE, "realtime", "dashboard3d.html"), "rb").read(); ctype = "text/html; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)


def serve(port):
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"dashboard: http://localhost:{port}", flush=True)


def serial_lines(port, baud=921600, record=None):
    import serial
    ser = serial.Serial(port, baud, timeout=0.2)
    rec = open(record, "a") if record else None
    while True:
        try:
            line = ser.readline().decode(errors="ignore")
        except (serial.SerialException, OSError) as e:      # board unplugged / reset: keep retrying, don't kill the link
            print(f"{port}: {e} - reconnecting", flush=True); ser.close(); time.sleep(1.0)
            try:
                ser = serial.Serial(port, baud, timeout=0.2)
            except (serial.SerialException, OSError):
                pass
            yield None; continue
        if line and rec:
            rec.write(f"{time.time():.3f},{port},{line.strip()}\n"); rec.flush()
        yield line or None


def hall_stream(lines, hall, on_window, ts_lines=False, speed=0):
    win = Windower(hall); next_t = None
    for item in lines:
        if ts_lines:
            t, line = item
        else:
            t, line = time.time(), item
        if next_t is None:
            next_t = t + 1.0
        while t >= next_t:
            on_window(*win.tick()); next_t += 1.0
            if speed:
                time.sleep(1.0 / speed)
        if line:
            if line[:2] in ("N,", "I,"):
                print(line.strip())
            win.add_line(line)


def parse_p(line):
    p = line.strip().split(",")
    if len(p) != 4 or p[0] != "P" or len(p[3]) < 256:
        return None
    raw = np.frombuffer(bytes.fromhex(p[3][:256]), dtype=np.int8).reshape(64, 2)
    return int(p[1]), int(p[2]), raw


def models(a):
    m1 = joblib.load(os.path.join(HERE, "models", "room_activity_esp1_random.joblib"))
    m3 = joblib.load(os.path.join(HERE, "models", "room_activity_esp3_random.joblib"))
    return m1, m3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["demo-room", "demo-hall", "room", "hall", "calibrate", "replay"])
    ap.add_argument("--port"); ap.add_argument("--ports", nargs="*"); ap.add_argument("--hall", default="hall.json")
    ap.add_argument("--room", default="room.json"); ap.add_argument("--calib", default="calib.npz")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "hall3d.joblib"))
    ap.add_argument("--seconds", type=int, default=30); ap.add_argument("--record"); ap.add_argument("--file")
    ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--http", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--boards", type=int, default=3); ap.add_argument("--hall-seed", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)

    if a.cmd == "demo-room":
        from hallsense.scenes3d import random_room
        from hallsense.sim3d import simulate_room
        sc = random_room(np.random.default_rng(a.seed), n_esp=a.boards, smart=True)
        plan = [("away", 25), ("walk", 15), ("sit", 50), ("walk", 10), ("lie", 80), ("walk", 12), ("away", 25)]
        cache = os.path.join(HERE, "cache", f"demo_room_{a.seed}_{a.boards}.joblib")
        if not os.path.exists(cache):
            print("simulating the room (one-time)...", flush=True)
            joblib.dump(simulate_room(sc, np.random.default_rng(a.seed), seconds=sum(d for _, d in plan), fs=50, plan=plan), cache)
        out, lab, B = joblib.load(cache)
        sj = room_scene_json(sc.W, sc.D, sc.H, sc.nodes, sc.furniture, ["router"] + ["esp"] * (len(sc.nodes) - 1))
        m1, m3 = models(a)
        eng = RoomEngine(sj, len(out), [tuple(l) for l in sc.links], "SIMULATION", m1, m3, sc.nodes)
        serve(a.http)
        fs = lab["fs"]
        while True:
            eng.buf = [collections.deque() for _ in out]; eng.hist.clear(); eng.t0 = None; eng.smooth = None; eng.still_for = 0; eng.br_run = 0; eng.calm_for = 0
            ks = [0] * len(out)
            for sec in range(int(out[0]["t"][-1])):
                for li, o in enumerate(out):
                    while ks[li] < len(o["t"]) and o["t"][ks[li]] < sec + 1:
                        eng.add(li, o["t"][ks[li]], o["iq"][ks[li]], o["rssi"][ks[li]]); ks[li] += 1
                f = min(len(lab["activity"]) - 1, int((sec + 0.5) * fs))
                st = ACT[lab["activity"][f]]
                xy = lab["xy"][f]
                eng.tick(sec + 1, dict(truth=st, truth_bpm=lab["breath_hz"] * 60 if st in ("lie", "sit", "stand") else None,
                                       truth_xy=None if not np.isfinite(xy[0]) else np.round(xy, 2).tolist()))
                time.sleep(1.0 / a.speed)

    if a.cmd == "demo-hall":
        from hallsense.scenes3d import random_hall3
        from hallsense.sim3d import simulate_hall_timeline3
        sc = random_hall3(np.random.default_rng(a.hall_seed), n_nodes=12, tiered=True, placement="walls", W=13.0, D=12.0)
        cache = os.path.join(HERE, "cache", f"demo_hall_{a.hall_seed}.npz")
        if not os.path.exists(cache):
            print("simulating a 4-minute lecture in a tiered hall (one-time, a few minutes)...", flush=True)
            np.savez_compressed(cache, **simulate_hall_timeline3(sc, np.random.default_rng(a.hall_seed), minutes=4.0))
        z = np.load(cache)
        hall = dict(W=sc.W, D=sc.D, H=sc.H, front=sc.tiers["front"], ids=list(range(len(sc.nodes))), nodes=sc.nodes,
                    seats=sc.seats, chest=sc.seats + [0, 0, 1.0], links=sc.links, tiers=sc.tiers)
        eng = HallEngine(hall, joblib.load(a.model), z["cal_mean"], z["cal_std"], "SIMULATION")
        serve(a.http)
        while True:
            for w in range(z["ses_mean"].shape[1]):
                eng.push(z["ses_mean"][:, w], z["ses_std"][:, w], truth=z["present"][w]); time.sleep(1.0 / a.speed)
            eng.bm.clear(); eng.bs.clear(); eng.ema = None; eng.hist.clear(); eng.t0 = time.time()

    if a.cmd == "room":
        r = json.load(open(a.room))
        ports = a.ports or [a.port]
        nodes = np.array([r["router"]] + [r["boards"][p] for p in ports], float)
        sj = room_scene_json(r["W"], r["D"], r.get("H", 2.8), nodes, r.get("furniture", []), ["router"] + ["esp"] * len(ports))
        m1, m3 = models(a)
        eng = RoomEngine(sj, len(ports), [(0, k + 1) for k in range(len(ports))], "LIVE", m1, m3, nodes)
        serve(a.http)
        lock = threading.Lock()

        def reader(k, port):
            t_prev, wraps = None, 0
            for line in serial_lines(port, record=a.record):
                if not line:
                    continue
                if line.startswith("I,"):
                    print(port, line.strip()); continue
                p = parse_p(line)
                if p is None:
                    continue
                rssi, ts, raw = p
                if t_prev is not None and ts < t_prev:
                    wraps += 1
                t_prev = ts
                with lock:
                    eng.add(k, (ts + wraps * 2 ** 32) / 1e6, raw, rssi)
        for k, p in enumerate(ports):
            threading.Thread(target=reader, args=(k, p), daemon=True).start()
        while True:
            time.sleep(1.0)
            with lock:
                latest = max((b[-1][0] for b in eng.buf if b), default=0)
                eng.tick(latest)

    hall = load_hall(a.hall)
    if a.cmd == "calibrate":
        ms, ss = [], []

        def cb(m, s):
            ms.append(m); ss.append(s)
            print(f"\rcalibration {len(ms)}/{a.seconds}  links alive {np.isfinite(m[:, 0]).sum()}/{len(m)}", end="", flush=True)
            if len(ms) >= a.seconds:
                np.savez(a.calib, cal_mean=np.stack(ms, 1), cal_std=np.stack(ss, 1)); print(f"\nsaved {a.calib}"); sys.exit(0)
        hall_stream(serial_lines(a.port, record=a.record), hall, cb)

    c = np.load(a.calib)
    eng = HallEngine(hall, joblib.load(a.model), c["cal_mean"], c["cal_std"], "LIVE" if a.cmd == "hall" else "REPLAY")
    serve(a.http)
    if a.cmd == "hall":
        hall_stream(serial_lines(a.port, record=a.record), hall, eng.push)
    else:
        def rows():
            for l in open(a.file):
                parts = l.split(",", 2)
                yield float(parts[0]), parts[2] if not parts[1].startswith("R") else l.split(",", 1)[1]
        hall_stream(rows(), hall, eng.push, ts_lines=True, speed=a.speed)
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
