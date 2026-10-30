"""AP-sense: see a person in your room with ZERO extra hardware and WITHOUT joining any network.

The laptop's Wi-Fi card listens in monitor mode to the beacons that every access point broadcasts
~10x a second at a fixed power and rate. Each (AP, laptop antenna) pair is a radio link through the room;
a body changes the multipath on every link a little differently. Two APs (home router + D-Link) on
5 GHz, two antennas each -> 4+ links, hopped between channels.

    sudo .venv/bin/python apsense.py run --chans 52,64 --record rec/ap.csv      # live + record
    .venv/bin/python apsense.py replay rec/ap.csv                                # offline
Dashboard: http://localhost:8000   calibration: http://localhost:8000/cal
"""
import argparse, collections, json, os, socket, struct, subprocess, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from laptop_sense import radiotap_signals


def beacon_ssid(p, L):
    i = L + 24 + 12
    while i + 2 <= len(p):
        tag, n = p[i], p[i + 1]
        if tag == 0:
            return p[i + 2:i + 2 + n].decode(errors="replace")
        i += 2 + n
    return ""


def capture(mon, chans, dwell, out):
    """yield (t, chan, bssid, ssid, [dBm per chain]) for every beacon, hopping over chans"""
    cur = [chans[0]]
    if len(chans) > 1:
        def hop():
            i = 0
            while True:
                c = chans[i % len(chans)]
                subprocess.run(["iw", "dev", mon, "set", "channel", str(c)], capture_output=True)
                cur[0] = c; time.sleep(dwell); i += 1
        threading.Thread(target=hop, daemon=True).start()
    else:
        subprocess.run(["iw", "dev", mon, "set", "channel", str(chans[0])], capture_output=True)
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(3)); s.bind((mon, 0))
    while True:
        p = s.recv(4096)
        sig, L = radiotap_signals(p)
        if not sig or len(p) < L + 40 or p[L] != 0x80:
            continue
        ch = sig[1:] if len(sig) > 1 else sig
        if any(v > -5 or v < -95 for v in ch):        # rtw89 reports junk on 2.4 GHz; drop it
            continue
        out((time.time(), cur[0], p[L + 10:L + 16].hex(":"), beacon_ssid(p, L), ch))


def read_rec(path):
    for line in open(path):
        q = line.rstrip("\n").split(",")
        yield float(q[0]), int(q[1]), q[2], q[3], [int(v) for v in q[4:]]



# --------------------------------------------------------------------------- engine
def bursts(t):
    """split a sorted time vector into dwell bursts (the card hops away for ~1 s between them)"""
    cut = np.where(np.diff(t) > 0.4)[0] + 1
    return np.split(np.arange(len(t)), cut)


class Engine:
    """Per second: which links are alive, how much they wobble (motion), and where the mean levels sit
    compared with what each calibration spot looked like (empty room = just another spot)."""

    def __init__(self, cfg, cal_path, win=4.0):
        self.cfg, self.cal_path, self.win = cfg, cal_path, win
        self.buf = collections.deque()
        self.ap_of = {b.lower(): i for i, ap in enumerate(cfg["aps"]) for b in ap["bssids"]}
        # one link per (BSSID, laptop antenna): two BSSIDs of one radio sit ~0.5 dB apart, mixing them fakes wobble
        self.keys = [(i, b.lower(), c) for i, ap in enumerate(cfg["aps"]) for b in ap["bssids"] for c in (0, 1)]
        self.cal = json.load(open(cal_path)) if os.path.exists(cal_path) else {"windows": []}
        self.pending = None; self.lock = threading.Lock(); self.fit()
        self.smooth = None; self.hist = collections.deque(maxlen=900); self.t0 = None; self.last = {}
        self.mhist = collections.deque(maxlen=3 * 3600)   # per-second per-link wobble, for the self-calibration

    # ---- input
    def add(self, t, chan, bssid, ch):
        i = self.ap_of.get(bssid)                      # the channel tag is unreliable around a hop; the MAC isn't
        if i is None:
            return
        with self.lock:
            for c, v in enumerate(ch[:2]):
                self.buf.append((t, (i, bssid, c), float(v)))
            while self.buf and self.buf[0][0] < t - 60:
                self.buf.popleft()

    # ---- features
    def features(self, now, win=None):
        win = win or self.win
        with self.lock:
            rows = [r for r in self.buf if r[0] >= now - win]
        m, mot, alive = [], [], []
        for k in self.keys:
            tt = np.array([r[0] for r in rows if r[1] == k]); vv = np.array([r[2] for r in rows if r[1] == k])
            if len(vv) < 4:
                m.append(np.nan); mot.append(np.nan); alive.append(0); continue
            o = np.argsort(tt); tt, vv = tt[o], vv[o]
            B = [vv[b] for b in bursts(tt) if len(b)]
            means = np.array([b.mean() for b in B])
            intra = np.sqrt(np.mean(np.concatenate([b - b.mean() for b in B]) ** 2))
            inter = np.abs(np.diff(means)).mean() if len(means) > 1 else 0.0
            m.append(vv.mean()); mot.append(intra + 0.5 * inter); alive.append(1)
        return np.array(m), np.array(mot), np.array(alive)

    # ---- calibration
    def start_cal(self, label, xy, secs, delay):
        self.pending = dict(label=label, xy=xy, t_start=time.time() + delay, t_end=time.time() + delay + secs, n=0, sess=round(time.time()))

    def cal_tick(self, now):
        p = self.pending
        if not p or now < p["t_start"]:
            return
        if now - p["t_start"] >= self.win:             # whole window inside the recording period
            m, mot, alive = self.features(now)
            if alive.all():
                self.cal["windows"].append(dict(label=p["label"], xy=p["xy"], t=now, sess=p["sess"], m=m.tolist(), mot=mot.tolist()))
                p["n"] += 1
        if now >= p["t_end"]:
            self.pending = None; self.save(); self.fit()

    def save(self):
        json.dump(self.cal, open(self.cal_path, "w"))

    def fit(self):
        """per-spot mean level vector, pooled per-link spread; motion threshold from the empty room"""
        W = self.cal["windows"]; self.model = None; self.cv = None
        labels = sorted({w["label"] for w in W})
        if not W:
            self.thr = 1.0; return
        e = np.array([w["mot"] for w in W if w["label"] == "empty"])
        self.thr = float(max(0.6, 1.8 * np.percentile(np.median(e, 1), 99))) if len(e) > 5 else 1.0
        if len(labels) < 2:
            return
        self.model = self._fit(W, labels); self.cv = self._cv(W, labels)

    def _fit(self, W, labels):
        mu, xy, res = [], [], []
        for L in labels:
            M = np.array([w["m"] for w in W if w["label"] == L])
            mu.append(M.mean(0)); res.append(M - M.mean(0))
            xy.append(next(w["xy"] for w in W if w["label"] == L))
        sd = np.maximum(0.5, np.concatenate(res).std(0))
        return dict(labels=labels, mu=np.array(mu), sd=sd, xy=xy)

    def _post(self, model, m):
        z = (m[None] - model["mu"]) / model["sd"]
        ll = -0.5 * np.nansum(z ** 2, 1) / max(1, np.isfinite(m).sum()) * 4      # tempered: links are correlated
        p = np.exp(ll - ll.max()); return p / p.sum()

    def _cv(self, W, labels):
        """honest estimate. RSSI drifts over minutes, so windows recorded back-to-back look alike whatever the
        position: a split inside one recording would learn the clock, not the place. If every spot was recorded in
        two or more rounds, train on the earlier rounds and test on each spot's LAST round; otherwise first 2/3 vs
        last 1/3 in time, flagged as optimistic."""
        tr, te = [], []
        sess = {L: sorted({w.get("sess", 0) for w in W if w["label"] == L}) for L in labels}
        rounds = all(len(v) >= 2 for v in sess.values())
        for L in labels:
            ws = sorted([w for w in W if w["label"] == L], key=lambda w: w["t"])
            if rounds:
                tr += [w for w in ws if w.get("sess", 0) != sess[L][-1]]; te += [w for w in ws if w.get("sess", 0) == sess[L][-1]]
            else:
                k = int(len(ws) * 2 / 3); tr += ws[:k]; te += ws[k:]
        if not te or len({w["label"] for w in tr}) < len(labels):
            return None
        mdl = self._fit(tr, labels); ok, err = 0, []
        for w in te:
            p = self._post(mdl, np.array(w["m"])); ok += mdl["labels"][int(np.argmax(p))] == w["label"]
            if w["xy"] is not None:
                sp = [i for i, xy in enumerate(mdl["xy"]) if xy is not None]
                q = p[sp] / p[sp].sum(); xy = (q[:, None] * np.array([mdl["xy"][i] for i in sp])).sum(0)
                err.append(float(np.hypot(*(xy - np.array(w["xy"])))))
        return dict(acc=ok / len(te), n=len(te), loc_err_m=float(np.median(err)) if err else None, rounds=rounds)

    # ---- every second
    def tick(self, now):
        self.cal_tick(now)
        m, mot, alive = self.features(now)
        room = {}
        if alive.any():
            mv = float(np.nanmedian(mot))
            moving = mv > self.thr
            room.update(motion=round(mv, 2), thr=round(self.thr, 2))
            if self.model is not None and alive.all():
                p = self._post(self.model, m)
                self.smooth = p if self.smooth is None or len(self.smooth) != len(p) else 0.7 * self.smooth + 0.3 * p
                p = self.smooth; labels = self.model["labels"]
                room["probs"] = {L: round(float(v), 3) for L, v in zip(labels, p)}
                pe = p[labels.index("empty")] if "empty" in labels else 0.0
                sp = [i for i, xy in enumerate(self.model["xy"]) if xy is not None]
                if sp:
                    q = p[sp] / p[sp].sum(); xy = (q[:, None] * np.array([self.model["xy"][i] for i in sp])).sum(0)
                    room["xy"] = np.round(xy, 2).tolist(); room["spot"] = labels[sp[int(np.argmax(q))]]
                state = "moving" if moving else ("empty" if pe > 0.5 else "still")
                if state == "empty":
                    room.pop("xy", None)
            else:
                state = self.auto(mot, room)
            room["state"] = room["state3"] = state
            room["calibrated"] = self.model is not None
        self.t0 = self.t0 or now
        tt = round(now - self.t0, 1)
        self.hist.append([tt, room.get("motion"), None])
        act = []
        for j in range(len(self.cfg["aps"])):
            ks = [i for i, k in enumerate(self.keys) if k[0] == j]
            act.append(float(np.clip(np.nanmean(mot[ks]) / (2 * self.thr), 0, 1)) if alive[ks].any() else 0.0)
        p = self.pending
        self.last = dict(m=np.round(m, 2).tolist(), mot=np.round(mot, 2).tolist(), keys=[list(k) for k in self.keys],
                         pending=None if not p else dict(label=p["label"], wait=round(max(0, p["t_start"] - now)),
                                                          left=round(max(0, p["t_end"] - now)), n=p["n"]),
                         cv=self.cv, spots=self.spots())
        return dict(mode="LIVE", t=tt, scene=scene_of(self.cfg), room=room, link_activity=act,
                    alive=int(sum(alive[[i for i, k in enumerate(self.keys) if k[0] == j]].any() for j in range(len(self.cfg["aps"])))),
                    history=list(self.hist))

    def auto(self, mot, room):
        """self-calibrating mode, no clicks needed. Each link's 'empty room' wobble is learnt as a low percentile of
        its own 20 s average over the last hours (an empty room is the quietest the link ever gets; integer RSSI
        then barely flickers). Motion = short-term wobble far above that link's usual level. Someone still =
        EVERY link's 20 s wobble stays above its floor (breathing / fidgeting flips the 1 dB steps); with nobody in
        the room at least one link goes flat.
        Walk test in the user's room (6 min): walking caught 109/120 s; sitting still vs empty was NOT separable."""
        self.mhist.append(mot)
        H = np.array(self.mhist, float)
        r4, r20 = np.nanmean(H[-4:], 0), np.nanmean(H[-20:], 0)
        aps = [[i for i, k in enumerate(self.keys) if k[0] == j] for j in range(len(self.cfg["aps"]))]
        if len(H) < 120:
            room["learning"] = f"learning this room: {len(H)}/120 s"
            return "moving" if np.nanmean(r4) > 1.5 else "quiet"
        R20 = np.array([np.nanmean(H[max(0, i - 19):i + 1], 0) for i in range(max(0, len(H) - 1800), len(H), 5)])
        floor = np.nanpercentile(R20, 5, axis=0)
        typ = np.nanmedian(H[-1800:], 0)
        mv = [np.nanmean(r4[k]) / max(0.3, np.nanmean(typ[k])) for k in aps]
        ex = [np.nanmean(r20[k] - floor[k]) for k in aps]
        exk = r20 - floor
        room["auto"] = dict(move_ratio=[round(float(v), 2) for v in mv], excess=[round(float(v), 2) for v in ex],
                            floor=np.round(floor, 2).tolist())
        j = int(np.nanargmax(mv)) if max(mv) > 2 else int(np.nanargmax(ex))
        L, A = np.array(self.cfg["laptop"][:2]), np.array(self.cfg["aps"][j]["xyz"][:2])
        room["zone"] = f"near the laptop-{self.cfg['aps'][j]['name']} line"
        room["xy"] = np.round(L + 0.45 * (A - L), 2).tolist()      # where that link is most sensitive (mid Fresnel zone)
        if max(mv) > 2:
            return "moving"
        # still-vs-empty from beacon RSSI did NOT hold up on the walk test (60% overall, sitting often read as empty),
        # so the live view only claims what was measured to work: motion, and which link it is on.
        room["still_hint"] = bool(np.nanmin(exk) > 0.1)
        room.pop("xy"); room.pop("zone")
        return "quiet"

    def spots(self):
        c = collections.Counter(w["label"] for w in self.cal["windows"])
        return [dict(label=L, n=n, xy=next(w["xy"] for w in self.cal["windows"] if w["label"] == L)) for L, n in c.items()]


def scene_of(cfg):
    from live import room_scene_json
    nodes = np.array([cfg["laptop"]] + [ap["xyz"] for ap in cfg["aps"]], float)
    sj = room_scene_json(cfg["W"], cfg["D"], cfg.get("H", 2.8), nodes, cfg.get("furniture", []), ["esp"] + ["router"] * len(cfg["aps"]))
    return sj


DEFAULT_ROOM = dict(W=4.0, D=3.5, H=2.8, laptop=[2.0, 0.4, 0.8],
                    aps=[dict(name="Router", chan=52, bssids=["ea:6e:44:f0:09:8d", "e8:6e:44:f6:09:8d"], xyz=[0.2, 3.0, 1.0]),
                         dict(name="D-Link", chan=64, bssids=["40:86:cb:cf:15:9c"], xyz=[3.8, 3.0, 1.0])])


# --------------------------------------------------------------------------- server
def make_server(eng, cfg_path, port):
    import live
    from http.server import ThreadingHTTPServer

    class H(live.Handler):
        def do_GET(self):
            p = self.path.split("?")[0]
            if p.startswith("/cal"):
                return self._send(open(os.path.join(HERE, "realtime", "apsense_cal.html"), "rb").read(), "text/html; charset=utf-8")
            if p.startswith("/api/info"):
                with live.LOCK:
                    room = live.STATE.get("room", {})
                return self._send(json.dumps(dict(cfg=eng.cfg, room=room, **eng.last)).encode(), "application/json")
            return super().do_GET()

        def do_POST(self):
            p = self.path.split("?")[0]
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if p == "/api/cal":
                eng.start_cal(body["label"], body.get("xy"), float(body.get("secs", 20)), float(body.get("delay", 3)))
            elif p == "/api/cal/delete":
                eng.cal["windows"] = [w for w in eng.cal["windows"] if w["label"] != body["label"]]; eng.save(); eng.fit()
            elif p == "/api/room":
                eng.cfg.update(body); json.dump(eng.cfg, open(cfg_path, "w"), indent=1)
            self._send(b'{"ok":true}', "application/json")

        def _send(self, body, ctype):
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"dashboard: http://localhost:{port}   calibration: http://localhost:{port}/cal", flush=True)


def main():
    import live
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "replay", "record"])
    ap.add_argument("file", nargs="?"); ap.add_argument("--mon", default="mon0")
    ap.add_argument("--chans", default="52,64"); ap.add_argument("--dwell", type=float, default=1.0)
    ap.add_argument("--record"); ap.add_argument("--http", type=int, default=8000)
    ap.add_argument("--room", default=os.path.join(HERE, "rec", "myroom.json"))
    ap.add_argument("--cal", default=os.path.join(HERE, "rec", "cal.json"))
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.room), exist_ok=True)
    if not os.path.exists(a.room):
        json.dump(DEFAULT_ROOM, open(a.room, "w"), indent=1)
    cfg = json.load(open(a.room))
    rec = open(a.record, "a") if a.record else None
    if a.cmd == "record":
        def out(r):
            t, c, b, s, ch = r
            rec.write(f"{t:.3f},{c},{b},{s.replace(',', ' ')}," + ",".join(map(str, ch)) + "\n"); rec.flush()
        return capture(a.mon, [int(c) for c in a.chans.split(",")], a.dwell, out)
    eng = Engine(cfg, a.cal)
    make_server(eng, a.room, a.http)

    def out(r):
        t, c, b, s, ch = r
        eng.add(t, c, b, ch)
        if rec:
            rec.write(f"{t:.3f},{c},{b},{s.replace(',', ' ')}," + ",".join(map(str, ch)) + "\n"); rec.flush()

    if a.cmd == "run":
        threading.Thread(target=capture, args=(a.mon, [int(c) for c in a.chans.split(",")], a.dwell, out), daemon=True).start()
        while True:
            time.sleep(1)
            try:
                st = eng.tick(time.time()); live.publish(st)
            except Exception as e:                     # never let one bad second kill the live demo
                print("\ntick error:", repr(e), flush=True); continue
            r = st["room"]; print(f"\r{r.get('state','…'):7s} motion={r.get('motion')} thr={r.get('thr')} xy={r.get('xy')}   ", end="", flush=True)
    else:                                              # replay a recording through the same engine
        R = list(read_rec(a.file)); nxt = R[0][0] + 1; t_wall = time.time()
        for t, c, b, s, ch in R:
            while t >= nxt:
                st = eng.tick(nxt); live.publish(st); nxt += 1
                if a.speed:
                    time.sleep(max(0, (nxt - R[0][0]) / a.speed - (time.time() - t_wall)))
            eng.add(t, c, b, ch)


if __name__ == "__main__":
    main()
