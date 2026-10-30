"""Laptop mode: no ESP32 needed. Your laptop's Wi-Fi card + any access point (phone hotspot works).
Linux only. RSSI only (normal laptop cards don't expose CSI), so: empty / still / moving, and
breathing when you sit or lie between the laptop and the phone. Same 3-D dashboard.

Step 0 - see what your card can do:
    python laptop_sense.py probe
Mode A, no root (coarse, whatever rate the driver refreshes the signal level):
    python laptop_sense.py run
Mode B, root, per-packet RSSI from every antenna chain (much better):
    sudo iw phy phy0 interface add mon0 type monitor && sudo ip link set mon0 up
    sudo python laptop_sense.py run --monitor mon0
Record first, look later (good for the experiment):
    python laptop_sense.py run --record laptop_rec.csv
Then open http://localhost:8000
"""
import argparse, collections, json, os, re, socket, struct, subprocess, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        return str(e)


def wifi_iface():
    """first managed interface (skips monitor ones like mon0, which a lazy regex used to match)"""
    out = sh("iw dev")
    for block in re.split(r"\n\s*Interface ", "\n" + out)[1:]:
        name = block.split()[0]
        if re.search(r"\btype managed\b", block):
            return name
    return None


def link_info(iface):
    out = sh(f"iw dev {iface} link")
    bssid = re.search(r"Connected to ([0-9a-f:]{17})", out)
    freq = re.search(r"freq: (\d+)", out)
    return (bssid.group(1) if bssid else None), (int(freq.group(1)) if freq else None)


def gateway(iface=None):
    out = sh(f"ip route show default dev {iface}" if iface else "ip route show default")
    m = re.search(r"via (\S+)", out)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- traffic
def traffic(gw, hz=50, iface=None):
    """make the access point talk to us ~hz times a second, without root: tiny DNS queries to the
    gateway (phones' hotspots and home routers answer DNS). Replies are downlink frames we measure."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.setblocking(False)
    if iface:      # wired + Wi-Fi on the same router: force the queries over Wi-Fi (needs root)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, iface.encode())
        except OSError:
            pass
    q = bytes.fromhex("12340100000100000000000007657861 6d706c6503636f6d0000010001".replace(" ", ""))
    while True:
        try:
            s.sendto(q, (gw, 53))
            while True:
                s.recv(512)
        except (BlockingIOError, OSError):
            pass
        time.sleep(1.0 / hz)


# --------------------------------------------------------------------------- sources
def source_proc(iface):
    """mode A: poll /proc/net/wireless (signal level of the last frames the driver saw)"""
    while True:
        for line in open("/proc/net/wireless"):
            if line.strip().startswith(iface + ":"):
                parts = line.split()
                lvl = float(parts[3].rstrip("."))
                yield time.time(), [lvl]
        time.sleep(0.01)


_RT = {0: (8, 8), 1: (1, 1), 2: (1, 1), 3: (4, 2), 4: (2, 2), 5: (1, 1), 6: (1, 1), 7: (2, 2), 8: (2, 2), 9: (2, 2),
       10: (1, 1), 11: (1, 1), 12: (1, 1), 13: (1, 1), 14: (2, 2), 15: (2, 2), 16: (1, 1), 17: (1, 1), 18: (8, 4),
       19: (3, 1), 20: (8, 4), 21: (12, 2), 22: (12, 8), 23: (12, 2), 24: (12, 2), 25: (6, 2), 26: (1, 1), 27: (4, 2)}


def radiotap_signals(pkt):
    """all dBm_AntSignal values in a radiotap header (first = combined, then one per antenna chain)"""
    if len(pkt) < 8 or pkt[0] != 0:
        return None, 0
    rt_len = struct.unpack_from("<H", pkt, 2)[0]
    words, off = [], 4
    while True:
        w = struct.unpack_from("<I", pkt, off)[0]; words.append(w); off += 4
        if not (w >> 31) & 1:
            break
    sig = []
    for w in words:
        for bit in range(29):
            if (w >> bit) & 1:
                size, align = _RT.get(bit, (None, None))
                if size is None:
                    return sig, rt_len
                off = (off + align - 1) // align * align
                if bit == 5:
                    sig.append(struct.unpack_from("b", pkt, off)[0])
                off += size
        if (w >> 30) & 1:      # vendor namespace: stop, we already have what we need
            break
    return sig, rt_len


def source_monitor(mon, bssid, beacons=False):
    """mode B: raw 802.11 frames from a monitor interface; keep frames transmitted by the AP.
    beacons=True keeps only beacons (fixed power + rate, ~10 Hz): data frames to other clients come at
    different rates/powers and swing RSSI by several dB, which the model reads as motion."""
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003)); s.bind((mon, 0))
    ap = bytes.fromhex(bssid.replace(":", ""))
    while True:
        pkt = s.recv(4096)
        sig, L = radiotap_signals(pkt)
        if not sig or len(pkt) < L + 16:
            continue
        if pkt[L + 10:L + 16] != ap:       # addr2 = transmitter
            continue
        if beacons and pkt[L] != 0x80:
            continue
        yield time.time(), sig[1:] if len(sig) > 1 else sig


# --------------------------------------------------------------------------- engine
class LaptopEngine:
    def __init__(self, model, sj, mode):
        self.model, self.sj, self.mode = model, sj, mode
        self.buf = collections.deque(); self.hist = collections.deque(maxlen=900); self.t0 = None
        self.smooth = None; self.calm = 0; self.brrun = 0; self.n = 0

    def add(self, t, sig):
        self.buf.append((t, sig)); self.n += 1
        while self.buf and self.buf[0][0] < t - 32:
            self.buf.popleft()

    def tick(self, now):
        from hallsense.rssifeat import rssi_windows
        from live import publish
        self.t0 = self.t0 or now
        room = {}
        if len(self.buf) > 30 and self.buf[-1][0] - self.buf[0][0] > 18:
            t = np.array([b[0] for b in self.buf]); C = min(len(b[1]) for b in self.buf)
            r = np.array([b[1][:C] for b in self.buf], float)
            X, _ = rssi_windows(t - t[0], r[:, :1].mean(1) if C == 1 else r.mean(1), win_s=min(20, t[-1] - t[0] - 0.5), hop_s=100)
            if len(X):
                pr = self.model.predict_proba(X[-1:])[0]
                self.smooth = pr if self.smooth is None else 0.6 * self.smooth + 0.4 * pr
                s3 = ["empty", "still", "moving"][int(np.argmax(self.smooth))]
                self.calm = self.calm + 1 if s3 != "moving" else 0
                room = dict(state=s3, state3=s3, probs=np.round(self.smooth, 3).tolist(),
                            rate_hz=round(len(self.buf) / max(1e-3, t[-1] - t[0]), 1))
                if t[-1] - t[0] > 29 and self.calm >= 25:
                    Xb, rb = rssi_windows(t - t[0], r.mean(1), win_s=30, hop_s=100)
                    if len(Xb) and Xb[-1, 7] > 2.5 and 6 <= rb[-1] <= 36:
                        room["breath_bpm"] = round(float(rb[-1]), 1)
                        if s3 == "empty":
                            room["state"] = room["state3"] = "still"
        tt = round(now - self.t0, 1)
        self.hist.append([tt, room.get("breath_bpm"), None])
        publish(dict(mode=self.mode, t=tt, scene=self.sj, room=room, link_activity=[min(1.0, max(0.0, (room.get("probs") or [1, 0, 0])[2]))],
                     alive=1 if room else 0, history=list(self.hist)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "run"])
    ap.add_argument("--iface"); ap.add_argument("--monitor"); ap.add_argument("--beacons", action="store_true"); ap.add_argument("--record")
    ap.add_argument("--http", type=int, default=8000); ap.add_argument("--hz", type=float, default=50)
    ap.add_argument("--room", help="room.json with router=[x,y,z] (the phone) and boards={'laptop':[x,y,z]}")
    a = ap.parse_args()
    iface = a.iface or wifi_iface()
    if a.cmd == "probe":
        print("interface:", iface)
        print(sh("lspci -nnk | grep -iA3 'network\\|wireless'"))
        print(sh(f"iw dev {iface} link"))
        phy = re.search(r"phy#(\d+)", sh("iw dev")); phy = phy.group(1) if phy else "0"
        modes = sh(f"iw phy phy{phy} info")
        m = re.search(r"Supported interface modes:\n((?:\s+\* .*\n)+)", modes)
        print("interface modes:", m.group(1).replace("\n", " ") if m else "?")
        print("valid interface combinations:", re.findall(r"valid interface combinations:[\s\S]*?\n\n", modes)[:1])
        print("\nIntel AX200/AX210 -> real CSI is possible on Linux with FeitCSI or PicoScenes (ask Claude).")
        return
    import joblib
    from live import serve, room_scene_json
    bssid, freq = link_info(iface)
    gw = gateway(iface)
    print(f"iface {iface}  AP {bssid}  {freq} MHz  gateway {gw}")
    if not bssid or not gw:
        sys.exit("connect the laptop to the phone hotspot (or any AP) first")
    if a.room:
        r = json.load(open(a.room)); W, D, H = r["W"], r["D"], r.get("H", 2.8)
        nodes = np.array([r["router"], list(r["boards"].values())[0]], float)
    else:
        W, D, H = 4.0, 3.5, 2.8; nodes = np.array([[0.2, 1.75, 1.0], [3.8, 1.75, 0.8]])
    sj = room_scene_json(W, D, H, nodes, [], ["router", "esp"])
    eng = LaptopEngine(joblib.load(os.path.join(HERE, "models", "laptop_rssi.joblib")), sj, "LIVE")
    serve(a.http)
    threading.Thread(target=traffic, args=(gw, a.hz, iface), daemon=True).start()
    src = source_monitor(a.monitor, bssid, a.beacons) if a.monitor else source_proc(iface)
    rec = open(a.record, "a") if a.record else None
    lock = threading.Lock()

    def reader():
        last = None
        for t, sig in src:
            if not a.monitor and last == sig:          # /proc repeats the last value: keep only updates
                continue
            last = sig
            with lock:
                eng.add(t, sig)
            if rec:
                rec.write(f"{t:.4f}," + ",".join(str(x) for x in sig) + "\n")
    threading.Thread(target=reader, daemon=True).start()
    while True:
        time.sleep(1)
        with lock:
            eng.tick(time.time())
        st = eng.hist[-1] if eng.hist else None
        print(f"\r{eng.n} samples  state={eng.smooth.round(2) if eng.smooth is not None else '...'}  ", end="", flush=True)


if __name__ == "__main__":
    main()
