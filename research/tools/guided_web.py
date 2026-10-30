"""The guided test as a web page: what to do now in big letters, a countdown, a bell at every change,
and two verdicts live beside it from the same signal: ours, and RuView's.

    cd research && .venv/bin/python tools/guided_web.py --board 192.168.1.34
    open http://127.0.0.1:8770

The board's lines are recorded to rec/guided_<time>.csv, fed live to our replay example (the room
engine, same model as the app), and sent as ADR-018 frames to RuView's sensing server, which is
started fresh when the test starts so that it learns the room while the room is empty. At the end,
the same second-by-second table as guided_test.py for each.
"""
import argparse
import json
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from drift_detector import DriftDetector
from vitals import Vitals
from guided_test import SETTLE, STEPS
from ruview_bridge import OURS, Bridge, ruview_says

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guided_web.html")
RUVIEW = os.path.expanduser("~/Projects/ext/RuView/v2")
ARABIC = {
    "LEAVE the room and close the door": "اخرج من الأوضة واقفل الباب",
    "COME IN and WALK around": "ادخل وامشي في الأوضة",
    "SIT on a chair and keep still": "اقعد على كرسي وما تتحركش",
    "LIE on the bed and keep still": "نام على السرير وما تتحركش",
    "GET UP and WALK around": "قوم وامشي",
    "STAND still": "اقف ثابت",
    "LIE on the bed and breathe normally": "نام على السرير واتنفس عادي",
    "BREATHE with the circle, 10 a minute": "اتنفس مع الدايرة: 10 مرات في الدقيقة",
    "BREATHE with the circle, 20 a minute": "اتنفس مع الدايرة: 20 مرة في الدقيقة",
    "HOLD your breath": "اكتم نفسك",
    "BREATHE normally": "اتنفس عادي",
}
CLASSES = ("empty", "still", "moving")
# Three minutes without leaving the room: is walking visible at all on this link?
QUICK = [
    ("still", 60, "LIE on the bed and keep still"),
    ("moving", 60, "GET UP and WALK around"),
    ("still", 60, "SIT on a chair and keep still"),
]
# Lying still the whole time: does the breathing rate we read follow a pace the person is given?
BREATHING = [
    ("still", 20, "LIE on the bed and breathe normally"),
    ("breath10", 60, "BREATHE with the circle, 10 a minute"),
    ("breath20", 60, "BREATHE with the circle, 20 a minute"),
    ("hold", 15, "HOLD your breath"),
    ("still", 30, "BREATHE normally"),
]
PACE = {"breath10": 10, "breath20": 20}
# Two minutes away, then lying still: does breathing tell someone lying still from nobody?
EMPTY = [
    ("empty", 120, "LEAVE the room and close the door"),
    ("still", 90, "LIE on the bed and breathe normally"),
]
SYSTEMS = ("ours", "drift", "ruview")


class Test:
    def __init__(self, board, hub, phone_at, steps):
        self.board, self.hub, self.phone_at, self.steps = board, hub, phone_at, steps
        self.plan, self.said = [], {s: [] for s in SYSTEMS}
        self.now_says = {s: None for s in SYSTEMS}
        self.lock = threading.Lock()
        self.ruview = self.replay = self.out = None
        self.phone, self.base, self.last_heard = None, None, 0.0
        self.drift, self.drift_link, self.drift_now = DriftDetector(), None, None
        self.vitals, self.vitals_now, self.vitals_seen = Vitals(), None, []
        threading.Thread(target=self.estimate_vitals, daemon=True).start()
        self.bridge = Bridge(("127.0.0.1", 5005))
        threading.Thread(target=self.listen, daemon=True).start()
        threading.Thread(target=self.poll_ruview, daemon=True).start()

    def start(self, prep):
        with self.lock:
            if self.plan and time.time() < self.plan[-1]["end"]:
                return
            t = time.time() + prep
            self.plan, self.said, self.vitals_seen = [], {s: [] for s in SYSTEMS}, []
            for label, seconds, text in self.steps:
                self.plan.append({"start": t, "end": t + seconds, "label": label, "text": ARABIC[text],
                                  "pace": PACE.get(label)})
                t += seconds
            self.base = os.path.join(HERE, "rec", time.strftime("guided_%Y%m%d_%H%M"))
            self.phone = None
            self.out = open(self.base + ".csv", "w")
        self.start_ruview()

    def start_ruview(self):
        if self.ruview:
            self.ruview.terminate()
            self.ruview.wait()
        data = self.base + "_ruview_data"
        os.makedirs(data, exist_ok=True)
        self.ruview = subprocess.Popen(
            [os.path.join(RUVIEW, "target/release/sensing-server"), "--source", "esp32", "--ui-path", "../ui",
             "--data-dir", data, "--no-edge-registry"],
            cwd=RUVIEW, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def start_replay(self, phone):
        self.phone = phone
        room = json.load(open(os.path.join(HERE, "rec", "room.json")))
        room["hub"], room["devices"] = self.hub, {phone: self.phone_at}
        room["names"] = {"router": "ESP32", phone: "phone"}
        json.dump(room, open(self.base + "_room.json", "w"), indent=1)
        self.replay = subprocess.Popen(
            [os.path.join(ROOT, "target/release/examples/replay"), "-", self.base + "_room.json", "1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self.read_replay, args=(self.replay,), daemon=True).start()

    def read_replay(self, replay):
        for line in replay.stdout:
            if "unix" in line:
                says = line.split()[1].lower()
                self.record("ours", float(line.rsplit("unix", 1)[1]), says)

    def record(self, system, unix, says):
        with self.lock:
            self.now_says[system] = says
            if self.plan and self.plan[0]["start"] <= unix < self.plan[-1]["end"]:
                self.said[system].append((unix, says))

    def listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2)
        asked = 0.0
        while True:
            now = time.time()
            # While the board reboots the laptop gets "port unreachable" back; that must not end the test.
            try:
                if now - asked > 1.0:
                    asked = now
                    sock.sendto(b"wujud?", (self.board, 5005))
                data, _ = sock.recvfrom(600)
            except OSError:
                continue
            line = data.decode(errors="ignore").strip()
            if "|P," not in line:
                continue
            # The first device heard after the start is the phone; the router sends no CSI.
            device, rest = line.split("|", 1)
            self.last_heard = now
            self.bridge.send(device, rest.split(","))
            self.drift_link = self.drift_link or device
            if device == self.drift_link:
                iq = bytes.fromhex(rest.rsplit(",", 1)[1])
                with self.lock:
                    self.vitals.add(now, iq)
                said = self.drift.add(now, iq)
                if said:
                    self.drift_now = said[1]
                    self.record("drift", float(said[0]), said[2])
            if not self.out:
                continue
            row = f"{now:.3f},udp:{self.board}@{device},{rest}\n"
            try:
                self.out.write(row)
                if self.phone is None:
                    self.start_replay(device)
                if device == self.phone:
                    self.replay.stdin.write(row)
            except (ValueError, AttributeError, BrokenPipeError):
                pass

    def estimate_vitals(self):
        while True:
            time.sleep(1.0)
            with self.lock:
                estimate = self.vitals.estimate()
            if not estimate:
                continue
            wave = estimate["wave"]
            # How strong the last 8 s of chest motion is against the whole window: falls when breath is held.
            estimate["recent"] = float(sum(v * v for v in wave[-80:]) / 80) ** 0.5
            with self.lock:
                self.vitals_now = estimate
                if self.plan and self.plan[0]["start"] <= time.time() < self.plan[-1]["end"]:
                    self.vitals_seen.append({k: estimate[k] for k in ("breath_bpm", "breath_snr", "recent")} |
                                            {"t": time.time()})

    def poll_ruview(self):
        while True:
            time.sleep(1.0)
            if not self.ruview:
                continue
            try:
                level = ruview_says("http://127.0.0.1:8080")
            except Exception:
                continue
            self.record("ruview", time.time(), OURS.get(level, level))

    def state(self):
        with self.lock:
            now = time.time()
            step = next((i for i, s in enumerate(self.plan) if s["start"] <= now < s["end"]), None)
            started = bool(self.plan)
            done = started and now >= self.plan[-1]["end"]
            if done and self.out:
                self.out.close()
                self.out = None
                json.dump({"segments": self.plan, "said": self.said, "vitals": self.vitals_seen},
                          open(self.base + ".json", "w"), indent=1)
            return {
                "now": now,
                "started": started,
                "done": done,
                "prep_left": max(0.0, self.plan[0]["start"] - now) if started else None,
                "step": step,
                "plan": self.plan,
                "says": self.now_says,
                "drift": self.drift_now,
                "vitals": self.vitals_now,
                "vitals_seen": self.vitals_seen,
                "signal": now - self.last_heard < 3.0,
                "preview": [{"text": ARABIC[text], "start": 0, "end": seconds} for _, seconds, text in self.steps],
                "tables": {s: self.table(s) for s in SYSTEMS} if started else None,
            }

    def table(self, system):
        counts = {c: {d: 0 for d in CLASSES} for c in CLASSES}
        for unix, said in self.said[system]:
            truth = next((s for s in self.plan if s["start"] <= unix < s["end"]), None)
            if truth and truth["label"] in CLASSES and unix - truth["start"] >= SETTLE and said in CLASSES:
                counts[truth["label"]][said] += 1
        return counts


def handler(test, prep):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, kind):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/state":
                self.send(json.dumps(test.state()).encode(), "application/json")
            else:
                self.send(open(PAGE, "rb").read(), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path == "/start":
                test.start(prep)
            self.send(b"{}", "application/json")

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default="192.168.1.34")
    parser.add_argument("--hub", default="2.0,0.5,0.8", help="where the board is, metres (rec/room.json frame)")
    parser.add_argument("--phone", default="3.4,3.25,0.55", help="where the phone lies, metres")
    parser.add_argument("--prep", type=int, default=60)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--quick", action="store_true", help="3 minutes, no leaving the room")
    parser.add_argument("--breathing", action="store_true", help="3 minutes lying down, breathing to a pace")
    parser.add_argument("--empty", action="store_true", help="2 minutes away, then 90 s lying still")
    args = parser.parse_args()
    point = lambda text: [float(v) for v in text.split(",")]
    test = Test(args.board, point(args.hub), point(args.phone), EMPTY if args.empty else BREATHING if args.breathing else QUICK if args.quick else STEPS)
    print(f"http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler(test, args.prep)).serve_forever()


if __name__ == "__main__":
    main()
