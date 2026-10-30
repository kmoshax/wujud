"""A guided test in a real room: tells you what to do and when, records the board, and scores what
the system said against what you were doing, second by second.

    cd research && .venv/bin/python tools/guided_test.py            # the laptop joined to the board's Wi-Fi
    cd research && .venv/bin/python tools/guided_test.py --ruview http://127.0.0.1:8080   # RuView on its own firmware

Steps: leave the room, walk, sit still, lie still, stand still, walk, leave. Writes rec/guided_<time>.csv
(the app's recording format), rec/guided_<time>.json (what you were asked to do, with times), and
the score. The first 25 s after every change are not scored: the model looks at 20-s windows.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
STEPS = [
    ("empty", 120, "LEAVE the room and close the door"),
    ("moving", 45, "COME IN and WALK around"),
    ("still", 90, "SIT on a chair and keep still"),
    ("still", 90, "LIE on the bed and keep still"),
    ("moving", 30, "GET UP and WALK around"),
    ("still", 60, "STAND still"),
    ("empty", 90, "LEAVE the room and close the door"),
]
SETTLE = 25.0
BIG = "\033[1m"
RESET = "\033[0m"


BELL = "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"


def say(text):
    print(f"\a\n{BIG}{'=' * 60}\n  {text}\n{'=' * 60}{RESET}", flush=True)
    if os.path.exists(BELL):
        subprocess.Popen(["paplay", BELL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def guide(steps, prep, tick):
    """Tells the person what to do and when; calls `tick(now)` in between. Returns the segments."""
    say(f"Get ready: the test starts in {prep} s")
    plan, t = [], time.time() + prep
    for label, seconds, text in steps:
        plan.append((t, t + seconds, label, text))
        t += seconds
    segments, current = [], -1
    while time.time() < plan[-1][1]:
        now = time.time()
        k = next((i for i, (a, b, *_) in enumerate(plan) if a <= now < b), -1)
        if k != current and k >= 0:
            current = k
            a, b, label, text = plan[k]
            say(f"{text}   ({int(b - a)} s)")
            segments.append({"start": a, "end": b, "label": label, "instruction": text})
        tick(now)
    say("Done, thank you. Scoring…")
    return segments


def record(board, path, steps, prep):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.2)
    devices, asked = set(), [0.0]
    with open(path, "w") as out:

        def tick(now):
            if now - asked[0] > 1.0:
                sock.sendto(b"wujud?", (board, 5005))
                asked[0] = now
            try:
                data, _ = sock.recvfrom(600)
            except socket.timeout:
                return
            line = data.decode(errors="ignore").strip()
            if "|" in line:
                device, rest = line.split("|", 1)
                devices.add(device)
                out.write(f"{now:.3f},udp:{board}@{device},{rest}\n")

        segments = guide(steps, prep, tick)
    return segments, sorted(devices)


def follow_ruview(api, steps, prep):
    """RuView on its own firmware: its verdict, read once a second while the person follows the steps."""
    from ruview_bridge import OURS, ruview_says

    said_at, asked = [], [0.0]

    def tick(now):
        if now - asked[0] >= 1.0:
            asked[0] = now
            try:
                level = ruview_says(api)
                said_at.append((now, OURS.get(level, level)))
            except Exception as error:
                print("RuView:", error, flush=True)
        time.sleep(0.05)

    return guide(steps, prep, tick), said_at


def score(recording, room_file, segments):
    run = subprocess.run(
        ["cargo", "run", "--release", "-q", "-p", "wujud", "--example", "replay", "--", recording, room_file, "1"],
        cwd=ROOT, capture_output=True, text=True,
    )
    said = [(float(line.rsplit("unix", 1)[1]), line.split()[1].lower()) for line in run.stdout.splitlines() if "unix" in line]
    table(said, segments)


def table(said_at, segments):
    """`said_at`: (unix time, "empty" | "still" | "moving") pairs, one a second."""
    rows = []
    for unix, said in said_at:
        truth = next((s for s in segments if s["start"] <= unix < s["end"]), None)
        if truth and unix - truth["start"] >= SETTLE:
            rows.append((truth["label"], said))
    classes = ("empty", "still", "moving")
    print(f"\n{'really':>8} | " + " ".join(f"{c:>7}" for c in classes) + " |  right")
    for c in classes:
        said = [s for t, s in rows if t == c]
        counts = [said.count(x) for x in classes]
        right = counts[classes.index(c)] / max(1, len(said))
        print(f"{c:>8} | " + " ".join(f"{n:7d}" for n in counts) + f" |  {100 * right:5.1f}%  of {len(said)} s")
    total = sum(1 for t, s in rows if t == s)
    print(f"\noverall {100 * total / max(1, len(rows)):.1f}% of {len(rows)} scored seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default="192.168.4.1")
    parser.add_argument("--prep", type=int, default=20, help="seconds before the first step")
    parser.add_argument("--score-only", help="score an existing guided_<time> recording (path without extension)")
    parser.add_argument("--ruview", metavar="URL", help="test RuView's own server instead, e.g. http://127.0.0.1:8080")
    args = parser.parse_args()
    if args.score_only:
        meta = json.load(open(args.score_only + ".json"))
        score(args.score_only + ".csv", args.score_only + "_room.json", meta["segments"])
        return
    stamp = time.strftime("%Y%m%d_%H%M")
    base = os.path.join(HERE, "rec", f"guided_{stamp}")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    if args.ruview:
        segments, said_at = follow_ruview(args.ruview, STEPS, args.prep)
        json.dump({"segments": segments, "ruview": said_at}, open(base + "_ruview.json", "w"), indent=1)
        table(said_at, segments)
        return
    segments, devices = record(args.board, base + ".csv", STEPS, args.prep)
    if not devices:
        sys.exit("no CSI arrived: is the laptop on the board's Wi-Fi, and is the phone connected to it?")
    template = json.load(open(os.path.join(HERE, "rec", "room.json")))
    known = list(template.get("devices", {}).values())
    template["devices"] = {d: known[i % len(known)] if known else [2.0, 2.0, 0.8] for i, d in enumerate(devices)}
    json.dump(template, open(base + "_room.json", "w"), indent=1)
    json.dump({"segments": segments, "devices": devices}, open(base + ".json", "w"), indent=1)
    score(base + ".csv", base + "_room.json", segments)


if __name__ == "__main__":
    main()
