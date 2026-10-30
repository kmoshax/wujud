"""Feeds our board's CSI to RuView's sensing server, so their whole pipeline runs on our data.

    .venv/bin/python tools/ruview_bridge.py --replay rec/clean_20261030_0050.csv   # a recording, in real time
    .venv/bin/python tools/ruview_bridge.py --board 192.168.4.1                    # the board, live
    .venv/bin/python tools/ruview_bridge.py --replay rec/guided_X.csv --score rec/guided_X.json

With --score, RuView's verdict is read once a second and graded against what the person was doing,
with the same table guided_test.py gives our own system.

Each CSI line becomes one ADR-018 frame (magic 0xC5110001) sent to RuView's UDP port. The I/Q buffer
is passed through untouched; their firmware sends the same ESP-IDF buffer. Each phone gets its own
node id.
"""
import argparse
import json
import os
import socket
import struct
import threading
import time
import urllib.request

from guided_test import table

CHANNEL_9_MHZ = 2452
OURS = {"absent": "empty", "present_still": "still", "present_moving": "moving", "active": "moving"}


def frame(node, sequence, rssi, iq):
    header = struct.pack("<IBBHIIbbBB", 0xC5110001, node, 1, len(iq) // 2, CHANNEL_9_MHZ, sequence, rssi, -92, 0, 0)
    return header + iq


class Bridge:
    def __init__(self, target):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = target
        self.nodes, self.sequence, self.sent = {}, 0, 0

    def send(self, device, fields):
        if len(fields) < 3 or fields[0] != "P":
            return
        node = self.nodes.setdefault(device, len(self.nodes) + 1)
        self.sequence += 1
        self.sock.sendto(frame(node, self.sequence, int(fields[1]), bytes.fromhex(fields[-1])), self.target)
        self.sent += 1


def ruview_says(api):
    request = urllib.request.Request(api + "/api/v1/sensing/latest")
    if os.environ.get("RUVIEW_API_TOKEN"):
        request.add_header("Authorization", "Bearer " + os.environ["RUVIEW_API_TOKEN"])
    with urllib.request.urlopen(request, timeout=2) as response:
        return json.load(response)["classification"]["motion_level"]


def watch(api, clock, said_at, stop):
    while not stop.is_set():
        try:
            level = ruview_says(api)
            said_at.append((clock(), OURS.get(level, level)))
        except Exception as error:
            print("RuView:", error, flush=True)
        stop.wait(1.0)


def replay(bridge, path, speed, clock_out):
    start_wall, start_rec = time.time(), None
    first = open(path).readline()
    start_rec = float(first.split(",", 1)[0])
    clock_out.append(lambda: start_rec + (time.time() - start_wall) * speed)
    for line in open(path):
        stamp, link, rest = line.rstrip("\n").split(",", 2)
        stamp = float(stamp)
        wait = (stamp - start_rec) / speed - (time.time() - start_wall)
        if wait > 0:
            time.sleep(wait)
        bridge.send(link.rsplit("@", 1)[-1], rest.split(","))
        if bridge.sent % 1000 == 0:
            print(f"{time.strftime('%H:%M:%S', time.localtime(stamp))} recorded, {bridge.sent} frames sent", flush=True)


def live(bridge, board):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.2)
    asked = 0.0
    while True:
        if time.time() - asked > 1.0:
            sock.sendto(b"wujud?", (board, 5005))
            asked = time.time()
        try:
            data, _ = sock.recvfrom(600)
        except socket.timeout:
            continue
        line = data.decode(errors="ignore").strip()
        if "|" in line:
            device, rest = line.split("|", 1)
            bridge.send(device, rest.split(","))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", help="a recording in the app's format")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--board", default="192.168.4.1")
    parser.add_argument("--ruview", default="127.0.0.1:5005")
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--score", help="the guided test's .json: grade RuView against it")
    args = parser.parse_args()
    host, port = args.ruview.split(":")
    bridge = Bridge((host, int(port)))
    if not args.replay:
        live(bridge, args.board)
        return
    clock, said_at, stop = [], [], threading.Event()
    sender = threading.Thread(target=replay, args=(bridge, args.replay, args.speed, clock))
    sender.start()
    while not clock:
        time.sleep(0.01)
    watcher = threading.Thread(target=watch, args=(args.api, clock[0], said_at, stop))
    watcher.start()
    sender.join()
    stop.set()
    watcher.join()
    json.dump(said_at, open(os.path.splitext(args.replay)[0] + "_ruview.json", "w"))
    if args.score:
        table(said_at, json.load(open(args.score))["segments"])


if __name__ == "__main__":
    main()
