"""Writes a live.py demo-room simulation and the Python RoomEngine's per-second 3-state decision to an .npz,
so the Rust RoomEngine can be replayed on exactly the same packets.
.venv/bin/python tools/export_room_stream.py <boards> <seed> <out.npz> [seconds]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import live
from wujud.scenes3d import random_room
from wujud.sim3d import simulate_room

PLAN = [("away", 25), ("walk", 15), ("sit", 50), ("walk", 10), ("lie", 80), ("walk", 12), ("away", 25)]
STATE3 = ["empty", "still", "moving"]
TRUTH3 = {"away": 0, "walk": 2, "sit": 1, "lie": 1, "stand": 1}

boards, seed, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
seconds = int(sys.argv[4]) if len(sys.argv) > 4 else sum(d for _, d in PLAN)
sc = random_room(np.random.default_rng(seed), n_esp=boards, smart=True)
out, lab, _ = simulate_room(sc, np.random.default_rng(seed), seconds=sum(d for _, d in PLAN), fs=50, plan=PLAN)
published = []
live.publish = lambda st: published.append(st)
m1, m3 = live.models(None)
sj = live.room_scene_json(sc.W, sc.D, sc.H, sc.nodes, sc.furniture, ["router"] + ["esp"] * (len(sc.nodes) - 1))
eng = live.RoomEngine(sj, len(out), [tuple(l) for l in sc.links], "SIMULATION", m1, m3, sc.nodes)
ks = [0] * len(out)
decisions, probs, truth = [], [], []
for sec in range(seconds):
    for li, o in enumerate(out):
        while ks[li] < len(o["t"]) and o["t"][ks[li]] < sec + 1:
            eng.add(li, o["t"][ks[li]], o["iq"][ks[li]], o["rssi"][ks[li]]); ks[li] += 1
    eng.tick(sec + 1)
    room = published[-1]["room"]
    decisions.append(STATE3.index(room["state3"]) if "state3" in room else -1)
    probs.append(room.get("probs", [np.nan] * 5))
    truth.append(TRUTH3[live.ACT[lab["activity"][min(len(lab["activity"]) - 1, int((sec + 0.5) * 50))]]])
arrays = dict(state3=np.array(decisions, np.int64), probs=np.array(probs, float), truth3=np.array(truth, np.int64),
              links=np.array([list(l) for l in sc.links], np.int64), nodes=np.asarray(sc.nodes, float),
              room=np.array([sc.W, sc.D, sc.H]))
for k, o in enumerate(out):
    keep = o["t"] < seconds
    arrays[f"t{k}"] = o["t"][keep].astype(float)
    arrays[f"iq{k}"] = o["iq"][keep].reshape(keep.sum(), -1).astype(np.int8)
    arrays[f"rssi{k}"] = o["rssi"][keep].astype(float)
np.savez_compressed(path, **arrays)
print(path, {k: v.shape for k, v in arrays.items()})
