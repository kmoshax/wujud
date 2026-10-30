"""The live.py demo-room scenario scored like crates/app/examples/demo_accuracy.rs: 3-state decision vs the simulator's
truth, on seconds at least 20 s after a change of activity.  .venv/bin/python bench/demo_accuracy.py <boards> <seed>"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import live
from hallsense.scenes3d import random_room
from hallsense.sim3d import simulate_room

PLAN = [("away", 25), ("walk", 15), ("sit", 50), ("walk", 10), ("lie", 80), ("walk", 12), ("away", 25)]
STATE3 = {"away": "empty", "walk": "moving", "sit": "still", "lie": "still", "stand": "still"}

boards, seed = int(sys.argv[1]), int(sys.argv[2])
sc = random_room(np.random.default_rng(seed), n_esp=boards, smart=True)
out, lab, _ = simulate_room(sc, np.random.default_rng([seed, 1]), seconds=sum(d for _, d in PLAN), fs=50, plan=PLAN)
published = []
live.publish = lambda st: published.append(st)
m1, m3 = live.models(None)
sj = live.room_scene_json(sc.W, sc.D, sc.H, sc.nodes, sc.furniture, ["router"] + ["esp"] * (len(sc.nodes) - 1))
eng = live.RoomEngine(sj, len(out), [tuple(l) for l in sc.links], "SIMULATION", m1, m3, sc.nodes)
fs = lab["fs"] if "fs" in lab else 50
ks = [0] * len(out); right = scored = since = 0; prev = None
for sec in range(sum(d for _, d in PLAN)):
    for li, o in enumerate(out):
        while ks[li] < len(o["t"]) and o["t"][ks[li]] < sec + 1:
            eng.add(li, o["t"][ks[li]], o["iq"][ks[li]], o["rssi"][ks[li]]); ks[li] += 1
    eng.tick(sec + 1)
    truth = STATE3[["away", "lie", "sit", "stand", "walk"][lab["activity"][min(len(lab["activity"]) - 1, int((sec + 0.5) * fs))]]]
    since = since + 1 if truth == prev else 0; prev = truth
    if since >= 20:
        scored += 1; right += published[-1]["room"].get("state3") == truth
print(f"PYTHON boards {boards} seed {seed}: ACCURACY {right}/{scored} = {100 * right / max(scored, 1):.1f}%")
