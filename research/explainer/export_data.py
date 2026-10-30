"""Export small, real samples from the simulator for the explainer page."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hallsense.scenes3d import random_room
from hallsense.sim3d import simulate_room, FREQS56
from hallsense.physics3d import link_paths3, link_channel, static_rest, diffuse_paths
from hallsense.roomfeat import to_grid, amplitude_from_iq

out = {}
sc = random_room(np.random.default_rng(11), smart=True)
plan = [("away", 20), ("walk", 12), ("lie", 40)]
res, lab, B = simulate_room(sc, np.random.default_rng(5), seconds=72, fs=50, plan=plan)
o = res[0]
a = sc.nodes[0]; b = sc.nodes[1]
# clean channel for the same bodies
paths = link_paths3(a, b, sc.W, sc.D, sc.H, sc.floor, sc.gam_walls) + diffuse_paths(a, b, sc.W, sc.D, sc.H, np.random.default_rng(1), 24)
rest = static_rest(a, b, sc.clutter, FREQS56, np.random.default_rng(1), 0)
Hc = link_channel(a, b, paths, rest, B, B.c.shape[1], FREQS56)
amp_meas = to_grid(o["t"], amplitude_from_iq(o["iq"]))            # 20 Hz grid
clean = np.abs(Hc); clean = clean / clean.mean(1, keepdims=True)
cg = clean[::int(lab["fs"] / 20)][: len(amp_meas)]
# pick the subcarrier where breathing is strongest during 'lie'
s0, s1 = 32 * 20, 72 * 20
var = np.var(20 * np.log10(cg[s0:s1]), 0); k = int(np.argmax(var))
km = int(np.argmax(np.var(20 * np.log10(np.maximum(amp_meas[s0:s1], 1e-3)), 0)))
t = np.arange(len(amp_meas)) / 20
out["signal"] = dict(t=np.round(t, 2).tolist(), clean=np.round(20 * np.log10(cg[:, k]), 3).tolist(),
                     measured=np.round(20 * np.log10(np.maximum(amp_meas[:, k], 1e-3)), 3).tolist(),
                     phases=[{"name": n, "start": s, "end": s + d} for (n, d), s in zip(plan, np.cumsum([0] + [d for _, d in plan])[:-1].tolist())],
                     breath_bpm=round(lab["breath_hz"] * 60, 1))
# spectrum of the 'lie' part (measured), all subcarriers combined
A = 20 * np.log10(np.maximum(amp_meas[s0:s1], 1e-3)); A = A - A.mean(0)
P = np.abs(np.fft.rfft(A * np.hanning(len(A))[:, None], n=8 * len(A), axis=0)) ** 2
f = np.fft.rfftfreq(8 * len(A), 1 / 20); m = f <= 1.0
sp = P[m].mean(1); sp = sp / sp.max()
out["spectrum"] = dict(f_bpm=np.round(f[m] * 60, 2).tolist(), p=np.round(sp, 4).tolist(), true_bpm=round(lab["breath_hz"] * 60, 1),
                       est_bpm=round(float(f[m][np.argmax(sp * (f[m] > 0.1))] * 60), 1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_samples.json"), "w"))
print("ok", out["spectrum"]["true_bpm"], out["spectrum"]["est_bpm"], len(out["signal"]["t"]))
