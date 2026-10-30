"""Hall mode on 3-D halls (flat + tiered amphitheatres, boards at mixed heights).
Trains on the 2.5-D halls + the 3-D halls, tests on 3-D halls never seen.
python evaluate3d.py -> results/results3d.json, results/fig3d_*.png, models/hall3d.joblib
"""
import glob, json, os, time, warnings
import numpy as np, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hallsense.model import load_scene, SeatModel, RTIBaseline
from hallsense.metrics import auc
from evaluate import score, summarize, BLUE, ORANGE, INK, INK2, GRID, SURF
warnings.filterwarnings("ignore")

def L(pat):
    return [load_scene(f) for f in sorted(glob.glob(pat))]

t0 = time.time()
train = L("data/train/*.npz") + L("data/train_rows/*.npz") + L("data3/train/*.npz")
print("train", len(train), flush=True)
model = SeatModel().fit(train); joblib.dump(model, "models/hall3d.joblib")
rti = RTIBaseline().fit(train[::5])
test = L("data3/test/*.npz")
rows = score(test, model, rti)
for r, s in zip(rows, test):
    r["tiered"] = bool(s["meta"].get("tiered"))
res = dict(test=dict(ours=summarize(rows), rti=summarize([{k: v for k, v in r.items() if k not in ("tol_f1", "block")} | {"count": r["rti_count"], "auc": r["rti_auc"]} for r in rows])),
           tiered=summarize([r for r in rows if r["tiered"]]), flat=summarize([r for r in rows if not r["tiered"]]), rows=rows)
print(json.dumps(res["test"]["ours"]), "\nRTI", json.dumps(res["test"]["rti"]), "\ntiered", json.dumps(res["tiered"]), "\nflat", json.dumps(res["flat"]), flush=True)
sw = L("data3/sweep/*.npz")
if sw:
    srows = score(sw, model)
    res["sweep"] = {int(n): summarize([r for r in srows if r["n_nodes"] == n]) for n in sorted({r["n_nodes"] for r in srows})}
    print("sweep", {n: (round(v["mae_pct_of_seats"], 1), round(v["auc"]["sparse(<25%)"], 2)) for n, v in res["sweep"].items()})
json.dump(res, open("results/results3d.json", "w"), indent=1, default=float)

t = np.array([r["true"] for r in rows]); c = np.array([r["count"] for r in rows]); b = np.array([r["rti_count"] for r in rows])
tier = np.array([r["tiered"] for r in rows])
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True); m = max(t.max(), c.max(), b.max()) * 1.05
for a, v, col, nm in ((ax[0], c, BLUE, "HallSense"), (ax[1], b, ORANGE, "Classic RTI")):
    a.plot([0, m], [0, m], color=INK2, lw=1, ls="--")
    a.scatter(t[tier], v[tier], s=30, color=col, edgecolor=SURF, linewidth=1.2, zorder=3, label="tiered hall")
    a.scatter(t[~tier], v[~tier], s=30, marker="s", facecolor=SURF, edgecolor=col, linewidth=1.5, zorder=3, label="flat hall")
    a.set_title(f"{nm}\nmean abs error = {np.mean(np.abs(v - t)):.1f} students", loc="left", fontsize=10, color=INK)
    a.set_xlabel("true number of seated students"); a.set_xlim(0, m); a.set_ylim(0, m)
ax[0].set_ylabel("estimated"); ax[0].legend(loc="upper left")
fig.suptitle("3-D halls never seen in training (12 ESP32 boards)", x=0.01, ha="left", color=INK)
fig.tight_layout(); fig.savefig("results/fig3d_count.png", dpi=160); plt.close(fig)
if "sweep" in res:
    ns = sorted(res["sweep"]); e = [res["sweep"][n]["mae_pct_of_seats"] for n in ns]
    fig, a = plt.subplots(figsize=(5.5, 3.6)); a.plot(ns, e, color=BLUE, lw=2, marker="o", ms=8, markeredgecolor=SURF)
    a.set_xticks(ns); a.set_ylim(0, max(e) * 1.25); a.set_xlabel("number of ESP32 boards"); a.set_ylabel("count error, % of capacity")
    a.set_title("How few boards can a hall get away with?", loc="left", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig("results/fig3d_nodes.png", dpi=160); plt.close(fig)
print(f"done {time.time() - t0:.0f}s")
