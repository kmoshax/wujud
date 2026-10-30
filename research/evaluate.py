"""Train on simulated halls, evaluate on halls the model has never seen, make figures.

python evaluate.py            -> results/results.json, results/*.png, models/wujud.joblib
"""
import glob, json, os, sys, time, warnings
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wujud.model import load_scene, SeatModel, RTIBaseline
from wujud.metrics import auc, seat_f1, tolerant_f1, block_corr

warnings.filterwarnings("ignore")
OUT = "results"
os.makedirs(OUT, exist_ok=True); os.makedirs("models", exist_ok=True)

# palette (validated reference slots, light mode)
BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def load_dir(d):
    fs = sorted(glob.glob(f"data/{d}/*.npz"))
    return [load_scene(f) for f in fs]


def score(scenes, model, rti=None):
    rows = []
    for s in scenes:
        r = model.predict(s); y = s["y"].astype(bool); n = int(y.sum())
        row = dict(path=os.path.basename(s["path"]), n_nodes=len(s["nodes"]), n_seats=len(y), true=n,
                   occ=float(y.mean()), count=r["count"], count_sum=r["count_sum"], count_glob=r["count_glob"],
                   auc=auc(r["p"], s["y"]), f1=seat_f1(r["p"] > 0.5, y),
                   tol_f1=tolerant_f1(r["p"] > 0.5, y, s["seats"]), block=block_corr(r["p"], s["y"], s["rc"]),
                   placement=s["meta"].get("placement"))
        if rti is not None:
            b = rti.predict(s)
            row.update(rti_count=b["count"], rti_auc=auc(b["p"], s["y"]), rti_f1=seat_f1(b["occ"], y))
        rows.append(row)
    return rows


def summarize(rows, key="count"):
    t = np.array([r["true"] for r in rows]); c = np.array([r[key] for r in rows]); S = np.array([r["n_seats"] for r in rows])
    big = t >= 20
    out = dict(n_scenes=len(rows), mae=float(np.mean(np.abs(c - t))),
               mae_pct_of_seats=float(np.mean(np.abs(c - t) / S) * 100),
               mape_20plus=float(np.mean(np.abs(c[big] - t[big]) / t[big]) * 100) if big.any() else None,
               within10=float(np.mean(np.abs(c[big] - t[big]) <= 0.1 * t[big]) * 100) if big.any() else None,
               max_true=int(t.max()))
    for k in ("auc", "tol_f1", "block"):
        if k in rows[0]:
            v = np.array([r[k] for r in rows], float)
            occ = np.array([r["occ"] for r in rows])
            out[k] = {"sparse(<25%)": float(np.nanmean(v[occ < 0.25])), "medium": float(np.nanmean(v[(occ >= 0.25) & (occ < 0.6)])),
                      "dense(>=60%)": float(np.nanmean(v[occ >= 0.6]))}
    return out


def main():
    t0 = time.time()
    train = load_dir("train") + load_dir("train_rows")
    print(f"train scenes: {len(train)} ({time.time() - t0:.0f}s)", flush=True)
    model = SeatModel().fit(train)
    rti = RTIBaseline().fit(train[::4])
    joblib.dump(model, "models/wujud.joblib")
    print(f"trained ({time.time() - t0:.0f}s)", flush=True)

    res = {}
    sets = {}
    for d in ("test", "test_rows", "harsh", "calm"):
        sc = load_dir(d)
        if not sc:
            continue
        sets[d] = sc
        rows = score(sc, model, rti)
        res[d] = dict(ours=summarize(rows), ours_seatsum=summarize(rows, "count_sum"),
                      ours_global=summarize(rows, "count_glob"),
                      rti=summarize([{k: v for k, v in r.items() if k not in ("tol_f1", "block")} | {"count": r["rti_count"], "auc": r["rti_auc"]} for r in rows]), rows=rows)
        print(d, json.dumps({k: v for k, v in res[d]["ours"].items()}, indent=None), flush=True)
        print("   rti", json.dumps(res[d]["rti"]), flush=True)

    # one-shot hall calibration: several lectures in the same deployed hall; the operator counts heads ONCE
    hc = load_dir("hallcal")
    if hc:
        rows = score(hc, model)
        by = {}
        for r in rows:
            by.setdefault(r["path"].split("_")[0], []).append(r)
        raw, cal, tt = [], [], []
        for h, rr in by.items():
            for k, ref in enumerate(rr):
                if ref["true"] < 15:
                    continue
                g = float(np.clip(ref["true"] / max(ref["count"], 1.0), 0.5, 2.0))
                for m, r in enumerate(rr):
                    if m != k:
                        raw.append(r["count"]); cal.append(r["count"] * g); tt.append(r["true"])
        raw, cal, tt = map(np.array, (raw, cal, tt))
        big = tt >= 20
        res["hallcal"] = dict(pairs=int(len(tt)), raw_mae=float(np.mean(np.abs(raw - tt))), cal_mae=float(np.mean(np.abs(cal - tt))),
                              raw_mape_20plus=float(np.mean(np.abs(raw - tt)[big] / tt[big]) * 100),
                              cal_mape_20plus=float(np.mean(np.abs(cal - tt)[big] / tt[big]) * 100),
                              cal_within10=float(np.mean(np.abs(cal - tt)[big] <= 0.1 * tt[big]) * 100),
                              raw_within10=float(np.mean(np.abs(raw - tt)[big] <= 0.1 * tt[big]) * 100),
                              points=[tt.tolist(), raw.tolist(), cal.tolist()])
        print("hallcal", json.dumps({k: v for k, v in res["hallcal"].items() if k != "points"}), flush=True)

    sweep = load_dir("sweep")
    if sweep:
        rows = score(sweep, model)
        res["sweep"] = {}
        for n in sorted({r["n_nodes"] for r in rows}):
            rr = [r for r in rows if r["n_nodes"] == n]
            res["sweep"][n] = summarize(rr)
        print("sweep", json.dumps({n: (v["mae_pct_of_seats"], v["auc"]["sparse(<25%)"]) for n, v in res["sweep"].items()}), flush=True)

    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=float)
    figures(res, sets, model)
    print(f"done ({time.time() - t0:.0f}s)")


def figures(res, sets, model):
    # 1. count: predicted vs true on unseen halls
    rows = res["test"]["rows"] + res.get("test_rows", {}).get("rows", [])
    t = np.array([r["true"] for r in rows]); c = np.array([r["count"] for r in rows]); b = np.array([r["rti_count"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    m = max(t.max(), c.max(), b.max()) * 1.05
    for a, v, col, name in ((ax[0], c, BLUE, "wujud (this work)"), (ax[1], b, ORANGE, "Classic RTI baseline")):
        a.plot([0, m], [0, m], color=INK2, lw=1, ls="--")
        a.scatter(t, v, s=28, color=col, edgecolor=SURF, linewidth=1.2, zorder=3)
        mae = np.mean(np.abs(v - t))
        a.set_title(f"{name}\nmean abs error = {mae:.1f} people", loc="left", color=INK, fontsize=10)
        a.set_xlabel("true number of seated students"); a.set_xlim(0, m); a.set_ylim(0, m)
    ax[0].set_ylabel("estimated number")
    fig.suptitle("Head-count in lecture halls the model never saw (simulation, 12 ESP32 nodes)", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_count.png", dpi=160); plt.close(fig)

    # 2. seat maps: sparse / medium / dense example
    sc = sets["test"]
    occs = np.array([s["y"].mean() for s in sc])
    picks = [int(np.argmin(np.abs(occs - q))) for q in (0.12, 0.4, 0.8)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    for a, k in zip(ax, picks):
        s = sc[k]; r = model.predict(s); y = s["y"].astype(bool)
        for i, j in s["links"]:
            a.plot(*np.c_[s["nodes"][i], s["nodes"][j]], color=GRID, lw=0.5, zorder=1)
        sca = a.scatter(*s["seats"].T, c=r["p"], cmap="Blues", vmin=0, vmax=1, s=46, marker="s", zorder=2,
                        edgecolor=SURF, linewidth=0.8)
        a.scatter(*s["seats"][y].T, s=9, color=INK, zorder=3, label="true student")
        a.scatter(*s["nodes"].T, s=60, marker="^", color=ORANGE, edgecolor=SURF, zorder=4, label="ESP32 node")
        a.set_aspect("equal"); a.set_xlim(-0.3, s["W"] + 0.3); a.set_ylim(-0.3, s["D"] + 0.3); a.grid(False)
        a.set_title(f"true {y.sum()} / est. {r['count']:.0f}   (seat AUC {auc(r['p'], s['y']):.2f})", loc="left", fontsize=10, color=INK)
        a.axhspan(0, s["front"], color=GRID, alpha=0.4, lw=0); a.text(0.2, s["front"] / 2, "lecturer zone", color=INK2, fontsize=8, va="center")
        a.set_xticks([]); a.set_yticks([])
    ax[0].legend(loc="lower right", fontsize=8, facecolor=SURF, frameon=True, edgecolor=GRID)
    cb = fig.colorbar(sca, ax=ax, fraction=0.015, pad=0.01); cb.set_label("P(seat occupied)")
    fig.suptitle("Seat map: square = model's probability, dot = where a student really sits", x=0.01, ha="left", color=INK)
    fig.savefig(f"{OUT}/fig2_maps.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 1b. one-shot hall calibration
    if "hallcal" in res:
        tt, raw, cal = map(np.array, res["hallcal"]["points"])
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
        m = max(tt.max(), raw.max(), cal.max()) * 1.05
        for a, v, title in ((ax[0], raw, "straight from simulation"), (ax[1], cal, "after ONE manual head-count in that hall")):
            a.plot([0, m], [0, m], color=INK2, lw=1, ls="--")
            a.scatter(tt, v, s=16, color=BLUE, edgecolor=SURF, linewidth=0.8, zorder=3, alpha=0.8)
            big = tt >= 20
            a.set_title(f"{title}\nmedian error {np.median(np.abs(v - tt)[big] / tt[big]) * 100:.0f}% (halls with 20+ students)", loc="left", fontsize=10, color=INK)
            a.set_xlabel("true number of seated students"); a.set_xlim(0, m); a.set_ylim(0, m)
        ax[0].set_ylabel("estimated number")
        fig.suptitle("Same hall, different lectures: calibrating once removes the hall's own bias", x=0.01, ha="left", color=INK)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig5_hallcal.png", dpi=160); plt.close(fig)

    # 3. node sweep
    if "sweep" in res:
        ns = sorted(int(n) for n in res["sweep"])
        e = [res["sweep"][n]["mae_pct_of_seats"] for n in ns]
        au = [res["sweep"][n]["auc"]["sparse(<25%)"] for n in ns]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].plot(ns, e, color=BLUE, lw=2, marker="o", ms=8, markeredgecolor=SURF)
        ax[0].set_title("Head-count error (% of hall capacity)", loc="left", fontsize=10, color=INK); ax[0].set_xlabel("number of ESP32 nodes")
        ax[0].set_ylim(0, max(e) * 1.2)
        ax[1].plot(ns, au, color=BLUE, lw=2, marker="o", ms=8, markeredgecolor=SURF)
        ax[1].axhline(0.5, color=INK2, lw=1, ls="--"); ax[1].text(ns[0], 0.51, "chance", color=INK2, fontsize=8)
        ax[1].set_title("Seat-level AUC, sparse hall (<25% full)", loc="left", fontsize=10, color=INK); ax[1].set_xlabel("number of ESP32 nodes")
        ax[1].set_ylim(0.45, 1.0)
        for a in ax: a.set_xticks(ns)
        fig.suptitle("How many boards do you need? (same halls, same students, perimeter placement)", x=0.01, ha="left", color=INK)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig3_nodes.png", dpi=160); plt.close(fig)

    # 4. robustness
    names = [("test", "normal"), ("test_rows", "row-end\nplacement"), ("harsh", "1.6x worse\nESP32 noise"), ("calm", "very calm\ncrowd")]
    names = [(k, l) for k, l in names if k in res]
    v = [res[k]["ours"]["mae_pct_of_seats"] for k, _ in names]
    vr = [res[k]["rti"]["mae_pct_of_seats"] for k, _ in names]
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - w / 2 - 0.01, v, w, color=BLUE, label="wujud")
    ax.bar(x + w / 2 + 0.01, vr, w, color=ORANGE, label="Classic RTI")
    for xi, vi in zip(x, v):
        ax.text(xi - w / 2, vi + 0.4, f"{vi:.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in names]); ax.grid(axis="x", visible=False)
    ax.set_ylabel("count error, % of capacity"); ax.legend()
    ax.set_title("Robustness on unseen halls", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_robustness.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
