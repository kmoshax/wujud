"""Paper-style figures from research/results/*.json -> research/results/paper/fig*.png

.venv/bin/python tools/paper_figures.py   (run from research/)
"""
import json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/paper"
J = lambda p: json.load(open(os.path.join("results", p)))
OURS, THEIRS, GREY, LIGHT = "#1f4e79", "#b0b0b0", "#555555", "#e8e8e8"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 10.5,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})




def label_bars(ax, bars, fmt="{:.2f}", dy=0.012):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(b.get_height()), ha="center", va="bottom", fontsize=8.5)


def fig_ruview():
    r = J("ruview_compare_native.json"); h, v = r["wujud (sim-only)"], r["RuView edge"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1, 2.1], wspace=0.35))
    bars = a.bar([0, 1], [h["motion_false_alarm"] * 100, v["motion_false_alarm"] * 100], color=[OURS, THEIRS], width=0.6)
    label_bars(a, bars, "{:.1f}%", 1.5)
    a.set_xticks([0, 1], ["wujud", "RuView"]); a.set_ylim(0, 115); a.set_ylabel("Motion false-alarm rate (%)")
    x = np.arange(2); w = 0.36
    b1 = b.bar(x - w / 2, [h["motion_auc"], h["still_vs_empty_auc"]], w, color=OURS, label="wujud (trained on simulation only)")
    b2 = b.bar(x + w / 2, [v["motion_auc"], v["still_vs_empty_auc"]], w, color=THEIRS, label="RuView on-device detector (original C)")
    label_bars(b, list(b1) + list(b2))
    b.axhline(0.5, color=GREY, lw=0.8, ls="--"); b.text(1.62, 0.51, "chance", fontsize=8, color=GREY, ha="right")
    b.set_xticks(x, ["Motion vs. rest", "Still person vs. empty"]); b.set_ylim(0, 1.18); b.set_ylabel("ROC AUC"); b.legend(frameon=False, loc="upper right")
    fig.savefig(f"{OUT}/fig1_ruview.png"); plt.close(fig)


def fig_sim2real():
    s = J("room_sim2real.json")["all"]; sim, real = s["sim_only"], s["real_trained"]
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    names = ["Motion recall", "Motion false alarm", "Still vs. empty AUC"]
    a = [sim["motion_recall"], sim["motion_fp"], sim["auc_empty_still"]]
    b = [real["motion_recall"], real["motion_fp"], real["auc_empty_still"]]
    x = np.arange(3); w = 0.36
    b1 = ax.bar(x - w / 2, a, w, color=OURS, label="Trained on simulation only")
    b2 = ax.bar(x + w / 2, b, w, color=THEIRS, label="Trained on real data from other rooms")
    label_bars(ax, list(b1) + list(b2), "{:.2f}", 0.015)
    ax.set_xticks(x, names); ax.set_ylim(0, 1.2); ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    fig.savefig(f"{OUT}/fig2_sim2real.png"); plt.close(fig)


def fig_hall():
    r = J("results3d.json"); rows = r["rows"]
    t = np.array([q["true"] for q in rows]); c = np.array([q["count"] for q in rows]); rt = np.array([q["rti_count"] for q in rows])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(wspace=0.32))
    m = max(t.max(), c.max(), rt.max()) * 1.05
    a.plot([0, m], [0, m], color=GREY, lw=0.8, ls="--")
    a.scatter(t, rt, s=14, facecolors="none", edgecolors=GREY, lw=0.8, label=f"RTI baseline (MAE {r['test']['rti']['mae_pct_of_seats']:.1f}% of seats)")
    a.scatter(t, c, s=14, color=OURS, label=f"wujud (MAE {r['test']['ours']['mae_pct_of_seats']:.1f}% of seats)")
    a.set_xlim(0, m); a.set_ylim(0, m); a.set_xlabel("True head-count"); a.set_ylabel("Estimated head-count"); a.legend(frameon=False, loc="upper left")
    sw = sorted(r["sweep"].items(), key=lambda kv: int(kv[0]))
    n = [int(k) for k, _ in sw]; e = [v["mae_pct_of_seats"] for _, v in sw]
    b.plot(n, e, "-o", color=OURS, ms=4)
    for xi, yi in zip(n, e):
        b.text(xi, yi + 0.6, f"{yi:.0f}%", ha="center", fontsize=8.5)
    b.set_xlim(min(n) - 1, max(n) + 1); b.set_xticks(n); b.set_ylim(0, max(e) + 4); b.set_xlabel("ESP32 boards on the walls"); b.set_ylabel("Count error (% of seats)")
    fig.savefig(f"{OUT}/fig3_hall.png"); plt.close(fig)


def fig_wall():
    c = J("room_world.json")["through_wall"]
    db = [q["wall_db"] for q in c]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.plot(db, [q["still_auc"] for q in c], "-o", color=OURS, ms=4, label="Still, breathing person vs. empty")
    ax.plot(db, [q["motion_auc"] for q in c], "-s", color=GREY, ms=4, label="Moving person vs. empty")
    ax.axhline(0.5, color=GREY, lw=0.8, ls="--")
    for x, name in ((6, "wooden door"), (18, "18-in. concrete"), (40, "reinforced concrete")):
        ax.axvline(x, color=LIGHT, lw=6, zorder=0); ax.text(x, 0.43, name, ha="center", fontsize=8, color=GREY)
    ax.set_ylim(0.4, 1.04); ax.set_xlabel("One-way wall attenuation (dB)"); ax.set_ylabel("ROC AUC")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.3))
    fig.savefig(f"{OUT}/fig4_wall.png"); plt.close(fig)


def fig_latch():
    o = J("occupancy_engine_replay.json"); fa = J("occupancy_real_fa.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    shade = {"empty": "#ffffff", "walk": "#f1f1f1", "still": "#dde7f1"}
    for ax, key in zip(axes, ("day1 realistic", "day2 realistic")):
        rows = o[key]; t = np.array([r[0] for r in rows]); truth = [r[1] for r in rows]
        d = np.array([np.nan if r[3] is None else r[3] for r in rows], float)
        start = 0
        for i in range(1, len(rows) + 1):
            if i == len(rows) or truth[i] != truth[start]:
                ax.axvspan(t[start], t[i - 1], color=shade[truth[start]], zorder=0)
                ax.text((t[start] + t[i - 1]) / 2, 60, {"empty": "empty", "walk": "walking", "still": "sitting still"}[truth[start]],
                        ha="center", fontsize=8, color=GREY)
                start = i
        ax.semilogy(t, d, color=OURS, lw=1.2)
        ax.axhline(0.4, color="#b23b3b", lw=0.9, ls="--"); ax.axhline(1.2, color="#b23b3b", lw=0.9, ls=":")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0.05, 120)
    axes[0].set_ylabel("Distance to fresh empty reference")
    axes[1].text(t[-1], 0.43, "threshold after motion", color="#b23b3b", fontsize=8, ha="right", va="bottom")
    axes[1].text(t[-1], 1.3, "threshold if no one walked in", color="#b23b3b", fontsize=8, ha="right", va="bottom")
    fig.savefig(f"{OUT}/fig5_latch.png"); plt.close(fig)


def fig_breath():
    b = J("breath_visibility.json"); s = J("sim_calibrate.json")
    calibrated = min(g["sim_visibility"] for g in s["grid"])
    vals = [b["sim"]["glrt"] * 100, calibrated * 100, s["target_visibility_train"] * 100, b["real"]["glrt"] * 100]
    names = ["Simulator\n(original)", "Simulator\n(calibrated)", "Real ESP32\n(train split)", "Real ESP32\n(all recordings)"]
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    bars = ax.bar(range(4), vals, color=[THEIRS, OURS, "#3a3a3a", "#3a3a3a"], width=0.6)
    label_bars(ax, bars, "{:.0f}%", 0.8)
    ax.set_xticks(range(4), names); ax.set_ylim(0, 62); ax.set_ylabel("Still windows with visible breathing (%)")
    fig.savefig(f"{OUT}/fig6_breathing.png"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_ruview, fig_sim2real, fig_hall, fig_wall, fig_latch, fig_breath):
        f(); print("ok", f.__name__)
