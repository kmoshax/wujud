"""Paper-style figures of the deep sim-to-real study: results/deep/summary.json -> results/deep/fig*.png

.venv/bin/python tools/deep_figures.py   (run from research/)
"""
import json, os, sys
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "results/deep"
OURS, THEIRS, GREY, LIGHT = "#1f4e79", "#b0b0b0", "#555555", "#e8e8e8"
EMPTY, STILL = "#7a7a7a", "#1f4e79"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 10.5,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
BASELINE = "baseline (hand features + GBDT, sim only)"
NOISE_LEVEL = "reference: receiver noise level only (no training)"


def pretty(name):
    if name == BASELINE:
        return "Hand features + GBDT (baseline)"
    if name == NOISE_LEVEL:
        return "Receiver noise level only"
    parts = name.split("_")
    encoder = {"cnn2d": "2-D CNN", "cnn1d": "1-D CNN", "transformer": "Transformer", "spectral": "Spectral CNN"}.get(parts[0], parts[0])
    extra = {"plain": "plain", "aug": "augmented", "dann03": "DANN 0.3", "dann1": "DANN 1.0", "ssl": "contrastive pretrain",
             "phase": "+phase", "ampn": "noise-normalised", "ampboth": "raw + noise-normalised", "aug05": "augmented x0.5", "ens3": "3-seed ensemble", "rssi": "+RSSI", "static": "+static profile", "all": "+phase+RSSI+static"}
    return f"{encoder}, " + ", ".join(extra.get(p, p) for p in parts[1:])


def fig_auc_forest(summary):
    names = [BASELINE] + sorted((n for n in summary["variants"] if n != BASELINE),
                                key=lambda n: summary["variants"][n]["selection"]["still_vs_empty_auc"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.26 * len(names) + 0.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    y = np.arange(len(names))
    for ax, split, xlabel in zip(axes, ("selection", "holdout"), ("Still vs. empty AUC, selection", "Still vs. empty AUC, holdout")):
        for i, n in enumerate(names):
            m = summary["variants"][n][split]
            lo, hi = m["ci"]["window"]
            color = THEIRS if n == BASELINE else (OURS if n == summary["chosen_on_selection"] else GREY)
            ax.plot([lo, hi], [i, i], color=color, lw=1.4)
            ax.plot(m["still_vs_empty_auc"], i, "o", color=color, ms=4.5)
            ax.plot(m["still_vs_empty_auc_recording"], i, "x", color=color, ms=4.5)
        ax.axvline(0.5, color=GREY, lw=0.8, ls="--")
        ax.axvline(summary["variants"][BASELINE][split]["still_vs_empty_auc"], color=THEIRS, lw=0.8, ls=":")
        ax.set_xlim(0.0, 1.0); ax.set_xlabel(xlabel)
    axes[0].set_yticks(y, [pretty(n) for n in names])
    axes[1].plot([], [], "o", color=GREY, label="window AUC (95 % CI over recordings)")
    axes[1].plot([], [], "x", color=GREY, label="recording AUC (first minute)")
    fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.6, -0.03))
    fig.subplots_adjust(bottom=0.09)
    fig.savefig(f"{OUT}/fig1_auc_variants.png"); plt.close(fig)


def fig_per_device(summary):
    chosen = summary["chosen_on_selection"]
    rows = []
    for split in ("selection", "holdout"):
        base = summary["variants"][BASELINE][split]["per_device"]
        mine = summary["variants"][chosen][split]["per_device"]
        for d in base:
            rows.append((f"{split[:3]}. {d[-6:]}", base[d]["auc"], mine[d]["auc"], base[d]["empty"], base[d]["still"]))
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w / 2, [r[1] for r in rows], w, color=THEIRS, label="Hand features + GBDT")
    ax.bar(x + w / 2, [r[2] for r in rows], w, color=OURS, label=pretty(chosen))
    ax.axhline(0.5, color=GREY, lw=0.8, ls="--")
    ax.set_xticks(x, [f"{r[0]}\n{r[3]} e / {r[4]} s" for r in rows], fontsize=7.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Window AUC within device")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.15))
    fig.savefig(f"{OUT}/fig2_per_device.png"); plt.close(fig)


def fig_time_of_day(summary):
    """recording mean still score vs local hour: shows whether a score tracks the session's time instead of the person"""
    from deep.data import RealRecordings
    real = RealRecordings()
    chosen = summary["chosen_on_selection"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=False, gridspec_kw=dict(wspace=0.3))
    for ax, (label, path, key) in zip(axes, [("Hand features + GBDT", "cache/deep/baseline_scores.npz", "holdout/{}/still"),
                                              (pretty(chosen), f"cache/deep/runs/{chosen}.holdout.npz", "{}/still")]):
        z = np.load(path)
        for r in real.recordings("holdout"):
            m = real.meta[r]
            if m["label"] == 2:
                continue
            hour = datetime.fromisoformat(m["collected_at"].replace("Z", "+00:00")).hour if m.get("collected_at") else np.nan
            score = z[key.format(m["file"])][:9].mean()
            ax.plot(hour, score, "o" if m["label"] else "s", color=STILL if m["label"] else EMPTY, ms=5, mfc="none" if m["long"] else None)
        ax.set_xlim(-0.5, 23.5); ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xlabel("Hour the recording started")
        ax.set_ylabel(f"Mean still score, first minute\n{label}", fontsize=8.5)
    axes[0].plot([], [], "s", color=EMPTY, label="empty"); axes[0].plot([], [], "o", color=STILL, label="still person")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(f"{OUT}/fig3_time_of_day.png"); plt.close(fig)


if __name__ == "__main__":
    summary = json.load(open(f"{OUT}/summary.json"))
    fig_auc_forest(summary)
    fig_per_device(summary)
    fig_time_of_day(summary)
