"""Paper-style figures and the results table of the limits study (crates/study, bin limits_study).

Reads research/results/limits/config_*.json, writes research/results/limits/fig_*.png and limits_table.md.
.venv/bin/python tools/limits_figures.py   (run from research/)
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "results/limits"
OURS, GREY, LIGHT, ACCENT = "#1f4e79", "#555555", "#b0b0b0", "#c0504d"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
FACTOR_ORDER = ["baseline", "band", "bandwidth", "hopping", "packet_rate", "links", "placement", "combination"]
FACTOR_LABEL = {"baseline": "Baseline", "band": "Band", "bandwidth": "Bandwidth", "hopping": "Hopping",
                "packet_rate": "Packet rate", "links": "Links", "placement": "Placement", "combination": "Combinations"}
LABEL = {
    "baseline": "Baseline (2.4 GHz, 52 sc, 50 pkt/s, 1 link)",
    "5ghz_conservative": "5 GHz, conservative", "5ghz_optimistic": "5 GHz, optimistic",
    "ht40_114": "HT40, 114 sc", "he20_242": "HE20, 242 sc (ESP32-C6)",
    "hop3_2g4": "Hop 3 ch. at 2.4 GHz", "hop4_5g": "Hop 4 ch. at 5 GHz",
    "rate25": "25 pkt/s", "rate100": "100 pkt/s", "rate200": "200 pkt/s",
    "links2": "2 boards (2 links)", "links3": "3 boards (3 links)", "mesh2": "2 boards + mesh (3 links)",
    "smart": "Smart placement",
    "smart_rate100": "Smart + 100 pkt/s", "smart_mesh2": "Smart + 2-board mesh", "5ghz_smart": "5 GHz + smart",
    "5ghz_he20": "5 GHz HE20 (ESP32-C5)", "5ghz_ht40_rate100": "5 GHz HT40 + 100 pkt/s",
    "best_2g4": "2.4 GHz HT40 + 100 pkt/s + smart + mesh",
    "best_conservative": "5 GHz HT40 + 100 pkt/s + smart + mesh (cons.)",
    "best_optimistic": "5 GHz HT40 + 100 pkt/s + smart + mesh (opt.)",
}


def load():
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(DIR, "config_*.json"))]
    order = {name: i for i, name in enumerate(LABEL)}
    return sorted(rows, key=lambda r: (FACTOR_ORDER.index(r["config"]["factor"]), order.get(r["config"]["name"], 99)))


def links(r):
    c = r["config"]
    return c["boards"] + (c["boards"] * (c["boards"] - 1) // 2 if c["mesh"] else 0)


def est(r, *path):
    v = r
    for p in path:
        v = v[p]
    return v["value"], v["se"]


def dot_panel(ax, rows, values, errors, xlabel, reference, xlim):
    y = np.arange(len(rows))[::-1]
    colors = [GREY if r["config"]["factor"] == "baseline" else OURS for r in rows]
    ax.axvline(reference, color=LIGHT, lw=0.8, ls="--", zorder=0)
    ax.errorbar(values, y, xerr=1.96 * np.asarray(errors), fmt="none", ecolor=LIGHT, elinewidth=1.2, capsize=0, zorder=1)
    ax.scatter(values, y, s=18, c=colors, zorder=2)
    ax.set_xlabel(xlabel); ax.set_xlim(*xlim); ax.set_ylim(-0.7, len(rows) - 0.3)
    group_edges(ax, rows, y)
    return y


def group_edges(ax, rows, y):
    for i in range(1, len(rows)):
        if rows[i]["config"]["factor"] != rows[i - 1]["config"]["factor"]:
            ax.axhline((y[i] + y[i - 1]) / 2, color=LIGHT, lw=0.5)


def fig_detection(rows):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 5.6), sharey=True, gridspec_kw=dict(wspace=0.12))
    still = [est(r, "windows", "still_vs_empty_auc") for r in rows]
    latch = [est(r, "latch", "distance_auc_still_vs_empty") for r in rows]
    vis = [est(r, "windows", "breathing_visible") for r in rows]
    base = rows[0]
    y = dot_panel(a, rows, [v for v, _ in still], [e for _, e in still], "Still vs. empty AUC (per 20-s window)",
                  est(base, "windows", "still_vs_empty_auc")[0], (0.5, 1.0))
    dot_panel(b, rows, [v for v, _ in latch], [e for _, e in latch], "Latch distance AUC",
              est(base, "latch", "distance_auc_still_vs_empty")[0], (0.5, 1.0))
    dot_panel(c, rows, [100 * v for v, _ in vis], [100 * e for _, e in vis], "Breathing visible (% of still windows)",
              100 * est(base, "windows", "breathing_visible")[0], (0, 100))
    a.set_yticks(y, [LABEL.get(r["config"]["name"], r["config"]["name"]) for r in rows])
    fig.savefig(f"{DIR}/fig_limits_detection.png"); plt.close(fig)


SHORT = {"5ghz_conservative": "5 GHz cons.", "5ghz_optimistic": "5 GHz opt.", "ht40_114": "HT40", "he20_242": "HE20",
         "hop3_2g4": "Hop 3 ch.", "hop4_5g": "Hop 4 ch. 5 GHz", "rate25": "25 pkt/s", "rate100": "100 pkt/s",
         "rate200": "200 pkt/s", "links2": "2 links", "links3": "3 boards", "mesh2": "2 boards + mesh", "smart": "Smart",
         "smart_rate100": "Smart + 100 pkt/s", "smart_mesh2": "Smart + mesh", "5ghz_smart": "5 GHz + smart",
         "5ghz_he20": "5 GHz HE20", "5ghz_ht40_rate100": "5 GHz HT40 + 100 pkt/s",
         "best_2g4": "HT40 + 100 pkt/s + smart + mesh", "best_conservative": "same at 5 GHz (cons.)",
         "best_optimistic": "same at 5 GHz (opt.)"}


def spread_labels(points, gap):
    placed = []
    for x, y, text in sorted(points, key=lambda p: (p[0], p[1])):
        same = [q for q in placed if q[0] == x]
        label_y = max(y, same[-1][3] + gap) if same else y
        placed.append((x, y, text, label_y))
    return placed


def fig_cost(rows):
    base_auc = est(rows[0], "windows", "still_vs_empty_auc")[0]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    points = []
    for r in rows[1:]:
        cost = r["config"]["extra_cost_usd"]; v, e = est(r, "windows", "still_vs_empty_auc")
        color = OURS if r["config"]["factor"] != "combination" else ACCENT
        ax.errorbar(cost, v - base_auc, yerr=1.96 * e, fmt="o", ms=4, color=color, ecolor=LIGHT, elinewidth=1.0)
        points.append((cost, v - base_auc, SHORT.get(r["config"]["name"], r["config"]["name"])))
    for x, y, text, label_y in spread_labels(points, 0.016):
        left = x == 0
        ax.annotate(text, (x, y), xytext=(x - 0.35 if left else x + 0.35, label_y), fontsize=7, va="center",
                    ha="right" if left else "left",
                    arrowprops=dict(arrowstyle="-", color=LIGHT, lw=0.5) if abs(label_y - y) > 0.004 else None)
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.set_xlim(-5.5, 19.5); ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Extra hardware cost over the baseline (USD)"); ax.set_ylabel(r"$\Delta$ still vs. empty AUC")
    ax.scatter([], [], color=OURS, s=14, label="One factor changed")
    ax.scatter([], [], color=ACCENT, s=14, label="Combination")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(f"{DIR}/fig_limits_cost.png"); plt.close(fig)


def fig_packet_rate(rows):
    by = {r["config"]["name"]: r for r in rows}
    names = [n for n in ("rate25", "baseline", "rate100", "rate200") if n in by]
    if len(names) < 3:
        return
    rate = [by[n]["config"]["packets_per_second"] for n in names]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw=dict(wspace=0.35))
    for ax, path, scale, label in ((a, ("windows", "still_vs_empty_auc"), 1, "Still vs. empty AUC"),
                                   (b, ("windows", "breathing_visible"), 100, "Breathing visible (%)")):
        v = np.array([est(by[n], *path)[0] for n in names]) * scale
        e = np.array([est(by[n], *path)[1] for n in names]) * scale
        ax.errorbar(rate, v, yerr=1.96 * e, fmt="-o", ms=4, color=OURS, ecolor=LIGHT)
        ax.set_xscale("log", base=2); ax.set_xticks(rate, [f"{x:.0f}" for x in rate])
        ax.set_xlabel("Packets per second"); ax.set_ylabel(label)
    fig.savefig(f"{DIR}/fig_limits_packet_rate.png"); plt.close(fig)


def table(rows):
    base = est(rows[0], "windows", "still_vs_empty_auc")[0]
    head = ("| configuration | links | extra cost (USD) | still-vs-empty AUC | latch distance AUC | breathing visible | "
            "rate error (bpm, median) | motion recall / false alarm | ΔAUC per $10 |")
    lines = [head, "|" + "---|" * 9]
    for r in rows:
        c, w, l = r["config"], r["windows"], r["latch"]
        auc, se = est(r, "windows", "still_vs_empty_auc")
        cost = c["extra_cost_usd"]
        per10 = "free" if cost == 0 and c["name"] != "baseline" else ("–" if cost == 0 else f"{10 * (auc - base) / cost:+.3f}")
        lines.append(
            f"| {c['name']} | {links(r)} | {cost:.0f} | {auc:.3f} ± {se:.3f} | "
            f"{l['distance_auc_still_vs_empty']['value']:.3f} ± {l['distance_auc_still_vs_empty']['se']:.3f} | "
            f"{100 * w['breathing_visible']['value']:.1f} ± {100 * w['breathing_visible']['se']:.1f} % | "
            f"{w['rate_error_median_bpm']['value']:.2f} ± {w['rate_error_median_bpm']['se']:.2f} | "
            f"{w['motion_recall']['value']:.3f} / {w['motion_false_alarm']['value']:.3f} | {per10} |")
    open(f"{DIR}/limits_table.md", "w").write("\n".join(lines) + "\n")
    json.dump([dict(name=r["config"]["name"], factor=r["config"]["factor"], links=links(r),
                    extra_cost_usd=r["config"]["extra_cost_usd"],
                    still_vs_empty_auc=r["windows"]["still_vs_empty_auc"],
                    latch_distance_auc=r["latch"]["distance_auc_still_vs_empty"],
                    breathing_visible=r["windows"]["breathing_visible"],
                    rate_error_median_bpm=r["windows"]["rate_error_median_bpm"],
                    motion_recall=r["windows"]["motion_recall"], motion_false_alarm=r["windows"]["motion_false_alarm"])
               for r in rows], open(f"{DIR}/limits_summary.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    rows = load()
    assert rows and rows[0]["config"]["name"] == "baseline", "run limits_study first"
    fig_detection(rows); fig_cost(rows); fig_packet_rate(rows); table(rows)
