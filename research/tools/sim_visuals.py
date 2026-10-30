"""Pictures of the simulator itself, drawn from the simulator's own code -> results/sim_visuals/

.venv/bin/python tools/sim_visuals.py   (run from research/)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wujud.config import LAM
from wujud.physics3d import link_paths3, diffuse_paths, link_channel, static_rest
from wujud.scenes3d import random_room, random_hall3, room_person, _standing
from wujud.sim3d import measure, _radio, FREQS56

OUT = "results/sim_visuals"
INK, ACCENT, MID, FAINT, BODY = "#1b1b1b", "#1f4e79", "#7a7a7a", "#c9c9c9", "#c0504d"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})
FS = 50.0


def demo_room(seed=7):
    rng = np.random.default_rng(seed)
    return random_room(rng, smart=True), rng


def link_of(sc):
    return sc.nodes[sc.router], sc.nodes[1]


def channel(sc, B, T, rng, n_diffuse=24):
    a, b = link_of(sc)
    paths = link_paths3(a, b, sc.W, sc.D, sc.H, sc.floor, sc.gam_walls) + diffuse_paths(a, b, sc.W, sc.D, sc.H, rng, n_diffuse)
    rest = static_rest(a, b, sc.clutter, FREQS56, rng, np.linalg.norm(b - a))
    return link_channel(a, b, paths, rest, B, T, FREQS56)


def esp32_amplitude(H, rng, remove_agc=True):
    r = _radio(2, rng)
    iq, _, _ = measure(H, rng, r["ptx"][0], r["evm"][1], r["agc"][1], 0.0, r["interf"], FS)
    amp = np.hypot(iq[..., 0].astype(float), iq[..., 1].astype(float))
    return amp / amp.mean(1, keepdims=True) if remove_agc else amp


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-9))


def box_faces(x, y, z, w, d, h):
    X = [x - w / 2, x + w / 2]; Y = [y - d / 2, y + d / 2]; Z = [z, z + h]
    v = np.array([[X[i], Y[j], Z[k]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    f = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    return [v[i] for i in f]


def room_edges(ax, W, D, H):
    for x0, y0, z0, x1, y1, z1 in [(0, 0, 0, W, 0, 0), (W, 0, 0, W, D, 0), (W, D, 0, 0, D, 0), (0, D, 0, 0, 0, 0),
                                   (0, 0, H, W, 0, H), (W, 0, H, W, D, H), (W, D, H, 0, D, H), (0, D, H, 0, 0, H),
                                   (0, 0, 0, 0, 0, H), (W, 0, 0, W, 0, H), (W, D, 0, W, D, H), (0, D, 0, 0, D, H)]:
        ax.plot([x0, x1], [y0, y1], [z0, z1], color=FAINT, lw=0.8)


def bare_3d(ax):
    ax.set_axis_off(); ax.set_box_aspect(None)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False


def fig_room_rays():
    sc, rng = demo_room()
    a, b = link_of(sc)
    fig = plt.figure(figsize=(7, 5.2)); ax = fig.add_subplot(projection="3d"); bare_3d(ax); ax.computed_zorder = False
    room_edges(ax, sc.W, sc.D, sc.H)
    for f in sc.furniture:
        ax.add_collection3d(Poly3DCollection(box_faces(f["x"], f["y"], f["z"], f["w"], f["d"], f["h"]),
                                             facecolor="#eeeeee", edgecolor=FAINT, lw=0.5, alpha=0.5))
    for p in diffuse_paths(a, b, sc.W, sc.D, sc.H, rng, 24):
        ax.plot(*p.pts.T, color=FAINT, lw=0.5)
    paths = link_paths3(a, b, sc.W, sc.D, sc.H, sc.floor, sc.gam_walls)
    for p in paths[1:]:
        ax.plot(*p.pts.T, color=MID, lw=1.1)
        ax.scatter(*p.pts[1], color=MID, s=8)
    ax.plot(*paths[0].pts.T, color=ACCENT, lw=2.4)
    chair = sc.meta["chair"]
    ax.add_collection3d(Poly3DCollection(box_faces(chair[0], chair[1], 0.4, 0.42, 0.3, 0.9), facecolor=BODY, edgecolor=BODY, alpha=0.55))
    ax.scatter(*a, marker="s", s=70, color=INK, depthshade=False, zorder=10)
    ax.scatter(*b, marker="^", s=80, color=ACCENT, depthshade=False, zorder=10)
    ax.set_xlim(0, sc.W); ax.set_ylim(0, sc.D); ax.set_zlim(0, sc.H); ax.set_box_aspect((sc.W, sc.D, sc.H))
    ax.view_init(elev=24, azim=-58)
    handles = [plt.Line2D([], [], color=ACCENT, lw=2.4), plt.Line2D([], [], color=MID, lw=1.1),
               plt.Line2D([], [], color=FAINT, lw=1), plt.Line2D([], [], marker="s", ls="", color=INK),
               plt.Line2D([], [], marker="^", ls="", color=ACCENT), plt.Line2D([], [], marker="s", ls="", color=BODY, alpha=0.6)]
    ax.legend(handles, ["line of sight", "wall, floor and ceiling reflections (image method)", "diffuse multipath",
                        "router", "ESP32", "person"], frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.95))
    fig.savefig(f"{OUT}/room_multipath.png"); plt.close(fig)


def fresnel_ellipse(a, b, n):
    d = np.linalg.norm(b[:2] - a[:2]); c = (a[:2] + b[:2]) / 2
    semi_major = (d + n * LAM / 2) / 2; semi_minor = np.sqrt(semi_major ** 2 - (d / 2) ** 2)
    th = np.linspace(0, 2 * np.pi, 200); ang = np.arctan2(*(b[:2] - a[:2])[::-1])
    x, y = semi_major * np.cos(th), semi_minor * np.sin(th)
    return c[0] + x * np.cos(ang) - y * np.sin(ang), c[1] + x * np.sin(ang) + y * np.cos(ang)


def fig_fresnel():
    sc, rng = demo_room(11)
    a, b = link_of(sc)
    mid = (a[:2] + b[:2]) / 2; u = (b[:2] - a[:2]) / np.linalg.norm(b[:2] - a[:2]); n_perp = np.array([-u[1], u[0]])
    room_room = min(mid[1], sc.D - mid[1], 1.5) - 0.15
    offsets = np.linspace(-room_room, room_room, 241)
    centres = mid + offsets[:, None] * n_perp
    B = _standing(centres[None], rng, FS)
    H = channel(sc, B, len(offsets), np.random.default_rng(3), n_diffuse=0)
    H0 = channel(sc, None, 1, np.random.default_rng(3), n_diffuse=0)
    change = 10 * np.log10(np.mean(np.abs(H) ** 2, 1) / np.mean(np.abs(H0) ** 2))
    fig, (l, r) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.3))
    l.add_patch(plt.Rectangle((0, 0), sc.W, sc.D, fill=False, ec=FAINT))
    for n, shade in ((3, "#eef2f7"), (2, "#dbe5ef"), (1, "#bcd0e3")):
        x, y = fresnel_ellipse(a, b, n); l.fill(x, y, color=shade, lw=0)
    l.plot([a[0], b[0]], [a[1], b[1]], color=ACCENT, lw=1.6)
    l.plot(*centres[::30].T, "s", color=BODY, ms=5, alpha=0.8)
    l.plot(*a[:2], "s", color=INK, ms=7); l.plot(*b[:2], "^", color=ACCENT, ms=8)
    l.set_aspect("equal"); l.set_xlim(-0.2, sc.W + 0.2); l.set_ylim(-0.2, sc.D + 0.2); l.set_xlabel("x (m)"); l.set_ylabel("y (m)")
    r1 = np.sqrt(LAM * (np.linalg.norm(b - a) / 2) ** 2 / np.linalg.norm(b - a))
    r.axvspan(-r1, r1, color="#bcd0e3", lw=0)
    r.plot(offsets, change, color=ACCENT, lw=1.4); r.axhline(0, color=FAINT, lw=0.8)
    r.set_xlabel("Distance of the person from the link (m)"); r.set_ylabel("Received power change (dB)")
    fig.savefig(f"{OUT}/fresnel_shadowing.png"); plt.close(fig)


def activity_channel(sc, act, seconds, seed):
    rng = np.random.default_rng(seed)
    T = int(seconds * FS)
    B, lab = room_person(sc, T, rng, FS, plan=[(act, seconds)])
    H = channel(sc, None if act == "away" else B, T, np.random.default_rng(seed + 1))
    return H, lab


def fig_csi():
    sc, _ = demo_room(7)
    acts = [("away", "empty room"), ("sit", "sitting still"), ("walk", "walking")]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.9), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.12, wspace=0.08))
    for j, (act, name) in enumerate(acts):
        H, _ = activity_channel(sc, act, 20, 100 + j)
        clean = db(H); clean -= clean.mean(0)
        meas = db(esp32_amplitude(H, np.random.default_rng(200 + j))); meas -= meas.mean(0)
        for i, img in enumerate((clean, meas)):
            ax = axes[i, j]
            ax.imshow(img.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-4, vmax=4,
                      extent=[0, 20, -28, 28], interpolation="nearest")
            ax.spines[:].set_visible(False)
        axes[0, j].text(10, 31, name, ha="center", va="bottom", fontsize=10)
        axes[1, j].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("physics\nsubcarrier"); axes[1, 0].set_ylabel("ESP32\nsubcarrier")
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-4, 4))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02); cb.set_label("amplitude change (dB)"); cb.outline.set_visible(False)
    fig.savefig(f"{OUT}/csi_activities.png"); plt.close(fig)


def fig_breathing():
    sc, _ = demo_room(7)
    H, lab = activity_channel(sc, "lie", 40, 300)
    clean = np.abs(H)
    swing = np.std(clean, 0) / clean.mean(0)
    t = np.arange(len(H)) / FS
    fig, (l, r) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.3))
    for k, shade in zip(np.argsort(swing)[[-1, len(swing) // 2, 0]], (ACCENT, MID, FAINT)):
        l.plot(t, 100 * (clean[:, k] / clean[:, k].mean() - 1), color=shade, lw=1.2)
    l.set_xlabel("Time (s)"); l.set_ylabel("Subcarrier amplitude (% of mean)"); l.set_xlim(0, 40)
    m = clean[:, np.argmax(swing)]
    x = m - m.mean(); n = len(x)
    f = np.fft.rfftfreq(8 * n, 1 / FS) * 60; P = np.abs(np.fft.rfft(x * np.hanning(n), 8 * n)) ** 2
    sel = (f > 4) & (f < 45)
    r.plot(f[sel], P[sel] / P[sel].max(), color=ACCENT, lw=1.3)
    r.axvline(lab["breath_hz"] * 60, color=BODY, lw=1, ls="--")
    r.set_xlabel("Breaths per minute"); r.set_ylabel("Spectrum (normalized)"); r.set_ylim(0, 1.08)
    fig.savefig(f"{OUT}/breathing.png"); plt.close(fig)


def fig_hall():
    rng = np.random.default_rng(5)
    sc = random_hall3(rng, n_nodes=12, tiered=True, placement="walls")
    occ = rng.random(len(sc.seats)) < 0.55
    fig = plt.figure(figsize=(7.2, 5.0)); ax = fig.add_subplot(projection="3d"); bare_3d(ax)
    room_edges(ax, sc.W, sc.D, sc.H)
    t = sc.tiers; rows = np.unique(sc.seat_rc[:, 0])
    for r in rows:
        y = t["y0"] + r * t["pitch"]; z = r * t["rise"]
        ax.add_collection3d(Poly3DCollection([[(0, y - t["pitch"] / 2, z), (sc.W, y - t["pitch"] / 2, z),
                                                (sc.W, y + t["pitch"] / 2, z), (0, y + t["pitch"] / 2, z)]],
                                             facecolor="#f2f2f2", edgecolor="#e2e2e2", lw=0.4))
    for i, j in sc.links:
        ax.plot(*np.stack([sc.nodes[i], sc.nodes[j]]).T, color=ACCENT, lw=0.35, alpha=0.35)
    s = sc.seats
    ax.scatter(s[~occ, 0], s[~occ, 1], s[~occ, 2] + 0.45, s=4, color=FAINT, depthshade=False)
    ax.scatter(s[occ, 0], s[occ, 1], s[occ, 2] + 0.9, s=9, color=BODY, depthshade=False)
    ax.scatter(*sc.nodes.T, marker="^", s=45, color=ACCENT, depthshade=False)
    ax.set_xlim(0, sc.W); ax.set_ylim(0, sc.D); ax.set_zlim(0, sc.H); ax.set_box_aspect((sc.W, sc.D, sc.H * 1.3))
    ax.view_init(elev=28, azim=-115)
    handles = [plt.Line2D([], [], marker="^", ls="", color=ACCENT), plt.Line2D([], [], color=ACCENT, lw=0.8, alpha=0.5),
               plt.Line2D([], [], marker="o", ls="", color=BODY, ms=4), plt.Line2D([], [], marker="o", ls="", color=FAINT, ms=3)]
    ax.legend(handles, ["ESP32 on the walls", f"{len(sc.links)} radio links", f"{occ.sum()} students", "empty seats"],
              frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.savefig(f"{OUT}/lecture_hall.png"); plt.close(fig)


def walk_across_link(sc, seconds, speed=0.7):
    a, b = link_of(sc)
    u = (b[:2] - a[:2]) / np.linalg.norm(b[:2] - a[:2]); n = np.array([-u[1], u[0]])
    reach = min((a[1] + b[1]) / 2, sc.D - (a[1] + b[1]) / 2) - 0.35
    stops = [a[:2] + f * (b[:2] - a[:2]) + side * reach * n for f, side in
             ((0.35, -1), (0.4, 1), (0.55, -1), (0.6, 1), (0.75, -1), (0.5, 1), (0.3, -1))]
    T = int(seconds * FS); path = [stops[0]]; k = 1
    while len(path) < T:
        step = stops[k % len(stops)] - path[-1]; dist = np.linalg.norm(step)
        if dist < speed / FS:
            k += 1; continue
        path.append(path[-1] + step / dist * speed / FS)
    return np.array(path)


def walk_animation(seconds=16, fps=12):
    sc, rng = demo_room(7)
    xy = walk_across_link(sc, seconds)
    H = channel(sc, _standing(xy[None], rng, FS, walking=True), len(xy), np.random.default_rng(400))
    amp = db(esp32_amplitude(H, np.random.default_rng(401))); amp -= np.median(amp, 0)
    a, b = link_of(sc); T = len(H)
    fig, (l, r) = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.25))
    fig.subplots_adjust(bottom=0.16)
    l.add_patch(plt.Rectangle((0, 0), sc.W, sc.D, fill=False, ec=FAINT))
    for f in sc.furniture:
        l.add_patch(plt.Rectangle((f["x"] - f["w"] / 2, f["y"] - f["d"] / 2), f["w"], f["d"], color="#eeeeee", lw=0))
    x, y = fresnel_ellipse(a, b, 1); l.fill(x, y, color="#dbe5ef", lw=0)
    link, = l.plot([a[0], b[0]], [a[1], b[1]], color=ACCENT, lw=1.6)
    l.plot(*a[:2], "s", color=INK, ms=7); l.plot(*b[:2], "^", color=ACCENT, ms=8)
    trail, = l.plot([], [], color=BODY, lw=1, alpha=0.4)
    person, = l.plot([], [], "o", color=BODY, ms=11)
    l.set_aspect("equal"); l.set_xlim(-0.2, sc.W + 0.2); l.set_ylim(-0.2, sc.D + 0.2); l.axis("off")
    img = r.imshow(np.full((amp.shape[1], T), np.nan), aspect="auto", origin="lower", cmap="RdBu_r", vmin=-5, vmax=5,
                   extent=[0, seconds, -28, 28], interpolation="nearest")
    cursor = r.axvline(0, color=INK, lw=0.8)
    r.set_xlabel("Time (s)"); r.set_ylabel("Subcarrier"); r.spines[:].set_visible(False)
    frames = np.arange(0, T, int(FS / fps))
    shown = np.full((amp.shape[1], T), np.nan)
    swing = np.abs(np.diff(amp, axis=0, prepend=amp[:1])).mean(1)
    swing = np.convolve(swing, np.ones(10) / 10, "same"); swing /= np.percentile(swing, 98)

    def draw(i):
        shown[:, :i + 1] = amp[:i + 1].T
        img.set_data(shown); cursor.set_xdata([i / FS, i / FS])
        person.set_data([xy[i, 0]], [xy[i, 1]]); trail.set_data(xy[max(0, i - 150):i + 1, 0], xy[max(0, i - 150):i + 1, 1])
        link.set_linewidth(1.2 + 3.5 * min(1.0, swing[i]))
        return img, cursor, person, trail, link

    anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(f"{OUT}/walking.gif", writer=animation.PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_room_rays, fig_fresnel, fig_csi, fig_breathing, fig_hall, walk_animation):
        f(); print("ok", f.__name__, flush=True)
