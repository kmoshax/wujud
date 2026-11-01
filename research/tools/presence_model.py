"""Empty / someone still / someone moving from ten seconds of one link, with nothing learnt about the
room it runs in. Trained on ESPectre's public ESP32 recordings (three rooms, five chips) and tested
room by room, then on our own guided recording, which it never sees in training.

    .venv/bin/python -I tools/presence_model.py cache/espectre/repo/data rec/guided_room_20261101_2120
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

LABELS = {"empty": 0, "static_presence": 1, "motion": 2}
NAMES = ["empty", "still", "moving"]
BIN = np.arange(64) - 32
SIGNAL = (np.abs(BIN) <= 26) & (BIN != 0)
WINDOW_S, HOP_S, GRID_HZ = 10.0, 2.0, 20.0
BANDS = {"breath": (0.1, 0.6), "micro": (0.6, 3.0), "noise": (4.0, 9.5)}


def amplitudes(iq):
    pairs = iq.astype(np.float32).reshape(len(iq), 64, 2)
    return np.hypot(pairs[..., 0], pairs[..., 1])[:, SIGNAL]


def band_power(series, lo, hi):
    spectrum = np.abs(np.fft.rfft(series - series.mean(axis=0), axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(series), 1 / GRID_HZ)
    return spectrum[(freqs >= lo) & (freqs < hi)].sum(axis=0)


def window_features(t, amp, rssi):
    """t: seconds; amp: packets x 52 amplitudes; rssi: dBm per packet."""
    db = 20 * np.log10(amp + 1)
    shape = db - db.mean(axis=1, keepdims=True)
    seconds = np.floor(t - t[0]).astype(int)
    per_second = [shape[seconds == s].mean(axis=0) for s in np.unique(seconds) if (seconds == s).sum() >= 10]
    drift = [np.abs(b - a).mean() for a, b in zip(per_second, per_second[1:])] or [0.0]
    turbulence = amp.std(axis=1) / np.maximum(amp.mean(axis=1), 1e-6)
    turb_lag1 = np.corrcoef(turbulence[:-1], turbulence[1:])[0, 1] if turbulence.std() > 0 else 0.0
    temporal_cv = amp.std(axis=0) / np.maximum(amp.mean(axis=0), 1e-6)
    grid = np.arange(t[0], t[-1], 1 / GRID_HZ)
    resampled = np.stack([np.interp(grid, t, shape[:, k]) for k in range(shape.shape[1])], axis=1)
    powers = {name: band_power(resampled, lo, hi) for name, (lo, hi) in BANDS.items()}
    centred = resampled - resampled.mean(axis=0)
    eig = np.linalg.eigvalsh(np.cov(centred.T))[::-1]
    eig_share = eig / max(eig.sum(), 1e-9)
    pc1 = centred @ np.linalg.eigh(np.cov(centred.T))[1][:, -1]
    pc_bands = {name: band_power(pc1[:, None], lo, hi)[0] for name, (lo, hi) in BANDS.items()}
    noise = np.maximum(powers["noise"], 1e-9)
    return [
        np.median(drift), np.percentile(drift, 90), np.max(drift),
        turbulence.std() / max(turbulence.mean(), 1e-6), np.nan_to_num(turb_lag1),
        np.median(temporal_cv), np.percentile(temporal_cv, 90),
        np.median(powers["breath"] / noise), np.median(powers["micro"] / noise),
        np.percentile(powers["micro"] / noise, 90),
        pc_bands["breath"] / max(pc_bands["noise"], 1e-9), pc_bands["micro"] / max(pc_bands["noise"], 1e-9),
        eig_share[0], eig_share[:3].sum(),
        np.std(rssi), np.ptp(rssi),
        shape.mean(axis=0).std(),
    ]


def windows(t, iq, rssi):
    amp = amplitudes(iq)
    start = t[0]
    while start + WINDOW_S <= t[-1]:
        keep = (t >= start) & (t < start + WINDOW_S)
        if keep.sum() >= WINDOW_S * 30:
            yield start, window_features(t[keep], amp[keep], rssi[keep])
        start += HOP_S


def espectre(root):
    info = json.load(open(os.path.join(root, "dataset_info.json")))
    environment = {}
    for entries in info["files"].values():
        for entry in entries:
            environment[os.path.basename(entry["filename"])] = (entry.get("environment"), entry.get("dataset_role"))
    rows = []
    for label in LABELS:
        for path in sorted(glob.glob(os.path.join(root, label, "*.npz"))):
            room, role = environment.get(os.path.basename(path), (None, None))
            if role == "exclude" or room is None:
                continue
            z = np.load(path, allow_pickle=False)
            t = z["device_ticks_us"].astype(np.float64) / 1e6
            order = np.argsort(t)
            for _, features in windows(t[order], z["csi_data"][order], z["rssi_dbm"][order].astype(float)):
                rows.append((features, LABELS[label], room, str(z["chip"])))
    return rows


def guided(base):
    """Our own recording: every link separately, labelled from the guided plan (5 s after each step starts)."""
    plan = json.load(open(base + "_plan.json"))
    links = {}
    with open(base + ".csv") as f:
        for line in f:
            parts = line.rstrip().split(",", 5)
            if len(parts) < 6 or parts[2] != "P" or len(parts[5]) != 256:
                continue
            t = float(parts[0])
            if t < plan[0]["start"] or t > plan[-1]["end"]:
                continue
            links.setdefault(parts[1], []).append((t, float(parts[3]), bytes.fromhex(parts[5])))
    rows = []
    for link, packets in links.items():
        t = np.array([p[0] for p in packets])
        rssi = np.array([p[1] for p in packets])
        iq = np.frombuffer(b"".join(p[2] for p in packets), dtype=np.int8).reshape(len(packets), 128)
        for start, features in windows(t, iq, rssi):
            middle = start + WINDOW_S / 2
            step = next((s for s in plan if s["start"] + 5 <= start and start + WINDOW_S <= s["end"]), None)
            if step:
                label = {"empty": 0, "still": 1, "moving": 2}[step["label"]]
                rows.append((features, label, step["english"], link, middle))
    return rows


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


def confusion(truth, said):
    m = np.zeros((3, 3), int)
    for a, b in zip(truth, said):
        m[a, b] += 1
    return m


def report(title, truth, said):
    m = confusion(truth, said)
    recall = [m[i, i] / max(m[i].sum(), 1) for i in range(3)]
    print(f"{title}: accuracy {np.trace(m) / max(m.sum(), 1) * 100:.1f}%  " +
          "  ".join(f"{NAMES[i]} {recall[i] * 100:.0f}%" for i in range(3)) + f"  (n={m.sum()})")
    return m


def main():
    data_root, guided_base = sys.argv[1], sys.argv[2]
    rows = espectre(data_root)
    X = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    rooms = np.array([r[2] for r in rows])
    print(f"ESPectre windows: {len(rows)}  rooms: {sorted(set(rooms))}")
    for room in sorted(set(rooms)):
        train, test = rooms != room, rooms == room
        clf = model().fit(X[train], y[train])
        report(f"  held-out room {room:12s}", y[test], clf.predict(X[test]))
    clf = model().fit(X, y)
    ours = guided(guided_base)
    Xg = np.array([r[0] for r in ours]); yg = np.array([r[1] for r in ours])
    report("our room, every link alone", yg, clf.predict(Xg))
    # all links together: average their probabilities per window time
    probs = clf.predict_proba(Xg)
    by_time = {}
    for (features, label, step, link, middle), p in zip(ours, probs):
        by_time.setdefault((round(middle), label, step), []).append(p)
    truth = [k[1] for k in by_time]; said = [int(np.argmax(np.mean(v, axis=0))) for v in by_time.values()]
    report("our room, links together", truth, said)
    for step in dict.fromkeys(k[2] for k in by_time):
        keys = [k for k in by_time if k[2] == step]
        right = sum(int(np.argmax(np.mean(by_time[k], axis=0))) == k[1] for k in keys)
        print(f"    {step[:34]:34s} {right}/{len(keys)}")
    np.save(os.path.join(os.path.dirname(guided_base), "presence_features.npy"),
            {"X": X, "y": y, "rooms": rooms, "Xg": Xg, "yg": yg}, allow_pickle=True)


if __name__ == "__main__":
    main()
