"""The movement network (tools/movement_net.py, learnt from simulated rooms only) on a real recording:
what it says second by second, as stretches, and how many bows and prostrations it counts.

    .venv/bin/python -I tools/real_movements.py cache/movements/movement_net.pt rec/2026-11-01-prayers.txt
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from movement_net import BINS, HZ, MOVEMENTS, MovementNet, link_inputs

SIGNAL_POWER = 2.0
ESP32_BINS = 64


def read(path):
    """Per board-to-board link: arrival times, RSSI and |CSI| per subcarrier."""
    links = defaultdict(lambda: ([], [], []))
    for row in open(path):
        parts = row.rstrip("\n").split(",")
        if len(parts) < 6 or parts[2] != "P" or "@" not in parts[1]:
            continue
        raw = bytes.fromhex(parts[5][:256])
        if len(raw) != 128:
            continue
        iq = np.frombuffer(raw, dtype=np.int8).astype(np.float32).reshape(ESP32_BINS, 2)
        t, rssi, amplitude = links[parts[1]]
        t.append(float(parts[0]))
        rssi.append(float(parts[3]))
        amplitude.append(np.hypot(iq[:, 0], iq[:, 1]))
    return {name: (np.array(t), np.array(r), np.array(a)) for name, (t, r, a) in links.items() if len(t) > 1000}


def steps(links):
    """Ten-per-second steps x links x 33, as crates/app/examples/export_movements.rs."""
    start = max(t[0] for t, _, _ in links.values())
    end = min(t[-1] for t, _, _ in links.values())
    count = int((end - start) * HZ)
    out = np.zeros((count, len(links), BINS + 1), dtype=np.float32)
    for l, (t, rssi, amplitude) in enumerate(links.values()):
        power = (amplitude ** 2).mean(0)
        order = np.arange(ESP32_BINS)
        if power[30] <= 0.5 and power[16] > 0.5:
            order = np.roll(order, -ESP32_BINS // 2)
        signal = [k for k in order if power[k] > SIGNAL_POWER]
        db = 20 * np.log10(amplitude[:, signal] + 1)
        index = ((t - start) * HZ).astype(int)
        last = None
        for s in range(count):
            mask = index == s
            if mask.any():
                level = db[mask].mean(0)
                shape = np.interp(np.linspace(0, 1, BINS), np.linspace(0, 1, len(signal)), level - level.mean())
                last = np.concatenate([shape, [rssi[mask].mean()]])
            if last is not None:
                out[s, l] = last
    return out, start


def main():
    weights, recording = sys.argv[1], sys.argv[2]
    saved = torch.load(weights, map_location="cpu", weights_only=False)
    dilations, state, config = saved["dilations"], saved["state"], saved["config"]
    model = MovementNet(state["encoder.0.weight"].shape[1], config["width"], dilations)
    model.load_state_dict(state)
    model.eval()
    links = read(recording)
    print(f"{len(links)} links: " + ", ".join(n.split("@")[1] for n in links))
    raw, start = steps(links)
    inputs = torch.tensor(link_inputs(raw, config["scale"], np.float32)).permute(1, 0, 2)[None]
    with torch.no_grad():
        logits = model(inputs)[0].numpy()
    seconds = logits.shape[1] // HZ
    said = logits[:, :seconds * HZ].reshape(len(MOVEMENTS), seconds, HZ).mean(-1).argmax(0)
    stretches, begin = [], 0
    for s in range(1, seconds + 1):
        if s == seconds or said[s] != said[begin]:
            stretches.append((begin, s, MOVEMENTS[said[begin]]))
            begin = s
    for a, b, what in stretches:
        if what != "still" or b - a >= 30:
            print(f"{a // 60:3d}:{a % 60:02d} {b - a:4d} s  {what}")
    counts = defaultdict(int)
    for a, b, what in stretches:
        counts[what] += 1
    total = {m: int((said == i).sum()) for i, m in enumerate(MOVEMENTS)}
    print(f"{seconds} s; stretches: {dict(counts)}; seconds: {total}")


if __name__ == "__main__":
    main()
