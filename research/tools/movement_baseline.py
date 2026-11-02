"""The baseline for tools/movement_net.py: per second, how much the links moved (mean |change| over
0.3 s and 1 s, the RSSI's departure from its mean), pooled over links (mean, max) and over 1, 3 and 7
seconds around it, into gradient-boosted trees. Same rooms, same split, same per-second scoring.

    .venv/bin/python -I tools/movement_baseline.py cache/movements/part*.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from movement_net import BINS, HZ, MOVEMENTS, load, report


def second_features(room):
    x = room["inputs"].astype(np.float32)
    energy = np.stack([np.abs(x[..., :BINS]).mean(-1), np.abs(x[..., BINS:2 * BINS]).mean(-1),
                       np.abs(x[..., 2 * BINS:3 * BINS]).mean(-1), np.abs(x[..., 3 * BINS]),
                       np.abs(x[..., 3 * BINS + 1])], -1)
    pooled = np.concatenate([energy.mean(1), energy.max(1)], -1)
    seconds = len(pooled) // HZ
    per_second = pooled[:seconds * HZ].reshape(seconds, HZ, -1).mean(1)
    columns = [per_second]
    for half in (1, 3):
        padded = np.pad(per_second, ((half, half), (0, 0)), mode="edge")
        columns.append(np.stack([padded[i:i + 2 * half + 1].mean(0) for i in range(seconds)]))
        columns.append(np.stack([padded[i:i + 2 * half + 1].std(0) for i in range(seconds)]))
    return np.concatenate(columns, -1), room["movement"][HZ // 2::HZ][:seconds]


def main():
    rooms = load(sys.argv[1:], scale=1)
    split = {"train": [r for r in rooms if r["seed"] <= 1600],
             "validation": [r for r in rooms if 1600 < r["seed"] <= 1800],
             "test": [r for r in rooms if r["seed"] > 1800]}
    data = {k: [second_features(r) for r in v] for k, v in split.items()}
    stack = lambda part: (np.concatenate([f for f, _ in data[part]]), np.concatenate([y for _, y in data[part]]))
    x, y = stack("train")
    model = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=0).fit(x, y)
    for part in ("validation", "test"):
        xp, yp = stack(part)
        matrix = np.zeros((len(MOVEMENTS), len(MOVEMENTS)), dtype=np.int64)
        np.add.at(matrix, (yp, model.predict(xp)), 1)
        print(report(matrix, f"baseline (motion energy + trees), {part} rooms")[0])


if __name__ == "__main__":
    main()
