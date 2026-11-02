"""Which body movement is under way (still, walk, wave, bow, prostrate, sit down, stand up), every
100 ms, from every link's ten-per-second readings; learnt from simulated rooms only
(crates/app/examples/export_movements.rs) and measured on rooms never seen while learning.

Each link's input is relative to itself: the spectrum shape's change over 0.3 s and 1 s, its departure
from its own 30-second mean, and the RSSI's departure from its 30-second mean and from 1 s ago. With
`scale=1` every link is also divided by its own typical 1 s change over the last 30 s, so a link that
sees the body strongly and one that barely does read alike. A dilated 1-D CNN reads each link
(centred, about +-6 s), the links are pooled (mean and max) every step, and a temporal head scores
the movement.

Rooms are split by seed: train, validation (choosing the epoch and settings) and test (read once).
Rooms with seeds above 5000 (export_movements with harsh 2-3.5) are a stress test, never learnt from.

    .venv/bin/python -I tools/movement_net.py cache/movements/part*.json [key=value ...]
"""
import json
import sys

import numpy as np
import torch
from torch import nn

MOVEMENTS = ["still", "walk", "wave", "bow", "prostrate", "sit_down", "stand_up"]
HZ, MEAN_S, BINS = 10, 30, 32
STRESS_FROM = 5000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG = {
    "scale": 1, "width": 64, "dilations": "1,2,4,8,16", "epochs": 24, "window": 256, "batch": 48,
    "train_until": 1600, "validate_until": 1800, "test": 0, "out": "cache/movements/movement_net.pt",
    "lr": 2e-3, "seed": 0, "device": DEVICE,
}


def rolling_mean(x, n):
    """Causal mean over the last n steps along axis 0."""
    c = np.cumsum(np.concatenate([np.zeros_like(x[:1]), x]), axis=0, dtype=np.float64)
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - n, 0)
    return ((c[idx] - c[lo]) / (idx - lo)[:, None, None]).astype(np.float32)


def lagged(x, n):
    return np.concatenate([np.repeat(x[:1], n, axis=0), x[:-n]], axis=0)


def link_inputs(raw, scale, dtype=np.float16):
    """raw: steps x links x 33 -> steps x links x 98, all relative to the link."""
    shape, rssi = raw[..., :BINS], raw[..., BINS:]
    short = shape - lagged(shape, 3)
    second = shape - lagged(shape, HZ)
    away = shape - rolling_mean(shape, MEAN_S * HZ)
    rssi_away = rssi - rolling_mean(rssi, MEAN_S * HZ)
    rssi_second = rssi - lagged(rssi, HZ)
    out = np.concatenate([short, second, away, rssi_away, rssi_second], axis=-1)
    if scale:
        typical = rolling_mean(np.abs(second).mean(-1, keepdims=True), MEAN_S * HZ) + 0.05
        out = out / typical
    return out.astype(dtype)


def load(paths, scale):
    rooms = []
    for path in paths:
        index = json.load(open(path))
        data = np.memmap(path.replace(".json", ".bin"), dtype=np.float32, mode="r")
        for entry in index:
            links = len(entry["links"])
            raw = np.array(data[entry["offset"]:entry["offset"] + entry["steps"] * links * (BINS + 1)])
            raw = raw.reshape(entry["steps"], links, BINS + 1)
            rooms.append({"seed": entry["room"], "inputs": link_inputs(raw, scale),
                          "movement": np.array(entry["movement"], dtype=np.int64)})
    return rooms


class MovementNet(nn.Module):
    def __init__(self, channels, width, dilations):
        super().__init__()
        layers, c = [], channels
        for d in dilations:
            layers += [nn.Conv1d(c, width, 5, padding=2 * d, dilation=d), nn.ReLU()]
            c = width
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Conv1d(2 * width, width, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(width, len(MOVEMENTS), 1))

    def forward(self, x):
        """x: B x L x T x C -> B x classes x T."""
        b, l, t, c = x.shape
        h = self.encoder(x.reshape(b * l, t, c).transpose(1, 2)).reshape(b, l, -1, t)
        return self.head(torch.cat([h.mean(1), h.amax(1)], 1))


def receptive_half(dilations):
    return 2 * sum(dilations) + 2


def predict(model, room):
    """Per-step logits of a whole room sequence (classes x steps)."""
    x = torch.tensor(room["inputs"], dtype=torch.float32).permute(1, 0, 2)[None].to(DEVICE)
    with torch.no_grad():
        return model(x)[0].cpu().numpy()


def per_second(room, logits):
    """Truth and guess per whole second: the truth at mid-second, the guess from the second's mean logit."""
    seconds = len(room["movement"]) // HZ
    truth = room["movement"][HZ // 2::HZ][:seconds]
    guess = logits[:, :seconds * HZ].reshape(len(MOVEMENTS), seconds, HZ).mean(-1).argmax(0)
    return truth, guess


def confusion(model, rooms):
    matrix = np.zeros((len(MOVEMENTS), len(MOVEMENTS)), dtype=np.int64)
    for room in rooms:
        truth, guess = per_second(room, predict(model, room))
        np.add.at(matrix, (truth, guess), 1)
    return matrix


def report(matrix, title):
    recall = matrix.diagonal() / np.maximum(matrix.sum(1), 1)
    precision = matrix.diagonal() / np.maximum(matrix.sum(0), 1)
    moves = recall[2:].mean()
    lines = [f"{title}: accuracy {matrix.trace() / matrix.sum() * 100:.1f}%, balanced {recall.mean() * 100:.1f}%, "
             f"movement classes (wave..stand_up) mean recall {moves * 100:.1f}%"]
    lines.append("            " + " ".join(f"{m[:9]:>9}" for m in MOVEMENTS) + "   recall precision")
    for i, m in enumerate(MOVEMENTS):
        lines.append(f"{m:>11} " + " ".join(f"{v:>9}" for v in matrix[i])
                     + f"   {recall[i] * 100:5.1f}%  {precision[i] * 100:5.1f}%")
    return "\n".join(lines), moves


def batches(rooms, window, size, rng):
    by_links = {}
    for room in rooms:
        by_links.setdefault(room["inputs"].shape[1], []).append(room)
    groups = list(by_links.values())
    weights = np.array([len(g) for g in groups], dtype=np.float64)
    count = sum(len(r["movement"]) for r in rooms) // (window // 2) // size
    for _ in range(count):
        group = groups[rng.choice(len(groups), p=weights / weights.sum())]
        xs, ys = [], []
        for _ in range(size):
            room = group[rng.integers(len(group))]
            start = rng.integers(0, len(room["movement"]) - window)
            xs.append(room["inputs"][start:start + window])
            ys.append(room["movement"][start:start + window])
        yield (torch.tensor(np.stack(xs), dtype=torch.float32).permute(0, 2, 1, 3),
               torch.tensor(np.stack(ys)))


def main():
    paths = [a for a in sys.argv[1:] if "=" not in a]
    for arg in (a for a in sys.argv[1:] if "=" in a):
        key, value = arg.split("=", 1)
        CONFIG[key] = type(CONFIG[key])(value)
    global DEVICE
    DEVICE = CONFIG["device"]
    dilations = [int(d) for d in CONFIG["dilations"].split(",")]
    rooms = load(paths, CONFIG["scale"])
    train = [r for r in rooms if r["seed"] <= CONFIG["train_until"]]
    validate = [r for r in rooms if CONFIG["train_until"] < r["seed"] <= CONFIG["validate_until"]]
    test = [r for r in rooms if CONFIG["validate_until"] < r["seed"] <= STRESS_FROM]
    stress = [r for r in rooms if r["seed"] > STRESS_FROM]
    print(f"{CONFIG}\n{len(train)} rooms to learn from, {len(validate)} to choose with, {len(test)} never seen; {DEVICE}",
          flush=True)
    counts = np.bincount(np.concatenate([r["movement"] for r in train]), minlength=len(MOVEMENTS))
    class_weight = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32, device=DEVICE)
    class_weight /= class_weight.mean()
    torch.manual_seed(CONFIG["seed"])
    model = MovementNet(train[0]["inputs"].shape[-1], CONFIG["width"], dilations).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, CONFIG["epochs"])
    half = receptive_half(dilations)
    edge = min(half, CONFIG["window"] // 4)
    best = -1.0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        rng = np.random.default_rng(CONFIG["seed"] * 1000 + epoch)
        for x, y in batches(train, CONFIG["window"], CONFIG["batch"], rng):
            logits = model(x.to(DEVICE))[..., edge:-edge]
            loss = nn.functional.cross_entropy(logits, y[:, edge:-edge].to(DEVICE), weight=class_weight)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        scheduler.step()
        model.eval()
        text, moves = report(confusion(model, validate), f"epoch {epoch} validation")
        print(text.splitlines()[0], flush=True)
        if moves > best:
            best = moves
            torch.save({"state": model.state_dict(), "config": CONFIG, "dilations": dilations}, CONFIG["out"])
    saved = torch.load(CONFIG["out"], weights_only=False)
    model.load_state_dict(saved["state"])
    model.eval()
    print(report(confusion(model, validate), "best epoch, validation rooms")[0])
    if CONFIG["test"]:
        print(report(confusion(model, test), "best epoch, TEST rooms (never seen)")[0])
        for links in sorted({r["inputs"].shape[1] for r in test}):
            subset = [r for r in test if r["inputs"].shape[1] == links]
            print(report(confusion(model, subset), f"TEST rooms with {links} links")[0].splitlines()[0])
        per_room = [np.diag(m).sum() / m.sum() for m in (confusion(model, [r]) for r in test)]
        print("TEST per-room accuracy: worst {:.0f}%, 10th percentile {:.0f}%, median {:.0f}%".format(
            *(100 * np.percentile(per_room, [0, 10, 50]))))
        if stress:
            print(report(confusion(model, stress), f"STRESS rooms (never seen, hardware harsher than any trained on)")[0])


if __name__ == "__main__":
    main()
