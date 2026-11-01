"""Where someone is and what they do (away / walk / sit / lie / stand), learnt from simulated rooms only,
for any room and any placement of three boards.

Every input is relative to the link itself, so nothing ties the model to a room: per second and link,
the disturbance and RSSI spread over their quiet level, each subcarrier's spread over its quiet level,
and how far the spectrum's shape moved from five seconds before, over the last five seconds. Location
is scored per grid point from how that point sits relative to each link (extra path, distance, along),
averaged over the links; activity comes from the links' pooled signals.

    .venv/bin/python -I tools/room_net.py cache/tracking/v2_part*.jsonl
"""
import json
import sys

import numpy as np
import torch
from torch import nn

ACTIVITIES = ["Away", "Walk", "Sit", "Lie", "Stand"]
GRID_M = 0.25
BINS = 32
CONTEXT_S = 5
SHAPE_LAG_S = 5
QUIET_PERCENTILE = 20
TEST_SHARE = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_bins(values):
    """Any number of subcarriers onto BINS points, so 56 simulated and 52 real ones line up."""
    values = np.asarray(values, dtype=np.float32)
    x = np.linspace(0, 1, values.shape[-1])
    return np.stack([np.interp(np.linspace(0, 1, BINS), x, v) for v in values.reshape(-1, values.shape[-1])]).reshape(
        *values.shape[:-1], BINS)


def link_signals(motion, shadow, spread, shape):
    """motion, shadow: T x L; spread, shape: T x L x K -> T x L x features, all relative to the link."""
    eps = 1e-6
    quiet = lambda a: np.maximum(np.percentile(a, QUIET_PERCENTILE, axis=0), eps)
    m = np.log(np.maximum(motion, eps) / quiet(motion))[..., None]
    s = np.log(np.maximum(shadow, eps) / quiet(shadow))[..., None]
    sp = to_bins(spread)
    sp = np.log(np.maximum(sp, eps) / quiet(sp))
    sh = to_bins(shape)
    lagged = np.concatenate([np.repeat(sh[:1], SHAPE_LAG_S, axis=0), sh[:-SHAPE_LAG_S]], axis=0)
    moved = np.abs(sh - lagged)
    per_second = np.concatenate([m, s, sp, moved], axis=-1)
    padded = np.concatenate([np.repeat(per_second[:1], CONTEXT_S - 1, axis=0), per_second], axis=0)
    return np.concatenate([padded[i:i + len(per_second)] for i in range(CONTEXT_S)], axis=-1).astype(np.float32)


def grid(width, depth):
    xs = np.arange(GRID_M / 2, width, GRID_M)
    ys = np.arange(GRID_M / 2, depth, GRID_M)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float32)


def geometry(points, nodes, links):
    out = []
    for a, b in links:
        pa, pb = np.array(nodes[a]), np.array(nodes[b])
        d, rel = pb - pa, points - pa
        length = np.linalg.norm(d)
        da, db = np.linalg.norm(points - pa, axis=1), np.linalg.norm(points - pb, axis=1)
        along = np.clip((rel @ d) / max(length ** 2, 1e-9), -0.5, 1.5)
        cross = np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) / max(length, 1e-9)
        out.append(np.stack([da + db - length, cross, along, np.full(len(points), length)], axis=1))
    return np.stack(out, axis=1).astype(np.float32)


def load_rooms(paths):
    rooms = {}
    for path in paths:
        for line in open(path):
            row = json.loads(line)
            rooms.setdefault(row["room"], []).append(row)
    out = []
    for room_id, rows in sorted(rooms.items()):
        first = rows[0]
        signals = link_signals(np.array([r["motion"] for r in rows]), np.array([r["shadowing"] for r in rows]),
                               np.array([r["spread"] for r in rows]), np.array([r["shape"] for r in rows]))
        points = grid(first["width"], first["depth"])
        out.append({
            "id": room_id,
            "signals": torch.tensor(signals),
            "geo": torch.tensor(geometry(points, first["nodes"], first["links"])),
            "points": points,
            "activity": torch.tensor([ACTIVITIES.index(r["activity"]) for r in rows]),
            "xy": np.array([r["xy"] if r["xy"] else [np.nan, np.nan] for r in rows], dtype=np.float32),
            "centre": np.array([first["width"] / 2, first["depth"] / 2]),
        })
    return out


class RoomNet(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.signal = nn.Sequential(nn.Linear(features, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.geo = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.pair = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.score = nn.Linear(64, 1)
        self.activity = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(ACTIVITIES)))

    def forward(self, signals, geo):
        """signals: T x L x F; geo: P x L x 4 -> location logits T x P, activity logits T x A."""
        s = self.signal(signals)
        g = self.geo(geo)
        pair = torch.cat([s.unsqueeze(1).expand(-1, g.shape[0], -1, -1), g.unsqueeze(0).expand(s.shape[0], -1, -1, -1)], -1)
        location = self.score(self.pair(pair).mean(dim=2)).squeeze(-1)
        pooled = torch.cat([s.mean(dim=1), s.max(dim=1).values], -1)
        return location, self.activity(pooled)


def run(model, room, train, optimiser=None):
    signals, geo = room["signals"].to(DEVICE), room["geo"].to(DEVICE)
    location, activity = model(signals, geo)
    truth_activity = room["activity"].to(DEVICE)
    walking = (room["activity"] == ACTIVITIES.index("Walk")).numpy() & ~np.isnan(room["xy"][:, 0])
    loss = nn.functional.cross_entropy(activity, truth_activity)
    if walking.any():
        points = room["points"]
        target = np.exp(-((points[None] - room["xy"][walking][:, None]) ** 2).sum(-1) / (2 * 0.3 ** 2))
        target = torch.tensor(target / target.sum(1, keepdims=True), device=DEVICE)
        loss = loss + -(target * torch.log_softmax(location[torch.tensor(walking)], 1)).sum(1).mean()
    if train:
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return location.detach(), activity.detach(), walking


def evaluate(model, rooms):
    errors, centre, right, total = [], [], np.zeros(len(ACTIVITIES)), np.zeros(len(ACTIVITIES))
    with torch.no_grad():
        for room in rooms:
            location, activity, walking = run(model, room, False)
            p = torch.softmax(location, 1).cpu().numpy()
            guess = p @ room["points"]
            errors += list(np.linalg.norm(guess[walking] - room["xy"][walking], axis=1))
            centre += list(np.linalg.norm(room["xy"][walking] - room["centre"], axis=1))
            said, truth = activity.argmax(1).cpu().numpy(), room["activity"].numpy()
            for a in range(len(ACTIVITIES)):
                total[a] += (truth == a).sum()
                right[a] += ((truth == a) & (said == a)).sum()
    return np.median(errors), np.median(centre), np.mean(np.array(errors) < 1.0), right / np.maximum(total, 1)


def main():
    rooms = load_rooms(sys.argv[1:])
    cut = int(len(rooms) * (1 - TEST_SHARE))
    train, test = rooms[:cut], rooms[cut:]
    features = rooms[0]["signals"].shape[-1]
    print(f"{len(train)} rooms to learn from, {len(test)} never seen; {features} inputs per link; {DEVICE}")
    torch.manual_seed(0)
    model = RoomNet(features).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(30):
        model.train()
        for i in np.random.default_rng(epoch).permutation(len(train)):
            run(model, train[i], True, optimiser)
        model.eval()
        err, centre, within, activity = evaluate(model, test)
        if epoch % 3 == 2:
            print(f"epoch {epoch}: unseen rooms: location median {err:.2f} m (centre {centre:.2f} m), "
                  f"within 1 m {within * 100:.0f}%  |  " + "  ".join(f"{a} {r * 100:.0f}%" for a, r in zip(ACTIVITIES, activity)),
                  flush=True)
    torch.save(model.state_dict(), "cache/tracking/room_net.pt")


if __name__ == "__main__":
    main()
