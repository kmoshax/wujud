"""Where a walking person is, learnt in simulation only, for any room and any board placement.

For every point of a grid over the room, each link is described by how that point sits relative to the
link (extra path length via the point, distance to the line, how far along it) and how much that link
is disturbed right now (relative to its own quiet level). A small network scores the point from those
link descriptions, averaged over the links, so it never sees a room id or absolute coordinates: it
learns the physics (which disturbed links point where), not a room.

    .venv/bin/python -I tools/learned_tracker.py <out.pt> cache/tracking/pilot.jsonl [more.jsonl ...]

Rooms from every file are split the same way (the last fifth of each never seen), so rooms with three
and with four devices are both learnt and both tested.
"""
import json
import sys

import numpy as np
import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRID_M = 0.25
SMOOTH_S = 3
QUIET_PERCENTILE = 20
TRAIN_SHARE = 0.8


def load(path):
    rooms = {}
    for line in open(path):
        row = json.loads(line)
        rooms.setdefault(row["room"], []).append(row)
    return rooms


def link_inputs(rows):
    """Per second and link: disturbance and shadowing over their quiet level, smoothed over a few seconds."""
    motion = np.array([r["motion"] for r in rows])
    shadow = np.array([r["shadowing"] for r in rows])
    quiet_m = np.maximum(np.percentile(motion, QUIET_PERCENTILE, axis=0), 1e-6)
    quiet_s = np.maximum(np.percentile(shadow, QUIET_PERCENTILE, axis=0), 1e-6)
    excess = np.stack([np.log(np.maximum(motion, 1e-6) / quiet_m), np.log(np.maximum(shadow, 1e-6) / quiet_s)], axis=-1)
    excess = np.nan_to_num(excess, nan=0.0, posinf=0.0, neginf=0.0)
    kernel = np.ones(SMOOTH_S) / SMOOTH_S
    return np.stack([np.apply_along_axis(lambda c: np.convolve(c, kernel, "same"), 0, excess[..., k])
                     for k in range(2)], axis=-1)


def grid(width, depth):
    xs = np.arange(GRID_M / 2, width, GRID_M)
    ys = np.arange(GRID_M / 2, depth, GRID_M)
    return np.array([[x, y] for y in ys for x in xs])


def geometry(points, nodes, links):
    """points x links x 4: extra path via the point, distance to the line, position along it, link length."""
    out = []
    for a, b in links:
        pa, pb = np.array(nodes[a]), np.array(nodes[b])
        length = np.linalg.norm(pb - pa)
        da = np.linalg.norm(points - pa, axis=1)
        db = np.linalg.norm(points - pb, axis=1)
        along = np.clip(((points - pa) @ (pb - pa)) / max(length ** 2, 1e-9), -0.5, 1.5)
        d, rel = pb - pa, points - pa
        cross = np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) / max(length, 1e-9)
        out.append(np.stack([da + db - length, cross, along, np.full(len(points), length)], axis=1))
    return np.stack(out, axis=1)


class PointScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.link = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.point = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, geo, signal):
        """geo: points x links x 4; signal: [seconds x] links x 2 -> one score per point [per second]."""
        if signal.dim() == 2:
            signal = signal.unsqueeze(0).expand(geo.shape[0], -1, -1)
            return self.point(self.link(torch.cat([geo, signal], dim=-1)).mean(dim=1)).squeeze(-1)
        t, p = signal.shape[0], geo.shape[0]
        pairs = torch.cat([geo.unsqueeze(0).expand(t, -1, -1, -1), signal.unsqueeze(1).expand(-1, p, -1, -1)], -1)
        return self.point(self.link(pairs).mean(dim=2)).squeeze(-1)


def samples(rooms, ids):
    out = []
    for room in ids:
        rows = rooms[room]
        signals = link_inputs(rows)
        first = rows[0]
        points = grid(first["width"], first["depth"])
        geo = torch.tensor(geometry(points, first["nodes"], first["links"]), dtype=torch.float32)
        for row, signal in zip(rows, signals):
            if row["activity"] == "Walk" and row["xy"] is not None:
                out.append((geo, torch.tensor(signal, dtype=torch.float32), points, np.array(row["xy"]),
                            np.array([first["width"] / 2, first["depth"] / 2])))
    return out


def evaluate(model, data):
    errors, centre = [], []
    with torch.no_grad():
        for geo, signal, points, truth, middle in data:
            p = torch.softmax(model(geo, signal), 0).numpy()
            guess = p @ points
            errors.append(np.linalg.norm(guess - truth))
            centre.append(np.linalg.norm(middle - truth))
    return np.median(errors), np.median(centre), np.mean(np.array(errors) < 1.0)


def room_batches(rooms, ids):
    """Per room: geometry, every walking second's signal, the truth, the grid and the centre."""
    out = []
    for room in ids:
        rows = rooms[room]
        signals = link_inputs(rows)
        first = rows[0]
        points = grid(first["width"], first["depth"])
        walking = [i for i, r in enumerate(rows) if r["activity"] == "Walk" and r["xy"] is not None]
        if not walking:
            continue
        out.append((torch.tensor(geometry(points, first["nodes"], first["links"]), dtype=torch.float32),
                    torch.tensor(signals[walking], dtype=torch.float32), points,
                    np.array([rows[i]["xy"] for i in walking]), np.array([first["width"] / 2, first["depth"] / 2]),
                    len(first["nodes"])))
    return out


def scores(model, batch):
    geo, signal = batch[0].to(DEVICE), batch[1].to(DEVICE)
    return model(geo, signal)


def report(model, batches, label):
    with torch.no_grad():
        for nodes in sorted({b[5] for b in batches}):
            errors, centre = [], []
            for b in (b for b in batches if b[5] == nodes):
                guess = torch.softmax(scores(model, b), 1).cpu().numpy() @ b[2]
                errors += list(np.linalg.norm(guess - b[3], axis=1))
                centre += list(np.linalg.norm(b[3] - b[4], axis=1))
            print(f"  {label} {nodes} devices: median {np.median(errors):.2f} m (centre {np.median(centre):.2f} m), "
                  f"within 1 m {np.mean(np.array(errors) < 1) * 100:.0f}%", flush=True)


def main():
    out_path, paths = sys.argv[1], sys.argv[2:]
    train, test = [], []
    for path in paths:
        rooms = load(path)
        ids = sorted(rooms)
        cut = int(len(ids) * TRAIN_SHARE)
        train += room_batches(rooms, ids[:cut])
        test += room_batches(rooms, ids[cut:])
    print(f"{len(train)} rooms to learn from, {len(test)} never seen; {DEVICE}", flush=True)
    torch.manual_seed(0)
    model = PointScorer().to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(40):
        for i in np.random.default_rng(epoch).permutation(len(train)):
            b = train[i]
            squared = ((b[2][None] - b[3][:, None]) ** 2).sum(-1)
            target = np.exp(-(squared - squared.min(1, keepdims=True)) / (2 * 0.3 ** 2))
            target = torch.tensor(target / target.sum(1, keepdims=True), dtype=torch.float32, device=DEVICE)
            loss = -(target * torch.log_softmax(scores(model, b), 1)).sum(1).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        if epoch % 8 == 7:
            print(f"epoch {epoch}:", flush=True)
            report(model, test, "never seen,")
    torch.save(model.cpu().state_dict(), out_path)


if __name__ == "__main__":
    main()
