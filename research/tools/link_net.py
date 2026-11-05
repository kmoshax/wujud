"""Where someone walks and what they do, from all the links at once, learnt from simulated rooms only.

Every link is a token, read together like the channels of a multichannel audio model: its disturbance
and RSSI spread over its quiet level, each subcarrier's spread over its quiet level and how the
spectrum's shape moved, over the last three seconds, plus the same disturbance relative to the other
links. A transformer lets the links see each other ("this one moves while that one does not"); then
every grid point attends over the links, through how it sits relative to each (extra path, distance to
the line, place along it, link length), and is scored. Rooms are flipped at random while learning, so
no favourite spot can be learnt; a room with nothing moving should land near its centre.

    LINK_NET_WIDTH=192 LINK_NET_LAYERS=3 LINK_NET_EPOCHS=36 .venv/bin/python -I tools/link_net.py <out.pt> cache/tracking/v2_part*.jsonl cache/tracking/v3_part*.jsonl cache/tracking/v2four_p*.jsonl cache/tracking/v3four_p*.jsonl

LINK_NET_START=<model.pt> starts from a trained model; with LINK_NET_EPOCHS=0 it is only scored on
each file's unseen rooms.
"""
import json
import os
import sys

import numpy as np
import torch
from torch import nn

ACTIVITIES = ["Away", "Walk", "Sit", "Lie", "Stand"]
GRID_M = 0.25
BINS = 32
CONTEXT_S = 5
SHAPE_LAG_S = 3
QUIET_PERCENTILE = 20
TEST_SHARE = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIDTH = int(os.environ.get("LINK_NET_WIDTH", 128))
LAYERS = int(os.environ.get("LINK_NET_LAYERS", 2))
EPOCHS = int(os.environ.get("LINK_NET_EPOCHS", 30))
# Start from a trained model (fine-tuning); with LINK_NET_EPOCHS=0 only score it, file by file.
START = os.environ.get("LINK_NET_START")
LEARNING_RATE = float(os.environ.get("LINK_NET_LR", 5e-4))


def to_bins(values):
    values = np.asarray(values, dtype=np.float32)
    x = np.linspace(0, 1, values.shape[-1])
    flat = values.reshape(-1, values.shape[-1])
    return np.stack([np.interp(np.linspace(0, 1, BINS), x, v) for v in flat]).reshape(*values.shape[:-1], BINS)


def link_tokens(motion, shadow, spread, shape):
    """T x L x F, every value relative to its own link, the first two also to the other links."""
    eps = 1e-6
    quiet = lambda a: np.maximum(np.percentile(a, QUIET_PERCENTILE, axis=0), eps)
    m = np.log(np.maximum(motion, eps) / quiet(motion))
    s = np.log(np.maximum(shadow, eps) / quiet(shadow))
    sp = to_bins(spread)
    sp = np.log(np.maximum(sp, eps) / quiet(sp))
    sh = to_bins(shape)
    lagged = np.concatenate([np.repeat(sh[:1], SHAPE_LAG_S, axis=0), sh[:-SHAPE_LAG_S]], axis=0)
    moved = np.abs(sh - lagged)
    rel = np.stack([m - m.mean(1, keepdims=True), s - s.mean(1, keepdims=True)], -1)
    per_second = np.concatenate([m[..., None], s[..., None], rel, sp, moved], axis=-1)
    padded = np.concatenate([np.repeat(per_second[:1], CONTEXT_S - 1, axis=0), per_second], axis=0)
    stacked = np.concatenate([padded[i:i + len(per_second)] for i in range(CONTEXT_S)], axis=-1)
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


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


def load(paths):
    rooms = {}
    for path in paths:
        current = {}
        for line in open(path):
            row = json.loads(line)
            current.setdefault(row["room"], []).append(row)
        for room_id, rows in current.items():
            rows = both_directions_averaged(rows)
            first = rows[0]
            rooms[(path, room_id)] = {
                "tokens": link_tokens(np.array([r["motion"] for r in rows]), np.array([r["shadowing"] for r in rows]),
                                      np.array([r["spread"] for r in rows]), np.array([r["shape"] for r in rows])),
                "nodes": np.array(first["nodes"], dtype=np.float32),
                "links": first["links"],
                "width": first["width"], "depth": first["depth"],
                "activity": np.array([ACTIVITIES.index(r["activity"]) for r in rows]),
                "xy": np.array([r["xy"] if r["xy"] else [np.nan, np.nan] for r in rows], dtype=np.float32),
                "path": path,
            }
        print(f"loaded {path}: {len(current)} rooms", flush=True)
    return rooms


def both_directions_averaged(rows):
    """A pair measured both ways becomes one link, its readings averaged, as the live view does
    (crates/app/src/link_net.rs: add_second)."""
    links = [tuple(l) for l in rows[0]["links"]]
    pairs = sorted({tuple(sorted(l)) for l in links})
    if len(pairs) == len(links):
        return rows
    members = [[i for i, l in enumerate(links) if tuple(sorted(l)) == p] for p in pairs]
    merged = []
    for row in rows:
        row = dict(row)
        for key in ("motion", "shadowing"):
            row[key] = [float(np.mean([row[key][i] for i in m])) for m in members]
        for key in ("spread", "shape"):
            row[key] = [np.mean([row[key][i] for i in m], axis=0).tolist() for m in members]
        row["links"] = [list(p) for p in pairs]
        merged.append(row)
    return merged


class LinkNet(nn.Module):
    def __init__(self, features, width=WIDTH, layers=LAYERS):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(features, width), nn.ReLU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0.1, batch_first=True)
        self.links = nn.TransformerEncoder(layer, layers)
        self.geo = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64))
        self.pair = nn.Sequential(nn.Linear(width + 64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.attend = nn.Linear(128, 1)
        self.score = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.activity = nn.Sequential(nn.Linear(width * 2, 128), nn.ReLU(), nn.Linear(128, len(ACTIVITIES)))

    def forward(self, tokens, geo):
        """tokens: T x L x F; geo: P x L x 4 -> location logits T x P, activity logits T x A."""
        c = self.links(self.embed(tokens))
        g = self.geo(geo)
        t, p = c.shape[0], g.shape[0]
        pair = self.pair(torch.cat([c[:, None].expand(-1, p, -1, -1), g[None].expand(t, -1, -1, -1)], -1))
        weights = torch.softmax(self.attend(pair), dim=2)
        location = self.score((weights * pair).sum(2)).squeeze(-1)
        return location, self.activity(torch.cat([c.mean(1), c.amax(1)], -1))


def flipped(room, rng):
    """The same room mirrored in x and/or y: nodes, truth and grid together."""
    nodes, xy = room["nodes"].copy(), room["xy"].copy()
    if rng.random() < 0.5:
        nodes[:, 0], xy[:, 0] = room["width"] - nodes[:, 0], room["width"] - xy[:, 0]
    if rng.random() < 0.5:
        nodes[:, 1], xy[:, 1] = room["depth"] - nodes[:, 1], room["depth"] - xy[:, 1]
    return nodes, xy


def run(model, room, seconds, nodes, xy, optimiser=None):
    points = grid(room["width"], room["depth"])
    geo = torch.tensor(geometry(points, nodes, room["links"]), device=DEVICE)
    tokens = torch.tensor(room["tokens"][seconds], device=DEVICE)
    location, activity = model(tokens, geo)
    truth = room["activity"][seconds]
    loss = nn.functional.cross_entropy(activity, torch.tensor(truth, device=DEVICE))
    walking = (truth == 1) & ~np.isnan(xy[seconds][:, 0])
    if walking.any():
        target_xy = xy[seconds][walking]
        squared = ((points[None] - target_xy[:, None]) ** 2).sum(-1)
        target = np.exp(-(squared - squared.min(1, keepdims=True)) / (2 * 0.3 ** 2))
        target = torch.tensor(target / target.sum(1, keepdims=True), device=DEVICE)
        loss = loss - (target * torch.log_softmax(location[torch.tensor(walking)], 1)).sum(1).mean()
    if optimiser:
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
    return location.detach(), activity.detach(), points, walking


def evaluate(model, rooms):
    by_devices = {}
    model.eval()
    with torch.no_grad():
        for room in rooms:
            seconds = np.arange(len(room["activity"]))
            location, activity, points, walking = run(model, room, seconds, room["nodes"], room["xy"])
            n = len(room["nodes"])
            stats = by_devices.setdefault(n, {"err": [], "centre": [], "right": np.zeros(5), "total": np.zeros(5), "idle": []})
            guess = torch.softmax(location, 1).cpu().numpy() @ points
            truth_xy = room["xy"][seconds][walking]
            stats["err"] += list(np.linalg.norm(guess[walking] - truth_xy, axis=1))
            centre = np.array([room["width"] / 2, room["depth"] / 2])
            stats["centre"] += list(np.linalg.norm(truth_xy - centre, axis=1))
            away = room["activity"] == 0
            stats["idle"] += list(np.linalg.norm(guess[away] - centre, axis=1))
            said, truth = activity.argmax(1).cpu().numpy(), room["activity"]
            for a in range(5):
                stats["total"][a] += (truth == a).sum()
                stats["right"][a] += ((truth == a) & (said == a)).sum()
    for n, s in sorted(by_devices.items()):
        recall = s["right"] / np.maximum(s["total"], 1)
        print(f"  {n} devices: location {np.median(s['err']):.2f} m (centre {np.median(s['centre']):.2f}), "
              f"within 1 m {np.mean(np.array(s['err']) < 1) * 100:.0f}%, empty room lands {np.median(s['idle']):.2f} m from centre | "
              + " ".join(f"{a} {r * 100:.0f}%" for a, r in zip(ACTIVITIES, recall)), flush=True)


def main():
    out_path, paths = sys.argv[1], sys.argv[2:]
    rooms = load(paths)
    by_path = {}
    for key, room in rooms.items():
        by_path.setdefault(room["path"], []).append(key)
    train, test = [], []
    for keys in by_path.values():
        keys.sort()
        cut = int(len(keys) * (1 - TEST_SHARE))
        train += [rooms[k] for k in keys[:cut]]
        test += [rooms[k] for k in keys[cut:]]
    features = train[0]["tokens"].shape[-1]
    print(f"{len(train)} rooms to learn from, {len(test)} never seen; {features} inputs per link; {DEVICE}", flush=True)
    torch.manual_seed(0)
    model = LinkNet(features).to(DEVICE)
    if START:
        model.load_state_dict(torch.load(START, map_location=DEVICE))
    if EPOCHS == 0:
        for path in by_path:
            print(path, flush=True)
            evaluate(model, [r for r in test if r["path"] == path])
        return
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    epochs = EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(epoch)
        for i in rng.permutation(len(train)):
            room = train[i]
            seconds = rng.choice(len(room["activity"]), size=min(96, len(room["activity"])), replace=False)
            nodes, xy = flipped(room, rng)
            run(model, room, seconds, nodes, xy, optimiser)
        scheduler.step()
        if epoch % 4 == 3:
            print(f"epoch {epoch}:", flush=True)
            evaluate(model, test)
            torch.save(model.state_dict(), out_path)


if __name__ == "__main__":
    main()
