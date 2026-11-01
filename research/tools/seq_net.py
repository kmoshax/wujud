"""Where someone is, what they do and how fast they breathe, from 20 seconds of ten-per-second link
readings, learnt from simulated rooms only (crates/app/examples/export_sequences.rs).

Each link's input is relative to itself: the spectrum shape's change over the last second, its
departure from its own 30-second mean, and the RSSI's departure from its 30-second mean. A 1-D CNN
reads each link's 20 seconds; location is scored per grid point from the links' embeddings and how
the point sits relative to each link (as tools/learned_tracker.py); activity and breathing come from
the links' pooled embeddings.

    .venv/bin/python -I tools/seq_net.py cache/tracking/seq_p*.json
"""
import json
import sys

import numpy as np
import torch
from torch import nn

ACTIVITIES = ["Away", "Walk", "Sit", "Lie", "Stand"]
HZ, WINDOW_S, MEAN_S = 10, 20, 30
BINS = 32
GRID_M = 0.25
TEST_SHARE = 0.2
WINDOWS_PER_ROOM = 48
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rolling_mean(x, n):
    """Causal mean over the last n steps along axis 0."""
    c = np.cumsum(np.concatenate([np.zeros_like(x[:1]), x]), axis=0)
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - n, 0)
    return (c[idx] - c[lo]) / (idx - lo)[:, None, None]


def link_inputs(raw):
    """raw: frames x links x 33 -> frames x links x 65, all relative to the link."""
    shape, rssi = raw[..., :BINS], raw[..., BINS:]
    lagged = np.concatenate([np.repeat(shape[:1], HZ, axis=0), shape[:-HZ]], axis=0)
    change = shape - lagged
    away = shape - rolling_mean(shape, MEAN_S * HZ)
    rssi_away = rssi - rolling_mean(rssi, MEAN_S * HZ)
    return np.concatenate([change, away, rssi_away], axis=-1).astype(np.float32)


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
    rooms = []
    for path in paths:
        index = json.load(open(path))
        data = np.fromfile(path.replace(".json", ".bin"), dtype=np.float32)
        for entry in index:
            links = len(entry["links"])
            raw = data[entry["offset"]:entry["offset"] + entry["frames"] * links * (BINS + 1)]
            raw = raw.reshape(entry["frames"], links, BINS + 1)
            points = grid(entry["width"], entry["depth"])
            truth = entry["truth"]
            rooms.append({
                "inputs": torch.tensor(link_inputs(raw)),
                "geo": torch.tensor(geometry(points, entry["nodes"], entry["links"])),
                "points": points,
                "activity": np.array([ACTIVITIES.index(t[0]) for t in truth]),
                "xy": np.array([t[1] if t[1] else [np.nan, np.nan] for t in truth], dtype=np.float32),
                "breath": np.array([t[2] if t[2] else np.nan for t in truth], dtype=np.float32),
                "centre": np.array([entry["width"] / 2, entry["depth"] / 2]),
            })
    return rooms


class SeqNet(nn.Module):
    def __init__(self, channels=65, width=96):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, width, 5, padding=2), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=8, dilation=4), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=16, dilation=8), nn.ReLU(),
        )
        self.summary = nn.Sequential(nn.Linear(width * 3, 128), nn.ReLU())
        self.geo = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.pair = nn.Sequential(nn.Linear(160, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU())
        self.score = nn.Linear(64, 1)
        self.activity = nn.Sequential(nn.Linear(256, 96), nn.ReLU(), nn.Linear(96, len(ACTIVITIES)))
        self.breath = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, windows, geo):
        """windows: B x L x T x C; geo: P x L x 4."""
        b, l, t, c = windows.shape
        h = self.encoder(windows.reshape(b * l, t, c).transpose(1, 2))
        s = self.summary(torch.cat([h.mean(-1), h.amax(-1), h[..., -HZ:].mean(-1)], -1)).reshape(b, l, -1)
        g = self.geo(geo)
        pair = torch.cat([s[:, None].expand(-1, g.shape[0], -1, -1), g[None].expand(b, -1, -1, -1)], -1)
        location = self.score(self.pair(pair).mean(2)).squeeze(-1)
        pooled = torch.cat([s.mean(1), s.amax(1)], -1)
        return location, self.activity(pooled), self.breath(pooled).squeeze(-1)


def batch(room, ends):
    """Windows ending at the given seconds, with their truth."""
    steps = WINDOW_S * HZ
    x = torch.stack([room["inputs"][(e + 1) * HZ - steps:(e + 1) * HZ] for e in ends]).permute(0, 2, 1, 3)
    return x, room["activity"][ends], room["xy"][ends], room["breath"][ends]


def step(model, room, ends, optimiser=None):
    x, activity, xy, breath = batch(room, ends)
    location, act_logits, bpm = model(x.to(DEVICE), room["geo"].to(DEVICE))
    loss = nn.functional.cross_entropy(act_logits, torch.tensor(activity, device=DEVICE))
    walking = (activity == 1) & ~np.isnan(xy[:, 0])
    if walking.any():
        target = np.exp(-((room["points"][None] - xy[walking][:, None]) ** 2).sum(-1) / (2 * 0.3 ** 2))
        target = torch.tensor(target / target.sum(1, keepdims=True), device=DEVICE)
        loss = loss - (target * torch.log_softmax(location[torch.tensor(walking)], 1)).sum(1).mean()
    breathing = ~np.isnan(breath)
    if breathing.any():
        loss = loss + 0.02 * nn.functional.smooth_l1_loss(bpm[torch.tensor(breathing)], torch.tensor(breath[breathing], device=DEVICE))
    if optimiser:
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return location.detach(), act_logits.detach(), bpm.detach(), activity, xy, breath, walking


def evaluate(model, rooms):
    errors, centre, bpm_err = [], [], []
    right, total = np.zeros(len(ACTIVITIES)), np.zeros(len(ACTIVITIES))
    with torch.no_grad():
        for room in rooms:
            ends = np.arange(WINDOW_S, len(room["activity"]), 2)
            location, act, bpm, activity, xy, breath, walking = step(model, room, ends)
            guess = torch.softmax(location, 1).cpu().numpy() @ room["points"]
            errors += list(np.linalg.norm(guess[walking] - xy[walking], axis=1))
            centre += list(np.linalg.norm(xy[walking] - room["centre"], axis=1))
            said = act.argmax(1).cpu().numpy()
            for a in range(len(ACTIVITIES)):
                total[a] += (activity == a).sum()
                right[a] += ((activity == a) & (said == a)).sum()
            ok = ~np.isnan(breath)
            bpm_err += list(np.abs(bpm.cpu().numpy()[ok] - breath[ok]))
    return np.median(errors), np.median(centre), np.mean(np.array(errors) < 1), right / np.maximum(total, 1), np.median(bpm_err), np.mean(np.array(bpm_err) < 2)


def main():
    rooms = load(sys.argv[1:])
    cut = int(len(rooms) * (1 - TEST_SHARE))
    train, test = rooms[:cut], rooms[cut:]
    print(f"{len(train)} rooms to learn from, {len(test)} never seen; {DEVICE}", flush=True)
    torch.manual_seed(0)
    model = SeqNet().to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(epoch)
        for i in rng.permutation(len(train)):
            room = train[i]
            ends = rng.integers(WINDOW_S, len(room["activity"]), WINDOWS_PER_ROOM)
            step(model, room, ends, optimiser)
        scheduler.step()
        if epoch % 2 == 1:
            model.eval()
            err, centre, within, act, bpm_med, bpm_2 = evaluate(model, test)
            print(f"epoch {epoch}: location {err:.2f} m (centre {centre:.2f}), within 1 m {within * 100:.0f}% | "
                  + " ".join(f"{a} {r * 100:.0f}%" for a, r in zip(ACTIVITIES, act))
                  + f" | breathing {bpm_med:.1f}/min, within 2: {bpm_2 * 100:.0f}%", flush=True)
            torch.save(model.state_dict(), "cache/tracking/seq_net.pt")


if __name__ == "__main__":
    main()
