"""How many people are in a room, how many walk, and where each walker is, from all the links at once,
learnt from simulated rooms only (link_net.py extended to a crowd).

The links are read as in link_net.py: one token per pair of devices, a transformer across them, then
every grid point attends over the links through its geometry. Two count heads read the pooled tokens:
everyone in the room (0..MAX_PEOPLE) and only those walking. The grid is scored per point (a heatmap,
not a softmax), learnt against the max of Gaussians at every walker; the walkers are its top peaks after
non-maximum suppression, as many as the walking count says. Per-person error is measured after matching
guesses to people (Hungarian). Rooms are split, never seconds: the last 20 % of every file is unseen.

    .venv/bin/python -I tools/crowd_net.py <out.pt> <report.json> cache/crowd/*.jsonl
"""
import json
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from link_net import ACTIVITIES, geometry, grid, link_tokens  # noqa: E402

MAX_PEOPLE = 5
WALK = ACTIVITIES.index("Walk")
AWAY = ACTIVITIES.index("Away")
SIGMA_M = 0.3
NMS_M = 0.6
TEST_SHARE = 0.2
SECONDS_PER_STEP = 48
EPOCHS = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(paths):
    rooms = []
    for path in paths:
        current = {}
        for line in open(path):
            row = json.loads(line)
            current.setdefault(row["room"], []).append(row)
        keys = sorted(current)
        cut = int(len(keys) * (1 - TEST_SHARE))
        for i, room_id in enumerate(keys):
            rows = current[room_id]
            first = rows[0]
            people = [[(ACTIVITIES.index(p["activity"]), p["xy"]) for p in r["people"]] for r in rows]
            walkers = np.full((len(rows), MAX_PEOPLE, 2), np.nan, dtype=np.float32)
            for t, everyone in enumerate(people):
                spots = [xy for a, xy in everyone if a == WALK and xy]
                if spots:
                    walkers[t, :len(spots)] = spots
            rooms.append({
                "tokens": link_tokens(np.array([r["motion"] for r in rows]), np.array([r["shadowing"] for r in rows]),
                                      np.array([r["spread"] for r in rows]), np.array([r["shape"] for r in rows])).astype(np.float16),
                "nodes": np.array(first["nodes"], dtype=np.float32),
                "links": first["links"],
                "width": first["width"], "depth": first["depth"],
                "present": np.array([sum(a != AWAY for a, _ in e) for e in people]),
                "walking": np.array([sum(a == WALK and xy is not None for a, xy in e) for e in people]),
                "walkers": walkers,
                "test": i >= cut,
                "id": (path, room_id),
            })
        print(f"loaded {path}: {len(keys)} rooms", flush=True)
    return rooms


class CrowdNet(nn.Module):
    def __init__(self, features, width=128):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(features, width), nn.ReLU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, 4, 256, dropout=0.1, batch_first=True)
        self.links = nn.TransformerEncoder(layer, 3)
        self.geo = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64))
        self.pair = nn.Sequential(nn.Linear(width + 64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.attend = nn.Linear(128, 1)
        self.score = nn.Sequential(nn.Linear(128 * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        pooled = width * 2
        self.present = nn.Sequential(nn.Linear(pooled, 128), nn.ReLU(), nn.Linear(128, MAX_PEOPLE + 1))
        self.walking = nn.Sequential(nn.Linear(pooled, 128), nn.ReLU(), nn.Linear(128, MAX_PEOPLE + 1))

    def forward(self, tokens, geo):
        """tokens: T x L x F; geo: P x L x 4 -> heatmap logits T x P, present and walking logits T x (MAX+1)."""
        c = self.links(self.embed(tokens))
        g = self.geo(geo)
        t, p = c.shape[0], g.shape[0]
        pair = self.pair(torch.cat([c[:, None].expand(-1, p, -1, -1), g[None].expand(t, -1, -1, -1)], -1))
        weights = torch.softmax(self.attend(pair), dim=2)
        attended = (weights * pair).sum(2)
        heat = self.score(torch.cat([attended, attended - attended.mean(1, keepdim=True)], -1)).squeeze(-1)
        pooled = torch.cat([c.mean(1), c.amax(1)], -1)
        return heat, self.present(pooled), self.walking(pooled)


def flipped(room, rng):
    nodes, walkers = room["nodes"].copy(), room["walkers"].copy()
    if rng.random() < 0.5:
        nodes[:, 0], walkers[..., 0] = room["width"] - nodes[:, 0], room["width"] - walkers[..., 0]
    if rng.random() < 0.5:
        nodes[:, 1], walkers[..., 1] = room["depth"] - nodes[:, 1], room["depth"] - walkers[..., 1]
    return nodes, walkers


def heat_target(points, walkers):
    """T x P: the max over walkers of a Gaussian bump at each (0 where nobody walks)."""
    squared = ((points[None, None] - walkers[:, :, None]) ** 2).sum(-1)
    bumps = np.exp(-squared / (2 * SIGMA_M ** 2))
    return np.nan_to_num(bumps, nan=0.0).max(1).astype(np.float32)


def forward_room(model, room, seconds, nodes, walkers):
    points = grid(room["width"], room["depth"])
    geo = torch.tensor(geometry(points, nodes, room["links"]), device=DEVICE)
    tokens = torch.tensor(room["tokens"][seconds].astype(np.float32), device=DEVICE)
    heat, present, walking = model(tokens, geo)
    return heat, present, walking, points


def loss_of(room, seconds, heat, present, walking, points, walkers):
    target = torch.tensor(heat_target(points, walkers[seconds]), device=DEVICE)
    positive = target.sum() / target.numel()
    heat_loss = nn.functional.binary_cross_entropy_with_logits(
        heat, target, pos_weight=torch.clamp(1 / positive.clamp(min=1e-3) - 1, max=50.0).detach())
    count = lambda logits, key: nn.functional.cross_entropy(
        logits, torch.tensor(np.minimum(room[key][seconds], MAX_PEOPLE), device=DEVICE))
    return heat_loss + count(present, "present") + count(walking, "walking")


def peaks(heat, points, k):
    """The k highest grid points at least NMS_M apart."""
    order = np.argsort(-heat)
    chosen = []
    for i in order:
        if len(chosen) == k:
            break
        if all(np.linalg.norm(points[i] - points[j]) >= NMS_M for j in chosen):
            chosen.append(i)
    return points[chosen]


def matched_errors(guess, truth):
    if len(guess) == 0 or len(truth) == 0:
        return []
    cost = np.linalg.norm(guess[:, None] - truth[None], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    return list(cost[rows, cols])


def evaluate(model, rooms, majority):
    model.eval()
    cells = {}
    with torch.no_grad():
        for room in rooms:
            n = len(room["nodes"])
            seconds = np.arange(len(room["present"]))
            heats, presents, walkings = [], [], []
            for chunk in np.array_split(seconds, max(1, len(seconds) // SECONDS_PER_STEP)):
                heat, present, walking, points = forward_room(model, room, chunk, room["nodes"], room["walkers"])
                heats.append(torch.sigmoid(heat).cpu().numpy())
                presents.append(present.argmax(1).cpu().numpy())
                walkings.append(walking.argmax(1).cpu().numpy())
            heat, said_present, said_walking = np.concatenate(heats), np.concatenate(presents), np.concatenate(walkings)
            centre = np.array([room["width"] / 2, room["depth"] / 2], dtype=np.float32)
            for t in seconds:
                truth = room["walkers"][t][~np.isnan(room["walkers"][t][:, 0])]
                present, walking = int(room["present"][t]), len(truth)
                cell = cells.setdefault((n, present), new_cell())
                cell["seconds"] += 1
                cell["present_right"] += int(said_present[t] == present)
                cell["present_near"] += int(abs(int(said_present[t]) - present) <= 1)
                cell["present_majority"] += int(majority[n]["present"] == present)
                cell["walking_right"] += int(said_walking[t] == walking)
                cell["walking_majority"] += int(majority[n]["walking"] == walking)
                by_walkers = cells.setdefault((n, "walking", walking), new_cell())
                by_walkers["seconds"] += 1
                by_walkers["walking_right"] += int(said_walking[t] == walking)
                by_walkers["walking_majority"] += int(majority[n]["walking"] == walking)
                if walking:
                    known_k = matched_errors(peaks(heat[t], points, walking), truth)
                    said_k = peaks(heat[t], points, int(said_walking[t]))
                    for cell_ in (cell, by_walkers):
                        cell_["err_known_count"] += known_k
                        cell_["err_said_count"] += matched_errors(said_k, truth)
                        cell_["missed"] += max(0, walking - len(said_k))
                        cell_["walkers"] += walking
                        cell_["centre"] += list(np.linalg.norm(truth - centre, axis=1))
    return cells


def new_cell():
    return {k: 0 for k in ["seconds", "present_right", "present_near", "present_majority", "walking_right",
                           "walking_majority", "missed", "walkers"]} | {"err_known_count": [], "err_said_count": [], "centre": []}


def summary(cells):
    out = []
    for key, c in sorted(cells.items(), key=lambda kv: str(kv[0])):
        med = lambda v: float(np.median(v)) if v else None
        row = {"devices": key[0], "group": "walking" if len(key) == 3 else "present", "count": key[-1],
               "seconds": c["seconds"],
               "walking_acc": c["walking_right"] / c["seconds"], "walking_majority_acc": c["walking_majority"] / c["seconds"],
               "err_known_count_m": med(c["err_known_count"]), "err_said_count_m": med(c["err_said_count"]),
               "within_1m": float(np.mean(np.array(c["err_known_count"]) < 1)) if c["err_known_count"] else None,
               "centre_m": med(c["centre"]), "missed_share": c["missed"] / max(c["walkers"], 1)}
        if len(key) == 2:
            row |= {"present_acc": c["present_right"] / c["seconds"], "present_within1": c["present_near"] / c["seconds"],
                    "present_majority_acc": c["present_majority"] / c["seconds"]}
        out.append(row)
    return out


def show(rows):
    fmt = lambda v: "  -  " if v is None else f"{v:5.2f}"
    for r in rows:
        extra = (f"present {r['present_acc'] * 100:3.0f}% (+-1 {r['present_within1'] * 100:3.0f}%, majority {r['present_majority_acc'] * 100:3.0f}%) "
                 if r["group"] == "present" else "")
        print(f"  {r['devices']} dev {r['group']:>7} {r['count']}: n={r['seconds']:5d} {extra}"
              f"walking {r['walking_acc'] * 100:3.0f}% (maj {r['walking_majority_acc'] * 100:3.0f}%) "
              f"err {fmt(r['err_known_count_m'])} / said-k {fmt(r['err_said_count_m'])} m, centre {fmt(r['centre_m'])}", flush=True)


def majorities(train):
    result = {}
    for n in sorted({len(r["nodes"]) for r in train}):
        pick = [r for r in train if len(r["nodes"]) == n]
        common = lambda key: int(np.bincount(np.concatenate([r[key] for r in pick])).argmax())
        result[n] = {"present": common("present"), "walking": common("walking")}
    return result


def main():
    out_path, report_path, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    rooms = load(paths)
    train = [r for r in rooms if not r["test"]]
    test = [r for r in rooms if r["test"]]
    majority = majorities(train)
    features = train[0]["tokens"].shape[-1]
    print(f"{len(train)} rooms to learn from, {len(test)} never seen; majority {majority}; {DEVICE}", flush=True)
    torch.manual_seed(0)
    model = CrowdNet(features).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, EPOCHS)
    for epoch in range(EPOCHS):
        model.train()
        rng = np.random.default_rng(epoch)
        total = 0.0
        for i in rng.permutation(len(train)):
            room = train[i]
            seconds = rng.choice(len(room["present"]), size=min(SECONDS_PER_STEP, len(room["present"])), replace=False)
            nodes, walkers = flipped(room, rng)
            heat, present, walking, points = forward_room(model, room, seconds, nodes, walkers)
            loss = loss_of(room, seconds, heat, present, walking, points, walkers)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += loss.item()
        scheduler.step()
        print(f"epoch {epoch}: loss {total / len(train):.3f}", flush=True)
        if epoch % 6 == 5 or epoch == EPOCHS - 1:
            torch.save(model.state_dict(), out_path)
            rows = summary(evaluate(model, test, majority))
            show(rows)
            json.dump({"epoch": epoch, "majority": majority, "rows": rows}, open(report_path, "w"), indent=1)


if __name__ == "__main__":
    main()
