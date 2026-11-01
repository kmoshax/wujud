"""link_net.pt as JSON for the live view (crates/app/src/link_net.rs), with one simulated second and the
network's answer for it, so the Rust side can check it computes the same thing.

    .venv/bin/python -I tools/export_link_net.py cache/tracking/link_net.pt cache/tracking/four_p5.jsonl ../crates/model/models/link_net.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from link_net import BINS, CONTEXT_S, GRID_M, QUIET_PERCENTILE, SHAPE_LAG_S, LinkNet, geometry, grid, link_tokens


def first_room(path):
    rows = []
    for line in open(path):
        row = json.loads(line)
        if rows and row["room"] != rows[0]["room"]:
            break
        rows.append(row)
    return rows


def main():
    weights_path, rows_path, out_path = sys.argv[1:4]
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    features = state["embed.0.weight"].shape[1]
    model = LinkNet(features)
    model.load_state_dict(state)
    model.eval()

    rows = first_room(rows_path)
    tokens = link_tokens(np.array([r["motion"] for r in rows]), np.array([r["shadowing"] for r in rows]),
                         np.array([r["spread"] for r in rows]), np.array([r["shape"] for r in rows]))
    second = next(i for i, r in enumerate(rows) if r["activity"] == "Walk" and i > 60)
    first = rows[0]
    points = grid(first["width"], first["depth"])
    with torch.no_grad():
        location, activity = model(torch.tensor(tokens[second:second + 1]),
                                   torch.tensor(geometry(points, first["nodes"], first["links"])))
    position = torch.softmax(location[0], 0).numpy() @ points

    round_all = lambda t: np.round(t.numpy().astype(np.float64), 6).tolist()
    out = {
        "about": "research/tools/link_net.py: every link a token, a transformer across links, attention over links per grid point; "
                 "trained on simulated rooms only (4 devices: 0.54 m median, 86% within 1 m on unseen rooms; 3 devices 0.71 m)",
        "grid_m": GRID_M, "bins": BINS, "context_s": CONTEXT_S, "shape_lag_s": SHAPE_LAG_S,
        "quiet_percentile": QUIET_PERCENTILE, "heads": 4,
        "activities": ["Away", "Walk", "Sit", "Lie", "Stand"],
        "weights": {name: round_all(value) for name, value in state.items()},
        "check": {
            "nodes": first["nodes"], "links": first["links"], "width": first["width"], "depth": first["depth"],
            "tokens": np.round(tokens[second].astype(np.float64), 6).tolist(),
            "position": position.tolist(),
            "activity": activity[0].numpy().tolist(),
        },
    }
    json.dump(out, open(out_path, "w"), separators=(",", ":"))
    print(f"wrote {out_path}: {features} inputs per link, check second {second} at {position.round(2)}")


if __name__ == "__main__":
    main()
