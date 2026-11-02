"""Does following the person through time beat reading each second alone? On simulated rooms the network
never saw, compares the per-second answer of link_net, the live view's easing (half way to each new
answer), and a Bayes filter on the grid: the last belief spread by how far someone walks in a second,
times the network's distribution raised to a power. The spread and power are chosen on other rooms.

    .venv/bin/python -I tools/track_filter.py cache/tracking/link_net.pt cache/tracking/v2_part4.jsonl,cache/tracking/v2four_p10.jsonl cache/tracking/v2_part5.jsonl,cache/tracking/v2four_p11.jsonl
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import link_net
from link_net import LinkNet, geometry, grid, load

EASE = 0.5


def network_answers(model, rooms):
    out = []
    with torch.no_grad():
        for room in rooms:
            points = grid(room["width"], room["depth"])
            geo = torch.tensor(geometry(points, room["nodes"], room["links"]), device=link_net.DEVICE)
            location, _ = model(torch.tensor(room["tokens"], device=link_net.DEVICE), geo)
            out.append((room, points, torch.softmax(location, 1).cpu().numpy()))
    return out


def follow(points, probs, walking, spread_m, power):
    """The filter's position each second; the belief only moves while someone walks."""
    distance = np.linalg.norm(points[:, None] - points[None], axis=-1)
    move = np.exp(-distance ** 2 / (2 * spread_m ** 2))
    move /= move.sum(0, keepdims=True)
    belief = np.full(len(points), 1 / len(points))
    guesses = []
    for p, w in zip(probs, walking):
        if w:
            belief = (move @ belief) * np.maximum(p, 1e-12) ** power
            belief /= belief.sum()
        guesses.append(belief @ points)
    return np.array(guesses)


def errors(answers, method):
    err, jumps = [], []
    for room, points, probs in answers:
        walking = room["activity"] == 1
        if not walking.any():
            continue
        guess = method(points, probs, walking)
        truth = room["xy"]
        ok = walking & ~np.isnan(truth[:, 0])
        err += list(np.linalg.norm(guess[ok] - truth[ok], axis=1))
        steps = np.linalg.norm(np.diff(guess, axis=0), axis=1)[ok[1:] & ok[:-1]]
        jumps += list(steps)
    err, jumps = np.array(err), np.array(jumps)
    return np.median(err), np.mean(err < 1), np.mean(jumps > 1.0)


def raw(points, probs, walking):
    return probs @ points


def eased(points, probs, walking):
    out, xy = [], None
    for p, w in zip(probs, walking):
        if w:
            now = p @ points
            xy = now if xy is None else xy + (now - xy) * EASE
        out.append(xy if xy is not None else p @ points)
    return np.array(out)


def main():
    weights, tune_paths, test_paths = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
    state = torch.load(weights, map_location=link_net.DEVICE, weights_only=True)
    features = state["embed.0.weight"].shape[1]
    link_net.CONTEXT_S = features // (4 + 2 * link_net.BINS)
    width = state["embed.0.weight"].shape[0]
    layers = len({k.split(".")[2] for k in state if k.startswith("links.layers.")})
    model = LinkNet(features, width, layers).to(link_net.DEVICE)
    model.load_state_dict(state)
    model.eval()
    tune = network_answers(model, list(load(tune_paths).values())[-40:])
    best = min(((s, p) for s in (0.3, 0.5, 0.8, 1.2) for p in (0.3, 0.5, 1.0)),
               key=lambda sp: errors(tune, lambda pt, pr, w: follow(pt, pr, w, *sp))[0])
    print(f"chosen on other rooms: spread {best[0]} m, power {best[1]}")
    test_rooms = load(test_paths)
    for devices in (3, 4):
        answers = network_answers(model, [r for r in list(test_rooms.values()) if len(r["nodes"]) == devices][-30:])
        for name, method in [("each second", raw), ("eased (live view)", eased),
                             ("Bayes filter", lambda pt, pr, w: follow(pt, pr, w, *best))]:
            median, within, jumpy = errors(answers, method)
            print(f"  {devices} devices, {name}: {median:.2f} m median, {within * 100:.0f}% within 1 m, "
                  f"{jumpy * 100:.1f}% of seconds jump over 1 m")


if __name__ == "__main__":
    main()
