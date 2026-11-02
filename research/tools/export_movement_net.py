"""movement_net.pt as JSON for a Rust port (crates/app/src/movement_net.rs until 774b4fe; dropped from the
app after it failed on the real prayers, see results/movements), with the start of one simulated room (raw
ten-per-second link steps), the network inputs of its last step and the network's logits for its last
whole second with full look-ahead, so the Rust side can check it computes the same thing.

    .venv/bin/python -I tools/export_movement_net.py cache/movements/movement_net.pt cache/movements/part3.json ../crates/model/models/movement_net.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from movement_net import BINS, HZ, MOVEMENTS, MovementNet, link_inputs, receptive_half

CHECK_SECONDS = 40


def main():
    weights_path, rooms_path, out_path = sys.argv[1:4]
    saved = torch.load(weights_path, map_location="cpu", weights_only=False)
    dilations, state, config = saved["dilations"], saved["state"], saved["config"]
    model = MovementNet(state["encoder.0.weight"].shape[1], config["width"], dilations)
    model.load_state_dict(state)
    model.eval()

    index = json.load(open(rooms_path))
    entry = next(e for e in reversed(index) if len(e["links"]) == 3)
    data = np.memmap(rooms_path.replace(".json", ".bin"), dtype=np.float32, mode="r")
    steps = CHECK_SECONDS * HZ
    raw = np.array(data[entry["offset"]:entry["offset"] + steps * 3 * (BINS + 1)]).reshape(steps, 3, BINS + 1)
    raw = np.round(raw, 5)
    exact = link_inputs(raw, config["scale"], dtype=np.float32)
    half = receptive_half(dilations)
    with torch.no_grad():
        logits = model(torch.tensor(exact).permute(1, 0, 2)[None])[0].numpy()
    second_end = steps - half
    check_logits = logits[:, second_end - HZ:second_end]

    round_all = lambda t: np.round(t.numpy().astype(np.float64), 7).tolist()
    out = {
        "about": "research/tools/movement_net.py: per-link dilated 1-D CNN over 10 Hz spectrum-shape and RSSI inputs "
                 "relative to the link, links pooled by mean and max, a temporal head; trained on simulated rooms only "
                 "(results in research/results/movements)",
        "movements": MOVEMENTS, "dilations": dilations, "scale": config["scale"],
        "weights": {name: round_all(value) for name, value in state.items()},
        "check": {
            "room": entry["room"],
            "raw": raw.astype(np.float64).tolist(),
            "last_inputs": exact[-1].astype(np.float64).tolist(),
            "logits": check_logits.astype(np.float64).tolist(),
        },
    }
    json.dump(out, open(out_path, "w"), separators=(",", ":"))
    print(f"wrote {out_path}: room {entry['room']}, logits of steps {second_end - HZ}..{second_end}, "
          f"truth {[MOVEMENTS[m] for m in entry['movement'][second_end - HZ:second_end:3]]}, "
          f"said {[MOVEMENTS[m] for m in check_logits.argmax(0)[::3]]}")


if __name__ == "__main__":
    main()
