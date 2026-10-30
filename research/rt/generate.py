"""Ray-traced CSI dataset: random rooms, five activities each, every link as the ESP32 reports it.

    python generate.py --out ../cache/rt --rooms 400 [--first-room 0] [--seconds 30] [--fast]

One .npz per room and activity (iq [links, packets, 64, 2] int8, rssi, t, truth as JSON). Rooms
already written are skipped, so an interrupted run continues where it stopped.
"""

import argparse
import json
import os
import time

import numpy as np

from channel import RoomChannel
from rooms import random_room
from sessions import ACTIVITIES, record


def room_json(room):
    return {
        "width": room.width,
        "depth": room.depth,
        "height": room.height,
        "hub": room.hub,
        "devices": room.devices,
        "boxes": [{"name": b.name, "centre": b.centre, "size": b.size, "material": b.material} for b in room.boxes],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--rooms", type=int, default=400)
    parser.add_argument("--first-room", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument(
        "--fast", action="store_true", help="pilot quality: 3 bounces, fewer rays, walking at 10 frames/s"
    )
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for index in range(args.first_room, args.first_room + args.rooms):
        paths = {a: os.path.join(args.out, f"room{index:05d}_{a}.npz") for a in ACTIVITIES}
        if all(os.path.exists(p) for p in paths.values()):
            continue
        rng = np.random.default_rng([index, 2026])
        room = random_room(rng, devices=int(rng.integers(1, 4)))
        channel = RoomChannel(room, max_depth=3, samples=100_000) if args.fast else RoomChannel(room)
        started = time.time()
        for activity, path in paths.items():
            packet_hz = rng.uniform(80, 130)
            iq, rssi, t, truth = record(channel, room, activity, args.seconds, packet_hz, rng, fast=args.fast)
            np.savez_compressed(
                path + ".partial.npz",
                iq=iq,
                rssi=rssi,
                t=t,
                activity=activity,
                truth=json.dumps(truth),
                room=json.dumps(room_json(room)),
            )
            os.replace(path + ".partial.npz", path)
        print(f"room {index}: {len(room.devices)} links, {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
