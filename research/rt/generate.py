"""Ray-traced CSI dataset: random worlds (a room inside a flat, or a lecture hall), five activities
of the measured person each, every link as the ESP32 reports it.

    python generate.py --out ../cache/rt --rooms 400 [--first-room 0] [--seconds 30] [--fast]
                       [--live ../cache/live/worker0.json]

One .npz per world and activity (iq [links, packets, 64, 2] int8, rssi, t, truth and world as JSON).
Worlds already written are skipped, so an interrupted run continues where it stopped; world N is the
same on every machine, so a team splits the work by ranges of N. With --live it also writes what it
is tracing now, for live.html.
"""

import argparse
import json
import os
import time

import numpy as np

from channel import SIGNAL_BINS, RoomChannel
from rooms import random_room
from sessions import ACTIVITIES, BYSTANDERS, record


def room_json(room):
    return {
        "kind": room.kind,
        "width": room.width,
        "depth": room.depth,
        "height": room.height,
        "hub": room.hub,
        "devices": room.devices,
        "walkable": room.walkable,
        "boxes": [{"name": b.name, "centre": b.centre, "size": b.size, "material": b.material} for b in room.boxes],
    }


class LiveView:
    """Writes the frame being traced to a small JSON file, at most a few times a second."""

    def __init__(self, path):
        self.path = path
        self.world = None
        self.last = 0.0
        self.context = {}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def due(self):
        return time.time() - self.last >= 0.1

    def __call__(self, frame, frames, people, h, rays):
        self.last = time.time()
        state = dict(
            self.context,
            frame=frame,
            frames=frames,
            world=self.world,
            people=[
                [{"centre": p.centre, "size": p.size, "orientation": p.orientation} for p in parts.values()]
                for parts in people
            ],
            rays=rays,
            csi=[(20 * np.log10(np.abs(link[SIGNAL_BINS]) + 1e-12)).round(2).tolist() for link in h],
            time=time.time(),
        )
        with open(self.path + ".tmp", "w") as f:
            json.dump(state, f, default=float)
        os.replace(self.path + ".tmp", self.path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--rooms", type=int, default=400)
    parser.add_argument("--first-room", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument(
        "--fast", action="store_true", help="pilot quality: 3 bounces, fewer rays, walking at 10 frames/s"
    )
    parser.add_argument("--live", help="JSON file the live view reads")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    live = LiveView(args.live) if args.live else None

    for index in range(args.first_room, args.first_room + args.rooms):
        paths = {a: os.path.join(args.out, f"room{index:05d}_{a}.npz") for a in ACTIVITIES}
        if all(os.path.exists(p) for p in paths.values()):
            continue
        rng = np.random.default_rng([index, 2026])
        room = random_room(rng, devices=int(rng.integers(1, 4)))
        options = dict(max_depth=3, samples=100_000) if args.fast else {}
        channel = RoomChannel(room, people=1 + BYSTANDERS, **options)
        started = time.time()
        if live:
            live.world = room_json(room)
        for activity, path in paths.items():
            packet_hz = rng.uniform(80, 130)
            if live:
                live.context = dict(world_index=index, activity=activity, kind=room.kind)
            iq, rssi, t, truth = record(channel, room, activity, args.seconds, packet_hz, rng, fast=args.fast, live=live)
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
        print(f"world {index} ({room.kind}): {len(room.devices)} links, {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
