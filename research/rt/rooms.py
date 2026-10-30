"""Random furnished rooms for the ray tracer: walls, floor, ceiling, furniture, and where the radios go.

A room is a list of boxes (each with an ITU material or a custom one) plus a hub (the ESP32 that
measures CSI) and one or more devices (phones / laptops whose packets it measures). Everything is
in metres, origin in a corner, z up.
"""

from dataclasses import dataclass, field

import numpy as np

# ITU-R P.2040 materials Sionna knows, with how often each is used for walls.
WALL_MATERIALS = {"concrete": 0.45, "brick": 0.35, "plasterboard": 0.15, "glass": 0.05}
FURNITURE_MATERIALS = {"wood": 0.7, "chipboard": 0.2, "metal": 0.1}
WALL_THICKNESS = 0.15


@dataclass
class Box:
    name: str
    centre: tuple
    size: tuple
    material: str
    yaw: float = 0.0


@dataclass
class Room:
    width: float
    depth: float
    height: float
    boxes: list
    hub: tuple
    devices: list
    # Free floor area people can walk in, as (x0, y0, x1, y1).
    walkable: tuple
    furniture: list = field(default_factory=list)


def pick(rng, weights):
    names = list(weights)
    return names[rng.choice(len(names), p=np.array(list(weights.values())) / sum(weights.values()))]


def shell(width, depth, height, rng):
    wall = pick(rng, WALL_MATERIALS)
    t = WALL_THICKNESS
    return [
        Box("floor", (width / 2, depth / 2, -t / 2), (width + 2 * t, depth + 2 * t, t), "concrete"),
        Box("ceiling", (width / 2, depth / 2, height + t / 2), (width + 2 * t, depth + 2 * t, t), "concrete"),
        Box("wall_south", (width / 2, -t / 2, height / 2), (width, t, height), wall),
        Box("wall_north", (width / 2, depth + t / 2, height / 2), (width, t, height), wall),
        Box("wall_west", (-t / 2, depth / 2, height / 2), (t, depth, height), wall),
        Box("wall_east", (width + t / 2, depth / 2, height / 2), (t, depth, height), wall),
    ]


# (kind, footprint w x d, height range); placed against a wall, not overlapping.
FURNITURE_KINDS = [
    ("bed", (1.9, 1.0), (0.4, 0.6)),
    ("desk", (1.2, 0.6), (0.72, 0.78)),
    ("wardrobe", (1.2, 0.6), (1.8, 2.2)),
    ("chest", (0.9, 0.45), (0.8, 1.0)),
    ("shelf", (0.8, 0.3), (1.2, 1.9)),
]


def place_against_wall(rng, width, depth, footprint, taken):
    for _ in range(40):
        w, d = footprint
        side = rng.integers(4)
        if side in (2, 3):
            w, d = d, w
        if side == 0:
            x, y = rng.uniform(w / 2, width - w / 2), d / 2
        elif side == 1:
            x, y = rng.uniform(w / 2, width - w / 2), depth - d / 2
        elif side == 2:
            x, y = w / 2, rng.uniform(d / 2, depth - d / 2)
        else:
            x, y = width - w / 2, rng.uniform(d / 2, depth - d / 2)
        rect = (x - w / 2, y - d / 2, x + w / 2, y + d / 2)
        if all(rect[2] <= o[0] or rect[0] >= o[2] or rect[3] <= o[1] or rect[1] >= o[3] for o in taken):
            return (x, y, w, d), rect
    return None


def furniture(rng, width, depth):
    boxes, taken = [], []
    kinds = rng.choice(len(FURNITURE_KINDS), size=rng.integers(2, 5), replace=False)
    for k in kinds:
        kind, footprint, heights = FURNITURE_KINDS[k]
        spot = place_against_wall(rng, width, depth, footprint, taken)
        if spot is None:
            continue
        (x, y, w, d), rect = spot
        h = rng.uniform(*heights)
        material = "metal" if kind == "wardrobe" and rng.random() < 0.25 else pick(rng, FURNITURE_MATERIALS)
        boxes.append(Box(f"furniture_{kind}", (x, y, h / 2), (w, d, h), material))
        taken.append(rect)
        if kind in ("chest", "wardrobe") and rng.random() < 0.4:
            # A mirror on its front: thin metal-backed glass, a strong reflector.
            front_y = rect[1] if y > depth / 2 else rect[3]
            boxes.append(Box(f"mirror_{kind}", (x, front_y, h * 0.65), (w * 0.8, 0.006, h * 0.5), "metal"))
    return boxes, taken


def random_room(rng, devices=2):
    width, depth = rng.uniform(3.0, 5.5), rng.uniform(2.8, 4.5)
    height = rng.uniform(2.6, 3.1)
    furn, taken = furniture(rng, width, depth)

    def radio_spot():
        # Near a wall, at the height of a shelf, desk or bed.
        side = rng.integers(4)
        along = rng.uniform(0.2, 0.8)
        x = {0: along * width, 1: along * width, 2: 0.25, 3: width - 0.25}[side]
        y = {0: 0.25, 1: depth - 0.25, 2: along * depth, 3: along * depth}[side]
        return (x, y, rng.uniform(0.3, 1.2))

    hub = radio_spot()
    others = []
    while len(others) < devices:
        spot = radio_spot()
        if np.hypot(spot[0] - hub[0], spot[1] - hub[1]) > 1.5:
            others.append(spot)
    return Room(
        width=width,
        depth=depth,
        height=height,
        boxes=shell(width, depth, height, rng) + furn,
        hub=hub,
        devices=others,
        walkable=(0.3, 0.3, width - 0.3, depth - 0.3),
        furniture=taken,
    )
