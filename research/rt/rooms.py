"""Random worlds for the ray tracer: a room inside a whole flat, or a lecture hall full of students.

A world is a list of boxes, each with an ITU-R P.2040 material (or "body" for people), plus a hub
(the ESP32 that measures CSI) and the devices whose packets it measures. Nothing is an empty box:
the measured room has neighbours behind plasterboard and brick, doors that are open or shut,
windows, kitchen appliances, clutter on the floor, and other people in other rooms; a hall has
tiered rows of seats and students sitting in them. Metres, origin in a corner, z up.
"""

from dataclasses import dataclass, field

import numpy as np

EXTERIOR_WALLS = {"concrete": 0.6, "brick": 0.4}
INTERIOR_WALLS = {"brick": 0.5, "plasterboard": 0.4, "concrete": 0.1}
FURNITURE_MATERIALS = {"wood": 0.6, "chipboard": 0.3, "metal": 0.1}
WALL = 0.15
INNER_WALL = 0.10
DOOR_WIDTH = 0.9
DOOR_HEIGHT = 2.05


@dataclass
class Box:
    name: str
    centre: tuple
    size: tuple
    material: str
    yaw: float = 0.0


@dataclass
class Room:
    kind: str  # "flat" or "hall"
    width: float
    depth: float
    height: float
    boxes: list
    hub: tuple
    devices: list
    # Where the measured person may walk (x0, y0, x1, y1): the measured room, or the lecturer's area.
    walkable: tuple
    # Footprints of furniture, (x0, y0, x1, y1), that people are placed and walk around.
    furniture: list = field(default_factory=list)
    # Other places where other people move: the flat's other rooms, the hall's aisles.
    elsewhere: list = field(default_factory=list)
    # Seat positions (x, y, seat height) of students who can fidget or stand up.
    seats: list = field(default_factory=list)


def pick(rng, weights):
    names = list(weights)
    p = np.array(list(weights.values()), float)
    return names[rng.choice(len(names), p=p / p.sum())]


def wall_with_openings(name, a, b, thickness, height, material, openings):
    """A wall from a to b (axis-aligned) with openings [(start, width, bottom, top)] along it."""
    (x0, y0), (x1, y1) = a, b
    horizontal = abs(y1 - y0) < 1e-9
    length = abs(x1 - x0) if horizontal else abs(y1 - y0)

    def piece(k, s0, s1, z0, z1):
        if s1 - s0 < 0.02 or z1 - z0 < 0.02:
            return None
        mid = (s0 + s1) / 2
        if horizontal:
            centre, size = (min(x0, x1) + mid, y0, (z0 + z1) / 2), (s1 - s0, thickness, z1 - z0)
        else:
            centre, size = (x0, min(y0, y1) + mid, (z0 + z1) / 2), (thickness, s1 - s0, z1 - z0)
        return Box(f"{name}_{k}", centre, size, material)

    boxes, cursor = [], 0.0
    for k, (start, width, bottom, top) in enumerate(sorted(openings)):
        boxes += [piece(f"{k}a", cursor, start, 0, height), piece(f"{k}b", start, start + width, 0, bottom),
                  piece(f"{k}c", start, start + width, top, height)]
        cursor = start + width
    boxes.append(piece("end", cursor, length, 0, height))
    return [b for b in boxes if b is not None]


FURNITURE = {
    # kind: (footprint w x d, height range, material weights)
    "bed": ((1.9, 1.0), (0.4, 0.6), {"wood": 1}),
    "desk": ((1.2, 0.6), (0.72, 0.78), FURNITURE_MATERIALS),
    "wardrobe": ((1.2, 0.6), (1.8, 2.2), {"wood": 0.7, "metal": 0.3}),
    "chest": ((0.9, 0.45), (0.8, 1.0), FURNITURE_MATERIALS),
    "shelf": ((0.8, 0.3), (1.2, 1.9), FURNITURE_MATERIALS),
    "sofa": ((2.0, 0.85), (0.8, 0.9), {"wood": 0.5, "chipboard": 0.5}),
    "tv": ((1.2, 0.08), (0.7, 0.75), {"metal": 1}),
    "table": ((1.2, 0.8), (0.74, 0.76), FURNITURE_MATERIALS),
    "fridge": ((0.7, 0.65), (1.7, 1.9), {"metal": 1}),
    "cooker": ((0.6, 0.6), (0.85, 0.9), {"metal": 1}),
    "counter": ((2.0, 0.6), (0.88, 0.92), {"chipboard": 0.6, "marble": 0.4}),
}
ROOM_TYPES = {
    "bedroom": ["bed", "wardrobe", "desk", "chest", "shelf"],
    "living": ["sofa", "tv", "table", "shelf", "chest"],
    "kitchen": ["fridge", "cooker", "counter", "table"],
    "study": ["desk", "shelf", "shelf", "chest", "sofa"],
}


def furnish(rng, room_rect, kinds, prefix):
    """Furniture against the walls of a room, and some clutter in the open; returns boxes and footprints."""
    x0, y0, x1, y1 = room_rect
    boxes, taken = [], []
    for kind in kinds:
        (w, d), heights, materials = FURNITURE[kind]
        for _ in range(40):
            side = rng.integers(4)
            fw, fd = (w, d) if side < 2 else (d, w)
            if fw > x1 - x0 - 0.2 or fd > y1 - y0 - 0.2:
                continue
            x = rng.uniform(x0 + fw / 2, x1 - fw / 2) if side < 2 else (x0 + fw / 2 if side == 2 else x1 - fw / 2)
            y = rng.uniform(y0 + fd / 2, y1 - fd / 2) if side >= 2 else (y0 + fd / 2 if side == 0 else y1 - fd / 2)
            rect = (x - fw / 2, y - fd / 2, x + fw / 2, y + fd / 2)
            if all(rect[2] <= o[0] or rect[0] >= o[2] or rect[3] <= o[1] or rect[1] >= o[3] for o in taken):
                h = rng.uniform(*heights)
                boxes.append(Box(f"{prefix}{kind}", (x, y, h / 2), (fw, fd, h), pick(rng, materials)))
                taken.append(rect)
                if kind in ("chest", "wardrobe") and rng.random() < 0.4:
                    front = rect[1] - 0.004 if y > (y0 + y1) / 2 else rect[3] + 0.004
                    if side >= 2:
                        boxes.append(Box(f"{prefix}mirror", (rect[2] + 0.004 if x < (x0 + x1) / 2 else rect[0] - 0.004, y, h * 0.65), (0.006, fd * 0.8, h * 0.5), "metal"))
                    else:
                        boxes.append(Box(f"{prefix}mirror", (x, front, h * 0.65), (fw * 0.8, 0.006, h * 0.5), "metal"))
                break
    # clutter: boxes, bags, plants, a radiator under a window
    for k in range(rng.integers(3, 9)):
        s = rng.uniform(0.15, 0.5, 3)
        x, y = rng.uniform(x0 + 0.2, x1 - 0.2), rng.uniform(y0 + 0.2, y1 - 0.2)
        rect = (x - s[0] / 2, y - s[1] / 2, x + s[0] / 2, y + s[1] / 2)
        if all(rect[2] <= o[0] or rect[0] >= o[2] or rect[3] <= o[1] or rect[1] >= o[3] for o in taken):
            boxes.append(Box(f"{prefix}clutter{k}", (x, y, s[2] / 2), tuple(s), pick(rng, {"wood": 1, "chipboard": 1, "metal": 0.3, "glass": 0.2})))
            taken.append(rect)
    return boxes, taken


def random_flat(rng, devices=2):
    """A flat of 2-4 rooms; one of them is measured. Interior walls have doors, open or shut."""
    width, depth, height = rng.uniform(8, 13), rng.uniform(6, 10), rng.uniform(2.6, 3.1)
    split_x = rng.uniform(0.35, 0.6) * width
    split_left = rng.uniform(0.4, 0.6) * depth if rng.random() < 0.8 else None
    split_right = rng.uniform(0.4, 0.6) * depth if rng.random() < 0.8 else None
    rooms = []
    for (rx0, rx1), split in (((0, split_x), split_left), ((split_x, width), split_right)):
        if split is None:
            rooms.append((rx0, 0, rx1, depth))
        else:
            rooms += [(rx0, 0, rx1, split), (rx0, split, rx1, depth)]

    outer, inner = pick(rng, EXTERIOR_WALLS), pick(rng, INTERIOR_WALLS)
    t = WALL

    def windows(length):
        return [(s, 1.2, 0.9, 2.1) for s in np.arange(1.0, length - 1.5, rng.uniform(2.5, 4.0))]

    boxes = [
        Box("floor", (width / 2, depth / 2, -t / 2), (width + 2 * t, depth + 2 * t, t), "concrete"),
        Box("ceiling", (width / 2, depth / 2, height + t / 2), (width + 2 * t, depth + 2 * t, t), "concrete"),
    ]
    for name, a, b, length in (("south", (0, -t / 2), (width, -t / 2), width), ("north", (0, depth + t / 2), (width, depth + t / 2), width),
                               ("west", (-t / 2, 0), (-t / 2, depth), depth), ("east", (width + t / 2, 0), (width + t / 2, depth), depth)):
        openings = windows(length)
        boxes += wall_with_openings(f"wall_{name}", a, b, t, height, outer, openings)
        boxes += [Box(f"window_{name}_{k}", (a[0] + (s + 0.6 if a[1] == b[1] else 0), a[1] + (s + 0.6 if a[0] == b[0] else 0), 1.5),
                      (1.2, 0.01, 1.2) if a[1] == b[1] else (0.01, 1.2, 1.2), "glass") for k, (s, *_ ) in enumerate(openings)]

    def door(length):
        return [(rng.uniform(0.3, max(0.31, length - DOOR_WIDTH - 0.3)), DOOR_WIDTH, DOOR_HEIGHT, height)]

    shut = []
    inner_walls = [(("inner_x", (split_x, 0), (split_x, depth), depth))]
    if split_left is not None:
        inner_walls.append(("inner_left", (0, split_left), (split_x, split_left), split_x))
    if split_right is not None:
        inner_walls.append(("inner_right", (split_x, split_right), (width, split_right), width - split_x))
    for name, a, b, length in inner_walls:
        openings = door(length)
        boxes += wall_with_openings(name, a, b, INNER_WALL, height, inner, openings)
        for start, w, *_ in openings:
            if rng.random() < 0.5:  # a shut door
                along = a[1] == b[1]
                c = (a[0] + start + w / 2, a[1], DOOR_HEIGHT / 2) if along else (a[0], a[1] + start + w / 2, DOOR_HEIGHT / 2)
                shut.append(Box(f"{name}_door", c, (w, 0.04, DOOR_HEIGHT) if along else (0.04, w, DOOR_HEIGHT), "wood"))
    boxes += shut

    measured = int(rng.integers(len(rooms)))
    kinds_of = list(ROOM_TYPES)
    furniture_rects, elsewhere = [], []  # footprints in every room: nobody walks through a wardrobe
    for k, rect in enumerate(rooms):
        inset = (rect[0] + 0.1, rect[1] + 0.1, rect[2] - 0.1, rect[3] - 0.1)
        room_type = "bedroom" if k == measured and rng.random() < 0.6 else kinds_of[rng.integers(len(kinds_of))]
        kinds = list(rng.permutation(ROOM_TYPES[room_type])[: rng.integers(2, 5)])
        f_boxes, taken = furnish(rng, inset, kinds, "furniture_" if k == measured else f"room{k}_")
        boxes += f_boxes
        furniture_rects += taken
        if k != measured:
            elsewhere.append((inset[0] + 0.3, inset[1] + 0.3, inset[2] - 0.3, inset[3] - 0.3))

    m = rooms[measured]
    hub, others = radios_in(rng, m, devices)
    if elsewhere and rng.random() < 0.2:  # one device in the next room: a link through a wall
        e = elsewhere[rng.integers(len(elsewhere))]
        others[-1] = (rng.uniform(e[0], e[2]), rng.uniform(e[1], e[3]), rng.uniform(0.5, 1.2))
    return Room("flat", width, depth, height, boxes, hub, others,
                walkable=(m[0] + 0.35, m[1] + 0.35, m[2] - 0.35, m[3] - 0.35),
                furniture=furniture_rects, elsewhere=elsewhere)


def radios_in(rng, rect, devices):
    x0, y0, x1, y1 = rect

    def spot():
        side, along = rng.integers(4), rng.uniform(0.2, 0.8)
        x = {0: x0 + along * (x1 - x0), 1: x0 + along * (x1 - x0), 2: x0 + 0.25, 3: x1 - 0.25}[side]
        y = {0: y0 + 0.25, 1: y1 - 0.25, 2: y0 + along * (y1 - y0), 3: y0 + along * (y1 - y0)}[side]
        return (x, y, rng.uniform(0.3, 1.2))

    hub, others = spot(), []
    for _ in range(200):
        if len(others) == devices:
            break
        s = spot()
        if np.hypot(s[0] - hub[0], s[1] - hub[1]) > 1.5:
            others.append(s)
    while len(others) < devices:
        others.append(spot())
    return hub, others


def random_hall(rng, devices=2):
    """A lecture hall: a lecturer's area at the front with the board on the desk, rows of seats behind
    (tiered or flat), students sitting in some of them, windows on one side, doors at the back.
    Half are classrooms (10-18 m, up to ~200 seats), half big amphitheatres (18-28 m, 300-700 seats,
    mostly full), because the demo hall can hold 500 students."""
    big = rng.random() < 0.5
    if big:
        width, depth, height = rng.uniform(18, 28), rng.uniform(16, 26), rng.uniform(5.0, 8.0)
    else:
        width, depth, height = rng.uniform(10, 18), rng.uniform(9, 16), rng.uniform(3.8, 5.5)
    tiered = rng.random() < 0.6
    outer = pick(rng, EXTERIOR_WALLS)
    t = WALL
    boxes = [
        Box("floor", (width / 2, depth / 2, -t / 2), (width + 2 * t, depth + 2 * t, t), "concrete"),
        Box("ceiling", (width / 2, depth / 2, height + t / 2), (width + 2 * t, depth + 2 * t, t), "concrete"),
    ]
    window_side = rng.integers(2)
    for name, a, b, length, has_windows in (
        ("front", (0, -t / 2), (width, -t / 2), width, False),
        ("back", (0, depth + t / 2), (width, depth + t / 2), width, False),
        ("west", (-t / 2, 0), (-t / 2, depth), depth, window_side == 0),
        ("east", (width + t / 2, 0), (width + t / 2, depth), depth, window_side == 1),
    ):
        openings = [(s, 1.5, 1.0, 2.6) for s in np.arange(1.5, length - 2, 3.0)] if has_windows else []
        if name == "back":
            openings = [(0.8, 1.6, 2.2, height), (width - 2.4, 1.6, 2.2, height)]
        boxes += wall_with_openings(f"wall_{name}", a, b, t, height, outer, openings)
    boxes.append(Box("board_wall", (width / 2, 0.03, 1.8), (min(6.0, width * 0.5), 0.02, 1.2), "chipboard"))

    lecturer = (1.0, 0.6, width - 1.0, 3.2)
    desk_x = rng.uniform(width * 0.3, width * 0.7)
    boxes.append(Box("furniture_desk", (desk_x, 1.6, 0.4), (1.8, 0.7, 0.8), pick(rng, {"wood": 1, "chipboard": 1})))
    furniture_rects = [(desk_x - 0.9, 1.25, desk_x + 0.9, 1.95)]

    rows, seats = [], []
    y, rise = 4.0, 0.0
    while y < depth - 1.2:
        rows.append((y, rise))
        if tiered:
            boxes.append(Box(f"step{len(rows)}", (width / 2, y + 0.45, rise / 2 + 0.01), (width - 0.3, 0.9, rise + 0.02), "concrete"))
        boxes.append(Box(f"seats{len(rows)}", (width / 2, y + 0.55, rise + 0.22), (width - 2.6, 0.45, 0.44), pick(rng, {"chipboard": 1, "metal": 0.6, "wood": 0.6})))
        boxes.append(Box(f"desks{len(rows)}", (width / 2, y + 0.1, rise + 0.73), (width - 2.6, 0.35, 0.03), pick(rng, {"chipboard": 1, "wood": 1})))
        for x in np.arange(1.6, width - 1.5, 0.6):
            seats.append((float(x), y + 0.55, rise + 0.45))
        y += 0.9
        rise += 0.15 if tiered else 0.0

    # seated students, still: part of the static scene
    filled = rng.random(len(seats)) < (rng.uniform(0.5, 1.0) if big else rng.uniform(0.15, 0.8))
    for k, (x, sy, sz) in enumerate(seats):
        if filled[k]:
            boxes.append(Box(f"student{k}_torso", (x, sy, sz + 0.33), (0.36, 0.22, 0.6), "body"))
            boxes.append(Box(f"student{k}_head", (x, sy - 0.02, sz + 0.75), (0.17, 0.2, 0.22), "body"))
    free_seats = [s for k, s in enumerate(seats) if not filled[k]]

    # the board on the lecturer's desk, the phones across the lecturer's area
    hub = (desk_x + rng.uniform(-0.8, 0.8), 1.35, 0.82)
    others = []
    while len(others) < devices:
        others.append((rng.uniform(1.2, width - 1.2), rng.uniform(0.4, 3.4), rng.uniform(0.8, 1.4)))
        if np.hypot(others[-1][0] - hub[0], others[-1][1] - hub[1]) < 1.5:
            others.pop()
    aisles = [(0.4, 4.0, 1.1, depth - 0.6), (width - 1.1, 4.0, width - 0.4, depth - 0.6)]
    return Room("hall", width, depth, height, boxes, hub, others, walkable=lecturer,
                furniture=furniture_rects, elsewhere=aisles, seats=free_seats)


def unique_names(room):
    """A world may have two shelves or two mirrors; the scene needs every shape named once."""
    seen = {}
    for box in room.boxes:
        count = seen.get(box.name, 0)
        seen[box.name] = count + 1
        if count:
            box.name = f"{box.name}_{count}"
    return room


def random_room(rng, devices=2):
    return unique_names(random_hall(rng, devices) if rng.random() < 0.25 else random_flat(rng, devices))
