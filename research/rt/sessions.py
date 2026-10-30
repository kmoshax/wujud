"""One recorded session of a room: a person doing one activity, as the ESP32 would have reported it.

The ray tracer is solved at a frame rate fast enough for the motion (walking 20 Hz, still 10 Hz);
packets in between take the channel interpolated between the two frames around them, then go
through the radio model.
"""

import numpy as np

from body import Person, hidden, lying, sitting, standing
from radio import Radio

ACTIVITIES = ("away", "lie", "sit", "stand", "walk")
FRAME_HZ = {"walk": 20.0, "lie": 5.0, "sit": 5.0, "stand": 5.0, "away": 1.0}


def free_spot(rng, room, margin=0.35):
    x0, y0, x1, y1 = room.walkable
    for _ in range(200):
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        clear_of_furniture = all(
            x < r[0] - margin or x > r[2] + margin or y < r[1] - margin or y > r[3] + margin for r in room.furniture
        )
        clear_of_radios = all(np.hypot(x - p[0], y - p[1]) > 0.4 for p in [room.hub, *room.devices])
        if clear_of_furniture and clear_of_radios:
            return np.array([x, y])
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def furniture_named(room, kind):
    return [b for b in room.boxes if b.name == f"furniture_{kind}"]


def breathing(rng, seconds, frame_hz):
    """Chest position in [-1, 1] per frame: a rate that wanders a little, as real breathing does."""
    bpm = rng.uniform(10, 22)
    rate = bpm / 60 * (1 + 0.08 * np.cumsum(rng.normal(0, 1, int(seconds * frame_hz))) / np.sqrt(frame_hz * 30))
    phase = 2 * np.pi * np.cumsum(rate) / frame_hz + rng.uniform(0, 2 * np.pi)
    return np.sin(phase), bpm


def walk_track(rng, room, seconds, frame_hz):
    speed = rng.uniform(0.5, 1.3)
    position, target = free_spot(rng, room), free_spot(rng, room)
    gait = 0.0
    xs, yaws, gaits = [], [], []
    for _ in range(int(seconds * frame_hz)):
        step = target - position
        if np.linalg.norm(step) < 0.15:
            target = free_spot(rng, room)
            step = target - position
        heading = np.arctan2(step[1], step[0])
        position = position + speed / frame_hz * step / np.linalg.norm(step)
        gait += 2 * np.pi * (0.9 + 0.6 * speed) / frame_hz
        xs.append(position.copy())
        yaws.append(heading)
        gaits.append(gait)
    return np.array(xs), np.array(yaws), np.array(gaits), speed


def poses(rng, room, activity, seconds, frame_hz):
    """Body part poses per frame, plus the truth (position, breathing rate) to keep."""
    frames = max(int(seconds * frame_hz), 2)
    person = Person(height=rng.uniform(1.55, 1.9), chest_excursion=rng.uniform(0.004, 0.012))
    if activity == "away":
        return [hidden()] * frames, {"xy": None, "breath_bpm": None}
    if activity == "walk":
        xs, yaws, gaits, speed = walk_track(rng, room, seconds, frame_hz)
        frames_out = [standing(person, tuple(x), y, gait_phase=g) for x, y, g in zip(xs, yaws, gaits)]
        return frames_out, {"xy": xs.tolist(), "breath_bpm": None, "speed": speed}

    breath, bpm = breathing(rng, seconds, frame_hz)
    beds, desks = furniture_named(room, "bed"), furniture_named(room, "desk")
    yaw = rng.uniform(0, 2 * np.pi)
    if activity == "lie" and beds:
        bed = beds[0]
        xy, yaw, surface = np.array(bed.centre[:2]), 0.0 if bed.size[0] >= bed.size[1] else np.pi / 2, bed.size[2]
    elif activity == "lie":
        xy, surface = free_spot(rng, room, margin=0.9), 0.02
    elif activity == "sit" and desks and rng.random() < 0.6:
        desk = desks[0]
        xy = np.array(desk.centre[:2]) + rng.normal(0, 0.1, 2)
        xy = np.clip(xy, [0.4, 0.4], [room.width - 0.4, room.depth - 0.4])
        surface = 0.45
    else:
        xy, surface = free_spot(rng, room), 0.45
    # Small sway: a person holding still still moves by a centimetre or so.
    sway = np.cumsum(rng.normal(0, 0.0007, (frames, 2)), axis=0)
    out = []
    for k in range(frames):
        at = tuple(xy + sway[k])
        if activity == "lie":
            out.append(lying(person, at, yaw, surface_height=surface, breath=breath[k]))
        elif activity == "sit":
            out.append(sitting(person, at, yaw, seat_height=surface, breath=breath[k]))
        else:
            out.append(standing(person, at, yaw, breath=breath[k]))
    return out, {"xy": xy.tolist(), "breath_bpm": bpm}


def record(channel, room, activity, seconds, packet_hz, rng, fast=False):
    """Returns iq `[links, packets, 64, 2]` int8, rssi `[links, packets]`, packet times, truth."""
    frame_hz = FRAME_HZ[activity] / (2 if fast and activity == "walk" else 1)
    frames, truth = poses(rng, room, activity, seconds, frame_hz)
    responses = []
    for k, parts in enumerate(frames):
        if k == 0 or activity != "away":
            channel.pose(parts)
            responses.append(channel.response())
        else:
            responses.append(responses[0])
    responses = np.array(responses)  # [frames, links, 64]

    times = np.cumsum(rng.exponential(1 / packet_hz, int(seconds * packet_hz * 1.2)))
    times = times[times < (len(frames) - 1) / frame_hz]
    radios = [Radio.random(rng) for _ in range(responses.shape[1])]
    iq = np.zeros((len(radios), len(times), 64, 2), np.int8)
    rssi = np.zeros((len(radios), len(times)), np.int16)
    for p, t in enumerate(times):
        f = t * frame_hz
        k = int(f)
        w = f - k
        h = (1 - w) * responses[k] + w * responses[min(k + 1, len(responses) - 1)]
        for link, radio in enumerate(radios):
            iq[link, p], rssi[link, p] = radio.packet(h[link], rng)
    return iq, rssi, times, truth
