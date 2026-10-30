"""An articulated person made of boxes: torso, head, two arms, two legs.

Each part is a box with a size, a centre and an orientation (yaw, pitch, roll in Sionna's z-y-x
order). Walking swings the legs and arms about the hips and shoulders; breathing moves the chest
wall by a few millimetres; sitting and lying fold or tip the body.
"""

from dataclasses import dataclass

import numpy as np

PARTS = ("torso", "head", "arm_left", "arm_right", "leg_left", "leg_right")
HIDDEN = (0.0, 0.0, -50.0)  # far below the floor: "nobody in the room"


@dataclass
class Person:
    height: float = 1.75
    # Front-to-back chest movement per breath, peak to peak (4-12 mm in adults at rest).
    chest_excursion: float = 0.008


@dataclass
class PartPose:
    centre: tuple
    size: tuple
    orientation: tuple


def rotation(yaw, pitch, roll):
    cz, sz, cy, sy, cx, sx = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz @ ry @ rx


def limb(pivot, length, thickness, yaw, swing):
    """A limb hanging from `pivot`, swung forward by `swing` radians."""
    r = rotation(yaw, swing, 0.0)
    centre = np.asarray(pivot) + r @ np.array([0.0, 0.0, -length / 2])
    return PartPose(tuple(centre), (thickness, thickness, length), (yaw, swing, 0.0))


def standing(person, xy, yaw, gait_phase=None, breath=0.0):
    """Upright; `gait_phase` (radians) swings the limbs as in walking, `breath` in [-1, 1]."""
    h = person.height
    leg, torso_h, head = 0.48 * h, 0.30 * h, 0.13 * h
    x, y = xy
    hip_z = leg
    side = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    swing = 0.0 if gait_phase is None else 0.45 * np.sin(gait_phase)
    bob = 0.0 if gait_phase is None else 0.02 * abs(np.sin(gait_phase))
    chest = 0.22 + person.chest_excursion / 2 * breath
    base = np.array([x, y, bob])
    parts = {
        "torso": PartPose(tuple(base + [0, 0, hip_z + torso_h / 2]), (chest, 0.36, torso_h), (yaw, 0.0, 0.0)),
        "head": PartPose(tuple(base + [0, 0, hip_z + torso_h + head / 2]), (0.19, 0.16, head), (yaw, 0.0, 0.0)),
    }
    for name, sign in (("left", 1), ("right", -1)):
        hip = base + [0, 0, hip_z] + 0.1 * sign * side
        shoulder = base + [0, 0, hip_z + torso_h * 0.95] + 0.22 * sign * side
        parts[f"leg_{name}"] = limb(hip, leg, 0.13, yaw, sign * swing)
        parts[f"arm_{name}"] = limb(shoulder, 0.62 * torso_h + 0.25, 0.09, yaw, -sign * 0.8 * swing)
    return parts


def sitting(person, xy, yaw, seat_height=0.45, breath=0.0):
    h = person.height
    torso_h, head = 0.30 * h, 0.13 * h
    x, y = xy
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    side = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    chest = 0.22 + person.chest_excursion / 2 * breath
    base = np.array([x, y, seat_height])
    parts = {
        "torso": PartPose(tuple(base + [0, 0, torso_h / 2]), (chest, 0.36, torso_h), (yaw, 0.0, 0.0)),
        "head": PartPose(tuple(base + [0, 0, torso_h + head / 2]), (0.19, 0.16, head), (yaw, 0.0, 0.0)),
    }
    for name, sign in (("left", 1), ("right", -1)):
        thigh_mid = base + 0.22 * fwd + 0.1 * sign * side
        parts[f"leg_{name}"] = PartPose(tuple(thigh_mid), (0.44, 0.13, 0.13), (yaw, 0.0, 0.0))
        shoulder = base + [0, 0, torso_h * 0.95] + 0.22 * sign * side
        parts[f"arm_{name}"] = limb(shoulder, 0.5, 0.09, yaw, 0.6)
    return parts


def lying(person, xy, yaw, surface_height=0.5, breath=0.0):
    """On the back along `yaw`; breathing lifts the chest (now facing up)."""
    h = person.height
    x, y = xy
    along = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    side = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    chest = 0.22 + person.chest_excursion / 2 * breath
    z = surface_height + chest / 2
    centre = np.array([x, y, z])
    flat = (yaw, np.pi / 2, 0.0)
    parts = {
        "torso": PartPose(tuple(centre), (chest, 0.36, 0.30 * h), flat),
        "head": PartPose(tuple(centre + along * (0.15 * h + 0.11)), (0.16, 0.19, 0.22), (yaw, 0.0, 0.0)),
    }
    for name, sign in (("left", 1), ("right", -1)):
        parts[f"leg_{name}"] = PartPose(
            tuple(centre - along * (0.15 * h + 0.24 * h) + 0.1 * sign * side), (0.13, 0.13, 0.48 * h), flat
        )
        parts[f"arm_{name}"] = PartPose(tuple(centre + 0.25 * sign * side), (0.09, 0.09, 0.45), flat)
    return parts


def hidden():
    return {name: PartPose(HIDDEN, (0.1, 0.1, 0.1), (0.0, 0.0, 0.0)) for name in PARTS}
