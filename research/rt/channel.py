"""Ray-traced channel of a world with people in it: one GPU solve gives every link's frequency response.

The hub transmits and every device receives; by reciprocity this is the same channel the hub
measures from the device's packets.
"""

import os
import tempfile

import numpy as np
import sionna.rt as rt
from scipy.special import fresnel

from body import HIDDEN, PARTS, rotation

CARRIER_HZ = 2.437e9  # channel 6
SUBCARRIER_SPACING_HZ = 312.5e3
# 802.11 L-LTF: bins -26..-1 and 1..26 carry pilots; the rest (and DC) are empty, as in the ESP32 buffer.
BIN_INDEX = np.arange(-32, 32)
SIGNAL_BINS = (np.abs(BIN_INDEX) <= 26) & (BIN_INDEX != 0)

# The ray tracer does not bend waves around a body: a person on the direct path cuts it off completely
# (checks/fdtd_body.py: -240 dB, against -13.5 dB from a full-wave FDTD solve). So the direct path is
# not traced: it is the free-space wave times the Fresnel-Kirchhoff diffraction of every body in the
# way (checks: 2.4 dB rms from FDTD across offsets 0-0.6 m), and the ray tracer gives everything else.
# Muscle-like tissue at 2.4 GHz (Gabriel et al. 1996): eps_r ~52.7, sigma ~1.7 S/m.
BODY = {"relative_permittivity": 52.7, "conductivity": 1.7}
# Rays launched from the board rarely hit something as small as a person, so the body's scattering in
# every direction (what makes walking anywhere in a room visible) is added as a radar scatterer per body
# part, directly and through one bounce off each wall, floor and ceiling of the measured area. Radar
# cross-section per part = BODY_RCS_PER_M2 x the part's largest face; calibrated so that walking
# disturbs the CSI as much, relative to an empty room, as in real ESP32 recordings (checks/README.md).
BODY_RCS_PER_M2 = float(os.environ.get("WUJUD_BODY_RCS", "5.0"))
WALL_BOUNCE = -0.4
SCATTERING = {"concrete": 0.4, "brick": 0.4, "plasterboard": 0.3, "glass": 0.1, "wood": 0.3, "chipboard": 0.3,
              "metal": 0.05, "marble": 0.2}
THICKNESS = {"concrete": 0.15, "brick": 0.15, "plasterboard": 0.03, "glass": 0.006, "wood": 0.03, "chipboard": 0.02,
             "metal": 0.002, "marble": 0.02}


def material_xml(name):
    if name == "body":
        return (
            '<bsdf type="radio-material" id="body">'
            f'<float name="relative_permittivity" value="{BODY["relative_permittivity"]}"/>'
            f'<float name="conductivity" value="{BODY["conductivity"]}"/>'
            '<float name="thickness" value="0.1"/><float name="scattering_coefficient" value="0.3"/></bsdf>'
        )
    return (
        f'<bsdf type="itu-radio-material" id="{name}"><string name="type" value="{name}"/>'
        f'<float name="thickness" value="{THICKNESS[name]}"/>'
        f'<float name="scattering_coefficient" value="{SCATTERING[name]}"/></bsdf>'
    )


def box_xml(name, centre, size, material, yaw=0.0):
    sx, sy, sz = (s / 2 for s in size)
    x, y, z = centre
    return (
        f'<shape type="cube" id="{name}"><transform name="to_world">'
        f'<scale x="{sx}" y="{sy}" z="{sz}"/><rotate z="1" angle="{np.degrees(yaw)}"/>'
        f'<translate x="{x}" y="{y}" z="{z}"/></transform><ref id="{material}" name="bsdf"/></shape>'
    )


class RoomChannel:
    def __init__(self, room, people=1, max_depth=4, samples=300_000, diffraction=False, diffuse=True):
        materials = {b.material for b in room.boxes} | {"body"}
        shapes = [box_xml(b.name, b.centre, b.size, b.material, b.yaw) for b in room.boxes]
        # Every moving person's parts are unit cubes, sized and placed every frame.
        shapes += [box_xml(f"person{k}_{p}", (0, 0, -50), (1, 1, 1), "body") for k in range(people) for p in PARTS]
        xml = '<scene version="2.1.0">' + "".join(material_xml(m) for m in materials) + "".join(shapes) + "</scene>"
        path = os.path.join(tempfile.mkdtemp(prefix="room_"), "room.xml")
        with open(path, "w") as f:
            f.write(xml)
        self.scene = rt.load_scene(path, merge_shapes_exclude_regex="person[0-9]+_.*")
        self.scene.frequency = CARRIER_HZ
        array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        self.scene.tx_array = array
        self.scene.rx_array = array
        self.scene.add(rt.Transmitter("hub", position=[float(v) for v in room.hub]))
        for k, device in enumerate(room.devices):
            self.scene.add(rt.Receiver(f"device_{k}", position=[float(v) for v in device]))
        self.people = [{p: self.scene.get(f"person{k}_{p}") for p in PARTS} for k in range(people)]
        self.solver = rt.PathSolver()
        self.solve_options = dict(
            max_depth=max_depth,
            samples_per_src=samples,
            diffuse_reflection=diffuse,
            diffraction=diffraction,
            edge_diffraction=diffraction,
            refraction=True,
        )
        self.solve_options["los"] = False
        self.frequencies = CARRIER_HZ + BIN_INDEX * SUBCARRIER_SPACING_HZ
        self.hub = np.array(room.hub, float)
        self.devices = np.array(room.devices, float).reshape(-1, 3)
        self.current = []
        x0, y0, x1, y1 = room.walkable
        margin = 0.35
        self.planes = [(0, x0 - margin), (0, x1 + margin), (1, y0 - margin), (1, y1 + margin), (2, 0.0), (2, room.height)]

    def pose(self, people):
        """`people`: one dict of part poses per moving person, in the order they were declared."""
        for objects, parts in zip(self.people, people):
            self.place(objects, parts)
        self.current = people

    def place(self, objects, parts):
        # Each change rebuilds the part's mesh on the GPU; most parts of most people do not move between
        # frames (bystanders out of sight, a still person's head), so only what changed is sent.
        placed = self.__dict__.setdefault("placed", {})
        for name, pose in parts.items():
            obj = objects[name]
            wanted = (tuple(map(float, pose.size)), tuple(map(float, pose.orientation)), tuple(map(float, pose.centre)))
            before = placed.get(id(obj))
            if before == wanted:
                continue
            if before is None or before[0] != wanted[0]:
                obj.scaling = list(wanted[0])
            if before is None or before[1] != wanted[1]:
                obj.orientation = list(wanted[1])
            if before is None or before[2] != wanted[2]:
                obj.position = list(wanted[2])
            placed[id(obj)] = wanted

    def response(self, seed=0, rays=False):
        """Complex frequency response per device, `[devices, 64]` in -32..31 bin order; with `rays`, also
        the strongest paths of the first link as polylines (for the live view)."""
        paths = self.solver(self.scene, seed=seed, **self.solve_options)
        h = paths.cfr(frequencies=self.frequencies, normalize=False, normalize_delays=False, out_type="numpy")
        h = np.asarray(h).reshape(len(self.scene.receivers), -1)
        h = h + self.direct_path() + self.body_scatter()
        h[:, ~SIGNAL_BINS] = 0
        return (h, strongest_paths(paths)) if rays else h


    def direct_path(self):
        """The hub-to-device line of sight of every link, `[devices, 64]`, diffracted round every body."""
        out = []
        for device in self.devices:
            distance = np.linalg.norm(device - self.hub)
            wavelength = 299_792_458.0 / self.frequencies
            free = wavelength / (4 * np.pi * distance) * np.exp(-2j * np.pi * distance / wavelength)
            for parts in self.current:
                free = free * body_diffraction(self.hub, device, parts, wavelength)
            out.append(free)
        return np.array(out)


    def body_scatter(self):
        """Every body part as a radar scatterer: hub -> part -> device, plus the same with one bounce off
        a wall, the floor or the ceiling on either side of the part (first-order images)."""
        wavelength = 299_792_458.0 / self.frequencies
        sources = [(self.hub, 1.0)] + [(mirror(self.hub, axis, at), WALL_BOUNCE) for axis, at in self.planes]
        out = np.zeros((len(self.devices), len(self.frequencies)), complex)
        for parts in self.current:
            for pose in parts.values():
                if pose.centre[2] < HIDDEN[2] / 2:
                    break
                size = sorted(pose.size)
                rcs = BODY_RCS_PER_M2 * size[1] * size[2]
                p = np.asarray(pose.centre, float)
                for k, device in enumerate(self.devices):
                    sinks = [(device, 1.0)] + [(mirror(device, axis, at), WALL_BOUNCE) for axis, at in self.planes]
                    for src, g1 in sources:
                        d1 = np.linalg.norm(p - src)
                        for snk, g2 in sinks:
                            if g1 != 1.0 and g2 != 1.0:
                                continue
                            d2 = np.linalg.norm(p - snk)
                            amplitude = g1 * g2 * np.sqrt(rcs / (4 * np.pi)) * wavelength / (4 * np.pi * d1 * d2)
                            out[k] += amplitude * np.exp(-2j * np.pi * (d1 + d2) / wavelength)
        return out


def mirror(point, axis, at):
    image = np.array(point, float)
    image[axis] = 2 * at - image[axis]
    return image


def fresnel_f(x):
    s, c = fresnel(x)
    return c + 1j * s


def body_diffraction(a, b, parts, wavelength):
    """Field behind a body relative to free space, for the straight path a -> b (Fresnel-Kirchhoff,
    Babinet: one minus what an aperture the shape of the body lets through). The body is the rectangle
    its parts cover, seen across the path at the point where it stands."""
    corners = []
    for pose in parts.values():
        if pose.centre[2] < HIDDEN[2] / 2:
            return 1.0
        half = np.array(pose.size) / 2
        r = rotation(*pose.orientation)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(np.asarray(pose.centre) + r @ (half * [sx, sy, sz]))
    corners = np.array(corners)
    along = (b - a) / np.linalg.norm(b - a)
    across = np.cross(along, [0, 0, 1.0])
    if np.linalg.norm(across) < 1e-6:
        return 1.0
    across /= np.linalg.norm(across)
    up = np.cross(across, along)
    t = (corners - a) @ along
    length = np.linalg.norm(b - a)
    if t.min() <= 0.05 or t.max() >= length - 0.05:
        return 1.0
    u, v = (corners - a) @ across, (corners - a) @ up

    def screen(d1):
        scale = np.sqrt(2 * length / (wavelength * d1 * (length - d1)))
        aperture = (1 / 2j) * (fresnel_f(scale * u.max()) - fresnel_f(scale * u.min())) * (
            fresnel_f(scale * v.max()) - fresnel_f(scale * v.min()))
        return 1 - aperture

    # A body is thick: one screen at its middle misses the shadow's depth, front and back screens
    # overdo it just beside the line; 0.7 / 0.3 of each was closest to FDTD (2.4 dB rms).
    middle, front_back = screen(t.mean()), screen(t.min()) * screen(t.max())
    return np.exp(0.7 * np.log(middle) + 0.3 * np.log(front_back))


def strongest_paths(paths, link=0, count=40):
    """Up to `count` paths of one link as lists of points, strongest first, each with its power in dB."""

    def array(x):
        return np.asarray(x.numpy() if hasattr(x, "numpy") else x)

    vertices = array(paths.vertices)[:, link, 0]  # [depth, paths, 3]
    valid = array(paths.valid)[link, 0]  # [paths]
    a_real, a_imag = (array(x)[link, 0, 0, 0] for x in paths.a)
    gain = a_real ** 2 + a_imag ** 2
    order = [i for i in np.argsort(-gain) if valid[i] and gain[i] > 0][:count]
    tx, rx = array(paths.sources)[:, 0], array(paths.targets)[:, link]
    out = []
    for i in order:
        points = [tx.tolist()] + [v.tolist() for v in vertices[:, i] if np.abs(v).sum() > 0] + [rx.tolist()]
        out.append({"points": points, "db": float(10 * np.log10(gain[i]))})
    return out
