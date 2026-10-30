"""Ray-traced channel of a room with a person in it: one GPU solve gives every link's frequency response.

The hub transmits and every device receives; by reciprocity this is the same channel the hub
measures from the device's packets.
"""

import os
import tempfile

import numpy as np
import sionna.rt as rt

from body import PARTS

CARRIER_HZ = 2.437e9  # channel 6
SUBCARRIER_SPACING_HZ = 312.5e3
# 802.11 L-LTF: bins -26..-1 and 1..26 carry pilots; the rest (and DC) are empty, as in the ESP32 buffer.
BIN_INDEX = np.arange(-32, 32)
SIGNAL_BINS = (np.abs(BIN_INDEX) <= 26) & (BIN_INDEX != 0)

# Without edge diffraction a body on the direct path shadows it by ~10 dB, within the 5-15 dB measured
# for people at 2.4 GHz; with it the shadow came out ~3 dB and each frame took 30% longer.
# Muscle-like tissue at 2.4 GHz (Gabriel et al. 1996): eps_r ~52.7, sigma ~1.7 S/m.
BODY = {"relative_permittivity": 52.7, "conductivity": 1.7}
SCATTERING = {"concrete": 0.4, "brick": 0.4, "plasterboard": 0.3, "glass": 0.1, "wood": 0.3, "chipboard": 0.3, "metal": 0.05}
THICKNESS = {"concrete": 0.15, "brick": 0.15, "plasterboard": 0.03, "glass": 0.006, "wood": 0.03, "chipboard": 0.02, "metal": 0.002}


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
    def __init__(self, room, max_depth=4, samples=300_000, diffraction=False, diffuse=True):
        materials = {b.material for b in room.boxes} | {"body"}
        shapes = [box_xml(b.name, b.centre, b.size, b.material, b.yaw) for b in room.boxes]
        # Person parts are unit cubes, sized and placed every frame.
        shapes += [box_xml(f"person_{p}", (0, 0, -50), (1, 1, 1), "body") for p in PARTS]
        xml = '<scene version="2.1.0">' + "".join(material_xml(m) for m in materials) + "".join(shapes) + "</scene>"
        path = os.path.join(tempfile.mkdtemp(prefix="room_"), "room.xml")
        with open(path, "w") as f:
            f.write(xml)
        self.scene = rt.load_scene(path, merge_shapes_exclude_regex="person_.*")
        self.scene.frequency = CARRIER_HZ
        array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        self.scene.tx_array = array
        self.scene.rx_array = array
        self.scene.add(rt.Transmitter("hub", position=[float(v) for v in room.hub]))
        for k, device in enumerate(room.devices):
            self.scene.add(rt.Receiver(f"device_{k}", position=[float(v) for v in device]))
        self.parts = {p: self.scene.get(f"person_{p}") for p in PARTS}
        self.solver = rt.PathSolver()
        self.solve_options = dict(
            max_depth=max_depth,
            samples_per_src=samples,
            diffuse_reflection=diffuse,
            diffraction=diffraction,
            edge_diffraction=diffraction,
            refraction=True,
        )
        self.frequencies = CARRIER_HZ + BIN_INDEX * SUBCARRIER_SPACING_HZ

    def pose(self, parts):
        for name, pose in parts.items():
            obj = self.parts[name]
            obj.scaling = [float(v) for v in pose.size]
            obj.orientation = [float(v) for v in pose.orientation]
            obj.position = [float(v) for v in pose.centre]

    def response(self, seed=0):
        """Complex frequency response per device, `[devices, 64]` in -32..31 bin order."""
        paths = self.solver(self.scene, seed=seed, **self.solve_options)
        h = paths.cfr(frequencies=self.frequencies, normalize=False, normalize_delays=False, out_type="numpy")
        h = np.asarray(h).reshape(len(self.scene.receivers), -1)
        h[:, ~SIGNAL_BINS] = 0
        return h
