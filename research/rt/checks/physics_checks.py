"""Sanity checks of the ray tracer against textbook physics, at the carrier the board uses.

    cd research/rt && ../.venv-rt/bin/python checks/physics_checks.py
"""
import os
import sys
import tempfile

import numpy as np
import sionna.rt as rt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from channel import CARRIER_HZ, THICKNESS, material_xml  # noqa: E402

C = 299_792_458.0
WAVELENGTH = C / CARRIER_HZ


def scene_with(boxes, materials):
    xml = '<scene version="2.1.0">' + "".join(material_xml(m) for m in materials)
    for name, centre, size, material in boxes:
        sx, sy, sz = (s / 2 for s in size)
        xml += (f'<shape type="cube" id="{name}"><transform name="to_world"><scale x="{sx}" y="{sy}" z="{sz}"/>'
                f'<translate x="{centre[0]}" y="{centre[1]}" z="{centre[2]}"/></transform><ref id="{material}" name="bsdf"/></shape>')
    path = os.path.join(tempfile.mkdtemp(), "s.xml")
    open(path, "w").write(xml + "</scene>")
    scene = rt.load_scene(path)
    scene.frequency = CARRIER_HZ
    scene.tx_array = scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    return scene


def gain_db(scene, tx, rx, **options):
    for name in list(scene.transmitters) + list(scene.receivers):
        scene.remove(name)
    scene.add(rt.Transmitter("tx", position=tx))
    scene.add(rt.Receiver("rx", position=rx))
    paths = rt.PathSolver()(scene, **options)
    h = np.asarray(paths.cfr(frequencies=[CARRIER_HZ], normalize=False, normalize_delays=False, out_type="numpy"))
    return 20 * np.log10(np.abs(h).ravel()[0])


def main():
    far = [("far", [0, 0, -100], [0.1, 0.1, 0.1], "concrete")]
    scene = scene_with(far, {"concrete"})
    print("free space (line of sight only) vs Friis 20 log10(lambda / 4 pi d):")
    for d in (1.0, 2.0, 4.0, 8.0):
        ray = gain_db(scene, [0, 0, 1], [d, 0, 1], max_depth=0)
        friis = 20 * np.log10(WAVELENGTH / (4 * np.pi * d))
        print(f"  d = {d:4.1f} m  ray tracer {ray:7.2f} dB   Friis {friis:7.2f} dB   diff {ray - friis:+.2f}")

    print("\nreflection off a concrete wall, normal incidence (ITU-R P.2040: |Gamma| about 0.39 at 2.4 GHz):")
    wall = [("wall", [3.0, 0, 1.0], [THICKNESS["concrete"], 20, 20], "concrete")]
    scene = scene_with(wall, {"concrete"})
    # tx and rx side by side 1 m from the wall: the reflected path is 2 m long, the direct one 0.2 m
    los = gain_db(scene, [2.0, -0.1, 1.0], [2.0, 0.1, 1.0], max_depth=0)
    both = gain_db(scene, [2.0, -0.1, 1.0], [2.0, 0.1, 1.0], max_depth=1, los=False, diffuse_reflection=False)
    path = 2 * np.hypot(1.0, 0.1)
    gamma = 10 ** ((both - 20 * np.log10(WAVELENGTH / (4 * np.pi * path))) / 20)
    print(f"  reflected path gain {both:.2f} dB over {path:.2f} m  ->  |Gamma| = {gamma:.2f}")


if __name__ == "__main__":
    main()
