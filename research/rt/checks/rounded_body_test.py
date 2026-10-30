import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sionna.rt as rt
from channel import CARRIER_HZ, material_xml

def cylinder_ply(path, sides=16):
    """Unit cylinder: radius 0.5 (so scaling gives the width), z from -0.5 to 0.5."""
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.c_[0.5 * np.cos(a), 0.5 * np.sin(a)]
    verts = [(x, y, z) for z in (-0.5, 0.5) for x, y in ring] + [(0, 0, -0.5), (0, 0, 0.5)]
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces += [(i, j, sides + j), (i, sides + j, sides + i), (2 * sides, j, i), (2 * sides + 1, sides + i, sides + j)]
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n"
                f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        f.writelines(f"{x} {y} {z}\n" for x, y, z in verts)
        f.writelines(f"3 {a} {b} {c}\n" for a, b, c in faces)

TX, RX, SIZE = (0.5, 1.5), (3.5, 1.5), (0.36, 0.22)
ply = os.path.join(tempfile.mkdtemp(), "cyl.ply"); cylinder_ply(ply)

def gain(offset, diffraction):
    cx, cy = 2.0, 1.5 + (offset if offset is not None else 50)
    xml = ('<scene version="2.1.0">' + material_xml("body") +
           f'<shape type="ply" id="person"><string name="filename" value="{ply}"/><transform name="to_world">'
           f'<scale x="{SIZE[1]}" y="{SIZE[0]}" z="40"/><translate x="{cx}" y="{cy}" z="0"/></transform><ref id="body" name="bsdf"/></shape></scene>')
    p = os.path.join(tempfile.mkdtemp(), "s.xml"); open(p, "w").write(xml)
    s = rt.load_scene(p); s.frequency = CARRIER_HZ
    s.tx_array = s.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    s.add(rt.Transmitter("tx", position=[*TX, 0.0])); s.add(rt.Receiver("rx", position=[*RX, 0.0]))
    paths = rt.PathSolver()(s, max_depth=3, diffraction=diffraction, edge_diffraction=diffraction, refraction=True, samples_per_src=300_000)
    return np.asarray(paths.cfr(frequencies=[CARRIER_HZ], normalize=False, normalize_delays=False, out_type="numpy")).ravel()[0]

fdtd = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fdtd_body.csv"), delimiter=",")
free = {d: gain(None, d) for d in (False, True)}
print(f"{'offset':>7} {'FDTD':>7} {'round':>7} {'round+diffr':>12}")
for off, f, *_ in fdtd:
    r = [20 * np.log10(abs(gain(off, d)) / abs(free[d]) + 1e-12) for d in (False, True)]
    print(f"{off:7.2f} {f:7.2f} {r[0]:7.2f} {r[1]:12.2f}", flush=True)
