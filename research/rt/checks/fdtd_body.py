"""The full-wave ruler: a body crossing a link, solved with Maxwell's equations (2-D FDTD, TMz) and with
the ray tracer, side by side.

A 2-D slice treats the body as infinitely tall, which is close for the part of a person that sits
in a link's Fresnel zone (about 0.3 m wide at 2.4 GHz over 3 m). The body is a 0.36 x 0.22 m block
of muscle-like tissue (eps_r 52.7, sigma 1.7 S/m); the two ends are 3 m apart. Both solvers report
the gain with the body at each offset from the line, relative to no body.

    cd research/rt && ../.venv-rt/bin/python checks/fdtd_body.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from channel import BODY, CARRIER_HZ  # noqa: E402

C, EPS0, MU0 = 299_792_458.0, 8.854e-12, 4e-7 * np.pi
WAVELENGTH = C / CARRIER_HZ
DX = WAVELENGTH / 25
SPONGE = 40
TX, RX = (0.5, 1.5), (3.5, 1.5)
BODY_SIZE = (0.36, 0.22)  # across the link, along the link


def fdtd_gain(offset, width=4.0, height=3.0, periods=45):
    """|Ez| at the receiver (complex phasor) with the body centred `offset` metres off the link line."""
    nx, ny = int(width / DX), int(height / DX)
    eps = np.ones((nx, ny))
    sigma = np.zeros((nx, ny))
    if offset is not None:
        x = (np.arange(nx) + 0.5) * DX
        y = (np.arange(ny) + 0.5) * DX
        X, Y = np.meshgrid(x, y, indexing="ij")
        cx, cy = (TX[0] + RX[0]) / 2, TX[1] + offset
        inside = (np.abs(X - cx) < BODY_SIZE[1] / 2) & (np.abs(Y - cy) < BODY_SIZE[0] / 2)
        eps[inside], sigma[inside] = BODY["relative_permittivity"], BODY["conductivity"]
    # absorbing sponge: conductivity rising towards every edge, matched in H so it does not reflect
    edge = np.minimum.reduce([np.arange(nx)[:, None] * np.ones(ny), (nx - 1 - np.arange(nx))[:, None] * np.ones(ny),
                              np.ones(nx)[:, None] * np.arange(ny), np.ones(nx)[:, None] * (ny - 1 - np.arange(ny))])
    sponge = np.clip((SPONGE - edge) / SPONGE, 0, 1) ** 3 * 0.8 * EPS0 / (DX / C / 2)
    sigma = sigma + sponge
    sigma_m = sponge * MU0 / EPS0

    dt = 0.98 * DX / (C * np.sqrt(2))
    ca = (1 - sigma * dt / (2 * EPS0 * eps)) / (1 + sigma * dt / (2 * EPS0 * eps))
    cb = (dt / (EPS0 * eps * DX)) / (1 + sigma * dt / (2 * EPS0 * eps))
    da = (1 - sigma_m * dt / (2 * MU0)) / (1 + sigma_m * dt / (2 * MU0))
    db = (dt / (MU0 * DX)) / (1 + sigma_m * dt / (2 * MU0))
    ez, hx, hy = np.zeros((nx, ny)), np.zeros((nx, ny)), np.zeros((nx, ny))
    sx, sy = int(TX[0] / DX), int(TX[1] / DX)
    rx, ry = int(RX[0] / DX), int(RX[1] / DX)
    steps = int(periods / (CARRIER_HZ * dt))
    omega = 2 * np.pi * CARRIER_HZ
    phasor, settle = 0j, steps // 2
    for n in range(steps):
        hx[:, :-1] = da[:, :-1] * hx[:, :-1] - db[:, :-1] * (ez[:, 1:] - ez[:, :-1])
        hy[:-1, :] = da[:-1, :] * hy[:-1, :] + db[:-1, :] * (ez[1:, :] - ez[:-1, :])
        ez[1:, 1:] = ca[1:, 1:] * ez[1:, 1:] + cb[1:, 1:] * ((hy[1:, 1:] - hy[:-1, 1:]) - (hx[1:, 1:] - hx[1:, :-1]))
        ramp = min(1.0, n / (steps * 0.15))
        ez[sx, sy] += ramp * np.sin(omega * n * dt)
        if n >= settle:
            phasor += ez[rx, ry] * np.exp(-1j * omega * n * dt)
    return phasor


def ray_gain(offset, diffraction):
    import sionna.rt as rt

    from channel import material_xml

    xml = '<scene version="2.1.0">' + material_xml("body")
    if offset is None:
        offset = 50.0
    cx, cy = (TX[0] + RX[0]) / 2, TX[1] + offset
    xml += (f'<shape type="cube" id="person"><transform name="to_world"><scale x="{BODY_SIZE[1] / 2}" y="{BODY_SIZE[0] / 2}" z="20"/>'
            f'<translate x="{cx}" y="{cy}" z="0"/></transform><ref id="body" name="bsdf"/></shape></scene>')
    path = os.path.join(tempfile.mkdtemp(), "s.xml")
    open(path, "w").write(xml)
    scene = rt.load_scene(path)
    scene.frequency = CARRIER_HZ
    scene.tx_array = scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter("tx", position=[TX[0], TX[1], 0.0]))
    scene.add(rt.Receiver("rx", position=[RX[0], RX[1], 0.0]))
    paths = rt.PathSolver()(scene, max_depth=3, diffraction=diffraction, edge_diffraction=diffraction, refraction=True,
                            samples_per_src=300_000)
    h = np.asarray(paths.cfr(frequencies=[CARRIER_HZ], normalize=False, normalize_delays=False, out_type="numpy"))
    return h.ravel()[0]


def main():
    offsets = np.round(np.arange(0.0, 0.61, 0.05), 2)
    free = fdtd_gain(None)
    rt_free = {d: ray_gain(None, d) for d in (False, True)}
    print(f"{'offset m':>9} {'FDTD dB':>9} {'rays dB':>9} {'rays+diffraction dB':>21}")
    rows = []
    for off in offsets:
        f = 20 * np.log10(abs(fdtd_gain(off)) / abs(free))
        r = [20 * np.log10(abs(ray_gain(off, d)) / abs(rt_free[d]) + 1e-12) for d in (False, True)]
        rows.append((off, f, *r))
        print(f"{off:9.2f} {f:9.2f} {r[0]:9.2f} {r[1]:21.2f}", flush=True)
    np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fdtd_body.csv"), np.array(rows),
               delimiter=",", header="offset_m,fdtd_db,rays_db,rays_diffraction_db", fmt="%.3f")


if __name__ == "__main__":
    main()
