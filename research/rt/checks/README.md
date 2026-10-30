# Is the simulator right? Checks against physics and real data

Every row below is a test of the simulator, rerunnable with `cd research/rt && ../.venv-rt/bin/python checks/<file>`. Results are from 2026-10-30.

## 1. Free space and walls (`physics_checks.py`)

| check | simulator | expected |
|---|---|---|
| line of sight, 1–8 m | equal to Friis to 0.00 dB | Friis `20 log10(λ / 4πd)` |
| concrete wall, normal incidence | \|Γ\| = 0.42 | about 0.39 (ITU-R P.2040, 15 cm slab) |

## 2. A body crossing a link: full-wave FDTD as the ruler (`fdtd_body.py`, `rounded_body_test.py`)

The setup is a 2-D FDTD solve of Maxwell's equations (TMz, cell λ/25, absorbing edges). A block of muscle-like tissue 0.36 × 0.22 m sits between two antennas 3 m apart. The table gives the gain relative to no body:

| offset from the line | FDTD | ray tracer, box body | ray tracer, round body |
|---|---|---|---|
| 0 m | −13.5 dB | **−240 dB** (cut off) | −240 dB |
| 0.2 m | −6.7 dB | −24.8 dB | 0.0 dB |
| 0.3–0.55 m | −3 to +2 dB | −5 to +6 dB | 0.0 dB |

- The ray tracer does not bend waves around a body; even with Sionna's diffraction switched on it found no path.
- A flat box reflects like a mirror and exaggerates the ripple.
- A rounded body loses the reflection altogether.

**What the simulator does now:**

- The direct path is not traced. It is computed as free space times the Fresnel–Kirchhoff diffraction of every body in the way.
- Each body is a thick obstacle: 0.7 of a screen at its middle, 0.3 of screens at its front and back.
- Against the FDTD this is off by **2.4 dB rms**, from the line out to 0.6 m.

## 3. How much walking disturbs the CSI (`compare_real.py` metrics)

The measure is the spread of each subcarrier's amplitude over 0.5 s, averaged over subcarriers, as a ratio to an empty room:

| | walking / empty | still / empty |
|---|---|---|
| real, ESPectre (20–25 recordings per class) | **1.85** | 0.98 |
| ray tracer alone | 1.0–1.6 (median ~1.3) | 0.92 |
| ray tracer + body scattering, RCS 1 × part area | median ~1.4 | |
| ray tracer + body scattering, RCS 4 × part area | median ~1.8 | |
| ray tracer + body scattering, RCS 16 × part area | median ~2.6 | |

**What changed:**

- Tracing rate (10, 20 or 40 Hz), rays (100k or 300k), bounces (3 or 4) and the body's diffuse coefficient (0.3 or 0.8) did not change the walking ratio.
- The cause is that rays launched from the board rarely hit something as small as a person.
- So each body part is added as a radar scatterer, directly and through one bounce off each wall, the floor and the ceiling.
- Its radar cross-section is 5 × the part's largest face, about 1–2 m² for a whole person, which is the order measured for people at 2.4 GHz.

## Not checked yet

- Breathing depth against a real still person with a known breathing rate.
- 5 GHz.
- The hall, against a real hall recording.
