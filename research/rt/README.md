# Ray-traced CSI dataset

Physics instead of approximations. Each world is ray-traced on the GPU with [Sionna RT](https://github.com/NVlabs/sionna-rt), NVIDIA's open-source radio ray tracer. Every path is followed:

- reflections off walls, floor, ceiling, furniture and mirrors;
- transmission through them;
- diffuse scattering.

Each surface uses its ITU-R P.2040 electrical properties. People are articulated bodies (`body.py`) with the dielectric properties of muscle at 2.4 GHz. They walk, swinging arms and legs, or they sit, stand or lie still while their chest moves with breathing.

The channel then goes through a model of the ESP32 receiver (`radio.py`): AGC in steps, thermal noise, the receive chain's ripple, the packet's random phase, int8 rounding, and RSSI with a drifting bias. That model is matched to the room firmware's own recordings.

## The worlds are not empty boxes

| world | what is in it |
|---|---|
| **a room inside a flat** (75 %) | A flat of 2–4 rooms. Exterior walls are concrete or brick with glass windows. Interior walls are brick or plasterboard, with doors that are open or shut. Each room is furnished by its type: a bedroom, living room, kitchen (fridge and cooker in metal) or study, with mirrors and clutter on the floor. One room is measured; in 20 % of worlds one device sits in the next room, so its link goes through a wall. |
| **a lecture hall** (25 %) | 10–18 m wide, 3.8–5.5 m high, with tiered (60 %) or flat rows of seats and desks. 15–80 % of the seats hold still students. Windows run down one side, doors are at the back, and there is a board. The ESP32 sits on the lecturer's desk and the phones are spread across the lecturer's area. |

Each session has **the measured person** doing one activity: away, lie, sit, stand or walk. Up to **two other people** also move, and they are part of the world, not part of the label:

- someone walking or standing in another room of the flat;
- a student shifting in a seat or standing up.

A neighbour walking next door while the measured room is empty is still "away", which is exactly the case a real deployment has to get right.

| file | what |
|---|---|
| `rooms.py` | random flats and halls: walls with doors and windows, furniture by room type, seats and students, where the hub and the 1–3 devices go |
| `body.py` | a person as six boxes, posed for walking, sitting, standing, lying and breathing |
| `channel.py` | builds the Sionna scene; one solve gives the 64-bin response of every link and, on request, the strongest rays |
| `radio.py` | the ESP32's view of a channel: int8 CSI buffer (centred order) and RSSI |
| `sessions.py` | one 30 s session: the measured person plus the others. Frames are traced at 20 Hz while walking and 5 Hz while still, and packets at 80–130 per second are interpolated between frames |
| `generate.py` | the dataset: one `.npz` per world and activity; it resumes where it stopped |
| `team_generate.sh` | a range of worlds split over several workers on one GPU, each with a live view |
| `live.html` | the live 3D view of what is being traced |
| `compare_real.py` | the same statistics for ray-traced sessions and a real recording |

## Setup

Sionna RT needs its own environment (Python 3.12, Mitsuba 3, Dr.Jit on CUDA) and an NVIDIA GPU:

```bash
uv venv --python 3.12 research/.venv-rt
VIRTUAL_ENV=research/.venv-rt uv pip install sionna-rt
```

## Generating as a team

World N is the same on every machine, so each person takes a different range:

| who | command |
|---|---|
| person 1 | `research/rt/team_generate.sh 0 500` |
| person 2 | `research/rt/team_generate.sh 500 500` |
| person 3 | `research/rt/team_generate.sh 1000 500` |

Each world is about 2 MB. Put every `research/cache/rt/*.npz` in one shared folder; the file names do not collide.

Run it in its own memory-capped unit:

```bash
systemd-run --user --collect -p MemoryMax=10G -p MemorySwapMax=1G research/rt/team_generate.sh 0 500
```

The ray tracer keeps every world it has traced in memory. The script starts a fresh process every two worlds to stop that growth. The cap is the safety net: a runaway is stopped on its own, and nothing else is taken down with it. On 2026-10-30, four uncapped workers grew to 17 GB of RAM plus 23 GB of swap, and the system's out-of-memory killer closed the desktop app they had been started from.

## Watching it live

Run `python -m http.server 8765` in `research/`, then open http://127.0.0.1:8765/rt/live.html. The view shows:

- the world in 3D: walls see-through, glass blue, metal shiny;
- the measured person in blue and the others in orange;
- the 40 strongest rays from the ESP32 to the first phone, brighter when stronger, with pulses running along them;
- the CSI of every link scrolling on the right.

The buttons switch between workers.

## How close to real (2026-10-30)

These are compared with the phone link of the room firmware (`compare_real.py`):

| | packet-to-packet change | 0.5 s disturbance | ripple across subcarriers |
|---|---|---|---|
| ray-traced, still | 0.22–0.28 dB | 0.15–0.18 dB | 1.4–3.0 dB |
| ray-traced, walking | 0.23 dB | 0.19 dB | 2.0 dB |
| real, lying on the bed | 0.30 dB | 0.22 dB | 2.4 dB |
| real, walking | 0.32 dB | 0.25 dB | 2.4 dB |

In both, walking changes the shape of the spectrum only a little more than lying still does: ×1.3 ray-traced and ×1.1 real.

## Known limits

- **Edge diffraction is off.** With it on, a body on the direct path shadows it by only ~3 dB. With it off, the shadow is ~10 dB, which is within the 5–15 dB measured for people, and it runs faster.
- **Rays are launched at random.** A person moving by a centimetre can make a path flicker on or off, up to ~1 dB in a frame. A person holding still is unaffected: their frames repeat to within 0.02 dB, so breathing comes through cleanly.
- **Bodies are made of boxes.** That is enough for the shadow, the reflection and breathing, but not for detailed shapes. A full-wave solver (FDTD, e.g. openEMS) on one small scene would be the way to calibrate the body; for whole rooms it would take ~30 min per frame.
- **Not yet modelled:** other Wi-Fi traffic, fans and pets, moving doors, and the differences between ESP32 models (C3, S3, C6).
