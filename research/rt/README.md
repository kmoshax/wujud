# Ray-traced CSI dataset

Physics instead of approximations. Each room is ray-traced on the GPU with [Sionna RT](https://github.com/NVlabs/sionna-rt), NVIDIA's open-source radio ray tracer. Every path is followed:

- reflections off walls, floor, ceiling, furniture and mirrors;
- transmission through them;
- diffuse scattering.

Each surface uses its ITU-R P.2040 electrical properties. The person is an articulated body (`body.py`) with the dielectric properties of muscle at 2.4 GHz. They walk, swinging arms and legs, or they sit, stand or lie still while their chest moves with breathing.

The channel then goes through a model of the ESP32 receiver (`radio.py`): AGC in steps, thermal noise, the receive chain's ripple, the packet's random phase, int8 rounding, and RSSI with a drifting bias. That model is matched to the room firmware's own recordings.

| file | what |
|---|---|
| `rooms.py` | random rooms: size, wall material, 2–4 pieces of furniture against the walls, mirrors, where the hub and 1–3 devices go |
| `body.py` | the person as six boxes, posed for walking, sitting, standing, lying and breathing |
| `channel.py` | builds the Sionna scene; one solve gives the 64-bin response of every link |
| `radio.py` | the ESP32's view of a channel: int8 CSI buffer (centred order) and RSSI |
| `sessions.py` | one 30 s session per activity: frames at 20 Hz while walking and 5 Hz while still, with packets interpolated between frames at 80–130 per second |
| `generate.py` | the dataset: one `.npz` per room and activity in `research/cache/rt/`; it resumes where it stopped |
| `compare_real.py` | the same statistics for ray-traced sessions and a real recording |

## Setup

Sionna RT needs its own environment (Python 3.12, Mitsuba 3, Dr.Jit on CUDA):

```bash
uv venv --python 3.12 research/.venv-rt
VIRTUAL_ENV=research/.venv-rt uv pip install sionna-rt
cd research/rt && ../.venv-rt/bin/python generate.py --out ../cache/rt --rooms 400
```

On an RTX 4050 laptop GPU one room (5 activities × 30 s, 1–3 links) takes 1–4 minutes.

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

- Edge diffraction is off. With it on, a body on the direct path shadows it by only ~3 dB. With it off, the shadow is ~10 dB, which is within the 5–15 dB measured for people. It also runs faster.
- The ray tracer launches rays at random. A person moving by a centimetre can make a path flicker on or off, up to ~1 dB in a frame. A person holding still is unaffected: their frames repeat exactly, so breathing comes through cleanly.
- Every room is a box. Doors, windows and other rooms are not modelled.
