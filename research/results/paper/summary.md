# wujud: Wi-Fi Sensing Trained on Physics Simulation Only

## Abstract
We present wujud, a Wi-Fi channel-state-information (CSI) sensing system for low-cost ESP32 boards that detects people in a room, counts students in a lecture hall, and detects a breathing person behind a wall. All models are trained exclusively on a 3-D physics simulator (image-method multipath, diffuse scattering, Fresnel–Kirchhoff body diffraction, breathing and fidget kinematics, Sabine absorption, wall attenuation, and ESP32 receiver impairments), and evaluated on real ESP32 recordings from the public ESPectre dataset (51 test recordings, five chip families, several rooms). Without seeing a single real sample, wujud detects motion with 100% recall and 0% false alarms, and outperforms a classifier trained on real data from other rooms. Run on the same recordings, the original on-device detector of RuView flags motion in 99.7% of windows, because its raw-phase statistic is dominated by per-packet clock offsets. We also find that a room's day-to-day drift exceeds the change caused by a still person (AUC 0.14 for a fixed empty-room calibration), and propose an occupancy latch that compares the static multipath profile with the empty room of a few minutes earlier; it raised no false presence on 108 minutes of real empty rooms.

## Key results
| Result | Value | Data |
|---|---|---|
| Motion detection (recall / false alarm) | 100% / 0% | real ESP32 |
| Motion false alarms, RuView on-device detector | 99.7% | real ESP32 |
| Still person vs. empty, AUC per 20 s (wujud / RuView) | 0.60 / 0.43 | real ESP32 |
| Still person vs. empty, trained on real data from other rooms | 0.52 | real ESP32 |
| Fixed empty-room calibration, tested on another day | AUC 0.14 | real ESP32 |
| Occupancy latch false presence, 108 min of empty rooms | 0% | real ESP32 |
| Lecture-hall count error (wujud / RTI), 12 boards | 9.0% / 29.4% of seats | simulation |
| Through-wall: still person detectable up to | ~18 dB | simulation |

## Limitations
- Distinguishing a still person from an empty room on a single link remains weak (AUC 0.52–0.65 across every method we tried, including models trained on real data); we attribute this to a physical limit of one link.
- Lecture-hall counting and through-wall results are from simulation and have not been validated on hardware.
- The occupancy latch matched, but did not exceed, the plain classifier on the two real enter-and-sit sessions available; its gain is measured in simulation only.
- The RuView comparison covers its on-device detector only; its server-side detector was not evaluated.

## Figure captions
1. Head-to-head with RuView's original on-device detector on real ESP32 recordings.
2. Simulation-to-real transfer: trained on simulation only vs. trained on real data from other rooms.
3. Lecture-hall counting in unseen halls vs. radio tomographic imaging, and error vs. number of boards (simulation).
4. Through-wall detection vs. wall attenuation (simulation).
5. Occupancy latch on real ESP32 data: empty room, walking, sitting still.
6. Breathing visibility: original simulator, calibrated simulator, and real ESP32 data.
