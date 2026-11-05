# Which reading tells how far apart two boards are (2026-11-05)

Measured with `wujud layout-study` and short probes on the same simulated rooms (seeds 1,000,000+,
150 rooms, 900 links, a person walking for 30 s). The probe code was removed after the run; the
numbers below are from that run.

| reading per link | correlation with log distance |
|---|---|
| median RSSI | -0.66 |
| how unevenly the power spreads over the subcarriers (Rician K) | -0.30 |
| how deep the RSSI dips when the person crosses (median minus 2nd percentile) | 0.11 |

- The simulated RSSI falls 14 dB per tenfold distance (exponent 1.40) and scatters 6.0 dB around
  that line (`layout_after.json`), so one link alone gives its distance only within about 2.7 times.
- Three Wi-Fi channels (1, 6, 11) instead of one take 0.04 m off the mean error
  (`layout_after.json`): frequency diversity, and so BLE's three advertising channels, adds little.
- On the real recording `research/rec/guided_room_20261101_2120_test.csv` the fitted exponent is 3.36:
  real RSSI falls faster with distance than the simulator's.
- The boards are ESP32-D0WD-V3 (esptool, 2026-11-05): no Wi-Fi FTM (that is ESP32-S2/S3/C3 and later)
  and no Bluetooth channel sounding.

## Prior tuning

On rooms 2,000,000+ (never scored): link fading 3 or 6 dB, board gain spread 2-6 dB, exponent
spread 0.3-0.8, common gain spread 0.5-3 dB and room side spread 0.5-2 m all gave a 4-board mean
error within 1.04-1.14 m. The information in the RSSI, not the priors, sets the error.

## The CSI phase slope (2026-11-05)

`research/tools/csi_delay_pairs.py` reads each pair's phase slope across the subcarriers, both
directions averaged.

- **Steady and gain-free:** on the real boards it holds to about 0.3 ns over 20 s, the two
  directions agree, and a board's gain does not touch it.
- **Not absolute time of flight:** packets between boards with unsynchronised clocks spread only
  1.3-3 ns (interquartile), so the receiver re-times every packet to its arrival and the absolute
  delay is gone.
- **What is left is the multipath shape around the first path.** With a person moving it drifts
  0.5-1.4 ns over minutes. In the first square the diagonals were the largest pairs (4.3 and 3.3
  ns); after the hub was swapped they were the smallest.
- **Weak in simulation too:** re-timed to the direct path, the slope correlates -0.27 with
  distance, against -0.78 for the noise-free channel power (150 rooms, 900 links; probe removed
  after the run).
