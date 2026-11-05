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
