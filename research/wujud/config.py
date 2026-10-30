"""Central constants for wujud.

Target hardware: ESP32 / ESP32-S3, 2.4 GHz HT20, CSI from the L-LTF
(52 data subcarriers, indices -26..-1, 1..26), int8 I/Q, RSSI in dBm.
Every number the firmware also uses lives here so sim and firmware agree.
"""
import numpy as np

C = 299_792_458.0
FC = 2.437e9                      # channel 6
LAM = C / FC                      # ~0.123 m
DF = 312.5e3
SC_IDX = np.r_[-26:0, 1:27]       # 52 L-LTF data subcarriers
FREQS = FC + SC_IDX * DF
K = len(SC_IDX)

# feature grouping: 52 subcarriers -> 13 groups of 4 adjacent subcarriers
N_GROUPS = 13
GROUP_OF = np.repeat(np.arange(N_GROUPS), 4)

FS = 20.0            # packets/s each node broadcasts
WIN_S = 1.0          # feature window (s) -> what each node reports per source per second
CALIB_S = 30         # empty-hall calibration length (s)
SESSION_S = 60       # observation length (s)

NOISE_FLOOR_DBM = -94.0
OUTLIER_DB = 8.0     # packets whose RSSI deviates more than this from the EMA are dropped
