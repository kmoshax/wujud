"""What the ESP32 makes of a channel: the int8 CSI buffer and the RSSI it reports.

Matched to the room firmware's own recordings (2026-10-30, ESP32 rev 3.1, phone at -51 dBm):
signal bins average |iq| ~16, packet-to-packet change in a quiet room ~0.35 dB, the receive
chain's own ripple across subcarriers ~2.4 dB. The buffer is in centred order (index i is bin
i - 32), bins -26..-1 and 1..26 carry signal, the rest are zero.
"""

from dataclasses import dataclass

import numpy as np

from channel import SIGNAL_BINS

TARGET_AMPLITUDE = 16.0
AGC_STEP_DB = 0.5
# The noise the CSI sees: ~27 dB SNR at -51 dBm, close to the measured 0.22-0.30 dB packet-to-packet change.
EFFECTIVE_NOISE_FLOOR_DBM = -78.0
RSSI_NOISE_DB = 0.6


@dataclass
class Radio:
    """One device-to-board link's hardware: transmit power, its receive-chain ripple, slow drifts."""

    tx_power_dbm: float
    ripple_db: np.ndarray
    rssi_bias_db: float = 0.0
    agc_db: float | None = None

    @classmethod
    def random(cls, rng):
        bins = np.arange(-32, 32)
        # A smooth ripple: a few cosines across the band, like the ESP32's analog filter.
        ripple = sum(
            rng.normal(0, 2.0) * np.cos(2 * np.pi * bins * rng.uniform(0.005, 0.04) + rng.uniform(0, 2 * np.pi))
            for _ in range(3)
        )
        return cls(tx_power_dbm=rng.uniform(12, 18), ripple_db=ripple)

    def packet(self, h, rng):
        """`h`: the channel `[64]` (centred bins, linear); returns (int8 `[64, 2]` as [imag, real], rssi)."""
        h = h * 10 ** (self.ripple_db / 20)
        power_dbm = self.tx_power_dbm + 10 * np.log10(np.mean(np.abs(h[SIGNAL_BINS]) ** 2) + 1e-30)
        self.rssi_bias_db += rng.normal(0, 0.02) - 0.001 * self.rssi_bias_db
        rssi = int(np.round(power_dbm + self.rssi_bias_db + rng.normal(0, RSSI_NOISE_DB)))

        # AGC: gain that brings the average amplitude near the target, in steps, with a little hysteresis.
        wanted = 20 * np.log10(TARGET_AMPLITUDE) - (power_dbm - self.tx_power_dbm)
        if self.agc_db is None or abs(wanted - self.agc_db) > 1.5 * AGC_STEP_DB:
            self.agc_db = AGC_STEP_DB * np.round(wanted / AGC_STEP_DB)
        y = h * 10 ** (self.agc_db / 20)

        snr_db = power_dbm - EFFECTIVE_NOISE_FLOOR_DBM
        noise_sigma = TARGET_AMPLITUDE / np.sqrt(2) * 10 ** (-snr_db / 20)
        y = y + noise_sigma * (rng.normal(size=64) + 1j * rng.normal(size=64))

        # Each packet starts at an unknown carrier phase and timing offset.
        bins = np.arange(-32, 32)
        y = y * np.exp(1j * (rng.uniform(0, 2 * np.pi) + rng.normal(0, 0.15) * bins))
        y[~SIGNAL_BINS] = 0
        iq = np.stack([np.clip(np.round(y.imag), -128, 127), np.clip(np.round(y.real), -128, 127)], axis=1)
        return iq.astype(np.int8), rssi
