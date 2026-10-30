"""Breathing and heart rate from one link's CSI, while the person keeps still.

The last WINDOW_SECONDS of packets are averaged into BIN_HZ frames of spectrum shape (dB per
subcarrier, the packet's level removed). Each subcarrier is detrended and band-passed; the strongest
common component across subcarriers (first principal component) is the chest's motion. Its spectral
peak gives the rate, and peak power over the median of the band gives a signal-to-noise ratio:
a breathing peak well above the noise means someone is there, even when nothing moves.

Heart rate uses the same component in 0.8-2.0 Hz. The heart moves the chest wall by well under a
millimetre, so on one single-antenna link it is experimental: shown with its own SNR, never alone.
"""
from collections import deque

import numpy as np

from drift_detector import shape_db

BIN_HZ = 10
WINDOW_SECONDS = 30
BREATH_HZ = (0.1, 0.5)
HEART_HZ = (0.8, 2.0)
PRESENT_SNR = 6.0


def principal_motion(frames, band, rate):
    """Band-passed first principal component of `frames` [time, subcarrier], unit variance."""
    x = frames - frames.mean(0)
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x))[:, None])
    spectrum = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    spectrum[(freqs < band[0]) | (freqs > band[1])] = 0
    filtered = np.fft.irfft(spectrum, n=len(x), axis=0)
    _, _, vt = np.linalg.svd(filtered, full_matrices=False)
    component = filtered @ vt[0]
    return component / (component.std() + 1e-12)


def rate_and_snr(signal, band, rate, pad=8):
    n = len(signal) * pad
    power = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n=n)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / rate)
    inside = (freqs >= band[0]) & (freqs <= band[1])
    peak = np.argmax(np.where(inside, power, 0))
    rest = inside & (np.abs(freqs - freqs[peak]) > 0.04)
    return 60 * freqs[peak], float(power[peak] / (np.median(power[rest]) + 1e-12))


class Vitals:
    def __init__(self):
        self.frames = deque(maxlen=WINDOW_SECONDS * BIN_HZ)
        self.bin, self.packets = None, []

    def add(self, t, iq):
        bin_ = int(t * BIN_HZ)
        if self.bin is not None and bin_ != self.bin and self.packets:
            self.frames.append(np.mean(self.packets, axis=0))
            self.packets = []
        self.bin = bin_
        self.packets.append(shape_db(iq))

    def estimate(self):
        """{breath_bpm, breath_snr, heart_bpm, heart_snr, wave} over the last window, or None."""
        if len(self.frames) < self.frames.maxlen * 0.8:
            return None
        frames = np.array(self.frames)
        breath = principal_motion(frames, BREATH_HZ, BIN_HZ)
        heart = principal_motion(frames, HEART_HZ, BIN_HZ)
        breath_bpm, breath_snr = rate_and_snr(breath, BREATH_HZ, BIN_HZ)
        heart_bpm, heart_snr = rate_and_snr(heart, HEART_HZ, BIN_HZ)
        return {
            "breath_bpm": breath_bpm,
            "breath_snr": breath_snr,
            "heart_bpm": heart_bpm,
            "heart_snr": heart_snr,
            "present": breath_snr > PRESENT_SNR,
            "wave": breath[-15 * BIN_HZ:].round(3).tolist(),
        }
