"""Motion from how fast the shape of the spectrum drifts, one second to the next.

Each second, the packets of that second are averaged into one spectrum shape (dB per subcarrier, the
packet's overall level removed, so gain and power steps cancel). Drift is the mean absolute change
between consecutive shapes. A still room (empty, or someone breathing) drifts ~0.03-0.05 dB; a person
walking across the link ~0.1-0.3 dB. Found on the 2026-10-31 recordings: moving vs still AUC 0.92 with
the phone on the window, while the 0.5-s amplitude spread could not tell them apart.
"""
from collections import deque

import numpy as np

BIN = np.arange(64) - 32
SIGNAL = (np.abs(BIN) <= 26) & (BIN != 0)
MOVING_ABOVE = 0.07
STILL_BELOW = 0.055
SMOOTHING_SECONDS = 3


def shape_db(iq):
    """One packet's spectrum shape: dB per signal subcarrier, its mean removed. `iq`: 128 int8 bytes."""
    pairs = np.frombuffer(iq, dtype=np.int8).astype(float).reshape(64, 2)
    db = 20 * np.log10(np.hypot(pairs[:, 0], pairs[:, 1])[SIGNAL] + 1)
    return db - db.mean()


class DriftDetector:
    def __init__(self):
        self.second, self.packets = None, []
        self.shapes = deque(maxlen=2)
        self.drifts = deque(maxlen=SMOOTHING_SECONDS)
        self.moving = False

    def add(self, t, iq):
        """Feeds one packet; returns (second, drift, "moving" | "still") when a second closes, else None."""
        second = int(t)
        if self.second is None:
            self.second = second
        out = None
        if second != self.second:
            out = self.close()
            self.second, self.packets = second, []
        self.packets.append(shape_db(iq))
        return out

    def close(self):
        if len(self.packets) < 20:
            self.shapes.clear()
            return None
        self.shapes.append(np.mean(self.packets, axis=0))
        if len(self.shapes) < 2:
            return None
        self.drifts.append(float(np.abs(self.shapes[1] - self.shapes[0]).mean()))
        drift = float(np.median(self.drifts))
        if drift > MOVING_ABOVE:
            self.moving = True
        elif drift < STILL_BELOW:
            self.moving = False
        return self.second + 1, drift, "moving" if self.moving else "still"


def run(rows):
    """`rows`: (unix time, iq bytes) in time order. Returns (second, drift, verdict) per second."""
    detector, out = DriftDetector(), []
    for t, iq in rows:
        said = detector.add(t, iq)
        if said:
            out.append(said)
    return out
