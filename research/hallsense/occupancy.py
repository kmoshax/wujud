"""Occupancy latch: still-person detection that uses the one thing a single link does perfectly (motion).

Why: on real ESP32 data the room's day-to-day drift is LARGER than what a still person changes
(bench/self_calib.py: calibrate on one empty day, test on another -> AUC 0.14). So comparing 'now' with an empty
baseline from yesterday or from boot cannot work. But a person cannot appear without walking in, and motion is
detected at 100% / 0% false alarms. So:
  - while the room is empty, keep a FRESH reference (the last few minutes of empty windows)
  - motion -> someone is here (moving)
  - motion stops -> compare the quiet room with the reference from minutes ago, not days:
      far from it  -> a person stayed (still)
      close to it  -> they left (empty), and the reference keeps refreshing
Same idea as PIR occupancy sensors' latch + timeout, but the 'timeout' is replaced by evidence.
"""
import collections
import numpy as np


class OccupancyLatch:
    def __init__(self, cols=None, thr=3.0, ref_len=24, min_ref=6, clear_after=3, ref_max_age=120, sd_floor=0.2,
                 no_entry_factor=3.0):
        self.sd_floor = sd_floor                          # dB: integer I/Q and AGC make anything finer meaningless
        # nobody can appear without walking in: with no motion since the room was last judged empty, a change
        # of the profile is much more likely slow drift (temperature, a door, a neighbour's AP) than a person,
        # so it needs no_entry_factor x more evidence. Right after motion the plain threshold applies.
        self.no_entry_factor, self.motion_since_empty = no_entry_factor, False
        self.cols, self.thr, self.min_ref, self.clear_after = cols, thr, min_ref, clear_after
        self.ref = collections.deque(maxlen=ref_len)     # recent EMPTY windows (2 min at 5 s hop)
        self.ref_max_age = ref_max_age                   # windows; an older reference is not trusted
        self.state, self.n, self.near, self.ref_t = "unknown", 0, 0, -10 ** 9

    def distance(self, x):
        """x: the window's static profile (hallsense.roomfeat.profile_windows); robust z vs the fresh empty reference"""
        R = np.array(self.ref); c = self.cols if self.cols is not None else slice(None)
        R = R[:, c]; mu = np.median(R, 0)
        sd = np.maximum(1.4826 * np.median(np.abs(R - mu), 0), self.sd_floor)
        return float(np.sqrt(np.mean(((x[c] - mu) / sd) ** 2)))

    def step(self, x, p_motion, p_still=None):
        """x: window feature vector; p_motion: classifier probability of motion; p_still: classifier
        still-vs-empty probability, used only when there is no fresh reference -> state, distance"""
        self.n += 1
        if p_motion > 0.5:
            self.state, self.near, self.motion_since_empty = "moving", 0, True
            return self.state, None
        fresh = len(self.ref) >= self.min_ref and self.n - self.ref_t <= self.ref_max_age
        if not fresh:                                     # e.g. after a long stay: the empty reference is stale
            if p_still is not None:
                self.state = "still" if p_still > 0.5 else "empty"
            elif self.state in ("moving", "still"):
                self.state = "still"                      # latched: someone came in and nobody was seen leaving
            else:
                self.state = "empty"
            if self.state == "empty":
                self.ref.append(x); self.ref_t = self.n
            return self.state, None
        d = self.distance(x)
        thr = self.thr if (self.motion_since_empty or self.state == "still") else self.thr * self.no_entry_factor
        if d > thr:
            self.state, self.near = "still", 0
        else:
            self.near += 1
            if self.state != "still" or self.near >= self.clear_after:
                self.state = "empty"
        if self.state == "empty":
            self.ref.append(x); self.ref_t = self.n; self.motion_since_empty = False
        return self.state, d
