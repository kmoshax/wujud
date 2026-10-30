"""Cached streams -> 20 s training crops (simulation, unlabeled real) and evaluation windows (real)."""
import json
import numpy as np
import torch
from deep.preprocess import WINDOW, HOP

CACHE = "cache/deep"
INPUTS = ("amp", "phase", "rssi", "static")


class SimSessions:
    def __init__(self, validation_fraction=0.05):
        self.amp = torch.from_numpy(np.load(f"{CACHE}/sim_amp.npy"))
        self.phase = torch.from_numpy(np.load(f"{CACHE}/sim_phase.npy"))
        self.rssi = torch.from_numpy(np.load(f"{CACHE}/sim_rssi.npy"))
        self.meta = json.load(open(f"{CACHE}/sim_meta.json"))
        self.label = torch.tensor([m["label"] for m in self.meta])
        rooms = np.array([m["room"] for m in self.meta])
        validation_rooms = rooms % round(1 / validation_fraction) == 0
        self.train_index = np.nonzero(~validation_rooms)[0]
        self.validation_index = np.nonzero(validation_rooms)[0]

    def balanced_sessions(self, rng, count):
        train_labels = self.label[self.train_index].numpy()
        weight = 1.0 / np.bincount(train_labels)[train_labels]
        return rng.choice(self.train_index, count, p=weight / weight.sum())


class RealRecordings:
    def __init__(self):
        self.amp = torch.from_numpy(np.load(f"{CACHE}/real_amp.npy"))
        self.phase = torch.from_numpy(np.load(f"{CACHE}/real_phase.npy"))
        self.rssi = torch.from_numpy(np.load(f"{CACHE}/real_rssi.npy"))
        self.meta = json.load(open(f"{CACHE}/real_meta.json"))
        lengths = np.array([m["grid"] for m in self.meta])
        self.offset = np.r_[0, np.cumsum(lengths)[:-1]]
        self.length = lengths

    def recordings(self, role):
        return [i for i, m in enumerate(self.meta) if m["role"] == role]

    def window_starts(self, recording):
        return self.offset[recording] + np.arange(0, self.length[recording] - WINDOW + 1, HOP)

    def random_crops(self, rng, role, count, length=WINDOW):
        recordings = np.array(self.recordings(role))
        chosen = rng.choice(recordings, count)
        starts = self.offset[chosen] + (rng.random(count) * (self.length[chosen] - length)).astype(int)
        return starts


def gather(streams, flat_starts, length):
    """streams: (amp [N,K], phase [N,K], rssi [N]) concatenated along time; -> crops [B,length,...] float32"""
    index = torch.as_tensor(flat_starts)[:, None] + torch.arange(length)[None]
    amp, phase, rssi = streams
    return amp[index].float(), phase[index].float(), rssi[index].float()


def gather_sessions(sim, sessions, starts, length):
    s = torch.as_tensor(sessions)[:, None]
    t = torch.as_tensor(starts)[:, None] + torch.arange(length)[None]
    return sim.amp[s, t].float(), sim.phase[s, t].float(), sim.rssi[s, t].float()


def noise_floor_db(amp):
    """per-window receiver noise level: median over subcarriers of the sample-to-sample jitter (dB)"""
    jitter = (amp[:, 1:] - amp[:, :-1]).std(1, keepdim=True) / 2 ** 0.5
    return jitter.median(2, keepdim=True).values.clamp(min=0.02)


def model_input(amp, phase, rssi, inputs):
    """crops [B,T,K] (amp dB, phase rad), [B,T] rssi -> [B,C,T,K]; everything relative to the window itself"""
    channels = []
    amp_mean = amp.mean(1, keepdim=True)
    if "amp" in inputs:
        channels.append(amp - amp_mean)
    if "ampn" in inputs:
        channels.append((amp - amp_mean) / noise_floor_db(amp))
    if "phase" in inputs:
        phasor = torch.polar(torch.ones_like(phase), phase)
        reference = phasor.mean(1, keepdim=True)
        channels.append(torch.angle(phasor * reference.conj() / (reference.abs() + 1e-6)))
    if "rssi" in inputs:
        channels.append(((rssi - rssi.mean(1, keepdim=True)) / 3.0)[..., None].expand_as(amp))
    if "static" in inputs:
        profile = amp_mean - amp_mean.mean(2, keepdim=True)
        channels.append((profile / 3.0).expand_as(amp))
    return torch.stack(channels, 1)
