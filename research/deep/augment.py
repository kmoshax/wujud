"""Receiver and environment nuisances the simulator may get wrong, applied to gridded crops on the GPU."""
import torch
import torch.nn.functional as F
from deep.preprocess import WINDOW


def time_warp(amp, phase, rssi):
    """crops of any length -> WINDOW samples (a longer crop plays the room faster)"""
    a = F.interpolate(amp.transpose(1, 2), size=WINDOW, mode="linear", align_corners=False).transpose(1, 2)
    r = F.interpolate(rssi[:, None], size=WINDOW, mode="linear", align_corners=False)[:, 0]
    return a, _warp_phasor(phase), r


def _warp_phasor(phase):
    phasor = torch.polar(torch.ones_like(phase), phase)
    parts = torch.cat([phasor.real, phasor.imag], 2).transpose(1, 2)
    warped = F.interpolate(parts, size=WINDOW, mode="linear", align_corners=False).transpose(1, 2)
    k = phase.shape[2]
    return torch.atan2(warped[..., k:], warped[..., :k])


def smooth_subcarrier_curve(batch, subcarriers, device, order=3):
    k = torch.linspace(-1, 1, subcarriers, device=device)
    basis = torch.stack([k ** i for i in range(1, order + 1)], 0)
    return torch.randn(batch, order, device=device) @ basis


def gain_drift(amp, strength):
    """slow, frequency-selective gain change across the window (dB)"""
    b, t, k = amp.shape
    shape = smooth_subcarrier_curve(b, k, amp.device)
    ramp = torch.linspace(-0.5, 0.5, t, device=amp.device)[None, :, None]
    return amp + strength[:, None, None] * ramp * shape[:, None, :]


def agc_jumps(amp, probability, strength):
    """at a random instant the frequency response shifts in a step (gain-table switch)"""
    b, t, k = amp.shape
    jump = torch.rand(b, device=amp.device) < probability
    at = (torch.rand(b, device=amp.device) * t).long()
    after = torch.arange(t, device=amp.device)[None] >= at[:, None]
    shape = smooth_subcarrier_curve(b, k, amp.device) * strength[:, None]
    return amp + (jump[:, None] & after)[..., None] * shape[:, None, :]


def amplitude_noise(amp, sigma):
    """quantisation/estimation noise: larger on weak subcarriers (std in dB ~ 1/linear amplitude)"""
    weakness = 10 ** (-(amp - amp.mean((1, 2), keepdim=True)) / 20)
    return amp + sigma[:, None, None] * weakness.clamp(max=4) * torch.randn_like(amp)


def packet_gaps(x, probability, max_gap=20):
    """grid samples lost to dropped packets: hold the last value"""
    b, t = x.shape[:2]
    keep = torch.ones(b, t, dtype=torch.bool, device=x.device)
    for _ in range(3):
        start = (torch.rand(b, device=x.device) * t).long()
        length = (torch.rand(b, device=x.device) * max_gap).long()
        span = torch.arange(t, device=x.device)[None]
        lost = (span >= start[:, None]) & (span < (start + length)[:, None]) & (torch.rand(b, 1, device=x.device) < probability)
        keep &= ~lost
    keep[:, 0] = True
    held = torch.cummax(torch.where(keep, torch.arange(t, device=x.device)[None], torch.zeros_like(keep, dtype=torch.long)), 1).values
    return held


def apply_index(x, held):
    shape = held.shape + (1,) * (x.dim() - 2)
    return torch.gather(x, 1, held.view(shape).expand_as(x))


def augment(amp, phase, rssi, level=1.0):
    b, device = amp.shape[0], amp.device
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(b, device=device)
    amp = gain_drift(amp, u(0, 2.0 * level))
    amp = agc_jumps(amp, 0.3 * level, u(0, 1.5 * level))
    amp = amplitude_noise(amp, u(0, 0.4 * level))
    phase = phase + u(0, 0.2 * level)[:, None, None] * torch.randn_like(phase)
    rssi = rssi + torch.round(u(0, 1.0 * level)[:, None] * torch.randn_like(rssi))
    held = packet_gaps(amp, 0.5 * level)
    amp, phase, rssi = apply_index(amp, held), apply_index(phase, held), apply_index(rssi, held)
    flip = torch.rand(b, device=device) < 0.5
    amp = torch.where(flip[:, None, None], amp.flip(2), amp)
    phase = torch.where(flip[:, None, None], -phase.flip(2), phase)
    return amp, phase, rssi
