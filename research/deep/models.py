"""Encoders over a [channels, time, subcarrier] window and the shared heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURES = 128


def conv_block(c_in, c_out, kernel, stride):
    padding = tuple(k // 2 for k in kernel)
    return nn.Sequential(nn.Conv2d(c_in, c_out, kernel, stride, padding, bias=False), nn.BatchNorm2d(c_out), nn.GELU())


class Cnn2d(nn.Module):
    """time x subcarrier image"""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(channels, 32, (7, 3), (2, 1)), conv_block(32, 32, (7, 3), (1, 1)),
            conv_block(32, 64, (7, 3), (2, 2)), conv_block(64, 64, (7, 3), (1, 1)),
            conv_block(64, 96, (5, 3), (2, 2)), conv_block(96, 128, (5, 3), (2, 2)))

    def forward(self, x):
        return self.net(x).mean((2, 3))


class Cnn1d(nn.Module):
    """subcarriers (and input kinds) as channels, convolution over time"""
    def __init__(self, channels, subcarriers=52):
        super().__init__()
        layers, c = [], channels * subcarriers
        for width, stride in [(128, 2), (128, 1), (128, 2), (128, 1), (128, 2), (128, 2)]:
            layers += [nn.Conv1d(c, width, 7, stride, 3, bias=False), nn.BatchNorm1d(width), nn.GELU()]
            c = width
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.permute(0, 1, 3, 2).flatten(1, 2)).mean(2)


class TemporalTransformer(nn.Module):
    """one token per 1 s of the window"""
    def __init__(self, channels, subcarriers=52, patch=20, depth=4):
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(channels * subcarriers * patch, FEATURES)
        self.position = nn.Parameter(torch.zeros(1, 21, FEATURES))
        self.cls = nn.Parameter(torch.zeros(1, 1, FEATURES))
        layer = nn.TransformerEncoderLayer(FEATURES, 4, 256, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(FEATURES)

    def tokens(self, x):
        b, c, t, k = x.shape
        return x.view(b, c, t // self.patch, self.patch, k).permute(0, 2, 1, 3, 4).flatten(2)

    def forward(self, x):
        z = self.embed(self.tokens(x))
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1)
        z = z + self.position[:, :z.shape[1]]
        return self.norm(self.encoder(z)[:, 0])


class Spectral(nn.Module):
    """log power spectrum of every subcarrier series (0-10 Hz) as a frequency x subcarrier image"""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(channels, 32, (5, 3), (1, 1)), conv_block(32, 64, (5, 3), (2, 2)),
            conv_block(64, 96, (5, 3), (2, 2)), conv_block(96, 128, (3, 3), (2, 2)))

    def forward(self, x):
        window = torch.hann_window(x.shape[2], device=x.device)[None, None, :, None]
        power = torch.fft.rfft((x - x.mean(2, keepdim=True)) * window, dim=2).abs() ** 2
        return self.net(torch.log10(power[:, :, 1:] + 1e-4) + 2.0).mean((2, 3))


ENCODERS = {"cnn2d": Cnn2d, "cnn1d": Cnn1d, "transformer": TemporalTransformer, "spectral": Spectral}


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.weight * grad, None


class Detector(nn.Module):
    """3-class head (empty / still / moving), a still-vs-empty head, and an optional domain head (DANN)"""
    def __init__(self, encoder, channels):
        super().__init__()
        self.encoder = ENCODERS[encoder](channels)
        self.classes = nn.Sequential(nn.Dropout(0.2), nn.Linear(FEATURES, 3))
        self.still = nn.Sequential(nn.Dropout(0.2), nn.Linear(FEATURES, 1))
        self.domain = nn.Sequential(nn.Linear(FEATURES, 64), nn.GELU(), nn.Linear(64, 1))
        self.projection = nn.Sequential(nn.Linear(FEATURES, FEATURES), nn.GELU(), nn.Linear(FEATURES, 64))

    def forward(self, x):
        z = self.encoder(x)
        return z, self.classes(z), self.still(z)[:, 0]

    def domain_logit(self, z, reversal_weight):
        return self.domain(GradientReversal.apply(z, reversal_weight))[:, 0]

    def embedding(self, z):
        return F.normalize(self.projection(z), dim=1)
