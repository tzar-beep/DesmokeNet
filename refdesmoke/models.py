"""Lightweight desmoking networks.

One class, three modes, identical decoder and heads so ablations compare like with like:
    single   smoky frame only (the single-image baseline)
    concat   smoky and reference stacked as 6 input channels
    gated    separate encoders; at every scale a learned gate decides how much reference
             content to inject. The full-resolution gate is the model's "reference validity"
             map and is returned for analysis.

Every mode predicts a residual image and a per-pixel log-scale `log_b` of a Laplace
distribution; b is the expected absolute error and drives the confidence map.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSBlock(nn.Module):
    """Depthwise-separable residual block (ConvNeXt-style, small)."""

    def __init__(self, c, expand=2):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.norm = nn.GroupNorm(1, c)
        self.pw1 = nn.Conv2d(c, c * expand, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(c * expand, c, 1)

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


def stage(c, depth):
    return nn.Sequential(*[DSBlock(c) for _ in range(depth)])


class Encoder(nn.Module):
    def __init__(self, cin, widths, depth):
        super().__init__()
        self.stem = nn.Conv2d(cin, widths[0], 3, padding=1)
        self.stages = nn.ModuleList(stage(w, depth) for w in widths)
        self.downs = nn.ModuleList(
            nn.Conv2d(widths[i], widths[i + 1], 2, stride=2) for i in range(len(widths) - 1))

    def forward(self, x):
        feats = []
        x = self.stem(x)
        for i, s in enumerate(self.stages):
            x = s(x)
            feats.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return feats


class GatedFusion(nn.Module):
    """fused = f_smoky + g * proj(f_ref), with g in [0, 1] predicted per pixel."""

    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(3 * c, c, 1), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1, groups=c), nn.GELU(),
            nn.Conv2d(c, 1, 1))
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, fs, fr):
        g = torch.sigmoid(self.gate(torch.cat([fs, fr, (fs - fr).abs()], 1)))
        return fs + g * self.proj(fr), g


class RefDesmoker(nn.Module):
    MODES = ("single", "concat", "gated")

    def __init__(self, mode="gated", widths=(32, 64, 128, 192), depth=2):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode, self.widths, self.depth = mode, tuple(widths), depth
        self.factor = 2 ** (len(widths) - 1)

        self.enc = Encoder(6 if mode == "concat" else 3, widths, depth)
        if mode == "gated":
            self.enc_ref = Encoder(3, widths, depth)
            self.fusions = nn.ModuleList(GatedFusion(w) for w in widths)

        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in range(len(widths) - 1))
        self.dec = nn.ModuleList(stage(w, depth) for w in widths[:-1])
        self.head_img = nn.Conv2d(widths[0], 3, 3, padding=1)
        self.head_conf = nn.Conv2d(widths[0], 1, 3, padding=1)

    def config(self):
        return {"mode": self.mode, "widths": list(self.widths), "depth": self.depth}

    def forward(self, smoky, ref=None):
        if self.mode != "single" and ref is None:
            raise ValueError(f"mode={self.mode!r} needs a reference frame")
        h, w = smoky.shape[-2:]
        ph, pw = (-h) % self.factor, (-w) % self.factor
        pad = lambda t: F.pad(t, (0, pw, 0, ph), mode="reflect") if (ph or pw) else t
        x = pad(smoky)

        gate = None
        if self.mode == "single":
            feats = self.enc(x)
        elif self.mode == "concat":
            feats = self.enc(torch.cat([x, pad(ref)], 1))
        else:
            fs, fr = self.enc(x), self.enc_ref(pad(ref))
            feats, gates = [], []
            for fuse, a, b in zip(self.fusions, fs, fr):
                f, g = fuse(a, b)
                feats.append(f)
                gates.append(g)
            gate = gates[0][..., :h, :w]

        y = feats[-1]
        for i in reversed(range(len(self.ups))):
            y = self.dec[i](self.ups[i](y) + feats[i])

        out = (x + self.head_img(y))[..., :h, :w]
        log_b = self.head_conf(y).clamp(-10, 3)[..., :h, :w]
        return {"out": out, "log_b": log_b, "gate": gate}


def build_model(cfg):
    return RefDesmoker(mode=cfg.get("mode", "gated"),
                       widths=cfg.get("widths", (32, 64, 128, 192)),
                       depth=cfg.get("depth", 2))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
