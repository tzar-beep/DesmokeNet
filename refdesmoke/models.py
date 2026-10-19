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

    def forward(self, fs, fr, prior=None):
        logit = self.gate(torch.cat([fs, fr, (fs - fr).abs()], 1))
        if prior is not None:
            logit = logit + F.interpolate(prior, size=logit.shape[-2:], mode="bilinear", align_corners=False)
        g = torch.sigmoid(logit)
        return fs + g * self.proj(fr), g


class ChangeHead(nn.Module):
    """Predicts reference validity logits from the deepest features (whole-frame context).

    Moved objects are invisible to a 3x3 neighbourhood, so per-scale gates alone fail to
    find them; this logit is added to every scale's gate as a shared prior.
    """

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3 * c, c, 1), DSBlock(c), DSBlock(c), nn.Conv2d(c, 1, 1))

    def forward(self, fs, fr):
        return self.net(torch.cat([fs, fr, (fs - fr).abs()], 1))


class RefDesmoker(nn.Module):
    MODES = ("single", "concat", "gated")

    def __init__(self, mode="gated", widths=(32, 64, 128, 192), depth=2, conf_from_gate=False,
                 global_gate=False):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if (conf_from_gate or global_gate) and mode != "gated":
            raise ValueError("conf_from_gate and global_gate need mode='gated'")
        self.mode, self.widths, self.depth = mode, tuple(widths), depth
        self.conf_from_gate, self.global_gate = conf_from_gate, global_gate
        self.factor = 2 ** (len(widths) - 1)

        self.enc = Encoder(6 if mode == "concat" else 3, widths, depth)
        if mode == "gated":
            self.enc_ref = Encoder(3, widths, depth)
            self.fusions = nn.ModuleList(GatedFusion(w) for w in widths)
            if global_gate:
                self.change_head = ChangeHead(widths[-1])

        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in range(len(widths) - 1))
        self.dec = nn.ModuleList(stage(w, depth) for w in widths[:-1])
        self.head_img = nn.Conv2d(widths[0], 3, 3, padding=1)
        # With conf_from_gate the confidence head also sees where the reference was rejected.
        self.head_conf = nn.Conv2d(widths[0] + int(conf_from_gate), 1, 3, padding=1)

    def config(self):
        return {"mode": self.mode, "widths": list(self.widths), "depth": self.depth,
                "conf_from_gate": self.conf_from_gate, "global_gate": self.global_gate}

    def forward(self, smoky, ref=None):
        if self.mode != "single" and ref is None:
            raise ValueError(f"mode={self.mode!r} needs a reference frame")
        h, w = smoky.shape[-2:]
        ph, pw = (-h) % self.factor, (-w) % self.factor
        pad = lambda t: F.pad(t, (0, pw, 0, ph), mode="reflect") if (ph or pw) else t
        x = pad(smoky)

        gate = gates = None
        if self.mode == "single":
            feats = self.enc(x)
        elif self.mode == "concat":
            feats = self.enc(torch.cat([x, pad(ref)], 1))
        else:
            fs, fr = self.enc(x), self.enc_ref(pad(ref))
            prior = self.change_head(fs[-1], fr[-1]) if self.global_gate else None
            feats, gates = [], []
            for fuse, a, b in zip(self.fusions, fs, fr):
                f, g = fuse(a, b, prior)
                feats.append(f)
                gates.append(g)
            gate = gates[0][..., :h, :w]

        y = feats[-1]
        for i in reversed(range(len(self.ups))):
            y = self.dec[i](self.ups[i](y) + feats[i])

        out = (x + self.head_img(y))[..., :h, :w]
        conf_in = torch.cat([y, gates[0]], 1) if self.conf_from_gate else y
        log_b = self.head_conf(conf_in).clamp(-10, 3)[..., :h, :w]
        # `gates` (all scales, padded size) is for the gate-supervision loss.
        return {"out": out, "log_b": log_b, "gate": gate, "gates": gates}


def build_model(cfg):
    return RefDesmoker(mode=cfg.get("mode", "gated"),
                       widths=cfg.get("widths", (32, 64, 128, 192)),
                       depth=cfg.get("depth", 2),
                       conf_from_gate=cfg.get("conf_from_gate", False),
                       global_gate=cfg.get("global_gate", False))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
