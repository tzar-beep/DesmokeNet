"""One interface over every method, shared by evaluation and the demo app.

    method = load_method("dcp")                  # classical, no training
    method = load_method("runs/gated/best.pt")   # trained checkpoint
    res = method(smoky, ref)                     # float RGB tensors (B, 3, H, W) in [0, 1]
    res["out"], res["log_b"] (or None), res["gate"] (or None)

`confidence_from_log_b` maps the predicted Laplace scale b (expected absolute error, in [0, 1]
intensity units) to a display confidence exp(-b / tau).
"""
from pathlib import Path

import numpy as np
import torch

from .dcp import dcp_desmoke
from .models import build_model

BASELINES = ("dcp", "identity", "paste_ref")


def confidence_from_log_b(log_b, tau=0.05):
    return torch.exp(-torch.exp(log_b) / tau)


class ModelMethod:
    def __init__(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = build_model(ckpt["config"]["model"]).to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.needs_ref = self.model.mode != "single"
        self.name = Path(ckpt_path).parent.name or Path(ckpt_path).stem

    @torch.no_grad()
    def __call__(self, smoky, ref=None):
        return self.model(smoky, ref if self.needs_ref else None)


class BaselineMethod:
    def __init__(self, name):
        self.name = name
        self.needs_ref = name == "paste_ref"

    @torch.no_grad()
    def __call__(self, smoky, ref=None):
        if self.name == "identity":
            out, log_b = smoky, None
        elif self.name == "paste_ref":
            out, log_b = ref, None
        else:  # dcp: log_b proxy from transmission so the app can show a confidence map
            outs, lbs = [], []
            for img in smoky.permute(0, 2, 3, 1).cpu().numpy():
                o, t = dcp_desmoke(img)
                outs.append(torch.from_numpy(o).permute(2, 0, 1))
                lbs.append(torch.from_numpy(np.log(0.25 * (1 - t) + 1e-3))[None])
            out = torch.stack(outs).to(smoky.device)
            log_b = torch.stack(lbs).to(smoky.device).float()
        return {"out": out, "log_b": log_b, "gate": None}


def load_method(spec, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if spec in BASELINES:
        return BaselineMethod(spec)
    return ModelMethod(spec, device)


def to_batch(img_u8, device):
    """HxWx3 uint8 RGB -> (1, 3, H, W) float tensor."""
    t = torch.from_numpy(np.ascontiguousarray(img_u8)).float().div(255)
    return t.permute(2, 0, 1)[None].to(device)


def to_u8(t):
    return (t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
