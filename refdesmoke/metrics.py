"""Evaluation metrics, including the density-binned protocol and uncertainty calibration.

Density-binned protocol
-----------------------
SmokeBench reports one average per method, which hides the thick-smoke failures. We split
pixels by local *visibility*: the ratio of local contrast in the smoky frame to local contrast
in the GT, computed on non-overlapping patches of the Y channel:

    visibility = std(smoky_patch) / std(gt_patch)

Patches with an almost flat GT (std < FLAT_STD) carry no structure to recover and are excluded.
Bins: thick < 0.3 <= medium < 0.6 <= thin. PSNR and SSIM are then computed per bin over all
test pixels, alongside the usual per-image averages. The thresholds are a protocol choice and
are reported with the results.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

BINS = {"thick": (0.0, 0.3), "medium": (0.3, 0.6), "thin": (0.6, float("inf"))}
FLAT_STD = 0.02
PATCH = 32


def rgb_to_y(x):
    """BT.601 luma in [0, 1] from RGB in [0, 1]; x is (B, 3, H, W)."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return (16 + 65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def psnr_from_mse(mse):
    return float("inf") if mse <= 0 else 10 * math.log10(1.0 / mse)


def _gaussian_window(size=11, sigma=1.5, device="cpu"):
    ax = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_map(x, y):
    """Per-pixel SSIM for single-channel (B, 1, H, W) tensors in [0, 1]."""
    win = _gaussian_window(device=x.device)
    pad = win.shape[-1] // 2
    f = lambda t: F.conv2d(F.pad(t, [pad] * 4, mode="reflect"), win)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))


def visibility_map(smoky_y, gt_y, patch=PATCH):
    """Patch visibility upsampled to pixels, plus a mask of textured (non-flat) patches."""
    h, w = gt_y.shape[-2:]
    ph, pw = (-h) % patch, (-w) % patch
    s = F.pad(smoky_y, (0, pw, 0, ph), mode="replicate")
    g = F.pad(gt_y, (0, pw, 0, ph), mode="replicate")
    std = lambda t: (F.avg_pool2d(t * t, patch) - F.avg_pool2d(t, patch) ** 2).clamp_min(0).sqrt()
    sg = std(g)
    vis = std(s) / (sg + 1e-6)
    up = lambda t: F.interpolate(t, scale_factor=patch, mode="nearest")[..., :h, :w]
    return up(vis), up((sg > FLAT_STD).float())


def sparsification_ause(err, unc, steps=20):
    """Area Under the Sparsification Error curve (lower is better) and Spearman rho.

    Removing the most-uncertain pixels first should lower the remaining error as fast as
    removing the truly worst pixels first (the oracle). AUSE is the gap between the two.
    """
    err, unc = np.asarray(err, np.float64), np.asarray(unc, np.float64)
    n = len(err)
    by_unc, by_err = np.argsort(-unc), np.argsort(-err)
    fracs = np.linspace(0, 0.95, steps)
    curve = lambda order: np.array([err[order[int(f * n):]].mean() for f in fracs])
    c_unc, c_orc = curve(by_unc), curve(by_err)
    ause = float(np.trapezoid(c_unc - c_orc, fracs))
    rank = lambda a: np.argsort(np.argsort(a))
    rho = float(np.corrcoef(rank(err), rank(unc))[0, 1])
    return {"ause": ause, "spearman": rho, "fracs": fracs.tolist(),
            "curve_unc": c_unc.tolist(), "curve_oracle": c_orc.tolist()}


class Evaluator:
    """Accumulates per-image and density-binned metrics over a test set (batch size 1)."""

    def __init__(self, unc_samples_per_image=20000, seed=0):
        self.per_image = []
        self.bins = {k: {"sse": 0.0, "n": 0, "ssim": 0.0} for k in list(BINS) + ["changed", "unchanged"]}
        self.err_s, self.unc_s = [], []
        self.k = unc_samples_per_image
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def add(self, smoky, out, gt, log_b=None, change_mask=None, name=""):
        out = out.clamp(0, 1)
        sy, oy, gy = rgb_to_y(smoky), rgb_to_y(out), rgb_to_y(gt)
        se = (oy - gy) ** 2
        smap = ssim_map(oy, gy)
        self.per_image.append({"name": name, "psnr": psnr_from_mse(se.mean().item()),
                               "ssim": smap.mean().item()})

        vis, textured = visibility_map(sy, gy)
        masks = {k: ((vis >= lo) & (vis < hi)).float() * textured for k, (lo, hi) in BINS.items()}
        if change_mask is not None and change_mask.sum() > 0:
            masks["changed"] = change_mask
            masks["unchanged"] = 1 - change_mask
        for k, m in masks.items():
            n = m.sum().item()
            if n > 0:
                self.bins[k]["sse"] += (se * m).sum().item()
                self.bins[k]["ssim"] += (smap * m).sum().item()
                self.bins[k]["n"] += int(n)

        if log_b is not None:
            err = (out - gt).abs().mean(1).flatten().cpu().numpy()
            unc = log_b.flatten().cpu().numpy()
            idx = self.rng.choice(len(err), size=min(self.k, len(err)), replace=False)
            self.err_s.append(err[idx])
            self.unc_s.append(unc[idx])

    def summary(self):
        res = {
            "n_images": len(self.per_image),
            "psnr": float(np.mean([r["psnr"] for r in self.per_image if math.isfinite(r["psnr"])])),
            "ssim": float(np.mean([r["ssim"] for r in self.per_image])),
            "bins": {},
        }
        for k, b in self.bins.items():
            if b["n"]:
                res["bins"][k] = {"psnr": psnr_from_mse(b["sse"] / b["n"]),
                                  "ssim": b["ssim"] / b["n"],
                                  "pixel_frac": b["n"]}
        total = sum(b["n"] for k, b in self.bins.items() if k in BINS)
        for k in BINS:
            if k in res["bins"]:
                res["bins"][k]["pixel_frac"] /= max(total, 1)
        for k in ("changed", "unchanged"):
            res["bins"].get(k, {}).pop("pixel_frac", None)
        if self.err_s:
            res["uncertainty"] = sparsification_ause(np.concatenate(self.err_s), np.concatenate(self.unc_s))
        return res


def format_table(results):
    """Markdown table: one row per method, columns = overall + bins."""
    cols = ["thick", "medium", "thin", "changed"]
    head = "| Method | PSNR | SSIM | " + " | ".join(f"{c} PSNR" for c in cols) + " | AUSE |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for name, r in results.items():
        b = r["bins"]
        cells = [f"{b[c]['psnr']:.2f}" if c in b else "-" for c in cols]
        ause = f"{r['uncertainty']['ause']:.4f}" if "uncertainty" in r else "-"
        lines.append(f"| {name} | {r['psnr']:.2f} | {r['ssim']:.3f} | " + " | ".join(cells) + f" | {ause} |")
    return "\n".join(lines)
