"""Does the gate tell changed from unchanged regions? Reports mean gate per scale and the
AUROC of (1 - full-res gate) as a change detector, for gated checkpoints.

    python scripts/gate_diagnostics.py --manifest data/synthetic/manifest.csv runs/*/best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refdesmoke.data import PairDataset  # noqa: E402
from refdesmoke.infer import load_method  # noqa: E402


def auroc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("checkpoints", nargs="+")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PairDataset(args.manifest, args.split)
    rng = np.random.default_rng(0)

    for ck in args.checkpoints:
        m = load_method(ck, device)
        if getattr(m, "model", None) is None or m.model.mode != "gated":
            print(f"{ck}: not a gated model, skipped")
            continue
        sums, scores, labels = None, [], []
        for i in range(len(ds)):
            b = ds[i]
            o = m.model(b["smoky"][None].to(device), b["ref"][None].to(device))
            mask = b["change_mask"][None].to(device)
            if sums is None:
                sums = np.zeros((len(o["gates"]), 4))
            for s, g in enumerate(o["gates"]):
                mk = F.interpolate(mask, size=g.shape[-2:], mode="area")
                sums[s] += [(g * mk).sum().item(), mk.sum().item(),
                            (g * (1 - mk)).sum().item(), (1 - mk).sum().item()]
            g0, mk0 = o["gate"].flatten().cpu().numpy(), mask.flatten().cpu().numpy()
            idx = rng.choice(len(g0), size=min(5000, len(g0)), replace=False)
            scores.append(1 - g0[idx])
            labels.append(mk0[idx] > 0.5)
        per_scale = "  ".join(f"s{s}: {a / max(b, 1):.2f}/{c / max(d, 1):.2f}" for s, (a, b, c, d) in enumerate(sums))
        print(f"{ck}\n  gate changed/unchanged  {per_scale}\n  change AUROC {auroc(np.concatenate(scores), np.concatenate(labels)):.3f}")


if __name__ == "__main__":
    main()
