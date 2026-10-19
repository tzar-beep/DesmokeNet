import torch
import torch.nn as nn
import torch.nn.functional as F


def laplace_nll(pred, target, log_b):
    """Negative log-likelihood of a per-pixel Laplace with scale b = exp(log_b)."""
    err = (pred - target).abs().mean(1, keepdim=True)
    return (err * torch.exp(-log_b) + log_b).mean()


def gate_loss(gates, change_mask):
    """Class-balanced BCE pushing every gate to 1 where the reference is valid, 0 where it changed.

    Changed pixels are a few percent of the image, so plain BCE is minimised by opening the
    gate everywhere; balancing gives both classes equal total weight.
    Global photometric changes are not in the mask: the reference still carries the right
    structure there, so the gate should stay open.
    """
    total = 0.0
    for g in gates:
        changed = F.interpolate(change_mask, size=g.shape[-2:], mode="area")
        frac = changed.mean().clamp(1e-3, 1 - 1e-3)
        weight = changed * (0.5 / frac) + (1 - changed) * (0.5 / (1 - frac))
        total = total + F.binary_cross_entropy(g.clamp(1e-4, 1 - 1e-4), 1 - changed, weight=weight)
    return total / len(gates)


class Criterion(nn.Module):
    """L1 trains the image; NLL trains the confidence head on a detached prediction.

    Detaching keeps the uncertainty term from down-weighting hard (thick-smoke) pixels in the
    image loss, which is exactly where we want the image to keep trying.
    """

    def __init__(self, w_l1=1.0, w_nll=0.1, w_lpips=0.0, w_gate=0.0):
        super().__init__()
        self.w_l1, self.w_nll, self.w_lpips, self.w_gate = w_l1, w_nll, w_lpips, w_gate
        self.lpips = None
        if w_lpips > 0:
            import lpips
            self.lpips = lpips.LPIPS(net="vgg", verbose=False).eval()
            for p in self.lpips.parameters():
                p.requires_grad_(False)

    def forward(self, pred, batch):
        gt = batch["gt"]
        out = pred["out"]
        terms = {"l1": (out - gt).abs().mean()}
        terms["nll"] = laplace_nll(out.detach(), gt, pred["log_b"])
        if self.lpips is not None:
            terms["lpips"] = self.lpips(out * 2 - 1, gt * 2 - 1).mean()
        if self.w_gate > 0 and pred.get("gates"):
            terms["gate"] = gate_loss(pred["gates"], batch["change_mask"])
        total = self.w_l1 * terms["l1"] + self.w_nll * terms["nll"]
        if self.lpips is not None:
            total = total + self.w_lpips * terms["lpips"]
        if "gate" in terms:
            total = total + self.w_gate * terms["gate"]
        terms["total"] = total
        return total, {k: float(v.detach()) for k, v in terms.items()}
