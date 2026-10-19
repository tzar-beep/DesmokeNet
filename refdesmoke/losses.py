import torch
import torch.nn as nn


def laplace_nll(pred, target, log_b):
    """Negative log-likelihood of a per-pixel Laplace with scale b = exp(log_b)."""
    err = (pred - target).abs().mean(1, keepdim=True)
    return (err * torch.exp(-log_b) + log_b).mean()


class Criterion(nn.Module):
    """L1 trains the image; NLL trains the confidence head on a detached prediction.

    Detaching keeps the uncertainty term from down-weighting hard (thick-smoke) pixels in the
    image loss, which is exactly where we want the image to keep trying.
    """

    def __init__(self, w_l1=1.0, w_nll=0.1, w_lpips=0.0):
        super().__init__()
        self.w_l1, self.w_nll, self.w_lpips = w_l1, w_nll, w_lpips
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
        total = self.w_l1 * terms["l1"] + self.w_nll * terms["nll"]
        if self.lpips is not None:
            total = total + self.w_lpips * terms["lpips"]
        terms["total"] = total
        return total, {k: float(v.detach()) for k, v in terms.items()}
