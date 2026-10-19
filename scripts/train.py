"""Train a desmoking model from a YAML config.

    python scripts/train.py --config configs/synthetic_gated.yaml
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refdesmoke.data import PairDataset  # noqa: E402
from refdesmoke.losses import Criterion  # noqa: E402
from refdesmoke.models import build_model, count_params  # noqa: E402
from scripts.evaluate import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, help="override config")
    ap.add_argument("--seed", type=int, help="override config")
    ap.add_argument("--out-dir", help="override config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.out_dir:
        cfg["train"]["out_dir"] = args.out_dir
    d, t = cfg["data"], cfg["train"]

    torch.manual_seed(t.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(t["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    train_ds = PairDataset(d["manifest"], d.get("train_split", "train"), crop=d["crop"], train=True,
                           ref_dropout=d.get("ref_dropout", 0.0))
    val_ds = PairDataset(d["manifest"], d.get("val_split", "val"), train=False)
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, drop_last=True,
                          num_workers=d.get("workers", 2), persistent_workers=d.get("workers", 2) > 0)
    val_dl = DataLoader(val_ds, batch_size=1)

    model = build_model(cfg["model"]).to(device)
    print(f"model {cfg['model']['mode']}: {count_params(model) / 1e6:.2f}M params, device {device}")
    crit = Criterion(**cfg.get("loss", {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4))
    total_steps = t["epochs"] * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    amp = t.get("amp", True) and device == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)

    def method(smoky, ref):
        return model(smoky, ref if model.mode != "single" else None)

    best, log = -1.0, open(out_dir / "log.jsonl", "a")
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        bar = tqdm(train_dl, desc=f"epoch {epoch}", leave=False)
        for b in bar:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            with torch.autocast(device, enabled=amp):
                pred = method(b["smoky"], b["ref"])
            pred = {k: [g.float() for g in v] if isinstance(v, list) else (v.float() if v is not None else None)
                    for k, v in pred.items()}
            loss, terms = crit(pred, b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            bar.set_postfix(l1=f"{terms['l1']:.4f}", nll=f"{terms['nll']:.3f}")

        if epoch % t.get("val_every", 1) == 0 or epoch == t["epochs"]:
            model.eval()
            with torch.no_grad():
                res = evaluate(method, val_dl, device, t.get("val_max_items"))
            thick = res["bins"].get("thick", {}).get("psnr", float("nan"))
            print(f"epoch {epoch}: PSNR {res['psnr']:.2f}  SSIM {res['ssim']:.3f}  thick PSNR {thick:.2f}")
            log.write(json.dumps({"epoch": epoch, "train": terms, "val": res}) + "\n")
            log.flush()
            ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": res}
            torch.save(ckpt, out_dir / "last.pt")
            if res["psnr"] > best:
                best = res["psnr"]
                torch.save(ckpt, out_dir / "best.pt")
    print(f"done. best val PSNR {best:.2f}; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
