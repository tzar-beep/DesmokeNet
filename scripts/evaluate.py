"""Evaluate methods with the density-binned protocol.

    python scripts/evaluate.py --manifest data/synthetic/manifest.csv --split test \
        --methods identity dcp paste_ref runs/single/best.pt runs/gated/best.pt \
        --out results/synthetic
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refdesmoke.data import PairDataset  # noqa: E402
from refdesmoke.infer import load_method  # noqa: E402
from refdesmoke.metrics import Evaluator, format_table  # noqa: E402


def evaluate(method, loader, device, max_items=None):
    ev = Evaluator()
    times = []
    for i, b in enumerate(loader):
        if max_items and i >= max_items:
            break
        smoky, gt, ref = (b[k].to(device) for k in ("smoky", "gt", "ref"))
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = method(smoky, ref)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        ev.add(smoky, pred["out"], gt, pred["log_b"], b["change_mask"].to(device), b["name"][0])
    res = ev.summary()
    if len(times) > 2:  # drop warm-up
        t = sorted(times[2:])
        res["latency_ms"] = {"median": 1000 * t[len(t) // 2], "p95": 1000 * t[int(0.95 * (len(t) - 1))]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods", nargs="+", default=["identity", "dcp", "paste_ref"])
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--max-items", type=int)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(PairDataset(args.manifest, args.split, train=False), batch_size=1)
    results = {}
    for spec in args.methods:
        m = load_method(spec, device)
        name = m.name if hasattr(m, "name") else spec
        print(f"evaluating {name} ...")
        results[name] = evaluate(m, loader, device, args.max_items)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    table = format_table(results)
    (out / "table.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
