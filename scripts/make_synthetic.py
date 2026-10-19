"""Build a synthetic paired dataset + manifest.

Without --clean-dir it draws procedural indoor-like scenes; that is only for testing the
pipeline end to end, not for results. With --clean-dir it adds Beer-Lambert smoke to your own
clean photos, which is useful for pre-training.

The test split gets *fixed* stale references and change masks written to disk, so it
exercises the same code path as a real captured dataset.

    python scripts/make_synthetic.py --out data/synthetic --scenes 60 --frames 4
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refdesmoke.data import load_rgb, make_stale_reference, save_rgb, smooth_noise, synthesize_smoke  # noqa: E402


def procedural_scene(rng, h, w):
    """Walls, floor, furniture-like boxes, a door and some texture."""
    img = np.zeros((h, w, 3), np.float32)
    wall = rng.uniform(0.3, 0.8, 3)
    floor = rng.uniform(0.15, 0.5, 3)
    horizon = int(h * rng.uniform(0.55, 0.75))
    img[:horizon] = wall
    img[horizon:] = floor
    img *= (0.85 + 0.3 * smooth_noise(h, w, rng, octaves=5, base=6))[..., None]
    for _ in range(rng.integers(4, 10)):
        bh, bw = int(h * rng.uniform(0.08, 0.35)), int(w * rng.uniform(0.05, 0.25))
        y = rng.integers(0, h - bh)
        x = rng.integers(0, w - bw)
        col = tuple(float(c) for c in rng.uniform(0, 1, 3))
        if rng.random() < 0.3:
            cv2.ellipse(img, (x + bw // 2, y + bh // 2), (bw // 2, bh // 2), 0, 0, 360, col, -1)
        else:
            cv2.rectangle(img, (x, y), (x + bw, y + bh), col, -1)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0.05, 0.05, 0.05), 2)
    for _ in range(rng.integers(5, 15)):  # edges / cables / shelves
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, tuple(float(c) for c in rng.uniform(0, 1, 3)), int(rng.integers(1, 4)))
    dx = int(rng.integers(0, w - w // 6))
    cv2.rectangle(img, (dx, int(h * 0.2)), (dx + w // 8, horizon), (0.35, 0.2, 0.1), -1)
    return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--clean-dir", help="folder of clean images (optional)")
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--frames", type=int, default=4, help="smoky frames per scene")
    ap.add_argument("--size", type=int, nargs=2, default=[256, 320], metavar=("H", "W"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    h, w = args.size
    if args.clean_dir:
        files = sorted(p for p in Path(args.clean_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        cleans = [cv2.resize(load_rgb(p), (w, h), interpolation=cv2.INTER_AREA) for p in files[:args.scenes]]
    else:
        cleans = [procedural_scene(rng, h, w) for _ in range(args.scenes)]

    n = len(cleans)
    splits = ["train"] * int(0.7 * n) + ["val"] * int(0.15 * n)
    splits += ["test"] * (n - len(splits))

    rows = []
    for s, (clean, split) in enumerate(zip(cleans, splits)):
        scene = f"scene{s:04d}"
        gt_rel = f"{scene}/gt.png"
        save_rgb(out / gt_rel, clean)
        ref_rel = mask_rel = ""
        if split == "test":
            donor = cleans[(s + 1) % n]
            ref, mask = make_stale_reference(clean, rng, donor)
            ref_rel, mask_rel = f"{scene}/ref.png", f"{scene}/change_mask.png"
            save_rgb(out / ref_rel, ref)
            cv2.imwrite(str(out / mask_rel), (mask * 255).astype(np.uint8))
        for f in range(args.frames):
            smoky, _ = synthesize_smoke(clean, rng)
            smoky_rel = f"{scene}/smoky_{f:02d}.png"
            save_rgb(out / smoky_rel, smoky)
            rows.append({"smoky": smoky_rel, "gt": gt_rel, "ref": ref_rel,
                         "change_mask": mask_rel, "scene": scene, "split": split})

    with open(out / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    counts = {sp: sum(r["split"] == sp for r in rows) for sp in ("train", "val", "test")}
    print(f"wrote {len(rows)} pairs to {out} {counts}")


if __name__ == "__main__":
    main()
