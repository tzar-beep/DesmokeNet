"""Turn a paired desmoking dataset (e.g. SmokeBench) into a manifest, optionally downscaled.

Finds the smoky and clean folders by common names, pairs images by file stem (or, when a
clean folder holds one image per scene, by scene folder), and writes manifest.csv. Splits come
from a train/test folder in the path if present; a val split is carved from train by scene.

    python scripts/prepare_pairs.py --src C:/data/smokebench/raw --out C:/data/smokebench/prepared --max-side 640

Run with --dry-run first to see which folders were detected.
"""
import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

SMOKY = re.compile(r"^(smoke|smoky|smoked|input|inputs|hazy|haze|degraded|lq|low)$", re.I)
CLEAN = re.compile(r"^(gt|clean|clear|target|targets|groundtruth|ground_truth|hq|high|label|labels)$", re.I)
SPLIT = re.compile(r"^(train|training|test|testing|val|valid|validation)$", re.I)
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def split_of(path):
    for part in path.parts:
        if SPLIT.match(part):
            p = part.lower()
            return "train" if p.startswith("train") else "val" if p.startswith("val") else "test"
    return "train"


def find_role_dirs(src):
    smoky, clean = [], []
    for d in sorted(p for p in src.rglob("*") if p.is_dir()):
        (smoky if SMOKY.match(d.name) else clean if CLEAN.match(d.name) else []).append(d)
    return smoky, clean


def images(d):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in EXTS)


def pair_dir(sd, cd):
    """Pair by identical stem; else by leading stem token; else one clean image per scene folder."""
    clean = images(cd)
    by_stem = {p.stem: p for p in clean}
    by_rel_parent = defaultdict(list)
    for p in clean:
        by_rel_parent[p.parent.relative_to(cd)].append(p)
    pairs = []
    for s in images(sd):
        g = by_stem.get(s.stem) or by_stem.get(re.split(r"[_\-]", s.stem)[0])
        if g is None:
            cands = by_rel_parent.get(s.parent.relative_to(sd), [])
            g = cands[0] if len(cands) == 1 else None
        if g is not None:
            pairs.append((s, g))
    return pairs


def scene_of(smoky_path, root):
    rel = smoky_path.relative_to(root)
    dirs = [p for p in rel.parts[:-1] if not SMOKY.match(p) and not SPLIT.match(p)]
    if dirs:
        return "/".join(dirs)
    m = re.match(r"([A-Za-z]*\d+)", smoky_path.stem)  # e.g. "scene12_0034" -> "scene12"
    return m.group(1) if m else smoky_path.stem


def save(src, dst, max_side):
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"unreadable image {src}")
    h, w = img.shape[:2]
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-side", type=int, default=0, help="downscale longest side (0 = keep)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of train scenes held out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=250, help="frames per pseudo-scene when names are bare numbers")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)

    smoky_dirs, clean_dirs = find_role_dirs(src)
    print(f"smoky folders: {[str(d.relative_to(src)) for d in smoky_dirs]}")
    print(f"clean folders: {[str(d.relative_to(src)) for d in clean_dirs]}")
    pairs = []
    for sd in smoky_dirs:  # match each smoky folder with the clean folder sharing its parent
        cd = next((c for c in clean_dirs if c.parent == sd.parent), None)
        if cd is None:
            print(f"  no clean folder next to {sd}, skipped")
            continue
        p = pair_dir(sd, cd)
        print(f"  {sd.relative_to(src)} <-> {cd.relative_to(src)}: {len(p)} pairs of {len(images(sd))} smoky images")
        pairs += p
    if not pairs:
        sys.exit("no pairs found: check the folder names and adjust SMOKY/CLEAN patterns")

    rows = [{"smoky_src": s, "gt_src": g, "scene": scene_of(s, src), "split": split_of(s.relative_to(src))}
            for s, g in pairs]
    if len({r["scene"] for r in rows}) == 1 and all(r["smoky_src"].stem.isdigit() for r in rows):
        # No scene labels (SmokeBench: 1.jpg, 2.jpg, ...). Frames are numbered in capture order, so
        # contiguous blocks stand in for scenes and keep near-duplicate frames out of validation.
        for r in rows:
            r["scene"] = f"{r['split']}_block{int(r['smoky_src'].stem) // args.block}"
        print(f"no scene labels found: using contiguous blocks of {args.block} frames as scenes")
    train_scenes = sorted({r["scene"] for r in rows if r["split"] == "train"})
    if not any(r["split"] == "val" for r in rows) and len(train_scenes) > 1:
        rng = random.Random(args.seed)
        val = set(rng.sample(train_scenes, max(1, round(args.val_frac * len(train_scenes)))))
        for r in rows:
            if r["split"] == "train" and r["scene"] in val:
                r["split"] = "val"
    counts = {sp: sum(r["split"] == sp for r in rows) for sp in ("train", "val", "test")}
    print(f"{len(rows)} pairs, {len({r['scene'] for r in rows})} scenes, splits {counts}")
    if args.dry_run:
        for r in rows[:5]:
            print("  example:", r["smoky_src"].relative_to(src), "->", r["gt_src"].relative_to(src), r["scene"], r["split"])
        return

    out.mkdir(parents=True, exist_ok=True)
    done_gt, manifest = {}, []
    for i, r in enumerate(rows):
        smoky_rel = Path("smoky") / r["smoky_src"].relative_to(src).with_suffix(".png")
        save(r["smoky_src"], out / smoky_rel, args.max_side)
        if r["gt_src"] not in done_gt:
            gt_rel = Path("gt") / r["gt_src"].relative_to(src).with_suffix(".png")
            save(r["gt_src"], out / gt_rel, args.max_side)
            done_gt[r["gt_src"]] = gt_rel
        manifest.append({"smoky": smoky_rel.as_posix(), "gt": done_gt[r["gt_src"]].as_posix(), "ref": "",
                         "change_mask": "", "scene": r["scene"], "split": r["split"]})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)}")
    with open(out / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest[0]))
        wr.writeheader()
        wr.writerows(manifest)
    print(f"wrote {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
