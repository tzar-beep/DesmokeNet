"""Datasets and synthesis for reference-guided desmoking.

A manifest CSV drives every dataset. Columns:
    smoky, gt            required, paths relative to the manifest's folder
    ref                  optional clean reference captured at an earlier time
    change_mask          optional binary mask of regions where ref differs from gt
    scene, split         scene id and train/val/test

When `ref` is missing (e.g. SmokeBench, where the only clean frame *is* the GT), a stale
reference is synthesized from the GT so the model cannot learn to simply copy it.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def save_rgb(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))


def to_tensor(img):
    if img.ndim == 2:
        img = img[..., None]
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def smooth_noise(h, w, rng, octaves=4, base=3):
    """Multi-octave value noise in [0, 1]; a cheap stand-in for Perlin noise."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh, gw = base * 2 ** o + 1, base * 2 ** o + 1
        grid = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def synthesize_smoke(clean, rng, beta_range=(0.3, 4.0)):
    """Beer-Lambert smoke: I = J*t + A*(1-t), t = exp(-beta * density).

    Density is non-uniform and biased towards the top of the frame, since indoor smoke
    accumulates under the ceiling. Returns the smoky image and the transmission map.
    """
    h, w, _ = clean.shape
    density = smooth_noise(h, w, rng)
    ceiling = np.linspace(1.0, rng.uniform(0.2, 0.8), h, dtype=np.float32)[:, None]
    density = density * (0.4 + 0.6 * ceiling)
    beta = rng.uniform(*beta_range)
    t = np.exp(-beta * density)[..., None]
    grey = rng.uniform(0.55, 0.95)
    airlight = np.clip(grey + rng.normal(0, 0.03, 3), 0, 1).astype(np.float32)
    smoky = clean * t + airlight * (1 - t)
    smoky += rng.normal(0, 0.01, smoky.shape).astype(np.float32)
    return np.clip(smoky, 0, 1).astype(np.float32), t[..., 0]


def make_stale_reference(gt, rng, donor=None, p_local=0.8, max_changes=3):
    """Simulate a clean reference recorded earlier than the GT.

    Global changes: per-channel gain (colour temperature), gamma, local illumination field.
    Local changes: objects moved within the scene or replaced with donor content.
    Returns the reference and a binary mask of locally changed regions.
    """
    h, w, _ = gt.shape
    ref = gt.copy()
    mask = np.zeros((h, w), np.float32)

    if rng.random() < p_local:
        for _ in range(rng.integers(1, max_changes + 1)):
            rh = int(h * rng.uniform(0.1, 0.3))
            rw = int(w * rng.uniform(0.1, 0.3))
            y, x = rng.integers(0, h - rh + 1), rng.integers(0, w - rw + 1)
            if donor is not None and rng.random() < 0.5:
                d = cv2.resize(donor, (w, h), interpolation=cv2.INTER_AREA)
                sy, sx = rng.integers(0, h - rh + 1), rng.integers(0, w - rw + 1)
                patch = d[sy:sy + rh, sx:sx + rw]
            else:  # object moved: content from elsewhere in the same scene
                sy, sx = rng.integers(0, h - rh + 1), rng.integers(0, w - rw + 1)
                patch = gt[sy:sy + rh, sx:sx + rw]
            ref[y:y + rh, x:x + rw] = patch
            mask[y:y + rh, x:x + rw] = 1.0

    gain = np.clip(1 + rng.normal(0, 0.08, 3), 0.7, 1.3).astype(np.float32)
    gamma = rng.uniform(0.75, 1.35)
    ref = np.clip(ref * gain, 0, 1) ** gamma
    if rng.random() < 0.5:  # a lamp on or off somewhere in the room
        field = 0.75 + 0.5 * smooth_noise(h, w, rng, octaves=2, base=2)
        ref = ref * field[..., None]
    if rng.random() < 0.3:  # slight camera shake between recordings
        dy, dx = rng.integers(-2, 3, size=2)
        ref = np.roll(ref, (dy, dx), axis=(0, 1))
    return np.clip(ref, 0, 1).astype(np.float32), mask


class PairDataset(Dataset):
    def __init__(self, manifest, split, crop=None, train=False, stale_ref=True, seed=0):
        self.root = Path(manifest).parent
        with open(manifest, newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows with split={split!r} in {manifest}")
        self.crop, self.train, self.stale_ref, self.seed = crop, train, stale_ref, seed

    def __len__(self):
        return len(self.rows)

    def _path(self, p):
        return self.root / p

    def __getitem__(self, i):
        # Training draws fresh randomness; evaluation is deterministic per item.
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed * 100003 + i)
        row = self.rows[i]
        smoky = load_rgb(self._path(row["smoky"]))
        gt = load_rgb(self._path(row["gt"]))

        if row.get("ref"):
            ref = load_rgb(self._path(row["ref"]))
            mask = np.zeros(gt.shape[:2], np.float32)
            if row.get("change_mask"):
                m = cv2.imread(str(self._path(row["change_mask"])), cv2.IMREAD_GRAYSCALE)
                mask = (m > 127).astype(np.float32)
        elif self.stale_ref:
            donor_row = self.rows[rng.integers(len(self.rows))]
            donor = load_rgb(self._path(donor_row["gt"]))
            ref, mask = make_stale_reference(gt, rng, donor)
        else:
            ref, mask = gt.copy(), np.zeros(gt.shape[:2], np.float32)

        arrs = [smoky, gt, ref, mask[..., None]]
        if self.crop:
            h, w = gt.shape[:2]
            c = min(self.crop, h, w)
            if self.train:
                y, x = rng.integers(0, h - c + 1), rng.integers(0, w - c + 1)
            else:
                y, x = (h - c) // 2, (w - c) // 2
            arrs = [a[y:y + c, x:x + c] for a in arrs]
        if self.train:
            if rng.random() < 0.5:
                arrs = [a[:, ::-1] for a in arrs]
            if rng.random() < 0.5:
                arrs = [a[::-1] for a in arrs]

        smoky, gt, ref, mask = (to_tensor(a) for a in arrs)
        return {"smoky": smoky, "gt": gt, "ref": ref, "change_mask": mask,
                "name": row["smoky"], "scene": row.get("scene", "")}
