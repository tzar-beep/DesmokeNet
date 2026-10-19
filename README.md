# RefDesmoke

Reference-guided, uncertainty-aware smoke removal for fixed surveillance cameras.
See [ACTION_PLAN.md](ACTION_PLAN.md) for the research plan.

## Setup

```bash
python -m venv ~/.venvs/desmoke
~/.venvs/desmoke/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
~/.venvs/desmoke/Scripts/python -m pip install -r requirements.txt
```

The environment and datasets live outside OneDrive so they don't sync.

## Quick start (synthetic data, checks the whole pipeline)

```bash
python scripts/make_synthetic.py --out data/synthetic --scenes 60 --frames 4
python scripts/train.py --config configs/synthetic_single.yaml
python scripts/train.py --config configs/synthetic_gated.yaml
python scripts/evaluate.py --manifest data/synthetic/manifest.csv --methods identity dcp paste_ref runs/synthetic_single/best.pt runs/synthetic_gated/best.pt --out results/synthetic
python app/app.py
```

Synthetic numbers only show that the code works. They are not paper results.

## Real data (SmokeBench)

1. Download `SmokeBench.zip` (1.5 GB, linked from [ncfjd/SmokeBench](https://github.com/ncfjd/SmokeBench))
   to `C:/data/smokebench/`, outside OneDrive.
2. Run the whole Phase 0 pilot. It unzips, builds a 640 px manifest with
   `scripts/prepare_pairs.py`, trains single / v2 / v3 on 2 seeds, then evaluates:
   ```bash
   bash scripts/run_pilot.sh
   ```
   Check the folder detection first with
   `python scripts/prepare_pairs.py --src C:/data/smokebench/raw --out x --dry-run`.
   `ref` is left empty, so a stale reference is synthesized from the GT.

## Layout

| Path | What |
|------|------|
| `refdesmoke/data.py` | manifest dataset, Beer-Lambert smoke synthesis, stale-reference synthesis |
| `refdesmoke/models.py` | `RefDesmoker` in `single` / `concat` / `gated` modes, confidence head |
| `refdesmoke/metrics.py` | density-binned PSNR/SSIM, changed-region metrics, AUSE calibration |
| `refdesmoke/dcp.py` | classical Dark Channel Prior baseline |
| `refdesmoke/infer.py` | one interface over models and baselines (used by eval and app) |
| `scripts/` | `train.py`, `evaluate.py`, `make_synthetic.py` |
| `app/app.py` | Gradio demo, image and video modes |
| `tests/` | `pytest tests` |
