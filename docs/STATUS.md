# Project status and handoff (2026-10-21)

## Where things stand
- **Idea:** reference-guided, uncertainty-aware desmoking for fixed CCTV cameras
  (see [ACTION_PLAN.md](../ACTION_PLAN.md)).
- **Phase 0 pilot on SmokeBench: GO.** v2 (2.21M params) reaches 26.91 dB vs 21.87 dB for a
  same-size single-image model (+6.1 dB in thick smoke), with a gate AUROC of 0.985 and 2× better
  confidence calibration. Full tables and caveats are in [experiments.md](experiments.md).
- **The main caveat:** the stale references are synthesized from the GT, so the numbers are an
  upper bound until real references are captured.
- **Method to carry forward:** v2 = `configs/smokebench_gated_v2.yaml`
  (gated fusion + global change head + balanced gate loss + confidence from gate).
  Reference dropout (v3) did not help on real data, so it is dropped.

## Not yet done (in order)
1. **Real stale-reference test set:** record it per [capture_protocol.md](capture_protocol.md).
   Needs a smoke machine. This is the result the paper stands on.
2. **Harder synthetic changes** in `make_stale_reference` (refdesmoke/data.py): shadows, soft
   edges, scale/perspective, people. Right now the gate may only be detecting pasted rectangles.
3. **Strong single-image baseline** trained to convergence (e.g. a ~5M-param `single` model,
   40+ epochs). Also cite the published SmokeBench numbers (MB-TaylorFormer 26.63 dB).
4. **v2 at full length:** 3 seeds × 40+ epochs, reporting mean ± std.
5. **Speed up data loading** before 3 and 4. Training on a laptop RTX 3050 is I/O-bound
   (~7 min/epoch for v2, ~4.5 min for single).
6. Literature search to confirm novelty ([lit_search_log.md](lit_search_log.md)); then write the
   paper; the demo app is already working (`python app/app.py`).

Estimated GPU time on the laptop: ~20 h; much less on a stronger GPU.

## Setting up on a new device
```bash
git clone https://github.com/tzar-beep/DesmokeNet.git && cd DesmokeNet
python -m venv .venv && .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests                      # 11 tests should pass
```
Data (not in git): copy `SmokeBench.zip` (1.5 GB, from github.com/ncfjd/SmokeBench), then:
```bash
unzip SmokeBench.zip -d /data/smokebench/raw
python scripts/prepare_pairs.py --src /data/smokebench/raw --out /data/smokebench/prepared
```
Then change the `manifest:` path in `configs/smokebench_*.yaml`, and raise `workers` and
`batch_size` if the machine allows it.

Not in git, so copy them by hand if you want them: `runs/` (pilot checkpoints, 270 MB) and
`results/` (pilot numbers; the tables are also in experiments.md).

## Known issues
- Long jobs must run detached (nohup/tmux or Start-Process): a job started from a Claude chat dies
  when the session ends, which is what killed v3 at epoch 11.
- On Windows, `workers: 4` crashed the DataLoader intermittently, so it is set to 2.
- **The code has not been pushed to GitHub yet:** push it from a terminal (`git remote add
  origin ...; git push -u origin main`) before switching devices.
