# Experiment log

Synthetic procedural data only: these runs debug the method; they are not paper results.
Setup: `data/synthetic` (60 scenes, 9 test scenes / 36 test images), 30 epochs, 128 px crops,
2 seeds, mean ± std. Changed regions are ~3% of test pixels, so `changed_thick` is noisy.

## 2026-10-19 — Does the model reject a stale reference?

| Variant | Params | PSNR | Thick | Changed | Changed∩thick | Unchanged | Spearman(conf, err) | Gate AUROC |
|---|---|---|---|---|---|---|---|---|
| single (no ref) | 0.94M | 22.63 ± 0.15 | 15.99 | **20.14** | **22.19** | 20.84 | **0.42** | – |
| gated | 1.80M | 25.43 ± 0.03 | **21.79** | 19.45 | 17.91 | 26.25 | 0.15 | 0.54 |
| + gate supervision (unbalanced BCE) | 1.80M | 25.45 ± 0.07 | 21.78 | 19.20 | 17.95 | 26.34 | 0.18 | 0.51 |
| v2: + global change head, balanced BCE, conf from gate, w_gate 0.05 | 2.21M | **25.57 ± 0.16** | 21.65 | 19.82 | 18.37 | **26.41** | 0.24 | 0.64 / 0.57 |
| v2, w_gate 0.2 | 2.21M | 25.08 ± 0.34 | 21.42 | 20.09 | 17.48 | 25.67 | 0.15 | 0.94 / 0.75 |

Gate AUROC: how well (1 − gate) detects changed pixels, given per seed (0.5 = chance).

**Findings**
1. The reference gives about +5.7 dB in thick smoke. This is the core claim, and so far it holds.
2. The first gated models ignored *where* the reference changed: the gate was spatially
   constant (AUROC ≈ 0.5). Per-scale gates see only a small neighbourhood, and plain BCE is
   minimised by opening the gate everywhere because changed pixels are rare.
3. A global change head plus a class-balanced gate loss makes the gate discriminative
   (AUROC up to 0.94) and recovers the changed regions (19.45 → 20.09 dB, matching the
   single-image model), at a small cost overall. The gate weight trades these off, and
   w_gate 0.2 is unstable across seeds.
4. **Still open:** in changed **and** thick regions, every reference model is 4+ dB below the
   single-image model. When the gate closes, the smoky-only path is weaker than a dedicated
   single-image network, because joint training lets it lean on the reference.
   Next: *reference dropout*. During training, randomly replace the reference with an
   unrelated frame (gate target 0) so the smoky path must work alone.
5. Confidence is still weakly correlated with error in reference models (ρ ≈ 0.2 vs 0.42 for
   single). It needs work before H3 can be claimed.

Reproduce: `configs/synthetic_{single,gated,gated_sup,gated_v2}.yaml` with `--seed 0/1`,
then `scripts/evaluate.py` and `scripts/gate_diagnostics.py`.
Note: `gated_sup` runs above used the earlier *unbalanced* gate loss; the loss is now balanced.
