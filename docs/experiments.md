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

## 2026-10-19 — Reference dropout (v3)

v2 (w_gate 0.05) + `ref_dropout: 0.2`: during training, 20% of samples get a reference from
another scene with the change mask set everywhere, so the smoky-only path must work alone.

| Variant | PSNR | Thick | Changed | Changed∩thick | Unchanged | AUSE ×100 | Spearman | Gate AUROC |
|---|---|---|---|---|---|---|---|---|
| single | 22.63 ± 0.15 | 15.99 | 20.14 | **22.19** | 20.84 | 2.01 | **0.42** | – |
| gated | 25.43 ± 0.03 | **21.79** | 19.45 | 17.91 | 26.25 | 2.03 | 0.15 | 0.54 |
| v2 | **25.57 ± 0.16** | 21.65 | 19.82 | 18.37 | **26.41** | **1.80** | 0.24 | 0.64 / 0.57 |
| v3 (v2 + ref dropout) | 25.01 ± 0.13 | 20.96 | **21.38 ± 0.49** | 19.87 ± 0.66 | 25.45 | 1.88 | 0.24 | 0.86 / 0.58 |

**Findings**
1. With reference dropout, a reference model beats the single-image model in changed regions
   for the first time (+1.2 dB), and changed∩thick improves by 1.5 dB over v2.
2. The cost is about 0.6 dB overall and 0.7 dB in thick smoke: the model trusts the
   reference a little less everywhere.
3. Changed∩thick is still about 2.3 dB below the single-image model. Caveat: this bin is tiny
   (~3% changed × ~11% thick pixels across 9 test scenes). The single model scores higher there
   than on thick smoke overall, which suggests the bin is noisy or biased by the synthetic
   rectangles. Don't tune against it further.
4. Gate quality varies a lot between seeds (AUROC 0.86 vs 0.58), as it did for v2 at
   w_gate 0.2. The change head is not converging reliably in 630 training steps.

**Decision:** stop tuning on procedural data. The test set (9 scenes) is too small and too
artificial to separate these variants reliably. Carry v2 and v3 forward as the two candidates
for the SmokeBench pilot, with more seeds and a larger test set.

Reproduce: `configs/synthetic_{single,gated,gated_sup,gated_v2,gated_v3}.yaml` with `--seed 0/1`,
then `scripts/evaluate.py` and `scripts/gate_diagnostics.py`.
Note: `gated_sup` runs above used the earlier *unbalanced* gate loss; the loss is now balanced.
