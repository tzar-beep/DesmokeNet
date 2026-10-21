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

## 2026-10-21 — Phase 0 pilot on SmokeBench (real smoke)

Setup: SmokeBench v2 (9,875 train / 100 test pairs, 320×240). Validation is 1,000 train pairs
held out as 4 contiguous blocks of 250 frames (file names carry no scene ids). 128 px crops,
batch 16, seed 0, LPIPS weight 0.1. Single-image and v2 trained 12 epochs. v3's run died during
epoch 11, so its **epoch-8** checkpoint is reported. The stale reference is **synthesized from the
GT** (`make_stale_reference`), with a fixed seed per test image. Test metrics are on the Y channel.

| Method | PSNR | SSIM | Thick | Medium | Thin | Changed | Changed∩thick | AUSE ×100 |
|---|---|---|---|---|---|---|---|---|
| identity (smoky input) | 20.39 | 0.811 | 13.07 | 17.15 | 25.63 | 19.08 | 12.67 | – |
| DCP (classical) | 13.90 | 0.725 | 12.90 | 14.44 | 14.45 | 13.84 | 12.85 | 2.90 |
| paste reference | 21.27 | 0.843 | 19.60 | 19.90 | 20.12 | 10.78 | 10.86 | – |
| single-image (0.94M) | 21.87 | 0.821 | 14.97 | 19.84 | 26.86 | **20.82** | 14.31 | 1.72 |
| **v2, reference-guided (2.21M)** | **26.91** | **0.899** | **21.07** | **24.68** | **28.05** | 20.25 | **14.34** | **0.77** |
| v3 = v2 + ref dropout (epoch 8) | 26.36 | 0.892 | 21.05 | 24.37 | 26.49 | 18.03 | 13.62 | 0.88 |

Gate as a change detector (AUROC): v2 **0.985** (gate 0.12 in changed regions vs 0.94 elsewhere),
v3 0.971.

**Findings**
1. **H1 holds on real smoke:** the reference gives +5.0 dB overall and **+6.1 dB in thick smoke**
   over the same-size single-image model (21.07 vs 14.97).
2. **The model does not just copy the reference:** v2 beats "paste reference" by 5.6 dB, and by
   9.5 dB in changed regions. The gate finds stale regions almost perfectly (AUROC 0.985).
3. **H3 looks promising:** the confidence map is far better calibrated with a reference (AUSE 0.77
   vs 1.72 ×10⁻², more than 2× lower).
4. **H2 is a tie, not a win:** in changed regions v2 is 0.6 dB below the single model, and in
   changed∩thick they are equal. Where the reference is wrong *and* smoke hides the scene, nothing
   can recover it; the honest output there is low confidence.
5. Reference dropout (v3) did **not** help on real data (changed 18.03 vs 20.25), unlike on
   synthetic data. It was also trained 4 fewer epochs. Drop it, or retest it at equal epochs.
6. Classical DCP is worse than the smoky input (13.9 vs 20.4 dB): the haze prior breaks on dense
   indoor smoke.

**Caveats (must be resolved before any paper claim)**
- **The reference is synthesized from the GT.** The global photometric shift and the pasted
  rectangles are easier than a real stale reference: real lighting changes, moved objects with
  shadows, and real camera drift. This makes the numbers an optimistic upper bound. The real
  stale-reference capture (docs/capture_protocol.md) is the test that matters.
- **The single-image baseline is weak:** 21.87 dB vs 22.8–26.6 dB for the published SmokeBench
  methods, after only 12 epochs at 0.94M parameters. The paper must compare against strong
  single-image models (e.g. MB-TaylorFormer, 26.63 dB published) trained to convergence.
  Note that v2 (26.91) is not directly comparable to the published numbers, because it uses extra
  input.
- One seed, 100 test images.

**Decision: GO**, with the method narrowed to v2. Next:
(1) record a small real stale-reference test set;
(2) a strong single-image baseline trained to convergence, plus published numbers;
(3) 3 seeds and full-length training for v2;
(4) harder synthetic changes (shadows, soft edges, scale/perspective) so the gate isn't just
    detecting pasted rectangles.

Reproduce: `bash scripts/run_pilot.sh` (results in `results/pilot`).
Synthetic configs: `configs/synthetic_{single,gated,gated_sup,gated_v2,gated_v3}.yaml` with `--seed 0/1`,
then `scripts/evaluate.py` and `scripts/gate_diagnostics.py`.
Note: `gated_sup` runs above used the earlier *unbalanced* gate loss; the loss is now balanced.
