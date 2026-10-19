# RefDesmoke — Action Plan

**Working title:** *Seeing Through Smoke with Memory: Reference-Guided, Uncertainty-Aware Desmoking for Fixed Surveillance Cameras*

## 1. The idea in one paragraph

SmokeBench (ACM MM 2025) shows that every current restoration method fails on thick smoke: the
content is gone, so models over-smooth or hallucinate. But a surveillance camera is fixed and
has usually recorded hours of clean footage of the same scene before the fire. We use a
**stale clean reference frame** as extra input. The model must learn *where the reference is
still valid* (static walls, doors, floor), *where it is not* (moved objects, people, lighting
changes), and must output a **per-pixel confidence map** so that downstream users never trust
hallucinated content in a safety-critical scene.

## 2. Research question and hypotheses

**RQ:** Does a stale clean reference let a lightweight model recover thick-smoke regions that
single-image methods cannot, and can the model tell when the reference no longer applies?

| # | Hypothesis | How we test it | Pass criterion |
|---|------------|----------------|----------------|
| H1 | Reference improves thick-smoke recovery | PSNR/SSIM/LPIPS split by smoke-density bin (thin / medium / thick) | Clear gain over best single-image baseline in the *thick* bin |
| H2 | Model does not blindly copy a stale reference | "Scene-change" test split (objects moved, lighting changed) | Beats naive "paste reference" and single-image model in changed regions |
| H3 | Confidence is calibrated | AUSE + Spearman(confidence, error) | Lower AUSE than error-agnostic baselines |
| H4 (opt.) | Desmoking helps detection without hallucinating | Person-detection AP with a fixed detector | AP up, false positives not up |

## 3. Contributions (what is new)

1. **Task:** reference-guided desmoking for fixed cameras, a setting not yet studied for smoke
   (to be confirmed by the structured literature search, Phase 0).
2. **Method:** a lightweight (1–5M params) dual-encoder network with **validity-gated fusion**
   and a **heteroscedastic confidence head**.
3. **Dataset/protocol:** a *stale-reference* test split (reference at T0, scene changed, clean
   GT at T1, smoke at T1+) and a **density-binned evaluation protocol** for thick smoke.
4. **Practical:** latency on an RTX 3050 laptop GPU and CPU, plus a working demo app.

## 4. Phases

### Phase 0: De-risk (weeks 1–2)
- [ ] Structured literature search (IEEE Xplore, Scholar, arXiv): "reference-based dehazing",
      "background-guided smoke removal", "exemplar restoration", "change-aware restoration",
      "video desmoking". Log queries + dates in `docs/lit_search_log.md`.
- [ ] Get SmokeBench data. Check whether any scenes repeat under different lighting
      (free stale references).
- [ ] Obtain full texts of DesmokeNet, StarNet, DEEP-CF; verify every number in the report.
- [ ] **Go/no-go pilot:** train `concat` RefUNet vs `single` UNet on SmokeBench with synthetic
      stale-reference augmentation. Compare in the *thick* bin.
      **Kill/pivot criterion:** if the reference gives < ~1 dB gain in the thick bin, rethink.

### Phase 1: Method (weeks 3–6)
- [ ] Gated dual-encoder model + confidence head (implemented in `refdesmoke/models.py`).
- [ ] Stale-reference synthesis: illumination shift, colour temperature, moved/added objects,
      small misalignment (implemented in `refdesmoke/data.py`).
- [ ] Baselines: DCP (classical), single-image UNet (same size), concat RefUNet, naive
      reference paste, MB-TaylorFormer / DehazeFormer numbers from SmokeBench.
- [ ] Ablations: no ref / concat / gated; with and without confidence; reference age/change type.

### Phase 2: Dataset (weeks 5–9, overlaps)
- [ ] Recording protocol (`docs/capture_protocol.md`), SmokeBench-style setup:
      fixed camera, smoke machine, sealed room.
      Per scene: **(a)** reference at T0 → **(b)** change scene (move 1–3 objects, change
      lights) → **(c)** clean GT at T1 → **(d)** smoke, sample frames.
- [ ] Target: 10–15 scenes, ~20 change configurations. Small and honest beats large and noisy.
- [ ] Hand-annotate change masks for the test split (enables H2 region metrics).

### Phase 3: Experiments (weeks 9–12)
- [ ] Main table: all methods × density bins × {SmokeBench, new split}.
- [ ] Calibration figure (sparsification curves), qualitative failure cases.
- [ ] Efficiency: params, FLOPs, median + p95 latency on RTX 3050 and CPU.
- [ ] 3 seeds, mean ± std, confidence intervals on main comparisons.

### Phase 4: Write-up (weeks 12–16)
- [ ] Paper draft. Candidate venues: ACM MM, WACV, ICIP; journals: IEEE TCSVT, Fire Technology,
      Pattern Recognition Letters. Check current deadlines.
- [ ] Release code, data and the demo app.

### Running throughout: Demo app
- Week 1: app works with the classical DCP method (no training needed).
- From Phase 1: app loads any trained checkpoint and shows restored frame + confidence map.
- Video mode: auto-selects a clean reference from early frames, exports side-by-side video.

## 5. Risks

| Risk | Mitigation |
|------|------------|
| Reference-based smoke removal already published | Phase 0 search before heavy investment; pivot to "uncertainty-aware thick-smoke desmoking" if so |
| SmokeBench reference = GT, so the model learns to copy | Stale-reference augmentation in training; scene-change test split for evaluation |
| Can't record new data (no smoke machine / room) | Use SmokeBench lighting variants + synthetic changes; state the limitation |
| 6 GB VRAM | 1–5M-param model, 128–256 px crops, AMP |
| Pilot fails | Kill criterion in Phase 0, decided within 2 weeks |

## 6. Repository map

```
refdesmoke/       library: data, models, losses, metrics, classical DCP
scripts/          train.py, evaluate.py, make_synthetic.py
app/app.py        Gradio demo (image + video)
configs/          experiment configs
tests/            end-to-end smoke tests on synthetic data
docs/             literature log, capture protocol
```
