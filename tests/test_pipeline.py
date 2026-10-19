import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from refdesmoke.data import make_stale_reference, synthesize_smoke  # noqa: E402
from refdesmoke.dcp import dcp_desmoke  # noqa: E402
from refdesmoke.losses import Criterion  # noqa: E402
from refdesmoke.metrics import Evaluator, sparsification_ause, visibility_map  # noqa: E402
from refdesmoke.models import RefDesmoker, count_params  # noqa: E402


@pytest.mark.parametrize("mode", RefDesmoker.MODES)
def test_model_shapes_and_size(mode):
    m = RefDesmoker(mode)
    x = torch.rand(2, 3, 70, 90)  # not a multiple of 8: exercises padding
    out = m(x, torch.rand_like(x))
    assert out["out"].shape == x.shape
    assert out["log_b"].shape == (2, 1, 70, 90)
    assert (out["gate"] is not None) == (mode == "gated")
    assert 0.3e6 < count_params(m) < 5e6


def test_ref_required():
    with pytest.raises(ValueError):
        RefDesmoker("gated")(torch.rand(1, 3, 32, 32))


def test_synthesis_ranges():
    rng = np.random.default_rng(0)
    clean = rng.random((64, 80, 3)).astype(np.float32)
    smoky, t = synthesize_smoke(clean, rng)
    assert smoky.shape == clean.shape and 0 <= smoky.min() and smoky.max() <= 1
    assert 0 < t.min() and t.max() <= 1
    ref, mask = make_stale_reference(clean, rng, donor=clean[::-1].copy(), p_local=1.0)
    assert ref.shape == clean.shape and mask.sum() > 0


def test_visibility_bins_order():
    rng = np.random.default_rng(1)
    gt = torch.rand(1, 1, 64, 64)
    thin = 0.8 * gt + 0.1
    thick = 0.1 * gt + 0.8
    v_thin, _ = visibility_map(thin, gt)
    v_thick, _ = visibility_map(thick, gt)
    assert v_thin.mean() > 0.6 > 0.3 > v_thick.mean()


def test_ause_oracle_is_zero_and_random_is_worse():
    rng = np.random.default_rng(0)
    err = rng.random(5000)
    assert sparsification_ause(err, err)["ause"] == pytest.approx(0, abs=1e-9)
    assert sparsification_ause(err, rng.random(5000))["ause"] > 0.05


def test_evaluator_perfect_prediction():
    ev = Evaluator()
    gt = torch.rand(1, 3, 64, 64)
    smoky = 0.2 * gt + 0.7
    ev.add(smoky, gt + 0.01, gt, torch.zeros(1, 1, 64, 64))
    s = ev.summary()
    assert s["psnr"] > 35 and "thick" in s["bins"] and "uncertainty" in s


def test_dcp_improves_synthetic_haze():
    rng = np.random.default_rng(0)
    # Satisfy the prior: each pixel has one dark channel, and a bright region shows the airlight.
    clean = rng.random((64, 64, 3)).astype(np.float32) * 0.8
    clean[np.arange(64)[:, None], np.arange(64)[None, :], rng.integers(0, 3, (64, 64))] = 0
    clean[:20, :20] = 0.9
    smoky = clean * 0.6 + 0.4 * 0.9
    out, t = dcp_desmoke(smoky)
    assert np.abs(out - clean).mean() < np.abs(smoky - clean).mean()


def test_gate_supervision_closes_gate_on_changes():
    torch.manual_seed(0)
    m = RefDesmoker("gated", widths=(16, 32), depth=1, conf_from_gate=True)
    crit = Criterion(w_gate=1.0)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    gt = torch.rand(4, 3, 32, 32)
    mask = torch.zeros(4, 1, 32, 32)
    mask[..., :16, :] = 1
    ref = gt.clone()
    ref[..., :16, :] = torch.rand(4, 3, 16, 32)
    batch = {"gt": gt, "smoky": 0.4 * gt + 0.5, "ref": ref, "change_mask": mask}
    for _ in range(80):
        loss, terms = crit(m(batch["smoky"], batch["ref"]), batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    g = m(batch["smoky"], batch["ref"])["gate"]
    assert g[..., :16, :].mean() < g[..., 16:, :].mean() - 0.15


def test_training_reduces_loss():
    torch.manual_seed(0)
    m = RefDesmoker("gated", widths=(16, 32), depth=1)
    crit = Criterion()
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    gt = torch.rand(4, 3, 32, 32)
    batch = {"gt": gt, "smoky": 0.4 * gt + 0.5, "ref": gt.clone()}
    losses = []
    for _ in range(60):
        loss, _ = crit(m(batch["smoky"], batch["ref"]), batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
