"""Why the Laya temperature fit stops at its clamp, and the tool that measures it.

The vendor fit (``laya.calibrate``) is pure CPU code. The first block feeds it synthetic
logit gaps that reproduce each plausible cause. It loads no weights. The second block tests
the diagnostic functions and the script, again on synthetic gaps.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from modeling.text import laya_temperature as lt
from modeling.text.laya_calibration import (
    MAX_OOF_ECE_RISE,
    REJECTED,
    LayaCalibratorError,
    fit_calibrator,
)

RNG_SEED = 7


def make_gaps(n, *, scale=1.0, shift=0.0, flip=False, informative=True, seed=RNG_SEED):
    """Gaps for a model whose true log odds are ``l``. ``scale`` makes it overconfident."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.0, 2.0, n)
    y = (rng.random(n) < expit(latent)).astype(int)
    gap = scale * latent + shift
    if not informative:
        gap = scale * rng.normal(0.0, 2.0, n)
    if flip:
        y = 1 - y
    return gap, y


def vendor_fit(gap, y):
    calibrate = pytest.importorskip("laya.calibrate")
    records = [
        (2, np.array([0.0, g], dtype=np.float32), np.array([1.0 - t, float(t)]), 2)
        for g, t in zip(gap, y, strict=True)
    ]
    return calibrate.fit_temperature_map(records)


# --- cause 1: the model is very overconfident -------------------------------------------
def test_overconfident_logits_push_the_vendor_fit_to_the_clamp_and_the_wide_fit_inside():
    gap, y = make_gaps(3000, scale=30.0)
    result = vendor_fit(gap, y)
    assert result["temperature"][2] == 5.0
    assert result["temperature_clamped"]["noul"]["bound"] == "max"
    fit = lt.fit_temperature(gap, y)
    assert not fit.at_bound
    assert 15.0 < fit.value < 60.0


# --- cause 2: the target is the wrong way round -----------------------------------------
def test_a_flipped_target_has_no_interior_minimum():
    gap, y = make_gaps(3000, scale=1.0, flip=True)
    assert vendor_fit(gap, y)["temperature"][2] == 5.0
    fit = lt.fit_temperature(gap, y)
    assert fit.at_bound and fit.bound == "upper"
    report = lt.diagnose(gap, y, None, checkpoint_temperature=5.0)
    assert any("no interior minimum" in note for note in report["findings"])


# --- cause 3: a base rate shift that a temperature cannot fix ---------------------------
def test_a_bias_is_visible_to_a_fit_with_an_intercept_and_not_to_a_temperature():
    gap, y = make_gaps(4000, scale=1.0, shift=3.0)
    fit = lt.fit_temperature(gap, y)
    platt = lt.fit_platt(gap, y)
    assert fit.nll - platt["nll"] > 0.02
    assert platt["intercept"] < -2.0
    report = lt.diagnose(gap, y, None, checkpoint_temperature=1.0)
    assert any("bias is present" in note for note in report["findings"])


# --- cause 4: the logits carry no signal -----------------------------------------------
def test_uninformative_logits_have_the_optimum_on_the_upper_bound():
    gap, y = make_gaps(3000, scale=10.0, informative=False)
    assert vendor_fit(gap, y)["temperature"][2] == 5.0
    assert lt.fit_temperature(gap, y).at_bound


# --- cause 5: one small slice ----------------------------------------------------------
def test_a_slice_of_400_rows_moves_the_fit_a_long_way():
    gap, y = make_gaps(6000, scale=5.0)
    sub = lt.subsample_argmins(gap, y, rows=400, repeats=60)
    assert sub["percentiles"]["5"] < 5.0 < sub["percentiles"]["95"] or (
        sub["share_above_vendor_max"] > 0.0
    )
    full = lt.fit_temperature(gap, y).value
    assert 3.0 < full < 8.0


# --- the objective itself is right ------------------------------------------------------
def test_the_two_option_objective_matches_the_vendor_softmax_nll():
    torch = pytest.importorskip("torch")
    gap, y = make_gaps(500, scale=3.0)
    for t in (0.5, 1.0, 4.0):
        z = torch.stack([torch.zeros(len(gap)), torch.tensor(gap, dtype=torch.float64)], 1) / t
        target = torch.stack([torch.tensor(1.0 - y), torch.tensor(y.astype(float))], 1)
        expected = float(-(target * torch.log_softmax(z, -1)).sum(-1).mean())
        assert lt.nll(gap, y, t) == pytest.approx(expected, rel=1e-6)


# --- the diagnostic functions -----------------------------------------------------------
def test_fit_temperature_finds_an_interior_optimum_and_reports_a_bound():
    gap, y = make_gaps(5000, scale=8.0)
    fit = lt.fit_temperature(gap, y)
    assert not fit.at_bound and fit.nll < fit.nll_at_one
    narrow = lt.fit_temperature(gap, y, lower=0.5, upper=5.0)
    assert narrow.at_bound and narrow.bound == "upper"


def test_nll_curve_covers_the_requested_range_and_its_minimum_matches_the_fit():
    gap, y = make_gaps(3000, scale=8.0)
    curve = lt.nll_curve(gap, y)
    assert curve["temperature"][0] == pytest.approx(0.05)
    assert curve["temperature"][-1] == pytest.approx(100.0)
    best = curve["temperature"][int(np.argmin(curve["nll"]))]
    assert best == pytest.approx(lt.fit_temperature(gap, y).value, rel=0.2)


def test_validate_inputs_refuses_bad_labels_and_non_finite_gaps():
    with pytest.raises(ValueError):
        lt.fit_temperature([0.0, np.inf], [0, 1])
    with pytest.raises(ValueError):
        lt.fit_temperature([0.0, 1.0], [0, 2])


def test_fold_ids_never_split_a_group():
    groups = np.repeat(np.arange(40), 5)
    folds = lt.fold_ids(groups)
    for g in range(40):
        assert len(set(folds[groups == g])) == 1
    assert set(folds) == set(range(5))
    with pytest.raises(ValueError):
        lt.fold_ids(np.array(["a", "b"]), n_splits=5)


def test_bootstrap_is_deterministic_and_brackets_the_value():
    gap, y = make_gaps(800, scale=4.0)
    groups = np.arange(len(y)) // 4
    prob = expit(gap / 4.0)
    first = lt.bootstrap_interval(y, prob, groups, lt.prob_ece, n_boot=200)
    again = lt.bootstrap_interval(y, prob, groups, lt.prob_ece, n_boot=200)
    assert first == again
    assert first["seed"] == 20260813 and first["cluster"] == "group"
    assert first["low"] <= first["high"]


def test_top1_ece_is_small_for_a_calibrated_model_and_large_for_an_overconfident_one():
    gap, y = make_gaps(20000, scale=1.0)
    assert lt.top1_ece(y, expit(gap)) < 0.03
    assert lt.top1_ece(y, expit(gap * 10)) > 0.1


def test_diagnose_reports_every_variant_out_of_fold_and_temperature_adds_nothing_to_isotonic():
    gap, y = make_gaps(3000, scale=12.0)
    groups = np.arange(len(y)) // 3
    report = lt.diagnose(gap, y, groups, checkpoint_temperature=5.0, n_boot=50)
    assert set(report["ece"]) == {
        "raw_checkpoint_temperature",
        "raw_temperature_one",
        "temperature",
        "isotonic",
        "temperature_isotonic",
        "isotonic_served",
    }
    iso = report["ece"]["isotonic"]["ece_prob"]["value"]
    both = report["ece"]["temperature_isotonic"]["ece_prob"]["value"]
    assert iso == pytest.approx(both, abs=1e-9)
    raw = report["ece"]["raw_checkpoint_temperature"]["ece_prob"]["value"]
    assert report["ece"]["temperature"]["ece_prob"]["value"] < raw
    assert report["temperature_fit"]["value"] > 5.0
    json.dumps(report, default=float)


def test_diagnose_without_groups_computes_only_the_raw_rows_and_says_so():
    gap, y = make_gaps(500, scale=3.0)
    report = lt.diagnose(gap, y, None, checkpoint_temperature=2.0, n_boot=20)
    assert set(report["ece"]) == {"raw_checkpoint_temperature", "raw_temperature_one"}
    assert "No group ids" in report["ece_protocol"]


def test_per_source_fit_shows_a_group_that_a_pooled_fit_hides():
    a, ya = make_gaps(1500, scale=2.0, seed=1)
    b, yb = make_gaps(1500, scale=40.0, seed=2)
    gap, y = np.concatenate([a, b]), np.concatenate([ya, yb])
    sources = np.array(["liar"] * 1500 + ["coaid"] * 1500)
    report = lt.diagnose(gap, y, None, checkpoint_temperature=5.0, sources=sources, n_boot=10)
    t = {k: v["temperature_fit"]["value"] for k, v in report["per_source"].items()}
    assert t["coaid"] > 5 * t["liar"]


# --- the calibrator guard ---------------------------------------------------------------
def test_fit_calibrator_reports_out_of_fold_ece_and_refuses_a_rise(monkeypatch):
    rng = np.random.default_rng(3)
    n = 1200
    groups = np.arange(n) // 2
    y = rng.integers(0, 2, n)
    scores = np.clip(0.5 * y + rng.normal(0.25, 0.2, n), 0, 1)
    _, report = fit_calibrator(scores, y, groups=groups)
    assert {"ece_before", "ece_after_oof"} <= set(report["oof"])
    assert report["oof"]["ece_after_oof"] <= report["oof"]["ece_before"] + MAX_OOF_ECE_RISE

    import modeling.text.laya_calibration as lc

    monkeypatch.setattr(
        lc, "isotonic_oof_ece", lambda *a, **k: {"ece_before": 0.05, "ece_after_oof": 0.09}
    )
    with pytest.raises(LayaCalibratorError) as info:
        fit_calibrator(scores, y, groups=groups)
    assert info.value.reason == REJECTED
    assert "out-of-fold" in str(info.value)


def test_fit_calibrator_without_groups_says_the_check_did_not_run():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 600)
    scores = np.clip(0.4 * y + rng.normal(0.3, 0.2, 600), 0, 1)
    _, report = fit_calibrator(scores, y)
    assert report["oof"] == {"oof_skipped": "no_groups"}


# --- the script -------------------------------------------------------------------------
def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "diagnose_laya_calibration.py"
    spec = importlib.util.spec_from_file_location("diagnose_laya_calibration", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_laya_calibration"] = module
    spec.loader.exec_module(module)
    return module


def test_the_script_reads_stored_gaps_without_a_gpu_and_writes_the_report(tmp_path, monkeypatch):
    script = load_script()
    gap, y = make_gaps(900, scale=10.0)
    ids = [f"v{i}" for i in range(len(y))]
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(
        {
            "split": "val",
            "id": ids,
            "text": "x",
            "label": y,
            "source_dataset": np.where(np.arange(len(y)) % 2, "liar", "coaid"),
            "group_id": [f"g{i // 3}" for i in range(len(y))],
        }
    ).to_parquet(run / "eval_rows.parquet", index=False)
    (run / "split.json").write_text(
        json.dumps({"description": "synthetic", "ids": {"val": ids, "test": []}})
    )
    out = tmp_path / "report.json"
    np.savez(
        out.with_suffix(".logits.npz"),
        gap=gap,
        positions=np.arange(len(y)),
        temperature=json.dumps({"applied": 5.0}),
        encode_max_len=512,
        n_val=len(y),
    )

    def no_gpu(*args, **kwargs):
        raise AssertionError("the GPU stage ran with --reuse-logits")

    monkeypatch.setattr(script.rc, "run_gpu_stage", no_gpu)
    code = script.main(
        [
            "--model", "laya_multi", "--out", str(out), "--comparison-dir", str(run),
            "--reuse-logits", "--n-boot", "20",
        ]
    )
    assert code == 0
    report = json.loads(out.read_text())
    assert report["model"] == "laya_multi" and report["rows"] == len(y)
    assert report["checkpoint_temperature"] == 5.0
    assert set(report["per_source"]) == {"liar", "coaid"}
    assert report["ece"]["isotonic"]["ece_prob"]["seed"] == 20260813
    assert report["nll_curve"]["temperature"][0] == pytest.approx(0.05)


def test_the_script_refuses_stored_gaps_of_another_validation_set(tmp_path):
    script = load_script()
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(
        {"split": "val", "id": ["a", "b"], "text": "x", "label": [0, 1],
         "source_dataset": "liar", "group_id": ["g1", "g2"]}
    ).to_parquet(run / "eval_rows.parquet", index=False)
    (run / "split.json").write_text(
        json.dumps({"description": "s", "ids": {"val": ["a", "b"], "test": []}})
    )
    out = tmp_path / "r.json"
    np.savez(out.with_suffix(".logits.npz"), gap=np.zeros(5), positions=np.arange(5),
             temperature=json.dumps({"applied": 1.0}), encode_max_len=1, n_val=5)
    code = script.main(["--model", "laya_en", "--out", str(out), "--comparison-dir", str(run),
                        "--reuse-logits"])
    assert code == 1
    assert not out.exists()
