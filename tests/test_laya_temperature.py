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

from modeling.text import laya_calibration as lc
from modeling.text import laya_temperature as lt
from modeling.text.laya_calibration import REJECTED, LayaCalibratorError, fit_calibrator

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


def records_of(gap, y):
    return [
        (2, np.array([0.0, g], dtype=np.float32), np.array([1.0 - t, float(t)]), 2)
        for g, t in zip(gap, y, strict=True)
    ]


def fake_fit_temperature_map(records):
    """A weight-free stand-in for ``laya.calibrate.fit_temperature_map``.

    It keeps the two facts that the clamp tests rely on: the objective is the NLL of the
    softmax over the options, and the answer is clamped to [0.5, 5] with a note that names
    the bound. The search is wide, as in the vendor code, so the clamp is what stops the fit.
    """
    gap = np.array([r[1][1] - r[1][0] for r in records], dtype=float)
    y = np.array([int(np.argmax(r[2])) for r in records])
    raw = lt.fit_temperature(gap, y, lower=0.5 * np.exp(-10), upper=5.0 * np.exp(10)).value
    applied = min(max(raw, lt.VENDOR_T_MIN), lt.VENDOR_T_MAX)
    clamped = {}
    if applied != raw:
        clamped["noul"] = {
            "fitted": raw,
            "applied": applied,
            "bound": "max" if raw > lt.VENDOR_T_MAX else "min",
        }
    return {"temperature": [1.0, 1.0, applied], "temperature_clamped": clamped}


@pytest.fixture(params=["fake", "real"])
def vendor_fit(request):
    """The vendor fit. The real one needs the laya extra and is skipped without it."""
    if request.param == "fake":
        return lambda gap, y: fake_fit_temperature_map(records_of(gap, y))
    calibrate = pytest.importorskip("laya.calibrate")
    return lambda gap, y: calibrate.fit_temperature_map(records_of(gap, y))


def test_the_fake_agrees_with_the_real_vendor_fit_when_it_is_installed():
    calibrate = pytest.importorskip("laya.calibrate")
    gap, y = make_gaps(2000, scale=30.0)
    real = calibrate.fit_temperature_map(records_of(gap, y))
    fake = fake_fit_temperature_map(records_of(gap, y))
    assert real["temperature"][2] == fake["temperature"][2]
    assert real["temperature_clamped"]["noul"]["bound"] == "max"
    assert fake["temperature_clamped"]["noul"]["bound"] == "max"


# --- cause 1: the model is very overconfident -------------------------------------------
def test_overconfident_logits_push_the_vendor_fit_to_the_clamp_and_the_wide_fit_inside(vendor_fit):
    gap, y = make_gaps(3000, scale=30.0)
    result = vendor_fit(gap, y)
    assert result["temperature"][2] == 5.0
    assert result["temperature_clamped"]["noul"]["bound"] == "max"
    fit = lt.fit_temperature(gap, y)
    assert not fit.at_bound
    assert 15.0 < fit.value < 60.0


# --- cause 2: the target is the wrong way round -----------------------------------------
def test_a_flipped_target_has_no_interior_minimum(vendor_fit):
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
def test_uninformative_logits_have_the_optimum_on_the_upper_bound(vendor_fit):
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
def test_fit_calibrator_reports_the_out_of_fold_interval():
    rng = np.random.default_rng(3)
    n = 1200
    groups = np.arange(n) // 2
    y = rng.integers(0, 2, n)
    scores = np.clip(0.5 * y + rng.normal(0.25, 0.2, n), 0, 1)
    _, report = fit_calibrator(scores, y, groups=groups)
    oof = report["oof"]
    assert {"ece_before", "ece_after_oof", "difference_low", "difference_high"} <= set(oof)
    assert oof["seed"] == 20260813 and len(oof["fold_seeds"]) == 5


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_a_perfectly_calibrated_raw_score_passes_at_400_rows(seed):
    rng = np.random.default_rng(seed)
    n = 400
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.random(n) < p).astype(int)
    _, report = fit_calibrator(p, y, groups=np.arange(n) // 2)
    assert report["oof"]["difference_low"] <= lc.OOF_DIFFERENCE_LIMIT


def test_a_calibrator_that_makes_the_scores_worse_is_refused(monkeypatch):
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 600)
    scores = np.clip(0.4 * y + rng.normal(0.3, 0.2, 600), 0, 1)
    monkeypatch.setattr(
        lc,
        "isotonic_oof_ece",
        lambda *a, **k: {
            "ece_before": 0.05, "ece_after_oof": 0.09,
            "difference_low": 0.01, "difference_high": 0.07,
        },
    )
    with pytest.raises(LayaCalibratorError) as info:
        fit_calibrator(scores, y, groups=np.arange(600) // 2)
    assert info.value.reason == REJECTED
    assert "out-of-fold" in str(info.value)


def test_a_missing_groups_check_is_counted_and_logged(caplog):
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 600)
    scores = np.clip(0.4 * y + rng.normal(0.3, 0.2, 600), 0, 1)
    with caplog.at_level("WARNING"):
        _, report = fit_calibrator(scores, y)
    assert report["oof"] == {"oof_skipped": "no_groups"}
    assert "did not run" in caplog.text


# --- the temperature refit and the rounding ----------------------------------------------
def test_a_wide_refit_before_the_rounding_keeps_ranks_that_the_clamp_loses():
    from sklearn.metrics import roc_auc_score

    gap, y = make_gaps(6000, scale=30.0)
    refit = lt.fit_temperature(gap, y)
    assert not refit.at_bound and refit.value > lt.VENDOR_T_MAX
    clamped = lt.served_scores(gap, lt.VENDOR_T_MAX)
    wide = lt.served_scores(gap, refit.value)
    assert len(np.unique(wide)) > len(np.unique(clamped))
    assert roc_auc_score(y, wide) > roc_auc_score(y, clamped)
    # Re-scaling the rounded value cannot undo the ties that the rounding made.
    from scipy.special import logit

    late = expit(logit(np.clip(clamped, 1e-4, 1 - 1e-4)) * lt.VENDOR_T_MAX / refit.value)
    assert len(np.unique(np.round(late, 4))) <= len(np.unique(clamped))


class FakeAgent:
    temperature = [1.0, 1.0, 5.0]
    temperature_by_options = {"noul:2": 5.0, "choice:2": 2.0}
    lang_temperatures = {"hi": {"temperature": [1, 1, 2]}}


def test_install_noul_temperature_acts_on_the_agent_and_removes_the_bucket_value():
    agent = FakeAgent()
    lt.install_noul_temperature(agent, 27.5)
    assert agent.temperature[2] == 27.5
    assert "noul:2" not in agent.temperature_by_options
    assert agent.temperature_by_options["choice:2"] == 2.0
    assert agent.lang_temperatures == {}
    with pytest.raises(ValueError):
        lt.install_noul_temperature(agent, float("nan"))


def write_state(directory, *, refit):
    from modeling.checkpoint_identity import directory_identity

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text('{"model_type": "fake"}')
    (directory / "model.safetensors").write_bytes(b"weights")
    rng = np.random.default_rng(7)
    scores = rng.uniform(0.6, 1.0, 400)
    labels = (rng.uniform(size=400) < 0.1 + 2.0 * (scores - 0.6)).astype(int)
    calibrator, report = fit_calibrator(scores, labels)
    identity = directory_identity(directory, ignore=lc.CALIBRATOR_FILES)
    lc.write_calibrator(
        directory, calibrator, report, demo=False, checkpoint_identity=identity,
        temperature_refit={"value": 27.5} if refit else None,
    )


def test_the_stamp_shows_a_refitted_temperature(tmp_path):
    plain = tmp_path / "plain"
    refit = tmp_path / "refit"
    write_state(plain, refit=False)
    write_state(refit, refit=True)
    a = lc.load_calibrator(plain)
    b = lc.load_calibrator(refit, temperature_refit=True)
    assert a.temperature is None and "+t" not in a.stamp
    assert b.temperature == 27.5 and "+t27.5@" in b.stamp
    assert a.stamp != b.stamp
    assert lc.read_temperature_refit(refit) == 27.5


def test_the_calibrator_and_the_config_must_agree_on_the_refit(tmp_path):
    plain = tmp_path / "plain"
    refit = tmp_path / "refit"
    write_state(plain, refit=False)
    write_state(refit, refit=True)
    with pytest.raises(LayaCalibratorError) as info:
        lc.load_calibrator(plain, temperature_refit=True)
    assert info.value.reason == lc.TEMPERATURE_MISSING
    with pytest.raises(LayaCalibratorError) as info:
        lc.load_calibrator(refit, temperature_refit=False)
    assert info.value.reason == lc.TEMPERATURE_UNEXPECTED
    with pytest.raises(LayaCalibratorError):
        lc.read_temperature_refit(plain)


def test_the_scorer_applies_the_refit_only_when_the_config_asks_for_it(tmp_path, monkeypatch):
    import sys
    import types

    from modeling.config import get_settings
    from modeling.text.laya_clf import LayaScorer

    directory = tmp_path / "multi"
    write_state(directory, refit=True)
    loaded = []

    def fake_load(checkpoint, **kwargs):
        agent = FakeAgent()
        agent.temperature = [1.0, 1.0, 5.0]
        agent.temperature_by_options = {"noul:2": 5.0}
        agent.lang_temperatures = {}
        loaded.append(agent)
        return agent

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=fake_load))
    base = {
        "route": {"hi": "multilingual_checkpoint"},
        "finetuned_checkpoints": {"hi": str(directory)},
    }
    off = LayaScorer(settings=get_settings(), _config={**base, "temperature_refit": False})
    off.agent_for("hi")
    assert loaded[-1].temperature[2] == 5.0
    on = LayaScorer(settings=get_settings(), _config={**base, "temperature_refit": True})
    on.agent_for("hi")
    assert loaded[-1].temperature[2] == 27.5
    assert loaded[-1].temperature_by_options == {}


def test_platt_standardises_and_reports_convergence():
    gap, y = make_gaps(3000, scale=1e4)
    platt = lt.fit_platt(gap, y)
    assert platt["converged"] and platt["convergence_warnings"] == []
    assert abs(platt["slope"] * 1e4 - 1.0) < 0.2


# --- the script -------------------------------------------------------------------------
def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "diagnose_laya_calibration.py"
    spec = importlib.util.spec_from_file_location("diagnose_laya_calibration", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_laya_calibration"] = module
    spec.loader.exec_module(module)
    return module


class Gpu:
    def __init__(self, free):
        self.free = list(free)
        self.calls = 0

    def __call__(self, index):
        value = self.free[min(self.calls, len(self.free) - 1)]
        self.calls += 1
        script = load_script()
        return script.rc.GpuMemory(index, 40960, 1000, value)


def make_run(tmp_path, monkeypatch, n=900, scale=10.0):
    script = load_script()
    gap, y = make_gaps(n, scale=scale)
    ids = [f"v{i}" for i in range(n)]
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(
        {
            "split": "val",
            "id": ids,
            "text": "x",
            "label": y,
            "source_dataset": np.where(np.arange(n) % 2, "liar", "coaid"),
            "group_id": [f"g{i // 3}" for i in range(n)],
        }
    ).to_parquet(run / "eval_rows.parquet", index=False)
    (run / "split.json").write_text(
        json.dumps({"description": "synthetic", "ids": {"val": ids, "test": []}})
    )
    monkeypatch.setattr(
        script.rc, "checkpoint_identity", lambda ctx, model: {"sha256": "abc", "kind": "directory"}
    )
    return script, run, gap, y, ids


def store(script, out, gap, ids, *, model="laya_multi", sha="abc", temperature=None, **extra):
    fields = {
        "gap": gap,
        "positions": np.arange(len(gap)),
        "ids": np.array(ids),
        "long_positions": np.array([], dtype=int),
        "blank_rows": 0,
        "model": model,
        "checkpoint_sha256": sha,
        "validation_ids_hash": script.lt.ids_hash(ids),
        "temperature": json.dumps(temperature or {"applied": 5.0}),
        "encode_max_len": 512,
        "device": "cuda:0",
    }
    fields.update(extra)
    return script.lt.write_gap_cache(out.with_name(out.stem + ".logits"), **fields)


def argv(run, out, *more, model="laya_multi"):
    return ["--model", model, "--out", str(out), "--comparison-dir", str(run), "--n-boot", "20",
            *more]


def test_the_script_reads_stored_gaps_without_a_gpu_and_writes_the_report(tmp_path, monkeypatch):
    script, run, gap, _y, ids = make_run(tmp_path, monkeypatch)
    out = tmp_path / "reports" / "report.json"
    store(script, out, gap, ids)

    def no_gpu(*args, **kwargs):
        raise AssertionError("the GPU stage ran with --reuse-logits")

    monkeypatch.setattr(script.rc, "run_gpu_stage", no_gpu)
    assert script.main(argv(run, out, "--reuse-logits")) == 0
    report = json.loads(out.read_text())
    assert report["model"] == "laya_multi" and report["rows"] == len(gap)
    assert report["valid"] is True and report["device"] == "cuda:0"
    assert report["checkpoint_temperature"] == 5.0
    assert set(report["per_source"]) == {"liar", "coaid"}
    assert report["ece"]["isotonic"]["ece_prob"]["seed"] == 20260813
    assert report["ece"]["isotonic"]["ece_top1"]["value"] >= 0.0
    assert report["nll_curve"]["temperature"][0] == pytest.approx(0.05)


@pytest.mark.parametrize(
    "change", [{"model": "laya_en"}, {"sha": "other"}, {"ids": "ids"}]
)
def test_a_stored_file_for_another_model_checkpoint_or_rows_is_refused(
    tmp_path, monkeypatch, change
):
    script, run, gap, _y, ids = make_run(tmp_path, monkeypatch, n=60)
    out = tmp_path / "r.json"
    if "ids" in change:
        store(script, out, gap, ids, validation_ids_hash="0" * 64)
    else:
        store(script, out, gap, ids, **change)
    assert script.main(argv(run, out, "--reuse-logits")) == 1
    assert not out.exists()


def test_a_logit_path_without_the_npz_suffix_is_read_where_it_was_written(tmp_path, monkeypatch):
    script, run, gap, _y, ids = make_run(tmp_path, monkeypatch, n=300)
    out = tmp_path / "r.json"
    written = store(script, out, gap, ids)
    bare = str(written)[: -len(".npz")]
    assert script.lt.gap_cache_path(bare) == written
    assert script.main(argv(run, out, "--reuse-logits", "--logits", bare)) == 0


def fake_collect(gap, ids, device="cuda:0", **temperature):
    def collect(ctx, model, frame, requested):
        n = len(gap)
        return {
            "gap": gap,
            "positions": np.arange(n),
            "long_positions": np.array([], dtype=int),
            "blank_rows": 0,
            "temperature": {"applied": 5.0, **temperature},
            "encode_max_len": 512,
            "device": device,
        }

    return collect


def test_wait_for_gpu_waits_then_proceeds(tmp_path, monkeypatch):
    script, run, gap, _y, ids = make_run(tmp_path, monkeypatch, n=300)
    monkeypatch.setattr(script, "collect_gaps", fake_collect(gap, ids))
    sleeps: list[float] = []
    clock = {"now": 0.0}

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    deps = script.rc.Deps(
        gpu_reader=Gpu([100, 100, 30000]), sleep=sleep, clock=lambda: clock["now"],
        out=__import__("io").StringIO(),
    )
    out = tmp_path / "deep" / "dir" / "r.json"
    code = script.main(argv(run, out, "--wait-for-gpu"), deps=deps)
    assert code == 0 and len(sleeps) == 2
    assert out.exists() and out.with_name("r.logits.npz").exists()


def test_without_wait_for_gpu_a_short_gpu_is_refused(tmp_path, monkeypatch):
    script, run, gap, _y, ids = make_run(tmp_path, monkeypatch, n=300)
    monkeypatch.setattr(script, "collect_gaps", fake_collect(gap, ids))
    deps = script.rc.Deps(
        gpu_reader=Gpu([100]), sleep=lambda s: None, clock=lambda: 0.0,
        out=__import__("io").StringIO(),
    )
    out = tmp_path / "r.json"
    assert script.main(argv(run, out), deps=deps) == script.rc.EXIT_GPU_BUSY
    assert not out.exists()


def test_the_agent_must_load_on_the_requested_device(tmp_path, monkeypatch):
    import sys
    import types

    script, run, gap, _y, ids = make_run(tmp_path, monkeypatch, n=60)

    class Agent:
        device = "cpu"
        cfg = {"max_len": 512}

    class Scorer:
        question_id = "misinfo"

        def __init__(self, settings=None, _config=None):
            self.config = _config

        def available(self):
            return types.SimpleNamespace(ok=True)

        def agent_for(self, lang):
            return Agent()

        def question(self):
            return {"misinfo": {}}

        def _is_long(self, text, lang):
            return False

    import modeling.text.laya_clf as clf

    monkeypatch.setattr(clf, "LayaScorer", Scorer)
    monkeypatch.setitem(
        sys.modules, "laya.calibrate", types.SimpleNamespace(records_from_labeled=None)
    )
    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(calibrate=None))
    deps = script.rc.Deps(
        gpu_reader=Gpu([30000]), sleep=lambda s: None, clock=lambda: 0.0,
        out=__import__("io").StringIO(),
    )
    out = tmp_path / "r.json"
    assert script.main(argv(run, out), deps=deps) == script.rc.EXIT_FAILED
    assert not out.exists()


@pytest.mark.parametrize(
    "temperature, expected",
    [
        ({"binning_map_installed": True}, "binning map"),
        ({"lang_temperatures": ["hi"]}, "own temperature"),
    ],
)
def test_the_report_is_invalid_when_the_agent_does_not_match_the_scorer(
    tmp_path, monkeypatch, temperature, expected
):
    script, run, gap, _y, ids = make_run(tmp_path, monkeypatch, n=300)
    out = tmp_path / "r.json"
    store(script, out, gap, ids, temperature={"applied": 5.0, **temperature})
    assert script.main(argv(run, out, "--reuse-logits")) == 0
    report = json.loads(out.read_text())
    assert report["valid"] is False
    assert any(expected in reason for reason in report["invalid_reasons"])


def test_the_report_is_invalid_when_the_rebuilt_scores_differ_from_the_score_file():
    script = load_script()
    check = {"compared": 100, "share_above_tolerance": 0.2}
    assert script.invalid_reasons({}, check)
    check["share_above_tolerance"] = script.SCORE_MISMATCH_LIMIT
    assert script.invalid_reasons({}, check) == []
