"""Score with a candidate misinfo checkpoint without editing configs/models.yaml."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import modeling.registry as registry
from modeling import candidate as C
from modeling import scoring as S
from modeling.config import module_config
from tests import test_fact_check_scoring as fcs
from tests import test_run_comparison as trc
from tests.test_run_comparison import (  # noqa: F401 - fixtures
    fake_laya,
    fake_misinfo,
    prepared,
    prepared_template,
    scored,
)


@pytest.fixture
def models(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        models_dir=tmp_path / "models", hf_repo=None, hf_token=None, gdrive_dir=None,
        hf_revision=None, allow_unverified_pull=False,
    )
    monkeypatch.setattr(registry, "get_settings", lambda: settings)
    monkeypatch.delenv(C.VERSION_ENV, raising=False)
    monkeypatch.delenv(C.ALLOW_UNVERIFIED_ENV, raising=False)
    registry.REASON_COUNTS.clear()
    return settings


def make_candidate(settings, version="cand-1") -> Path:
    folder = settings.models_dir / "misinfo" / version
    folder.mkdir(parents=True)
    (folder / "model.bin").write_bytes(b"weights")
    (folder / "calibrator.json").write_text("{}")
    (folder / "config.json").write_text('{"model_type": "roberta"}')
    return folder


def test_a_candidate_without_a_hash_is_refused_by_default(models):
    make_candidate(models)
    with pytest.raises(C.CandidateRefused) as raised:
        C.resolve_candidate("misinfo", "cand-1")
    assert raised.value.reason == C.CandidateReason.UNVERIFIED_REFUSED
    assert registry.REASON_COUNTS[C.CandidateReason.UNVERIFIED_REFUSED] == 1


def test_the_flag_loads_it_unverified_with_a_loud_warning(models, caplog):
    make_candidate(models)
    with caplog.at_level(logging.WARNING):
        checkpoint = C.resolve_candidate("misinfo", "cand-1", unverified_ok=True)
    assert checkpoint.verified is False and checkpoint.sha256
    assert "UNVERIFIED CANDIDATE" in caplog.text


def test_the_env_flag_works_like_the_argument(models, monkeypatch):
    make_candidate(models)
    monkeypatch.setenv(C.ALLOW_UNVERIFIED_ENV, "1")
    assert C.resolve_candidate("misinfo", "cand-1").verified is False


def test_a_configured_hash_is_checked(models, monkeypatch):
    folder = make_candidate(models)
    digest = registry._dir_sha256(folder)
    config = {"version": "v0.2.0", "sha256": {"cand-1": digest}}
    monkeypatch.setattr(registry, "module_config", lambda module: config)
    monkeypatch.setattr(C, "module_config", lambda module: config)
    assert C.resolve_candidate("misinfo", "cand-1").verified is True
    config["sha256"]["cand-1"] = "0" * 64
    with pytest.raises(C.CandidateRefused) as raised:
        C.resolve_candidate("misinfo", "cand-1")
    assert raised.value.reason == registry.CheckpointReason.SHA256_MISMATCH


@pytest.mark.parametrize("version", ["../x", "a/b", "", "v0.2.0"])
def test_a_bad_version_or_the_configured_one_is_refused(models, version):
    assert version != "cand-1"
    with pytest.raises(C.CandidateRefused):
        C.resolve_candidate("misinfo", version or " ", unverified_ok=True)


def test_a_missing_folder_is_never_pulled(models):
    with pytest.raises(C.CandidateRefused) as raised:
        C.resolve_candidate("misinfo", "cand-9", unverified_ok=True)
    assert raised.value.reason == C.CandidateReason.NOT_FOUND


# ---------------------------------------------------------------------------
# the batch stage
# ---------------------------------------------------------------------------
@pytest.fixture
def stage_settings(monkeypatch):
    from modeling import config as cfg

    cfg.get_settings.cache_clear()
    yield cfg.get_settings()
    cfg.get_settings.cache_clear()


def test_the_stage_stamps_the_candidate_and_never_the_configured_version(
    stage_settings, models, monkeypatch, tmp_path
):
    folder = make_candidate(models)
    ctx = fcs.make_ctx(tmp_path, stage_settings, monkeypatch)
    # make_ctx replaces registry.resolve. The candidate path must not call it.
    monkeypatch.setattr(
        registry, "resolve", lambda *a, **k: pytest.fail("the candidate went through resolve")
    )
    C.set_request("cand-1", unverified=True)
    try:
        out = S._misinfo_stage(ctx)
    finally:
        for name in (C.VERSION_ENV, C.ALLOW_UNVERIFIED_ENV):
            import os

            os.environ.pop(name, None)
    rows = ctx.store.read("record_scores")
    stamps = {json.dumps(dict(v), sort_keys=True) for v in rows["model_versions"].dropna()}
    misinfo_stamps = [s for s in stamps if "misinfo" in s]
    assert misinfo_stamps and all("cand-1+" in s for s in misinfo_stamps)
    assert not any(module_config("misinfo")["version"] in s for s in misinfo_stamps)
    assert "CANDIDATE cand-1" in out[0]["note"]
    assert folder.exists()


def test_the_stage_reports_a_refused_candidate_and_scores_nothing(
    stage_settings, models, monkeypatch, tmp_path
):
    make_candidate(models)
    ctx = fcs.make_ctx(tmp_path, stage_settings, monkeypatch)
    monkeypatch.setenv(C.VERSION_ENV, "cand-1")
    out = S._misinfo_stage(ctx)
    assert "candidate_unverified_refused" in out[0]["note"]
    assert not len(ctx.store.read("record_scores"))


def test_the_stage_uses_the_configured_version_without_a_candidate(
    stage_settings, models, monkeypatch, tmp_path
):
    ctx = fcs.make_ctx(tmp_path, stage_settings, monkeypatch)
    S._misinfo_stage(ctx)
    rows = ctx.store.read("record_scores")
    assert any(module_config("misinfo")["version"] in json.dumps(dict(v))
               for v in rows["model_versions"].dropna())


# ---------------------------------------------------------------------------
# the comparison script
# ---------------------------------------------------------------------------
@pytest.fixture
def comparison_run(request):
    """A finished demo comparison. Resolved by name, so the imported fixtures stay unshadowed."""
    return request.getfixturevalue("scored")


def test_the_comparison_scores_with_a_candidate_and_stamps_its_version(
    models, comparison_run, tmp_path
):
    make_candidate(models, "cand-1")
    out = tmp_path / "candidate_out"
    shutil.copytree(comparison_run, out)
    args = ["score", "--demo", "--out", str(out), "--models", "misinfo",
            "--misinfo-version", "cand-1"]
    deps, *_ = trc.make_deps()
    assert trc.run(args, deps) == 1  # no sha256 in the config and no flag

    deps, *_ = trc.make_deps()
    assert trc.run([*args, "--allow-unverified-candidate"], deps) == 0
    provenance = json.loads((out / "scores" / "misinfo.provenance.json").read_text())
    assert provenance["model_version"] == "cand-1"
    # The run of the configured version stays as it was.
    original = json.loads((comparison_run / "scores" / "misinfo.provenance.json").read_text())
    assert original["model_version"] == module_config("misinfo")["version"]
    assert len(pd.read_parquet(out / "scores" / "misinfo.parquet"))


def test_the_unverified_flag_alone_is_a_usage_error(comparison_run):
    deps, *_ = trc.make_deps()
    args = ["score", "--demo", "--out", str(comparison_run), "--allow-unverified-candidate"]
    code = trc.run(args, deps)
    assert code == trc.rc.EXIT_USAGE
