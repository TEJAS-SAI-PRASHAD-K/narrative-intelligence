"""Score with a candidate misinfo checkpoint without editing configs/models.yaml."""

from __future__ import annotations

import json
import logging
import re
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


def test_no_environment_variable_selects_or_allows_a_candidate(models, monkeypatch):
    make_candidate(models)
    monkeypatch.setenv("MISINFO_CANDIDATE_ALLOW_UNVERIFIED", "1")
    monkeypatch.setenv("MISINFO_CANDIDATE_VERSION", "cand-1")
    monkeypatch.setenv("MISINFO_CANDIDATE_REPLACE_OTHER_VERSION", "1")
    with pytest.raises(C.CandidateRefused):
        C.resolve_candidate("misinfo", "cand-1")
    assert not hasattr(C, "scoped_request") and not hasattr(C, "requested_version")


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
    ctx.candidate = C.CandidateOptions("cand-1", unverified=True)
    out = S._misinfo_stage(ctx)
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
    ctx.candidate = C.CandidateOptions("cand-1")
    out = S._misinfo_stage(ctx)
    assert "candidate_unverified_refused" in out[0]["note"]
    assert not len(ctx.store.read("record_scores"))


def test_a_value_with_no_misinfo_stamp_counts_as_another_version():
    existing = pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d"],
            "misinfo_prob": [0.5, 0.6, None, 0.7],
            "model_versions": [{}, {"misinfo": "cand-1+ab"}, {}, {"misinfo": "v0.2.0+ff"}],
        }
    )
    count = S._rows_with_other_misinfo_stamp(existing, {"a", "b", "c", "d"}, "cand-1+ab")
    assert count == 2  # a: no stamp. d: another stamp. b matches. c has no value.


def test_the_stage_uses_the_configured_version_without_a_candidate(
    stage_settings, models, monkeypatch, tmp_path
):
    ctx = fcs.make_ctx(tmp_path, stage_settings, monkeypatch)
    S._misinfo_stage(ctx)
    rows = ctx.store.read("record_scores")
    assert any(module_config("misinfo")["version"] in json.dumps(dict(v))
               for v in rows["model_versions"].dropna())


def seed_rows_from_another_version(ctx):
    """Write misinfo_prob rows that the configured version scored, for the candidate to meet."""
    S._misinfo_stage(ctx)
    return ctx.store.read("record_scores").set_index("record_id")["misinfo_prob"].copy()


def test_a_candidate_run_over_rows_of_another_version_refuses_and_writes_nothing(
    stage_settings, models, monkeypatch, tmp_path, caplog
):
    make_candidate(models)
    ctx = fcs.make_ctx(tmp_path, stage_settings, monkeypatch)
    before = seed_rows_from_another_version(ctx)
    assert before.notna().any()
    monkeypatch.setattr(registry, "resolve", lambda *a, **k: pytest.fail("resolve called"))
    ctx.candidate = C.CandidateOptions("cand-1", unverified=True)
    with caplog.at_level(logging.WARNING):
        out = S._misinfo_stage(ctx)
    assert out[0]["refused"] == C.CandidateReason.REPLACE_REQUIRED
    assert "1 row(s) carry another misinfo stamp" in out[0]["note"]
    assert "would overwrite misinfo_prob of 1 row" in caplog.text
    after = ctx.store.read("record_scores").set_index("record_id")["misinfo_prob"]
    assert after.equals(before)


def test_the_replace_flag_lets_the_candidate_overwrite(
    stage_settings, models, monkeypatch, tmp_path
):
    make_candidate(models)
    ctx = fcs.make_ctx(tmp_path, stage_settings, monkeypatch)
    seed_rows_from_another_version(ctx)
    monkeypatch.setattr(registry, "resolve", lambda *a, **k: pytest.fail("resolve called"))
    ctx.candidate = C.CandidateOptions("cand-1", unverified=True, replace_other=True)
    out = S._misinfo_stage(ctx)
    assert "refused" not in out[0]
    rows = ctx.store.read("record_scores")
    assert any("cand-1+" in json.dumps(dict(v)) for v in rows["model_versions"].dropna())


def invoke_score(monkeypatch, *args):
    from typer.testing import CliRunner

    import modeling.cli as cli

    monkeypatch.setattr(cli, "_bootstrap", lambda verbose=False: None)
    return CliRunner().invoke(cli.app, ["score", "misinfo", *args])


def test_the_cli_exits_1_for_a_refused_candidate(models, monkeypatch):
    make_candidate(models)
    result = invoke_score(monkeypatch, "--misinfo-version", "cand-1")
    assert result.exit_code == 1
    assert "candidate_unverified_refused" in result.output.replace("\n", "")
    result = invoke_score(
        monkeypatch, "--misinfo-version", "missing", "--allow-unverified-candidate"
    )
    assert result.exit_code == 1


def test_the_cli_passes_the_options_to_the_stages_and_exits_1_on_a_stage_refusal(
    models, monkeypatch
):
    make_candidate(models)
    seen = {}

    def fake_run_stages(stages, **kwargs):
        seen["candidate"] = kwargs.get("candidate")
        return [{"stage": "misinfo", "table": "record_scores", "refused": "some_reason",
                 "note": "n"}]

    monkeypatch.setattr(S, "run_stages", fake_run_stages)
    result = invoke_score(
        monkeypatch, "--misinfo-version", "cand-1", "--allow-unverified-candidate",
        "--replace-other-version",
    )
    assert result.exit_code == 1 and "some_reason" in result.output
    assert seen["candidate"] == C.CandidateOptions("cand-1", True, True)


def test_a_candidate_set_in_the_shell_does_not_select_a_candidate(models, monkeypatch):
    make_candidate(models)
    monkeypatch.setenv("MISINFO_CANDIDATE_VERSION", "cand-1")
    seen = {}
    monkeypatch.setattr(
        S, "run_stages", lambda stages, **kw: seen.update(kw) or [{"stage": "misinfo"}]
    )
    result = invoke_score(monkeypatch)
    assert result.exit_code == 0 and seen["candidate"] is None


def test_replace_needs_a_candidate(models, monkeypatch):
    result = invoke_score(monkeypatch, "--replace-other-version")
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# the comparison script
# ---------------------------------------------------------------------------
@pytest.fixture
def comparison_run(request):
    """A finished demo comparison. Resolved by name, so the imported fixtures stay unshadowed."""
    return request.getfixturevalue("scored")


def candidate_args(out, *extra):
    return ["score", "--demo", "--out", str(out), "--misinfo-version", "cand-1", *extra]


def test_the_comparison_scores_the_candidate_under_its_own_key(models, comparison_run, tmp_path):
    make_candidate(models, "cand-1")
    out = tmp_path / "candidate_out"
    shutil.copytree(comparison_run, out)
    original_file = (out / "scores" / "misinfo.parquet").read_bytes()
    deps, *_ = trc.make_deps()
    assert trc.run(candidate_args(out), deps) == 1  # no sha256 in the config and no flag
    assert not (out / "scores" / "misinfo@cand-1.parquet").exists()

    deps, *_ = trc.make_deps()
    assert trc.run(candidate_args(out, "--allow-unverified-candidate"), deps) == 0
    provenance = json.loads((out / "scores" / "misinfo@cand-1.provenance.json").read_text())
    # The same `<version>+<sha8>` stamp as the batch scorer.
    assert provenance["model"] == "misinfo@cand-1"
    assert re.fullmatch(r"cand-1\+[0-9a-f]{8}", provenance["model_version"])
    # The configured version stays as it was, so both load together.
    assert (out / "scores" / "misinfo.parquet").read_bytes() == original_file
    from modeling.eval.factcheck_eval import load_score_dir

    both = load_score_dir(out / "scores", names=["misinfo", "misinfo@cand-1"])
    assert set(both) == {"misinfo", "misinfo@cand-1"}


def test_a_score_file_of_another_version_is_not_overwritten_without_force(
    models, comparison_run, tmp_path, capsys
):
    make_candidate(models, "cand-1")
    out = tmp_path / "candidate_out"
    shutil.copytree(comparison_run, out)
    flag = "--allow-unverified-candidate"
    deps, *_ = trc.make_deps()
    assert trc.run(candidate_args(out, flag), deps) == 0
    sidecar = out / "scores" / "misinfo@cand-1.provenance.json"
    recorded = json.loads(sidecar.read_text())
    recorded["model_version"] = "cand-1+00000000"
    sidecar.write_text(json.dumps(recorded))
    before = (out / "scores" / "misinfo@cand-1.parquet").read_bytes()
    # The checkpoint identity is stale too, so the stage decides to score again.
    recorded["checkpoint"] = {"kind": "directory", "sha256": "stale"}
    sidecar.write_text(json.dumps(recorded))

    capsys.readouterr()
    deps, *_ = trc.make_deps()
    assert trc.run(candidate_args(out, flag), deps) == 1
    assert trc.rc.SCORE_FILE_OTHER_VERSION in capsys.readouterr().err
    assert (out / "scores" / "misinfo@cand-1.parquet").read_bytes() == before

    deps, *_ = trc.make_deps()
    assert trc.run(candidate_args(out, flag, "--force"), deps) == 0
    stamp = json.loads(sidecar.read_text())["model_version"]
    assert re.fullmatch(r"cand-1\+[0-9a-f]{8}", stamp)


def test_a_score_file_with_no_sidecar_is_not_overwritten_without_force(
    models, comparison_run, tmp_path, capsys
):
    make_candidate(models, "cand-1")
    out = tmp_path / "candidate_out"
    shutil.copytree(comparison_run, out)
    flag = "--allow-unverified-candidate"
    deps, *_ = trc.make_deps()
    assert trc.run(candidate_args(out, flag), deps) == 0
    (out / "scores" / "misinfo@cand-1.provenance.json").unlink()
    before = (out / "scores" / "misinfo@cand-1.parquet").read_bytes()
    capsys.readouterr()
    deps, *_ = trc.make_deps()
    assert trc.run(candidate_args(out, flag), deps) == 1
    assert trc.rc.SCORE_FILE_NO_SIDECAR in capsys.readouterr().err
    assert (out / "scores" / "misinfo@cand-1.parquet").read_bytes() == before
    deps, *_ = trc.make_deps()
    assert trc.run(candidate_args(out, flag, "--force"), deps) == 0


def test_the_version_check_runs_before_the_checkpoint_loads(
    models, comparison_run, tmp_path, monkeypatch
):
    make_candidate(models, "cand-1")
    out = tmp_path / "candidate_out"
    shutil.copytree(comparison_run, out)
    flag = "--allow-unverified-candidate"
    deps, *_ = trc.make_deps()
    assert trc.run(candidate_args(out, flag), deps) == 0
    sidecar = out / "scores" / "misinfo@cand-1.provenance.json"
    recorded = json.loads(sidecar.read_text())
    recorded["model_version"] = "cand-1+00000000"
    recorded["checkpoint"] = {"kind": "directory", "sha256": "stale"}
    sidecar.write_text(json.dumps(recorded))
    loads = []
    original_load = trc.FakeMisinfo.load
    monkeypatch.setattr(
        trc.FakeMisinfo, "load", lambda self, d: loads.append(d) or original_load(self, d)
    )
    deps, *_ = trc.make_deps()
    assert trc.run(candidate_args(out, flag), deps) == 1
    assert loads == []


def test_the_unverified_flag_alone_is_a_usage_error(comparison_run):
    deps, *_ = trc.make_deps()
    args = ["score", "--demo", "--out", str(comparison_run), "--allow-unverified-candidate"]
    code = trc.run(args, deps)
    assert code == trc.rc.EXIT_USAGE
