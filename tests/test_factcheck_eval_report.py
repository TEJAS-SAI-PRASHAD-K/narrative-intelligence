"""End to end: score files and fact-check data in, JSON and Markdown out, and the CLI."""

from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from modeling import cli
from modeling.config import get_settings
from modeling.eval.factcheck_eval import (
    EvalInputError,
    ManifestMismatch,
    build_manifest,
    load_factcheck_publishers,
    run_comparison,
    write_manifest,
)
from tests.test_factcheck_eval import COLUMNS, VAL_A, make_test_rows, small_case_rows
from tests.test_factcheck_manifest import _write_factchecks

runner = CliRunner()

SIMHASHES = [
    0xAAAAAAAAAAAAAAAA,
    0x5555555555555555,
    0xF0F0F0F0F0F0F0F0,
    0x0F0F0F0F0F0F0F0F,
    0xFF00FF00FF00FF00,
]
CLAIMS = [f"c{i}" for i in range(1, 6)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Data and artifacts under tmp_path, scores for two models, five fact-check rows."""
    data, artifacts = tmp_path / "data", tmp_path / "artifacts"
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts))
    get_settings.cache_clear()
    scores = tmp_path / "scores"
    scores.mkdir()
    rows = VAL_A + make_test_rows([0.9, 0.9, 0.1, 0.1]) + small_case_rows()
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for name in ("tfidf", "cand"):
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, scores / f"{name}.parquet")
    _write_factchecks(
        data,
        [
            {"record_id": c, "claim": f"claim text {c}", "simhash": h}
            for c, h in zip(CLAIMS, SIMHASHES, strict=True)
        ],
    )
    yield {
        "scores": scores,
        "data": data,
        "artifacts": artifacts,
        "manifest": tmp_path / "manifest.csv",
    }
    get_settings.cache_clear()


def freeze(env):
    from modeling.eval.factcheck_eval import freeze_manifest_from_data

    freeze_manifest_from_data(env["data"], env["manifest"])


def test_run_writes_json_and_markdown_with_the_required_content(env):
    result, paths = run_comparison(
        env["scores"],
        manifest_path=env["manifest"],
        n_bootstrap=100,
        seed=3,
        versions={"cand": "v0.2.0"},
    )
    assert [p.name for p in paths] == ["factcheck_eval.json", "factcheck_eval.md"]
    assert paths[0].parent == env["artifacts"] / "eval" / "factcheck" / "comparison"

    payload = json.loads(paths[0].read_text())  # parses as strict JSON
    assert payload["models"]["cand"]["version"] == "v0.2.0"
    assert payload["models"]["cand"]["operating_point"]["threshold"] == 0.8
    assert payload["cells"]["cand"]["en"]["claims"]["n_scored"] == 3
    assert payload["run"]["seed"] == 3
    assert result["seed"] == 3

    text = paths[1].read_text()
    assert "evaluation set not frozen" in text
    assert "only the claims that publishers chose to debunk" in text
    assert "0.8000" in text  # the threshold
    assert "no value" in text or "undefined" in text  # the hi title cell has flag rate 0
    assert "Hindi gate" in text and "FAIL" in text
    assert "skip_reason" in text


def test_a_frozen_set_is_not_flagged_as_unfrozen(env):
    freeze(env)
    _, paths = run_comparison(env["scores"], manifest_path=env["manifest"], n_bootstrap=50, seed=1)
    text = paths[1].read_text()
    assert "evaluation set not frozen" not in text
    assert json.loads(paths[0].read_text())["evaluation_set"]["manifest_rows"] == 5


def test_manifest_mismatch_stops_the_harness_and_writes_nothing(env):
    freeze(env)
    # Change one claim after freezing.
    _write_factchecks(
        env["data"],
        [
            {"record_id": c, "claim": f"claim text {c}" + (" EDITED" if c == "c2" else ""),
             "simhash": h}
            for c, h in zip(CLAIMS, SIMHASHES, strict=True)
        ],
    )
    with pytest.raises(ManifestMismatch, match="claim hash differs: c2"):
        run_comparison(env["scores"], manifest_path=env["manifest"], n_bootstrap=50)
    assert not (env["artifacts"] / "eval" / "factcheck").exists()


def test_a_manifest_row_missing_from_the_data_stops_the_harness(env):
    rows = build_manifest(
        [{"record_id": "gone", "claim": "x", "label": "false", "url": "https://x.org"}]
    )
    write_manifest(rows, env["manifest"])
    with pytest.raises(ManifestMismatch, match="missing from the fact-check rows: gone"):
        run_comparison(env["scores"], manifest_path=env["manifest"], n_bootstrap=50)


def test_cli_runs_and_prints_the_decision(env):
    result = runner.invoke(
        cli.app,
        [
            "evaluate-factchecks",
            str(env["scores"]),
            "--manifest",
            str(env["manifest"]),
            "--n-bootstrap",
            "50",
            "--model-version",
            "cand=v0.2.0",
            "--label",
            "t",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "evaluation set not frozen" in result.output
    assert "cand: do not ship" in result.output
    assert "Hindi gate FAIL" in result.output
    assert (env["artifacts"] / "eval" / "factcheck" / "t" / "factcheck_eval.md").exists()


def test_cli_refuses_a_changed_frozen_set(env):
    freeze(env)
    text = env["manifest"].read_text().replace("c3,", "c3x,")
    env["manifest"].write_text(text)
    result = runner.invoke(
        cli.app,
        ["evaluate-factchecks", str(env["scores"]), "--manifest", str(env["manifest"])],
    )
    assert result.exit_code == 1
    assert "refused" in result.output
    assert not (env["artifacts"] / "eval" / "factcheck").exists()


def test_cli_refuses_a_malformed_score_file(env):
    (env["scores"] / "cand.parquet").write_text("not parquet")
    result = runner.invoke(
        cli.app,
        ["evaluate-factchecks", str(env["scores"]), "--manifest", str(env["manifest"])],
    )
    assert result.exit_code == 1 and "refused" in result.output
    assert not (env["artifacts"] / "eval" / "factcheck").exists()


def test_cli_refuses_a_bad_model_version_argument(env):
    result = runner.invoke(
        cli.app, ["evaluate-factchecks", str(env["scores"]), "--model-version", "novalue"]
    )
    assert result.exit_code != 0


def test_freeze_command_writes_hashes_not_text_and_refuses_to_replace(env):
    first = runner.invoke(cli.app, ["freeze-factchecks", "--out", str(env["manifest"])])
    assert first.exit_code == 0, first.output
    text = env["manifest"].read_text()
    assert text.splitlines()[0] == "record_id,url,claim_sha256,verdict"
    assert len(text.splitlines()) == 6
    assert "claim text" not in text

    again = runner.invoke(cli.app, ["freeze-factchecks", "--out", str(env["manifest"])])
    assert again.exit_code == 1 and "already exists" in again.output
    forced = runner.invoke(
        cli.app, ["freeze-factchecks", "--out", str(env["manifest"]), "--overwrite"]
    )
    assert forced.exit_code == 0


def test_real_config_lists_the_fact_check_publishers():
    publishers = load_factcheck_publishers()
    assert {"boomlive.in", "thequint.com", "vishvasnews.com", "factcrescendo.com"} <= publishers


def test_empty_publisher_list_refuses(tmp_path):
    path = tmp_path / "factcheck.yaml"
    path.write_text("publishers: {}\n")
    with pytest.raises(EvalInputError, match="lists no publishers"):
        load_factcheck_publishers(path)
