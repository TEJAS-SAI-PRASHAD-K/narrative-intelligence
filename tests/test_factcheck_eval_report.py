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
from tests._factcheck_eval_support import COLUMNS, VAL_A, make_test_rows, small_case_rows
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


def test_the_report_states_each_pass_collapse_count_and_precision_not_measured(env):
    from tests.test_claim_dedup_embedding import write_links

    sample = env["data"] / "factchecks" / "claim_dedup_sample.csv"
    sample.write_text(
        "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,is_same_claim\n",
        encoding="utf-8",
    )
    write_links(
        env["data"],
        {"c1": "c1", "c2": "c1", "c3": "c3", "c4": "c4", "c5": "c5"},
        sample=sample,
    )

    result, paths = run_comparison(
        env["scores"], manifest_path=env["manifest"], n_bootstrap=50, seed=2
    )

    passes = result["evaluation_set"]["claim_dedup"]["passes"]
    assert passes["simhash"]["collapsed"] == 0
    assert passes["embedding"]["collapsed"] == 1
    assert result["evaluation_set"]["claim_dedup"]["precision"]["precision"] is None
    text = paths[1].read_text()
    assert "Claim deduplication (read side)" in text
    assert "| simhash | ok | 0 |" in text
    assert "| embedding | ok | 1 |" in text
    assert "not measured" in text
    assert "embedding claim dedup pass is not_run" not in text


def test_a_labelled_hand_check_records_embedding_precision(env):
    from tests.test_claim_dedup_embedding import write_links

    sample = env["data"] / "factchecks" / "claim_dedup_sample.csv"
    sample.write_text(
        "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,"
        "labeller_kind,labeller_1,labeller_2,label_1,label_2,is_same_claim\n"
        "c2,c1,0.91,false,x,false,y,human,ana,raj,yes,yes,yes\n"
        "c3,c1,0.88,false,x,false,y,human,ana,raj,no,no,no\n",
        encoding="utf-8",
    )
    write_links(
        env["data"],
        {"c1": "c1", "c2": "c1", "c3": "c1", "c4": "c4", "c5": "c5"},
        sample=sample,
    )

    result, paths = run_comparison(
        env["scores"], manifest_path=env["manifest"], n_bootstrap=50, seed=2
    )

    precision = result["evaluation_set"]["claim_dedup"]["precision"]
    assert precision["status"] == "measured"
    assert precision["precision"] == pytest.approx(0.5)
    text = paths[1].read_text()
    assert "embedding precision of collapses: 0.500" in text
    assert "labeller: human" in text
    assert "PROVISIONAL" not in text


def test_a_model_unreviewed_sample_is_marked_in_the_report(env):
    from tests.test_claim_dedup_embedding import write_links

    sample = env["data"] / "factchecks" / "claim_dedup_sample.csv"
    sample.write_text(
        "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,"
        "labeller_kind,labeller_1,labeller_2,label_1,label_2,is_same_claim\n"
        "c2,c1,0.91,false,x,false,y,model-unreviewed,m1,m2,yes,yes,yes\n"
        "c3,c1,0.88,false,x,false,y,model-unreviewed,m1,m2,no,no,no\n",
        encoding="utf-8",
    )
    write_links(
        env["data"],
        {"c1": "c1", "c2": "c1", "c3": "c1", "c4": "c4", "c5": "c5"},
        sample=sample,
    )

    result, paths = run_comparison(
        env["scores"], manifest_path=env["manifest"], n_bootstrap=50, seed=2
    )

    precision = result["evaluation_set"]["claim_dedup"]["precision"]
    assert precision["status"] == "provisional"
    assert precision["reason"] == "labeller_not_human"
    text = paths[1].read_text()
    assert "labeller: model, unreviewed by a person" in text
    assert "not a measured precision" in text
    assert "reason code labeller_not_human".lower() in text.lower()


def test_a_stale_sample_is_provisional_in_the_report(env):
    from tests.test_claim_dedup_embedding import write_links

    sample = env["data"] / "factchecks" / "claim_dedup_sample.csv"
    sample.write_text(
        "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,"
        "labeller_kind,labeller_1,labeller_2,label_1,label_2,is_same_claim\n"
        "c2,c1,0.91,false,x,false,y,human,ana,raj,yes,yes,yes\n"
        "c9,c1,0.88,false,x,false,y,human,ana,raj,no,no,no\n",
        encoding="utf-8",
    )
    write_links(
        env["data"],
        {"c1": "c1", "c2": "c1", "c3": "c3", "c4": "c4", "c5": "c5"},
        sample=sample,
    )

    result, paths = run_comparison(
        env["scores"], manifest_path=env["manifest"], n_bootstrap=50, seed=2
    )

    precision = result["evaluation_set"]["claim_dedup"]["precision"]
    assert precision["status"] == "provisional"
    assert precision["reason"] == "sample_pairs_stale"
    assert precision["n_stale"] == 1
    assert "1 of 2 labelled pairs are not in the current embedding links" in paths[1].read_text()


def test_a_refused_sample_does_not_stop_the_report(env):
    from tests.test_claim_dedup_embedding import write_links

    sample = env["data"] / "factchecks" / "claim_dedup_sample.csv"
    sample.write_text(
        "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,"
        "labeller_kind,labeller_1,labeller_2,label_1,label_2,is_same_claim\n"
        "c2,c1,0.91,false,x,false,y,human,ana,raj,yes,,\n",
        encoding="utf-8",
    )
    write_links(
        env["data"],
        {"c1": "c1", "c2": "c1", "c3": "c3", "c4": "c4", "c5": "c5"},
        sample=sample,
    )

    result, paths = run_comparison(
        env["scores"], manifest_path=env["manifest"], n_bootstrap=50, seed=2
    )

    assert result["evaluation_set"]["claim_dedup"]["precision"]["status"] == "refused"
    assert "half-filled" in paths[1].read_text()


def test_a_moved_sample_falls_back_beside_the_links(env):
    from tests.test_claim_dedup_embedding import write_links

    sample = env["data"] / "factchecks" / "claim_dedup_sample.csv"
    sample.write_text(
        "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,"
        "labeller_kind,labeller_1,labeller_2,label_1,label_2,is_same_claim\n"
        "c2,c1,0.91,false,x,false,y,human,ana,raj,yes,yes,yes\n",
        encoding="utf-8",
    )
    write_links(
        env["data"],
        {"c1": "c1", "c2": "c1", "c3": "c3", "c4": "c4", "c5": "c5"},
        sample=env["data"] / "gone" / "claim_dedup_sample.csv",
    )

    result, paths = run_comparison(
        env["scores"], manifest_path=env["manifest"], n_bootstrap=50, seed=2
    )

    precision = result["evaluation_set"]["claim_dedup"]["precision"]
    assert precision["precision"] == 1.0
    assert "does not exist" in precision["sample_note"]
    assert "does not exist" in paths[1].read_text()


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


def test_fact_checkers_without_an_extraction_tier_are_excluded_too():
    publishers = load_factcheck_publishers()
    assert {"altnews.in", "factly.in", "newschecker.in", "digiteye.in"} <= publishers


def test_every_fact_checker_feed_in_sources_is_listed():
    import yaml

    from modeling.config import CONFIG_DIR, normalise_domain

    # sources.yaml lists the fact-checker feeds first, under "--- fact-checkers"
    # headings, and the mainstream outlets after the first "--- Indian mainstream".
    text = (CONFIG_DIR / "sources.yaml").read_text()
    marker = "# --- Indian mainstream"
    assert marker in text, "sources.yaml no longer separates fact-checkers from mainstream feeds"
    head = text.split(marker)[0]
    feeds = yaml.safe_load(text)["news_rss"]["feeds"]
    fact_checker_feeds = [f for f in feeds if f in head]
    hosts = {normalise_domain(f.split("/")[2]) for f in fact_checker_feeds}
    assert hosts and hosts <= load_factcheck_publishers()


def test_other_fact_checkers_alone_are_enough(tmp_path):
    path = tmp_path / "factcheck.yaml"
    path.write_text("publishers: {}\nother_fact_checkers: [www.Example.in]\n")
    assert load_factcheck_publishers(path) == frozenset({"example.in"})


def test_empty_publisher_list_refuses(tmp_path):
    path = tmp_path / "factcheck.yaml"
    path.write_text("publishers: {}\n")
    with pytest.raises(EvalInputError, match="lists no publishers"):
        load_factcheck_publishers(path)


def test_the_report_shows_used_below_and_band_counts(env):
    from tests.test_claim_dedup_embedding import write_links

    header = (
        "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,"
        "labeller_kind,labeller_1,labeller_2,label_1,label_2,is_same_claim\n"
    )
    sample = env["data"] / "factchecks" / "claim_dedup_sample.csv"
    sample.write_text(
        header
        + "c2,c1,0.91,f,x,f,y,human,ana,raj,yes,yes,yes\n"
        + "c3,c1,0.70,f,x,f,y,human,ana,raj,no,no,no\n",
        encoding="utf-8",
    )
    write_links(
        env["data"],
        {"c1": "c1", "c2": "c1", "c3": "c1", "c4": "c4", "c5": "c5"},
        sample=sample,
    )

    result, paths = run_comparison(
        env["scores"], manifest_path=env["manifest"], n_bootstrap=50, seed=2
    )

    precision = result["evaluation_set"]["claim_dedup"]["precision"]
    assert (precision["n_labelled"], precision["chain"]["n"]) == (2, 1)
    assert precision["n_stale"] == 0
    text = paths[1].read_text()
    assert "embedding precision of collapses: 0.500 (1 of 2 labelled link pairs" in text
    assert "Direct links (similarity to the representative at or above 0.85): 1 of 1" in text
    assert "Chain links (below 0.85, joined through other members): 0 of 1" in text
    assert "| direct [0.85, 0.9) | 0 | 0 | no pairs in this band |" in text
    assert "| direct [0.9, 0.95) | 1 | 1 | 1.000 |" in text
    assert "| chain [0.0, 0.85) | 1 | 0 | 0.000 |" in text


def test_the_public_report_prints_no_machine_local_path(env, tmp_path):
    from tests.test_claim_dedup_embedding import write_links

    sample = env["data"] / "factchecks" / "claim_dedup_sample.csv"
    sample.write_text(
        "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,"
        "labeller_kind,labeller_1,labeller_2,label_1,label_2,is_same_claim\n"
        "c2,c1,0.91,false,x,false,y,human,ana,raj,yes,,\n",
        encoding="utf-8",
    )
    write_links(
        env["data"],
        {"c1": "c1", "c2": "c1", "c3": "c3", "c4": "c4", "c5": "c5"},
        sample=sample,
    )

    _, paths = run_comparison(
        env["scores"], manifest_path=env["manifest"], n_bootstrap=50, seed=2
    )

    for path in paths:
        assert str(tmp_path) not in path.read_text(encoding="utf-8")
    markdown = paths[1].read_text(encoding="utf-8")
    assert "half-filled" in markdown
    assert "Sample `factchecks/claim_dedup_sample.csv`" in markdown
    manifest = json.loads(paths[0].read_text())["evaluation_set"]["manifest"]
    assert manifest == "manifest.csv (outside the repository)"


def test_a_moved_sample_note_prints_no_machine_local_path(env, tmp_path):
    from tests.test_claim_dedup_embedding import write_links

    sample = env["data"] / "factchecks" / "claim_dedup_sample.csv"
    sample.write_text(
        "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,"
        "labeller_kind,labeller_1,labeller_2,label_1,label_2,is_same_claim\n"
        "c2,c1,0.91,false,x,false,y,human,ana,raj,yes,yes,yes\n",
        encoding="utf-8",
    )
    write_links(
        env["data"],
        {"c1": "c1", "c2": "c1", "c3": "c3", "c4": "c4", "c5": "c5"},
        sample=env["data"] / "gone" / "claim_dedup_sample.csv",
    )
    _, paths = run_comparison(
        env["scores"], manifest_path=env["manifest"], n_bootstrap=50, seed=2
    )

    for path in paths:
        assert str(tmp_path) not in path.read_text(encoding="utf-8")
    markdown = paths[1].read_text(encoding="utf-8")
    assert "(gone/claim_dedup_sample.csv) does not exist" in markdown


def test_display_path_prints_a_public_form_for_each_location(tmp_path):
    from ingest.config import REPO_ROOT
    from modeling.eval.factcheck_eval import display_path

    manifest = REPO_ROOT / "configs" / "factcheck_eval_manifest.csv"
    assert display_path(manifest) == "configs/factcheck_eval_manifest.csv"
    repo_data = REPO_ROOT / "data" / "factchecks" / "factchecks.parquet"
    assert display_path(repo_data, REPO_ROOT / "data") == "data/factchecks/factchecks.parquet"
    data = tmp_path / "data"
    assert display_path(data / "factchecks" / "claim_links.json", data) == (
        "factchecks/claim_links.json"
    )
    outside = tmp_path / "elsewhere" / "manifest.csv"
    assert display_path(outside, data) == "manifest.csv (outside the repository)"
    assert display_path(outside) == "manifest.csv (outside the repository)"


def test_redact_paths_rewrites_paths_in_nested_values_only(tmp_path):
    from modeling.eval.factcheck_eval import redact_paths

    data = tmp_path / "data"
    outside = tmp_path / "elsewhere" / "x.csv"
    value = {
        "sample": f"{data}/factchecks/s.csv",
        "error": f"{data}/factchecks/s.csv line 2: bad label; see {outside}.",
        "list": [f"{outside}", 1, None, 0.5],
        "url": "https://example.org/a/b",
        "ratio": "1/2 and 3 / 4",
    }
    assert redact_paths(value, data) == {
        "sample": "factchecks/s.csv",
        "error": "factchecks/s.csv line 2: bad label; see x.csv (outside the repository).",
        "list": ["x.csv (outside the repository)", 1, None, 0.5],
        "url": "https://example.org/a/b",
        "ratio": "1/2 and 3 / 4",
    }


def test_known_paths_with_a_space_are_redacted_by_value(tmp_path):
    from modeling.eval.factcheck_eval import redact_paths

    spaced = tmp_path / "with space"
    data = spaced / "data"
    manifest = spaced / "manifest file.csv"
    text = f"Sample `{data}/factchecks/s.csv` and manifest {manifest}."
    assert redact_paths(
        {"text": text, "sample": f"{data}/factchecks/s.csv"},
        data,
        known=[str(data), str(manifest)],
    ) == {
        "text": (
            "Sample `factchecks/s.csv` and manifest manifest file.csv (outside the repository)."
        ),
        "sample": "factchecks/s.csv",
    }


def test_a_moved_sample_with_a_space_in_its_path_prints_no_machine_local_path(env, tmp_path):
    from tests.test_claim_dedup_embedding import write_links

    sample = env["data"] / "factchecks" / "claim_dedup_sample.csv"
    sample.write_text(
        "record_id,kept_record_id,similarity,label,claim,kept_label,kept_claim,"
        "labeller_kind,labeller_1,labeller_2,label_1,label_2,is_same_claim\n"
        "c2,c1,0.91,false,x,false,y,human,ana,raj,yes,yes,yes\n",
        encoding="utf-8",
    )
    write_links(
        env["data"],
        {"c1": "c1", "c2": "c1", "c3": "c3", "c4": "c4", "c5": "c5"},
        sample=tmp_path / "old place" / "claim dedup sample.csv",
    )
    _, paths = run_comparison(
        env["scores"], manifest_path=env["manifest"], n_bootstrap=50, seed=2
    )

    for path in paths:
        assert str(tmp_path) not in path.read_text(encoding="utf-8")
    markdown = paths[1].read_text(encoding="utf-8")
    assert "claim dedup sample.csv (outside the repository)) does not exist" in markdown


def test_redaction_catches_a_path_after_a_key_and_windows_and_file_urls(tmp_path):
    from modeling.eval.factcheck_eval import redact_paths

    data = tmp_path / "data"
    assert redact_paths("manifest: /home/ana/m.csv", data) == (
        "manifest: m.csv (outside the repository)"
    )
    assert redact_paths(r"read C:\Users\ana\data\x.csv", data) == (
        "read x.csv (outside the repository)"
    )
    assert redact_paths("file:///home/ana/x.csv", data) == "x.csv (outside the repository)"
    assert redact_paths(f"file://{data}/factchecks/s.csv", data) == "factchecks/s.csv"


def test_redaction_leaves_a_url_scheme_and_a_route_as_written(tmp_path):
    from modeling.eval.factcheck_eval import redact_paths

    text = "https://example.org/a/b and /api/v1/items and /health"
    assert redact_paths(text, tmp_path / "data") == text


def test_redaction_rewrites_only_data_repo_and_system_root_paths(tmp_path):
    from modeling.eval.factcheck_eval import redact_paths

    data = tmp_path / "data"
    inside = data / "factchecks" / "s.csv"
    assert redact_paths(f"{inside}", data) == "factchecks/s.csv"
    assert redact_paths("/home/ana/x.csv", data) == "x.csv (outside the repository)"
    assert redact_paths("/var/lib/x.csv", data) == "x.csv (outside the repository)"
    assert redact_paths("~/notes/x.csv", data) == "x.csv (outside the repository)"
    assert redact_paths("/etc/hosts and /usr/share/x", data) == "/etc/hosts and /usr/share/x"
