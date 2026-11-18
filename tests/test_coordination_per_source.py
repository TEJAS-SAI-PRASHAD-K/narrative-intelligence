"""Per-source modularity against the time-shuffled null, on synthetic corpora.

Every corpus here is built by hand. A planted group posts one URL inside a short
window, and each author also has filler posts at author-specific times. The
within-author shuffle moves the planted posts onto filler times, so the null has
no coincidences left to find.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from modeling.accounts.coordination import (
    CoordinationDetector,
    null_comparison,
    null_model_section,
    write_report,
)

BASE = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    from modeling import config as C

    C.get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    value = C.get_settings()
    value.ensure_dirs()
    yield value
    C.get_settings.cache_clear()


def _row(source, author, n, when, urls, parent=None):
    return {
        "id": f"{source}:{author}-{n}",
        "author_id": f"{source}:{author}",
        "source": source,
        "text": f"post {n} by {author}",
        "timestamp": when,
        "parent_id": parent,
        "urls": urls,
        "domains": [],
        "hashtags": [],
        "simhash": None,
    }


def planted_source(source, groups=2, per_group=6, bursts=8, common=4):
    """Two groups that each share a URL inside a burst, plus a common background.

    Every author also posts to ``common`` shared URLs in fixed slots, so the shuffled
    corpus keeps real edges and the null has a distribution, not a row of empty graphs.
    """
    rows = []
    for g in range(groups):
        for a in range(per_group):
            author = f"g{g}a{a}"
            for b in range(bursts):
                when = BASE + timedelta(days=3 * b + g, minutes=a)
                rows.append(_row(source, author, b, when, [f"https://x.example/{g}/{b}"]))
            for k in range(common):
                when = BASE + timedelta(days=40 + 2 * k, minutes=a)
                rows.append(_row(source, author, 100 + k, when, [f"https://bg.example/{k}"]))
    return rows


def quiet_source(source, authors=12, with_parent=True):
    """Authors who share nothing: no edge can form."""
    rows = []
    for a in range(authors):
        for n in range(3):
            rows.append(
                _row(
                    source,
                    f"u{a}",
                    n,
                    BASE + timedelta(hours=a * 5 + n * 70),
                    [f"https://z.example/{source}/{a}/{n}"],
                    parent=f"{source}:root-{a}-{n}" if with_parent else None,
                )
            )
    return rows


def mixed_corpus():
    rows = planted_source("telegram")
    rows += quiet_source("youtube")
    rows += quiet_source("mastodon", authors=4)
    return pd.DataFrame(rows)


def detector(settings, shuffles=20):
    value = CoordinationDetector(settings)
    value.null_shuffles = shuffles
    # The research path names its sources, as `coordination-report --sources` does.
    value.eligible_sources = {"telegram", "youtube", "mastodon", "reddit"}
    return value


def test_the_null_has_real_edges_and_the_p_value_is_empirical(settings):
    result = detector(settings).detect(mixed_corpus(), per_source=True)
    row = result.per_source["telegram"]

    assert row["status"] == "measured"
    # A sparse shuffled graph has high modularity, so this corpus does not beat its
    # null. The test checks the arithmetic of the row, not a verdict.
    assert row["p_value"] == pytest.approx(
        (1 + sum(v >= row["observed"] for v in run_values(settings))) / 21, abs=1e-4
    )
    assert row["null_shuffles"] == 20
    assert row["null_empty_shuffles"] == 0
    assert row["null_mean_edges"] > 0
    assert row["p_floor"] == pytest.approx(1 / 21, abs=1e-4)
    assert row["null_low"] <= row["null_mean"] <= row["null_high"]


def run_values(settings):
    frame = pd.DataFrame(planted_source("telegram"))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return detector(settings)._null_run(frame)["values"]


def test_a_source_with_too_little_graph_gets_a_reason_code_and_stays_in_the_report(settings):
    result = detector(settings).detect(mixed_corpus(), per_source=True)

    assert result.per_source["mastodon"]["status"] == "too_few_authors"
    assert result.per_source["mastodon"]["authors"] == 4
    assert result.per_source["youtube"]["status"] == "no_edges"
    assert set(result.per_source) == {"telegram", "youtube", "mastodon"}
    table = null_model_section(result)
    assert "too_few_authors" in table and "no_edges" in table


def test_a_source_with_a_handful_of_edges_is_not_measured(settings):
    rows = quiet_source("youtube", authors=12)
    for author in ("u0", "u1"):
        rows.append(_row("youtube", author, 9, BASE + timedelta(days=40), ["https://s.example/"],
                         parent="youtube:p"))
    result = detector(settings).detect(pd.DataFrame(rows), per_source=True)

    assert result.per_source["youtube"]["status"] == "too_few_edges"
    assert result.per_source["youtube"]["edges"] == 1


def test_the_same_seed_gives_the_same_null_distribution(settings):
    first = detector(settings).detect(mixed_corpus(), per_source=True)
    second = detector(settings).detect(mixed_corpus(), per_source=True)

    assert first.per_source == second.per_source
    assert first.overall == second.overall
    assert settings.seed == 20260813


def test_overall_comparison_is_reported_beside_the_sources(settings):
    result = detector(settings).detect(mixed_corpus(), per_source=True)

    assert result.overall["status"] == "measured"
    assert result.overall["observed"] == pytest.approx(result.modularity, abs=1e-4)
    assert result.summary()["per_source"]["telegram"]["status"] == "measured"


def test_per_source_graphs_never_use_edges_between_sources(settings):
    rows = planted_source("telegram") + planted_source("mastodon")
    for row in rows:
        if row["source"] == "mastodon":
            row["author_id"] = row["author_id"] + "m"
            row["id"] = row["id"] + "m"
    frame = pd.DataFrame(rows)
    det = detector(settings, shuffles=3)
    result = det.detect(frame, per_source=True)

    for source in ("telegram", "mastodon"):
        alone = det._to_graph(det.build_edges(frame[frame["source"] == source]))
        assert result.per_source[source]["edges"] == alone.number_of_edges()
        assert result.per_source[source]["authors"] == 12


def test_reason_rows_survive_an_empty_overall_graph(settings):
    corpus = pd.DataFrame(quiet_source("youtube") + quiet_source("mastodon", authors=4))
    result = detector(settings, shuffles=3).detect(corpus, per_source=True)

    assert result.edges == []
    assert result.per_source["youtube"]["status"] == "no_edges"
    assert result.per_source["mastodon"]["status"] == "too_few_authors"
    assert "no_edges" in null_model_section(result)


def test_per_source_is_off_by_default(settings):
    result = detector(settings, shuffles=3).detect(mixed_corpus())
    assert result.per_source == {}


def test_empty_null_shuffles_are_counted_and_not_zero_filled(settings):
    det = detector(settings, shuffles=6)
    det.build_edges = lambda frame: []
    run = det._null_run(mixed_corpus().assign(timestamp=lambda f: pd.to_datetime(f["timestamp"])))

    assert run["values"] == [] and run["empty"] == 6
    row = null_comparison(0.4, run)
    assert row["status"] == "no_null" and row["p_value"] is None
    degenerate = null_comparison(0.4, {"values": [0.1, 0.2], "shuffles": 6, "empty": 4,
                                       "mean_edges": 1.0})
    assert degenerate["status"] == "null_degenerate"
    assert degenerate["null_empty_shuffles"] == 4
    assert degenerate["p_value"] == pytest.approx(1 / 3, abs=1e-4)


def test_a_row_with_no_null_renders_as_no_null(settings):
    result = detector(settings).detect(mixed_corpus(), run_null_model=False, per_source=True)
    row = result.per_source["telegram"]
    assert row["status"] == "no_null" and row["observed"] > 0
    assert "no_null" in null_model_section(result)


def test_forwarded_telegram_posts_are_dropped_and_counted(settings):
    import json

    rows = planted_source("telegram")
    for i, row in enumerate(rows):
        row["raw"] = json.dumps({"forwarded_from": {"id": 1}} if i % 4 == 0 else {})
    det = detector(settings, shuffles=2)
    det.eligible_sources = {"telegram"}
    work, excluded = det.eligible_records(pd.DataFrame(rows))

    assert excluded == {"telegram_forward": len([i for i in range(len(rows)) if i % 4 == 0])}
    assert len(work) == len(rows) - excluded["telegram_forward"]


def test_telegram_is_outside_the_production_eligible_set(settings):
    det = CoordinationDetector(settings)
    assert "telegram" not in det.eligible_sources
    assert det.threading_optional == {"mastodon", "telegram"}
    _, excluded = det.eligible_records(pd.DataFrame(quiet_source("telegram", authors=3)))
    assert excluded == {"telegram:source_not_eligible": 9}


def test_cli_rejects_zero_shuffles_and_warns_about_the_p_floor(settings):
    from typer.testing import CliRunner

    from modeling.cli import app

    bad = CliRunner().invoke(app, ["coordination-report", "--null-shuffles", "0", "--demo"])
    assert bad.exit_code != 0


def _null(values):
    return {"values": values, "shuffles": len(values), "empty": 0, "mean_edges": 5.0}


def test_null_comparison_arithmetic():
    row = null_comparison(0.5, _null([0.1, 0.2, 0.3, 0.6]))
    assert row["p_value"] == pytest.approx(2 / 5)
    assert row["null_mean"] == pytest.approx(0.3)
    assert row["null_low"] <= 0.1 + 0.01
    assert row["null_high"] >= 0.5
    assert null_comparison(0.9, _null([0.1, 0.2]))["p_value"] == pytest.approx(1 / 3, abs=1e-4)
    assert null_comparison(0.9, _null([]))["p_value"] is None


def test_a_flat_source_outside_the_optional_list_is_excluded(settings):
    flat = quiet_source("reddit", authors=3, with_parent=False)
    work, excluded = detector(settings).eligible_records(pd.DataFrame(flat))

    assert len(work) == 0
    assert excluded == {"reddit:no_threading": len(flat)}


def test_report_files_are_written(settings, tmp_path):
    result = detector(settings, shuffles=5).detect(mixed_corpus(), per_source=True)
    paths = write_report(result, tmp_path / "out")

    assert [p.name for p in paths] == ["report.md", "report.json"]
    text = paths[0].read_text()
    assert "| telegram | 12 |" in text
    assert "| mastodon | 4 |" in text and "too_few_authors" in text
