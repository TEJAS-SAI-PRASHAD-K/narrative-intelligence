"""Per-source modularity against the time-shuffled null, on synthetic corpora.

Every corpus here is built by hand. A planted group posts one URL inside a short
window, and each author also has filler posts at author-specific times. The
within-author shuffle moves the planted posts onto filler times, so the null has
no coincidences left to find.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
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


def planted_source(source, groups=2, per_group=6, bursts=2, filler=40):
    """Two groups that each share a URL inside a burst, plus filler posts."""
    rows = []
    for g in range(groups):
        for a in range(per_group):
            author = f"g{g}a{a}"
            for b in range(bursts):
                when = BASE + timedelta(days=3 * b + g, minutes=a)
                rows.append(_row(source, author, b, when, [f"https://x.example/{g}/{b}"]))
            for f in range(filler):
                when = BASE + timedelta(days=10 + 20 * f + a, hours=g + 1, minutes=7 * f)
                rows.append(_row(source, author, 100 + f, when, [f"https://y.example/{author}/{f}"]))
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
                    [f"https://z.example/{a}/{n}"],
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
    return value


def test_planted_source_beats_its_null_and_the_p_value_is_empirical(settings):
    result = detector(settings).detect(mixed_corpus())
    row = result.per_source["telegram"]

    assert row["status"] == "measured"
    assert row["observed"] > row["null_high"]
    assert row["null_shuffles"] == 20
    assert row["p_value"] == pytest.approx(1 / 21, abs=1e-4)
    assert row["null_low"] <= row["null_mean"] <= row["null_high"]


def test_a_source_with_too_little_graph_gets_a_reason_code_and_stays_in_the_report(settings):
    result = detector(settings).detect(mixed_corpus())

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
    result = detector(settings).detect(pd.DataFrame(rows))

    assert result.per_source["youtube"]["status"] == "too_few_edges"
    assert result.per_source["youtube"]["edges"] == 1


def test_the_same_seed_gives_the_same_null_distribution(settings):
    first = detector(settings).detect(mixed_corpus())
    second = detector(settings).detect(mixed_corpus())

    assert first.per_source == second.per_source
    assert first.overall == second.overall
    assert detector(settings).null_seed == 20260813


def test_overall_comparison_is_reported_beside_the_sources(settings):
    result = detector(settings).detect(mixed_corpus())

    assert result.overall["status"] == "measured"
    assert result.overall["observed"] == pytest.approx(result.modularity, abs=1e-4)
    assert result.summary()["per_source"]["telegram"]["status"] == "measured"


def test_per_source_graphs_never_use_edges_between_sources(settings):
    rows = planted_source("telegram") + planted_source("mastodon")
    for row in rows:
        if row["source"] == "mastodon":
            row["author_id"] = row["author_id"] + "m"
            row["id"] = row["id"] + "m"
    result = detector(settings, shuffles=5).detect(pd.DataFrame(rows))

    assert result.per_source["telegram"]["authors"] == 12
    assert result.per_source["mastodon"]["authors"] == 12


def test_null_comparison_arithmetic():
    row = null_comparison(0.5, [0.1, 0.2, 0.3, 0.6])
    assert row["p_value"] == pytest.approx(2 / 5)
    assert row["null_mean"] == pytest.approx(0.3)
    assert row["null_low"] <= 0.1 + 0.01
    assert row["null_high"] >= 0.5
    assert null_comparison(0.9, [0.1, 0.2])["p_value"] == pytest.approx(1 / 3, abs=1e-4)
    assert null_comparison(0.9, [])["p_value"] is None


def test_telegram_is_eligible_without_threads_and_a_flat_source_is_not(settings):
    flat = quiet_source("reddit", authors=3, with_parent=False)
    tele = quiet_source("telegram", authors=3, with_parent=False)
    work, excluded = CoordinationDetector(settings).eligible_records(pd.DataFrame(flat + tele))

    assert set(work["source"]) == {"telegram"}
    assert excluded == {"reddit:no_threading": len(flat)}


def test_engagement_columns_are_not_read(settings):
    corpus = mixed_corpus()
    corpus["engagement"] = [None] * len(corpus)
    first = detector(settings, shuffles=3).detect(corpus, run_null_model=False)
    second = detector(settings, shuffles=3).detect(mixed_corpus(), run_null_model=False)

    assert first.modularity == second.modularity
    assert not np.isnan(first.modularity)


def test_report_files_are_written(settings, tmp_path):
    result = detector(settings, shuffles=5).detect(mixed_corpus())
    paths = write_report(result, tmp_path / "out")

    assert [p.name for p in paths] == ["report.md", "report.json"]
    assert "telegram" in paths[0].read_text()
