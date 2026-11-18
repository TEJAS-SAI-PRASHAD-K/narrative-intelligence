"""Fact-check articles get no misinformation score.

A fact-check article quotes the false claim it debunks, so the misinformation
models score it high, and fusion would raise the risk of every narrative a
fact-checker answered. The ``misinfo`` stage gives those records a null value and
``misinfo:fact_check_publisher``. The ``laya`` side is in ``test_laya_stage.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from modeling import config as C
from modeling import scoring as S
from modeling.io import ScoredStore, _as_version_dict
from modeling.registry import Checkpoint

TEXT = "a claim that is comfortably longer than twenty characters"
DEBUNK = "Fact check: a viral video claiming the bridge collapsed is from 2016"
HINDI = "यह एक दावा है जो बीस अक्षरों से काफी लंबा है"


@pytest.fixture
def settings(monkeypatch):
    C.get_settings.cache_clear()
    monkeypatch.delenv("SCORE_UNKNOWN_LANGUAGE", raising=False)
    yield C.get_settings()
    C.get_settings.cache_clear()


class FakeMisinfo:
    seen: list[str] = []

    def __init__(self, settings):
        pass

    def load(self, path):
        return True

    def predict(self, texts):
        type(self).seen = list(texts)
        return [0.9] * len(texts)


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": "outlet", "source": "news", "source_detail": "thehindu.com",
             "text": TEXT, "lang": "en"},
            {"id": "boom", "source": "news", "source_detail": "boomlive.in",
             "text": DEBUNK, "lang": "en"},
            # Listed under other_fact_checkers, with no extraction tier, and with www.
            {"id": "altnews", "source": "news", "source_detail": "www.AltNews.in",
             "text": DEBUNK + " again", "lang": "en"},
            {"id": "hindi", "source": "news", "source_detail": "vishvasnews.com",
             "text": HINDI, "lang": "hi"},
        ]
    )


def make_ctx(tmp_path, settings, monkeypatch, records=None, store=None):
    import modeling.registry as registry
    import modeling.text.misinfo_clf as clf

    monkeypatch.setattr(
        registry, "resolve",
        lambda module, version, required=False: Checkpoint(
            module="misinfo", version=version, path=tmp_path, origin="local", sha256="ab" * 32
        ),
    )
    monkeypatch.setattr(clf, "MisinfoClassifier", FakeMisinfo)
    records = frame() if records is None else records
    return S.StageContext(
        settings=settings,
        reader=None,
        store=store or ScoredStore(settings, root=tmp_path / "scored"),
        records=records,
        authors=pd.DataFrame(),
        demo=True,
        dry_run=False,
        known_record_ids=set(records["id"]),
    )


def table(ctx) -> pd.DataFrame:
    return ctx.store.read("record_scores").set_index("record_id")


def test_fact_check_articles_are_not_scored_and_carry_the_reason(
    settings, monkeypatch, tmp_path
):
    ctx = make_ctx(tmp_path, settings, monkeypatch)
    out = S._misinfo_stage(ctx)

    assert FakeMisinfo.seen == [TEXT]
    rows = table(ctx)
    assert rows.loc["outlet", "misinfo_prob"] == pytest.approx(0.9)
    for rid in ("boom", "altnews"):
        assert pd.isna(rows.loc[rid, "misinfo_prob"])
        assert list(rows.loc[rid, "skip_reasons"]) == ["misinfo:fact_check_publisher"]
    assert "3 fact_check_publisher" in out[0]["note"]


def test_the_publisher_reason_wins_over_the_language_reason(settings, monkeypatch, tmp_path):
    ctx = make_ctx(tmp_path, settings, monkeypatch)
    S._misinfo_stage(ctx)
    assert list(table(ctx).loc["hindi", "skip_reasons"]) == ["misinfo:fact_check_publisher"]


def test_a_value_from_an_earlier_run_is_cleared_and_its_stamp_retracted(
    settings, monkeypatch, tmp_path
):
    ctx = make_ctx(tmp_path, settings, monkeypatch)
    stamp = {"misinfo": "v0.0.1+old", "toxicity": "v0.1.0"}
    ctx.store.write(
        "record_scores",
        pd.DataFrame(
            [{"record_id": "boom", "source": "news", "misinfo_prob": 0.97,
              "toxicity": 0.2, "model_versions": stamp, "skip_reasons": []}]
        ),
    )
    out = S._misinfo_stage(ctx)

    row = table(ctx).loc["boom"]
    assert pd.isna(row["misinfo_prob"])
    assert row["toxicity"] == pytest.approx(0.2)
    assert _as_version_dict(row["model_versions"]) == {"toxicity": "v0.1.0"}
    assert "cleared 1 earlier misinfo value" in out[0]["note"]


def test_a_value_another_module_wrote_is_left_alone(settings, monkeypatch, tmp_path):
    ctx = make_ctx(tmp_path, settings, monkeypatch)
    ctx.store.write(
        "record_scores",
        pd.DataFrame(
            [{"record_id": "outlet", "source": "news", "misinfo_prob": 0.4,
              "model_versions": {"laya": "v0.2.1"}, "skip_reasons": []},
             {"record_id": "hindi", "source": "news", "misinfo_prob": 0.3,
              "model_versions": {"laya": "v0.2.1"}, "skip_reasons": []}]
        ),
    )
    records = frame()
    records.loc[records["id"] == "hindi", "source_detail"] = "amarujala.com"
    ctx = make_ctx(tmp_path, settings, monkeypatch, records=records, store=ctx.store)
    S._misinfo_stage(ctx)

    row = table(ctx).loc["hindi"]
    assert row["misinfo_prob"] == pytest.approx(0.3)
    assert list(row["skip_reasons"]) == ["misinfo:unsupported_language"]
    assert _as_version_dict(row["model_versions"]) == {"laya": "v0.2.1"}


def test_a_second_run_changes_nothing(settings, monkeypatch, tmp_path):
    ctx = make_ctx(tmp_path, settings, monkeypatch)
    S._misinfo_stage(ctx)
    before = table(ctx)
    S._misinfo_stage(ctx)
    pd.testing.assert_frame_equal(before, table(ctx))


def test_records_without_source_detail_are_scored_with_a_warning(
    settings, monkeypatch, tmp_path, caplog
):
    records = frame().drop(columns=["source_detail"])
    ctx = make_ctx(tmp_path, settings, monkeypatch, records=records)
    S._misinfo_stage(ctx)
    assert "no `source_detail` column" in caplog.text
    assert table(ctx).loc["boom", "misinfo_prob"] == pytest.approx(0.9)


def test_a_missing_fact_check_config_fails_loudly(settings, monkeypatch, tmp_path):
    real = C.fact_check_domains
    monkeypatch.setattr(C, "fact_check_domains", lambda path=None: real(tmp_path / "none.yaml"))
    ctx = make_ctx(tmp_path, settings, monkeypatch)
    with pytest.raises(C.FactCheckDomainsMissing):
        S._misinfo_stage(ctx)
