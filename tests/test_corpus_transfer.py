"""Corpus-transfer sampling and evaluation, on a small synthetic sample and fake scores."""

from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from modeling import cli as modeling_cli
from modeling.eval import corpus_transfer as ct

runner = CliRunner()
LONG = "a sentence long enough to be scored by the stage"


def allow_en(lang):
    return lang is not None and str(lang) == "en"


def corpus() -> pd.DataFrame:
    rows = []
    for i in range(6):
        rows.append((f"r{i}", "reddit", "en", f"{LONG} {i}"))
    for i in range(6):
        rows.append((f"n{i}", "news", "en", f"{LONG} {i}"))
    for i in range(4):
        rows.append((f"h{i}", "news", "hi", f"{LONG} hindi {i}"))
    rows.append(("short", "news", "en", "too short"))
    rows.append(("nolang", "reddit", None, LONG))
    return pd.DataFrame(rows, columns=["id", "source", "lang", "text"])


class TestBuildSample:
    def test_every_sampled_row_is_scorable_and_unexcluded_rows_are_counted(self):
        result = ct.build_sample(corpus(), n=8, language_allowed=allow_en, seed=1)
        assert set(result.frame["lang"]) == {"en"}
        assert not set(result.frame["id"]) & {"short", "nolang", "h0", "h1", "h2", "h3"}
        assert result.excluded == {ct.LANGUAGE_NOT_SCORED: 5, ct.TEXT_TOO_SHORT: 1}
        assert len(result.frame) == 8

    def test_sheet_has_blank_labels_and_no_score_column(self):
        frame = ct.build_sample(corpus(), n=8, language_allowed=allow_en).frame
        assert list(frame.columns) == ["id", "source", "lang", "text", "label"]
        assert (frame["label"] == "").all()

    def test_stratified_by_source(self):
        frame = ct.build_sample(corpus(), n=8, language_allowed=allow_en).frame
        assert frame["source"].value_counts().to_dict() == {"news": 4, "reddit": 4}

    def test_language_filter_hi_with_english_only_module_counts_everything_excluded(self):
        result = ct.build_sample(corpus(), n=8, language_allowed=allow_en, lang="hi")
        assert len(result.frame) == 0
        assert result.pool == 4
        assert result.excluded == {ct.LANGUAGE_NOT_SCORED: 4}

    def test_language_filter_hi_samples_hindi_when_the_module_scores_it(self):
        result = ct.build_sample(
            corpus(), n=8, language_allowed=lambda lang: lang in ("en", "hi"), lang="hi"
        )
        assert sorted(result.frame["id"]) == ["h0", "h1", "h2", "h3"]

    def test_language_filter_en_drops_unknown_language_from_the_pool(self):
        result = ct.build_sample(corpus(), n=20, language_allowed=allow_en, lang="en")
        assert "nolang" not in set(result.frame["id"])
        assert result.pool == 13

    def test_unsupported_language_is_rejected(self):
        with pytest.raises(ValueError):
            ct.build_sample(corpus(), n=5, language_allowed=allow_en, lang="fr")

    def test_tops_up_when_a_source_is_small(self):
        records = corpus()
        frame = ct.build_sample(records[records["id"] != "n5"], n=11, language_allowed=allow_en)
        assert len(frame.frame) == 11


class TestSampleCommand:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        class S:
            artifacts_dir = tmp_path / "artifacts"
            seed = 0
            language_allowed = staticmethod(allow_en)

        monkeypatch.setattr(modeling_cli, "get_settings", lambda: S)
        monkeypatch.setattr(modeling_cli, "_bootstrap", lambda verbose=False: None)
        monkeypatch.setattr("modeling.io.CorpusReader.__init__", lambda self, settings=None: None)
        monkeypatch.setattr(
            "modeling.io.CorpusReader.records", lambda self, **kw: corpus()
        )
        return S

    def test_writes_sheet_and_reports_exclusions(self, env):
        result = runner.invoke(
            modeling_cli.app, ["sample-for-labelling", "misinfo", "--n", "8", "--lang", "en"]
        )
        assert result.exit_code == 0, result.output
        sheet = pd.read_csv(
            env.artifacts_dir / "hand_labels" / "misinfo_corpus_sample.csv",
            dtype=str, keep_default_na=False,
        )
        assert len(sheet) == 8 and (sheet["label"] == "").all()
        assert "text_too_short=1" in result.output

    def test_unscorable_language_is_a_clear_failure(self, env):
        result = runner.invoke(
            modeling_cli.app, ["sample-for-labelling", "misinfo", "--lang", "hi"]
        )
        assert result.exit_code == 1
        assert "nothing to label" in result.output

    def test_bad_lang_is_rejected(self, env):
        result = runner.invoke(
            modeling_cli.app, ["sample-for-labelling", "misinfo", "--lang", "fr"]
        )
        assert result.exit_code != 0
