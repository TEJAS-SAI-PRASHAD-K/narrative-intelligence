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
            language_allowed = staticmethod(
                lambda module, lang: module == "misinfo" and allow_en(lang)
            )

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


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
VERSION = "v0.1.0"
POS = [0.9, 0.8, 0.6, 0.4]
NEG = [0.7, 0.5, 0.3, 0.1]


def make_sheet() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eight usable labelled rows plus every kind of row that must stay out.

    Hand-calculated at threshold 0.5 (score >= 0.5 is flagged):
    TP=3 (0.9, 0.8, 0.6), FN=1 (0.4), FP=2 (0.7, 0.5), TN=2 (0.3, 0.1).
    misinfo class: precision 3/5, recall 3/4, F1 = 2*0.6*0.75/1.35 = 2/3.
    other class: precision 2/3, recall 2/4, F1 = 4/7. Macro-F1 = (2/3 + 4/7)/2.
    ROC-AUC: positives beat negatives in 4+4+3+2 = 13 of 16 pairs.
    """
    rows, scores = [], []

    def add(rid, label, score, lang="en", text=LONG, versions=None):
        rows.append({"id": rid, "source": "news", "lang": lang, "text": text, "label": label})
        if score is not None or versions is not None:
            scores.append(
                {"record_id": rid, "misinfo_prob": score,
                 "model_versions": versions or {"misinfo": f"{VERSION}+abcd1234"}}
            )

    for i, s in enumerate(POS):
        add(f"p{i}", "1", s)
    for i, s in enumerate(NEG):
        add(f"n{i}", "0", s)
    add("hi", "1", None, lang="hi")  # null score, language the module does not score
    add("short", "0", None, text="tiny")  # null score, too short
    add("missing", "1", None)  # scorable but the scorer never produced a score
    add("stale", "1", 0.99, versions={"misinfo": "v0.0.9+ffffffff"})
    add("blank1", "", 0.99)  # unlabelled rows carry extreme scores on purpose
    add("blank2", "  ", 0.99)
    add("blank3", "", 0.01)
    add("junk", "yes", 0.99)
    return pd.DataFrame(rows), pd.DataFrame(scores)


def run(sheet, scores, **kw):
    kw.setdefault("min_labelled", 8)
    return ct.evaluate_transfer(
        sheet, scores, language_allowed=allow_en, model_version=VERSION,
        threshold=0.5, n_bootstrap=50, **kw,
    )


class TestEvaluateTransfer:
    def test_metrics_match_hand_calculation(self):
        out = run(*make_sheet())
        assert out.computed
        m = out.metrics
        assert out.used == 8
        assert m["precision"] == pytest.approx(0.6, abs=1e-4)
        assert m["recall"] == pytest.approx(0.75, abs=1e-4)
        assert m["f1"] == pytest.approx(2 / 3, abs=1e-4)
        assert m["macro_f1_value"] == pytest.approx((2 / 3 + 4 / 7) / 2, abs=1e-3)
        assert m["roc_auc"].value == pytest.approx(13 / 16)
        assert m["flag_rate"] == pytest.approx(5 / 8)
        assert m["positive_rate"] == pytest.approx(0.5)

    def test_null_scores_are_excluded_with_reasons_not_scored_as_negative(self):
        out = run(*make_sheet())
        assert out.excluded == {
            ct.LANGUAGE_NOT_SCORED: 1,
            ct.TEXT_TOO_SHORT: 1,
            ct.NO_SCORE: 1,
            ct.STALE_MODEL_VERSION: 1,
        }
        # Treating the three null-score positives as 0 would drop recall to 3/7.
        assert out.metrics["recall"] == pytest.approx(0.75, abs=1e-4)

    def test_blank_label_is_not_zero_and_invalid_is_counted_apart(self):
        out = run(*make_sheet())
        assert (out.unlabelled, out.invalid_labels) == (3, 1)
        assert out.labelled == 12
        assert out.sample_rows == 16

    def test_a_real_zero_score_is_used_not_excluded(self):
        sheet, scores = make_sheet()
        scores.loc[scores["record_id"] == "n3", "misinfo_prob"] = 0.0
        out = run(sheet, scores)
        assert out.used == 8 and out.metrics["roc_auc"].value == pytest.approx(13 / 16)

    def test_refuses_below_the_named_threshold_with_the_default(self):
        sheet, scores = make_sheet()
        out = ct.evaluate_transfer(
            sheet, scores, language_allowed=allow_en, model_version=VERSION, threshold=0.5
        )
        assert ct.MIN_LABELLED_ROWS == 50
        assert not out.computed and out.metrics == {}
        assert "need at least 50" in out.refusal
        assert out.labelled == 12

    def test_refuses_when_only_blank_labels(self):
        sheet, scores = make_sheet()
        sheet["label"] = ""
        out = run(sheet, scores)
        assert not out.computed and out.labelled == 0 and out.unlabelled == 16

    def test_refuses_when_too_few_labelled_rows_have_a_usable_score(self):
        sheet, scores = make_sheet()
        out = run(sheet, scores, min_labelled=9)
        assert not out.computed
        assert out.labelled == 12 and out.used == 8
        assert "usable score" in out.refusal

    def test_refuses_with_one_class(self):
        sheet, scores = make_sheet()
        sheet.loc[sheet["label"] == "0", "label"] = "1"
        assert "one class" in run(sheet, scores).refusal

    def test_no_scores_at_all_counts_every_row_as_no_score(self):
        sheet, _ = make_sheet()
        out = run(sheet, None, min_labelled=1)
        assert not out.computed
        assert out.excluded[ct.NO_SCORE] == 10

    def test_sheet_labels_are_the_only_label_source(self):
        sheet, scores = make_sheet()
        flipped = sheet.copy()
        flipped["label"] = flipped["label"].replace({"1": "0", "0": "1"})
        assert run(flipped, scores).metrics["roc_auc"].value == pytest.approx(3 / 16)


class TestEvaluateCommand:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        class S:
            artifacts_dir = tmp_path / "artifacts"
            seed = 0
            language_allowed = staticmethod(
                lambda module, lang: module == "misinfo" and allow_en(lang)
            )

        monkeypatch.setattr(modeling_cli, "get_settings", lambda: S)
        monkeypatch.setattr(modeling_cli, "_bootstrap", lambda verbose=False: None)
        monkeypatch.setattr(
            "modeling.training.evaluate_module",
            lambda module, **kw: __import__("modeling.training", fromlist=["x"]).TrainingResult(
                module, VERSION, None, skipped=True, notes=["no saved predictions"]
            ),
        )
        monkeypatch.setattr("modeling.io.ScoredStore.__init__", lambda self, settings=None: None)
        sheet, scores = make_sheet()
        monkeypatch.setattr("modeling.io.ScoredStore.read", lambda self, name: scores)
        monkeypatch.setattr(
            "modeling.config.module_config",
            lambda module: {"version": VERSION, "decision_threshold": 0.5},
        )
        monkeypatch.setattr(ct, "MIN_LABELLED_ROWS", 8)
        (S.artifacts_dir / "hand_labels").mkdir(parents=True)
        sheet.to_csv(S.artifacts_dir / "hand_labels" / "misinfo_corpus_sample.csv", index=False)
        return S

    def test_reports_transfer_when_a_labelled_sample_exists(self, env):
        result = runner.invoke(modeling_cli.app, ["evaluate", "misinfo"])
        assert result.exit_code == 0, result.output
        assert "corpus transfer" in result.output
        assert "labelled 12" in result.output
        assert "ROC-AUC 0.812" in result.output
        assert "language_not_scored=1" in result.output

    def test_says_not_measured_when_no_sample_exists(self, env):
        (env.artifacts_dir / "hand_labels" / "misinfo_corpus_sample.csv").unlink()
        result = runner.invoke(modeling_cli.app, ["evaluate", "misinfo"])
        assert result.exit_code == 0
        assert "NOT MEASURED" in result.output

    def test_refusal_exits_nonzero_and_prints_no_metrics(self, env):
        path = env.artifacts_dir / "hand_labels" / "misinfo_corpus_sample.csv"
        sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
        sheet["label"] = ""
        sheet.to_csv(path, index=False)
        result = runner.invoke(modeling_cli.app, ["evaluate", "misinfo"])
        assert result.exit_code == 1
        assert "REFUSED" in result.output and "ROC-AUC" not in result.output

    def test_other_modules_have_no_transfer_section(self, env):
        result = runner.invoke(modeling_cli.app, ["evaluate", "stance"])
        assert "corpus transfer" not in result.output
