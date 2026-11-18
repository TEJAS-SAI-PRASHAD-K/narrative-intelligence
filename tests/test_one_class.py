"""A sample with one class: what the transfer evaluators still report. Synthetic data only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from modeling import cli as modeling_cli
from modeling.eval import corpus_transfer as ct
from modeling.eval import labelling as lb
from modeling.eval import laya_transfer as lt
from modeling.eval import one_class as oc
from modeling.eval import weighted as wt
from tests.test_labelling import env  # noqa: F401

runner = CliRunner()
VERSION = "v0.1.0"
LONG = "a sentence long enough to be scored by the stage"
NOTE = "labeller: model, unreviewed by a person"


def allow(lang):
    return lang == "en"


def build(n=60, positives=0, kind="human", lang="en", module="misinfo"):
    """Sheet, scores and design. Even rows are in stratum A (pool 120), odd rows in B (pool 30).

    Row ``i`` has probability 0.8 when ``i % 4 == 0`` and 0.2 otherwise, so exactly one row
    in four is flagged at 0.5. Every fifth row reports on a false claim. The first
    ``positives`` rows are labelled 1.
    """
    rows, scores, design = [], [], []
    for i in range(n):
        rid = f"x{i}"
        rows.append(
            {
                "id": rid,
                "source": "news",
                "lang": lang,
                "text": LONG,
                "adjudicated_label": "1" if i < positives else "0",
                "adjudicated_reports_on_false_claim": "1" if i % 5 == 0 else "0",
                "labeller_kind": kind,
            }
        )
        scores.append(
            {
                "record_id": rid,
                "misinfo_prob": 0.8 if i % 4 == 0 else 0.2,
                "model_versions": {module: f"{VERSION}+ab"},
            }
        )
        design.append(
            {
                "id": rid,
                "stratum": "A" if i % 2 == 0 else "B",
                "pool": "120" if i % 2 == 0 else "30",
                "drawn": str(n // 2),
                "weight": "4" if i % 2 == 0 else "1",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(scores), pd.DataFrame(design)


def english(threshold=0.5, with_design=True, **kw):
    sheet, scores, design = build(**kw)
    resolved = lb.resolve_sheet(
        sheet, design=design if with_design else None, allow_model_labels=True
    )
    return ct.evaluate_transfer(
        resolved,
        scores,
        language_allowed=allow,
        model_version=VERSION,
        threshold=threshold,
        min_labelled=20,
        n_bootstrap=200,
    )


def hindi(threshold=0.5, **kw):
    sheet, scores, design = build(lang="hi", module="laya", **kw)
    resolved = lb.resolve_sheet(sheet, design=design, allow_model_labels=True)
    return lt.evaluate_laya_transfer(
        resolved,
        scores,
        model_version=VERSION,
        threshold=threshold,
        threshold_source="laya.decision_threshold" if threshold is not None else "unset",
        min_labelled=20,
        n_bootstrap=200,
    )


class TestWilson:
    def test_zero_of_150_has_an_upper_limit_of_about_0025(self):
        low, high = wt.wilson_interval(0, 150)
        assert low == 0.0 and high == pytest.approx(0.025, abs=0.0005)

    def test_no_rows_gives_nan(self):
        assert all(np.isnan(v) for v in wt.wilson_interval(0, 0))

    def test_all_flagged_has_an_upper_limit_of_one(self):
        assert wt.wilson_interval(20, 20)[1] == 1.0


class TestNegativeOnlySample:
    def test_the_auc_stays_refused_with_its_reason_code(self):
        out = english()
        assert not out.computed and oc.ONE_CLASS in out.refusal
        assert "one class" in out.refusal and out.metrics == {}

    def test_positive_rate_is_zero_with_a_wilson_interval(self):
        report = english().one_class
        rate = report["positive_rate"]
        assert (rate["k"], rate["n"], rate["value"]) == (0, 60, 0.0)
        assert rate["ci_high"] == pytest.approx(wt.wilson_interval(0, 60)[1])

    def test_flag_rate_unweighted_and_weighted(self):
        flag = english().one_class["flag"]
        assert flag["threshold"] == 0.5
        assert flag["unweighted_wilson"]["k"] == 15
        assert flag["unweighted_wilson"]["value"] == pytest.approx(0.25)
        assert flag["unweighted_bootstrap"].value == pytest.approx(0.25)
        # rows 0, 4, 8, ... are flagged: 15 even rows, so stratum A flags 15 of 30 and B none
        assert flag["weighted_bootstrap"].value == pytest.approx(15 / 30 * 120 / 150)
        assert flag["weighted_bootstrap"].low <= flag["weighted_bootstrap"].high

    def test_flag_rate_per_stratum(self):
        per = english().one_class["per_stratum"]
        assert (per["A"]["k"], per["A"]["n"], per["A"]["pool"]) == (15, 30, 120)
        assert (per["B"]["k"], per["B"]["n"]) == (0, 30)

    def test_flag_rate_on_reporting_rows_against_the_rest_with_a_difference(self):
        reporting = english().one_class["reporting"]
        assert reporting["n_negatives_reporting"] == 12 and reporting["n_negatives_other"] == 48
        rates = reporting["unweighted"]
        # rows 0, 10, 20, ... report: 0, 20, 40 are flagged (every fourth row) among 12 rows
        assert rates["reporting"].value == pytest.approx(3 / 12)
        assert rates["other"].value == pytest.approx(12 / 48)
        assert rates["difference"].value == pytest.approx(rates["reporting"].value - 0.25)
        assert reporting["wilson"]["reporting"] == {
            "flagged": 3,
            "n": 12,
            "ci_low": pytest.approx(wt.wilson_interval(3, 12)[0]),
            "ci_high": pytest.approx(wt.wilson_interval(3, 12)[1]),
        }
        assert reporting["weighted"]["difference"] is not None

    def test_ece_equals_the_mean_probability_and_the_output_says_so(self):
        out = english()
        cal = out.one_class["calibration"]
        mean = 0.25 * 0.8 + 0.75 * 0.2
        assert cal["mean_probability"].value == pytest.approx(mean)
        assert cal["ece"].value == pytest.approx(mean)
        assert cal["ece_point"] == pytest.approx(mean)
        assert "equals the mean predicted probability" in "\n".join(out.render())

    def test_render_lists_each_measure_and_the_refusal(self):
        text = "\n".join(english().render())
        for part in (
            "REFUSED",
            "observed positive rate 0 of 60",
            "flag rate at threshold 0.5 (misinfo.decision_threshold)",
            "flag rate per stratum",
            "difference, reporting minus other",
            "mean predicted probability",
        ):
            assert part in text, part

    def test_without_a_design_file_the_weighted_and_stratum_figures_are_not_measured(self):
        out = english(with_design=False)
        assert out.one_class["flag"]["weighted_bootstrap"] is None
        assert "per_stratum" not in out.one_class
        text = "\n".join(out.render())
        assert "not measured, the sheet has no design file" in text

    def test_model_labels_carry_the_note_on_the_first_and_last_line(self):
        lines = english(kind="model-unreviewed").render()
        assert NOTE in lines[1] and NOTE in lines[-1]
        assert NOTE not in "\n".join(english().render())

    def test_a_sample_of_positives_reports_the_rate_and_calibration_only(self):
        out = english(positives=60)
        report = out.one_class
        assert report["class_present"] == "positive"
        assert report["positive_rate"]["value"] == 1.0
        assert "recall" in report["flag"]["not_measured"]
        assert report["calibration"]["ece"].value == pytest.approx(1 - (0.25 * 0.8 + 0.75 * 0.2))

    def test_both_classes_still_compute_and_report_the_reporting_difference(self):
        out = english(positives=30)
        assert out.computed and out.one_class is None
        assert out.reporting["unweighted"]["difference"] is not None
        assert out.reporting["weighted"]["difference"] is not None
        assert "difference, reporting minus other" in "\n".join(out.render())


class TestLayaNegativeOnly:
    def test_laya_reports_the_flag_rate_at_a_given_threshold(self):
        out = hindi()
        assert not out.computed and oc.ONE_CLASS in out.refusal
        assert out.one_class["flag"]["threshold_source"] == "laya.decision_threshold"
        assert out.one_class["flag"]["unweighted_wilson"]["k"] == 15
        assert "flag rate at threshold 0.5" in "\n".join(out.render())

    def test_a_null_threshold_is_never_invented(self):
        out = hindi(threshold=None)
        assert out.one_class["flag"] == {"not_measured": oc.NO_THRESHOLD}
        assert "per_stratum" not in out.one_class and out.one_class["reporting"] is None
        text = "\n".join(out.render())
        assert "flag rate not measured, no threshold configured" in text
        assert out.one_class["calibration"]["ece"] is not None


class TestCommand:
    @pytest.fixture
    def sheets(self, env, monkeypatch):  # noqa: F811
        monkeypatch.setattr(
            "modeling.config.module_config",
            lambda module: {
                "version": VERSION,
                "decision_threshold": 0.5 if module == "misinfo" else None,
            },
        )
        monkeypatch.setattr("modeling.eval.corpus_transfer.MIN_LABELLED_ROWS", 20)
        monkeypatch.setattr("modeling.eval.laya_transfer.MIN_LABELLED_ROWS", 20)
        scores = []
        for lang, module in (("en", "misinfo"), ("hi", "laya")):
            sheet, score, design = build(lang=lang, module=module, kind="model-unreviewed")
            sheet["label_a"] = sheet["label_b"] = sheet["adjudicated_label"]
            sheet["reports_on_false_claim_a"] = sheet["adjudicated_reports_on_false_claim"]
            sheet["reports_on_false_claim_b"] = sheet["adjudicated_reports_on_false_claim"]
            for frame, col in ((sheet, "id"), (score, "record_id"), (design, "id")):
                frame[col] = [f"{lang}-{i}" for i in frame[col]]
            path = env.artifacts_dir / "hand_labels" / f"misinfo_corpus_sample_{lang}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            sheet.to_csv(path, index=False)
            design.to_csv(lb.design_path(path), index=False)
            scores.append(score)
        both = pd.concat(scores)
        monkeypatch.setattr("modeling.io.ScoredStore.__init__", lambda self, settings=None: None)
        monkeypatch.setattr("modeling.io.ScoredStore.read", lambda self, name: both)
        return env

    def args(self, *extra):
        return ["relabel-eval", "--allow-model-labels", "--n-bootstrap", "50", *extra]

    def test_exit_code_is_1_and_the_json_holds_the_figures(self, sheets):
        result = runner.invoke(modeling_cli.app, self.args("--laya-threshold", "0.5"))
        assert result.exit_code == 1, result.output
        out = sheets.artifacts_dir / "eval" / "labelling"
        for lang in ("en", "hi"):
            data = json.loads((out / lang / "transfer.json").read_text())
            assert data["computed"] is False and data["labeller"] == NOTE[len("labeller: ") :]
            assert data["seed"] == 20260813
            one = data["one_class"]
            assert one["reason_code"] == "one_class_only"
            assert one["flag"]["unweighted_wilson"]["k"] == 15
            assert one["reporting"]["unweighted"]["difference"]["value"] is not None
            assert one["positive_rate"]["value"] == 0.0
            text = (out / lang / "transfer.md").read_text()
            assert NOTE in text and "flag rate per stratum" in text

    def test_laya_without_a_threshold_says_so_in_the_markdown(self, sheets):
        result = runner.invoke(modeling_cli.app, self.args())
        assert result.exit_code == 1
        out = sheets.artifacts_dir / "eval" / "labelling"
        assert "no threshold configured" in (out / "hi" / "transfer.md").read_text()
        data = json.loads((out / "hi" / "transfer.json").read_text())
        assert data["one_class"]["flag"]["not_measured"] == "no threshold configured"
