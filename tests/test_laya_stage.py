"""The ``laya`` scoring stage and the calibrator it needs.

No weights, no network. The scorer is a subclass of ``LayaScorer`` whose agent is a fake, so
the routing, the checkpoint check and the version stamp are the real code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeling import config as C
from modeling import scoring as S
from modeling.checkpoint_identity import directory_identity
from modeling.eval.calibrate import Calibrator
from modeling.io import ScoredStore, _as_version_dict
from modeling.text import laya_calibration as LC
from modeling.text.laya_clf import LayaAvailability, LayaScorer, LayaSkipReason

HINDI = "यह एक दावा है जो बीस अक्षरों से काफी लंबा है"
HINDI_2 = "दूसरा दावा जो काफी लंबा है और जांच के लायक है"
ENGLISH = "an english claim that is long enough to be scored by someone"
BODY = "लंबा लेख का मुख्य भाग जो शीर्षक से बिल्कुल अलग है और सैकड़ों शब्दों का हो सकता है"
RAW = 0.99
LAYA_VERSION = C.module_config("laya")["version"]


def synthetic_validation(n: int = 400, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Over-confident scores: high everywhere, but only loosely tied to the label."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.6, 1.0, n)
    labels = (rng.uniform(size=n) < 0.1 + 2.0 * (scores - 0.6)).astype(int)
    return scores, labels


class FakeAgent:
    def __init__(self, raw):
        self.raw = raw

    def _result(self, questions):
        return {"answers": {next(iter(questions)): {"noul": self.raw}}}

    def predict(self, state, questions, **kwargs):
        return self._result(questions)

    predict_long = predict

    def predict_batch(self, states, questions, **kwargs):
        return [self._result(questions) for _ in states]


class FakeScorer(LayaScorer):
    installed = True
    raw = RAW
    scored_texts: list[str] = []

    def available(self):
        if not type(self).installed:
            return LayaAvailability(ok=False, reason="laya is not installed (fake)")
        return LayaAvailability(ok=True, laya_version="0.0")

    def agent_for(self, lang):
        outer = type(self)

        class Agent(FakeAgent):
            def predict_batch(self, states, questions, **kwargs):
                outer.scored_texts.extend(states)
                return super().predict_batch(states, questions, **kwargs)

        return Agent(outer.raw)


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("LAYA_HI_CHECKPOINT", str(tmp_path / "models" / f"multi-{LAYA_VERSION}"))
    monkeypatch.delenv("SCORE_UNKNOWN_LANGUAGE", raising=False)
    C.get_settings.cache_clear()
    yield C.get_settings()
    C.get_settings.cache_clear()


def install_calibrator(directory: Path, seed: int = 7, *, demo: bool = False) -> None:
    scores, labels = synthetic_validation(seed=seed)
    calibrator, report = LC.fit_calibrator(scores, labels)
    identity = directory_identity(directory, ignore=LC.CALIBRATOR_FILES)
    LC.write_calibrator(directory, calibrator, report, demo=demo, checkpoint_identity=identity)


def rewrite_state(directory: Path, **changes) -> dict:
    """Edit the calibrator file in place. ``None`` deletes a key."""
    path = directory / LC.CALIBRATOR_FILE
    state = json.loads(path.read_text())
    for key, value in changes.items():
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
    path.write_text(json.dumps(state))
    return state


@pytest.fixture
def checkpoint(tmp_path, settings):
    directory = tmp_path / "models" / f"multi-{LAYA_VERSION}"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text('{"model_type": "fake"}')
    (directory / "model.safetensors").write_bytes(b"weights")
    install_calibrator(directory)
    return directory


@pytest.fixture(autouse=True)
def fake_scorer(monkeypatch):
    import modeling.text.laya_clf as clf

    FakeScorer.installed = True
    FakeScorer.raw = RAW
    FakeScorer.scored_texts = []
    monkeypatch.setattr(clf, "LayaScorer", FakeScorer)
    return FakeScorer


def record(rid, title, lang, text=BODY):
    raw = json.dumps({"title": title}) if title is not None else json.dumps({"url": "x"})
    return {"id": rid, "source": "news", "text": text, "lang": lang, "raw": raw}


def records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            record("hi1", HINDI, "hi"),
            record("hi2", HINDI_2, "hi"),
            record("hi_short", "छोटा", "hi"),
            record("hi_notitle", None, "hi"),
            record("en1", ENGLISH, "en"),
            record("unk", HINDI, None),
        ]
    )


def make_ctx(tmp_path, settings, frame=None, store=None, dry_run=False):
    frame = records() if frame is None else frame
    store = store or ScoredStore(settings, root=tmp_path / "scored")
    return S.StageContext(
        settings=settings,
        reader=None,
        store=store,
        records=frame,
        authors=pd.DataFrame(),
        demo=True,
        dry_run=dry_run,
        known_record_ids=set(frame["id"]),
    )


def table(ctx) -> dict[str, dict]:
    frame = ctx.store.read("record_scores")
    return {row["record_id"]: row for row in frame.to_dict(orient="records")}


def seed(ctx, rows):
    now = pd.Timestamp.now(tz="UTC")
    ctx.store.write(
        "record_scores", pd.DataFrame([{**r, "scored_at": now} for r in rows]),
        known_keys=ctx.known_record_ids,
    )


def prob_of(row):
    value = row.get("misinfo_prob")
    return None if value is None or value != value else float(value)


def reasons_of(row):
    value = row.get("skip_reasons")
    return [str(r) for r in (value if value is not None else [])]


# ---------------------------------------------------------------------------
# what the stage writes
# ---------------------------------------------------------------------------
def test_hindi_is_written_with_the_calibrated_value_and_not_the_raw_one(
    tmp_path, settings, checkpoint
):
    ctx = make_ctx(tmp_path, settings)
    S._laya_stage(ctx)
    rows = table(ctx)
    loaded = LC.load_calibrator(checkpoint)
    expected = float(loaded.calibrator.transform([RAW])[0])
    assert prob_of(rows["hi1"]) == pytest.approx(expected, abs=1e-6)
    assert abs(expected - RAW) > 0.05, "the fixture must separate the calibrated and raw values"
    assert prob_of(rows["hi1"]) != pytest.approx(RAW, abs=1e-3)
    assert prob_of(rows["hi2"]) == pytest.approx(expected, abs=1e-6)


def test_english_and_unknown_language_are_never_written(tmp_path, settings, checkpoint):
    ctx = make_ctx(tmp_path, settings)
    S._laya_stage(ctx)
    rows = table(ctx)
    assert "en1" not in rows
    assert "unk" not in rows
    assert ENGLISH not in FakeScorer.scored_texts


def test_the_title_reaches_the_scorer_and_the_body_never_does(tmp_path, settings, checkpoint):
    S._laya_stage(make_ctx(tmp_path, settings))
    assert sorted(FakeScorer.scored_texts) == sorted([HINDI, HINDI_2])
    assert BODY not in FakeScorer.scored_texts


def test_a_record_with_no_title_gets_no_value_and_the_reason(tmp_path, settings, checkpoint):
    ctx = make_ctx(tmp_path, settings)
    out = S._laya_stage(ctx)
    row = table(ctx)["hi_notitle"]
    assert prob_of(row) is None
    assert reasons_of(row) == ["laya:no_title"]
    assert "laya" not in _as_version_dict(row["model_versions"])
    assert "1 no_title" in out[0]["note"]


def test_the_title_is_read_the_way_the_comparison_reads_it():
    import importlib.util
    import sys

    from modeling.titles import title_of

    script = Path(__file__).resolve().parent.parent / "scripts" / "run_comparison.py"
    spec = importlib.util.spec_from_file_location("run_comparison_for_titles", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_comparison_for_titles"] = module
    spec.loader.exec_module(module)
    assert module.title_of is title_of
    assert title_of('{"title": "  a title "}') == "a title"
    assert title_of({"title": "   "}) is None
    assert title_of("not json") is None
    assert title_of(None) is None


def test_the_stamp_names_the_checkpoint_and_the_calibrator(tmp_path, settings, checkpoint):
    ctx = make_ctx(tmp_path, settings)
    S._laya_stage(ctx)
    stamp = _as_version_dict(table(ctx)["hi1"]["model_versions"])["laya"]
    digest = LC.load_calibrator(checkpoint).digest
    assert stamp == f"{LAYA_VERSION}+finetuned:multi-{LAYA_VERSION}+isotonic@{digest}"


def test_an_existing_misinfo_prob_is_not_overwritten(tmp_path, settings, checkpoint):
    ctx = make_ctx(tmp_path, settings)
    seed(
        ctx,
        [
            {"record_id": "hi1", "source": "news", "misinfo_prob": 0.123,
             "model_versions": {"misinfo": "v0.2.0+abcd1234"}},
            {"record_id": "hi2", "source": "news", "misinfo_prob": 0.456, "model_versions": {}},
        ],
    )
    S._laya_stage(ctx)
    rows = table(ctx)
    assert prob_of(rows["hi1"]) == pytest.approx(0.123, abs=1e-6)
    assert prob_of(rows["hi2"]) == pytest.approx(0.456, abs=1e-6)
    assert "laya" not in _as_version_dict(rows["hi1"]["model_versions"])
    assert "laya" not in _as_version_dict(rows["hi2"]["model_versions"])
    assert HINDI not in FakeScorer.scored_texts and HINDI_2 not in FakeScorer.scored_texts


def test_a_record_that_misinfo_now_owns_loses_the_laya_stamp(tmp_path, settings, checkpoint):
    ctx = make_ctx(tmp_path, settings)
    S._laya_stage(ctx)
    ctx.store.write(
        "record_scores",
        pd.DataFrame(
            [{"record_id": "hi1", "source": "news", "misinfo_prob": 0.2,
              "model_versions": {"misinfo": "v0.2.0+abcd1234"}}]
        ),
        known_keys=ctx.known_record_ids,
    )
    S._laya_stage(ctx)
    row = table(ctx)["hi1"]
    assert prob_of(row) == pytest.approx(0.2, abs=1e-6)
    assert _as_version_dict(row["model_versions"]) == {"misinfo": "v0.2.0+abcd1234"}


def test_every_skip_has_a_laya_reason_and_no_value(tmp_path, settings, checkpoint):
    ctx = make_ctx(tmp_path, settings)
    S._laya_stage(ctx)
    row = table(ctx)["hi_short"]
    assert prob_of(row) is None
    assert reasons_of(row) == ["laya:text_too_short"]
    assert "laya" not in _as_version_dict(row["model_versions"])


def test_an_invalid_raw_score_is_skipped_not_written(tmp_path, settings, checkpoint):
    FakeScorer.raw = 1.7
    ctx = make_ctx(tmp_path, settings)
    S._laya_stage(ctx)
    row = table(ctx)["hi1"]
    assert prob_of(row) is None
    assert reasons_of(row) == [f"laya:{LayaSkipReason.INVALID_SCORE}"]


def test_other_reasons_on_the_row_are_kept(tmp_path, settings, checkpoint):
    ctx = make_ctx(tmp_path, settings)
    seed(ctx, [{"record_id": "hi1", "source": "news",
                "skip_reasons": ["misinfo:unsupported_language", "laya:stale"]}])
    S._laya_stage(ctx)
    assert reasons_of(table(ctx)["hi1"]) == ["misinfo:unsupported_language"]


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------
def test_a_second_run_scores_nothing_and_writes_nothing(tmp_path, settings, checkpoint):
    ctx = make_ctx(tmp_path, settings)
    S._laya_stage(ctx)
    FakeScorer.scored_texts = []
    before = ctx.store.read("record_scores")
    out = S._laya_stage(ctx)
    assert FakeScorer.scored_texts == []
    assert out[0]["written"] == 0 and out[0]["updated"] == 0
    pd.testing.assert_frame_equal(before, ctx.store.read("record_scores"))


def test_a_new_calibrator_changes_the_stamp_and_scores_again(tmp_path, settings, checkpoint):
    ctx = make_ctx(tmp_path, settings)
    S._laya_stage(ctx)
    first = _as_version_dict(table(ctx)["hi1"]["model_versions"])["laya"]
    install_calibrator(checkpoint, seed=99)
    FakeScorer.scored_texts = []
    S._laya_stage(ctx)
    assert HINDI in FakeScorer.scored_texts
    assert _as_version_dict(table(ctx)["hi1"]["model_versions"])["laya"] != first


def test_a_record_that_becomes_a_skip_loses_its_value_and_stamp(tmp_path, settings, checkpoint):
    ctx = make_ctx(tmp_path, settings)
    S._laya_stage(ctx)
    assert prob_of(table(ctx)["hi1"]) is not None
    ctx.records.loc[ctx.records["id"] == "hi1", "raw"] = json.dumps({"title": "छोटा"})
    S._laya_stage(ctx)
    row = table(ctx)["hi1"]
    assert prob_of(row) is None
    assert "laya" not in _as_version_dict(row["model_versions"])
    assert reasons_of(row) == ["laya:text_too_short"]
    assert ScoredStore.already_scored(
        ctx.store, "record_scores", {"laya": "anything"}
    ) == set()


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def _refusal_case(tmp_path, settings, ctx, code):
    out = S._laya_stage(ctx)
    rows = table(ctx)
    assert out[0]["refused"] == {"hi": code}
    assert "REFUSED" in out[0]["note"] and f"laya:{code}" in out[0]["note"]
    for rid in ("hi1", "hi2", "hi_short", "hi_notitle"):
        assert prob_of(rows[rid]) is None
        assert reasons_of(rows[rid]) == [f"laya:{code}"]
        assert "laya" not in _as_version_dict(rows[rid]["model_versions"])
    assert "en1" not in rows
    assert FakeScorer.scored_texts == []


def test_refused_when_laya_is_not_installed(tmp_path, settings, checkpoint, caplog):
    FakeScorer.installed = False
    ctx = make_ctx(tmp_path, settings)
    _refusal_case(tmp_path, settings, ctx, "model_unavailable")
    assert "not installed" in caplog.text


def test_refused_when_the_checkpoint_directory_is_missing(tmp_path, settings, checkpoint):
    for item in checkpoint.iterdir():
        item.unlink()
    checkpoint.rmdir()
    _refusal_case(tmp_path, settings, make_ctx(tmp_path, settings), "checkpoint_missing")


def test_refused_when_no_checkpoint_is_set_and_the_base_is_not_used(
    tmp_path, settings, checkpoint, monkeypatch
):
    monkeypatch.delenv("LAYA_HI_CHECKPOINT")
    C.get_settings.cache_clear()
    new_settings = C.get_settings()
    _refusal_case(tmp_path, new_settings, make_ctx(tmp_path, new_settings), "checkpoint_missing")


def test_refused_when_the_calibrator_is_missing(tmp_path, settings, checkpoint):
    (checkpoint / LC.CALIBRATOR_FILE).unlink()
    _refusal_case(tmp_path, settings, make_ctx(tmp_path, settings), "calibrator_missing")


@pytest.mark.parametrize(
    "content",
    ["not json", "[1, 2]", '{"method": "isotonic", "fitted": true}', ""],
)
def test_refused_when_the_calibrator_is_unreadable(tmp_path, settings, checkpoint, content):
    (checkpoint / LC.CALIBRATOR_FILE).write_text(content, encoding="utf-8")
    _refusal_case(tmp_path, settings, make_ctx(tmp_path, settings), "calibrator_unreadable")


@pytest.mark.parametrize(
    "changes",
    [
        {"fitted": "yes"},
        {"fitted": 1},
        {"demo": "no"},
        {"demo": None},
        {"checkpoint_identity": None},
        {"fit_range": None},
        {"fit_range": [1.0, 0.5]},
        {"fit_range": ["a", "b"]},
    ],
)
def test_refused_when_a_calibrator_field_is_not_strict(tmp_path, settings, checkpoint, changes):
    rewrite_state(checkpoint, **changes)
    _refusal_case(tmp_path, settings, make_ctx(tmp_path, settings), "calibrator_unreadable")


@pytest.mark.parametrize(
    "changes",
    [
        {"method": "none", "fitted": True},
        {"method": "isotonic", "fitted": False},
        {"method": "platt", "fitted": True, "coef": [[-3.0]], "intercept": [1.0]},
        # a constant isotonic map and a Platt map with a zero coefficient
        {"method": "isotonic", "fitted": True, "x": [0.6, 1.0], "y": [0.5, 0.5]},
        {"method": "platt", "fitted": True, "coef": [[0.0]], "intercept": [0.3]},
        # a near-constant map, below the minimum of distinct values at four decimals
        {"method": "isotonic", "fitted": True, "x": [0.6, 1.0], "y": [0.5, 0.50004]},
    ],
)
def test_refused_when_the_calibrator_is_rejected(tmp_path, settings, checkpoint, changes):
    rewrite_state(checkpoint, **changes)
    _refusal_case(tmp_path, settings, make_ctx(tmp_path, settings), "calibrator_rejected")


def test_refused_when_the_calibrator_is_marked_demo(tmp_path, settings, checkpoint):
    rewrite_state(checkpoint, demo=True)
    _refusal_case(tmp_path, settings, make_ctx(tmp_path, settings), "calibrator_demo")


@pytest.mark.parametrize("change", ["weights", "new_file", "config"])
def test_refused_when_the_weights_differ_from_the_ones_fitted(
    tmp_path, settings, checkpoint, change
):
    if change == "weights":
        (checkpoint / "model.safetensors").write_bytes(b"other weights")
    elif change == "new_file":
        (checkpoint / "extra.bin").write_bytes(b"x")
    else:
        (checkpoint / "config.json").write_text('{"model_type": "other"}')
    _refusal_case(
        tmp_path, settings, make_ctx(tmp_path, settings), "calibrator_checkpoint_mismatch"
    )


def test_the_calibrator_files_do_not_change_the_identity(checkpoint):
    before = directory_identity(checkpoint, ignore=LC.CALIBRATOR_FILES)
    install_calibrator(checkpoint, seed=3)
    assert directory_identity(checkpoint, ignore=LC.CALIBRATOR_FILES) == before
    assert directory_identity(checkpoint) != before


def test_the_stamp_follows_the_identity_of_the_weights(tmp_path, settings, checkpoint):
    first = LC.load_calibrator(checkpoint).digest
    (checkpoint / "model.safetensors").write_bytes(b"other weights")
    identity = directory_identity(checkpoint, ignore=LC.CALIBRATOR_FILES)
    rewrite_state(checkpoint, checkpoint_identity=identity)
    assert LC.load_calibrator(checkpoint).digest != first


def test_a_refusal_clears_values_from_an_earlier_good_run(tmp_path, settings, checkpoint):
    ctx = make_ctx(tmp_path, settings)
    S._laya_stage(ctx)
    assert prob_of(table(ctx)["hi1"]) is not None
    FakeScorer.scored_texts = []
    (checkpoint / LC.CALIBRATOR_FILE).unlink()
    _refusal_case(tmp_path, settings, ctx, "calibrator_missing")


def test_a_refusal_never_touches_a_misinfo_value(tmp_path, settings, checkpoint):
    (checkpoint / LC.CALIBRATOR_FILE).unlink()
    ctx = make_ctx(tmp_path, settings)
    seed(ctx, [{"record_id": "hi1", "source": "news", "misinfo_prob": 0.3,
                "model_versions": {"misinfo": "v0.2.0+abcd1234"}}])
    S._laya_stage(ctx)
    row = table(ctx)["hi1"]
    assert prob_of(row) == pytest.approx(0.3, abs=1e-6)
    assert reasons_of(row) == []


def test_english_in_scoring_languages_is_refused(tmp_path, settings, checkpoint, monkeypatch):
    original = C.module_config

    def with_english(module):
        config = original(module)
        if module == "laya":
            config["scoring_languages"] = ["hi", "en"]
        return config

    monkeypatch.setattr(C, "module_config", with_english)
    ctx = make_ctx(tmp_path, settings)
    out = S._laya_stage(ctx)
    assert out[0]["refused"] == "scoring_languages_invalid"
    assert len(ctx.store.read("record_scores")) == 0


def test_dry_run_writes_nothing(tmp_path, settings, checkpoint):
    ctx = make_ctx(tmp_path, settings, dry_run=True)
    out = S._laya_stage(ctx)
    assert "dry-run" in out[0]["note"]
    assert len(ctx.store.read("record_scores")) == 0


def test_the_stage_runs_after_misinfo_in_the_config():
    stages = C.scoring_config()["stages"]
    assert stages.index("misinfo") < stages.index("laya") < stages.index("cluster")
    assert C.module_config("laya")["scoring_languages"] == ["hi"]


def test_the_aux_rerun_keeps_laya_skip_reasons():
    frame = pd.DataFrame({"record_id": ["a"], "skip_reasons": [["aux_reason"]]})
    existing = pd.DataFrame({"record_id": ["a"], "skip_reasons": [["laya:text_too_short"]]})
    out = S._keep_misinfo_skips(frame, existing)
    assert list(out["skip_reasons"].iloc[0]) == ["aux_reason", "laya:text_too_short"]


# ---------------------------------------------------------------------------
# the calibrator
# ---------------------------------------------------------------------------
def test_fit_reports_before_and_after_and_the_label_statement():
    scores, labels = synthetic_validation()
    calibrator, report = LC.fit_calibrator(scores, labels)
    assert report["rows"] == 400
    assert report["brier_after"] < report["brier_before"]
    assert report["ece_after"] < report["ece_before"]
    assert "Hindi" in report["label_statement"] and "English" in report["label_statement"]
    assert calibrator.fitted


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_state_to_file_to_load_gives_identical_transforms(tmp_path, method):
    scores, labels = synthetic_validation()
    calibrator, report = LC.fit_calibrator(scores, labels, method)
    (tmp_path / "config.json").write_text("{}")
    identity = directory_identity(tmp_path, ignore=LC.CALIBRATOR_FILES)
    LC.write_calibrator(tmp_path, calibrator, report, demo=False, checkpoint_identity=identity)
    loaded = LC.load_calibrator(tmp_path)
    grid = np.concatenate([np.linspace(0, 1, 201), scores])
    assert np.array_equal(calibrator.transform(grid), loaded.calibrator.transform(grid))
    assert loaded.method == method
    assert json.loads((tmp_path / LC.REPORT_FILE).read_text())["method"] == method


def test_a_rejected_fit_raises_and_a_one_class_fit_raises():
    scores = np.linspace(0.5, 1.0, 300)
    # Labels fall as the score rises, so the fit has to be a decreasing map.
    labels = (scores < 0.75).astype(int)
    with pytest.raises(LC.LayaCalibratorError) as info:
        LC.fit_calibrator(scores, labels, "platt")
    assert info.value.reason == "calibrator_rejected"
    with pytest.raises(LC.LayaCalibratorError):
        LC.fit_calibrator(scores, np.ones(300, dtype=int))


def test_inverted_scores_are_refused_by_the_auc_floor():
    scores, labels = synthetic_validation()
    with pytest.raises(LC.LayaCalibratorError, match="ROC-AUC"):
        LC.fit_calibrator(1.0 - scores, labels)


def test_uninformative_scores_are_refused_by_the_auc_floor():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.6, 1.0, 600)
    labels = rng.integers(0, 2, 600)
    with pytest.raises(LC.LayaCalibratorError, match="ROC-AUC") as info:
        LC.fit_calibrator(scores, labels)
    assert info.value.reason == "calibrator_rejected"


def test_the_fit_report_holds_the_auc_the_range_and_the_distinct_count():
    scores, labels = synthetic_validation()
    _, report = LC.fit_calibrator(scores, labels)
    assert report["raw_roc_auc"] > LC.MIN_FIT_AUC
    assert report["distinct_outputs"] >= LC.MIN_DISTINCT_OUTPUTS
    assert report["fit_range"] == [float(scores.min()), float(scores.max())]


def test_calibrator_state_round_trips_through_the_class():
    scores, labels = synthetic_validation()
    calibrator, _ = LC.fit_calibrator(scores, labels)
    again = Calibrator.from_state(json.loads(json.dumps(calibrator.state())))
    assert np.array_equal(calibrator.transform(scores), again.transform(scores))


# ---------------------------------------------------------------------------
# the scored loader reads what the stage writes
# ---------------------------------------------------------------------------
def test_the_scored_loader_reads_a_laya_row_without_a_misinfo_stamp(
    tmp_path, settings, checkpoint
):
    from app.etl.scored_loader import _read, _version_of

    ctx = make_ctx(tmp_path, settings)
    S._laya_stage(ctx)
    rows = {r["record_id"]: r for r in _read(tmp_path / "scored", "record_scores")}
    hindi = rows["hi1"]
    versions = dict(hindi["model_versions"])
    assert set(versions) == {"laya"}
    assert hindi["misinfo_prob"] is not None
    assert _version_of(hindi["model_versions"]).startswith("phase2:")
    # A skipped row carries no stamp, so its version digest is empty.
    assert _version_of(rows["hi_short"]["model_versions"]) is None


# ---------------------------------------------------------------------------
# collapse detection
# ---------------------------------------------------------------------------
def test_the_summary_counts_out_of_range_values_and_distinct_outputs(
    tmp_path, settings, checkpoint
):
    FakeScorer.raw = 0.99
    out = S._laya_stage(make_ctx(tmp_path, settings))
    info = out[0]["calibration"]["hi"]
    assert info["scored"] == 2 and info["out_of_range"] == 0
    assert info["distinct_calibrated"] == 1
    # Two titles gave one raw value, so one calibrated value: below the minimum.
    assert info["collapse_warning"] is True
    assert "COLLAPSE WARNING" in out[0]["note"]


def test_a_collapse_is_a_warning_and_not_a_refusal(tmp_path, settings, checkpoint, caplog):
    FakeScorer.raw = 0.99
    ctx = make_ctx(tmp_path, settings)
    S._laya_stage(ctx)
    assert "COLLAPSE WARNING" in caplog.text
    assert prob_of(table(ctx)["hi1"]) is not None


def test_more_than_half_out_of_range_warns(tmp_path, settings, checkpoint, caplog):
    FakeScorer.raw = 0.05  # the fit range starts near 0.6
    out = S._laya_stage(make_ctx(tmp_path, settings))
    info = out[0]["calibration"]["hi"]
    assert info["out_of_range"] == 2 and info["collapse_warning"] is True


def test_no_warning_when_the_values_are_spread(tmp_path, settings, checkpoint, monkeypatch):
    class Varied(FakeScorer):
        def agent_for(self, lang):
            class Agent(FakeAgent):
                def predict_batch(self, states, questions, **kwargs):
                    return [
                        {"answers": {next(iter(questions)): {"noul": 0.6 + 0.4 * i / 20}}}
                        for i, _ in enumerate(states)
                    ]

            return Agent(0.0)

    many = pd.DataFrame([record(f"h{i}", f"{HINDI} {i}", "hi") for i in range(21)])
    import modeling.text.laya_clf as clf

    monkeypatch.setattr(clf, "LayaScorer", Varied)
    out = S._laya_stage(make_ctx(tmp_path, settings, frame=many))
    info = out[0]["calibration"]["hi"]
    assert info["collapse_warning"] is False
    assert info["distinct_calibrated"] >= LC.MIN_DISTINCT_OUTPUTS
