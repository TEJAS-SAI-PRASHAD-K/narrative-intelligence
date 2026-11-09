"""The saved TF-IDF baseline: fit on the split, refuse non-English, refuse a flipped model."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from modeling.datasets.splits import group_train_val_test
from modeling.eval import tfidf_artifact as ta
from modeling.eval.baselines import tfidf_logreg

BAD = "shocking secret cure doctors hide miracle fake hoax"
GOOD = "council meeting budget report schedule published today"


def _frame(n: int = 120) -> pd.DataFrame:
    rows = []
    for i in range(n):
        bad = i % 2 == 0
        rows.append(
            {
                "text": f"{BAD if bad else GOOD} item{i % 7}",
                "label": int(bad),
                "author": f"a{i // 2}",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def fitted(tmp_path):
    frame, split = group_train_val_test(_frame(), group_col="author", label_col="label")
    path = tmp_path / "m.joblib"
    pipeline, meta = ta.fit_tfidf_on_split(frame, split, path=path)
    return frame, split, path, pipeline, meta


def test_fit_uses_only_training_rows_and_records_metadata(fitted):
    _, split, path, _, meta = fitted
    assert meta["n_train"] == len(split.train)
    assert meta["split_description"] == split.describe()
    assert meta["positive_label"] == 1 and meta["positive_class"] == "misinformation"
    assert meta["sklearn_version"]
    assert path.exists()


def test_round_trip_gives_identical_scores(fitted):
    frame, split, path, pipeline, _ = fitted
    test = frame.iloc[split.test]
    ids = [f"r{i}" for i in range(len(test))]
    langs = ["en"] * len(test)
    first = ta.score_tfidf(ta.load_tfidf(path), ids, test["text"].tolist(), langs)
    second = ta.score_tfidf(ta.load_tfidf(path), ids, test["text"].tolist(), langs)
    assert first.values == second.values
    direct = pipeline.predict_proba(test["text"].tolist())[:, 1]
    assert np.allclose([first.values[i] for i in ids], direct)


def test_scores_match_tfidf_logreg_on_same_rows(fitted):
    frame, split, path, _, _ = fitted
    tr, te = frame.iloc[split.train], frame.iloc[split.test]
    from modeling.config import get_settings

    result = tfidf_logreg(
        tr["text"].tolist(), tr["label"].to_numpy(), te["text"].tolist(),
        te["label"].to_numpy(), module="t", split_description="x", seed=get_settings().seed,
    )
    ids = [str(i) for i in range(len(te))]
    out = ta.score_tfidf(ta.load_tfidf(path), ids, te["text"].tolist(), ["en"] * len(te))
    assert np.allclose([out.values[i] for i in ids], result.scores)


def test_non_english_null_and_empty_are_skipped_with_reasons_exhaustively(fitted):
    _, _, path, _, _ = fitted
    ids = ["en1", "hi1", "none1", "nan1", "empty1", "blank_lang", "en2"]
    texts = [BAD, "यह एक झूठी खबर है", BAD, BAD, "   ", BAD, GOOD]
    langs = ["en", "hi", None, float("nan"), "en", "", "EN"]
    out = ta.score_tfidf(ta.load_tfidf(path), ids, texts, langs)
    assert set(out.values) | set(out.skipped) == set(ids)
    assert not set(out.values) & set(out.skipped)
    assert set(out.values) == {"en1", "en2"}
    assert out.skipped["hi1"] == "unsupported_language"
    assert out.skipped["none1"] == "unknown_language"
    assert out.skipped["nan1"] == "unknown_language"
    assert out.skipped["blank_lang"] == "unknown_language"
    assert out.skipped["empty1"] == "empty_text"
    assert out.tally()["unknown_language"] == 3
    assert out.values["en1"] > out.values["en2"]


def test_duplicate_ids_and_length_mismatch_raise(fitted):
    model = ta.load_tfidf(fitted[2])
    with pytest.raises(ValueError):
        ta.score_tfidf(model, ["a", "a"], [BAD, BAD], ["en", "en"])
    with pytest.raises(ValueError):
        ta.score_tfidf(model, ["a"], [BAD, BAD], ["en"])


def _rewrite(path, mutate):
    payload = joblib.load(path)
    mutate(payload)
    joblib.dump(payload, path)


def test_flipped_metadata_refuses_to_load(fitted):
    path = fitted[2]
    _rewrite(path, lambda p: p["metadata"].update(positive_label=0))
    with pytest.raises(ta.OrientationError):
        ta.load_tfidf(path)


def test_flipped_classes_refuses_to_load(fitted):
    path = fitted[2]

    def flip(payload):
        clf = payload["pipeline"].steps[-1][1]
        clf.classes_ = clf.classes_[::-1]

    _rewrite(path, flip)
    with pytest.raises(ta.OrientationError):
        ta.load_tfidf(path)


def test_missing_artifact_and_default_path(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        ta.load_tfidf(tmp_path / "nope.joblib")
    assert ta.default_artifact_path().parts[-2:] == ("baselines", ta.DEFAULT_ARTIFACT_NAME)


def test_single_class_training_labels_are_refused():
    frame = _frame()
    frame["label"] = 1
    frame, split = group_train_val_test(frame, group_col="author", dedupe=False)
    with pytest.raises(ValueError):
        ta.fit_tfidf_on_split(frame, split, save=False)
