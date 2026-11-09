"""Input schema validation for the fact-check evaluation harness."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from modeling.eval.factcheck_eval import (
    EvalInputError,
    InputsMismatch,
    ScoreFileError,
    SplitMismatch,
    check_shared_inputs,
    check_shared_test_split,
    load_score_dir,
    load_score_file,
    validate_frame,
)

COLUMNS = ["part", "id", "lang", "label", "score", "source_detail", "skip_reason"]


def row(part, id_, *, lang="en", label=None, score=None, source=None, skip=None):
    return {
        "part": part,
        "id": id_,
        "lang": lang,
        "label": label,
        "score": score,
        "source_detail": source,
        "skip_reason": skip,
    }


def good_rows():
    return [
        row("val", "v1", label=1, score=0.9),
        row("val", "v2", label=0, score=0.2),
        row("test", "t1", label=1, score=0.8),
        row("test", "t2", label=0, score=0.1),
        row("claims", "c1", score=0.7),
        row("corpus_title", "k1", score=0.3, source="example.com"),
    ]


def write(path, rows, schema=None):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    pq.write_table(table, path)
    return path


def test_a_good_file_loads_and_is_normalised(tmp_path):
    rows = good_rows() + [row("claims", "c2", lang="hi", skip="unsupported_language")]
    scores = load_score_file(write(tmp_path / "m.parquet", rows))
    assert scores.name == "m"
    assert len(scores.sha256) == 64
    claims = scores.part("claims").set_index("id")
    assert pd.isna(claims.loc["c2", "score"])
    assert claims.loc["c2", "skip_reason"] == "unsupported_language"
    assert pd.isna(claims.loc["c1", "skip_reason"])


def test_a_null_score_without_a_reason_gets_the_unspecified_code(tmp_path):
    rows = good_rows() + [row("claims", "c2")]
    scores = load_score_file(write(tmp_path / "m.parquet", rows))
    assert scores.part("claims").set_index("id").loc["c2", "skip_reason"] == "unspecified"


def test_missing_column_refuses(tmp_path):
    frame = pd.DataFrame(good_rows(), columns=COLUMNS).drop(columns=["skip_reason"])
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp_path / "m.parquet")
    with pytest.raises(ScoreFileError, match="missing column"):
        load_score_file(tmp_path / "m.parquet")


def test_score_stored_as_string_refuses(tmp_path):
    rows = [{**r, "score": None if r["score"] is None else str(r["score"])} for r in good_rows()]
    path = write(tmp_path / "m.parquet", rows)
    with pytest.raises(ScoreFileError, match="'score' must be a float"):
        load_score_file(path)


def test_not_a_parquet_file_refuses(tmp_path):
    (tmp_path / "m.parquet").write_text("not parquet")
    with pytest.raises(ScoreFileError, match="not a readable Parquet"):
        load_score_file(tmp_path / "m.parquet")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.append(row("elsewhere", "x", score=0.5)), "unknown part"),
        (lambda rows: rows.append(row("claims", "c1", score=0.5)), "share a"),
        (lambda rows: rows.append(row("claims", "c9", score=1.5)), r"\[0, 1\]"),
        (lambda rows: rows.append(row("claims", "c9", score=float("inf"))), "infinite"),
        (lambda rows: rows.append(row("claims", "c9", score=0.5, skip="x")), "both a score"),
        (lambda rows: rows.append(row("val", "v9", label=2, score=0.5)), "0 or 1"),
        (lambda rows: rows.append(row("val", "v9", label=None, score=0.5)), "null label"),
        (lambda rows: rows.append(row("claims", "c9", label=1, score=0.5)), "label must be null"),
        (lambda rows: rows.append(row("test", "v1", label=1, score=0.5)), "both val and test"),
        (lambda rows: rows.append(row("claims", "", score=0.5)), "null or empty id"),
        (lambda rows: rows.append(row(None, "z", score=0.5)), "null part"),
    ],
)
def test_malformed_values_refuse(mutation, message):
    rows = good_rows()
    mutation(rows)
    with pytest.raises(ScoreFileError, match=message):
        validate_frame(pd.DataFrame(rows, columns=COLUMNS), "m")


def test_a_file_without_val_or_test_rows_refuses():
    no_val = [r for r in good_rows() if r["part"] != "val"]
    with pytest.raises(ScoreFileError, match="no 'val' rows"):
        validate_frame(pd.DataFrame(no_val, columns=COLUMNS), "m")
    no_test = [r for r in good_rows() if r["part"] != "test"]
    with pytest.raises(ScoreFileError, match="no 'test' rows"):
        validate_frame(pd.DataFrame(no_test, columns=COLUMNS), "m")


def test_empty_file_refuses():
    with pytest.raises(ScoreFileError, match="no rows"):
        validate_frame(pd.DataFrame(columns=COLUMNS), "m")


def test_unknown_extra_columns_are_ignored_and_reported():
    frame = pd.DataFrame(good_rows(), columns=COLUMNS)
    frame["note"] = "x"
    scores = validate_frame(frame, "m")
    assert scores.ignored_columns == ["note"]
    assert "note" not in scores.frame.columns


def test_directory_with_no_parquet_refuses(tmp_path):
    with pytest.raises(EvalInputError, match="no <model>.parquet"):
        load_score_dir(tmp_path)


def test_different_test_ids_refuse():
    a = validate_frame(pd.DataFrame(good_rows(), columns=COLUMNS), "a")
    other = [r for r in good_rows() if r["id"] != "t2"]
    b = validate_frame(pd.DataFrame(other, columns=COLUMNS), "b")
    with pytest.raises(SplitMismatch, match="same test ids"):
        check_shared_test_split({"a": a, "b": b})


def test_same_test_ids_with_different_labels_refuse():
    a = validate_frame(pd.DataFrame(good_rows(), columns=COLUMNS), "a")
    flipped = [{**r, "label": 1 - r["label"]} if r["id"] == "t1" else r for r in good_rows()]
    b = validate_frame(pd.DataFrame(flipped, columns=COLUMNS), "b")
    with pytest.raises(SplitMismatch, match="different labels"):
        check_shared_test_split({"a": a, "b": b})


def test_identical_test_ids_pass_even_with_other_rows_different():
    a = validate_frame(pd.DataFrame(good_rows(), columns=COLUMNS), "a")
    extra = good_rows() + [row("claims", "c2", score=0.5)]
    b = validate_frame(pd.DataFrame(extra, columns=COLUMNS), "b")
    check_shared_test_split({"a": a, "b": b})


def inputs_rows():
    return good_rows() + [row("corpus_text", "x1", score=0.2, source="example.com")]


@pytest.mark.parametrize(
    "part, dropped", [("claims", "c1"), ("corpus_title", "k1"), ("corpus_text", "x1")]
)
def test_models_that_scored_different_claim_or_corpus_rows_refuse(part, dropped):
    a = validate_frame(pd.DataFrame(inputs_rows(), columns=COLUMNS), "a")
    kept = [r for r in inputs_rows() if not (r["part"] == part and r["id"] == dropped)]
    b = validate_frame(pd.DataFrame(kept, columns=COLUMNS), "b")
    with pytest.raises(InputsMismatch, match=f"same {part} rows"):
        check_shared_inputs({"a": a, "b": b})


def test_models_that_scored_other_corpus_ids_of_the_same_size_refuse():
    a = validate_frame(pd.DataFrame(inputs_rows(), columns=COLUMNS), "a")
    swapped = [{**r, "id": "k9"} if r["id"] == "k1" else r for r in inputs_rows()]
    b = validate_frame(pd.DataFrame(swapped, columns=COLUMNS), "b")
    with pytest.raises(InputsMismatch, match=r"only in b: \['k9'\]"):
        check_shared_inputs({"a": a, "b": b})


def test_models_with_the_same_claim_and_corpus_ids_pass():
    a = validate_frame(pd.DataFrame(inputs_rows(), columns=COLUMNS), "a")
    b = validate_frame(pd.DataFrame(reversed(inputs_rows()), columns=COLUMNS), "b")
    check_shared_inputs({"a": a, "b": b})


def test_evaluate_refuses_models_scored_on_different_corpus_rows():
    from modeling.eval.factcheck_eval import ClaimUniverse, evaluate

    a = validate_frame(pd.DataFrame(inputs_rows(), columns=COLUMNS), "tfidf")
    kept = [r for r in inputs_rows() if r["id"] != "k1"]
    b = validate_frame(pd.DataFrame(kept, columns=COLUMNS), "cand")
    universe = ClaimUniverse(eval_ids=frozenset({"c1"}), excluded={}, frozen=False)
    with pytest.raises(InputsMismatch):
        evaluate(
            {"tfidf": a, "cand": b},
            universe=universe,
            publishers=frozenset({"x.example"}),
            n_bootstrap=20,
            seed=1,
        )
