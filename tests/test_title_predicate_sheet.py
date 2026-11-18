"""The tier 4 label sheet, its key file and the scorer."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ingest.config import CONFIG_DIR, Settings
from ingest.factcheck import DropReason, FactCheckExtractor
from ingest.title_predicate_eval import (
    LABELLER_KINDS,
    MEASURED_KINDS,
    SHEET_COLUMNS,
    SheetError,
    allocate,
    check_sheet_path,
    default_key_path,
    draw_sample,
    join_sheet_and_key,
    read_key,
    read_sheet,
    score_rows,
    score_sheet,
    wilson,
    write_sheet,
)


def real_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "factcheck.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def extractor(tmp_path):
    return FactCheckExtractor(settings=Settings(data_dir=tmp_path), config=real_config())


EN_VERDICTS = [
    "Video Of Flooded Street Is AI-Generated",
    "Image Of Minister Is Digitally Altered",
    "Viral Letter About Holiday Is Fake",
    "Old Video Of Crowd Falsely Shared As Recent Rally",
    "Fact Check: No, This Video Is Not Fake",
]
HI_VERDICTS = ["वायरल पोस्ट फर्जी है", "यह वीडियो एआई जनरेटेड है", "वायरल तस्वीर एडिटेड है"]
NO_VERDICT = ["Budget Session Begins Monday", "Weekly Roundup Of Fact Checks", "Parliament Debate"]
HI_NO_VERDICT = ["आज का मौसम: कई राज्यों में बारिश", "बजट सत्र सोमवार से शुरू"]


def record(n: int, title: str, domain: str = "altnews.in") -> dict:
    return {
        "id": f"news:{domain}/{n}",
        "url": f"https://{domain}/{n}",
        "source_detail": domain,
        "title": title,
        "text": "",
        "tags": [],
    }


def population(n_en=300, n_hi=80):
    rows = []
    for i in range(n_en):
        pool = EN_VERDICTS if i % 5 == 0 else NO_VERDICT
        rows.append(record(i, f"{pool[i % len(pool)]} number {i}", "altnews.in"))
    for i in range(n_hi):
        pool = HI_VERDICTS if i % 8 == 0 else HI_NO_VERDICT
        rows.append(record(1000 + i, f"{pool[i % len(pool)]} {i}", "altnews.in"))
    return rows


# --- allocation and the draw -------------------------------------------------


def test_allocation_gives_hindi_its_minimum_when_the_pool_allows():
    pools = {("en", True): 40, ("en", False): 900, ("hi", True): 10, ("hi", False): 60}
    quotas = allocate(pools, 100)
    assert sum(quotas.values()) == 100
    assert quotas[("hi", True)] + quotas[("hi", False)] >= 30
    assert quotas[("en", True)] <= 35


def test_allocation_takes_all_hindi_when_the_pool_is_short():
    pools = {("en", True): 40, ("en", False): 900, ("hi", True): 2, ("hi", False): 10}
    quotas = allocate(pools, 100)
    assert quotas[("hi", True)] == 2 and quotas[("hi", False)] == 10
    assert sum(quotas.values()) == 100


def test_allocation_spills_a_short_matched_pool_to_the_unmatched_one():
    quotas = allocate({("en", True): 3, ("en", False): 500, ("hi", False): 100}, 100)
    assert quotas[("en", True)] == 3
    assert sum(quotas.values()) == 100


def test_a_small_pool_is_taken_whole():
    assert allocate({("en", True): 3, ("en", False): 4}, 100) == {
        ("en", True): 3,
        ("en", False): 4,
    }


def test_the_draw_is_seeded_and_does_not_depend_on_input_order(extractor):
    pop = population()
    a = draw_sample(pop, extractor, n=100)
    b = draw_sample(list(reversed(pop)), extractor, n=100)
    c = draw_sample(pop, extractor, n=100, seed=1)
    assert len(a[0]) == 100
    assert a == b
    assert [r["record_id"] for r in a[0]] != [r["record_id"] for r in c[0]]


def test_the_draw_has_hindi_rows_and_unmatched_headlines(extractor):
    sheet, key = draw_sample(population(), extractor, n=100)
    rows = key["rows"]
    assert sum(r["lang"] == "hi" for r in rows) >= 30
    assert sum(r["predicted_verdict"] == "none" for r in rows) >= 40
    assert any(r["predicted_verdict"] in ("false", "true", "middle") for r in rows)
    assert len(sheet) == len(rows) == key["n_drawn"] == 100


def test_the_key_records_the_weights_and_the_pattern_hash(extractor):
    from ingest.title_predicate import pattern_set_hash

    _, key = draw_sample(population(), extractor, n=100)
    assert key["pattern_hash"] == pattern_set_hash(real_config())
    assert key["seed"] == 20260813
    for stratum in key["strata"].values():
        assert stratum["weight"] == pytest.approx(stratum["population"] / stratum["sampled"])
    # The weights add back up to the pool, so the figures describe the population.
    assert sum(r["weight"] for r in key["rows"]) == pytest.approx(380)
    for row in key["rows"]:
        assert row["weight"] == key["strata"][row["stratum"]]["weight"]


def test_the_sheet_is_blind(extractor):
    sheet, key = draw_sample(population(), extractor, n=20)
    assert tuple(sheet[0]) == SHEET_COLUMNS
    assert not [c for c in SHEET_COLUMNS if "predict" in c or "pattern" in c or "weight" in c]
    for needed in ("record_id", "headline", "url", "label_1", "label_2", "adjudicated_label",
                   "labeller_1", "labeller_2", "labeller_kind"):
        assert needed in SHEET_COLUMNS
    assert all(r["label_1"] == r["label_2"] == r["labeller_kind"] == "" for r in sheet)
    assert {"predicted_verdict", "predicted_pattern", "predicted_rating"} <= set(key["rows"][0])
    assert {r["predicted_verdict"] for r in key["rows"]} <= {"false", "true", "middle", "none"}


def test_the_sheet_never_shows_a_prediction_to_the_labellers(extractor, tmp_path):
    sheet, key = draw_sample(population(), extractor, n=20)
    path, _ = write_sheet(sheet, key, tmp_path / "s.csv")
    text = path.read_text(encoding="utf-8-sig")
    assert "predicted" not in text and "en_ai_generated" not in text


def test_the_prediction_is_the_extractors_on_invisible_characters(extractor):
    titles = [
        "Video Is Fa­ke",
        "Video Is AI‌-Generated",
        "वायरल पोस्ट फर्‌जी‍ है",
        "Photo Is Mor­phed",
        "Budget ​Session Begins",
    ]
    sheet, key = draw_sample([record(i, t) for i, t in enumerate(titles)], extractor, n=100)
    by_id = {r["record_id"]: r for r in key["rows"]}
    assert len(sheet) == len(titles)
    for row in sheet:
        raw = titles[int(row["record_id"].rsplit("/", 1)[1])]
        extracted = extractor.from_title_predicate(record(0, raw))
        predicted = by_id[row["record_id"]]["predicted_verdict"]
        # The sheet shows the text that the extractor read.
        assert "­" not in row["headline"] and "‌" not in row["headline"]
        assert "‍" not in row["headline"] and "​" not in row["headline"]
        if extracted is None:
            assert predicted == "none"
        else:
            assert predicted == extracted.label
            assert row["headline"] == extracted.claim


def test_the_prediction_in_the_key_matches_the_extractor_for_every_row(extractor):
    sheet, key = draw_sample(population(), extractor, n=100)
    by_id = {r["record_id"]: r for r in key["rows"]}
    for row in sheet:
        extracted = extractor.from_title_predicate(
            {"id": row["record_id"], "url": "u", "source_detail": "altnews.in",
             "title": row["headline"]}
        )
        predicted = by_id[row["record_id"]]["predicted_verdict"]
        if predicted in ("false", "true"):
            assert extracted is not None and extracted.label == predicted
        elif predicted == "none":
            assert extracted is None


def test_a_record_with_no_title_is_counted_not_drawn(extractor):
    pop = population(10, 0) + [record(99, "")]
    sheet, _ = draw_sample(pop, extractor, n=100)
    assert len(sheet) == 10
    assert extractor.dropped[DropReason.NO_CLAIM_TEXT.value] == 1


# --- paths and files ---------------------------------------------------------


def test_a_sheet_path_inside_the_repo_is_refused_unless_ignored(tmp_path):
    repo = Path(__file__).resolve().parent.parent
    with pytest.raises(SheetError, match="inside the repository"):
        check_sheet_path(repo / "configs" / "sheet.csv")
    assert check_sheet_path(repo / "data" / "factchecks" / "sheet.csv").name == "sheet.csv"
    assert check_sheet_path(tmp_path / "sheet.csv") == (tmp_path / "sheet.csv").resolve()


def test_the_key_path_is_checked_too(extractor, tmp_path):
    sheet, key = draw_sample(population(10, 0), extractor, n=5)
    repo = Path(__file__).resolve().parent.parent
    with pytest.raises(SheetError, match="inside the repository"):
        write_sheet(sheet, key, tmp_path / "s.csv", repo / "configs" / "key.json")
    assert not (tmp_path / "s.csv").exists()


def test_write_sheet_refuses_to_replace_a_file_and_writes_a_bom(extractor, tmp_path):
    sheet, key = draw_sample(population(10, 5), extractor, n=8)
    path, key_path = write_sheet(sheet, key, tmp_path / "sheet.csv")
    assert key_path == default_key_path(path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with pytest.raises(SheetError, match="exists"):
        write_sheet(sheet, key, path)
    write_sheet(sheet, key, path, overwrite=True)
    assert read_key(key_path)["n_drawn"] == len(sheet)


def test_a_sheet_with_a_bom_is_read_and_the_hindi_survives(extractor, tmp_path):
    sheet, key = draw_sample(population(10, 20), extractor, n=30)
    path, _ = write_sheet(sheet, key, tmp_path / "sheet.csv")
    text = path.read_text(encoding="utf-8-sig")
    assert any(re.search("[ऀ-ॿ]", line) for line in text.splitlines())
    filled = fill(path, sheet)
    assert read_sheet(filled)[0]["record_id"] == sheet[0]["record_id"]


def fill(path: Path, sheet: list[dict], **columns) -> Path:
    """Fill every label column of a written sheet, with the BOM kept."""
    rows = []
    for index, row in enumerate(sheet):
        rows.append(
            {
                **row,
                "label_1": "none",
                "label_2": "none",
                "labeller_1": "Asha",
                "labeller_2": "Ravi",
                "labeller_kind": "human",
                **{k: v(index) if callable(v) else v for k, v in columns.items()},
            }
        )
    out = path.with_name("filled.csv")
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SHEET_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return out


# --- scoring: statistics -----------------------------------------------------

KEY = {"seed": 1, "pattern_hash": "h", "strata": {}}


def mk(i, predicted, label_1, label_2=None, *, adjudicated="", lang="en",
       publisher="altnews.in", weight=1.0, kind="human", labellers=("asha", "ravi")):
    return {
        "record_id": f"r{i}",
        "url": "u",
        "headline": "h",
        "label_1": label_1,
        "label_2": label_2 or label_1,
        "adjudicated_label": adjudicated,
        "labeller_1": labellers[0],
        "labeller_2": labellers[1],
        "labeller_kind": kind,
        "predicted_verdict": predicted,
        "lang": lang,
        "publisher": publisher,
        "weight": weight,
    }


def score(rows, **kwargs):
    kwargs.setdefault("current_pattern_hash", "h")
    kwargs.setdefault("min_adjudicated_rows", 1)
    kwargs.setdefault("min_language_rows", 0)
    return score_rows(rows, KEY, **kwargs)


def hand_case(**kwargs):
    # predicted/label: 6 false-false, 2 false-none, 2 none-false, 10 none-none.
    spec = [("false", "false")] * 6 + [("false", "none")] * 2
    spec += [("none", "false")] * 2 + [("none", "none")] * 10
    return [mk(i, p, y, **kwargs) for i, (p, y) in enumerate(spec)]


def test_wilson_matches_the_textbook_value():
    low, high = wilson(0.5, 100)
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(0.5962, abs=1e-3)
    low, high = wilson(1.0, 10)
    assert high == pytest.approx(1.0) and low == pytest.approx(0.7225, abs=1e-3)


def test_scoring_matches_the_hand_computed_values_with_wilson_intervals():
    result = score(hand_case())
    false = result["metrics"]["false"]
    assert false["precision"]["value"] == pytest.approx(6 / 8)
    assert false["recall"]["value"] == pytest.approx(6 / 8)
    low, high = wilson(0.75, 8)
    assert false["precision"]["ci_low"] == pytest.approx(low)
    assert false["precision"]["ci_high"] == pytest.approx(high)
    assert false["precision"]["denominator_rows"] == 8
    assert result["metrics"]["verdict"]["precision"]["value"] == pytest.approx(0.75)
    assert result["metrics"]["true"]["recall"]["value"] is None
    assert result["metrics"]["true"]["recall"]["reason"]
    assert result["cohen_kappa"] == 1.0
    assert "bootstrap" not in json.dumps(result).lower()


def test_scoring_is_deterministic():
    assert score(hand_case()) == score(hand_case())


def test_weights_change_the_estimate_as_a_stratified_estimate_must():
    # Two true positives of weight 1 and one false positive of weight 3.
    rows = [
        mk(0, "false", "false"),
        mk(1, "false", "false"),
        mk(2, "false", "none", weight=3.0),
    ]
    precision = score(rows)["metrics"]["false"]["precision"]
    assert precision["value"] == pytest.approx(2 / 5)
    n_eff = 25 / (1 + 1 + 9)
    assert precision["n_effective"] == pytest.approx(n_eff)
    assert precision["ci_low"] == pytest.approx(wilson(2 / 5, n_eff)[0])


def test_recall_is_weighted_so_unmatched_headlines_count_at_their_rate():
    # One matched true positive at weight 1, one unmatched false headline at weight 9.
    rows = [mk(0, "false", "false"), mk(1, "none", "false", weight=9.0)]
    assert score(rows)["metrics"]["false"]["recall"]["value"] == pytest.approx(0.1)


def test_agreement_fills_the_adjudicated_label_and_a_disagreement_needs_one():
    rows = hand_case()
    rows[0] = mk(0, "false", "false", "none")
    result = score(rows)
    assert result["n_unresolved"] == 1 and result["unresolved_record_ids"] == ["r0"]
    assert result["is_measured_error_rate"] is False
    assert any("no adjudicated label" in r for r in result["not_measured_reasons"])
    rows[0] = mk(0, "false", "false", "none", adjudicated="false")
    result = score(rows)
    assert result["n_unresolved"] == 0 and result["n_adjudicated"] == 20
    assert result["metrics"]["false"]["precision"]["value"] == pytest.approx(6 / 8)
    assert result["cohen_kappa"] < 1.0


def test_the_worst_case_counts_every_unresolved_disagreement_as_an_error():
    rows = hand_case()
    rows[0] = mk(0, "false", "false", "none")  # predicted false, labellers split
    rows[19] = mk(19, "none", "false", "none")  # missed by the reader, labellers split
    result = score(rows)
    best = result["metrics"]["false"]
    worst = result["worst_case_metrics"]["false"]
    # Main figure leaves both rows out: 5 hits of 7 predicted, 5 of 7 labelled false.
    assert best["precision"]["value"] == pytest.approx(5 / 7)
    assert best["recall"]["value"] == pytest.approx(5 / 7)
    # Worst case: the split rows are in the denominators and are not hits.
    assert worst["precision"]["value"] == pytest.approx(5 / 8)
    assert worst["recall"]["value"] == pytest.approx(5 / 9)
    assert worst["precision"]["value"] < best["precision"]["value"]


def test_precision_and_recall_are_also_reported_against_each_labeller():
    rows = hand_case()
    rows[6] = mk(6, "false", "none", "false")  # labeller 2 says false
    result = score(rows)
    assert result["against_labeller_1"]["false"]["precision"]["value"] == pytest.approx(6 / 8)
    assert result["against_labeller_2"]["false"]["precision"]["value"] == pytest.approx(7 / 8)


def test_metrics_are_reported_per_language_and_per_publisher():
    rows = hand_case(lang="en", publisher="altnews.in")
    rows += [mk(100 + i, "false", "false", lang="hi", publisher="factly.in") for i in range(4)]
    result = score(rows)
    assert result["per_language"]["hi"]["n_rows"] == 4
    assert result["per_language"]["hi"]["metrics"]["false"]["precision"]["value"] == 1.0
    assert result["per_language"]["en"]["metrics"]["false"]["precision"]["value"] == (
        pytest.approx(0.75)
    )
    assert set(result["per_publisher"]) == {"altnews.in", "factly.in"}
    assert result["n_by_language"] == {"en": 20, "hi": 4}


def human_rows(n=100, n_hi=40, **kwargs):
    rows = []
    for i in range(n):
        predicted = "false" if i % 4 == 0 else "none"
        rows.append(mk(i, predicted, predicted, lang="hi" if i < n_hi else "en", **kwargs))
    return rows


def full_score(rows, **kwargs):
    return score_rows(
        rows, KEY, current_pattern_hash="h", min_adjudicated_rows=100, min_kappa=0.6,
        min_language_rows=30, **kwargs
    )


def test_a_complete_human_sheet_is_measured_and_gives_a_config_block():
    result = full_score(human_rows())
    assert result["is_measured_error_rate"] is True and result["not_measured_reasons"] == []
    block = result["suggested_config_block"]
    assert block["status"] == "measured" and block["labeller_kind"] == "human"
    assert block["pattern_hash"] == "h" and block["n_headlines"] == 100
    assert block["n_by_language"] == {"en": 60, "hi": 40}


@pytest.mark.parametrize("kind", [k for k in LABELLER_KINDS if k not in MEASURED_KINDS])
def test_only_human_labels_are_ever_measured(kind):
    result = full_score(human_rows(kind=kind))
    assert result["labeller_kind"] == kind
    assert result["is_measured_error_rate"] is False
    assert result["suggested_config_block"]["status"] == "not_measured"
    assert any("not 'human'" in r for r in result["not_measured_reasons"])


def test_the_labeller_kind_carries_into_the_output_and_mixed_kinds_say_mixed():
    rows = human_rows()
    rows[0]["labeller_kind"] = "model-unreviewed"
    result = full_score(rows)
    assert result["labeller_kind"] == "mixed"
    assert result["labeller_kind_counts"] == {"human": 99, "model-unreviewed": 1}
    assert result["is_measured_error_rate"] is False


def test_fewer_than_100_adjudicated_rows_is_not_measured():
    result = full_score(human_rows(n=99, n_hi=40))
    assert result["is_measured_error_rate"] is False
    assert any("fewer than 100" in r for r in result["not_measured_reasons"])


def test_a_language_with_fewer_than_30_rows_is_not_measured():
    result = full_score(human_rows(n_hi=29))
    assert result["is_measured_error_rate"] is False
    assert any("29 hi rows" in r for r in result["not_measured_reasons"])


def test_a_kappa_below_the_floor_is_not_measured():
    rows = human_rows()
    for i in range(0, 100, 2):  # labeller 2 flips half the rows, all adjudicated
        rows[i]["label_2"] = "true" if rows[i]["label_1"] != "true" else "none"
        rows[i]["adjudicated_label"] = rows[i]["label_1"]
    result = full_score(rows)
    assert result["cohen_kappa"] < 0.6 and result["is_measured_error_rate"] is False
    assert any("kappa" in r for r in result["not_measured_reasons"])


def test_a_key_from_another_pattern_set_is_not_measured():
    result = score_rows(
        human_rows(), KEY, current_pattern_hash="other", min_adjudicated_rows=100,
        min_kappa=0.6, min_language_rows=30,
    )
    assert result["is_measured_error_rate"] is False
    assert any("pattern set" in r for r in result["not_measured_reasons"])


# --- scoring: the sheet and the commands -------------------------------------


@pytest.fixture
def drawn(extractor, tmp_path):
    sheet, key = draw_sample(population(), extractor, n=100)
    path, key_path = write_sheet(sheet, key, tmp_path / "sheet.csv")
    return sheet, key, path, key_path


def test_a_blank_sheet_from_the_sampler_is_refused(drawn):
    _, _, path, _ = drawn
    with pytest.raises(SheetError, match="not fully labelled"):
        read_sheet(path)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("label_1", "", "label_1 is empty"),
        ("label_2", "", "label_2 is empty"),
        ("label_1", "maybe", "not one of"),
        ("adjudicated_label", "maybe", "adjudicated_label"),
        ("labeller_1", "", "labeller_1 is empty"),
        ("labeller_kind", "", "labeller_kind"),
        ("labeller_kind", "robot", "labeller_kind"),
        ("labeller_2", "asha", "same labeller"),
        ("labeller_2", "ASHA", "same labeller"),
    ],
)
def test_a_sheet_with_a_gap_is_refused_and_nothing_is_written(drawn, tmp_path, column, value,
                                                              message):
    sheet, _, path, key_path = drawn
    filled = fill(path, sheet, labeller_1="Asha")
    rows = list(csv.DictReader(filled.open(encoding="utf-8-sig")))
    rows[3][column] = value
    with filled.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SHEET_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "out.json"
    with pytest.raises(SheetError, match=message):
        score_sheet(filled, key_path, out)
    assert not out.exists()


def test_a_sheet_and_key_for_different_headlines_are_refused(drawn, tmp_path):
    sheet, key, path, key_path = drawn
    filled = fill(path, sheet)
    data = json.loads(key_path.read_text())
    data["rows"] = data["rows"][:-1]
    key_path.write_text(json.dumps(data))
    with pytest.raises(SheetError, match="different headlines"):
        score_sheet(filled, key_path, tmp_path / "out.json")


def test_a_sheet_with_a_missing_column_or_repeated_id_is_refused(drawn, tmp_path):
    sheet, _, path, _ = drawn
    filled = fill(path, sheet)
    text = filled.read_text(encoding="utf-8-sig").replace("labeller_kind", "kind")
    broken = tmp_path / "broken.csv"
    broken.write_text(text, encoding="utf-8-sig")
    with pytest.raises(SheetError, match="missing column"):
        read_sheet(broken)
    rows = list(csv.DictReader(filled.open(encoding="utf-8-sig")))
    rows[1]["record_id"] = rows[0]["record_id"]
    dup = tmp_path / "dup.csv"
    with dup.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SHEET_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SheetError, match="more than once"):
        read_sheet(dup)


def test_labels_and_labeller_names_are_case_folded(drawn):
    sheet, _, path, _ = drawn
    filled = fill(path, sheet, label_1="NONE", labeller_1="Asha", labeller_2="RAVI")
    row = read_sheet(filled)[0]
    assert row["label_1"] == "none" and row["labeller_1"] == "asha" and row["labeller_2"] == "ravi"


def test_the_whole_path_from_sheet_to_json_carries_the_kind_and_the_hash(drawn, tmp_path):
    sheet, key, path, key_path = drawn
    filled = fill(path, sheet)
    out = tmp_path / "score.json"
    result = score_sheet(filled, key_path, out)
    on_disk = json.loads(out.read_text())
    assert on_disk["labeller_kind"] == "human"
    assert on_disk["pattern_hash"] == key["pattern_hash"] == on_disk["current_pattern_hash"]
    assert on_disk["n_rows"] == 100 and on_disk["n_unresolved"] == 0
    assert set(on_disk["per_language"]) == {"en", "hi"}
    assert result["interval"].startswith("Wilson")
    assert join_sheet_and_key(read_sheet(filled), key)[0]["weight"] > 0


def test_the_sample_command_refuses_a_path_inside_the_repo():
    from ingest.cli import app

    repo = Path(__file__).resolve().parent.parent
    result = CliRunner().invoke(app, ["title-predicate-sample", "--out", str(repo / "sheet.csv")])
    assert result.exit_code == 1
    assert "refused" in re.sub(r"\x1b\[[0-9;]*m", "", result.output)


def test_the_sample_command_needs_at_least_one_headline(tmp_path):
    from ingest.cli import app

    result = CliRunner().invoke(
        app, ["title-predicate-sample", "--out", str(tmp_path / "s.csv"), "--n", "0"]
    )
    assert result.exit_code != 0 and not (tmp_path / "s.csv").exists()


def test_the_score_command_refuses_an_unlabelled_sheet_and_scores_a_full_one(drawn, tmp_path):
    from ingest.cli import app

    sheet, _, path, key_path = drawn
    out = tmp_path / "out.json"
    refused = CliRunner().invoke(app, ["title-predicate-score", str(path), "--out", str(out)])
    assert refused.exit_code == 1 and not out.exists()
    filled = fill(path, sheet)
    ok = CliRunner().invoke(
        app, ["title-predicate-score", str(filled), "--key", str(key_path), "--out", str(out)]
    )
    assert ok.exit_code == 0, ok.output
    assert json.loads(out.read_text())["labeller_kind"] == "human"
    assert not math.isnan(json.loads(out.read_text())["agreement"])
