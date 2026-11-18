"""Extra negatives for the misinfo training frame. Fakes only: no weights, no network."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import modeling.training as T
from modeling.cli import app
from modeling.config import ModelingSettings
from modeling.datasets.splits import group_train_val_test
from modeling.text import extra_negatives as X
from modeling.text.misinfo_clf import TrainedMisinfoModel

ANSI = re.compile(r"\x1b\[[0-9;]*m")
runner = CliRunner()


def reporting_file(tmp_path: Path, rows: int = 60, groups: int = 6, **overrides) -> Path:
    frame = pd.DataFrame(
        {
            "text": [f"report {i} on a viral claim about topic {i}" for i in range(rows)],
            "group_id": [f"checker{i % groups}.in" for i in range(rows)],
            "domain": [f"checker{i % groups}.in" for i in range(rows)],
        }
    )
    for key, value in overrides.items():
        frame[key] = value
    path = tmp_path / "reporting.parquet"
    frame.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------
def test_a_good_file_is_label_zero_reporting_with_namespaced_groups(tmp_path):
    extra = X.load_extra_negatives(reporting_file(tmp_path))
    frame = extra.frame
    assert set(frame["label"]) == {0}
    assert set(frame["source_dataset"]) == {"reporting"}
    assert frame["group_id"].str.startswith("reporting:checker").all()
    assert extra.rows_in_file == 60 and extra.groups_in_file == 6
    assert re.fullmatch(r"[0-9a-f]{64}", extra.sha256)


def test_the_domain_column_is_optional(tmp_path):
    path = reporting_file(tmp_path)
    pd.read_parquet(path).drop(columns="domain").to_parquet(path, index=False)
    assert set(X.load_extra_negatives(path).frame["domain"]) == {"reporting"}


@pytest.mark.parametrize(
    ("build", "reason"),
    [
        (lambda p: pd.DataFrame({"text": ["a"]}).to_parquet(p),
         X.ExtraNegativesReason.MISSING_COLUMNS),
        (lambda p: pd.DataFrame({"text": [], "group_id": []}).to_parquet(p),
         X.ExtraNegativesReason.NO_ROWS),
        (lambda p: pd.DataFrame({"text": ["ok", "  "], "group_id": ["a", "b"]}).to_parquet(p),
         X.ExtraNegativesReason.EMPTY_TEXT),
        (lambda p: pd.DataFrame({"text": ["ok", None], "group_id": ["a", "b"]}).to_parquet(p),
         X.ExtraNegativesReason.EMPTY_TEXT),
        (lambda p: pd.DataFrame({"text": ["ok", "no"], "group_id": ["a", ""]}).to_parquet(p),
         X.ExtraNegativesReason.EMPTY_GROUP),
        (lambda p: p.write_text("not parquet"), X.ExtraNegativesReason.UNREADABLE),
    ],
)
def test_a_bad_file_is_refused_with_a_reason_code(tmp_path, build, reason):
    path = tmp_path / "bad.parquet"
    build(path)
    with pytest.raises(X.ExtraNegativesRefused) as raised:
        X.load_extra_negatives(path)
    assert raised.value.reason == reason


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(X.ExtraNegativesRefused) as raised:
        X.load_extra_negatives(tmp_path / "none.parquet")
    assert raised.value.reason == X.ExtraNegativesReason.FILE_MISSING


def test_overlaps_with_the_benchmark_are_dropped_and_counted(tmp_path):
    extra = X.load_extra_negatives(reporting_file(tmp_path, rows=12, groups=4))
    benchmark = pd.DataFrame(
        {
            "text": [
                "  REPORT 0 on a viral claim about topic 0",
                "report 1 on a viral claim about topic 1",
            ],
            "label": [1, 0],
        }
    )
    kept = X.drop_overlaps(extra, benchmark)
    assert len(kept.frame) == 10
    assert kept.dropped[X.ExtraNegativesReason.CONFLICT_WITH_BENCHMARK] == 1
    assert kept.dropped[X.ExtraNegativesReason.DUPLICATE_OF_BENCHMARK] == 1
    assert kept.rows_in_file == 12


def test_duplicates_inside_the_file_are_counted_under_their_own_reason(tmp_path):
    path = reporting_file(tmp_path, rows=30, groups=6)
    frame = pd.read_parquet(path)
    frame.loc[5, "text"] = frame.loc[4, "text"]
    frame.to_parquet(path, index=False)
    kept = X.drop_overlaps(X.load_extra_negatives(path), pd.DataFrame({"text": [], "label": []}))
    assert kept.dropped[X.ExtraNegativesReason.DUPLICATE] == 1
    assert len(kept.frame) == 29


def test_near_duplicates_of_the_union_are_dropped_under_their_own_reason(tmp_path):
    base = "the minister said the bridge in the city collapsed after heavy rain last night"
    path = reporting_file(tmp_path, rows=12, groups=4)
    frame = pd.read_parquet(path)
    frame.loc[0, "text"] = base + " today"
    frame.loc[1, "text"] = base + " today!"
    frame.to_parquet(path, index=False)
    benchmark = pd.DataFrame({"text": [base + " today."], "label": [1]})
    kept = X.drop_overlaps(X.load_extra_negatives(path), benchmark)
    assert kept.dropped[X.ExtraNegativesReason.NEAR_DUPLICATE] == 2
    assert kept.dropped[X.ExtraNegativesReason.DUPLICATE] == 0
    assert len(kept.frame) == 10


@pytest.mark.parametrize("groups", [2, 5])
def test_too_few_groups_are_refused_and_the_message_says_nothing_trains(tmp_path, groups):
    extra = X.load_extra_negatives(reporting_file(tmp_path, rows=60, groups=groups))
    with pytest.raises(X.ExtraNegativesRefused) as raised:
        X.split_extra_negatives(extra, seed=1)
    assert raised.value.reason == X.ExtraNegativesReason.TOO_FEW_GROUPS
    assert "nothing" in str(raised.value) and str(X.MIN_GROUPS) in str(raised.value)


def test_a_group_sits_on_one_side_of_the_extra_split(tmp_path):
    extra = X.load_extra_negatives(reporting_file(tmp_path, rows=90, groups=9))
    train, val, test = X.split_extra_negatives(extra, seed=20260813)
    sides = [set(part["group_id"]) for part in (train, val, test)]
    assert not (sides[0] & sides[1] or sides[0] & sides[2] or sides[1] & sides[2])
    assert sum(len(part) for part in (train, val, test)) == 90
    assert set(extra.sides) == {"train", "val", "heldout"}
    assert sum(side["groups"] for side in extra.sides.values()) == 9


def test_the_publisher_bootstrap_is_wider_than_wilson_when_publishers_differ():
    scores = np.r_[np.full(50, 0.9), np.full(50, 0.1)]
    groups = np.array(["a"] * 50 + ["b"] * 50)
    boot = X.group_bootstrap_rate(scores, groups, 0.5, n_boot=500, seed=1)
    wilson = X.wilson_interval(50, 100)
    assert boot[0] < wilson[0] and boot[1] > wilson[1]
    assert all(np.isnan(v) for v in X.group_bootstrap_rate(scores, np.array(["a"] * 100), 0.5))


def test_precision_threshold_is_read_from_the_labelled_rows():
    scores = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9])
    labels = np.array([0, 1, 0, 1, 1, 1])
    assert X.precision_threshold(scores, labels, 0.85) == pytest.approx(0.7)
    assert X.precision_threshold(scores, np.zeros(6, dtype=int), 0.85) is None


def test_wilson_interval_brackets_the_rate():
    low, high = X.wilson_interval(7, 8)
    assert low < 7 / 8 < high <= 1.0
    assert all(np.isnan(v) for v in X.wilson_interval(0, 0))


# ---------------------------------------------------------------------------
# the training path, with a fake model
# ---------------------------------------------------------------------------
class FakeClassifier:
    instances: list[FakeClassifier] = []
    fallback_model = "fake-small"
    base_model = "fake-base"

    def __init__(self, settings):
        self.settings = settings
        self.fit_calls: list[dict] = []
        self.predict_calls: list[list[str]] = []
        type(self).instances.append(self)

    def fine_tune(self, train_texts, train_labels, val_texts, val_labels, *, output_dir, **kw):
        self.fit_calls.append(
            {"train": list(train_texts), "val": list(val_texts), "labels": train_labels,
             "output_dir": Path(output_dir)}
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "model.bin").write_bytes(b"fake weights")
        return TrainedMisinfoModel(Path(output_dir), "fake-base", None, 0.5, metadata={})

    def predict(self, texts):
        self.predict_calls.append(list(texts))
        rng = np.random.default_rng(0)
        return rng.uniform(0, 1, len(texts))


class FakeDataset:
    def __init__(self, frame):
        self.frame = frame

    def available(self, path=None, *, demo=False):
        return True

    def load(self, path=None, *, demo=False):
        return SimpleNamespace(frame=self.frame, summary=lambda: {"rows": len(self.frame)})


def benchmark_frame() -> pd.DataFrame:
    rows = 400
    return pd.DataFrame(
        {
            "text": [f"benchmark statement number {i} about policy" for i in range(rows)],
            "label": [(i // 8) % 2 for i in range(rows)],
            "speaker": [f"speaker{i // 8}" for i in range(rows)],
        }
    )


@pytest.fixture
def trainer_env(tmp_path, monkeypatch):
    FakeClassifier.instances.clear()
    settings_holder = {}
    # `register` reads the models folder from the registry settings, not from the argument.
    monkeypatch.setattr("modeling.registry.get_settings", lambda: settings_holder["s"])
    monkeypatch.setattr("modeling.text.misinfo_clf.MisinfoClassifier", FakeClassifier)
    monkeypatch.setattr(
        T, "get_dataset",
        lambda key: FakeDataset(benchmark_frame()) if key == "liar" else _Absent(),
    )
    monkeypatch.setattr(T, "_domain_holdout_note", lambda *a, **k: "")
    monkeypatch.setattr(T, "_corpus_transfer_note", lambda *a, **k: "")
    settings = ModelingSettings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        artifacts_dir=tmp_path / "artifacts",
        _env_file=None,
    )
    settings_holder["s"] = settings
    return SimpleNamespace(settings=settings, tmp=tmp_path)


class _Absent:
    def available(self, path=None, *, demo=False):
        return False


def train(env, **kwargs):
    return T.train_module("misinfo", settings=env.settings, **kwargs)


def test_the_benchmark_split_does_not_change_with_extra_negatives(trainer_env):
    plain = train(trainer_env)
    assert not plain.skipped, plain.notes
    base = FakeClassifier.instances[-1]

    path = reporting_file(trainer_env.tmp, rows=120, groups=8)
    with_extra = train(trainer_env, extra_negatives=path, output_version="cand-1")
    assert not with_extra.skipped, with_extra.notes
    cand = FakeClassifier.instances[-1]

    def is_benchmark(text: str) -> bool:
        return text.startswith("benchmark")

    plain_fit, cand_fit = base.fit_calls[0], cand.fit_calls[0]
    assert [t for t in cand_fit["train"] if is_benchmark(t)] == plain_fit["train"]
    # The calibration set is the benchmark validation set, with no reporting row in it.
    assert cand_fit["val"] == plain_fit["val"]
    # The first predict call scores the benchmark test rows, in the same order.
    assert cand.predict_calls[0] == base.predict_calls[0]
    assert any(not is_benchmark(t) for t in cand_fit["train"])
    # The reporting rows are label 0.
    extra_labels = [
        label for t, label in zip(cand_fit["train"], cand_fit["labels"], strict=True)
        if not is_benchmark(t)
    ]
    assert extra_labels and set(extra_labels) == {0}


def test_the_benchmark_split_equals_the_splitter_on_the_benchmark_alone(trainer_env):
    path = reporting_file(trainer_env.tmp, rows=120, groups=8)
    train(trainer_env, extra_negatives=path, output_version="cand-1")
    fit = FakeClassifier.instances[-1].fit_calls[0]
    from modeling.text.misinfo_clf import build_training_frame

    frame = build_training_frame({"liar": benchmark_frame()})
    work, split = group_train_val_test(
        frame, group_col="group_id", label_col="label", seed=trainer_env.settings.seed
    )
    expected = work.iloc[split.train]["text"].tolist()
    assert [t for t in fit["train"] if t.startswith("benchmark")] == expected


def test_the_report_records_the_file_its_hash_and_counts(trainer_env):
    path = reporting_file(trainer_env.tmp, rows=120, groups=8)
    result = train(trainer_env, extra_negatives=path, output_version="cand-1")
    eval_root = trainer_env.settings.artifacts_dir / "eval" / "misinfo"
    metrics_path = eval_root / "cand-1" / "metrics.json"
    recorded = json.loads(metrics_path.read_text())["dataset"]["extra_negatives"]
    assert recorded["file"] == "reporting.parquet"
    assert recorded["sha256"] == X.file_sha256(path)
    assert recorded["rows"] == 120 and recorded["groups"] == 8
    report_text = (metrics_path.parent / "report.md").read_text()
    assert "Held-out reporting negatives" in report_text
    registry_file = trainer_env.settings.models_dir / "misinfo" / "cand-1" / "registry.json"
    assert json.loads(registry_file.read_text())["extra_negatives"]["sha256"] == recorded["sha256"]
    assert any("cand-1" in note for note in result.notes)


def test_extra_negatives_without_an_output_version_are_refused_and_nothing_is_written(
    trainer_env,
):
    path = reporting_file(trainer_env.tmp, rows=120, groups=8)
    result = train(trainer_env, extra_negatives=path)
    assert result.skipped
    assert X.ExtraNegativesReason.OUTPUT_VERSION_REQUIRED in result.notes[0]
    assert not (trainer_env.settings.models_dir / "misinfo").exists()
    assert not FakeClassifier.instances


def test_fine_tune_refuses_a_folder_that_holds_files(tmp_path):
    from modeling.text.misinfo_clf import MisinfoClassifier, OutputDirNotEmpty

    folder = tmp_path / "v0.2.0"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"shipped weights")
    classifier = MisinfoClassifier(
        ModelingSettings(data_dir=tmp_path, models_dir=tmp_path, _env_file=None)
    )
    with pytest.raises(OutputDirNotEmpty, match="output_dir_not_empty"):
        classifier.fine_tune(["a"], np.array([0]), ["b"], np.array([0]), output_dir=folder)
    assert (folder / "model.safetensors").read_bytes() == b"shipped weights"


def test_the_calibration_set_is_the_benchmark_validation_set_and_the_base_rate_is_logged(
    trainer_env, caplog
):
    import logging

    path = reporting_file(trainer_env.tmp, rows=120, groups=8)
    with caplog.at_level(logging.INFO):
        train(trainer_env, extra_negatives=path, output_version="cand-1")
    fit = FakeClassifier.instances[-1].fit_calls[0]
    assert all(t.startswith("benchmark") for t in fit["val"])
    assert "benchmark validation positive rate" in caplog.text


def test_the_held_out_ids_are_saved_without_text_and_the_report_has_the_caveats(trainer_env):
    path = reporting_file(trainer_env.tmp, rows=120, groups=8)
    train(trainer_env, extra_negatives=path, output_version="cand-1")
    directory = trainer_env.settings.artifacts_dir / "eval" / "misinfo" / "cand-1"
    ids = pd.read_parquet(directory / "reporting_heldout_ids.parquet")
    assert list(ids.columns) == ["record_id", "group_id", "domain", "text_sha256"]
    meta = json.loads((directory / "reporting_heldout.json").read_text())
    assert meta["rows"] == len(ids) and meta["extra_negatives_sha256"] == X.file_sha256(path)
    report = (directory / "report.md").read_text()
    assert "fact_check_publisher" in report and "not measure other outlets" in report
    assert "publisher bootstrap interval" in report and "row-level Wilson" in report
    assert "publishers" in report
    recorded = json.loads((directory / "metrics.json").read_text())["dataset"]["extra_negatives"]
    assert set(recorded["sides"]) == {"train", "val", "heldout"}
    assert "threshold_precision_085" in recorded["heldout"]


def test_any_checkpoint_scores_the_same_held_out_rows(trainer_env, tmp_path, monkeypatch):
    path = reporting_file(trainer_env.tmp, rows=120, groups=8)
    train(trainer_env, extra_negatives=path, output_version="cand-1")
    directory = trainer_env.settings.artifacts_dir / "eval" / "misinfo" / "cand-1"
    ids = directory / "reporting_heldout_ids.parquet"

    import modeling.cli as cli
    import modeling.registry as registry

    monkeypatch.setattr(cli, "_bootstrap", lambda verbose=False: None)
    monkeypatch.setattr(registry, "get_settings", lambda: trainer_env.settings)
    FakeClassifier.load = lambda self, model_dir: True
    out = tmp_path / "rates.json"
    scores = tmp_path / "scores.parquet"
    result = runner.invoke(
        app,
        ["score-reporting-heldout", "--extra-negatives", str(path), "--heldout", str(ids),
         "--version", "cand-1", "--allow-unverified-candidate", "--threshold", "0.5",
         "--out", str(out), "--scores-out", str(scores)],
    )
    assert result.exit_code == 0, ANSI.sub("", result.output)
    recorded = json.loads(out.read_text())
    held = pd.read_parquet(ids)
    assert recorded["rows"] == len(held)
    assert set(pd.read_parquet(scores)["text_sha256"]) == set(held["text_sha256"])
    assert recorded["checkpoint"].startswith("cand-1+")


def test_a_changed_extra_file_is_refused_for_the_held_out_list(trainer_env):
    path = reporting_file(trainer_env.tmp, rows=120, groups=8)
    train(trainer_env, extra_negatives=path, output_version="cand-1")
    ids = (trainer_env.settings.artifacts_dir / "eval" / "misinfo" / "cand-1"
           / "reporting_heldout_ids.parquet")
    frame = pd.read_parquet(path)
    frame.loc[0, "text"] = frame.loc[0, "text"] + " edited"
    frame.to_parquet(path, index=False)
    with pytest.raises(X.ExtraNegativesRefused) as raised:
        X.select_heldout(path, ids)
    assert raised.value.reason == X.ExtraNegativesReason.HELDOUT_MISMATCH


def test_output_version_writes_there_and_refuses_an_existing_folder(trainer_env):
    path = reporting_file(trainer_env.tmp, rows=120, groups=8)
    train(trainer_env, extra_negatives=path, output_version="cand-1")
    target = trainer_env.settings.models_dir / "misinfo" / "cand-1"
    assert (target / "model.bin").is_file()
    before = (target / "registry.json").read_text()

    again = train(trainer_env, extra_negatives=path, output_version="cand-1")
    assert again.skipped
    assert X.ExtraNegativesReason.OUTPUT_EXISTS in again.notes[0]
    assert (target / "registry.json").read_text() == before


@pytest.mark.parametrize("version", ["../escape", "a/b", "", " x", "v0.2.0"])
def test_a_bad_or_configured_output_version_is_refused(trainer_env, version):
    result = train(trainer_env, output_version=version)
    assert result.skipped
    assert not (trainer_env.settings.models_dir / "misinfo").exists()


def test_a_bad_extra_file_stops_the_run_before_any_training(trainer_env):
    bad = trainer_env.tmp / "bad.parquet"
    pd.DataFrame({"text": ["a"]}).to_parquet(bad)
    result = train(trainer_env, extra_negatives=bad, output_version="cand-2")
    assert result.skipped and X.ExtraNegativesReason.MISSING_COLUMNS in result.notes[0]
    assert not FakeClassifier.instances


def test_the_options_apply_to_misinfo_only(trainer_env):
    result = T.train_module(
        "stance", settings=trainer_env.settings, extra_negatives=trainer_env.tmp / "x.parquet"
    )
    assert result.skipped and X.ExtraNegativesReason.WRONG_MODULE in result.notes[0]


def test_without_the_options_the_run_writes_the_configured_version(trainer_env):
    from modeling.config import module_config

    train(trainer_env)
    version = str(module_config("misinfo")["version"])
    fit = FakeClassifier.instances[-1].fit_calls[0]
    assert fit["output_dir"] == trainer_env.settings.models_dir / "misinfo" / version
    assert all(t.startswith("benchmark") for t in fit["train"])
    metrics = trainer_env.settings.artifacts_dir / "eval" / "misinfo" / version / "metrics.json"
    assert "extra_negatives" not in json.loads(metrics.read_text())["dataset"]


# ---------------------------------------------------------------------------
# the builder command
# ---------------------------------------------------------------------------
def corpus_records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list("abcdefg"),
            "source": "news_rss",
            "source_detail": [
                "boomlive.in", "www.AltNews.in", "thehindu.com", "boomlive.in",
                "boomlive.in", "vishvasnews.com", "boomlive.in",
            ],
            "text": ["debunk one", "debunk two", "outlet story", "debunk three", "",
                     "hindi debunk", None],
            "lang": ["en", "en", "en", "en", "en", "hi", "en"],
        }
    )


def test_select_reporting_rows_matches_the_scorer_domain_test():
    from modeling.config import fact_check_domains

    frame, dropped = X.select_reporting_rows(corpus_records(), fact_check_domains(), lang="en")
    assert sorted(frame["text"]) == ["debunk one", "debunk three", "debunk two"]
    assert set(frame["group_id"]) == {"boomlive.in", "altnews.in"}
    assert sorted(frame["record_id"]) == ["a", "b", "d"]
    reasons = X.ExtraNegativesReason
    assert dropped[reasons.LANGUAGE_MISMATCH] == 1
    assert dropped[reasons.ROW_EMPTY_TEXT] == 2
    # The domain filter is the filter, so it is not reported as a drop.
    assert "not_fact_check_domain" not in dropped


def test_the_builder_drops_records_that_have_a_claim_row_of_any_label():
    from modeling.config import fact_check_domains

    frame, dropped = X.select_reporting_rows(
        corpus_records(), fact_check_domains(), claim_record_ids={"a", "d", "zzz"}
    )
    assert list(frame["record_id"]) == ["b"]
    assert dropped[X.ExtraNegativesReason.CLAIM_RECORD] == 2


def test_the_builder_drops_text_that_contains_a_frozen_claim_after_normalising():
    from modeling.config import fact_check_domains

    records = corpus_records()
    records.loc[1, "text"] = "Intro.  The Bridge   COLLAPSED in the flood, says viral post. End."
    frame, dropped = X.select_reporting_rows(
        records, fact_check_domains(), frozen_claim_texts=["the bridge collapsed in the flood"]
    )
    assert "b" not in set(frame["record_id"])
    assert dropped[X.ExtraNegativesReason.FROZEN_CLAIM_TEXT] == 1
    assert dropped[X.ExtraNegativesReason.CLAIM_RECORD] == 0


def test_leak_guards_read_every_claim_row_and_the_manifest_claim_texts(tmp_path, monkeypatch):
    import ingest.factcheck as factcheck
    from modeling.eval.factcheck_eval import MANIFEST_COLUMNS, claim_hash

    rows = [
        {"record_id": "news:x/1", "claim": "claim one is false", "label": "false"},
        {"record_id": "news:x/2", "claim": "claim two is true", "label": "true"},
        {"record_id": "news:x/3", "claim": "claim three is false", "label": "false"},
    ]
    monkeypatch.setattr(factcheck, "read_factcheck_rows", lambda settings=None: rows)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        ",".join(MANIFEST_COLUMNS) + "\n"
        f"news:x/1,u,{claim_hash('claim one is false')},false\n",
        encoding="utf-8",
    )
    ids, texts = X.leak_guards(manifest)
    assert ids == {"news:x/1", "news:x/2", "news:x/3"}
    assert texts == ["claim one is false"]


def test_leak_guards_refuse_without_the_manifest(tmp_path):
    with pytest.raises(X.ExtraNegativesRefused) as raised:
        X.leak_guards(tmp_path / "none.csv")
    assert raised.value.reason == X.ExtraNegativesReason.MANIFEST_MISSING


def patch_guards(monkeypatch, ids=frozenset(), texts=()):
    monkeypatch.setattr(X, "leak_guards", lambda manifest=None, **kw: (set(ids), list(texts)))


def test_the_command_writes_the_file_and_prints_counts_per_domain(tmp_path, monkeypatch):
    import modeling.io as io

    monkeypatch.setattr(io.CorpusReader, "records", lambda self, **kw: corpus_records())
    patch_guards(monkeypatch, ids={"d"})
    out = tmp_path / "nested" / "reporting.parquet"
    result = runner.invoke(app, ["reporting-negatives", "--out", str(out)])
    text = ANSI.sub("", result.output)
    assert result.exit_code == 0, text
    assert "boomlive.in" in text and "altnews.in" in text and "2 rows" in text
    assert X.ExtraNegativesReason.CLAIM_RECORD in text
    assert "not_fact_check_domain" not in text
    written = pd.read_parquet(out)
    assert list(written.columns) == ["record_id", "text", "group_id", "domain"]
    assert len(written) == 2


def test_the_command_refuses_without_the_manifest(tmp_path, monkeypatch):
    import modeling.io as io

    monkeypatch.setattr(io.CorpusReader, "records", lambda self, **kw: corpus_records())
    out = tmp_path / "reporting.parquet"
    result = runner.invoke(
        app, ["reporting-negatives", "--out", str(out), "--manifest", str(tmp_path / "no.csv")]
    )
    assert result.exit_code == 1 and not out.exists()
    assert X.ExtraNegativesReason.MANIFEST_MISSING in ANSI.sub("", result.output)


def test_the_command_writes_nothing_when_no_row_matches(tmp_path, monkeypatch):
    import modeling.io as io

    records = corpus_records().assign(lang="hi")
    monkeypatch.setattr(io.CorpusReader, "records", lambda self, **kw: records)
    patch_guards(monkeypatch)
    out = tmp_path / "reporting.parquet"
    result = runner.invoke(app, ["reporting-negatives", "--out", str(out)])
    assert result.exit_code == 1 and not out.exists()
    assert "no_rows" in ANSI.sub("", result.output)


def test_train_help_lists_the_new_options():
    text = ANSI.sub("", runner.invoke(app, ["train", "--help"], terminal_width=200).output)
    assert "--extra-negatives" in text and "--output-version" in text
