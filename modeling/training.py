"""Training orchestration: load, split, baseline, fit, calibrate, report.

One entry point per module, all following the same order, because the order is
the methodology:

1. **Load** the benchmark from a local path, or refuse with instructions.
2. **Split** through ``datasets/splits.py``. Never anywhere else.
3. **Baselines first.** They run in seconds and they set the bar. Running them
   before the expensive model means the bar is fixed before anyone has a stake
   in clearing it.
4. **Fit** the main model.
5. **Calibrate** on the validation split.
6. **Report** — metrics with CIs, baseline comparison, calibration curve, error
   analysis, model card — and save the raw predictions so the report can be
   regenerated without retraining.

If the main model does not beat its baselines, that is logged loudly and written
into the report. It is not a reason to keep tuning.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.config import ModelingSettings, get_settings, module_config, set_all_seeds
from modeling.datasets import DatasetUnavailable, get_dataset
from modeling.datasets.splits import domain_holdout, group_train_val_test
from modeling.eval import baselines as B
from modeling.eval.error_analysis import analyze, write_markdown
from modeling.eval.metrics import (
    ClassificationReport,
    classification_report,
)
from modeling.eval.report import save_predictions, write_report

log = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    module: str
    version: str
    report: ClassificationReport | None
    artifacts: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: bool = False

    def headline(self) -> str:
        if self.skipped:
            return "SKIPPED: " + "; ".join(self.notes)
        if self.report is None:
            return "no report produced: " + "; ".join(self.notes)
        return self.report.headline()


#: ``module name -> trainer``. A decorator registry rather than a dict literal
#: inside ``train_module``, mirroring ``modeling/scoring.py``'s ``@stage``: a
#: module's trainer is then declared next to the trainer, and adding one cannot
#: be half-done.
TrainerFn = Callable[..., "TrainingResult"]
_TRAINERS: dict[str, TrainerFn] = {}


def trainer(name: str) -> Callable[[TrainerFn], TrainerFn]:
    def decorate(fn: TrainerFn) -> TrainerFn:
        _TRAINERS[name] = fn
        return fn

    return decorate


def trainable_modules() -> list[str]:
    """Modules with a registered training path. Backs the CLI's help text."""
    return sorted(_TRAINERS)


def train_module(
    module: str,
    *,
    data_path: Path | None = None,
    demo: bool = False,
    epochs: int | None = None,
    settings: ModelingSettings | None = None,
) -> TrainingResult:
    settings = settings or get_settings()
    settings.ensure_dirs()
    set_all_seeds()

    handler = _TRAINERS.get(module)
    if handler is None:
        return TrainingResult(
            module,
            module_config(module).get("version", "v0.0.0-unset"),
            None,
            skipped=True,
            notes=[
                f"no training path for {module!r}. Trainable modules: "
                f"{', '.join(trainable_modules())}."
            ],
        )
    try:
        return handler(settings, data_path=data_path, demo=demo, epochs=epochs)
    except DatasetUnavailable as exc:
        version = str(module_config(module).get("version", "v0.0.0-unset"))
        log.warning("%s: %s", module, exc)
        return TrainingResult(module, version, None, skipped=True, notes=[str(exc)])


def evaluate_module(
    module: str,
    *,
    data_path: Path | None = None,
    demo: bool = False,
    settings: ModelingSettings | None = None,
) -> TrainingResult:
    """Evaluate a trained checkpoint without retraining it."""
    from modeling.eval.report import load_predictions

    settings = settings or get_settings()
    version = str(module_config(module).get("version", "v0.0.0-unset"))
    saved = load_predictions(module, version, settings)
    if saved is None:
        return TrainingResult(
            module, version, None, skipped=True,
            notes=[f"no saved predictions for {module}/{version}; run `modeling train {module}`"],
        )
    report = classification_report(
        saved["y_true"].to_numpy(),
        saved["y_pred"].to_numpy(),
        y_score=saved["y_score"].to_numpy() if "y_score" in saved else None,
        module=module,
        split_description=str(saved.attrs.get("split", "see metrics.json")),
        class_names=None,
        seed=settings.seed,
        is_demo=demo,
    )
    return TrainingResult(module, version, report)


# ---------------------------------------------------------------------------
# misinfo
# ---------------------------------------------------------------------------
@trainer("misinfo")
def _train_misinfo(
    settings: ModelingSettings,
    *,
    data_path: Path | None,
    demo: bool,
    epochs: int | None,
) -> TrainingResult:
    from modeling.registry import register
    from modeling.text.misinfo_clf import LABEL_NAMES, MisinfoClassifier, build_training_frame

    version = str(module_config("misinfo").get("version", "v0.0.0-unset"))
    notes: list[str] = []

    loaded: dict[str, pd.DataFrame] = {}
    dataset_summaries: dict[str, Any] = {}
    for key in ("liar", "fakenewsnet", "coaid"):
        dataset = get_dataset(key)
        if not dataset.available(data_path, demo=demo):
            notes.append(f"{key} not available; excluded from training")
            log.warning("%s not on disk; training without it", key)
            continue
        result = dataset.load(data_path, demo=demo)
        loaded[key] = result.frame
        dataset_summaries[key] = result.summary()

    if not loaded:
        raise DatasetUnavailable(
            "none of LIAR, FakeNewsNet or CoAID is available. Every one is a manual "
            "download; see `modeling datasets` for the steps, or use --demo."
        )

    frame = build_training_frame(loaded)
    if len(frame) < 30:
        return TrainingResult(
            "misinfo", version, None, skipped=True,
            notes=notes + [f"only {len(frame)} usable rows; refusing to report a metric"],
        )

    work, split = group_train_val_test(
        frame, group_col="group_id", label_col="label", seed=settings.seed
    )
    split_description = split.describe()
    log.info("misinfo training set: %d rows, %s", len(work), split_description)

    train = work.iloc[split.train]
    val = work.iloc[split.val]
    test = work.iloc[split.test]

    # --- baselines first, so the bar is fixed before anyone has a stake -----
    baseline_results = [
        B.majority_baseline(
            train["label"].to_numpy(), test["label"].to_numpy(),
            module="misinfo", split_description=split_description,
            seed=settings.seed, is_demo=demo,
        ),
        B.tfidf_logreg(
            train["text"].tolist(), train["label"].to_numpy(),
            test["text"].tolist(), test["label"].to_numpy(),
            module="misinfo", split_description=split_description,
            seed=settings.seed, is_demo=demo,
        ),
    ]
    for baseline in baseline_results:
        log.info("baseline %s", baseline.headline())

    # --- fine-tune ---------------------------------------------------------
    classifier = MisinfoClassifier(settings)
    output_dir = settings.models_dir / "misinfo" / version
    # The demo path deliberately uses the smaller base model. `--demo` exists so
    # the pipeline is executable on a clean clone, and making it wait on a
    # 500 MB roberta-base download defeats that. Nothing trained on fixtures is
    # a result either way, so the smaller backbone costs nothing real -- but the
    # report records which backbone produced the numbers.
    base_model = classifier.fallback_model if demo else None
    trained = classifier.fine_tune(
        train["text"].tolist(), train["label"].to_numpy(),
        val["text"].tolist(), val["label"].to_numpy(),
        output_dir=output_dir, base_model=base_model, epochs=epochs,
    )
    if demo:
        notes.append(f"demo run used {trained.base_model} rather than {classifier.base_model}")

    scores = classifier.predict(test["text"].tolist())
    predictions = (scores >= trained.threshold).astype(int)
    report = classification_report(
        test["label"].to_numpy(), predictions, y_score=scores,
        module="misinfo", split_description=split_description,
        class_names=LABEL_NAMES, seed=settings.seed, is_demo=demo,
    )
    log.info("misinfo %s", report.headline())

    comparison = B.compare(report, baseline_results)
    if not comparison["clears_every_baseline"]:
        notes.append(
            "the fine-tune does not cleanly separate from every baseline on this test set"
        )

    # --- per-dataset and cross-domain breakdowns ---------------------------
    extra_sections: dict[str, str] = {}
    extra_sections["Per-benchmark breakdown"] = _per_dataset_table(
        test, predictions, scores, settings, demo
    )
    domain_note = _domain_holdout_note(loaded, classifier, settings, demo, epochs)
    if domain_note:
        extra_sections["Cross-domain transfer (PolitiFact -> GossipCop)"] = domain_note
    extra_sections["Corpus transfer"] = _corpus_transfer_note(settings)

    # --- artifacts ---------------------------------------------------------
    predictions_frame = pd.DataFrame(
        {
            "text": test["text"].to_numpy(),
            "y_true": test["label"].to_numpy(),
            "y_pred": predictions,
            "y_score": scores,
            "source_dataset": test["source_dataset"].to_numpy(),
            "domain": test["domain"].to_numpy(),
        }
    )
    artifacts = [save_predictions("misinfo", version, predictions_frame, settings=settings)]
    calibration = _calibration_from(trained)
    artifacts += write_report(
        module="misinfo", version=version, report=report, baselines=comparison,
        calibration=calibration, dataset_summary=dataset_summaries,
        extra_sections=extra_sections, settings=settings,
    )

    analysis = analyze(predictions_frame, module="misinfo", seed=settings.seed)
    artifacts.append(write_markdown(analysis, settings=settings))

    checkpoint = register(
        "misinfo", version, output_dir,
        metadata={
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "base_model": trained.base_model,
            "split": split_description,
            "metrics": report.as_dict(),
            "is_demo": demo,
        },
    )
    notes.append(f"checkpoint {checkpoint.uri}")
    return TrainingResult("misinfo", version, report, artifacts=artifacts, notes=notes)


def _calibration_from(trained) -> Any:
    from modeling.eval.calibrate import CalibrationResult

    payload = (trained.metadata or {}).get("calibration")
    if not payload:
        return None
    return CalibrationResult(
        method=payload["method"],
        brier_before=payload["brier_before"],
        brier_after=payload["brier_after"],
        reliability_before=payload["reliability_before"],
        reliability_after=payload["reliability_after"],
        n_calibration=payload["n_calibration"],
        degraded=payload.get("degraded", False),
        note=payload.get("note", ""),
    )


def _per_dataset_table(
    test: pd.DataFrame, predictions: np.ndarray, scores: np.ndarray,
    settings: ModelingSettings, demo: bool,
) -> str:
    """Per-benchmark metrics, so the union does not hide a weak component.

    LIAR, FakeNewsNet and CoAID are three different problems. A single averaged
    F1 over their union tells you nothing about which one the model actually
    learned.
    """
    lines = [
        "The three benchmarks are three different problems: politicians' statements, "
        "news headlines, and COVID-era health claims. A single averaged F1 over their "
        "union hides which one the model actually learned.",
        "",
        "| benchmark | n | macro F1 | positive rate |",
        "|---|---|---|---|",
    ]
    for name in sorted(test["source_dataset"].unique()):
        mask = (test["source_dataset"] == name).to_numpy()
        if mask.sum() < 10:
            lines.append(f"| {name} | {int(mask.sum())} | too small to report | — |")
            continue
        sub = classification_report(
            test["label"].to_numpy()[mask], predictions[mask], y_score=scores[mask],
            module="misinfo", split_description="subset of the grouped test split",
            seed=settings.seed, is_demo=demo,
        )
        lines.append(
            f"| {name} | {int(mask.sum())} | {sub.macro_f1.value:.3f} "
            f"[{sub.macro_f1.low:.3f}, {sub.macro_f1.high:.3f}] | {sub.positive_rate:.3f} |"
        )
    return "\n".join(lines)


def _domain_holdout_note(
    loaded: dict[str, pd.DataFrame], classifier, settings: ModelingSettings,
    demo: bool, epochs: int | None,
) -> str:
    """Train on PolitiFact, test on GossipCop.

    This number is more honest than in-domain F1 and reviewers respect it: it is
    the closest thing in the benchmark suite to "what happens on data you have
    not seen the production pipeline of".
    """
    frame = loaded.get("fakenewsnet")
    if frame is None or "domain" not in frame.columns:
        return ""
    if frame["domain"].nunique() < 2:
        return "_Only one FakeNewsNet domain present; the transfer number cannot be computed._"

    try:
        work, split = domain_holdout(
            frame, domain_col="domain", held_out="gossipcop", group_col="claim_id"
        )
    except ValueError as exc:
        return f"_Domain holdout unavailable: {exc}_"

    if len(split.train) < 20 or len(split.test) < 10:
        return (
            f"_Too few rows for a domain holdout (train {len(split.train)}, "
            f"test {len(split.test)})._"
        )

    baseline = B.tfidf_logreg(
        work["text"].iloc[split.train].tolist(), work["label"].iloc[split.train].to_numpy(),
        work["text"].iloc[split.test].tolist(), work["label"].iloc[split.test].to_numpy(),
        module="misinfo", split_description="train PolitiFact -> test GossipCop",
        seed=settings.seed, is_demo=demo,
    )
    return "\n".join(
        [
            "Trained on PolitiFact (political fact-checks), tested on GossipCop "
            "(celebrity gossip) — a genuine domain shift inside one benchmark.",
            "",
            "Reported here with the TF-IDF baseline rather than the fine-tune, because "
            "re-fine-tuning for one table costs a full training run; the baseline's drop "
            "measures the shift itself, which is the quantity of interest.",
            "",
            "- in-domain reference: see the main table above",
            f"- PolitiFact -> GossipCop, TF-IDF baseline: {baseline.report.macro_f1}",
            f"- test rows: {len(split.test)}",
            "",
            "**Expect the fine-tune to drop similarly.** House style is memorizable; the "
            "underlying task is not.",
        ]
    )


def _corpus_transfer_note(settings: ModelingSettings) -> str:
    """The hand-labelled corpus-transfer check.

    100 real corpus records, hand-labelled, scored by this model. This is the
    single most informative table in the report and it cannot be automated --
    the labels have to come from a person.
    """
    path = settings.artifacts_dir / "hand_labels" / "misinfo_corpus_sample.csv"
    if not path.exists():
        return "\n".join(
            [
                "**Not yet measured.** Benchmark F1 is not production accuracy and must not "
                "be quoted as if it were: LIAR is politicians' statements, FakeNewsNet is "
                "news headlines, and this project's corpus is Reddit comments, Mastodon "
                "toots and GDELT article metadata.",
                "",
                "To measure the gap:",
                "",
                "```bash",
                "python -m modeling.cli sample-for-labelling misinfo --n 100",
                "```",
                "",
                f"That writes `{path.relative_to(settings.artifacts_dir.parent)}` with a blank "
                "`label` column. Fill it in by hand, rerun `modeling evaluate misinfo`, and "
                "this section becomes a table. Until then, the honest statement is that the "
                "transfer gap is **unmeasured and expected to be large**.",
            ]
        )

    frame = pd.read_csv(path)
    labelled = frame.loc[frame["label"].notna()]
    if len(labelled) < 20:
        return (
            f"_Only {len(labelled)} of {len(frame)} sampled records have been hand-labelled; "
            "at least 20 are needed before reporting a transfer number._"
        )
    if "score" not in labelled.columns:
        return "_Hand labels present but not yet scored; rerun `modeling evaluate misinfo`._"

    report = classification_report(
        labelled["label"].to_numpy().astype(int),
        (labelled["score"].to_numpy() >= 0.5).astype(int),
        y_score=labelled["score"].to_numpy(),
        module="misinfo",
        split_description=f"{len(labelled)} hand-labelled corpus records",
        seed=settings.seed,
    )
    return "\n".join(
        [
            f"**{len(labelled)} real corpus records, hand-labelled.** This is the number that "
            "describes production behaviour; the benchmark numbers above describe the "
            "benchmarks.",
            "",
            f"- macro F1: {report.macro_f1}",
            f"- PR-AUC: {report.pr_auc}" if report.pr_auc else "",
            f"- positive rate in the sample: {report.positive_rate:.3f}",
        ]
    )


# ---------------------------------------------------------------------------
# bot
# ---------------------------------------------------------------------------
@trainer("bot")
def _train_bot(
    settings: ModelingSettings,
    *,
    data_path: Path | None,
    demo: bool,
    epochs: int | None,
) -> TrainingResult:
    from modeling.accounts.bot_clf import train_bot_classifier

    return train_bot_classifier(settings, data_path=data_path, demo=demo)


# ---------------------------------------------------------------------------
# stance
# ---------------------------------------------------------------------------
@trainer("stance")
def _train_stance(
    settings: ModelingSettings,
    *,
    data_path: Path | None,
    demo: bool,
    epochs: int | None,
) -> TrainingResult:
    """Fine-tune a pre-existing NLI checkpoint on FNC-1.

    The test set is the organizers' competition set, the published comparison
    point. Train and val are split by headline (``claim_id``), because one
    headline appears against many bodies and a body-level split puts every
    validation headline in train too. Train rows whose headline also occurs in
    the test set are purged. Nothing dedupes on ``text``: in FNC-1 one body
    legitimately appears against many headlines, so near-duplicate collapse
    would delete most of the corpus.
    """
    from modeling.registry import register
    from modeling.text.stance_clf import STANCE_LABELS, StanceClassifier

    version = str(module_config("stance").get("version", "v0.0.0-unset"))
    notes: list[str] = []

    dataset = None
    for key in ("fnc1", "stance"):
        candidate = get_dataset(key)
        if candidate.available(data_path, demo=demo):
            dataset = candidate
            break
    if dataset is None:
        raise DatasetUnavailable(
            get_dataset("fnc1").info.instructions(
                get_dataset("fnc1").resolve_path(data_path, demo)
            )
        )

    loaded = dataset.load(data_path, demo=demo)
    frame = loaded.frame
    dataset_summaries = {loaded.info.key: loaded.summary()}

    if len(frame) < 40:
        return TrainingResult(
            "stance", version, None, skipped=True,
            notes=[f"only {len(frame)} usable pairs; refusing to report a metric"],
        )

    work, split, split_description, split_notes = stance_split(frame, settings)
    notes.extend(split_notes)

    label_index = {label: i for i, label in enumerate(STANCE_LABELS)}
    work = work.assign(label_id=work["label"].map(label_index))
    if work["label_id"].isna().any():
        unknown = sorted(set(work.loc[work["label_id"].isna(), "label"]))
        raise ValueError(f"labels outside the contract vocabulary: {unknown}")
    work["label_id"] = work["label_id"].astype(int)

    train = work.iloc[split.train]
    val = work.iloc[split.val]
    test = work.iloc[split.test]
    log.info("stance training set: %d rows, %s", len(work), split_description)

    attested = sorted(set(work["label"].unique()))
    missing = [label for label in STANCE_LABELS if label not in attested]
    if missing:
        notes.append(
            f"labels {missing} are UNATTESTED in this benchmark; a model trained on it "
            "can never predict them. This is a label-set coverage gap, and it belongs "
            "in the model card rather than in a silently-invented mapping."
        )

    def pairs_of(part: pd.DataFrame) -> list[tuple[str, str]]:
        return list(zip(part["target"].astype(str), part["text"].astype(str), strict=True))

    y_test = test["label_id"].to_numpy()
    class_names = list(STANCE_LABELS)

    # --- baselines first, so the bar is fixed before anyone has a stake -----
    baseline_results = [
        B.majority_baseline(
            train["label_id"].to_numpy(), y_test,
            module="stance", split_description=split_description,
            seed=settings.seed, is_demo=demo,
        ),
        B.tfidf_logreg(
            train["text"].tolist(), train["label_id"].to_numpy(),
            test["text"].tolist(), y_test,
            module="stance", split_description=split_description,
            seed=settings.seed, is_demo=demo,
        ),
    ]

    # The zero-shot run of the *same* checkpoint we are about to fine-tune.
    # This is the baseline that actually matters here: it answers "was the
    # fine-tune worth it", which no amount of comparing against TF-IDF does.
    classifier = StanceClassifier(settings)
    zero_shot_note = _zero_shot_stance_baseline(
        classifier, pairs_of(test), y_test, class_names,
        split_description, settings, demo, baseline_results,
    )
    for baseline in baseline_results:
        log.info("baseline %s", baseline.headline())

    # --- fine-tune ---------------------------------------------------------
    classifier = StanceClassifier(settings)
    output_dir = settings.models_dir / "stance" / version
    base_model = classifier.fallback_model if demo else None
    trained = classifier.fine_tune(
        pairs_of(train), train["label_id"].to_numpy(),
        pairs_of(val), val["label_id"].to_numpy(),
        output_dir=output_dir, base_model=base_model, epochs=epochs,
    )
    if demo:
        notes.append(f"demo run used {trained.base_model} rather than {classifier.base_model}")

    predicted = classifier.predict(pairs_of(test))
    y_pred = np.array([label_index.get(p.label, -1) for p in predicted])
    confidence = np.array([p.confidence if p.confidence is not None else 0.0 for p in predicted])

    # y_score stays None: it is the positive-class probability of a *binary*
    # problem, and there is no such thing here. Passing the 4-class max would
    # produce a PR-AUC that looks like a result and means nothing.
    report = classification_report(
        y_test, y_pred, y_score=None,
        module="stance", split_description=split_description,
        class_names=class_names, seed=settings.seed, is_demo=demo,
    )
    log.info("stance %s", report.headline())

    comparison = B.compare(report, baseline_results)
    if not comparison["clears_every_baseline"]:
        notes.append(
            "the fine-tune does not cleanly separate from every baseline on this test "
            "set -- including, check this first, the zero-shot run of its own base model"
        )

    extra_sections: dict[str, str] = {
        "Zero-shot NLI baseline": zero_shot_note,
        "Calibration": (
            "**`stance_conf` is not calibrated.** `modeling/eval/calibrate.py` fits a "
            "binary isotonic or Platt curve and stance is 4-class, so there is nothing "
            "honest to fit here without building per-class one-vs-rest calibration "
            "first. Nothing in `configs/fusion.yaml` multiplies `stance_conf`, so "
            "shipping it uncalibrated costs nothing downstream -- but it must be read "
            "as a ranking signal, the way the README's table says to read `toxicity`, "
            "and never as a probability."
        ),
        "Class imbalance": _stance_balance_table(train, test),
    }

    predictions_frame = pd.DataFrame(
        {
            "text": test["text"].to_numpy(),
            "target": test["target"].to_numpy(),
            "y_true": y_test,
            "y_pred": y_pred,
            "y_score": confidence,
            "source_dataset": test["source_dataset"].to_numpy(),
            "domain": test["official_split"].to_numpy()
            if "official_split" in test
            else np.array(["fnc1"] * len(test)),
        }
    )
    artifacts = [save_predictions("stance", version, predictions_frame, settings=settings)]
    artifacts += write_report(
        module="stance", version=version, report=report, baselines=comparison,
        calibration=None, dataset_summary=dataset_summaries,
        extra_sections=extra_sections, settings=settings,
    )

    analysis = analyze(predictions_frame, module="stance", seed=settings.seed)
    artifacts.append(write_markdown(analysis, settings=settings))

    checkpoint = register(
        "stance", version, output_dir,
        metadata={
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "base_model": trained.base_model,
            "split": split_description,
            "metrics": report.as_dict(),
            "calibration": "none (4-class)",
            "is_demo": demo,
        },
    )
    notes.append(f"checkpoint {checkpoint.uri}")
    return TrainingResult("stance", version, report, artifacts=artifacts, notes=notes)


def stance_split(
    frame: pd.DataFrame, settings: ModelingSettings
) -> tuple[pd.DataFrame, Any, str, list[str]]:
    """Headline-grouped train/val and, when present, FNC-1's own test set.

    Returns ``(work, split, description, notes)``; the split indices are
    positional into ``work``, not into ``frame``.
    """
    notes: list[str] = []
    official = set(frame.get("official_split", pd.Series(dtype=str)).astype(str).unique())
    if "competition_test" in official:
        work, split = domain_holdout(
            frame,
            domain_col="official_split",
            held_out="competition_test",
            group_col="claim_id",
            purge_col="claim_id",
        )
        purged = split.dropped.get("purged_overlap_with_test", 0)
        split_description = (
            "FNC-1's own competition test set (bodies disjoint from train); "
            "train/val grouped by headline (claim_id), headlines shared with the test set "
            f"purged from train ({purged} rows); sizes = "
            f"{len(split.train)}/{len(split.val)}/{len(split.test)}; seed={settings.seed}"
        )
        notes.append("test set is the organizers' competition split, not a resampled one")
        if purged:
            notes.append(
                f"purged_overlap_with_test: {purged} train rows dropped because their "
                "headline also occurs in the competition test set"
            )
    else:
        # Only the train file is on disk. Fall back to a grouped split by body,
        # and say so -- the number is then not comparable to published FNC-1
        # results, which all use the competition test set.
        work, split = group_train_val_test(
            frame, group_col="claim_id", label_col="label",
            seed=settings.seed, dedupe=False,
        )
        split_description = split.describe()
        notes.append(
            "competition_test_stances.csv was not on disk; this is a resampled "
            "headline-grouped split and is NOT comparable to published FNC-1 numbers"
        )
    return work, split, split_description, notes


def _zero_shot_stance_baseline(
    classifier,
    test_pairs: list[tuple[str, str]],
    y_test: np.ndarray,
    class_names: list[str],
    split_description: str,
    settings: ModelingSettings,
    demo: bool,
    baseline_results: list[Any],
) -> str:
    """Score the test set with the un-fine-tuned NLI head and record it.

    Appends to ``baseline_results`` in place when it succeeds, so the fine-tune
    has to clear it like any other baseline. Returns the markdown section.
    """
    from modeling.eval.baselines import BaselineResult
    from modeling.text.stance_clf import STANCE_LABELS

    if not bool(classifier.config.get("zero_shot", {}).get("enabled", True)):
        return "Disabled in `configs/models.yaml` (`stance.zero_shot.enabled: false`)."
    if not classifier.load_zero_shot():
        return (
            "**Not run.** The configured base model could not be loaded, or its head is "
            "not a three-way entailment/neutral/contradiction head, so it cannot be read "
            "as stance. Without this the report cannot say whether fine-tuning was worth "
            "it; re-run with the base model cached (`modeling warm-cache`)."
        )

    label_index = {label: i for i, label in enumerate(STANCE_LABELS)}
    predicted = classifier.predict(test_pairs)
    y_pred = np.array([label_index.get(p.label, -1) for p in predicted])
    report = classification_report(
        y_test, y_pred, y_score=None,
        module="stance", split_description=split_description,
        class_names=class_names, seed=settings.seed, is_demo=demo,
    )
    baseline_results.append(
        BaselineResult(
            name="zero-shot NLI (same base model)",
            report=report,
            # No positive-class score exists for a 4-class head; the softmax
            # maximum is a confidence, not a probability of a class of interest.
            scores=np.array([p.confidence or 0.0 for p in predicted]),
            predictions=y_pred,
        )
    )
    assigned = int((np.array([p.reason for p in predicted]) == "nli_neutral_below_cut").sum())
    return (
        f"The base model **{classifier.base_model}**, run with its own NLI head and no "
        f"fine-tuning: macro-F1 **{report.macro_f1.value:.4f}**.\n\n"
        "entailment -> support, contradiction -> deny, neutral -> discuss. The NLI head "
        "has no `unrelated` class, because `unrelated` is a retrieval judgement rather "
        "than an entailment one, so it is assigned by thresholding the directional mass "
        f"(`zero_shot.unrelated_below`); that fired on {assigned} of {len(predicted)} "
        "test pairs. This is the number the fine-tune has to beat to justify its cost."
    )


def _stance_balance_table(train: pd.DataFrame, test: pd.DataFrame) -> str:
    """Label balance per split. FNC-1's imbalance is the headline caveat."""
    rows = ["| label | train | test |", "|---|---|---|"]
    train_counts = train["label"].value_counts()
    test_counts = test["label"].value_counts()
    for label in sorted(set(train_counts.index) | set(test_counts.index)):
        n_train = int(train_counts.get(label, 0))
        n_test = int(test_counts.get(label, 0))
        rows.append(
            f"| `{label}` | {n_train} ({100 * n_train / max(1, len(train)):.1f}%) "
            f"| {n_test} ({100 * n_test / max(1, len(test)):.1f}%) |"
        )
    rows.append("")
    rows.append(
        "FNC-1 is ~73% `unrelated` and ~1.7% `deny`. A model that never predicts `deny` "
        "still scores high accuracy, which is why accuracy is absent from this report. "
        "`deny` is also the class the product cares most about -- a post pushing back on "
        "a circulating claim -- so read its per-class recall first."
    )
    return "\n".join(rows)

