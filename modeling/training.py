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
from modeling.eval.calibrate import Calibrator
from modeling.eval.error_analysis import analyze, write_markdown
from modeling.eval.metrics import (
    ClassificationReport,
    classification_report,
    threshold_at_precision,
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

    The split is the organizers' own protocol rather than a fresh random one:
    FNC-1 ships a competition test set whose article bodies are disjoint from
    train, which is both the strongest available group key and the published
    comparison point. ``domain_holdout`` reproduces it exactly, and does not
    dedupe -- in FNC-1 one body legitimately appears against many headlines, so
    near-duplicate collapse on ``text`` would delete most of the corpus.
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

    official = set(frame.get("official_split", pd.Series(dtype=str)).astype(str).unique())
    if "competition_test" in official:
        work, split = domain_holdout(
            frame,
            domain_col="official_split",
            held_out="competition_test",
            group_col="body_id",
        )
        split_description = (
            "FNC-1's own competition test set (bodies disjoint from train); "
            f"train/val grouped by body_id; sizes = "
            f"{len(split.train)}/{len(split.val)}/{len(split.test)}; seed={settings.seed}"
        )
        notes.append("test set is the organizers' competition split, not a resampled one")
    else:
        # Only the train file is on disk. Fall back to a grouped split by body,
        # and say so -- the number is then not comparable to published FNC-1
        # results, which all use the competition test set.
        work, split = group_train_val_test(
            frame, group_col="body_id", label_col="label",
            seed=settings.seed, dedupe=False,
        )
        split_description = split.describe()
        notes.append(
            "competition_test_stances.csv was not on disk; this is a resampled "
            "body-grouped split and is NOT comparable to published FNC-1 numbers"
        )

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


# ---------------------------------------------------------------------------
# deepfake
# ---------------------------------------------------------------------------
@trainer("deepfake")
def _train_deepfake(
    settings: ModelingSettings,
    *,
    data_path: Path | None,
    demo: bool,
    epochs: int | None,
) -> TrainingResult:
    """Install a pre-existing detector and evaluate it on DFDC. No training.

    This is registered as a "trainer" because it is the module's acquire-and-
    evaluate entry point, not because anything is fitted -- ``epochs`` is
    ignored and the report says so in as many words.

    **Why evaluating on DFDC is honest here, when training on it would not be.**
    The committed DFDC copy is pre-extracted face crops whose ``original``
    field was dropped, so a fake cannot be tied to the real clip it came from.
    Training on that risks identity leakage, because DFDC swaps faces between
    actors recorded in the same sessions -- a model could memorise a face and be
    scored on that same face from the other side of the split. But nothing is
    fitted here, and a set used purely for testing has no internal boundary to
    leak across. The full argument is in ``modeling/datasets/dfdc.py``.

    **The grain is the video, not the frame.** 3,745 crops come from 381 clips.
    Reporting per frame would inflate n roughly tenfold and quietly weight long
    clips more heavily; the frame scores are aggregated to one verdict per
    video with the same top-k mean used in production.
    """
    from modeling.accounts.bot_clf import _operating_point_section
    from modeling.media.deepfake_clf import DeepfakeScorer, aggregate_top_k
    from modeling.registry import register

    version = str(module_config("deepfake").get("version", "v0.0.0-unset"))
    notes: list[str] = ["NO TRAINING WAS PERFORMED: this is a pre-existing detector."]
    if epochs is not None:
        notes.append("--epochs is ignored; nothing is fitted here")

    dataset = get_dataset("dfdc")
    if not dataset.available(data_path, demo=demo):
        raise DatasetUnavailable(dataset.info.instructions(dataset.resolve_path(data_path, demo)))
    loaded = dataset.load(data_path, demo=demo)
    frame = loaded.frame
    dataset_summaries = {loaded.info.key: loaded.summary()}

    if "path" not in frame.columns:
        return TrainingResult(
            "deepfake", version, None, skipped=True,
            notes=notes + [
                "the DFDC layout on disk carries no per-frame image paths, so a "
                "pretrained detector cannot be pointed at anything"
            ],
        )

    scorer = DeepfakeScorer(settings)
    output_dir = settings.models_dir / "deepfake" / version
    try:
        installed = scorer.install_pretrained(output_dir)
    except Exception as exc:
        return TrainingResult(
            "deepfake", version, None, skipped=True,
            notes=notes + [f"could not install a pre-existing detector: {exc}"],
        )
    if not scorer.load(output_dir):
        return TrainingResult(
            "deepfake", version, None, skipped=True,
            notes=notes + [f"installed {installed} but could not load it back"],
        )
    notes.append(f"detector: {installed} (no fine-tuning)")

    # --- score every crop, then collapse to one verdict per video ---------
    frame_scores = _score_dfdc_crops(scorer, frame)
    scored = frame.assign(frame_score=frame_scores)
    usable = scored.loc[scored["frame_score"].notna()]
    unreadable = len(scored) - len(usable)
    if unreadable:
        notes.append(f"{unreadable} crops could not be read and are excluded")
    if usable.empty:
        return TrainingResult(
            "deepfake", version, None, skipped=True,
            notes=notes + ["no crop could be scored; refusing to report a metric"],
        )

    config = module_config("deepfake")
    top_k = int(config.get("aggregate_top_k", 5))
    per_video = (
        usable.groupby("source_video")
        .apply(
            lambda part: pd.Series(
                {
                    "y_true": int(part["label"].iloc[0]),
                    "raw_score": float(aggregate_top_k(part["frame_score"].tolist(), top_k)),
                    "n_frames": int(len(part)),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per_video["y_true"] = per_video["y_true"].astype(int)

    # --- the threshold and the calibrator need their own data --------------
    #
    # Nothing is *trained* here, but a decision threshold and a calibration
    # curve are both *fitted*, and fitting them on the rows we then report is
    # tuning on test however little else is learned. The raw softmax is also
    # useless as a threshold on this data: the detector calls almost every
    # video manipulated at 0.5, which scores exactly the majority baseline.
    #
    # So DFDC is split in half by source_video: one half fits the calibrator
    # and picks the operating point, the other half is reported. Grouping is by
    # video for the usual reason -- frames of one clip on both sides is *the*
    # cause of implausible deepfake numbers.
    per_video, split = group_train_val_test(
        per_video,
        group_col="source_video",
        label_col="y_true",
        test_size=0.5,
        val_size=0.0,
        seed=settings.seed,
        dedupe=False,
    )
    fit_rows = per_video.iloc[split.train]
    test_rows = per_video.iloc[split.test]
    if fit_rows["y_true"].nunique() < 2 or test_rows["y_true"].nunique() < 2:
        return TrainingResult(
            "deepfake", version, None, skipped=True,
            notes=notes + [
                "splitting DFDC in half leaves one side single-class; cannot fit a "
                "threshold and report on held-out video without inventing one"
            ],
        )

    calibrator = Calibrator(str(config.get("calibration", "isotonic")))
    calibration = calibrator.fit(
        fit_rows["raw_score"].to_numpy(), fit_rows["y_true"].to_numpy()
    )
    log.info(calibration.summary())

    fit_calibrated = calibrator.transform(fit_rows["raw_score"].to_numpy())
    operating_point = threshold_at_precision(
        fit_rows["y_true"].to_numpy(),
        fit_calibrated,
        float(config.get("precision_target", 0.85)),
    )
    threshold = float(operating_point["threshold"])
    log.info(
        "operating point from the held-apart half: threshold %.3f -> precision %.3f, "
        "recall %.3f (target %s)",
        threshold,
        operating_point["precision"],
        operating_point["recall"],
        "met" if operating_point["target_met"] else "NOT MET",
    )
    if not operating_point["target_met"]:
        notes.append(
            f"no threshold reaches the {config.get('precision_target', 0.85)} precision "
            "target on the calibration half; the best achievable point is reported"
        )

    split_description = (
        f"DFDC as a pure test set at video grain, split in half by source_video: "
        f"{len(fit_rows)} videos fit the calibrator and the operating point, "
        f"{len(test_rows)} held-out videos are reported "
        f"(from {len(usable)} face crops; seed={settings.seed}). Nothing was trained."
    )
    notes.append(split_description)

    y_true = test_rows["y_true"].to_numpy().astype(int)
    y_score = calibrator.transform(test_rows["raw_score"].to_numpy())
    y_pred = (y_score >= threshold).astype(int)
    per_video = test_rows

    # --- baselines. The majority baseline is the one that matters here: ----
    # DFDC-as-shipped is 80% fake, so "always fake" scores well and any
    # detector that cannot clear it has told us nothing.
    baseline_results = [
        B.majority_baseline(
            fit_rows["y_true"].to_numpy(), y_true,
            module="deepfake", split_description=split_description,
            seed=settings.seed, is_demo=demo,
        ),
    ]

    report = classification_report(
        y_true, y_pred, y_score=y_score,
        module="deepfake", split_description=split_description,
        class_names=["authentic", "manipulated"], seed=settings.seed, is_demo=demo,
    )
    log.info("deepfake %s", report.headline())

    comparison = B.compare(report, baseline_results)
    if not comparison["clears_every_baseline"]:
        notes.append(
            "the pretrained detector does not clear the majority-class baseline on "
            "DFDC. Read the transfer section before mounting it."
        )

    extra_sections = {
        "Operating point": _operating_point_section(operating_point, config),
        "This is cross-dataset transfer, and that is the point": _deepfake_transfer_note(
            installed, report, per_video, threshold
        ),
        "Why no training happened": (
            "FaceForensics++ is 17 GB behind a signed agreement and is not on disk. The "
            "DFDC copy that is on disk is pre-extracted face crops with the `original` "
            "field dropped, so a fake cannot be tied to the real clip it came from; "
            "training on it risks identity leakage, because DFDC swaps faces between "
            "actors recorded in the same sessions. Mounting someone else's trained "
            "detector and testing it here sidesteps both problems: nothing is fitted, "
            "so there is no split to leak across, and the module's stated role -- a "
            "cross-dataset generalisation check -- is exactly what this measures."
        ),
    }

    predictions_frame = pd.DataFrame(
        {
            "text": per_video["source_video"].astype(str).to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
            "source_dataset": np.array(["dfdc"] * len(per_video)),
            "domain": np.array(["crops"] * len(per_video)),
        }
    )
    artifacts = [save_predictions("deepfake", version, predictions_frame, settings=settings)]
    artifacts += write_report(
        module="deepfake", version=version, report=report, baselines=comparison,
        calibration=calibration, dataset_summary=dataset_summaries,
        extra_sections=extra_sections, settings=settings,
    )

    # The calibrator and threshold are part of the mounted model: without them
    # the raw softmax goes into a column Phase 4 thresholds at 0.50, and the
    # detector's answer is "manipulated" for almost everything.
    import json as _json

    (output_dir / "calibrator.json").write_text(
        _json.dumps(calibrator.state()), encoding="utf-8"
    )
    (output_dir / "model.json").write_text(
        _json.dumps(
            {
                "hf_model": installed,
                "threshold": threshold,
                "methods": [],
                "fine_tuned": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    checkpoint = register(
        "deepfake", version, output_dir,
        metadata={
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "hf_model": installed,
            "fine_tuned": False,
            "threshold": threshold,
            "calibration": calibration.as_dict(),
            "split": split_description,
            "metrics": report.as_dict(),
            "is_demo": demo,
        },
    )
    notes.append(f"checkpoint {checkpoint.uri}")
    return TrainingResult("deepfake", version, report, artifacts=artifacts, notes=notes)


def _score_dfdc_crops(scorer, frame: pd.DataFrame) -> list[float | None]:
    """P(manipulated) per committed crop.

    The crops are already extracted faces, so they go straight to the model
    rather than back through face detection -- re-detecting a face inside a
    face crop is just a second chance to fail.
    """
    from PIL import Image

    paths = frame["path"].astype(str).tolist()
    out: list[float | None] = [None] * len(paths)
    batch_size = max(1, int(scorer.batch_size))
    for start in range(0, len(paths), batch_size):
        chunk_positions = list(range(start, min(start + batch_size, len(paths))))
        images: list[Any] = []
        kept: list[int] = []
        for position in chunk_positions:
            try:
                with Image.open(paths[position]) as handle:
                    images.append(np.array(handle.convert("RGB")))
                kept.append(position)
            except Exception as exc:  # unreadable crop -> null, with a reason
                log.warning("could not read %s: %s", paths[position], exc)
        if not images:
            continue
        scores = scorer.score_crops(images)
        for position, score in zip(kept, scores, strict=False):
            out[position] = float(score)
        if start and start % (batch_size * 20) == 0:
            log.info("scored %d/%d crops", start, len(paths))
    return out


def _deepfake_transfer_note(
    installed: str, report: ClassificationReport, per_video: pd.DataFrame, threshold: float
) -> str:
    """State the transfer caveat with the numbers attached."""
    n_fake = int((per_video["y_true"] == 1).sum())
    n_real = int((per_video["y_true"] == 0).sum())
    return (
        f"**{installed}** was trained by someone else, on someone else's data, and is "
        f"evaluated here on DFDC with no adaptation. That is cross-dataset transfer, "
        "which is the single hardest regime in deepfake detection: published detectors "
        "routinely lose most of their headline accuracy when the manipulation method, "
        "codec and capture conditions change. A weak number here is the expected "
        "result, not a bug in this pipeline.\n\n"
        f"Test set: {len(per_video)} videos, {n_fake} manipulated / {n_real} authentic "
        f"({100 * n_fake / max(1, len(per_video)):.0f}% positive), decision threshold "
        f"{threshold}. Macro-F1 **{report.macro_f1.value:.4f}**.\n\n"
        "**How to read this.** If the detector does not clear the majority-class "
        "baseline, it has measured nothing and the module should stay off rather than "
        "ship a `deepfake_prob` that is noise -- `configs/fusion.yaml` thresholds that "
        "column at 0.50 and a noisy value propagates into the risk score. The detector "
        "also targets a different manipulation family than DFDC's face swaps; the "
        "FF++-trained alternative named in `configs/models.yaml` is the closer domain "
        "match, at the cost of `trust_remote_code`."
    )
