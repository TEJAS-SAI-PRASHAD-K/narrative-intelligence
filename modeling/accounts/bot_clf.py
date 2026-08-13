"""Bot classifier: XGBoost primary, RandomForest comparison, SHAP explanations.

**Grouped 5-fold CV, grouped by campaign where the dataset has one.** Cresci-2017
ships seven distinct botnets, each running one content template. Grouping by
account puts siblings from the same botnet in train and test, and the model
reports near-perfect F1 for having memorized seven signatures. Grouping by
campaign is strictly stronger and is what ``modeling/datasets/cresci.py``
returns.

**The operating point is not 0.5.** In this product a false "bot" flag is an
accusation about a person, and it is costlier than a miss. The threshold is
chosen from the precision-recall curve at a stated precision target (0.85 by
default) and the recall that costs is reported next to it.

**SHAP, per account, into the contract.** ``author_scores.bot_top_features``
carries the top five signed contributions for that specific account, and the
dashboard's "why is this flagged" panel reads it directly. A global importance
ranking would not answer the question the user is actually asking.

**Domain shift, stated up front.** TwiBot-22 and Cresci-2017 are Twitter. This
project's corpus is Mastodon, Reddit and YouTube. Cross-platform transfer is
**unmeasured and should be assumed degraded** until fifty Mastodon accounts are
hand-labelled as a sanity check; the model card says so and
``modeling/eval/error_analysis.py`` has the taxonomy ready for when they are.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.config import ModelingSettings, get_settings, module_config
from modeling.datasets import DatasetUnavailable, get_dataset
from modeling.datasets.splits import grouped_kfold
from modeling.eval import baselines as B
from modeling.eval.calibrate import Calibrator
from modeling.eval.metrics import (
    classification_report,
    fold_summary,
    threshold_at_precision,
)

log = logging.getLogger(__name__)

LABEL_NAMES = ["human", "bot"]


@dataclass
class BotModel:
    """A trained bot classifier plus its calibrator, threshold and feature names."""

    estimator: Any
    calibrator: Calibrator
    feature_names: list[str]
    threshold: float
    tiers_used: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        raw = self.estimator.predict_proba(matrix)[:, 1]
        return self.calibrator.transform(raw)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "estimator.pkl").open("wb") as fh:
            pickle.dump(self.estimator, fh)
        (directory / "model.json").write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "threshold": self.threshold,
                    "tiers_used": self.tiers_used,
                    "calibrator": self.calibrator.state(),
                    **self.metadata,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> BotModel | None:
        directory = Path(directory)
        estimator_path = directory / "estimator.pkl"
        meta_path = directory / "model.json"
        if not (estimator_path.exists() and meta_path.exists()):
            return None
        try:
            with estimator_path.open("rb") as fh:
                estimator = pickle.load(fh)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            log.warning("could not load bot checkpoint at %s: %s", directory, exc)
            return None
        return cls(
            estimator=estimator,
            calibrator=Calibrator.from_state(meta["calibrator"]),
            feature_names=meta["feature_names"],
            threshold=float(meta["threshold"]),
            tiers_used=meta.get("tiers_used", ["universal"]),
            metadata=meta,
        )


def build_estimator(kind: str, config: dict[str, Any], seed: int):
    """XGBoost primary, RandomForest comparison."""
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            log.warning("xgboost is not installed; using RandomForest as the primary model")
            return build_estimator("random_forest", config, seed)
        params = dict(config.get("xgboost") or {})
        return XGBClassifier(
            random_state=seed,
            n_jobs=1,  # determinism over throughput; the labelled set is small
            eval_metric="logloss",
            tree_method="hist",
            **params,
        )
    from sklearn.ensemble import RandomForestClassifier

    params = dict(config.get("random_forest") or {})
    return RandomForestClassifier(
        random_state=seed, n_jobs=1, class_weight="balanced", **params
    )


def shap_contributions(
    estimator: Any, matrix: np.ndarray, feature_names: list[str], top_k: int = 5
) -> list[list[dict[str, Any]]]:
    """Per-row top-k signed SHAP contributions.

    Falls back to global gain importance if SHAP is unavailable, and says so --
    a global ranking answers a different question ("what does the model use in
    general") from the one the UI asks ("why this account"), and conflating them
    would be a quiet lie in a panel labelled "why".
    """
    try:
        import shap
    except ImportError:
        log.warning(
            "shap is not installed; bot_top_features will carry GLOBAL importances, which "
            "do not explain an individual account. Install the 'modeling' extra."
        )
        return _global_fallback(estimator, matrix, feature_names, top_k)

    try:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(matrix)
        if isinstance(values, list):  # older APIs return one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:  # (n, features, classes)
            values = values[:, :, 1]
    except Exception as exc:  # pragma: no cover - explainer incompatibility
        log.warning("SHAP failed (%s); falling back to global importances", exc)
        return _global_fallback(estimator, matrix, feature_names, top_k)

    out: list[list[dict[str, Any]]] = []
    for row in values:
        order = np.argsort(-np.abs(row))[:top_k]
        out.append(
            [
                {"name": feature_names[i], "contribution": round(float(row[i]), 5)}
                for i in order
            ]
        )
    return out


def _global_fallback(
    estimator: Any, matrix: np.ndarray, feature_names: list[str], top_k: int
) -> list[list[dict[str, Any]]]:
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return [[] for _ in range(len(matrix))]
    order = np.argsort(-np.asarray(importances))[:top_k]
    shared = [
        {
            "name": f"{feature_names[i]} (global)",
            "contribution": round(float(importances[i]), 5),
        }
        for i in order
    ]
    return [list(shared) for _ in range(len(matrix))]


def train_bot_classifier(
    settings: ModelingSettings | None = None,
    *,
    data_path: Path | None = None,
    demo: bool = False,
):
    """Load a bot benchmark, run grouped CV, calibrate, report."""
    from modeling.eval.error_analysis import analyze, write_markdown
    from modeling.eval.report import save_predictions, write_report
    from modeling.registry import register
    from modeling.training import TrainingResult

    settings = settings or get_settings()
    config = module_config("bot")
    version = str(config.get("version", "v0.0.0-unset"))
    notes: list[str] = []

    frame, dataset_key, group_col, summary = _load_bot_benchmark(data_path, demo)
    if frame is None:
        raise DatasetUnavailable(
            "neither Cresci-2017 nor TwiBot-22 is available. Both are behind a request "
            "form; see `modeling datasets`, or use --demo."
        )

    # These benchmarks ship account metadata, not post histories, so the
    # computable tier is the social-graph one. The corpus-side tier is decided
    # at scoring time by features.available_tiers, and the intersection is what
    # the scorer actually uses.
    matrix, feature_names = _benchmark_matrix(frame)
    labels = frame["label"].to_numpy().astype(int)
    tiers_used = ["social_graph_benchmark"]

    work, folds = grouped_kfold(
        frame.assign(**{f"f{i}": matrix[:, i] for i in range(matrix.shape[1])}),
        group_col=group_col,
        label_col="label",
        n_splits=int(config.get("n_folds", 5)),
        seed=settings.seed,
    )
    split_description = folds[0].describe().replace("train/val/test", "train/test")

    seed = settings.seed
    fold_reports = []
    oof_scores = np.zeros(len(work), dtype=float)
    oof_mask = np.zeros(len(work), dtype=bool)

    for fold in folds:
        estimator = build_estimator(str(config.get("primary", "xgboost")), config, seed)
        estimator.fit(matrix[fold.train], labels[fold.train])
        scores = estimator.predict_proba(matrix[fold.test])[:, 1]
        oof_scores[fold.test] = scores
        oof_mask[fold.test] = True
        fold_reports.append(
            classification_report(
                labels[fold.test],
                (scores >= 0.5).astype(int),
                y_score=scores,
                module="bot",
                split_description=f"fold {fold.notes['fold']}, {split_description}",
                class_names=LABEL_NAMES,
                seed=seed,
                is_demo=demo,
            )
        )

    cv = fold_summary(fold_reports)
    log.info("bot 5-fold grouped CV: %s", cv)

    # Calibrate on the out-of-fold predictions: every score there was produced
    # by a model that did not see that row, which is the closest thing to a
    # held-out validation set when the labelled set is too small to carve one.
    calibrator = Calibrator(str(config.get("calibration", "isotonic")))
    calibration = calibrator.fit(oof_scores[oof_mask], labels[oof_mask])
    log.info(calibration.summary())
    calibrated = calibrator.transform(oof_scores)

    operating_point = threshold_at_precision(
        labels[oof_mask], calibrated[oof_mask], float(config.get("precision_target", 0.85))
    )
    log.info(
        "operating point: threshold %.3f -> precision %.3f, recall %.3f (target %s)",
        operating_point["threshold"],
        operating_point["precision"],
        operating_point["recall"],
        "met" if operating_point["target_met"] else "NOT MET",
    )
    if not operating_point["target_met"]:
        notes.append(
            f"no threshold reaches the {config.get('precision_target')} precision target; "
            "the best achievable point is reported instead"
        )

    predictions = (calibrated >= operating_point["threshold"]).astype(int)
    report = classification_report(
        labels,
        predictions,
        y_score=calibrated,
        module="bot",
        split_description=f"out-of-fold, {split_description}",
        class_names=LABEL_NAMES,
        seed=seed,
        is_demo=demo,
    )

    # --- baselines ---------------------------------------------------------
    baseline_results = [
        B.majority_baseline(
            labels, labels, module="bot", split_description=split_description,
            seed=seed, is_demo=demo,
        )
    ]
    if "follower_following_ratio" in feature_names:
        first = folds[0]
        baseline_results.append(
            B.single_feature_logreg(
                matrix[first.train], labels[first.train],
                matrix[first.test], labels[first.test],
                feature_names, "follower_following_ratio",
                module="bot", split_description=split_description, seed=seed, is_demo=demo,
            )
        )
    comparison = B.compare(report, baseline_results)

    # --- comparison estimator ---------------------------------------------
    comparison_kind = str(config.get("comparison", "random_forest"))
    rf = build_estimator(comparison_kind, config, seed)
    first = folds[0]
    rf.fit(matrix[first.train], labels[first.train])
    rf_scores = rf.predict_proba(matrix[first.test])[:, 1]
    rf_report = classification_report(
        labels[first.test], (rf_scores >= 0.5).astype(int), y_score=rf_scores,
        module="bot", split_description=f"fold 0, {split_description}",
        class_names=LABEL_NAMES, seed=seed, is_demo=demo,
    )

    # --- final model on everything ----------------------------------------
    final = build_estimator(str(config.get("primary", "xgboost")), config, seed)
    final.fit(matrix, labels)
    top_k = int(config.get("shap_top_k", 5))
    contributions = shap_contributions(final, matrix, feature_names, top_k)

    model = BotModel(
        estimator=final,
        calibrator=calibrator,
        feature_names=feature_names,
        threshold=float(operating_point["threshold"]),
        tiers_used=tiers_used,
        metadata={
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_key,
            "group_col": group_col,
            "cv": cv,
            "operating_point": operating_point,
            "calibration": calibration.as_dict(),
            "is_demo": demo,
        },
    )
    output_dir = settings.models_dir / "bot" / version
    model.save(output_dir)

    predictions_frame = pd.DataFrame(
        {
            "author_id": frame["account_id"].to_numpy(),
            "y_true": labels,
            "y_pred": predictions,
            "y_score": calibrated,
            "followers": frame.get("followers", pd.Series(np.nan, index=frame.index)).to_numpy(),
            "following": frame.get("following", pd.Series(np.nan, index=frame.index)).to_numpy(),
            "post_count": frame.get("post_count", pd.Series(np.nan, index=frame.index)).to_numpy(),
        }
    )
    artifacts = [save_predictions("bot", version, predictions_frame, settings=settings)]
    artifacts += write_report(
        module="bot",
        version=version,
        report=report,
        baselines=comparison,
        calibration=calibration,
        dataset_summary=summary,
        extra_sections={
            "Cross-validation": _cv_section(cv, split_description),
            "Operating point": _operating_point_section(operating_point, config),
            f"Comparison estimator ({comparison_kind})": (
                f"{rf_report.headline()}\n\nReported so the choice of XGBoost as primary is "
                "an observation rather than an assumption."
            ),
            "Feature explanations": _shap_section(contributions, feature_names, top_k),
            "Cross-platform transfer": _transfer_section(dataset_key),
        },
        settings=settings,
    )

    analysis = analyze(predictions_frame, module="bot", seed=seed)
    artifacts.append(write_markdown(analysis, settings=settings))

    checkpoint = register(
        "bot", version, output_dir,
        metadata={
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_key,
            "split": split_description,
            "metrics": report.as_dict(),
            "is_demo": demo,
        },
    )
    notes.append(f"checkpoint {checkpoint.uri}")
    return TrainingResult("bot", version, report, artifacts=artifacts, notes=notes)


def _load_bot_benchmark(data_path: Path | None, demo: bool):
    """Prefer Cresci (it has campaign structure); fall back to TwiBot."""
    for key in ("cresci", "twibot"):
        dataset = get_dataset(key)
        if not dataset.available(data_path, demo=demo):
            continue
        loaded = dataset.load(data_path, demo=demo)
        return loaded.frame, key, dataset.group_col, {key: loaded.summary()}
    return None, "", "", {}


def _benchmark_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Features computable from a bot benchmark's account metadata.

    These benchmarks ship profile rows, not post histories, so the temporal and
    content features in ``accounts/features.py`` cannot be computed here. What
    *is* shared with this project's Mastodon corpus is the social-graph tier,
    and that is deliberately all this uses -- a feature the corpus cannot supply
    is a feature that will not exist at inference time.
    """
    now = pd.Timestamp.now(tz="UTC")
    followers = pd.to_numeric(frame.get("followers"), errors="coerce")
    following = pd.to_numeric(frame.get("following"), errors="coerce")
    posts = pd.to_numeric(frame.get("post_count"), errors="coerce")
    created = pd.to_datetime(frame.get("created_at"), errors="coerce", utc=True)
    age_days = (now - created).dt.total_seconds() / 86400

    columns = {
        "followers": followers.fillna(0.0),
        "following": following.fillna(0.0),
        "follower_following_ratio": followers.fillna(0.0) / (following.fillna(0.0) + 1.0),
        "post_count": posts.fillna(0.0),
        "posts_per_account_day": posts.fillna(0.0) / age_days.clip(lower=1.0).fillna(1.0),
        "account_age_days": age_days.fillna(0.0),
        # Indicators, so an absent value never masquerades as a measured zero.
        "followers_is_missing": followers.isna().astype(float),
        "account_age_is_missing": age_days.isna().astype(float),
    }
    names = list(columns)
    matrix = np.column_stack([columns[name].to_numpy(dtype=float) for name in names])
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), names


def _cv_section(cv: dict[str, Any], split_description: str) -> str:
    lines = [
        f"5-fold grouped cross-validation, {split_description}.",
        "",
        f"- macro F1: **{cv.get('macro_f1_mean')} ± {cv.get('macro_f1_std')}** across folds",
        f"- per fold: {cv.get('macro_f1_per_fold')}",
        f"- worst fold: {cv.get('worst_fold_macro_f1')}",
        "",
        "The per-fold numbers are given alongside the mean deliberately: a mean that "
        "hides one catastrophic fold is worse than no summary at all.",
    ]
    if cv.get("pr_auc_mean") is not None:
        lines.insert(3, f"- PR-AUC: **{cv['pr_auc_mean']} ± {cv['pr_auc_std']}**")
    return "\n".join(lines)


def _operating_point_section(point: dict[str, float], config: dict[str, Any]) -> str:
    target = config.get("precision_target", 0.85)
    return "\n".join(
        [
            f"Threshold **{point['threshold']:.3f}**, chosen from the precision-recall curve "
            f"at a precision target of {target} — not 0.5.",
            "",
            f"- precision at this threshold: {point['precision']:.3f}",
            f"- recall at this threshold: {point['recall']:.3f}",
            "",
            "**Why a precision target.** A false 'bot' flag is an accusation about a person. "
            "In this product that costs more than a miss, so the operating point buys "
            "precision with recall, and the recall it costs is stated rather than buried.",
        ]
        + (
            []
            if point["target_met"]
            else [
                "",
                f"> **The {target} target was not reachable on this data.** The best available "
                "point is reported above; treat the model as not yet deployable at the "
                "intended precision.",
            ]
        )
    )


def _shap_section(
    contributions: list[list[dict[str, Any]]], feature_names: list[str], top_k: int
) -> str:
    from collections import Counter

    counter: Counter[str] = Counter()
    for row in contributions:
        for entry in row:
            counter[entry["name"]] += 1
    lines = [
        f"Per-account SHAP contributions, top {top_k} per account, written into "
        "`author_scores.bot_top_features`. The dashboard's \"why is this account "
        "flagged\" panel reads that column directly.",
        "",
        "How often each feature appears in an account's top-5:",
        "",
        "| feature | accounts |",
        "|---|---|",
    ]
    for name, count in counter.most_common(12):
        lines.append(f"| `{name}` | {count} |")
    return "\n".join(lines)


def _transfer_section(dataset_key: str) -> str:
    return "\n".join(
        [
            f"Trained on **{dataset_key}**, which is Twitter data. This project's corpus is "
            "Mastodon, Reddit and YouTube.",
            "",
            "**Cross-platform transfer is unmeasured and should be assumed degraded.** The "
            "feature set is deliberately restricted to the social-graph tier that Mastodon "
            "also supplies, which bounds the shift but does not remove it: follower counts "
            "mean different things on a follow-graph platform and on a federated one, and "
            "Reddit has no follower concept at all.",
            "",
            "To measure it: hand-label 50 Mastodon accounts, score them with this model, and "
            "report the result here. Until that exists, `bot_prob` on non-Twitter accounts is "
            "a research signal and not a finding.",
        ]
    )
