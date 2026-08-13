"""Eval artifacts: ``artifacts/eval/<module>/<version>/``.

Each run writes ``metrics.json``, ``report.md``, ``confusion.png`` and
``reliability.png``, plus the raw predictions. The predictions are what makes
``modeling report`` able to regenerate every chart and table **without
retraining** -- which is the difference between a report a grader can reproduce
in seconds and one that needs a GPU.

Every artifact carries the run fingerprint: seed, device, library versions and
the input manifest hash. A metric with no provenance cannot be defended when it
disagrees with a rerun.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.config import ModelingSettings, get_settings, run_fingerprint
from modeling.eval.calibrate import CalibrationResult
from modeling.eval.metrics import ClassificationReport

log = logging.getLogger(__name__)


def eval_dir(module: str, version: str, settings: ModelingSettings | None = None) -> Path:
    path = (settings or get_settings()).eval_dir / module / version
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_predictions(
    module: str,
    version: str,
    frame: pd.DataFrame,
    *,
    settings: ModelingSettings | None = None,
) -> Path:
    """Persist per-row predictions so reports regenerate without retraining."""
    path = eval_dir(module, version, settings) / "predictions.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    log.info("saved %d predictions -> %s", len(frame), path)
    return path


def load_predictions(
    module: str, version: str, settings: ModelingSettings | None = None
) -> pd.DataFrame | None:
    path = eval_dir(module, version, settings) / "predictions.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_report(
    *,
    module: str,
    version: str,
    report: ClassificationReport,
    baselines: dict[str, Any] | None = None,
    calibration: CalibrationResult | None = None,
    dataset_summary: dict[str, Any] | None = None,
    extra_sections: dict[str, str] | None = None,
    settings: ModelingSettings | None = None,
) -> list[Path]:
    """Write metrics.json + report.md (+ charts when matplotlib is available)."""
    settings = settings or get_settings()
    target = eval_dir(module, version, settings)
    written: list[Path] = []

    payload: dict[str, Any] = {
        "module": module,
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": run_fingerprint(),
        "metrics": report.as_dict(),
        "baselines": baselines,
        "calibration": calibration.as_dict() if calibration else None,
        "dataset": dataset_summary,
    }
    metrics_path = target / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    written.append(metrics_path)

    report_path = target / "report.md"
    report_path.write_text(
        _render_markdown(module, version, report, baselines, calibration, dataset_summary,
                         extra_sections),
        encoding="utf-8",
    )
    written.append(report_path)

    written.extend(_write_charts(target, report, calibration))
    return written


def _render_markdown(
    module: str,
    version: str,
    report: ClassificationReport,
    baselines: dict[str, Any] | None,
    calibration: CalibrationResult | None,
    dataset_summary: dict[str, Any] | None,
    extra_sections: dict[str, str] | None,
) -> str:
    lines: list[str] = []
    lines.append(f"# {module} — evaluation report ({version})")
    lines.append("")

    if report.is_demo:
        lines.append(
            "> **These numbers are not a result.** They were computed on the committed "
            "demo fixture, which is shape-faithful and value-meaningless. They demonstrate "
            "that the training and evaluation path executes end to end. Reproduce with a "
            "real benchmark on disk before citing anything below."
        )
        lines.append("")

    lines.append(f"**Headline:** {report.headline()}")
    lines.append("")
    lines.append(f"- Split: `{report.split_description}`")
    lines.append(f"- Test rows: {report.n_test}")
    lines.append(f"- Positive rate in test: {report.positive_rate:.3f}")
    lines.append("")

    if dataset_summary:
        lines.append("## Training data")
        lines.append("")
        for key, value in dataset_summary.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "Accuracy is deliberately absent. On an imbalanced target it rewards predicting "
        "the majority class, and PR-AUC is the number that degrades when the model starts "
        "crying wolf."
    )
    lines.append("")
    lines.append("| metric | value | 95% CI |")
    lines.append("|---|---|---|")
    lines.append(
        f"| macro F1 | {report.macro_f1.value:.3f} | "
        f"[{report.macro_f1.low:.3f}, {report.macro_f1.high:.3f}] |"
    )
    if report.pr_auc:
        lines.append(
            f"| PR-AUC | {report.pr_auc.value:.3f} | "
            f"[{report.pr_auc.low:.3f}, {report.pr_auc.high:.3f}] |"
        )
    if report.roc_auc:
        lines.append(
            f"| ROC-AUC | {report.roc_auc.value:.3f} | "
            f"[{report.roc_auc.low:.3f}, {report.roc_auc.high:.3f}] |"
        )
    if report.brier is not None:
        lines.append(f"| Brier | {report.brier:.4f} | — |")
    lines.append("")

    lines.append("### Per class")
    lines.append("")
    lines.append("| class | precision | recall | F1 | support |")
    lines.append("|---|---|---|---|---|")
    for name, values in report.per_class.items():
        lines.append(
            f"| {name} | {values['precision']:.3f} | {values['recall']:.3f} | "
            f"{values['f1']:.3f} | {values['support']} |"
        )
    lines.append("")

    lines.append("### Confusion matrix")
    lines.append("")
    header = "| actual \\ predicted | " + " | ".join(report.class_names) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(report.class_names) + 1))
    for name, row in zip(report.class_names, report.confusion, strict=True):
        lines.append(f"| **{name}** | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")

    if baselines:
        lines.append("## Baselines")
        lines.append("")
        lines.append(
            "A baseline exists to answer *what did the expensive model buy*. Overlapping "
            "confidence intervals are reported as 'not separable', never as a win."
        )
        lines.append("")
        lines.append("| baseline | macro F1 | delta | verdict |")
        lines.append("|---|---|---|---|")
        for name, verdict in (baselines.get("verdicts") or {}).items():
            baseline_f1 = verdict["baseline_macro_f1"]
            lines.append(
                f"| {name} | {baseline_f1['value']:.3f} "
                f"[{baseline_f1['ci_low']:.3f}, {baseline_f1['ci_high']:.3f}] | "
                f"{verdict['delta']:+.3f} | {verdict['verdict']} |"
            )
        lines.append("")
        if not baselines.get("clears_every_baseline"):
            lines.append(
                "> **The model does not cleanly clear every baseline.** That is the finding, "
                "reported here rather than tuned away."
            )
            lines.append("")

    if calibration:
        lines.append("## Calibration")
        lines.append("")
        lines.append(
            "Phase 4 multiplies these scores together, so they must be probabilities rather "
            "than arbitrary decision values."
        )
        lines.append("")
        lines.append(f"- {calibration.summary()}")
        if calibration.note:
            lines.append(f"- Note: {calibration.note}")
        if calibration.degraded:
            lines.append(
                "- **Calibration made the Brier score worse.** Reported rather than hidden; "
                "the usual cause is a validation split too small for the chosen method."
            )
        lines.append("")
        lines.append("| predicted | observed | n |")
        lines.append("|---|---|---|")
        after = calibration.reliability_after
        for p, o, c in zip(after["predicted"], after["observed"], after["counts"], strict=True):
            lines.append(f"| {p:.3f} | {o:.3f} | {c} |")
        lines.append("")

    for title, body in (extra_sections or {}).items():
        lines.append(f"## {title}")
        lines.append("")
        lines.append(body)
        lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    fingerprint = run_fingerprint()
    lines.append(f"- seed: `{fingerprint['seed']}`")
    lines.append(f"- device: `{fingerprint['device']}`")
    lines.append(f"- input manifest hash: `{fingerprint['input_manifest_hash']}`")
    lines.append(f"- languages: `{fingerprint['languages']}`")
    lines.append("")
    lines.append("<details><summary>library versions</summary>")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(fingerprint["library_versions"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    lines.append(
        f"Regenerate from saved predictions with `python -m modeling.cli report {module}`."
    )
    lines.append("")
    return "\n".join(lines)


def _write_charts(
    target: Path, report: ClassificationReport, calibration: CalibrationResult | None
) -> list[Path]:
    """Confusion matrix and reliability diagram. Skipped without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed; skipping charts")
        return []

    written: list[Path] = []

    figure, axis = plt.subplots(figsize=(4.5, 4))
    matrix = np.asarray(report.confusion, dtype=float)
    normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
    axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(report.class_names)), report.class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(report.class_names)), report.class_names)
    axis.set_xlabel("predicted")
    axis.set_ylabel("actual")
    axis.set_title(f"{report.module} confusion (row-normalized)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                color="white" if normalized[i, j] > 0.5 else "black",
            )
    figure.tight_layout()
    path = target / "confusion.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    written.append(path)

    if calibration:
        figure, axis = plt.subplots(figsize=(4.5, 4))
        axis.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
        for label, curve in (
            ("before", calibration.reliability_before),
            ("after", calibration.reliability_after),
        ):
            if curve["predicted"]:
                axis.plot(curve["predicted"], curve["observed"], "o-", label=label)
        axis.set_xlabel("predicted probability")
        axis.set_ylabel("observed frequency")
        axis.set_title(f"{report.module} reliability")
        axis.legend()
        figure.tight_layout()
        path = target / "reliability.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        written.append(path)

    return written


def regenerate_reports(
    module: str | None = None, settings: ModelingSettings | None = None
) -> list[Path]:
    """Rebuild every report from saved predictions, without retraining.

    Backs ``modeling report``. This is the acceptance criterion "regenerates
    artifacts/eval/** from saved predictions without retraining", and it works
    because ``save_predictions`` stored the per-row outputs alongside the
    metrics.
    """
    settings = settings or get_settings()
    root = settings.eval_dir
    if not root.exists():
        return []

    written: list[Path] = []
    for module_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if module and module_dir.name != module:
            continue
        for version_dir in sorted(p for p in module_dir.iterdir() if p.is_dir()):
            predictions = version_dir / "predictions.parquet"
            metrics = version_dir / "metrics.json"
            if not predictions.exists() or not metrics.exists():
                continue
            frame = pd.read_parquet(predictions)
            saved = json.loads(metrics.read_text(encoding="utf-8"))
            from modeling.eval.metrics import classification_report

            report = classification_report(
                frame["y_true"].to_numpy(),
                frame["y_pred"].to_numpy(),
                y_score=frame["y_score"].to_numpy() if "y_score" in frame else None,
                module=module_dir.name,
                split_description=saved["metrics"]["split"],
                is_demo=saved["metrics"].get("is_demo", False),
            )
            written.extend(
                write_report(
                    module=module_dir.name,
                    version=version_dir.name,
                    report=report,
                    baselines=saved.get("baselines"),
                    dataset_summary=saved.get("dataset"),
                    settings=settings,
                )
            )
    return written
