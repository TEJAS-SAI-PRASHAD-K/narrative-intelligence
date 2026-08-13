"""Module ablation: what does each signal actually contribute?

Run the pipeline with each module disabled in turn -- text only, +accounts,
+media, +coordination -- and report how detection quality changes on a held-out
set of hand-labelled narratives.

================================================================================
THE FUSION BELOW IS NOT THE PRODUCT FORMULA.
================================================================================

Phase 4 owns the fused 0-100 risk score, deliberately, because the weighting is
a documented product decision rather than a model output. What lives here is a
**provisional fusion for measurement only**: an equally-weighted mean of the
available normalized components. It exists so that "adding coordination changed
the ranking by this much" is a sentence with a number in it, and for no other
purpose.

Three properties make it fit for measurement and unfit for shipping:

* **Equal weights.** No component is privileged, because privileging one is
  exactly the product decision this phase is not making.
* **Null-aware.** A missing component is dropped from the mean rather than
  imputed to zero. A record with no deepfake score is not a record with a
  deepfake score of zero.
* **Rank-oriented.** It is evaluated by agreement with hand labels on ranking,
  not by absolute calibration, because an unweighted mean of calibrated
  probabilities is not itself calibrated.

Any code outside this module that imports ``provisional_fusion`` is a bug.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from modeling.config import ModelingSettings, get_settings
from modeling.io import ScoredStore

log = logging.getLogger(__name__)

#: Each ablation configuration: name -> the components it may use.
CONFIGURATIONS: dict[str, tuple[str, ...]] = {
    "text only": ("misinfo", "toxicity"),
    "+ accounts": ("misinfo", "toxicity", "bot"),
    "+ coordination": ("misinfo", "toxicity", "bot", "coordination"),
    "+ media": ("misinfo", "toxicity", "bot", "coordination", "deepfake"),
}

PROVISIONAL_FUSION_WARNING = (
    "PROVISIONAL FUSION — FOR MEASUREMENT ONLY. This is an equally-weighted mean of "
    "the available components, defined in modeling/eval/ablation.py. It is NOT the "
    "product's risk score: Phase 4 owns that formula, because the weighting is a "
    "documented product decision rather than a model output."
)


@dataclass
class AblationRow:
    configuration: str
    components: tuple[str, ...]
    n_scored: int
    coverage: dict[str, float]
    spearman: float | None = None
    top_k_precision: float | None = None
    note: str = ""


@dataclass
class AblationResult:
    rows: list[AblationRow] = field(default_factory=list)
    n_narratives: int = 0
    n_labelled: int = 0
    label_source: str = "none"

    def render(self) -> str:
        lines = [
            "# Module ablation",
            "",
            f"> {PROVISIONAL_FUSION_WARNING}",
            "",
        ]
        if self.n_labelled == 0:
            lines.append(
                "**No hand-labelled narratives are available**, so the quality columns are "
                "empty and only component coverage is reported. Produce labels with "
                "`modeling sample-for-labelling narratives`, then rerun."
            )
            lines.append("")

        lines.append(
            f"{self.n_narratives} narratives; {self.n_labelled} hand-labelled "
            f"(source: {self.label_source})."
        )
        lines.append("")
        lines.append("| configuration | components | scored | Spearman rho | precision@10 | note |")
        lines.append("|---|---|---|---|---|---|")
        for row in self.rows:
            rho = f"{row.spearman:.3f}" if row.spearman is not None else "—"
            precision = (
                f"{row.top_k_precision:.3f}" if row.top_k_precision is not None else "—"
            )
            lines.append(
                f"| {row.configuration} | {', '.join(row.components)} | {row.n_scored} | "
                f"{rho} | {precision} | {row.note} |"
            )
        lines.append("")
        lines.append("## Component coverage")
        lines.append("")
        lines.append(
            "The share of narratives for which each component produced a value. A "
            "component at 0% contributes nothing to its configuration, and its row in the "
            "table above is therefore identical to the row below it — that is information, "
            "not a bug."
        )
        lines.append("")
        if self.rows:
            components = sorted({c for row in self.rows for c in row.coverage})
            lines.append("| component | coverage |")
            lines.append("|---|---|")
            for component in components:
                value = next(
                    (row.coverage[component] for row in self.rows if component in row.coverage),
                    0.0,
                )
                lines.append(f"| `{component}` | {value:.0%} |")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "warning": PROVISIONAL_FUSION_WARNING,
            "n_narratives": self.n_narratives,
            "n_labelled": self.n_labelled,
            "label_source": self.label_source,
            "rows": [
                {
                    "configuration": row.configuration,
                    "components": list(row.components),
                    "n_scored": row.n_scored,
                    "coverage": row.coverage,
                    "spearman": row.spearman,
                    "top_k_precision": row.top_k_precision,
                    "note": row.note,
                }
                for row in self.rows
            ],
        }


def provisional_fusion(components: pd.DataFrame, allowed: tuple[str, ...]) -> pd.Series:
    """Equally-weighted, null-aware mean of the allowed components.

    **For measurement only.** See the module docstring; Phase 4 owns the real
    formula.

    Null-aware means a missing component is *dropped from the mean*, not imputed
    to zero. Imputing zero would make "we did not assess this" indistinguishable
    from "we assessed this and it was clean", which is the exact conflation the
    whole null discipline in this phase exists to prevent.
    """
    usable = [c for c in allowed if c in components.columns]
    if not usable:
        return pd.Series(np.nan, index=components.index)
    values = components[usable].apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1, skipna=True)


def run_ablation(
    *, demo: bool = False, settings: ModelingSettings | None = None
) -> AblationResult:
    """Build the ablation table from the scored tables on disk."""
    settings = settings or get_settings()
    root = settings.scored_dir.parent / "scored_demo" if demo else None
    store = ScoredStore(settings, root=root)

    narratives = store.read("narratives")
    if not len(narratives):
        return AblationResult([], 0, 0, "none")

    components = _narrative_components(store, narratives)
    labels = _load_labels(settings, narratives)

    rows: list[AblationRow] = []
    for name, allowed in CONFIGURATIONS.items():
        fused = provisional_fusion(components, allowed)
        scored = int(fused.notna().sum())
        coverage = {
            component: float(components[component].notna().mean())
            if component in components.columns
            else 0.0
            for component in allowed
        }
        row = AblationRow(
            configuration=name,
            components=allowed,
            n_scored=scored,
            coverage=coverage,
        )
        if labels is not None and scored >= 5:
            row.spearman, row.top_k_precision = _quality(fused, labels)
        elif labels is None:
            row.note = "no hand labels"
        else:
            row.note = "too few scored narratives"
        missing = [c for c, v in coverage.items() if v == 0.0]
        if missing:
            row.note = (row.note + "; " if row.note else "") + (
                f"{', '.join(missing)} contributed nothing (0% coverage)"
            )
        rows.append(row)

    result = AblationResult(
        rows=rows,
        n_narratives=len(narratives),
        n_labelled=0 if labels is None else int(labels.notna().sum()),
        label_source="artifacts/hand_labels/narratives.csv" if labels is not None else "none",
    )
    log.info("ablation: %d configurations over %d narratives", len(rows), len(narratives))
    return result


def _narrative_components(store: ScoredStore, narratives: pd.DataFrame) -> pd.DataFrame:
    """Per-narrative component scores, aggregated from the scored tables."""
    membership = store.read("narrative_membership")
    record_scores = store.read("record_scores")
    author_scores = store.read("author_scores")
    media_scores = store.read("media_scores")

    frame = pd.DataFrame(index=narratives["narrative_id"].astype(str))
    frame["misinfo"] = narratives.set_index(
        narratives["narrative_id"].astype(str)
    )["severity"].reindex(frame.index)

    if len(membership) and len(record_scores):
        joined = membership.merge(record_scores, on="record_id", how="left")
        for source, target in (("toxicity", "toxicity"), ("anomaly_score", "anomaly")):
            if source in joined.columns:
                frame[target] = (
                    joined.groupby("narrative_id")[source]
                    .mean()
                    .reindex(frame.index)
                )

    if len(author_scores) and len(membership):
        exploded = author_scores.explode("narratives_touched")
        exploded = exploded.loc[exploded["narratives_touched"].notna()]
        if len(exploded):
            grouped = exploded.groupby(exploded["narratives_touched"].astype(str))
            for source, target in (
                ("bot_prob", "bot"),
                ("coordination_score", "coordination"),
            ):
                if source in exploded.columns:
                    frame[target] = grouped[source].max().reindex(frame.index)

    if len(media_scores) and len(membership):
        joined = membership.merge(media_scores, on="record_id", how="inner")
        if len(joined) and "deepfake_prob" in joined.columns:
            frame["deepfake"] = (
                joined.groupby("narrative_id")["deepfake_prob"].max().reindex(frame.index)
            )
    return frame


def _load_labels(settings: ModelingSettings, narratives: pd.DataFrame) -> pd.Series | None:
    """Hand-labelled narrative risk, if a human has produced any."""
    path = settings.artifacts_dir / "hand_labels" / "narratives.csv"
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        log.warning("could not read %s: %s", path, exc)
        return None
    if "narrative_id" not in frame.columns or "risk" not in frame.columns:
        log.warning("%s must have narrative_id and risk columns", path)
        return None
    series = frame.set_index(frame["narrative_id"].astype(str))["risk"]
    return series.reindex(narratives["narrative_id"].astype(str))


def _quality(fused: pd.Series, labels: pd.Series) -> tuple[float | None, float | None]:
    """Rank agreement with the hand labels, plus precision at the top.

    Rank agreement rather than a calibration metric: an unweighted mean of
    calibrated probabilities is not itself calibrated, so its absolute values
    are not interpretable even when its ordering is.
    """
    from scipy.stats import spearmanr

    aligned = pd.DataFrame({"fused": fused, "label": labels}).dropna()
    if len(aligned) < 5:
        return None, None
    rho = float(spearmanr(aligned["fused"], aligned["label"]).statistic)

    k = min(10, len(aligned))
    top = aligned.nlargest(k, "fused")
    threshold = aligned["label"].quantile(0.75)
    precision = float((top["label"] >= threshold).mean())
    return rho, precision


def write_ablation(
    result: AblationResult, *, settings: ModelingSettings | None = None
) -> list:
    """Persist the ablation table alongside the other eval artifacts."""
    import json

    settings = settings or get_settings()
    target = settings.eval_dir / "ablation"
    target.mkdir(parents=True, exist_ok=True)
    written = []

    path = target / "ablation.md"
    path.write_text(result.render() + "\n", encoding="utf-8")
    written.append(path)

    path = target / "ablation.json"
    path.write_text(json.dumps(result.as_dict(), indent=2, default=str) + "\n", encoding="utf-8")
    written.append(path)
    return written


def pairwise_component_overlap(components: pd.DataFrame) -> dict[str, float]:
    """How often two components are both present.

    A configuration that adds a component present on 3% of narratives has not
    been meaningfully tested, and this is the cheapest way to notice.
    """
    out: dict[str, float] = {}
    for a, b in combinations(components.columns, 2):
        both = (components[a].notna() & components[b].notna()).mean()
        out[f"{a}+{b}"] = round(float(both), 4)
    return out
