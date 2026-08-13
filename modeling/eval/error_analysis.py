"""Error analysis: sample the failures, categorize them, count them.

A confusion matrix says *how many* the model got wrong. It never says *what kind
of wrong*, and the kind is what tells you whether a model is deployable. "F1
0.78" and "F1 0.78, and 40% of the false positives are satire" are different
findings, and only the second one leads anywhere.

This module samples at least 50 false positives and 50 false negatives per
module, applies a keyword-and-shape taxonomy as a **first pass**, and writes the
uncategorized remainder out for a human to read. The automated pass is a triage
aid, not the analysis: the deliverable is prose in
``artifacts/error_analysis/<module>.md`` that a person wrote after reading the
uncategorized examples.

The taxonomies below encode failure modes that are known in advance from the
literature and from the shape of this corpus. They are deliberately
conservative -- a pattern that fires on everything categorizes nothing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.config import ModelingSettings, get_settings

log = logging.getLogger(__name__)


def _as_list(value) -> list:
    from modeling.io import as_list

    return as_list(value)

MIN_SAMPLES = 50


@dataclass
class FailureCategory:
    """One named failure mode, with a cheap detector."""

    name: str
    description: str
    pattern: re.Pattern | None = None
    predicate: Any = None

    def matches(self, row: dict[str, Any]) -> bool:
        text = str(row.get("text", ""))
        if self.pattern is not None and self.pattern.search(text):
            return True
        if self.predicate is not None:
            try:
                return bool(self.predicate(row))
            except Exception:  # pragma: no cover - defensive
                return False
        return False


def _re(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


#: Text-classifier failure modes (misinfo, stance).
TEXT_TAXONOMY = [
    FailureCategory(
        "satire_or_parody",
        "Satirical or parody content read as a sincere claim. The single most "
        "common false positive for misinformation classifiers, because satire "
        "and disinformation share surface form by design.",
        pattern=_re(r"\b(satir\w*|parod\w*|the onion|babylon bee|/s\b|onion\b)"),
    ),
    FailureCategory(
        "sarcasm_or_irony",
        "Sarcasm inverts the intended stance while leaving the surface wording "
        "intact. Breaks stance detection especially.",
        pattern=_re(r"(\byeah right\b|\bsure,? because\b|\btotally\b.*\bnot\b|\bobviously\b.*\?)"),
    ),
    FailureCategory(
        "quoting_the_claim_to_debunk_it",
        "A post that quotes a false claim in order to refute it. Lexically "
        "near-identical to the claim itself; the model sees the claim.",
        pattern=_re(r"\b(debunk\w*|no evidence|false claim|fact.?check\w*|this is not true)\b"),
    ),
    FailureCategory(
        "very_short_text",
        "Too little text to carry a claim. GDELT article metadata and one-line "
        "comments dominate this bucket.",
        predicate=lambda row: len(str(row.get("text", ""))) < 60,
    ),
    FailureCategory(
        "opinion_not_claim",
        "An expression of preference or feeling with no checkable proposition. "
        "The label set has no place for it, so the model must guess.",
        pattern=_re(r"^\s*(i (think|feel|believe|hate|love)|imo\b|imho\b)"),
    ),
    FailureCategory(
        "url_or_quote_only",
        "Almost entirely a link or a block quote, with no assertion of its own.",
        predicate=lambda row: bool(_as_list(row.get("urls")))
        and len(re.sub(r"https?://\S+", "", str(row.get("text", ""))).strip()) < 40,
    ),
]

#: Account-classifier failure modes (bot).
ACCOUNT_TAXONOMY = [
    FailureCategory(
        "low_follower_human",
        "A real person with few followers and a lopsided follow ratio. Looks "
        "like a fake-follower account on exactly the features that separate them.",
        predicate=lambda row: (row.get("followers") or 0) < 50
        and (row.get("following") or 0) > 300,
    ),
    FailureCategory(
        "new_account_human",
        "A genuine account created recently. Account age is a strong bot feature "
        "and a weak one for anyone who just joined.",
        predicate=lambda row: (row.get("account_age_days") or 9999) < 30,
    ),
    FailureCategory(
        "high_volume_human",
        "A prolific human -- a journalist, a community moderator, a hobbyist. "
        "Posting rate alone does not distinguish them from a scheduler.",
        predicate=lambda row: (row.get("posts_per_day") or 0) > 20,
    ),
    FailureCategory(
        "organisational_account",
        "A brand, outlet or bot-by-design account (news feeds, weather bots). "
        "Automated and legitimate -- the label conflates the two.",
        predicate=lambda row: bool(row.get("author_is_outlet")),
    ),
    FailureCategory(
        "sparse_history",
        "Too few posts to compute the behavioural features the model relies on.",
        predicate=lambda row: (row.get("post_count") or 0) < 5,
    ),
]

#: Deepfake failure modes.
MEDIA_TAXONOMY = [
    FailureCategory(
        "small_or_blurry_face",
        "Face crop below the model's effective resolution. Compression artefacts "
        "and manipulation artefacts become indistinguishable.",
        predicate=lambda row: (row.get("face_size") or 999) < 80,
    ),
    FailureCategory(
        "heavy_compression",
        "Re-encoded video. The known weak point: training on c23 and testing on "
        "platform-transcoded footage is a domain shift the model loses to.",
        predicate=lambda row: str(row.get("compression", "")) in {"c40", "recompressed"},
    ),
    FailureCategory(
        "occlusion_or_profile",
        "Face partly hidden or turned away, so the artefact regions are not visible.",
        predicate=lambda row: bool(row.get("occluded")),
    ),
    FailureCategory(
        "unseen_manipulation_method",
        "Generated by a method absent from training. Cross-method generalisation "
        "is the known weakness of this whole model family.",
        predicate=lambda row: bool(row.get("held_out_method")),
    ),
]

TAXONOMIES = {
    "misinfo": TEXT_TAXONOMY,
    "stance": TEXT_TAXONOMY,
    "bot": ACCOUNT_TAXONOMY,
    "deepfake": MEDIA_TAXONOMY,
}


@dataclass
class ErrorAnalysis:
    module: str
    false_positives: pd.DataFrame
    false_negatives: pd.DataFrame
    fp_counts: dict[str, int] = field(default_factory=dict)
    fn_counts: dict[str, int] = field(default_factory=dict)
    uncategorized_fp: pd.DataFrame = field(default_factory=pd.DataFrame)
    uncategorized_fn: pd.DataFrame = field(default_factory=pd.DataFrame)
    undersampled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "n_false_positives": len(self.false_positives),
            "n_false_negatives": len(self.false_negatives),
            "fp_categories": self.fp_counts,
            "fn_categories": self.fn_counts,
            "uncategorized_fp": len(self.uncategorized_fp),
            "uncategorized_fn": len(self.uncategorized_fn),
            "undersampled": self.undersampled,
        }


def analyze(
    frame: pd.DataFrame,
    *,
    module: str,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    seed: int = 0,
    n_samples: int = MIN_SAMPLES,
) -> ErrorAnalysis:
    """Sample and categorize the model's failures."""
    taxonomy = TAXONOMIES.get(module, TEXT_TAXONOMY)
    y_true = frame[y_true_col].to_numpy()
    y_pred = frame[y_pred_col].to_numpy()

    fp = frame.loc[(y_true == 0) & (y_pred == 1)]
    fn = frame.loc[(y_true == 1) & (y_pred == 0)]
    undersampled = len(fp) < n_samples or len(fn) < n_samples
    if undersampled:
        log.warning(
            "%s error analysis: only %d FP and %d FN available (target %d each). "
            "The taxonomy counts below are correspondingly thin.",
            module,
            len(fp),
            len(fn),
            n_samples,
        )

    rng = np.random.default_rng(seed)
    fp_sample = _sample(fp, n_samples, rng)
    fn_sample = _sample(fn, n_samples, rng)

    fp_counts, fp_left = _categorize(fp_sample, taxonomy)
    fn_counts, fn_left = _categorize(fn_sample, taxonomy)

    return ErrorAnalysis(
        module=module,
        false_positives=fp_sample,
        false_negatives=fn_sample,
        fp_counts=fp_counts,
        fn_counts=fn_counts,
        uncategorized_fp=fp_left,
        uncategorized_fn=fn_left,
        undersampled=undersampled,
    )


def _sample(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    index = rng.choice(len(frame), size=n, replace=False)
    return frame.iloc[sorted(index)].copy()


def _categorize(
    frame: pd.DataFrame, taxonomy: list[FailureCategory]
) -> tuple[dict[str, int], pd.DataFrame]:
    """First-pass triage. A row can match several categories; all are counted.

    Counting every match rather than the first one is deliberate: "satire, and
    also very short" is genuinely both, and forcing a single bucket would hide
    the interaction that actually explains the failure.
    """
    counts = dict.fromkeys((c.name for c in taxonomy), 0)
    uncategorized_index = []
    for position, row in enumerate(frame.to_dict(orient="records")):
        hit = False
        for category in taxonomy:
            if category.matches(row):
                counts[category.name] += 1
                hit = True
        if not hit:
            uncategorized_index.append(position)
    return counts, frame.iloc[uncategorized_index].copy()


def write_markdown(
    analysis: ErrorAnalysis,
    *,
    settings: ModelingSettings | None = None,
    prose: str = "",
    n_examples: int = 8,
) -> Path:
    """Write ``artifacts/error_analysis/<module>.md``.

    The generated file is a *scaffold*: counts, examples, and an explicit
    placeholder for the written analysis. The counts are automated; the reading
    is not, and the file says so rather than pretending a keyword match is an
    error analysis.
    """
    settings = settings or get_settings()
    settings.error_analysis_dir.mkdir(parents=True, exist_ok=True)
    path = settings.error_analysis_dir / f"{analysis.module}.md"
    taxonomy = {c.name: c for c in TAXONOMIES.get(analysis.module, TEXT_TAXONOMY)}

    lines = [f"# {analysis.module} — error analysis", ""]
    lines.append(
        f"Sampled {len(analysis.false_positives)} false positives and "
        f"{len(analysis.false_negatives)} false negatives from the held-out test set."
    )
    if analysis.undersampled:
        lines.append("")
        lines.append(
            f"> **Fewer than {MIN_SAMPLES} errors of one kind were available.** The counts "
            "below are thin and should be read as indicative, not as proportions."
        )
    lines.append("")
    lines.append(
        "The category counts are a keyword-and-shape triage pass, not the analysis. "
        "A row may match more than one category, and all matches are counted -- "
        "\"satire, and also very short\" is genuinely both. The analysis is the prose "
        "below, written after reading the uncategorized examples."
    )
    lines.append("")

    for title, counts, total in (
        ("False positives", analysis.fp_counts, len(analysis.false_positives)),
        ("False negatives", analysis.fn_counts, len(analysis.false_negatives)),
    ):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| category | count | share | what it means |")
        lines.append("|---|---|---|---|")
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            share = f"{100 * count / total:.0f}%" if total else "—"
            description = taxonomy[name].description if name in taxonomy else ""
            lines.append(f"| `{name}` | {count} | {share} | {description} |")
        lines.append("")

    for title, frame in (
        ("Uncategorized false positives", analysis.uncategorized_fp),
        ("Uncategorized false negatives", analysis.uncategorized_fn),
    ):
        lines.append(f"## {title} ({len(frame)})")
        lines.append("")
        if not len(frame):
            lines.append("_None — every sampled error matched at least one known category._")
        else:
            lines.append("These are the ones worth reading. New categories come from here.")
            lines.append("")
            for row in frame.head(n_examples).to_dict(orient="records"):
                snippet = str(row.get("text", row.get("author_id", "")))[:220]
                score = row.get("y_score")
                score_text = f" _(score {score:.3f})_" if isinstance(score, float) else ""
                lines.append(f"- {snippet}{score_text}")
        lines.append("")

    lines.append("## Written analysis")
    lines.append("")
    lines.append(
        prose
        or "_To be written after reading the uncategorized examples above. State which "
        "failure modes are systematic rather than incidental, which are fixable within "
        "this model family, and which are limits of the label set itself._"
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", path)
    return path
