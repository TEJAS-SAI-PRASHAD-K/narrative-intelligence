"""The fusion score.

**This is a plain, documented Python function, not a model.** That is the point.
Every number the UI flags a narrative with can be decomposed into three
components, each of those into named sub-signals, each of those into the raw
inputs that produced it -- and the weights that were applied are served
alongside. A score nobody can explain in one click is the thing this product
exists to argue against, so the explanation is the return value, not an
afterthought.

    fusion = 100 * (w1*narrative_severity + w2*coordination + w3*authenticity)

Three rules, all enforced here rather than by convention:

1. **Weights and thresholds live in ``configs/fusion.yaml``**, never in code.
   Every config carries a ``version`` that is written onto every row it
   produces, so a stale score is detectable rather than suspected.
2. **Missing inputs propagate.** A component that could not be computed is
   ``None``, is named in ``missing``, and the remaining weights renormalize over
   it. Treating "not measured" as "measured zero" would systematically
   under-flag every narrative whose deepfake pass never ran.
3. **Everything normalizes the same way**, through ``app/scoring/normalize.py``,
   because mixing normalizations makes scores incomparable in a way that is
   invisible until somebody tries to reconcile two of them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.scoring.normalize import Normalizer, renormalized_weights, weighted_mean

log = logging.getLogger(__name__)

FORMULA = "100 * (w1*narrative_severity + w2*coordination + w3*authenticity)"

COMPONENT_DEFINITIONS = {
    "narrative_severity": (
        "How harmful the narrative's content is: aggregated misinformation "
        "likelihood over its posts, the Compass verification risk for its claim, "
        "mean toxicity and the share of negative sentiment."
    ),
    "coordination": (
        "How inauthentically it spread: the share of member authors scoring "
        "bot-like, the density of near-duplicate co-posting between them, how "
        "bursty its timeline is against its own mean, and how concentrated its "
        "authors are in a single cohort."
    ),
    "authenticity": (
        "Whether its evidence is what it claims to be: deepfake detections on "
        "attached media, aggregated risk of the domains it links, and the share "
        "of its authors that are anonymous or deleted."
    ),
}

SUB_DEFINITIONS = {
    "misinfo_likelihood_agg": (
        "Engagement-weighted mean of member posts' misinformation likelihood. "
        "Weighted by engagement because a false claim nobody saw is not the same "
        "problem as one that reached a hundred thousand people."
    ),
    "compass_risk": (
        "The Compass verification status mapped to [0,1]. `insufficient_evidence` "
        "sits mid-scale, not at zero: failing to source a claim is not evidence "
        "that the claim is harmless."
    ),
    "toxicity": "Mean toxicity over member posts the toxicity model scored.",
    "negative_sentiment": "Share of scored member posts classified negative.",
    "bot_like_ratio": (
        "Share of member authors above the bot threshold, over the authors the "
        "bot classifier could score. Outlet 'authors' on news sources are "
        "excluded rather than counted as human."
    ),
    "co_post_similarity_density": (
        "Density of co-posting-similarity edges among member authors, against a "
        "complete graph on the same author set."
    ),
    "temporal_burstiness": (
        "Share of the narrative's volume falling in buckets above the burst "
        "multiple of its own mean. Coordination looks like a spike."
    ),
    "cohort_concentration": (
        "Share of member posts written by the single largest cohort. A narrative "
        "carried entirely by one segment is a different object from one with "
        "broad reach."
    ),
    "deepfake_hits": "Share of checked media on member posts flagged as manipulated.",
    "domain_risk_agg": "Mean risk score of the domains member posts link to.",
    "anonymous_author_ratio": (
        "Share of member authors that are deleted or carry no handle. Absence of "
        "attribution is not proof of bad faith, which is why this is a minority "
        "of one component rather than a flag of its own."
    ),
}


@dataclass
class NarrativeSignals:
    """The raw inputs for one narrative.

    Every field is optional and ``None`` means "not measured". The dataclass
    exists so the caller cannot accidentally pass a zero where a null belongs --
    it has to be explicit at the call site.
    """

    narrative_id: str
    misinfo_likelihood_agg: float | None = None
    compass_status: str | None = None
    toxicity: float | None = None
    negative_sentiment: float | None = None

    bot_like_ratio: float | None = None
    co_post_similarity_density: float | None = None
    temporal_burstiness: float | None = None
    cohort_concentration: float | None = None

    deepfake_hits: float | None = None
    domain_risk_agg: float | None = None
    anonymous_author_ratio: float | None = None

    #: Extra context echoed into the components blob for the drilldown.
    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    score: float | None
    priority: str
    components: dict[str, Any]
    scoring_version: str

    @property
    def missing(self) -> list[str]:
        return list(self.components.get("missing", ()))


def load_config() -> dict[str, Any]:
    """Read ``configs/fusion.yaml``. Not cached, deliberately.

    A rescore has to pick up an edited weight without a restart -- that is the
    whole point of the file being config rather than code, and the acceptance
    criterion says so explicitly. The file is small and read once per run.
    """
    from app.config import load_yaml

    config = load_yaml("fusion")
    if not config:
        raise RuntimeError(
            "configs/fusion.yaml is missing or empty. The fusion weights are "
            "configuration, not defaults baked into the code, so there is "
            "deliberately nothing to fall back to."
        )
    return config


@dataclass(frozen=True)
class Normalizers:
    """Per-project reference distributions, built once per scoring run."""

    by_signal: dict[str, Normalizer]

    def get(self, name: str) -> Normalizer:
        return self.by_signal.get(name, Normalizer(method="raw"))

    @classmethod
    def build(cls, populations: dict[str, list[float | None]], config: dict) -> Normalizers:
        norm_config = config.get("normalization", {})
        method = norm_config.get("method", "percentile")
        min_sample = int(norm_config.get("min_sample", 20))
        return cls(
            by_signal={
                name: Normalizer.from_values(values, method=method, min_sample=min_sample)
                for name, values in populations.items()
            }
        )


def fusion_score(
    signals: NarrativeSignals,
    *,
    config: dict[str, Any] | None = None,
    normalizers: Normalizers | None = None,
) -> FusionResult:
    """Score one narrative. Returns the score **and** everything behind it."""
    config = config or load_config()
    version = str(config.get("version", "fusion-unversioned"))
    weights = config["weights"]
    normalizers = normalizers or Normalizers(by_signal={})

    severity_parts = {
        "misinfo_likelihood_agg": normalizers.get("misinfo_likelihood_agg")(
            signals.misinfo_likelihood_agg
        ),
        "compass_risk": _compass_risk(signals.compass_status, config),
        "toxicity": normalizers.get("toxicity")(signals.toxicity),
        "negative_sentiment": normalizers.get("negative_sentiment")(signals.negative_sentiment),
    }
    coordination_parts = {
        "bot_like_ratio": normalizers.get("bot_like_ratio")(signals.bot_like_ratio),
        "co_post_similarity_density": normalizers.get("co_post_similarity_density")(
            signals.co_post_similarity_density
        ),
        "temporal_burstiness": normalizers.get("temporal_burstiness")(signals.temporal_burstiness),
        "cohort_concentration": normalizers.get("cohort_concentration")(
            signals.cohort_concentration
        ),
    }
    authenticity_parts = {
        "deepfake_hits": normalizers.get("deepfake_hits")(signals.deepfake_hits),
        "domain_risk_agg": normalizers.get("domain_risk_agg")(signals.domain_risk_agg),
        "anonymous_author_ratio": normalizers.get("anonymous_author_ratio")(
            signals.anonymous_author_ratio
        ),
    }

    severity, severity_missing, severity_renorm = weighted_mean(
        severity_parts, config["narrative_severity"]
    )
    coordination, coordination_missing, coordination_renorm = weighted_mean(
        coordination_parts, config["coordination"]
    )
    authenticity, authenticity_missing, authenticity_renorm = weighted_mean(
        authenticity_parts, config["authenticity"]
    )

    components_values = {
        "narrative_severity": severity,
        "coordination": coordination,
        "authenticity": authenticity,
    }
    combined, missing, renormalized = weighted_mean(components_values, weights)
    applied_weights = renormalized_weights(components_values, weights)

    score = None if combined is None else round(100.0 * combined, 2)
    components = {
        "formula": FORMULA,
        "normalization": _describe_normalization(config),
        "scoring_version": version,
        "missing": missing,
        "weights_renormalized": renormalized,
        "components": [
            _component(
                "narrative_severity",
                severity,
                weights["narrative_severity"],
                applied_weights["narrative_severity"],
                severity_parts,
                config["narrative_severity"],
                severity_missing,
                severity_renorm,
                signals,
            ),
            _component(
                "coordination",
                coordination,
                weights["coordination"],
                applied_weights["coordination"],
                coordination_parts,
                config["coordination"],
                coordination_missing,
                coordination_renorm,
                signals,
            ),
            _component(
                "authenticity",
                authenticity,
                weights["authenticity"],
                applied_weights["authenticity"],
                authenticity_parts,
                config["authenticity"],
                authenticity_missing,
                authenticity_renorm,
                signals,
            ),
        ],
        "raw_inputs": signals.inputs,
    }

    return FusionResult(
        score=score,
        priority=priority_for(score, config),
        components=components,
        scoring_version=version,
    )


def _component(
    name: str,
    value: float | None,
    configured_weight: float,
    applied_weight: float,
    parts: dict[str, float | None],
    sub_weights: dict[str, float],
    sub_missing: list[str],
    sub_renormalized: bool,
    signals: NarrativeSignals,
) -> dict[str, Any]:
    applied_sub = renormalized_weights(parts, sub_weights)
    return {
        "name": name,
        "value": None if value is None else round(value, 4),
        # Both weights travel. The configured one explains the design; the
        # applied one is what makes the arithmetic on screen reconcile.
        "weight": round(applied_weight, 4),
        "configured_weight": configured_weight,
        "contribution": None if value is None else round(value * applied_weight, 4),
        "definition": COMPONENT_DEFINITIONS[name],
        "missing": sub_missing,
        "weights_renormalized": sub_renormalized,
        "inputs": {
            sub: {
                "value": None if parts[sub] is None else round(parts[sub], 4),
                "weight": round(applied_sub.get(sub, 0.0), 4),
                "definition": SUB_DEFINITIONS[sub],
            }
            for sub in sub_weights
        },
    }


def _compass_risk(status: str | None, config: dict[str, Any]) -> float | None:
    """Compass verification status -> [0,1].

    Returns ``None`` when no Compass context exists, which is different from a
    context that found nothing: "not fact-checked" and "fact-checked and
    unsourceable" are different states and must not collapse into one number.
    """
    if not status:
        return None
    return config.get("compass_risk_values", {}).get(status)


def priority_for(score: float | None, config: dict[str, Any]) -> str:
    """Derived from the score by the configured bounds. Never assigned by hand.

    An unscorable narrative is `low` rather than `high`: with no evidence at all,
    escalating it would flood the queue with narratives nobody has measured, and
    the UI surfaces the missing components separately.
    """
    if score is None:
        return "low"
    bounds = config.get("priority", {})
    if score >= float(bounds.get("high", 70.0)):
        return "high"
    if score >= float(bounds.get("medium", 40.0)):
        return "medium"
    return "low"


def _describe_normalization(config: dict[str, Any]) -> str:
    norm = config.get("normalization", {})
    method = norm.get("method", "percentile")
    if method == "percentile":
        return (
            f"percentile-rank within project (min sample {norm.get('min_sample', 20)}; "
            "below that the raw value is used and the component says so). Chosen over "
            "min-max against a fixed reference range because this is a within-case "
            "triage tool: 'high' should mean high for this investigation, and a fixed "
            "reference range is a guess that ages badly."
        )
    return f"min-max against the project range (method={method})"
