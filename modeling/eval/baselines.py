"""Baselines. Every supervised module reports these next to its main model.

The point of a baseline is not to be beaten. It is to answer "what did the
expensive thing actually buy?" -- and sometimes the answer is "nothing", which is
a finding worth more than a tuned number.

Three baselines, in increasing order of how embarrassing it is to lose to them:

* **Majority class.** Predicts the most common label always. Its accuracy is the
  number that makes accuracy useless on imbalanced data, and seeing it printed
  next to the model is the cheapest possible guard against that mistake.
* **TF-IDF + logistic regression.** Word and character n-grams into a linear
  model. Trains in seconds on CPU. On short social text it is a genuinely strong
  baseline, and a fine-tuned transformer that does not clear it is not worth its
  inference cost.
* **Single-feature logistic regression** (accounts). One feature -- the
  follower/following ratio -- against the whole engineered set. If 30 features
  do not beat 1, the feature engineering is decoration.

A zero-shot LLM prompt is also supported as a fourth comparison, because it
usefully shows what the fine-tune buys over an off-the-shelf model. It is a
*comparison point only*: LLM-as-classifier is forbidden in the product for the
reasons in the LLM usage policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from modeling.eval.metrics import ClassificationReport, classification_report

log = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    name: str
    report: ClassificationReport
    scores: np.ndarray
    predictions: np.ndarray

    def headline(self) -> str:
        return f"{self.name}: {self.report.headline()}"


def majority_baseline(
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    module: str,
    split_description: str,
    seed: int = 0,
    is_demo: bool = False,
) -> BaselineResult:
    """Always predict the training majority class."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(np.zeros((len(y_train), 1)), y_train)
    predictions = model.predict(np.zeros((len(y_test), 1)))
    # A constant classifier has no ranking, so its "probability" is the class
    # prior. Reported so the PR-AUC column shows the no-skill floor explicitly.
    prior = float(np.mean(np.asarray(y_train) == 1))
    scores = np.full(len(y_test), prior)
    return BaselineResult(
        name="majority class",
        report=classification_report(
            y_test,
            predictions,
            y_score=scores,
            module=module,
            split_description=split_description,
            seed=seed,
            is_demo=is_demo,
        ),
        scores=scores,
        predictions=predictions,
    )


def tfidf_logreg(
    texts_train: list[str],
    y_train: np.ndarray,
    texts_test: list[str],
    y_test: np.ndarray,
    *,
    module: str,
    split_description: str,
    seed: int = 0,
    is_demo: bool = False,
) -> BaselineResult:
    """Word + character n-gram TF-IDF into logistic regression.

    Character n-grams are included deliberately: they catch the orthographic
    signature of low-effort content (ALL CAPS, repeated punctuation, unusual
    spacing) that word features miss entirely, and on social text that signature
    carries real signal.

    ``class_weight="balanced"`` because the positive class is the minority and an
    unweighted linear model on imbalanced data collapses to the majority.
    """
    model = build_tfidf_pipeline(seed)
    model.fit(texts_train, y_train)
    scores = model.predict_proba(texts_test)[:, 1]
    predictions = model.predict(texts_test)
    return BaselineResult(
        name="tf-idf + logreg",
        report=classification_report(
            y_test,
            predictions,
            y_score=scores,
            module=module,
            split_description=split_description,
            seed=seed,
            is_demo=is_demo,
        ),
        scores=scores,
        predictions=predictions,
    )


def build_tfidf_pipeline(seed: int = 0) -> Pipeline:
    from sklearn.pipeline import FeatureUnion

    return make_pipeline(
        FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=50_000,
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_features=50_000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        ),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
            C=1.0,
        ),
    )


def single_feature_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    feature: str,
    *,
    module: str,
    split_description: str,
    seed: int = 0,
    is_demo: bool = False,
) -> BaselineResult:
    """One feature against the whole engineered set.

    If 30 features do not beat 1, the feature engineering is decoration and the
    model card should say so.
    """
    if feature not in feature_names:
        raise KeyError(f"{feature!r} not in {feature_names[:10]}...")
    column = feature_names.index(feature)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    model.fit(X_train[:, [column]], y_train)
    scores = model.predict_proba(X_test[:, [column]])[:, 1]
    predictions = model.predict(X_test[:, [column]])
    return BaselineResult(
        name=f"logreg on {feature} only",
        report=classification_report(
            y_test,
            predictions,
            y_score=scores,
            module=module,
            split_description=split_description,
            seed=seed,
            is_demo=is_demo,
        ),
        scores=scores,
        predictions=predictions,
    )


def compare(main: ClassificationReport, baselines: list[BaselineResult]) -> dict[str, Any]:
    """Did the main model actually beat its baselines?

    Uses non-overlapping bootstrap intervals as the bar. Overlapping intervals
    are reported as "not separable at this test size" rather than as a win --
    the honest reading of a 2-point difference on 500 rows.
    """
    out: dict[str, Any] = {"main_macro_f1": main.macro_f1.as_dict(), "verdicts": {}}
    beaten_all = True
    for baseline in baselines:
        separated = main.macro_f1.separated_from(baseline.report.macro_f1)
        better = main.macro_f1.value > baseline.report.macro_f1.value
        if separated and better:
            verdict = "beats"
        elif separated and not better:
            verdict = "LOSES TO"
            beaten_all = False
        else:
            verdict = "not separable at this test size"
            beaten_all = False
        out["verdicts"][baseline.name] = {
            "baseline_macro_f1": baseline.report.macro_f1.as_dict(),
            "delta": round(main.macro_f1.value - baseline.report.macro_f1.value, 4),
            "verdict": verdict,
        }
        if verdict == "LOSES TO":
            log.warning(
                "the main model LOSES to the %s baseline (%.3f vs %.3f). Report this and "
                "stop -- do not tune until the number looks better.",
                baseline.name,
                main.macro_f1.value,
                baseline.report.macro_f1.value,
            )
    out["clears_every_baseline"] = beaten_all
    return out


def zero_shot_llm(
    texts_test: list[str],
    y_test: np.ndarray,
    *,
    module: str,
    split_description: str,
    prompt: str,
    seed: int = 0,
    is_demo: bool = False,
) -> BaselineResult | None:
    """Zero-shot LLM on the same test set, as a comparison point only.

    Shows what the fine-tune buys over an off-the-shelf model. Returns ``None``
    with no API key, because a baseline that cannot run must not silently become
    a baseline that scores zero.

    This is *not* a licence to use an LLM as the product's classifier -- see the
    LLM usage policy. It produces no reportable train/test methodology, its cost
    scales with corpus size, and a grader cannot audit it.
    """
    from modeling.text.summarize import LLMClient

    client = LLMClient()
    if not client.available:
        log.info("no ANTHROPIC_API_KEY; skipping the zero-shot LLM baseline")
        return None

    predictions = []
    for text in texts_test:
        answer = client.classify(prompt, text)
        predictions.append(1 if answer else 0)
    predictions = np.asarray(predictions)
    return BaselineResult(
        name=f"zero-shot {client.model}",
        report=classification_report(
            y_test,
            predictions,
            y_score=predictions.astype(float),
            module=module,
            split_description=split_description,
            seed=seed,
            is_demo=is_demo,
        ),
        scores=predictions.astype(float),
        predictions=predictions,
    )
