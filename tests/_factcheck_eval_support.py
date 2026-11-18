"""Synthetic score rows and runners shared by the fact-check evaluation tests."""

from __future__ import annotations

import pandas as pd

from modeling.eval.factcheck_eval import ClaimUniverse, evaluate, validate_frame

COLUMNS = ["part", "id", "lang", "label", "score", "source_detail", "skip_reason"]
PUBLISHERS = frozenset({"boomlive.in", "thequint.com"})
B = 300  # resamples for tests that do not assert on the default of 2000


def r(part, id_, score, *, label=None, lang="en", source=None, skip=None):
    return {
        "part": part,
        "id": id_,
        "lang": lang,
        "label": label,
        "score": score,
        "source_detail": source,
        "skip_reason": skip,
    }


def model(name, rows):
    return validate_frame(pd.DataFrame(rows, columns=COLUMNS), name)


def universe(ids, *, frozen=False, excluded=None):
    return ClaimUniverse(
        eval_ids=frozenset(ids), excluded=dict(excluded or {}), frozen=frozen
    )


# Validation rows with a threshold that can be read off by hand.
# (label, score): (1, .9) (1, .8) (0, .3) (0, .1).
#   t=0.9: precision 1,   recall 1/2
#   t=0.8: precision 1,   recall 1       <- lowest threshold reaching 0.85 with best recall
#   t=0.3: precision 2/3, recall 1
#   t=0.1: precision 1/2, recall 1
# so the threshold is 0.8.
VAL_A = [
    r("val", "v1", 0.9, label=1),
    r("val", "v2", 0.8, label=1),
    r("val", "v3", 0.3, label=0),
    r("val", "v4", 0.1, label=0),
]
TEST_ROWS = [("t1", 1), ("t2", 1), ("t3", 0), ("t4", 0)]


def make_test_rows(scores):
    return [r("test", i, s, label=y) for (i, y), s in zip(TEST_ROWS, scores, strict=True)]


def small_case_rows():
    """Claims, corpus titles and corpus text for the hand-worked lift case."""
    # Claims (all known false). Flagged = score >= 0.8:  c1 .9, c2 .85, c3 .8 flagged;
    # c4 .5, c5 .2 not.   en: c1 c2 c3 -> recall 3/3.   hi: c4 c5 -> recall 0/2.
    # all: 3/5 = 0.6.
    claims = [
        r("claims", "c1", 0.9),
        r("claims", "c2", 0.85),
        r("claims", "c3", 0.8),
        r("claims", "c4", 0.5, lang="hi"),
        r("claims", "c5", 0.2, lang="hi"),
    ]
    # Titles. en k1..k5 = .95 .8 .1 .2 .3 -> 2 flagged of 5 (0.4).
    #         hi k6..k10 = .1 .2 .3 .4 .5   -> 0 flagged of 5 (flag rate 0).
    #         all: 2 of 10 = 0.2.
    title_scores = [(0.95, "en"), (0.8, "en"), (0.1, "en"), (0.2, "en"), (0.3, "en")]
    title_scores += [(s, "hi") for s in (0.1, 0.2, 0.3, 0.4, 0.5)]
    titles = [
        r("corpus_title", f"k{i}", s, lang=lang, source="news.example")
        for i, (s, lang) in enumerate(title_scores, 1)
    ]
    # Text. en x6 = .9 .85 .8 .1 .1 .1 -> 3 flagged of 6 (0.5).
    #       hi x4 = .9 .1 .1 .1        -> 1 flagged of 4 (0.25).
    #       all: 4 of 10 = 0.4.
    text_scores = [(s, "en") for s in (0.9, 0.85, 0.8, 0.1, 0.1, 0.1)]
    text_scores += [(s, "hi") for s in (0.9, 0.1, 0.1, 0.1)]
    texts = [
        r("corpus_text", f"x{i}", s, lang=lang, source="news.example")
        for i, (s, lang) in enumerate(text_scores, 1)
    ]
    return claims + titles + texts


def run(models, **kwargs):
    kwargs.setdefault("universe", universe(["c1", "c2", "c3", "c4", "c5"]))
    kwargs.setdefault("publishers", PUBLISHERS)
    kwargs.setdefault("n_bootstrap", B)
    kwargs.setdefault("seed", 7)
    return evaluate(models, **kwargs)


def pair(extra=None, name="cand"):
    """A candidate and a tfidf baseline with the same hand-worked rows."""
    rows = VAL_A + make_test_rows([0.9, 0.9, 0.1, 0.1]) + small_case_rows() + (extra or [])
    return {"tfidf": model("tfidf", rows), name: model(name, rows)}
