"""The fact-check evaluation statistics, against values worked out by hand.

Each expected number below is derived in a comment from the synthetic rows, not
by calling the implementation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from modeling.eval.factcheck_eval import (
    HINDI_MIN_CLAIMS,
    EvalInputError,
    evaluate,
    lift_estimate,
    percentile_with_undefined,
    validate_frame,
)
from tests._factcheck_eval_support import (
    PUBLISHERS,
    VAL_A,
    B,
    make_test_rows,
    model,
    pair,
    r,
    run,
    small_case_rows,
    universe,
)


# ---------------------------------------------------------------------------
# Lift, recall and flag rate
# ---------------------------------------------------------------------------
def test_threshold_for_the_small_case_is_the_hand_derived_0_8():
    result = run(pair())
    op = result["models"]["cand"]["operating_point"]
    assert op["threshold"] == 0.8
    assert op["val_precision"] == 1.0 and op["val_recall"] == 1.0 and op["target_met"]


def test_lift_values_for_a_small_case():
    cells = run(pair())["cells"]["cand"]
    # all: recall 3/5 = 0.6, title flag rate 2/10 = 0.2 -> lift 3.0; text 4/10 -> 1.5
    assert cells["all"]["claims"]["recall"]["value"] == pytest.approx(0.6)
    assert cells["all"]["corpus_title"]["flag_rate"]["value"] == pytest.approx(0.2)
    assert cells["all"]["lift_title"]["value"] == pytest.approx(3.0)
    assert cells["all"]["corpus_text"]["flag_rate"]["value"] == pytest.approx(0.4)
    assert cells["all"]["lift_text"]["value"] == pytest.approx(1.5)
    # en: recall 3/3 = 1, title flag 2/5 = 0.4 -> 2.5;  text flag 3/6 = 0.5 -> 2.0
    assert cells["en"]["claims"]["recall"]["value"] == 1.0
    assert cells["en"]["lift_title"]["value"] == pytest.approx(2.5)
    assert cells["en"]["lift_text"]["value"] == pytest.approx(2.0)
    # counts behind the cells
    assert cells["en"]["claims"]["n_scored"] == 3
    assert cells["en"]["corpus_title"]["n_flagged"] == 2


def test_zero_flag_rate_gives_no_lift_and_no_crash():
    cells = run(pair())["cells"]["cand"]
    # hi titles: 0 of 5 flagged, so recall 0/2 over flag rate 0 is undefined.
    hi = cells["hi"]
    assert hi["corpus_title"]["flag_rate"]["value"] == 0.0
    lift = hi["lift_title"]
    assert lift["status"] == "undefined" and lift["value"] is None
    assert lift["ci_low"] is None and lift["ci_high"] is None
    assert "flag rate is 0" in lift["reason"]
    assert lift["n_zero_flag_resamples"] == B
    json.dumps(hi, allow_nan=False)  # no NaN and no infinity anywhere


def test_lift_estimate_zero_flag_rate_directly():
    out = lift_estimate(3, 5, 0, 10, n_bootstrap=100, seed=1)
    assert out["value"] is None and out["status"] == "undefined"


def test_resamples_with_zero_flag_rate_are_counted_and_unbound_the_upper_limit():
    # 1 flagged of 10: a resample of 10 rows has no flagged row with probability
    # 0.9 ** 10 = 0.3487, so about 697 of 2000 resamples are undefined (> 2.5%).
    out = lift_estimate(5, 5, 1, 10, n_bootstrap=2000, seed=3)
    assert 600 < out["n_zero_flag_resamples"] < 800
    assert out["value"] == pytest.approx(1.0 / 0.1)
    assert out["ci_high"] is None and "cannot be bounded" in out["reason"]
    assert out["ci_low"] is not None and np.isfinite(out["ci_low"])


def test_no_zero_flag_resamples_gives_a_finite_interval():
    # 50 flagged of 100: probability of a zero-flag resample is 2 ** -100.
    out = lift_estimate(5, 10, 50, 100, n_bootstrap=500, seed=3)
    assert out["n_zero_flag_resamples"] == 0
    assert out["reason"] is None
    assert out["value"] == pytest.approx(1.0)  # recall 0.5 / flag rate 0.5
    assert out["ci_low"] < 1.0 < out["ci_high"]


def test_percentile_with_undefined_equals_numpy_when_nothing_is_undefined():
    values = np.random.default_rng(0).random(101)
    for q in (2.5, 50, 97.5):
        assert percentile_with_undefined(values, 0, q) == pytest.approx(np.percentile(values, q))


def test_percentile_with_undefined_ranks_undefined_above_everything():
    # 8 finite values 1..8 and 2 undefined: ranks 0..9. q=50 -> position 4.5, between
    # ranks 4 and 5 (values 5 and 6) -> 5.5.  q=97.5 -> position 8.775, rank 8 is
    # undefined -> infinity.
    finite = np.arange(1.0, 9.0)
    assert percentile_with_undefined(finite, 2, 50) == pytest.approx(5.5)
    assert percentile_with_undefined(finite, 2, 97.5) == float("inf")


def test_always_flag_model_has_recall_one_flag_rate_one_lift_one():
    # Every score is 1.0: validation precision is the positive rate 1/2, below the
    # 0.85 target, so the best threshold is 1.0 and everything is flagged.
    rows = (
        [r("val", "v1", 1.0, label=1), r("val", "v2", 1.0, label=0)]
        + make_test_rows([1.0, 1.0, 1.0, 1.0])
        + [r("claims", f"c{i}", 1.0) for i in range(1, 4)]
        + [r("corpus_title", f"k{i}", 1.0, source="n.example") for i in range(1, 6)]
        + [r("corpus_text", f"x{i}", 1.0, source="n.example") for i in range(1, 6)]
    )
    models = {"tfidf": model("tfidf", rows), "always": model("always", rows)}
    result = run(models, universe=universe(["c1", "c2", "c3"]))
    cell = result["cells"]["always"]["all"]
    assert result["models"]["always"]["operating_point"]["target_met"] is False
    assert cell["claims"]["recall"]["value"] == 1.0
    assert cell["corpus_title"]["flag_rate"]["value"] == 1.0
    assert cell["lift_title"]["value"] == 1.0
    assert (cell["lift_title"]["ci_low"], cell["lift_title"]["ci_high"]) == (1.0, 1.0)
    assert any("precision" in n for n in result["notices"])


def test_thresholds_are_computed_per_model():
    # Model b has the same shape as a, with scores scaled down. Its own validation
    # rows give threshold 0.5 (precision 1, recall 1 at t = 0.5), a's give 0.8.
    val_b = [
        r("val", "v1", 0.6, label=1),
        r("val", "v2", 0.5, label=1),
        r("val", "v3", 0.2, label=0),
        r("val", "v4", 0.1, label=0),
    ]
    rest = make_test_rows([0.9, 0.9, 0.1, 0.1]) + small_case_rows()
    models = {
        "tfidf": model("tfidf", VAL_A + rest),
        "other": model("other", val_b + rest),
    }
    ops = run(models)["models"]
    assert ops["tfidf"]["operating_point"]["threshold"] == 0.8
    assert ops["other"]["operating_point"]["threshold"] == 0.5


def test_a_shared_threshold_would_change_the_cell():
    # Same rows as above. At 0.5 (model "other"), the claim scores .9 .85 .8 .5 flag,
    # i.e. 4 of 5; at 0.8 (model "tfidf") 3 of 5 flag. The cells must differ.
    val_b = [
        r("val", "v1", 0.6, label=1),
        r("val", "v2", 0.5, label=1),
        r("val", "v3", 0.2, label=0),
        r("val", "v4", 0.1, label=0),
    ]
    rest = make_test_rows([0.9, 0.9, 0.1, 0.1]) + small_case_rows()
    models = {"tfidf": model("tfidf", VAL_A + rest), "other": model("other", val_b + rest)}
    cells = run(models)["cells"]
    assert cells["tfidf"]["all"]["claims"]["n_flagged"] == 3
    assert cells["other"]["all"]["claims"]["n_flagged"] == 4


# ---------------------------------------------------------------------------
# Exclusions and null scores
# ---------------------------------------------------------------------------
def test_fact_check_publisher_rows_are_excluded_from_the_flag_rate():
    # Titles: 4 ordinary rows with 1 flagged, plus 4 rows from boomlive.in and
    # thequint.com that are all flagged. Without the exclusion the flag rate would
    # be 5/8; with it, 1/4.
    titles = [
        r("corpus_title", "n1", 0.9, source="news.example"),
        r("corpus_title", "n2", 0.1, source="news.example"),
        r("corpus_title", "n3", 0.1, source="news.example"),
        r("corpus_title", "n4", 0.1, source="news.example"),
        r("corpus_title", "p1", 0.9, source="boomlive.in"),
        r("corpus_title", "p2", 0.9, source="boomlive.in"),
        r("corpus_title", "p3", 0.9, source="www.thequint.com"),
        r("corpus_title", "p4", 0.9, source="THEQUINT.COM"),
    ]
    rows = VAL_A + make_test_rows([0.9, 0.9, 0.1, 0.1]) + [r("claims", "c1", 0.9)] + titles
    rows += [r("corpus_text", "x1", 0.1, source="news.example")]
    result = run({"tfidf": model("tfidf", rows), "m": model("m", rows)},
                 universe=universe(["c1"]))
    title = result["cells"]["m"]["all"]["corpus_title"]
    assert title["n_rows"] == 4 and title["n_flagged"] == 1
    assert title["flag_rate"]["value"] == pytest.approx(0.25)
    info = result["models"]["m"]["corpus"]["corpus_title"]
    assert info["n_excluded_fact_check_publisher"] == 4


def test_null_scores_leave_the_denominator_and_are_counted_by_reason():
    # 5 claims, c4 and c5 have no score. Flagged (>= 0.8): c1 c2 c3 of the 3 scored.
    # Recall is 3/3 = 1.0. Treating null as "not flagged" would give 3/5 = 0.6.
    rows = VAL_A + make_test_rows([0.9, 0.9, 0.1, 0.1]) + [
        r("claims", "c1", 0.9),
        r("claims", "c2", 0.85),
        r("claims", "c3", 0.8),
        r("claims", "c4", None, lang="hi", skip="unsupported_language"),
        r("claims", "c5", None, lang="hi", skip="empty_text"),
        r("corpus_title", "k1", 0.9, source="n.example"),
        r("corpus_title", "k2", None, source="n.example", skip="unsupported_language"),
        r("corpus_text", "x1", 0.9, source="n.example"),
    ]
    result = run({"tfidf": model("tfidf", rows), "m": model("m", rows)})
    claims = result["cells"]["m"]["all"]["claims"]
    assert claims["n_rows"] == 5 and claims["n_scored"] == 3 and claims["n_skipped"] == 2
    assert claims["skips"] == {"empty_text": 1, "unsupported_language": 1}
    assert claims["recall"]["value"] == 1.0
    title = result["cells"]["m"]["all"]["corpus_title"]
    assert title["n_scored"] == 1 and title["skips"] == {"unsupported_language": 1}
    assert title["flag_rate"]["value"] == 1.0


def test_a_cell_with_no_valid_score_says_no_score_with_the_reason_and_no_zero():
    # Every Hindi claim is null: the hi cell is "no score", not recall 0.
    rows = VAL_A + make_test_rows([0.9, 0.9, 0.1, 0.1]) + [
        r("claims", "c1", 0.9),
        r("claims", "c2", None, lang="hi", skip="unsupported_language"),
        r("claims", "c3", None, lang="hi", skip="unsupported_language"),
        r("corpus_title", "k1", 0.9, source="n.example"),
        r("corpus_title", "k2", None, lang="hi", source="n.example", skip="unsupported_language"),
        r("corpus_text", "x1", 0.9, source="n.example"),
    ]
    result = run({"tfidf": model("tfidf", rows), "m": model("m", rows)},
                 universe=universe(["c1", "c2", "c3"]))
    hi = result["cells"]["m"]["hi"]
    recall = hi["claims"]["recall"]
    assert recall["status"] == "no_score" and recall["value"] is None
    assert "unsupported_language=2" in recall["reason"]
    assert hi["lift_title"]["status"] == "no_score"
    assert hi["lift_title"]["value"] is None
    # No corpus rows at all for hi text is a different reason: no rows.
    assert "no corpus text rows" in hi["corpus_text"]["flag_rate"]["reason"]


def test_claim_ids_outside_the_kept_set_are_excluded_and_counted_by_reason():
    extra = [
        r("claims", "dup", 0.9),      # collapsed duplicate of c1
        r("claims", "ghost", 0.9),    # in no fact-check row
        r("claims", "later", 0.9),    # real claim added after freezing
    ]
    models = pair(extra)
    uni = universe(
        ["c1", "c2", "c3", "c4", "c5"],
        frozen=True,
        excluded={"dup": "collapsed_duplicate", "later": "not_in_frozen_set"},
    )
    result = run(models, universe=uni)
    excluded = result["models"]["cand"]["claims_excluded"]
    assert excluded == {
        "collapsed_duplicate": 1,
        "not_in_frozen_set": 1,
        "unknown_claim_id": 1,
    }
    # The three extra rows are flagged, but recall is still 3/5.
    assert result["cells"]["cand"]["all"]["claims"]["n_rows"] == 5
    assert result["cells"]["cand"]["all"]["claims"]["recall"]["value"] == pytest.approx(0.6)


def test_evaluation_claims_missing_from_the_scores_are_reported():
    result = run(pair(), universe=universe(["c1", "c2", "c3", "c4", "c5", "c6", "c7"]))
    assert result["models"]["cand"]["evaluation_claims_missing_from_scores"] == 2
    assert any("2 evaluation claim(s) have no row" in n for n in result["notices"])


# ---------------------------------------------------------------------------
# Test split, paired bootstrap, ship decision
# ---------------------------------------------------------------------------
def labelled_set(n, good_scores):
    """A balanced test set. ``good_scores``: perfect scores (0.9 for 1, 0.1 for 0) or inverted."""
    labels = [i % 2 for i in range(n)]
    scores = [(0.9 if y else 0.1) if good_scores else (0.1 if y else 0.9) for y in labels]
    pairs = zip(labels, scores, strict=True)
    return [r("test", f"t{i}", s, label=y) for i, (y, s) in enumerate(pairs)]


PERFECT_VAL = [
    r("val", "v1", 0.9, label=1),
    r("val", "v2", 0.9, label=1),
    r("val", "v3", 0.1, label=0),
    r("val", "v4", 0.1, label=0),
]
TAIL = [
    r("claims", "c1", 0.9),
    r("corpus_title", "k1", 0.1, source="n.example"),
    r("corpus_text", "x1", 0.1, source="n.example"),
]


def ship_models(candidate_good, baseline_good):
    # Validation is perfect for both, so each model's threshold is 0.9. On test the
    # "inverted" model gets every row wrong: macro-F1 0. The perfect model scores 1.
    return {
        "tfidf": model("tfidf", PERFECT_VAL + labelled_set(40, baseline_good) + TAIL),
        "cand": model("cand", PERFECT_VAL + labelled_set(40, candidate_good) + TAIL),
    }


def test_ship_when_the_interval_lower_limit_is_above_zero():
    result = run(ship_models(True, False), universe=universe(["c1"]))
    comparison = result["comparisons"]["cand"]
    assert comparison["difference"]["value"] == 1.0  # F1 1.0 minus F1 0.0
    assert comparison["difference"]["ci_low"] == 1.0 > 0
    assert comparison["ship"] is True


def test_do_not_ship_when_the_interval_lies_below_zero():
    result = run(ship_models(False, True), universe=universe(["c1"]))
    comparison = result["comparisons"]["cand"]
    assert comparison["difference"]["value"] == -1.0
    assert comparison["difference"]["ci_high"] < 0
    assert comparison["ship"] is False


def test_a_model_compared_with_an_identical_copy_has_interval_exactly_zero_and_does_not_ship():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 60)
    scores = np.where(rng.random(60) < 0.75, labels * 0.8 + 0.1, (1 - labels) * 0.8 + 0.1)
    test = [
        r("test", f"t{i}", float(s), label=int(y))
        for i, (y, s) in enumerate(zip(labels, scores, strict=True))
    ]
    rows = PERFECT_VAL + test + TAIL
    result = run({"tfidf": model("tfidf", rows), "copy": model("copy", rows)},
                 universe=universe(["c1"]))
    diff = result["comparisons"]["copy"]["difference"]
    assert (diff["value"], diff["ci_low"], diff["ci_high"]) == (0.0, 0.0, 0.0)
    assert result["comparisons"]["copy"]["ship"] is False  # 0 is not above 0


def test_null_test_scores_leave_the_pair_and_are_counted():
    base_rows = PERFECT_VAL + labelled_set(40, True) + TAIL
    cand_rows = [
        {**row, "score": None, "skip_reason": "unsupported_language"}
        if row["part"] == "test" and row["id"] in {"t0", "t1", "t2"}
        else row
        for row in base_rows
    ]
    result = run({"tfidf": model("tfidf", base_rows), "cand": model("cand", cand_rows)},
                 universe=universe(["c1"]))
    comparison = result["comparisons"]["cand"]
    assert comparison["n_paired"] == 37 and comparison["n_left_out"] == 3
    assert result["models"]["cand"]["test"]["skips"] == {"unsupported_language": 3}
    assert result["models"]["cand"]["test"]["n_scored"] == 37


def test_test_macro_f1_per_model_hand_value():
    # TEST_ROWS labels 1 1 0 0, scores .9 .9 .1 .1 with threshold .8 -> predictions
    # 1 1 0 0, perfect: macro-F1 1.0.  Scores .9 .1 .9 .1 -> predictions 1 0 1 0:
    # class 1 tp1 fp1 fn1 -> 2/4 = .5; class 0 tp1 fp1 fn1 -> .5; macro .5.
    rows_a = VAL_A + make_test_rows([0.9, 0.9, 0.1, 0.1]) + small_case_rows()
    rows_b = VAL_A + make_test_rows([0.9, 0.1, 0.9, 0.1]) + small_case_rows()
    result = run({"tfidf": model("tfidf", rows_a), "m": model("m", rows_b)})
    assert result["models"]["tfidf"]["test"]["macro_f1"]["value"] == 1.0
    assert result["models"]["m"]["test"]["macro_f1"]["value"] == 0.5
    # Only 4 paired rows: fewer than 10, so no interval and no ship.
    comparison = result["comparisons"]["m"]
    assert comparison["ship"] is False and "fewer than 10" in comparison["reason"]


def test_missing_baseline_refuses():
    models = pair()
    del models["tfidf"]
    with pytest.raises(EvalInputError, match="baseline model 'tfidf'"):
        run(models)


def test_no_publishers_refuses():
    with pytest.raises(EvalInputError, match="no fact-check publishers"):
        run(pair(), publishers=frozenset())


# ---------------------------------------------------------------------------
# Determinism and notices
# ---------------------------------------------------------------------------
def dump(result):
    return json.dumps(result, sort_keys=True, allow_nan=False)


def test_same_seed_gives_identical_output_and_a_different_seed_does_not():
    one = dump(run(pair(), seed=11))
    two = dump(run(pair(), seed=11))
    other = dump(run(pair(), seed=12))
    assert one == two
    assert one != other


def test_output_does_not_depend_on_row_order():
    base = pair()
    shuffled = {}
    for name, scores in base.items():
        frame = scores.frame.sample(frac=1.0, random_state=4).reset_index(drop=True)
        shuffled[name] = validate_frame(frame, name)
    assert dump(run(base)) == dump(run(shuffled))


def test_unfrozen_set_is_stated_in_the_notices_and_frozen_is_not():
    assert any("evaluation set not frozen" in n for n in run(pair())["notices"])
    frozen = run(pair(), universe=universe(["c1", "c2", "c3", "c4", "c5"], frozen=True))
    assert not any("not frozen" in n for n in frozen["notices"])
    assert frozen["evaluation_set"]["frozen"] is True


def test_a_claim_links_threshold_that_differs_from_config_is_noticed():
    uni = universe(["c1", "c2", "c3", "c4", "c5"])
    uni.claim_dedup = {
        "passes": {"embedding": {"status": "ok", "collapsed": 1, "threshold": 0.5}}
    }
    notices = run(pair(), universe=uni)["notices"]
    assert any("claim links artifact used cosine >= 0.5" in n for n in notices)
    assert any("dedup-claims" in n for n in notices)


def test_result_states_what_the_evaluation_set_contains():
    statement = run(pair())["evaluation_set"]["statement"]
    assert "only the claims that publishers chose to debunk" in statement


def test_default_resamples_is_2000():
    from modeling.eval.factcheck_eval import DEFAULT_N_BOOTSTRAP

    assert DEFAULT_N_BOOTSTRAP == 2000


def test_the_project_seed_is_the_default_seed():
    from modeling.config import get_settings

    result = evaluate(
        pair(),
        universe=universe(["c1", "c2", "c3", "c4", "c5"]),
        publishers=PUBLISHERS,
        n_bootstrap=50,
    )
    assert result["seed"] == get_settings().seed


# ---------------------------------------------------------------------------
# Hindi gate
# ---------------------------------------------------------------------------
def hindi_rows(n_claims, claim_flag_all=True, corpus_flagged=100, corpus_n=1000):
    """Hindi claims plus a Hindi title corpus with a known flag rate.

    Threshold is 0.8 (VAL_A). Every claim scores 0.9 (flagged) or, when
    ``claim_flag_all`` is false, flagged only at the corpus rate.
    """
    claims = []
    for i in range(n_claims):
        flagged = claim_flag_all or i < round(n_claims * corpus_flagged / corpus_n)
        claims.append(r("claims", f"h{i}", 0.9 if flagged else 0.1, lang="hi"))
    titles = [
        r("corpus_title", f"hk{i}", 0.9 if i < corpus_flagged else 0.1, lang="hi",
          source="news.example")
        for i in range(corpus_n)
    ]
    texts = [r("corpus_text", "hx0", 0.1, lang="hi", source="news.example")]
    return claims, titles, texts


def gate_for(n_claims, **kwargs):
    claims, titles, texts = hindi_rows(n_claims, **kwargs)
    rows = VAL_A + make_test_rows([0.9, 0.9, 0.1, 0.1]) + claims + titles + texts
    result = run(
        {"tfidf": model("tfidf", rows), "m": model("m", rows)},
        universe=universe([c["id"] for c in claims]),
    )
    return result["hindi_gate"]["m"]


def test_hindi_gate_fails_at_49_claims_and_passes_at_50():
    # Recall 1.0 over flag rate 100/1000 = 0.1 gives lift 10. A resample of 1000
    # titles has about 100 +/- 9.5 flagged, so the lower limit stays far above 1.
    below = gate_for(HINDI_MIN_CLAIMS - 1)
    at = gate_for(HINDI_MIN_CLAIMS)
    assert below["n_hindi_claims_scored"] == 49
    assert below["lift_condition_met"] is True and below["claims_condition_met"] is False
    assert below["pass"] is False and "49 Hindi claims" in below["reason"]
    assert at["n_hindi_claims_scored"] == 50
    assert at["pass"] is True
    assert at["lift"] == pytest.approx(10.0) and at["lift_ci_low"] > 1


def test_hindi_gate_fails_when_the_lift_interval_reaches_one():
    # 50 claims, 10% of them flagged, against a 10% flag rate: lift 1.0 and the
    # lower limit is below 1.
    gate = gate_for(50, claim_flag_all=False)
    assert gate["n_hindi_claims_scored"] == 50 and gate["claims_condition_met"]
    assert gate["lift"] == pytest.approx(1.0)
    assert gate["lift_ci_low"] < 1
    assert gate["pass"] is False and "not above 1" in gate["reason"]


def test_hindi_gate_with_no_hindi_score_fails_with_the_reason():
    rows = VAL_A + make_test_rows([0.9, 0.9, 0.1, 0.1]) + [
        r("claims", "h0", None, lang="hi", skip="unsupported_language"),
        r("corpus_title", "k0", 0.9, source="n.example"),
        r("corpus_text", "x0", 0.9, source="n.example"),
    ]
    result = run({"tfidf": model("tfidf", rows), "m": model("m", rows)},
                 universe=universe(["h0"]))
    gate = result["hindi_gate"]["m"]
    assert gate["pass"] is False and gate["n_hindi_claims_scored"] == 0
    assert gate["lift_ci_low"] is None


# ---------------------------------------------------------------------------
# Control models: reported as a baseline, never as a candidate to ship
# ---------------------------------------------------------------------------
def control_result(name="laya_multi_base", **kwargs):
    claims, titles, texts = hindi_rows(HINDI_MIN_CLAIMS)
    # A candidate that is perfect on test while the baseline is inverted: it clears the
    # ship rule, and with 50 Hindi claims at lift 10 it clears the Hindi gate as well.
    rows = PERFECT_VAL + claims + titles + texts
    models = {
        "tfidf": model("tfidf", rows + labelled_set(40, False)),
        name: model(name, rows + labelled_set(40, True)),
    }
    return run(models, universe=universe([c["id"] for c in claims]), **kwargs)


def test_the_zero_shot_control_is_never_shippable_whatever_its_numbers():
    result = control_result()
    comparison = result["comparisons"]["laya_multi_base"]
    assert comparison["difference"]["ci_low"] > 0  # the rule would have shipped it
    assert comparison["ship_by_numbers"] is True
    assert comparison["ship"] is False and comparison["control"] is True
    assert "control model" in comparison["reason"]
    gate = result["hindi_gate"]["laya_multi_base"]
    assert gate["pass_by_numbers"] is True
    assert gate["pass"] is False and gate["control"] is True
    # The numbers stay in the report as a baseline.
    assert gate["lift"] == pytest.approx(10.0)
    assert result["models"]["laya_multi_base"]["test"]["macro_f1"]["value"] == 1.0
    assert result["controls"] == ["laya_multi_base"]


def test_the_report_says_do_not_ship_and_fail_control_for_the_control():
    from modeling.eval.factcheck_eval import render_markdown

    text = render_markdown(control_result())
    assert "**do not ship (control)**" in text
    assert "**FAIL (control)**" in text
    assert "laya_multi_base (control)" in text


def test_the_report_shows_fact_check_rows_and_evaluation_claims_per_tier():
    from modeling.eval.factcheck_eval import render_markdown

    result = control_result()
    result["evaluation_set"]["rows_per_tier"] = {"claimreview": 7, "dom_block": 3}
    result["evaluation_set"]["claims_per_tier"] = {"claimreview": 7}
    text = render_markdown(result)
    assert "Fact-check rows per extraction tier: `claimreview` 7, `dom_block` 3" in text
    assert "Evaluation claims per extraction tier: `claimreview` 7" in text
    assert result["evaluation_set"]["rows_per_tier"] == {"claimreview": 7, "dom_block": 3}
    assert result["evaluation_set"]["claims_per_tier"] == {"claimreview": 7}


def test_the_report_hides_the_claim_line_when_it_matches_the_rows():
    from modeling.eval.factcheck_eval import render_markdown

    result = control_result()
    result["evaluation_set"]["rows_per_tier"] = {"dom_block": 3}
    result["evaluation_set"]["claims_per_tier"] = {"dom_block": 3}
    text = render_markdown(result)
    assert "Fact-check rows per extraction tier: `dom_block` 3" in text
    assert "Evaluation claims per extraction tier" not in text


def test_the_result_carries_the_per_tier_counts_of_the_universe():
    claims, titles, texts = hindi_rows(HINDI_MIN_CLAIMS)
    ids = [c["id"] for c in claims]
    rows = PERFECT_VAL + claims + titles + texts
    models = {
        "tfidf": model("tfidf", rows + labelled_set(40, False)),
        "m": model("m", rows + labelled_set(40, True)),
    }
    base = universe(ids)
    base.tier_counts = {"dom_block": len(ids)}
    base.eval_tier_counts = {"dom_block": len(ids)}
    result = run(models, universe=base)
    assert result["evaluation_set"]["rows_per_tier"] == {"dom_block": len(ids)}
    assert result["evaluation_set"]["claims_per_tier"] == {"dom_block": len(ids)}


def test_a_model_that_is_not_a_control_still_ships_on_the_same_numbers():
    result = control_result(name="laya_multi")
    assert result["comparisons"]["laya_multi"]["ship"] is True
    assert "control" not in result["comparisons"]["laya_multi"]
    assert result["hindi_gate"]["laya_multi"]["pass"] is True
    assert result["controls"] == []


def test_the_baseline_cannot_be_a_control():
    with pytest.raises(EvalInputError, match="control"):
        control_result(controls=["tfidf"])


# ---------------------------------------------------------------------------
# corpus_text rates are not like for like across model families
# ---------------------------------------------------------------------------
def test_every_corpus_text_flag_rate_and_lift_carries_the_notice_and_title_does_not():
    from modeling.eval.factcheck_eval import CORPUS_TEXT_NOTICE, render_markdown

    result = run(pair())
    text = render_markdown(result)
    title_part, text_part = text.split("## Recall, flag rate and lift: corpus full text")
    text_part = text_part.split("## Hindi gate")[0]
    title_part = title_part.split("## Recall, flag rate and lift: corpus titles")[1]

    assert CORPUS_TEXT_NOTICE in text_part and CORPUS_TEXT_NOTICE not in title_part
    assert "predict_long" in CORPUS_TEXT_NOTICE and "256 tokens" in CORPUS_TEXT_NOTICE
    rows = [line for line in text_part.splitlines() if line.startswith("| ") and "---" not in line]
    body = rows[1:]  # the first row is the header
    assert len(body) == 2 * 3  # two models, languages en, hi and all
    for row in body:
        flag_cell, lift_cell = row.split(" | ")[5], row.split(" | ")[6]
        assert flag_cell.endswith("(not like for like)"), row
        assert lift_cell.endswith("(not like for like)"), row
    assert "(not like for like)" not in title_part
    assert result["evaluation_set"]["corpus_text_notice"] == CORPUS_TEXT_NOTICE


def _domain_rows():
    return [
        r("corpus_title", "d1", 0.1, source="www.BoomLive.in"),
        r("corpus_title", "d2", 0.1, source="boomlive.in"),
        r("corpus_title", "d3", 0.1, source="thequint.com"),
        r("corpus_text", "d4", 0.1, source="boomlive.in"),
        r("corpus_text", "d5", 0.1, source="news.example"),
    ]


def test_exclusions_are_counted_per_domain_with_a_zero_domain():
    models = pair(_domain_rows())
    publishers = frozenset({"boomlive.in", "thequint.com", "altnews.in"})
    result = run(models, publishers=publishers)
    corpus = result["models"]["cand"]["corpus"]
    assert corpus["corpus_title"]["excluded_fact_check_publisher_by_domain"] == {
        "altnews.in": 0, "boomlive.in": 2, "thequint.com": 1,
    }
    assert corpus["corpus_text"]["excluded_fact_check_publisher_by_domain"] == {
        "altnews.in": 0, "boomlive.in": 1, "thequint.com": 0,
    }
    for part in ("corpus_title", "corpus_text"):
        info = corpus[part]
        assert sum(info["excluded_fact_check_publisher_by_domain"].values()) == (
            info["n_excluded_fact_check_publisher"]
        )


def test_exclusion_sum_check_fails_loudly_on_a_mismatch():
    from modeling.eval import factcheck_eval as fe

    fe._check_domain_sum("corpus_title", {"a.in": 1, "b.in": 2}, 3)
    with pytest.raises(fe.EvalInputError, match="add up"):
        fe._check_domain_sum("corpus_title", {"a.in": 1, "b.in": 2}, 4)


def test_markdown_has_one_table_row_per_fact_checker_domain():
    from modeling.eval.factcheck_eval import render_markdown

    publishers = frozenset({"boomlive.in", "thequint.com", "altnews.in"})
    md = render_markdown(run(pair(_domain_rows()), publishers=publishers))
    assert "### Corpus rows excluded per fact-checker domain" in md
    assert "| domain | corpus_title | corpus_text |" in md
    assert "| altnews.in | 0 | 0 |" in md
    assert "| boomlive.in | 2 | 1 |" in md
    assert "| thequint.com | 1 | 0 |" in md
    assert "| total | 3 | 1 |" in md


def test_publishers_are_normalised_once_so_case_and_www_still_match():
    result = run(pair(_domain_rows()), publishers=frozenset({"WWW.BoomLive.in", "thequint.com"}))
    by_domain = result["models"]["cand"]["corpus"]["corpus_title"][
        "excluded_fact_check_publisher_by_domain"
    ]
    assert by_domain == {"boomlive.in": 2, "thequint.com": 1}


def test_a_fact_checker_subdomain_is_not_excluded_but_is_counted():
    rows = [r("corpus_title", "s1", 0.1, source="hindi.boomlive.in")]
    publishers = frozenset({"boomlive.in"})
    result = run(pair(_domain_rows() + rows), publishers=publishers)
    title = result["models"]["cand"]["corpus"]["corpus_title"]
    assert title["fact_check_subdomain_not_excluded"] == 1
    assert title["excluded_fact_check_publisher_by_domain"] == {"boomlive.in": 2}
    text = result["models"]["cand"]["corpus"]["corpus_text"]
    assert text["fact_check_subdomain_not_excluded"] == 0
    from modeling.eval.factcheck_eval import render_markdown

    md = render_markdown(result)
    assert "1/0 corpus_title/corpus_text rows are on a subdomain" in md
    assert "fact_check_subdomain_not_excluded" in md


def test_report_on_the_real_config_has_eight_domain_rows():
    from modeling.eval.factcheck_eval import load_factcheck_publishers, render_markdown

    publishers = load_factcheck_publishers()
    assert len(publishers) == 8
    md = render_markdown(run(pair(_domain_rows()), publishers=publishers))
    block = md[md.index("### Corpus rows excluded") :]
    rows = [ln for ln in block.split("\n\n")[2].splitlines()[2:] if not ln.startswith("| total")]
    assert len(rows) == 8


def test_models_that_differ_get_one_block_each_with_escaped_names():
    from modeling.eval.factcheck_eval import render_markdown

    # The evaluator requires equal corpus inputs, so edit the result to reach the branch.
    result = run(pair(_domain_rows(), name="c|d"), publishers=frozenset({"boomlive.in"}))
    other = result["models"]["tfidf"]["corpus"]["corpus_title"]
    other["excluded_fact_check_publisher_by_domain"] = {"boomlive.in": 0}
    md = render_markdown(result)
    assert "| model | domain | corpus_title | corpus_text |" in md
    assert "| c\\|d | boomlive.in | 2 | 1 |" in md
    assert "| tfidf | boomlive.in | 0 | 1 |" in md
    assert "| tfidf | total | 0 | 1 |" in md


def test_a_model_with_no_corpus_rows_is_named_under_the_table():
    from modeling.eval.factcheck_eval import render_markdown

    rows = VAL_A + make_test_rows([0.9, 0.9, 0.1, 0.1]) + small_case_rows()
    empty = [x for x in rows if not x["part"].startswith("corpus")]
    models = {"tfidf": model("tfidf", empty), "cand": model("cand", empty)}
    md = render_markdown(run(models, publishers=frozenset({"boomlive.in"})))
    assert "cand has 0 corpus rows, so its counts are 0 by absence." in md
