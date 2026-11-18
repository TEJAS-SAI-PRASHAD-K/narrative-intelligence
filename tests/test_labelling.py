"""Weighted sampling, the labeller gate, agreement, weighted metrics and Laya Hindi transfer.

All data is synthetic. No model runs: scores are written into a fake ``record_scores``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from modeling import cli as modeling_cli
from modeling.eval import labelling as lb
from modeling.eval import laya_transfer as lt
from modeling.eval import weighted as wt

runner = CliRunner()
LONG = "a sentence long enough to be scored by the stage"
FC = frozenset({"altnews.in", "boomlive.in"})


def allow(lang):
    return lang in ("en", "hi")


def config(**kw):
    return lb.parse_sampling_config(
        {
            "seed": 20260813,
            "likely_positive_sources": ["reddit", "mastodon"],
            "strata": [
                {"name": "claim_match", "share": 0.4},
                {"name": "reddit", "share": 0.2},
                {"name": "mastodon", "share": 0.2},
                {"name": "other", "share": 0.2},
            ],
            **kw,
        }
    )


def raw(title):
    return json.dumps({"title": title})


def corpus():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        rows.append(
            (
                f"r{i}",
                "reddit",
                "reddit.com",
                "en",
                f"{LONG} r{i}",
                raw(f"t r{i}"),
                int(rng.integers(1 << 40, 1 << 62)),
            )
        )
    for i in range(10):
        rows.append(
            (
                f"m{i}",
                "mastodon",
                "mastodon.social",
                "en",
                f"{LONG} m{i}",
                raw("t"),
                int(rng.integers(1 << 40, 1 << 62)),
            )
        )
    for i in range(40):
        rows.append(
            (
                f"n{i}",
                "news",
                "thehindu.com",
                "en",
                f"{LONG} n{i}",
                raw("t"),
                int(rng.integers(1 << 40, 1 << 62)),
            )
        )
    for i in range(5):
        rows.append((f"f{i}", "news", "www.altnews.in", "en", f"{LONG} f{i}", raw("t"), 0))
    for i in range(12):
        title = "" if i < 2 else f"यह एक पर्याप्त लंबा हिंदी शीर्षक है {i}"
        rows.append(
            (
                f"h{i}",
                "news",
                "bbc.com",
                "hi",
                f"{LONG} hindi {i}",
                raw(title),
                int(rng.integers(1 << 40, 1 << 62)),
            )
        )
    rows.append(
        ("short", "news", "x.com", "en", "tiny", raw("t"), int(rng.integers(1 << 40, 1 << 62)))
    )
    rows[3] = (*rows[3][:6], R3)
    cols = ["id", "source", "source_detail", "lang", "text", "raw", "simhash"]
    return pd.DataFrame(rows, columns=cols)


# the claim hash differs from record r3 by two bits and from every other record by many
R3 = 123456789123456789
CLAIM = R3 ^ 0b11


class TestSimhashMatches:
    def test_within_radius_matches_and_beyond_does_not(self):
        found = lb.simhash_matches(["a", "b", "c"], [R3, R3 ^ 0b11110000, 0], [CLAIM], 3)
        assert found == {"a": 2}

    def test_a_zero_claim_hash_is_no_signal(self):
        assert lb.simhash_matches(["a"], [0], [0], 3) == {}

    def test_high_bits_above_2_to_63_work(self):
        big = (1 << 63) | 5
        assert lb.simhash_matches(["a"], [big], [big ^ 1], 3) == {"a": 1}

    def test_embedding_matches_by_cosine(self):
        claims = np.array([[1.0, 0.0]])
        records = np.array([[0.99, 0.1], [0.0, 1.0]])
        assert lb.embedding_matches(["a", "b"], records, claims, 0.9) == {"a"}


class TestAllocate:
    def test_follows_shares_and_totals_n(self):
        alloc = lb.allocate({"a": 100, "b": 100, "c": 100}, {"a": 0.5, "b": 0.25, "c": 0.25}, 20)
        assert sum(alloc.values()) == 20 and alloc["a"] == 10

    def test_a_small_pool_gives_its_share_to_the_others(self):
        alloc = lb.allocate({"a": 2, "b": 100}, {"a": 0.5, "b": 0.5}, 20)
        assert alloc == {"a": 2, "b": 18}

    def test_empty_strata_are_dropped(self):
        assert lb.allocate({"a": 0, "b": 5}, {"a": 0.5, "b": 0.5}, 3) == {"b": 3}

    def test_n_below_the_number_of_live_strata_is_refused_with_the_reason(self):
        with pytest.raises(lb.AllocationError, match="below the 3 strata"):
            lb.allocate({"a": 9, "b": 9, "c": 9}, {"a": 0.5, "b": 0.3, "c": 0.2}, 2)

    def test_every_live_stratum_gets_at_least_one_row(self):
        alloc = lb.allocate({"a": 90, "b": 90, "c": 90}, {"a": 0.98, "b": 0.01, "c": 0.01}, 10)
        assert min(alloc.values()) >= 1 and sum(alloc.values()) == 10


class TestBuildWeightedSample:
    def draw(self, **kw):
        kw.setdefault("claims", lb.ClaimEvidence(hashes=[CLAIM]))
        return lb.build_weighted_sample(
            corpus(),
            n=kw.pop("n", 20),
            language_allowed=allow,
            fact_checker_domains=FC,
            config=config(),
            lang=kw.pop("lang", "en"),
            **kw,
        )

    def test_fact_checker_articles_never_enter_and_are_counted(self):
        built = self.draw(n=60)
        assert not any(i.startswith("f") for i in built.frame["id"])
        assert built.excluded[lb.FACT_CHECK_PUBLISHER] == 5
        assert built.excluded["text_too_short"] == 1

    def test_claim_matches_land_in_the_first_stratum(self):
        built = self.draw(n=60)
        design = built.design.set_index("id")
        assert design.loc["r3", "stratum"] == "S1"
        assert built.strata["S1"] == {"name": "claim_match", "pool": 1, "drawn": 1}

    def test_design_weight_is_pool_over_drawn_and_sums_to_the_pool(self):
        built = self.draw(n=20)
        design = built.design
        assert list(design["id"]) == list(built.frame["id"])
        for code, meta in built.strata.items():
            rows = design[design["stratum"] == code]
            assert len(rows) == meta["drawn"]
            if meta["drawn"]:
                assert rows["weight"].astype(float).sum() == pytest.approx(meta["pool"], rel=1e-4)

    def test_the_sheet_has_no_design_column_and_its_label_columns_are_blank(self):
        frame = self.draw().frame
        assert list(frame.columns) == list(lb.SHEET_COLUMNS)
        assert not {"stratum", "weight", "pool", "drawn"} & set(frame.columns)
        assert (frame[list(lb.BLANK_COLUMNS)] == "").all().all()
        assert (frame["sample_seed"] == 20260813).all()

    def test_row_order_is_not_grouped_by_stratum(self):
        built = self.draw(n=60)
        order = list(built.design["stratum"])
        runs = 1 + sum(a != b for a, b in zip(order, order[1:], strict=False))
        # Grouped rows give one run per stratum. A shuffle gives many more.
        assert runs > 2 * built.design["stratum"].nunique()

    def test_same_seed_same_rows_and_a_different_seed_differs(self):
        a, b = self.draw(seed=1), self.draw(seed=1)
        c = self.draw(seed=2)
        assert list(a.frame["id"]) == list(b.frame["id"])
        assert list(a.frame["id"]) != list(c.frame["id"])
        assert (c.frame["sample_seed"] == 2).all()

    def test_no_embeddings_is_a_counted_degradation_not_a_silent_one(self):
        built = self.draw()
        assert built.degraded[lb.EMBEDDING_UNAVAILABLE] == 1
        assert any("embedding match path did not run" in n for n in built.notes)

    def test_embedding_path_adds_matches(self):
        ids = [f"n{i}" for i in range(40)]
        matrix = np.zeros((40, 2), dtype=np.float32)
        matrix[:, 1] = 1.0
        matrix[7] = [1.0, 0.0]  # n7 matches the claim by embedding only
        evidence = lb.ClaimEvidence(
            hashes=[CLAIM],
            claim_matrix=np.array([[1.0, 0.0]], dtype=np.float32),
            record_ids=ids,
            record_matrix=matrix,
            min_similarity=0.85,
            embedding_reason=None,
        )
        built = self.draw(n=100, claims=evidence)
        assert built.claim_match_counts == {"simhash": 1, "embedding": 1, "both": 0, "any": 2}
        assert built.strata["S1"]["pool"] == 2
        assert built.degraded[lb.EMBEDDING_RECORD_MISSING] == 40

    def test_no_claims_on_disk_is_counted(self):
        built = self.draw(claims=lb.ClaimEvidence())
        assert built.degraded[lb.NO_FALSE_CLAIMS] == 1
        assert built.strata["S1"]["pool"] == 0

    def test_hindi_uses_the_laya_skip_logic_on_the_title(self):
        records = corpus()
        # h2 has a title under Laya's MIN_CHARS, h3 has a fact-checker source.
        records.loc[records["id"] == "h2", "raw"] = raw("छोटा")
        records.loc[records["id"] == "h3", "source_detail"] = "boomlive.in"
        built = lb.build_weighted_sample(
            records,
            n=20,
            language_allowed=allow,
            fact_checker_domains=FC,
            config=config(),
            lang="hi",
        )
        assert built.excluded == {
            lb.NO_TITLE: 2,
            "text_too_short": 1,
            lb.FACT_CHECK_PUBLISHER: 1,
        }
        assert set(built.frame["id"]).isdisjoint({"h0", "h1", "h2", "h3"})
        assert (built.frame["title"].str.len() >= 20).all()

    def test_a_pool_with_no_claim_match_warns_and_records_the_zero(self):
        built = self.draw(claims=lb.ClaimEvidence(hashes=[1 << 40]))
        assert built.claim_match_counts["any"] == 0
        assert any(n.startswith("WARNING: no scorable record matches") for n in built.notes)

    def test_n_below_the_live_strata_is_refused(self):
        with pytest.raises(lb.AllocationError):
            self.draw(n=2)

    def test_all_strata_empty_returns_an_empty_frame(self):
        built = lb.build_weighted_sample(
            corpus().iloc[0:0],
            n=5,
            language_allowed=allow,
            fact_checker_domains=FC,
            config=config(),
        )
        assert len(built.frame) == 0


class TestConfigFile:
    def test_the_shipped_config_parses(self):
        cfg = lb.load_sampling_config()
        assert cfg.seed == lb.DEFAULT_SEED == 20260813
        assert {"reddit", "mastodon", "youtube", "telegram"} <= set(cfg.likely_sources)
        assert cfg.simhash_hamming == 3 and cfg.min_similarity == 0.85

    def test_a_config_without_other_is_rejected(self):
        with pytest.raises(ValueError):
            lb.parse_sampling_config({"strata": [{"name": "reddit", "share": 1}]})


# ---------------------------------------------------------------------------
# the labeller gate
# ---------------------------------------------------------------------------
def sheet_frame(kind="human", n=6):
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"x{i}",
                "lang": "en",
                "label_a": str(i % 2),
                "label_b": str(i % 2),
                "reports_on_false_claim_a": "0",
                "reports_on_false_claim_b": "0",
                "adjudicated_label": str(i % 2),
                "adjudicated_reports_on_false_claim": "1" if i == 0 else "0",
                "labeller_kind": kind,
            }
        )
    return pd.DataFrame(rows)


class TestResolveSheet:
    def test_human_labels_pass_with_no_note(self):
        out = lb.resolve_sheet(sheet_frame(), design=design_of(sheet_frame()))
        assert out.refusal is None and out.labeller_note is None and out.has_design
        assert list(out.frame["label"]) == ["0", "1", "0", "1", "0", "1"]
        assert out.frame["reports"].iloc[0] == 1

    @pytest.mark.parametrize("kind", ["model-unreviewed", "llm-unreviewed"])
    def test_model_labels_are_refused_by_default(self, kind):
        out = lb.resolve_sheet(sheet_frame(kind))
        assert lb.MODEL_LABELS_REFUSED in out.refusal

    def test_opt_in_allows_model_labels_and_sets_the_note(self):
        out = lb.resolve_sheet(sheet_frame("model-unreviewed"), allow_model_labels=True)
        assert out.refusal is None
        assert out.labeller_note == "labeller: model, unreviewed by a person"

    def test_a_mixed_sheet_is_refused_without_the_flag(self):
        frame = sheet_frame()
        frame.loc[0, "labeller_kind"] = "model-unreviewed"
        assert lb.resolve_sheet(frame).refusal
        assert lb.resolve_sheet(frame, allow_model_labels=True).labeller_note

    @pytest.mark.parametrize("flag", [False, True])
    def test_an_unmarked_labelled_row_is_refused_even_with_the_flag(self, flag):
        out = lb.resolve_sheet(sheet_frame(""), allow_model_labels=flag)
        assert lb.UNMARKED_LABELLER_KIND in out.refusal

    def test_a_blank_label_needs_no_kind(self):
        frame = sheet_frame("")
        frame["adjudicated_label"] = ""
        assert lb.resolve_sheet(frame).refusal is None

    def test_a_legacy_sheet_reads_the_label_column(self):
        frame = sheet_frame().rename(columns={"adjudicated_label": "label"})
        out = lb.resolve_sheet(frame)
        assert out.refusal is None and list(out.frame["label"])[:2] == ["0", "1"]

    def test_a_labelled_row_without_a_design_row_is_refused(self):
        frame = sheet_frame()
        design = design_of(frame).iloc[1:]
        assert lb.DESIGN_MISSING in lb.resolve_sheet(frame, design=design).refusal

    def test_without_a_design_file_the_sheet_is_read_unweighted(self):
        out = lb.resolve_sheet(sheet_frame())
        assert out.refusal is None and not out.has_design and "stratum" not in out.frame

    def test_the_design_joins_on_id_whatever_the_row_order(self):
        frame = sheet_frame()
        out = lb.resolve_sheet(frame.iloc[::-1], design=design_of(frame))
        assert list(out.frame["id"]) == list(frame.iloc[::-1]["id"])
        assert out.frame.set_index("id").loc["x1", "stratum"] == "S2"


def design_of(frame):
    """Rows x0, x2, x4 are in stratum S1 (pool 12), the rest in S2 (pool 6)."""
    first = [i % 2 == 0 for i in range(len(frame))]
    return pd.DataFrame(
        {
            "id": list(frame["id"]),
            "stratum": ["S1" if f else "S2" for f in first],
            "pool": ["12" if f else "6" for f in first],
            "drawn": ["3"] * len(frame),
            "weight": ["4" if f else "2" for f in first],
        }
    )


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------
class TestAgreement:
    def test_kappa_matches_a_hand_calculation(self):
        # 10 pairs: both 1 on 4, both 0 on 4, a=1 b=0 on 1, a=0 b=1 on 1.
        a = np.array([1] * 4 + [0] * 4 + [1, 0])
        b = np.array([1] * 4 + [0] * 4 + [0, 1])
        # observed 0.8, expected 0.5 * 0.5 + 0.5 * 0.5 = 0.5, kappa 0.6
        assert lb.cohen_kappa(a, b) == pytest.approx(0.6)

    def test_kappa_is_undefined_when_both_use_one_class(self):
        assert lb.cohen_kappa(np.ones(5, dtype=int), np.ones(5, dtype=int)) is None

    def test_perfect_disagreement_is_negative(self):
        a = np.array([1, 0, 1, 0])
        assert lb.cohen_kappa(a, 1 - a) == pytest.approx(-1.0)

    def test_the_bootstrap_is_seeded(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, 60)
        b = np.where(rng.random(60) < 0.8, a, 1 - a)
        one = lb.kappa_interval(a, b, n_bootstrap=300)
        two = lb.kappa_interval(a, b, n_bootstrap=300)
        assert (one.low, one.high) == (two.low, two.high)
        assert one.low < one.value < one.high

    def test_report_lists_disagreements_and_counts_incomplete_rows(self):
        frame = sheet_frame(n=12)
        frame.loc[3, "label_b"] = "1"  # x3 was 1 for a already, so flip a row where they differ
        frame.loc[4, "label_b"] = "1"  # x4: a 0, b 1
        frame.loc[5, "label_b"] = ""  # incomplete
        report = lb.agreement_report(frame, n_bootstrap=100)
        item = report["label"]
        assert item.n_pairs == 11 and item.incomplete == 1
        assert [d["id"] for d in item.disagreements] == ["x4"]
        assert item.raw_agreement == pytest.approx(10 / 11)
        assert any("x4" in line for line in lb.render_agreement(report, 20260813))

    def test_fill_agreed_leaves_disagreements_blank(self):
        frame = sheet_frame()
        frame["adjudicated_label"] = ""
        frame["adjudicated_reports_on_false_claim"] = ""
        frame.loc[1, "label_b"] = "0"
        filled, counts = lb.fill_agreed(frame)
        assert filled.loc[1, "adjudicated_label"] == ""
        assert filled.loc[0, "adjudicated_label"] == "0"
        assert counts["left_for_adjudication"] == 1


# ---------------------------------------------------------------------------
# weighted metrics
# ---------------------------------------------------------------------------
class TestWeighted:
    def test_rates_match_a_hand_calculation(self):
        y = np.array([1, 1, 0, 0])
        flag = np.array([1, 0, 1, 0])
        w = np.array([1.0, 3.0, 2.0, 4.0])
        r = wt.rates(y, flag, w)
        assert r["positive_rate"] == pytest.approx(0.4)
        assert r["flag_rate"] == pytest.approx(0.3)
        assert r["precision"] == pytest.approx(1 / 3)
        assert r["recall"] == pytest.approx(0.25)

    def test_an_empty_denominator_is_none_not_zero(self):
        r = wt.rates(np.array([1, 0]), np.array([0, 0]), np.ones(2))
        assert r["precision"] is None and r["recall"] == 0.0

    def test_ece_of_a_calibrated_set_is_zero_and_of_an_overconfident_one_is_not(self):
        y = np.array([1, 1, 1, 0] * 5)
        assert wt.expected_calibration_error(y, np.full(20, 0.75), np.ones(20)) == pytest.approx(0)
        assert wt.expected_calibration_error(y, np.full(20, 0.95), np.ones(20)) == pytest.approx(
            0.2
        )

    def test_ece_weights_change_the_result_within_one_bin(self):
        y = np.array([1, 0])
        p = np.array([0.6, 0.6])
        assert wt.expected_calibration_error(y, p, np.ones(2)) == pytest.approx(0.1)
        assert wt.expected_calibration_error(y, p, np.array([1.0, 3.0])) == pytest.approx(0.35)

    def test_rate_intervals_are_seeded_and_enclose_the_point(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 80)
        flag = np.where(rng.random(80) < 0.7, y, 1 - y)
        a = wt.rate_intervals(y, flag, np.ones(80), n_bootstrap=200, seed=3)
        b = wt.rate_intervals(y, flag, np.ones(80), n_bootstrap=200, seed=3)
        assert a["precision"].low == b["precision"].low
        assert a["precision"].low <= a["precision"].value <= a["precision"].high


class TestStratumWeights:
    def frames(self):
        rows = []
        for code, pool, n in (("S1", 40, 4), ("S2", 10, 5), ("S3", 6, 3)):
            for i in range(n):
                rows.append({"id": f"{code}{i}", "stratum": code, "pool": pool})
        all_rows = pd.DataFrame(rows)
        return all_rows, all_rows

    def test_the_weight_is_pool_over_the_rows_used_not_over_the_rows_drawn(self):
        all_rows, _ = self.frames()
        used = all_rows[~all_rows["id"].isin(["S10", "S11"])]  # two S1 rows were dropped
        sw = wt.stratum_weights(all_rows, used)
        assert sw.per_stratum["S1"] == {"pool": 40, "n_used": 2}
        by_id = dict(zip(used["id"], sw.weights, strict=True))
        assert by_id["S12"] == pytest.approx(20.0)  # 40 / 2, not 40 / 4
        assert by_id["S20"] == pytest.approx(2.0)
        # The weights still add up to the whole pool of every stratum with a used row.
        assert sw.weights.sum() == pytest.approx(56.0)

    def test_a_stratum_with_no_used_row_is_named_and_warned_about(self):
        all_rows, _ = self.frames()
        used = all_rows[all_rows["stratum"] != "S3"]
        sw = wt.stratum_weights(all_rows, used)
        assert sw.empty == ["S3"]
        assert "stratum S3 (pool 6) has no used row" in wt.stratum_warning(sw)[0]

    def test_kish_n_eff_of_equal_weights_is_n_and_of_skewed_weights_is_less(self):
        assert wt.kish_n_eff(np.ones(10)) == pytest.approx(10.0)
        assert wt.kish_n_eff(np.array([1.0, 1.0, 8.0])) == pytest.approx(100 / 66)

    def test_the_bootstrap_resamples_inside_each_stratum(self):
        # Stratum A holds only y=1 rows and stratum B only y=0 rows. A plain bootstrap
        # would draw a different A and B count each time. Inside strata, the positive
        # rate is the same on every draw, so the interval has zero width.
        y = np.array([1] * 6 + [0] * 6)
        strata = np.array(["A"] * 6 + ["B"] * 6)
        w = np.ones(12)
        stat = {"pos": lambda idx: float(y[idx].mean())}
        within = wt.bootstrap_statistics(y, w, stat, n_bootstrap=100, seed=1, strata=strata)["pos"]
        plain = wt.bootstrap_statistics(y, w, stat, n_bootstrap=100, seed=1)["pos"]
        assert within.low == within.high == pytest.approx(0.5)
        assert plain.high > plain.low

    def test_false_positive_rate_is_the_flag_rate_among_rows_labelled_0(self):
        y = np.array([0, 0, 0, 0, 1])
        flag = np.array([1, 0, 0, 0, 1])
        assert wt.false_positive_rate(y, flag, np.ones(5)) == pytest.approx(0.25)
        weights = np.array([3.0, 1.0, 1.0, 1.0, 1.0])
        assert wt.false_positive_rate(y, flag, weights) == pytest.approx(0.5)
        assert wt.false_positive_rate(np.ones(3, dtype=int), np.ones(3), np.ones(3)) is None


# ---------------------------------------------------------------------------
# transfer on a resolved sheet
# ---------------------------------------------------------------------------
VERSION = "v0.1.0"


def scored_sheet(n_pos=30, n_neg=30, kind="human", lang="en"):
    """Sheet, scores and design. Positives are stratum S1 (pool 90), negatives S2 (pool 30)."""
    rows, scores, design = [], [], []
    for i in range(n_pos + n_neg):
        positive = i < n_pos
        rid = f"{'p' if positive else 'n'}{i}"
        prob = 0.9 - 0.4 * (i % 3 == 0) if positive else 0.2 + 0.5 * (i % 4 == 0)
        rows.append(
            {
                "id": rid,
                "source": "news",
                "lang": lang,
                "text": LONG,
                "adjudicated_label": str(int(positive)),
                "adjudicated_reports_on_false_claim": "1" if i % 5 == 0 else "0",
                "labeller_kind": kind,
            }
        )
        scores.append(
            {"record_id": rid, "misinfo_prob": prob, "model_versions": {"laya": f"{VERSION}+ab"}}
        )
        design.append(
            {
                "id": rid,
                "stratum": "S1" if positive else "S2",
                "pool": "90" if positive else "30",
                "drawn": str(n_pos if positive else n_neg),
                "weight": "3" if positive else "1",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(scores), pd.DataFrame(design)


def resolved(kind="human", allow_flag=False, with_design=True, **kw):
    sheet, scores, design = scored_sheet(kind=kind, **kw)
    return (
        lb.resolve_sheet(
            sheet, design=design if with_design else None, allow_model_labels=allow_flag
        ),
        scores,
    )


class TestEnglishTransfer:
    def run(self, kind="human", allow_flag=False, **kw):
        from modeling.eval import corpus_transfer as ct

        sheet, scores = resolved(kind, allow_flag, **kw)
        scores["model_versions"] = [{"misinfo": f"{VERSION}+ab"}] * len(scores)
        return ct.evaluate_transfer(
            sheet,
            scores,
            language_allowed=allow,
            model_version=VERSION,
            threshold=0.5,
            n_bootstrap=100,
        )

    def test_reweighted_and_reporting_sections_render_with_n_eff(self):
        out = self.run()
        assert out.computed and out.weighted is not None and out.reporting is not None
        # S1 weighs 90 / 30 = 3 per row and S2 weighs 1, so the corpus positive rate is 0.75
        assert out.weighted["positive_rate"].value == pytest.approx(0.75)
        text = "\n".join(out.render())
        assert "reweighted to the corpus" in text and "n_eff" in text
        assert "false-positive rate" in text and "unweighted, sample rows" in text
        assert "unreviewed" not in text

    def test_false_positive_rates_by_reporting_are_weighted_and_unweighted(self):
        out = self.run()
        r = out.reporting
        assert r["n_negatives_reporting"] + r["n_negatives_other"] == 30
        assert r["unweighted"]["reporting"].value is not None
        assert r["weighted"]["other"].value is not None

    def test_a_dropped_row_renormalises_the_weights_and_is_counted_per_stratum(self):
        from modeling.eval import corpus_transfer as ct

        sheet, scores = resolved()
        scores["model_versions"] = [{"misinfo": f"{VERSION}+ab"}] * len(scores)
        scores = scores[scores["record_id"] != "p0"]  # p0 has no score
        out = ct.evaluate_transfer(
            sheet,
            scores,
            language_allowed=allow,
            model_version=VERSION,
            threshold=0.5,
            n_bootstrap=50,
        )
        assert out.drops_by_stratum == {"S1": {ct.NO_SCORE: 1}}
        assert out.strata_info.per_stratum["S1"] == {"pool": 90, "n_used": 29}
        assert out.strata_info.weights.max() == pytest.approx(90 / 29)
        assert "S1: no_score=1" in "\n".join(out.render())

    def test_model_labels_stamp_the_first_and_last_line(self):
        lines = self.run("model-unreviewed", True).render()
        note = "labeller: model, unreviewed by a person"
        assert lines[1].strip() == note and lines[-1].strip() == note

    def test_the_evaluator_refuses_model_labels_itself(self):
        from modeling.eval import corpus_transfer as ct

        sheet, scores, design = scored_sheet(kind="llm-unreviewed")
        out = ct.evaluate_transfer(
            lb.resolve_sheet(sheet, design=design),
            scores,
            language_allowed=allow,
            model_version=VERSION,
            threshold=0.5,
        )
        assert not out.computed and "no person reviewed" in out.refusal

    def test_a_plain_frame_is_rejected(self):
        from modeling.eval import corpus_transfer as ct

        sheet, scores, _ = scored_sheet()
        with pytest.raises(TypeError, match="ResolvedSheet"):
            ct.evaluate_transfer(
                sheet, scores, language_allowed=allow, model_version=VERSION, threshold=0.5
            )
        with pytest.raises(TypeError, match="ResolvedSheet"):
            lt.evaluate_laya_transfer(sheet, scores, model_version=VERSION, threshold=0.5)

    def test_the_default_seed_is_the_labelling_seed(self):
        import inspect

        from modeling.eval import corpus_transfer as ct

        for fn in (ct.evaluate_transfer, lt.evaluate_laya_transfer):
            assert inspect.signature(fn).parameters["seed"].default == 20260813


class TestLayaHindi:
    def run(self, sheet, scores, **kw):
        kw.setdefault("threshold", 0.5)
        return lt.evaluate_laya_transfer(
            sheet, scores, model_version=VERSION, n_bootstrap=100, **kw
        )

    def test_metrics_ece_and_weighted_ece_have_intervals_and_n_eff(self):
        sheet, scores = resolved(lang="hi")
        out = self.run(sheet, scores)
        assert out.computed
        assert out.sample["ece"].low <= out.sample["ece"].high
        assert out.corpus["ece"].value != out.sample["ece"].value
        text = "\n".join(out.render())
        assert "ece" in text and "n_eff" in text

    def test_ece_states_its_bins_and_n_and_warns_below_200_rows(self):
        sheet, scores = resolved(lang="hi")
        text = "\n".join(self.run(sheet, scores).render())
        assert "10 equal-width bins, n 60" in text
        assert "caveat: n is below 200" in text
        assert "ece definition: sum over bins" in text

    def test_no_threshold_means_precision_and_recall_are_not_measured(self):
        sheet, scores = resolved(lang="hi")
        out = self.run(sheet, scores, threshold=None)
        assert out.computed and "precision" not in out.sample and "ece" in out.sample
        assert "precision not measured, no decision threshold" in "\n".join(out.render())

    def test_only_laya_stamped_current_values_count(self):
        sheet, scores = resolved(lang="hi")
        scores["model_versions"] = scores["model_versions"].astype(object)
        scores.at[0, "model_versions"] = {"misinfo": VERSION}
        scores.at[1, "model_versions"] = {"laya": "v0.0.1+zz"}
        scores.loc[2, "misinfo_prob"] = None
        out = self.run(sheet, scores)
        assert out.excluded == {
            lt.NOT_A_LAYA_SCORE: 1,
            lt.STALE_MODEL_VERSION: 1,
            lt.NO_SCORE: 1,
        }
        assert out.drops_by_stratum == {
            "S1": {lt.NOT_A_LAYA_SCORE: 1, lt.STALE_MODEL_VERSION: 1, lt.NO_SCORE: 1}
        }

    def test_refuses_below_the_minimum(self):
        sheet, scores = resolved(n_pos=5, n_neg=5, lang="hi")
        out = self.run(sheet, scores)
        assert not out.computed and "MIN_LABELLED_ROWS" in out.refusal

    def test_the_evaluator_refuses_model_labels_and_stamps_them_when_allowed(self):
        sheet, scores = resolved("model-unreviewed", False, lang="hi")
        assert "no person reviewed" in self.run(sheet, scores).refusal
        sheet, scores = resolved("model-unreviewed", True, lang="hi")
        out = self.run(sheet, scores)
        assert out.computed and out.render()[1].strip() == sheet.labeller_note


class TestJsonable:
    def test_a_nan_interval_becomes_null_and_the_payload_is_strict_json(self):
        from modeling.eval.metrics import Interval
        from modeling.eval.relabel import _jsonable

        value = {"x": Interval(0.5, float("nan"), float("nan"), 0), "y": [float("nan")]}
        text = json.dumps(_jsonable(value), allow_nan=False)
        assert json.loads(text) == {
            "x": {"value": 0.5, "ci_low": None, "ci_high": None, "n_bootstrap": 0},
            "y": [None],
        }

    def test_a_transfer_result_serialises_without_arrays(self):
        from modeling.eval import corpus_transfer as ct
        from modeling.eval.relabel import _jsonable

        sheet, scores = resolved()
        scores["model_versions"] = [{"misinfo": f"{VERSION}+ab"}] * len(scores)
        out = ct.evaluate_transfer(
            sheet,
            scores,
            language_allowed=allow,
            model_version=VERSION,
            threshold=0.5,
            n_bootstrap=30,
        )
        payload = json.dumps(_jsonable(out), allow_nan=False)
        assert json.loads(payload)["strata_info"]["per_stratum"]["S1"]["pool"] == 90


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
class Settings:
    seed = 0
    data_dir = "unused"

    def __init__(self, root):
        self.artifacts_dir = root / "artifacts"

    @staticmethod
    def language_allowed(module, lang):
        return lang in ("en", "hi")


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(tmp_path)
    monkeypatch.setattr(modeling_cli, "get_settings", lambda: settings)
    monkeypatch.setattr("modeling.config.get_settings", lambda: settings)
    monkeypatch.setattr(modeling_cli, "_bootstrap", lambda verbose=False: None)
    monkeypatch.setattr("modeling.io.CorpusReader.__init__", lambda self, settings=None: None)
    monkeypatch.setattr("modeling.io.CorpusReader.records", lambda self, **kw: corpus())
    monkeypatch.setattr(
        "ingest.factcheck.read_deduplicated_factchecks",
        lambda s: type(
            "R", (), {"kept": [{"label": "false", "claim_simhash": CLAIM, "claim": "c"}]}
        )(),
    )
    monkeypatch.setattr("ingest.config.Settings", lambda **kw: None)
    return settings


class TestSampleCommand:
    def test_draws_a_shuffled_en_sheet_a_design_file_and_a_sampling_record(self, env):
        result = runner.invoke(
            modeling_cli.app, ["sample-for-labelling", "misinfo", "--n", "20", "--lang", "en"]
        )
        assert result.exit_code == 0, result.output
        path = env.artifacts_dir / "hand_labels" / "misinfo_corpus_sample_en.csv"
        sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert list(sheet.columns) == list(lb.SHEET_COLUMNS)
        assert not {"stratum", "weight", "pool", "drawn"} & set(sheet.columns)
        assert len(sheet) == 20 and (sheet["sample_seed"] == "20260813").all()
        design = pd.read_csv(lb.design_path(path), dtype=str)
        assert set(design["id"]) == set(sheet["id"]) and "stratum" in design.columns
        record = json.loads(path.with_suffix(".sampling.json").read_text())
        assert record["seed"] == 20260813 and record["strata"]["S1"]["name"] == "claim_match"
        assert record["claim_matches_in_scorable_pool"]["any"] >= 1
        assert record["claim_match"]["simhash_hamming"] == 3
        assert "embedding_unavailable" in result.output
        assert "fact_check_publisher=5" in result.output
        assert "claim matches in the scorable pool: simhash=1" in result.output

    def test_hi_uses_the_laya_language_list(self, env):
        result = runner.invoke(
            modeling_cli.app, ["sample-for-labelling", "misinfo", "--n", "6", "--lang", "hi"]
        )
        assert result.exit_code == 0, result.output
        sheet = pd.read_csv(
            env.artifacts_dir / "hand_labels" / "misinfo_corpus_sample_hi.csv", dtype=str
        )
        assert set(sheet["lang"]) == {"hi"} and sheet["title"].notna().all()

    def test_lang_is_required(self, env):
        result = runner.invoke(modeling_cli.app, ["sample-for-labelling", "misinfo"])
        assert result.exit_code != 0
        assert not list((env.artifacts_dir / "hand_labels").glob("*_all*"))

    def test_a_too_small_n_is_refused_with_the_reason(self, env):
        result = runner.invoke(
            modeling_cli.app, ["sample-for-labelling", "misinfo", "--n", "2", "--lang", "en"]
        )
        assert (
            result.exit_code == 1 and "Every such stratum needs at least one row" in result.output
        )

    def test_a_second_run_does_not_overwrite_labels(self, env):
        args = ["sample-for-labelling", "misinfo", "--n", "6", "--lang", "en"]
        assert runner.invoke(modeling_cli.app, args).exit_code == 0
        again = runner.invoke(modeling_cli.app, args)
        assert again.exit_code == 1 and "--force" in again.output
        assert runner.invoke(modeling_cli.app, [*args, "--force"]).exit_code == 0

    def test_bad_lang_is_rejected(self, env):
        result = runner.invoke(
            modeling_cli.app, ["sample-for-labelling", "misinfo", "--lang", "fr"]
        )
        assert result.exit_code != 0


class TestEmbeddingsFile:
    def write(self, tmp_path, model):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"id": ["a"], "embedding": [[1.0, 0.0]]})
        if model:
            table = table.replace_schema_metadata({b"model": model.encode()})
        path = tmp_path / "emb.parquet"
        pq.write_table(table, path)
        return path

    def evidence(self, path, expected):
        calls = []

        def factory():
            calls.append(1)
            return None, "model_unavailable"

        out = lb.claim_evidence(
            [{"label": "false", "claim_simhash": 5, "claim": "c"}],
            config(claim_match={"min_similarity": 0.85, "simhash_hamming": 3}),
            embeddings_path=path,
            embedder_factory=factory,
            expected_model=expected,
        )
        return out, calls

    def test_a_file_that_names_another_model_is_refused(self, tmp_path):
        out, calls = self.evidence(self.write(tmp_path, "model-x"), "model-y")
        assert out.embedding_reason == lb.EMBEDDING_MODEL_MISMATCH and not calls
        assert any("model-x" in n for n in out.notes)

    def test_a_file_with_no_model_id_is_flagged_as_unverified(self, tmp_path):
        out, calls = self.evidence(self.write(tmp_path, None), "model-y")
        assert calls and any(lb.EMBEDDING_MODEL_UNVERIFIED in n for n in out.notes)
        assert out.embedding_reason == "model_unavailable"

    def test_a_matching_model_id_passes_to_the_embedder(self, tmp_path):
        out, calls = self.evidence(self.write(tmp_path, "model-y"), "model-y")
        assert calls and not out.notes

    def test_a_missing_file_is_a_counted_reason(self, tmp_path):
        out, _ = self.evidence(tmp_path / "none.parquet", "m")
        assert out.embedding_reason == lb.EMBEDDING_UNAVAILABLE


class TestRelabelEval:
    @pytest.fixture
    def filled(self, env, monkeypatch):
        monkeypatch.setattr(
            "modeling.config.module_config",
            lambda module: {"version": VERSION, "decision_threshold": 0.5},
        )
        monkeypatch.setattr("modeling.eval.corpus_transfer.MIN_LABELLED_ROWS", 20)
        monkeypatch.setattr("modeling.eval.laya_transfer.MIN_LABELLED_ROWS", 20)
        scores = {}
        for lang in ("en", "hi"):
            sheet, score, design = scored_sheet(lang=lang)
            sheet["label_a"] = sheet["adjudicated_label"]
            sheet["label_b"] = sheet["adjudicated_label"]
            sheet.loc[0, "label_b"] = "0"
            sheet["reports_on_false_claim_a"] = sheet["adjudicated_reports_on_false_claim"]
            sheet["reports_on_false_claim_b"] = sheet["adjudicated_reports_on_false_claim"]
            for frame, col in ((sheet, "id"), (score, "record_id"), (design, "id")):
                frame[col] = [f"{lang}-{i}" for i in frame[col]]
            path = env.artifacts_dir / "hand_labels" / f"misinfo_corpus_sample_{lang}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            sheet.to_csv(path, index=False)
            design.to_csv(lb.design_path(path), index=False)
            scores[lang] = score
        both = pd.concat(scores.values())
        both.loc[both["record_id"].str.startswith("en"), "model_versions"] = [
            {"misinfo": f"{VERSION}+ab"}
        ] * int(both["record_id"].str.startswith("en").sum())
        monkeypatch.setattr("modeling.io.ScoredStore.__init__", lambda self, settings=None: None)
        monkeypatch.setattr("modeling.io.ScoredStore.read", lambda self, name: both)
        return env

    def test_one_command_rewrites_every_output(self, filled):
        result = runner.invoke(modeling_cli.app, ["relabel-eval", "--n-bootstrap", "50"])
        assert result.exit_code == 0, result.output
        out = filled.artifacts_dir / "eval" / "labelling"
        for lang in ("en", "hi"):
            for name in ("agreement.md", "disagreements.csv", "transfer.md", "transfer.json"):
                assert (out / lang / name).exists(), (lang, name)
        data = json.loads((out / "hi" / "transfer.json").read_text())
        assert data["computed"] and "labeller" not in data
        # Under the stubbed settings the seed is 0, so 20260813 comes from the default.
        assert data["seed"] == 20260813
        assert json.loads((out / "en" / "transfer.json").read_text())["seed"] == 20260813
        assert "bootstrap seed 20260813" in (out / "en" / "agreement.md").read_text()
        assert "Cohen kappa" in (out / "en" / "agreement.md").read_text()
        assert "en-p0" in (out / "en" / "disagreements.csv").read_text()

    def test_model_labels_are_refused_then_stamped_on_every_file(self, filled):
        for lang in ("en", "hi"):
            path = filled.artifacts_dir / "hand_labels" / f"misinfo_corpus_sample_{lang}.csv"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame["labeller_kind"] = "model-unreviewed"
            frame.to_csv(path, index=False)
        refused = runner.invoke(modeling_cli.app, ["relabel-eval", "--n-bootstrap", "50"])
        assert refused.exit_code == 1 and "REFUSED" in refused.output
        out = filled.artifacts_dir / "eval" / "labelling"
        assert "REFUSED" in (out / "hi" / "transfer.md").read_text()
        allowed = runner.invoke(
            modeling_cli.app, ["relabel-eval", "--allow-model-labels", "--n-bootstrap", "50"]
        )
        assert allowed.exit_code == 0, allowed.output
        note = "labeller: model, unreviewed by a person"
        for lang in ("en", "hi"):
            for name in ("agreement.md", "transfer.md"):
                assert note in (out / lang / name).read_text(), (lang, name)
            assert json.loads((out / lang / "transfer.json").read_text())["labeller"] == (
                "model, unreviewed by a person"
            )

    def test_evaluate_misinfo_refuses_model_labels_without_the_flag(self, filled, monkeypatch):
        monkeypatch.setattr(
            "modeling.training.evaluate_module",
            lambda module, **kw: __import__("modeling.training", fromlist=["x"]).TrainingResult(
                module, VERSION, None, skipped=True, notes=[]
            ),
        )
        path = filled.artifacts_dir / "hand_labels" / "misinfo_corpus_sample_en.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame["labeller_kind"] = "llm-unreviewed"
        frame.to_csv(path, index=False)
        refused = runner.invoke(modeling_cli.app, ["evaluate", "misinfo"])
        assert refused.exit_code == 1 and "no person reviewed" in refused.output
        allowed = runner.invoke(modeling_cli.app, ["evaluate", "misinfo", "--allow-model-labels"])
        assert allowed.exit_code == 0, allowed.output
        assert "labeller: model, unreviewed by a person" in allowed.output

    def test_missing_sheets_are_reported_not_skipped_silently(self, env, tmp_path):
        result = runner.invoke(modeling_cli.app, ["relabel-eval", "--n-bootstrap", "50"])
        assert result.exit_code == 1 and "NOT MEASURED" in result.output

    def test_label_agreement_command_prints_the_report(self, filled):
        path = filled.artifacts_dir / "hand_labels" / "misinfo_corpus_sample_en.csv"
        result = runner.invoke(modeling_cli.app, ["label-agreement", str(path)])
        assert result.exit_code == 0 and "raw agreement" in result.output
        assert "disagreements: 1" in result.output

    def test_laya_command_reports_calibration(self, filled):
        result = runner.invoke(
            modeling_cli.app, ["evaluate-laya-hindi", "--threshold", "0.5", "--n-bootstrap", "50"]
        )
        assert result.exit_code == 0, result.output
        assert "ece" in result.output and "not the configured threshold" in result.output
