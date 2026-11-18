"""Tier 4 reader and extractor: verdicts read from headlines.

All headlines are written for the tests. Nothing here reads the network or a corpus.
"""

from __future__ import annotations

import pytest
import yaml

from ingest.config import CONFIG_DIR, Settings
from ingest.factcheck import DropReason, FactCheckExtractor, Tier, normalize_rating
from ingest.title_predicate import (
    TitlePredicateConfigError,
    TitlePredicateReader,
    headline_lang,
    pattern_set_hash,
    prepare_headline,
)


def real_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "factcheck.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reader() -> TitlePredicateReader:
    return TitlePredicateReader(real_config())


@pytest.fixture
def extractor(tmp_path):
    return FactCheckExtractor(settings=Settings(data_dir=tmp_path), config=real_config())


def record(n: int, title: str, domain: str = "altnews.in") -> dict:
    return {
        "id": f"news:{domain}/{n}",
        "url": f"https://{domain}/{n}",
        "source_detail": domain,
        "title": title,
        "text": "",
        "tags": [],
    }


# --- the config itself -------------------------------------------------------


def test_every_pattern_in_the_config_compiles_and_names_a_rating_in_the_map(reader):
    config = real_config()
    ratings = {normalize_rating(k) for k in config["ratings"]}
    assert reader.patterns["en"] and reader.patterns["hi"]
    assert reader.guards["en"] and reader.guards["hi"]
    for patterns in reader.patterns.values():
        for pattern in patterns:
            assert normalize_rating(pattern.rating) in ratings, pattern.id


def test_tier_4_is_off_in_the_primary_evaluation_by_default():
    block = real_config()["title_predicate"]
    assert block["include_in_primary_evaluation"] is False
    assert block["measured_error_rate"]["status"] == "not_measured"
    assert block["min_kappa"] == 0.6
    assert block["min_adjudicated_rows"] == 100
    assert block["min_language_rows"] == 30


def test_both_publishers_use_the_title_predicate_tier():
    publishers = real_config()["publishers"]
    assert publishers["altnews.in"]["tier"] == Tier.TITLE_PREDICATE.value
    assert publishers["factly.in"]["tier"] == Tier.TITLE_PREDICATE.value


def test_the_fact_checker_skip_list_is_the_same_eight_domains():
    from modeling.config import fact_check_domains

    assert fact_check_domains() == frozenset(
        {
            "boomlive.in",
            "thequint.com",
            "vishvasnews.com",
            "factcrescendo.com",
            "newschecker.in",
            "altnews.in",
            "factly.in",
            "digiteye.in",
        }
    )
    assert "altnews.in" not in (real_config().get("other_fact_checkers") or [])
    assert "factly.in" not in (real_config().get("other_fact_checkers") or [])


def pattern(pid="a", rating="False", text="x"):
    return {"id": pid, "rating": rating, "pattern": text}


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"title_predicate": {}},
        {"title_predicate": {"patterns": {"fr": []}}},
        {"title_predicate": {"patterns": {"en": [pattern(text="(")]}}},
        {"title_predicate": {"patterns": {"en": [{"id": "a", "rating": "False"}]}}},
        {"title_predicate": {"patterns": {"en": [pattern(rating=False)]}}},
        {"title_predicate": {"patterns": {"en": [pattern("a"), pattern("a", text="y")]}}},
        {"title_predicate": {"patterns": {"en": [pattern()]}, "guards": {"fr": []}}},
        {"title_predicate": {"patterns": {"en": [pattern()]}, "guards": {"en": [{"id": "g"}]}}},
        {
            "title_predicate": {
                "patterns": {"en": [pattern()]},
                "guards": {"en": [{"id": "g", "pattern": "("}]},
            }
        },
    ],
)
def test_a_broken_config_raises(config):
    with pytest.raises(TitlePredicateConfigError):
        TitlePredicateReader(config)


def test_the_pattern_hash_follows_the_patterns_and_the_guards():
    config = real_config()
    base = pattern_set_hash(config)
    assert base == TitlePredicateReader(config).pattern_hash
    assert pattern_set_hash(real_config()) == base
    config["title_predicate"]["patterns"]["en"][0]["pattern"] += "x"
    assert pattern_set_hash(config) != base
    config = real_config()
    config["title_predicate"]["guards"]["en"][0]["pattern"] += "x"
    assert pattern_set_hash(config) != base
    config = real_config()
    config["title_predicate"]["include_in_primary_evaluation"] = True
    assert pattern_set_hash(config) == base


# --- the reader --------------------------------------------------------------

#: (headline, expected pattern id). Written for these tests in the shapes the two
#: publishers use. They are not copied from a corpus.
MATCHING = [
    ("Video Of Flooded Street Is AI-Generated, Not Real", "en_ai_generated"),
    ("Image Of Minister Is Digitally Altered", "en_morphed_or_edited"),
    ("Clip Of Rally Is Scripted, Shows Actors", "en_scripted"),
    (
        "Old Video From Pakistan Falsely Shared As Recent Attack In India",
        "en_old_media_shared",
    ),
    ("Photo Of Temple Falsely Shared As Newly Built", "en_falsely_shared"),
    ("Claim That Tea Cures Fever Is Misleading", "en_misleading"),
    ("Fact Check: No, The Prime Minister Did Not Say This", "en_no_prefix"),
    ("There Is No Evidence For The Viral Claim About Banned Drink", "en_no_evidence"),
    ("Viral Letter About Holiday Is Fake", "en_fake"),
    ("Old Photo Shared As Recent Rally", "en_old_media_shared"),
    ("Morphed Image Of Rahul Gandhi Goes Viral", "en_morphed_or_edited"),
    ("Doctored Video Of Minister Circulates", "en_morphed_or_edited"),
    ("यह वीडियो एआई जनरेटेड है", "hi_ai_generated"),
    ("वीडियो AI-generated है", "hi_ai_generated"),
    ("वायरल तस्वीर एडिटेड है", "hi_morphed"),
    ("सोशल मीडिया पर वायरल पोस्ट फर्जी है", "hi_fake"),
    ("मंदिर की तस्वीर के साथ किया जा रहा दावा भ्रामक है", "hi_misleading"),
    # The old-media pattern is checked before the fake pattern.
    ("पुराना वीडियो फर्जी दावे के साथ वायरल", "hi_old_media_viral"),
]

#: Headlines that must give no verdict. The first group is every headline of the
#: adversarial review: a negated verdict, genuine media, a quoted verdict word, a topic
#: headline and a word that is no verdict.
NOT_MATCHING = [
    "Video Is Not AI-Generated, Experts Confirm It Is Real",
    "Fact Check: No, This Video Is Not Fake",
    "Video Of Temple Is Real, Not Fake",
    "Man Calls Real Video 'Fake' But It Is Genuine",
    "Video Of PM Is Genuine, Claims Of It Being Fake Are False",
    "How To Spot Fake News: A Guide",
    "Fake News Law Passed In State Assembly",
    "Government Issues Advisory On Deepfake Menace",
    "AI-Generated Images Flood Social Media After Cyclone",
    "Court Says There Is No Proof Against Activist",
    "वीडियो फर्जी नहीं है, असली है",
    "दावा झूठा नहीं, सही है",
    "EVM में छेड़छाड़ का वीडियो वायरल, जानें सच",
    "Parliament Passes The Budget After A Week Of Debate",
    "How We Verify Images: A Guide For Readers",
    "आज का मौसम: कई राज्यों में बारिश की चेतावनी",
    "Is This Video Fake?",
    "Fact Check: Is This Image AI-Generated?",
    "क्या यह वीडियो फर्जी है?",
]


@pytest.mark.parametrize(("headline", "pattern_id"), MATCHING)
def test_a_verdict_headline_matches_its_pattern(reader, headline, pattern_id):
    match = reader.read(headline)
    assert match is not None and match.pattern_id == pattern_id


@pytest.mark.parametrize("headline", NOT_MATCHING)
def test_a_headline_with_no_verdict_matches_nothing(reader, headline):
    assert reader.read(headline) is None


def test_a_guard_cancels_a_match_and_is_named(reader):
    match, guard = reader.classify("Fact Check: No, This Video Is Not Fake")
    assert match is None and guard == "en_guard_negated"
    match, guard = reader.classify("Video Of PM Is Genuine, Claims Of It Being Fake Are False")
    assert match is None and guard == "en_guard_genuine"
    match, guard = reader.classify("दावा झूठा नहीं, सही है")
    assert match is None and guard == "hi_guard_negated"
    assert reader.classify("Weekly Roundup") == (None, None)


def test_old_media_wins_over_the_fake_pattern_in_both_languages(reader):
    assert reader.read("Old Video Of Crowd Falsely Shared, The Claim Is Fake").rating == (
        "Missing Context"
    )
    assert reader.read("पुराना वीडियो फर्जी दावे के साथ वायरल").rating == "Missing Context"


@pytest.mark.parametrize(
    "headline",
    [
        "Video Is AI‑Generated",  # non-breaking hyphen
        "Video Is AI–Generated",  # en dash
        "Video Is AI Generated",
        "Video Is AI-generated",
        "Video Is AI­-Generated",  # soft hyphen
        "Video Is AI‍-Generated",  # zero-width joiner
    ],
)
def test_ai_generated_allows_the_separators_that_sites_use(reader, headline):
    assert reader.read(prepare_headline(headline)).pattern_id == "en_ai_generated"


@pytest.mark.parametrize(
    "headline",
    [
        "Fact-Check: No, The PM Did Not Say This",
        "Fact Check – No, The PM Did Not Say This",
        "fact check—No, The PM Did Not Say This",
        "FactCheck: No, The PM Did Not Say This",
    ],
)
def test_the_fact_check_prefix_allows_any_dash(reader, headline):
    assert reader.read(headline).pattern_id == "en_no_prefix"


def test_a_headline_in_two_scripts_is_tried_against_both_lists(reader):
    assert reader.read("Video of EVM फर्जी है").pattern_id == "hi_fake"
    assert reader.read("यह Video Is AI-Generated").pattern_id == "en_ai_generated"
    assert reader.read("वायरल दावा: Photo Is Morphed").pattern_id == "en_morphed_or_edited"


def test_the_language_comes_from_the_script():
    assert headline_lang("यह फर्जी है") == "hi"
    assert headline_lang("This is fake") == "en"


def test_the_first_matching_pattern_wins():
    config = {
        "title_predicate": {
            "patterns": {
                "en": [
                    {"id": "first", "rating": "Misleading", "pattern": "fake"},
                    {"id": "second", "rating": "False", "pattern": "fake"},
                ]
            }
        }
    }
    assert TitlePredicateReader(config).read("a fake claim").pattern_id == "first"


# --- the extractor -----------------------------------------------------------


def test_a_matching_headline_becomes_a_tier_4_row(extractor):
    row = extractor.from_title_predicate(record(1, "Video Of Flooded Street Is AI-Generated"))
    assert row is not None
    assert row.tier == "title_predicate"
    assert row.label == "false"
    assert row.rating_raw == "AI Generated"
    assert row.claim == "Video Of Flooded Street Is AI-Generated"
    assert row.publisher == "Alt News"


def test_a_hindi_headline_becomes_a_tier_4_row(extractor):
    row = extractor.from_title_predicate(record(2, "वायरल पोस्ट फर्जी है"))
    assert row is not None and row.tier == "title_predicate" and row.claim_lang == "hi"


def test_invisible_characters_do_not_hide_a_verdict(extractor):
    row = extractor.from_title_predicate(record(3, "Video Is Fa­ke"))
    assert row is not None and row.claim == "Video Is Fake"
    row = extractor.from_title_predicate(record(4, "वायरल पोस्ट फर्‌जी‍ है"))
    assert row is not None and row.rating_raw == "Fake"


def test_a_headline_with_no_match_is_counted_and_logged(extractor, caplog):
    with caplog.at_level("DEBUG", logger="ingest.factcheck"):
        assert extractor.from_title_predicate(record(5, "Budget Session Begins Monday")) is None
    assert extractor.dropped == {DropReason.TITLE_PREDICATE_NO_MATCH.value: 1}
    assert "title_predicate_no_match" in caplog.text


def test_a_guarded_headline_is_counted_apart_from_no_match(extractor, caplog):
    with caplog.at_level("DEBUG", logger="ingest.factcheck"):
        guarded = record(6, "Fact Check: No, This Video Is Not Fake")
        assert extractor.from_title_predicate(guarded) is None
    assert extractor.dropped == {DropReason.TITLE_PREDICATE_GUARDED.value: 1}
    assert "en_guard_negated" in caplog.text


def test_a_middle_rating_is_dropped_through_the_rating_map(extractor):
    headline = "Old Video Of Crowd Falsely Shared As Recent Rally"
    assert extractor.from_title_predicate(record(7, headline)) is None
    assert extractor.dropped == {DropReason.RATING_IS_MIDDLE.value: 1}


def test_an_empty_headline_is_counted(extractor):
    assert extractor.from_title_predicate(record(8, "  ")) is None
    assert extractor.dropped == {DropReason.NO_CLAIM_TEXT.value: 1}


def test_a_config_without_patterns_is_counted_as_bad_config(tmp_path):
    config = real_config()
    del config["title_predicate"]
    ex = FactCheckExtractor(settings=Settings(data_dir=tmp_path), config=config)
    assert ex.from_title_predicate(record(9, "Is Fake")) is None
    assert ex.dropped == {DropReason.TITLE_PREDICATE_BAD_CONFIG.value: 1}


def test_run_reads_the_tier_without_fetching_a_page(extractor):
    rows = list(
        extractor.run(
            [
                record(1, "Photo Is Morphed", "factly.in"),
                record(2, "Weekly Roundup", "altnews.in"),
            ]
        )
    )
    assert [r.tier for r in rows] == ["title_predicate"]
    assert extractor.summary()["fetch_budget_left"] == 200
    assert extractor.dropped == {DropReason.TITLE_PREDICATE_NO_MATCH.value: 1}
