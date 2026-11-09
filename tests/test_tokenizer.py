"""Word tokenization of Indic scripts, and the English output that must not move."""

from __future__ import annotations

import pytest

from ingest.factcheck import normalize_rating
from ingest.normalize import _WORD_RE, extract_hashtags, simhash

# Expected values computed on the code before the tokenizer change, when
# `_WORD_RE` was `\w+`. English has no combining marks, so none of them may move.
ENGLISH_PINNED = [
    (
        "The Prime Minister said that vaccines were banned in 2021, officials deny it.",
        ["the", "prime", "minister", "said", "that", "vaccines", "were", "banned", "in",
         "2021", "officials", "deny", "it"],
        17674515204117543356,
    ),
    (
        "Fake: Delhi's water is #unsafe! Don't share",
        ["fake", "delhi", "s", "water", "is", "unsafe", "don", "t", "share"],
        1048192035169834174,
    ),
    ("Short claim", ["short", "claim"], 11824590805164772723),
]


@pytest.mark.parametrize(("text", "tokens", "expected_hash"), ENGLISH_PINNED)
def test_english_tokens_and_simhash_are_unchanged(text, tokens, expected_hash):
    assert _WORD_RE.findall(text.lower()) == tokens
    assert simhash(text) == expected_hash


def test_hindi_words_with_vowel_signs_stay_whole():
    text = "भारत में टीके पर प्रतिबंध है।"
    assert _WORD_RE.findall(text) == ["भारत", "में", "टीके", "पर", "प्रतिबंध", "है"]


def test_hindi_danda_is_not_part_of_a_word():
    assert _WORD_RE.findall("यह सच है।वह झूठ है॥") == ["यह", "सच", "है", "वह", "झूठ", "है"]


def test_telugu_words_stay_whole():
    text = "భారతదేశంలో టీకాలు నిషేధించబడ్డాయి"
    assert _WORD_RE.findall(text) == ["భారతదేశంలో", "టీకాలు", "నిషేధించబడ్డాయి"]


def test_hindi_simhash_ignores_the_sentence_mark():
    assert simhash("भारत में टीके पर प्रतिबंध है।") == simhash("भारत में टीके पर प्रतिबंध है")


def test_hindi_simhash_is_nonzero_for_marked_words():
    assert simhash("टीके") != 0


def test_rating_key_keeps_hindi_words_whole_and_drops_punctuation():
    assert normalize_rating("झूठ!!") == "झूठ"
    assert normalize_rating("Half True!! (Mostly)") == "half true mostly"


def test_hindi_hashtag_is_extracted_whole():
    assert extract_hashtags("आज #भारत में बहस") == ["भारत"]


def test_ascii_hashtag_with_digits_is_extracted_unchanged():
    # extract_hashtags lowercases its output, so the expected value is the lowercase tag.
    assert extract_hashtags("Watch this #Delhi2024 today") == ["delhi2024"]


def test_danda_after_hindi_hashtag_is_not_part_of_the_tag():
    assert extract_hashtags("यह सच है #भारत। वह झूठ है") == ["भारत"]
