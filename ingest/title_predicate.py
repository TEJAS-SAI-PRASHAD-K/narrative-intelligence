"""Tier 4: read a verdict from the predicate of a fact-check headline.

Alt News and Factly publish no ClaimReview and no verdict block. Their verdict is
in the headline ("... Is AI-Generated"). This module matches a headline against
the regular expressions in `configs/factcheck.yaml` under `title_predicate`. It
holds no pattern of its own, so a change of wording on a site is a config edit.

The result is inference, not extraction. The extractor writes it under the tier
`title_predicate`, and the evaluation keeps it out of the primary set until a
measured error rate allows it.

Two kinds of regular expression are configured for each language:

- `patterns` find a verdict. They are tried in the order written and the first
  match wins, so the most specific pattern comes first.
- `guards` block a verdict. A guard that matches anywhere in the headline cancels
  the match. Guards cover a negated verdict ("Is Not Fake"), a headline that says
  the media is genuine, and a verdict word inside quotation marks. The reader fails
  closed: a guarded headline gets no verdict.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from ingest.config import load_yaml
from ingest.normalize import clean_text

#: Verdict name for a headline that matches no pattern.
NO_MATCH = "no_match"

_DEVANAGARI = re.compile("[ऀ-ॿ]")
_LATIN = re.compile("[A-Za-z]")
_LANGS = ("en", "hi")


class TitlePredicateConfigError(ValueError):
    """The `title_predicate` block of the config is missing or unusable."""


@dataclass(frozen=True)
class PredicatePattern:
    """One compiled headline pattern and the rating string it stands for."""

    id: str
    lang: str
    rating: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class Guard:
    """A compiled guard. It cancels a verdict when it matches the headline."""

    id: str
    lang: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class PredicateMatch:
    """The first pattern that matched a headline."""

    pattern_id: str
    lang: str
    rating: str
    matched_text: str


def prepare_headline(title: str | None) -> str:
    """The headline as the pipeline reads it: NFKC, invisible characters removed.

    The extractor, the sampler and the reader all call this, so a label sheet shows
    the text that the reader saw. A zero-width joiner or a soft hyphen inside a word
    would otherwise split a pattern match in one place and not in the other.
    """
    return " ".join(clean_text(title).split())


def headline_lang(title: str) -> str:
    """``hi`` when the headline holds any Devanagari letter, else ``en``."""
    return "hi" if _DEVANAGARI.search(title or "") else "en"


def _languages_to_try(text: str) -> tuple[str, ...]:
    """Both lists for a headline in two scripts, else the list of its script."""
    has_hindi = bool(_DEVANAGARI.search(text))
    if has_hindi and _LATIN.search(text):
        return ("hi", "en")
    return ("hi",) if has_hindi else ("en",)


def normalize_headline(title: str) -> str:
    """NFC form with single spaces, so a pattern does not depend on invisible variants."""
    return " ".join(unicodedata.normalize("NFC", title or "").split())


def pattern_set_hash(config: dict[str, Any] | None) -> str:
    """SHA-256 of the configured patterns and guards, for a stale-measurement check.

    A measured error rate describes one pattern set. Any edit to a pattern, a guard
    or their order changes this value, so a figure from an older set is detectable.
    """
    block = (config or {}).get("title_predicate") or {}
    payload = {"patterns": block.get("patterns") or {}, "guards": block.get("guards") or {}}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TitlePredicateReader:
    """Matches headlines against the configured patterns, first match wins."""

    def __init__(self, config: dict[str, Any] | None):
        block = (config or {}).get("title_predicate")
        if not isinstance(block, dict):
            raise TitlePredicateConfigError("configs/factcheck.yaml has no title_predicate block")
        raw = block.get("patterns")
        if not isinstance(raw, dict):
            raise TitlePredicateConfigError(
                "title_predicate.patterns must map a language to a list"
            )
        self.pattern_hash = pattern_set_hash(config)
        self.patterns: dict[str, list[PredicatePattern]] = {lang: [] for lang in _LANGS}
        self.guards: dict[str, list[Guard]] = {lang: [] for lang in _LANGS}
        seen: set[str] = set()
        for lang, entries in raw.items():
            if lang not in self.patterns:
                raise TitlePredicateConfigError(
                    f"title_predicate.patterns has language {lang!r}; allowed: {list(_LANGS)}"
                )
            for entry in entries or []:
                self.patterns[lang].append(self._compile(lang, entry, seen))
        if not any(self.patterns.values()):
            raise TitlePredicateConfigError("title_predicate.patterns holds no pattern")
        guards = block.get("guards") or {}
        if not isinstance(guards, dict):
            raise TitlePredicateConfigError("title_predicate.guards must map a language to a list")
        for lang, entries in guards.items():
            if lang not in self.guards:
                raise TitlePredicateConfigError(
                    f"title_predicate.guards has language {lang!r}; allowed: {list(_LANGS)}"
                )
            for entry in entries or []:
                self.guards[lang].append(self._compile_guard(lang, entry, seen))

    @staticmethod
    def _regex(pattern_id: str, source: Any) -> re.Pattern[str]:
        if not (isinstance(source, str) and source):
            raise TitlePredicateConfigError(f"pattern {pattern_id!r} has no pattern text")
        try:
            return re.compile(unicodedata.normalize("NFC", source), re.IGNORECASE)
        except re.error as exc:
            raise TitlePredicateConfigError(f"pattern {pattern_id!r}: {exc}") from exc

    @classmethod
    def _compile(cls, lang: str, entry: Any, seen: set[str]) -> PredicatePattern:
        if not isinstance(entry, dict):
            raise TitlePredicateConfigError(f"a {lang} pattern entry is not a mapping: {entry!r}")
        pattern_id = str(entry.get("id") or "").strip()
        rating = entry.get("rating")
        # A bare `False` in YAML is a boolean, so the rating must be a string.
        if not (pattern_id and isinstance(rating, str) and rating.strip()):
            raise TitlePredicateConfigError(
                f"a {lang} pattern needs an id and a string rating (quote False): {entry!r}"
            )
        if pattern_id in seen:
            raise TitlePredicateConfigError(f"pattern id {pattern_id!r} is used twice")
        seen.add(pattern_id)
        return PredicatePattern(
            pattern_id, lang, rating.strip(), cls._regex(pattern_id, entry.get("pattern"))
        )

    @classmethod
    def _compile_guard(cls, lang: str, entry: Any, seen: set[str]) -> Guard:
        if not isinstance(entry, dict):
            raise TitlePredicateConfigError(f"a {lang} guard entry is not a mapping: {entry!r}")
        guard_id = str(entry.get("id") or "").strip()
        if not guard_id:
            raise TitlePredicateConfigError(f"a {lang} guard needs an id: {entry!r}")
        if guard_id in seen:
            raise TitlePredicateConfigError(f"pattern id {guard_id!r} is used twice")
        seen.add(guard_id)
        return Guard(guard_id, lang, cls._regex(guard_id, entry.get("pattern")))

    def classify(self, title: str) -> tuple[PredicateMatch | None, str | None]:
        """The first match for ``title``, and the id of the guard that cancelled it.

        Returns ``(match, None)`` for a verdict, ``(None, guard_id)`` when a pattern
        matched and a guard cancelled it, and ``(None, None)`` when nothing matched.
        A headline in two scripts is tried against both language lists.
        """
        text = normalize_headline(title)
        if not text:
            return None, None
        languages = _languages_to_try(text)
        for lang in languages:
            for pattern in self.patterns[lang]:
                found = pattern.regex.search(text)
                if not found:
                    continue
                for guard_lang in languages:
                    for guard in self.guards[guard_lang]:
                        if guard.regex.search(text):
                            return None, guard.id
                return PredicateMatch(pattern.id, lang, pattern.rating, found.group(0)), None
        return None, None

    def read(self, title: str) -> PredicateMatch | None:
        """The verdict match for ``title``, or ``None``."""
        return self.classify(title)[0]


def reader_from_config(config: dict[str, Any] | None = None) -> TitlePredicateReader:
    """A reader over ``config``, or over `configs/factcheck.yaml` when none is given."""
    return TitlePredicateReader(config if config is not None else load_yaml("factcheck"))
