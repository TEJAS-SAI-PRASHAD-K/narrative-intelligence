"""Normalization primitives. One function per concern, all pure, all unit-tested.

Deliberately *not* done here: lowercasing, punctuation stripping, stopword
removal, stemming. Phase 2 embeds this text with transformers and needs the
original surface form -- "BREAKING!!!" and "breaking" are different signals.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections.abc import Iterable, Sequence
from functools import lru_cache
from html import unescape
from typing import Any
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger(__name__)

# --- constants ------------------------------------------------------------

#: Zero-width and other invisible characters. Common in copy-paste botnets and
#: in unicode-obfuscated spam, so we strip them rather than embed them.
_INVISIBLE = dict.fromkeys(
    [
        0x00AD,  # soft hyphen
        0x200B,  # zero width space
        0x200C,  # zero width non-joiner
        0x200D,  # zero width joiner
        0x200E,  # LTR mark
        0x200F,  # RTL mark
        0x2060,  # word joiner
        0xFEFF,  # BOM / zero width no-break space
    ]
)

_URL_RE = re.compile(r"""(?i)\bhttps?://[^\s<>"'`\\]+""")
_HASHTAG_RE = re.compile(r"(?<![\w&])#(\w{1,139})", re.UNICODE)
_MENTION_RE = re.compile(r"(?<![\w])@([A-Za-z0-9_.\-]{1,64}(?:@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})?)")
_WS_RUN = re.compile(r"[^\S\n]+")  # horizontal whitespace only
_NEWLINE_RUN = re.compile(r"\n{3,}")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_TRAILING_PUNCT = ".,;:!?)]}'\"»”’"

#: Reddit/YouTube tombstones. Text that is one of these carries no information.
DELETED_MARKERS = frozenset({"[deleted]", "[removed]", "deleted", "removed", "[deleted by user]"})

#: Link shorteners worth one HEAD request to expand. Kept short on purpose:
#: resolving every link in the corpus is a Phase 2 enrichment job, not this.
SHORTENER_DOMAINS = frozenset(
    {
        "bit.ly",
        "t.co",
        "tinyurl.com",
        "goo.gl",
        "ow.ly",
        "buff.ly",
        "dlvr.it",
        "ift.tt",
        "youtu.be",
        "trib.al",
        "shar.es",
        "rb.gy",
        "cutt.ly",
        "is.gd",
        "wp.me",
        "amzn.to",
        "nyti.ms",
        "reut.rs",
        "apne.ws",
        "cnn.it",
        "bbc.in",
    }
)

#: Tracking parameters that fragment otherwise-identical URLs and inflate the
#: apparent diversity of a coordinated link-drop campaign.
_TRACKING_PARAMS = re.compile(
    r"(?i)(^|&)(utm_[a-z_]+|fbclid|gclid|mc_[a-z]+|igshid|ref_src|ref_url|s|si|CMP|cmpid|smid)=[^&]*"
)

# tldextract downloads the public suffix list on first use. We pin it to the
# bundled snapshot so tests (and airgapped runs) never touch the network.
_extractor = None


def _get_extractor():
    global _extractor
    if _extractor is None:
        import tldextract

        _extractor = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)
    return _extractor


# --- text -----------------------------------------------------------------


# --- html parser backend --------------------------------------------------

#: Resolved once per process. ``False`` means "not yet looked up"; ``None``
#: means "looked up and unusable", which puts the regex fallbacks in play.
_html_parser_cls: Any = False


def html_parser_class() -> Any:
    """The selectolax parser class, or ``None`` if the library is unusable.

    selectolax ships two backends and which one you get depends on the version:

    * ``selectolax.lexbor.LexborHTMLParser`` -- current, HTML5-conformant.
    * ``selectolax.parser.HTMLParser`` -- the Modest backend, **removed in
      selectolax 1.0**. Importing it on 1.0+ raises ``ImportError`` with a
      message telling you to switch to lexbor.

    Both call sites below used to import the Modest backend inside a bare
    ``except Exception`` and fall back to a regex. So the day the installed
    selectolax crossed 1.0, this project silently stopped parsing HTML and
    started regexing it -- no error, no log line, no test failure that said
    *why*. The regex cannot do what :func:`extract_html_links` documents as its
    whole purpose, so every Mastodon hashtag and mention anchor has been
    entering ``urls`` and ``domains`` and polluting the Domain Risk pillar.

    The lesson is in the structure, not the version pin: an optional
    accelerator may be swapped for a fallback silently, but a **core
    dependency** disappearing must be loud. Hence the one-time warning. If this
    ever logs, the numbers downstream are different from the documented ones.
    """
    global _html_parser_cls
    if _html_parser_cls is not False:
        return _html_parser_cls
    try:
        from selectolax.lexbor import LexborHTMLParser

        _html_parser_cls = LexborHTMLParser
        return _html_parser_cls
    except ImportError:
        pass
    try:
        from selectolax.parser import HTMLParser

        _html_parser_cls = HTMLParser
    except ImportError:
        # Not pragma'd: this path is covered by a test, because it is the path
        # that quietly changed the corpus last time.
        log.warning(
            "selectolax is unusable: neither the lexbor nor the Modest backend could be "
            "imported. Falling back to REGEX html handling, which cannot exclude hashtag "
            "and mention anchors from outbound links -- `urls` and `domains` will include "
            'internal navigation links. Fix with `pip install -e ".[sources]"` or '
            "`pip install -U selectolax`."
        )
        _html_parser_cls = None
    return _html_parser_cls


def strip_html(s: str | None) -> str:
    """HTML -> plaintext. Handles Mastodon status markup and RSS-escaped entities.

    Block boundaries become newlines so that "a</p><p>b" does not become "ab".
    """
    if not s:
        return ""
    if "<" not in s and "&" not in s:
        return s
    # Convert block boundaries to newlines *before* tag stripping.
    s = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", s)
    s = re.sub(r"(?i)</\s*(p|div|li|tr|h[1-6]|blockquote)\s*>", "\n", s)
    parser = html_parser_class()
    if parser is None:
        text = re.sub(r"<[^>]+>", "", s)
    else:
        text = parser(s).text(separator="")
    # RSS commonly double-escapes; unescape twice at most, never in a loop.
    text = unescape(text)
    if "&" in text and re.search(r"&(?:amp|lt|gt|quot|#\d+);", text):
        text = unescape(text)
    return text


def extract_html_links(html: str | None) -> list[str]:
    """Outbound ``href`` targets from markup, before the tags are stripped.

    Mastodon renders a link as ``<a href="https://real/url">real/ur…</a>``: the
    visible text is *truncated*, so regexing the stripped text loses the actual
    destination. Internal navigation links (hashtag and mention anchors) are
    excluded -- they are not outbound links and would pollute the domain counts
    that feed the Domain Risk pillar.
    """
    if not html or "<a" not in html.lower():
        return []
    parser = html_parser_class()
    if parser is None:
        # Degraded: this returns hashtag and mention anchors too, because a
        # regex cannot see the class and rel attributes that identify them.
        # html_parser_class() has already warned.
        return re.findall(r"""(?i)<a[^>]+href=["'](https?://[^"']+)["']""", html)
    nodes = parser(html).css("a")
    out: list[str] = []
    for node in nodes:
        href = node.attributes.get("href") or ""
        if not href.lower().startswith(("http://", "https://")):
            continue
        classes = (node.attributes.get("class") or "").lower()
        rel = (node.attributes.get("rel") or "").lower()
        if "hashtag" in classes or "mention" in classes or "tag" in rel.split():
            continue
        out.append(href)
    return out


def clean_text(s: str | None) -> str:
    """NFKC-normalize, drop invisible characters, collapse whitespace.

    Newlines survive (capped at two consecutive) because paragraph structure is
    information; horizontal whitespace runs collapse to a single space.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_INVISIBLE)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS_RUN.sub(" ", s)
    s = _NEWLINE_RUN.sub("\n\n", s)
    return "\n".join(line.strip() for line in s.split("\n")).strip()


def is_deleted_text(s: str | None) -> bool:
    """True for platform tombstones (``[deleted]``, ``[removed]``)."""
    if s is None:
        return False
    return s.strip().lower() in DELETED_MARKERS


# --- urls -----------------------------------------------------------------


def _trim_url(url: str) -> str:
    url = url.strip()
    # Trailing punctuation from prose: "see https://x.com/a." -> drop the dot.
    while url and url[-1] in _TRAILING_PUNCT:
        # Keep a closing paren if the URL contains a matching opening one.
        if url[-1] == ")" and url.count("(") > url.count(")") - 1:
            break
        url = url[:-1]
    return url


def canonicalize_url(url: str) -> str:
    """Lowercase the host, drop the fragment and known tracking parameters."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    query = _TRACKING_PARAMS.sub("", parts.query or "").lstrip("&")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, query, ""))


def extract_urls(text: str | None, raw: Any = None) -> list[str]:
    """Outbound links for a record.

    Structured entities from the source win over regex when present -- the
    platform already parsed them and we should not re-guess. ``raw`` may be a
    list of urls, a dict containing url-ish fields, or ``None``.
    """
    urls: list[str] = []
    for candidate in _structured_urls(raw):
        urls.append(candidate)
    for match in _URL_RE.finditer(text or ""):
        urls.append(match.group(0))

    out: list[str] = []
    seen: set[str] = set()
    for url in urls:
        url = canonicalize_url(_trim_url(url))
        if not url or len(url) > 2048:
            continue
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _structured_urls(raw: Any) -> Iterable[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.startswith("http") else []
    if isinstance(raw, (list, tuple, set)):
        found: list[str] = []
        for item in raw:
            found.extend(_structured_urls(item))
        return found
    if isinstance(raw, dict):
        found = []
        for key in ("url", "href", "expanded_url", "unshortened_url", "link", "documentidentifier"):
            value = raw.get(key)
            if isinstance(value, str) and value.startswith("http"):
                found.append(value)
        for key in ("urls", "links", "entities"):
            if key in raw:
                found.extend(_structured_urls(raw[key]))
        return found
    return []


def resolve_domain(url: str | None) -> str | None:
    """Registrable domain, lowercased, ``www.`` implicitly gone.

    ``https://WWW.News.BBC.co.uk/x?y`` -> ``bbc.co.uk``. Returns ``None`` when
    the input has no registrable domain (bare IPs, ``mailto:``, junk).
    """
    if not url:
        return None
    if "://" not in url:
        # A scheme with no authority (mailto:, tel:, javascript:) has no host.
        if re.match(r"(?i)^[a-z][a-z0-9+.\-]*:", url):
            return None
        url = "http://" + url
    elif not url.lower().startswith(("http://", "https://")):
        return None
    ext = _get_extractor()(url)
    if not ext.domain or not ext.suffix:
        return None
    return f"{ext.domain}.{ext.suffix}".lower()


def resolve_domains(urls: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for url in urls:
        domain = resolve_domain(url)
        if domain and domain not in seen:
            seen.add(domain)
            out.append(domain)
    return out


def is_shortlink(url: str) -> bool:
    return (resolve_domain(url) or "") in SHORTENER_DOMAINS


def unshorten(url: str, session: Any = None, timeout: float = 5.0) -> str:
    """Expand a shortened URL with a single HEAD request. Network; opt-in.

    Failure returns the input unchanged -- an unresolved shortlink is worth more
    than a dropped record.
    """
    if not is_shortlink(url):
        return url
    try:
        import requests

        client = session or requests
        resp = client.head(url, allow_redirects=True, timeout=timeout)
        return canonicalize_url(resp.url) or url
    except Exception as exc:  # pragma: no cover - network path
        log.debug("unshorten failed for %s: %s", url, exc)
        return url


# --- entities -------------------------------------------------------------


def extract_hashtags(text: str | None, raw: Any = None) -> list[str]:
    """Hashtags, lowercased, without the ``#``. Structured entities preferred."""
    tags: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict) and item.get("name"):
                tags.append(str(item["name"]))
            elif isinstance(item, str):
                tags.append(item.lstrip("#"))
    tags.extend(_HASHTAG_RE.findall(text or ""))
    return _dedupe_lower(tags)


def extract_mentions(text: str | None, raw: Any = None) -> list[str]:
    """Mentions without the leading ``@``. Fediverse handles keep their instance."""
    mentions: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict) and (item.get("acct") or item.get("username")):
                mentions.append(str(item.get("acct") or item.get("username")))
            elif isinstance(item, str):
                mentions.append(item.lstrip("@"))
    mentions.extend(_MENTION_RE.findall(text or ""))
    return _prefer_qualified(_dedupe_lower(mentions))


def _prefer_qualified(mentions: list[str]) -> list[str]:
    """Collapse ``colleague`` into ``colleague@instance.tld`` when both appear.

    Mastodon renders a remote mention as a bare ``@colleague`` in the visible
    text while the structured entity carries the full ``colleague@instance.tld``.
    Keeping both counts one account twice and splits it into two nodes in Phase
    2's coordination graph.
    """
    qualified_locals = {m.split("@", 1)[0] for m in mentions if "@" in m}
    return [m for m in mentions if "@" in m or m not in qualified_locals]


def _dedupe_lower(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value).strip().lstrip("#@").lower()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


# --- script ---------------------------------------------------------------

#: Unicode block -> script name, for the scripts this corpus can contain.
#: Explicit ranges rather than ``unicodedata.name()`` per character: the name
#: lookup is ~40x slower and this runs on every record, including 15k-character
#: articles.
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0041, 0x005A, "Latin"),
    (0x0061, 0x007A, "Latin"),
    (0x00C0, 0x024F, "Latin"),
    (0x0370, 0x03FF, "Greek"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0590, 0x05FF, "Hebrew"),
    # Urdu, Kashmiri and Sindhi are written in Perso-Arabic.
    (0x0600, 0x06FF, "Arabic"),
    (0x0750, 0x077F, "Arabic"),
    # The languages of India, in Unicode block order.
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x4E00, 0x9FFF, "Han"),
)

#: Scripts used by exactly one major Indian language, so the script alone
#: identifies it. Devanagari is absent on purpose -- it carries Hindi, Marathi,
#: Nepali, Sanskrit, Bhojpuri and Konkani, so script is not a language there
#: and pretending otherwise would mislabel every Marathi record as Hindi.
SCRIPT_TO_LANG: dict[str, str] = {
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Gujarati": "gu",
    "Gurmukhi": "pa",
    "Oriya": "or",
    "Sinhala": "si",
}


@lru_cache(maxsize=4096)
def _script_of(char: str) -> str | None:
    code = ord(char)
    for start, end, name in _SCRIPT_RANGES:
        if start <= code <= end:
            return name
    return None


def script_profile(text: str | None) -> dict[str, float]:
    """Share of alphabetic characters belonging to each writing system.

    Shares sum to 1.0 over the characters that could be attributed; characters
    in no listed block are excluded from the denominator rather than lumped
    into a bucket, so a share is always "of the text we can read".

    Empty input gives ``{}``, not ``{"Latin": 0.0}`` -- no evidence is not the
    same as evidence of nothing.
    """
    counts: dict[str, int] = {}
    total = 0
    for char in text or "":
        if not char.isalpha():
            continue
        script = _script_of(char)
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
        total += 1
    if not total:
        return {}
    return {name: count / total for name, count in counts.items()}


def dominant_script(text: str | None, min_share: float = 0.5) -> str | None:
    """The one script carrying more than ``min_share`` of the letters, if any."""
    profile = script_profile(text)
    if not profile:
        return None
    name, share = max(profile.items(), key=lambda kv: kv[1])
    return name if share >= min_share else None


def is_code_mixed(text: str | None, min_share: float = 0.10) -> bool:
    """True when two or more writing systems each carry ``min_share`` of the text.

    This is *script* mixing, which is only one of the two kinds of code-mixing
    Indian social media produces. It catches "यह video बिलकुल fake है"; it
    cannot catch fully romanized "yeh video bilkul fake hai", which has one
    script and two languages. Use :func:`looks_romanized_hindi` for that.
    """
    profile = script_profile(text)
    return sum(1 for share in profile.values() if share >= min_share) >= 2


# --- romanized Hindi ------------------------------------------------------

#: High-frequency Hindi/Urdu function and discourse words as typed in Latin
#: script, restricted to tokens that are NOT also English words.
#:
#: The exclusions matter more than the inclusions. "to", "is", "par", "ye",
#: "log", "the", "hum", "main", "ab", "do" and "me" are all frequent romanized
#: Hindi *and* ordinary English, and every one of them is deliberately absent:
#: including any of them trades a false "this English comment is Hindi" for a
#: marginal recall gain, and a wrong language label is worse than a missing one
#: because the Phase 2 scorer acts on it.
#:
#: Hindi and Urdu share this vocabulary almost entirely in speech, so this
#: detects the pair, not Hindi alone. It detects no other Indian language:
#: romanized Tamil, Telugu and Malayalam score zero against it, by construction
#: rather than by accident. Extending it is a per-language lexicon each time.
ROMANIZED_HINDI_MARKERS = frozenset(
    """
    hai hain hun hoon nahi nahin nhi kya kyun kyon kaise kaisa kaisi jo woh yeh
    bhi aur ko ka ki se mein mera meri mere mujhe mujhse tumhe tumhara aap aapka
    aapko aapne apna apne unka unke uska iska inka kuch sab sabko sabke bahut
    bohot bilkul sirf abhi pehle baad liye karo karna kiya karta karte karti
    kijiye raha rahi rahe tha thi gaya gayi gaye diya dena lena hona hoga hogi
    chahiye lekin agar phir matlab acha accha achha theek thik sach jhooth jhoot
    khabar batao bataya bataye dekha dekho dekhiye suno bhejo bhejein bhej
    jaldi dhyan jankari galat sahi logo logon yahan wahan kahan kaun kab
    kitna kitne jarur zaroor zarur waise aisa aise jaisa jaise unhe inhe hamara
    hamare humara tumne usne isne kisne koi kisi wala wali wale banaya banana
    dijiye milega milta sakta sakte sakti chahta chahte raho rakho samajh samjho
    bola bole boli kehte kehta kaha kahte nikla nikli nikle lagta lagti lage
    """.split()
)

_LATIN_TOKEN_RE = re.compile(r"[a-z]+")

#: Two markers minimum, not one. Measured: six adversarial English sentences
#: ("jo biden said that", "the ki is a japanese concept", "se habla espanol
#: here") each land exactly one marker, and the second-marker requirement is
#: the only thing that rejects them. See the thresholds note in
#: :func:`looks_romanized_hindi`.
_ROMANIZED_MIN_MARKERS = 2
_ROMANIZED_MIN_SHARE = 0.06


def romanized_hindi_markers(text: str | None) -> tuple[int, float]:
    """``(marker count, marker share of Latin tokens)``. Pure; no model."""
    tokens = _LATIN_TOKEN_RE.findall((text or "").lower())
    if not tokens:
        return 0, 0.0
    hits = sum(1 for token in tokens if token in ROMANIZED_HINDI_MARKERS)
    return hits, hits / len(tokens)


def looks_romanized_hindi(
    text: str | None,
    min_markers: int = _ROMANIZED_MIN_MARKERS,
    min_share: float = _ROMANIZED_MIN_SHARE,
) -> bool:
    """Whether Latin-script text is Hindi/Urdu typed in Latin letters.

    Deliberately a lexicon and not a classifier. langdetect has no romanized
    Hindi class at all, so it cannot abstain on this input -- it picks the
    nearest of its 55 trained languages and commits. Measured on 10 hand-written
    romanized Hindi sentences it returned Swahili five times, Estonian twice,
    Somali twice and Turkish once: **0/10**, and never once English, which is
    the detail that matters. A confident `sw` is not a near miss; it sends the
    record down the "non-English, do not score" path and out of the analysis
    entirely, silently.

    Thresholds were swept against the ingested corpus (454 real English news
    records) and two hand-written probe sets. Every setting from
    ``(1, 0.06)`` upward gave zero false positives on the real English; the
    binding case was short informal English, where ``min_markers=1`` admits
    the adversarial examples above. ``(2, 0.06)`` is the loosest setting that
    rejects all of them.

    **Recall is not measured.** The positive set is hand-written, and written
    by the same person who chose the lexicon, so its 20/20 is circular and is
    an upper bound rather than a result. The false-positive rate is the only
    number here measured against data nobody curated for it. Closing this
    properly needs hand-labelled romanized records out of the real corpus,
    which is a Phase 2 labelling task, not a config change.
    """
    hits, share = romanized_hindi_markers(text)
    return hits >= min_markers and share >= min_share


# --- language -------------------------------------------------------------

_MIN_LANG_CHARS = 20


def detect_lang(text: str | None, min_chars: int = _MIN_LANG_CHARS) -> str | None:
    """ISO 639-1 language code, or ``None``.

    Under ~20 characters langdetect is close to a coin flip, so we return
    ``None`` rather than a guess: an honest null is cheaper to handle downstream
    than a confident wrong label.

    Two script-aware corrections sit on top of langdetect, in this order:

    1. **Romanized Hindi/Urdu.** Checked *before* langdetect runs, because
       langdetect cannot abstain on input it has no class for and answers with
       a confident wrong language instead. See :func:`looks_romanized_hindi`.
    2. **Short text in a single-language script.** langdetect abstains below
       ``min_chars``, but Tamil, Telugu, Gujarati and the other scripts in
       :data:`SCRIPT_TO_LANG` are used by exactly one major language, so the
       script settles it with no statistics at all. Devanagari is excluded --
       it carries Hindi, Marathi and Nepali, and guessing Hindi there would
       mislabel every short Marathi record.

    On the 548-record corpus ingested on 2026-10-09 neither correction fires:
    news articles are long, and langdetect already got all 90 Devanagari
    records right. Both exist for the short, romanized, social-media text that
    the YouTube, Reddit and Mastodon adapters produce.
    """
    if not text:
        return None
    stripped = text.strip()

    # Latin-script Hindi, before langdetect gets a chance to call it Swahili.
    if looks_romanized_hindi(stripped):
        return "hi"

    if len(stripped) < min_chars:
        # Too short for statistics, but maybe not too short for the alphabet.
        script = dominant_script(stripped)
        return SCRIPT_TO_LANG.get(script) if script else None
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # deterministic output across runs
        code = detect(stripped)
    except Exception:
        return None
    return code.split("-")[0].lower() if code else None


# --- near-duplicate hashing ----------------------------------------------


def _shingles(text: str, n: int = 3) -> list[str]:
    """Lowercased word n-grams. Lowercasing here affects hashing only, not ``text``."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return []
    if len(words) < n:
        return [" ".join(words)]
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def simhash(text: str | None, n: int = 3, bits: int = 64) -> int:
    """64-bit simhash over word 3-grams.

    Cheap near-duplicate detection for Phase 2: two texts within a small Hamming
    distance are almost certainly the same claim reposted. Returns ``0`` for
    text with no word content -- treat 0 as "no signal", not as a real hash.
    """
    shingles = _shingles(text or "", n=n)
    if not shingles:
        return 0
    vector = [0] * bits
    for shingle in shingles:
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=bits // 8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(bits):
            vector[bit] += 1 if (value >> bit) & 1 else -1
    out = 0
    for bit in range(bits):
        if vector[bit] > 0:
            out |= 1 << bit
    return out


def hamming(a: int, b: int) -> int:
    """Bit distance between two simhashes."""
    return bin(a ^ b).count("1")


# --- convenience for adapters --------------------------------------------


def build_text_fields(
    raw_text: str | None,
    *,
    is_html: bool = False,
    structured_urls: Any = None,
    structured_tags: Any = None,
    structured_mentions: Any = None,
) -> dict[str, Any]:
    """Run the whole text pipeline once and hand back the schema fields.

    Adapters call this so the six of them cannot drift apart in how they clean
    text -- the drift would show up in Phase 2 as a spurious platform effect.
    """
    if is_html:
        # Pull hrefs first: tag stripping discards the real destinations.
        href_urls = extract_html_links(raw_text)
        structured_urls = (
            href_urls
            if structured_urls is None
            else [
                *href_urls,
                *(structured_urls if isinstance(structured_urls, list) else [structured_urls]),
            ]
        )
        text = clean_text(strip_html(raw_text))
    else:
        text = clean_text(raw_text)
    urls = extract_urls(text, structured_urls)
    return {
        "text": text,
        "lang": detect_lang(text),
        "urls": urls,
        "domains": resolve_domains(urls),
        "hashtags": extract_hashtags(text, structured_tags),
        "mentions": extract_mentions(text, structured_mentions),
        "simhash": simhash(text),
    }
