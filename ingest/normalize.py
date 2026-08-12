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
    text = None
    try:
        from selectolax.parser import HTMLParser

        text = HTMLParser(s).text(separator="")
    except Exception:  # pragma: no cover - selectolax is a core dep, this is belt-and-braces
        text = re.sub(r"<[^>]+>", "", s)
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
    try:
        from selectolax.parser import HTMLParser

        nodes = HTMLParser(html).css("a")
    except Exception:  # pragma: no cover - fall back to attribute regex
        return re.findall(r"""(?i)<a[^>]+href=["'](https?://[^"']+)["']""", html)
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
    return _dedupe_lower(mentions)


def _dedupe_lower(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value).strip().lstrip("#@").lower()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


# --- language -------------------------------------------------------------

_MIN_LANG_CHARS = 20


def detect_lang(text: str | None, min_chars: int = _MIN_LANG_CHARS) -> str | None:
    """ISO 639-1 language code, or ``None``.

    Under ~20 characters langdetect is close to a coin flip, so we return
    ``None`` rather than a guess: an honest null is cheaper to handle downstream
    than a confident wrong label.
    """
    if not text:
        return None
    stripped = text.strip()
    if len(stripped) < min_chars:
        return None
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
