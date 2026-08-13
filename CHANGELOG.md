# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-13

Phase 1: the data and ingestion layer. Produces a reproducible, schema-normalized,
multi-platform corpus. No models, no API, no dashboard — those are Phases 2–6.

### Added

**Schema (the project's contract)**
- `Record` and `Author` Pydantic v2 models, plus `EngagementMetrics` and a `DropReason`
  enum, in `ingest/schema.py`.
- Validation-enforced invariants: timezone-aware UTC timestamps (naive datetimes are
  rejected, never coerced), source-namespaced `id`/`author_id`/`parent_id`/
  `conversation_id`, always-present engagement keys where `null` ≠ `0`, and no extra
  top-level fields.

**Normalization** (`ingest/normalize.py`)
- Pure, unit-tested functions: `strip_html`, `clean_text` (NFKC, zero-width stripping,
  whitespace collapse; surface form preserved for Phase 2's transformers),
  `extract_urls`, `canonicalize_url`, `resolve_domain`, `extract_hashtags`,
  `extract_mentions`, `detect_lang`, and a 64-bit `simhash` over word 3-grams.

**Storage** (`ingest/store.py`)
- Parquet corpus partitioned `source=<source>/date=<YYYY-MM-DD>/`, written against an
  explicit Arrow schema; `raw` stored as a JSON string so per-source payload drift cannot
  break the physical schema.
- Id-level dedupe against what is already on disk, author roll-up merging, and
  `data/manifest.json` with source URL, SHA256, byte size and row count per artifact.

**Infrastructure**
- `ingest/ratelimit.py`: token bucket, exponential backoff with jitter, and an HTTP
  session that honours `Retry-After` / `X-RateLimit-Reset` by sleeping to the reset.
- `ingest/checkpoint.py`: atomically-written per-source cursors and a YouTube quota
  ledger that charges units before a call and hard-stops on the daily budget.
- `ingest/sources/base.py`: one run loop for all adapters — buffered flush, dedupe, drop
  accounting by reason code, and `SourceUnavailable` for graceful degradation.

**Source adapters** — all six, each tested against recorded fixtures
- `reddit_convokit` (primary Reddit): threaded conversations, stable speaker ids; deleted
  bodies dropped, deleted authors kept and flagged.
- `mastodon`: paginated public/hashtag timelines plus a bounded live tail; boosts emitted
  as their own records so the cross-instance amplification edge survives.
- `gdelt`: DOC 2.0 topic search and the raw 15-minute drops; `mentions` kept as a side
  artifact rather than forced into the record schema.
- `news_rss`: RSS/Atom with optional NewsAPI, budgeted and robots.txt-gated full-text
  extraction, and syndicated copy kept rather than deduplicated.
- `youtube`: quota-budgeted discovery, cheap hydration, and threaded comments.
- `reddit_kaggle`: per-slug explicit column maps; zstd-streaming loader for Academic
  Torrents dumps; threading reported as absent rather than fabricated.

**CLI and deliverables**
- `ingest/cli.py`: `fetch`, `fetch-all`, `stats`, `validate`, `manifest`, `show-config`,
  `mastodon-register`, `mastodon-stream`.
- `notebooks/01_corpus_eda.ipynb`, generated from `notebooks/build_eda_notebook.py` so its
  diffs stay reviewable; ends with an explicit coverage-and-bias statement.
- `scripts/download_benchmarks.py` for LIAR, CoAID and FakeNewsNet with checksums.
- `configs/sources.yaml` and `configs/topics.yaml`: changing the case under study requires
  no code change.
- Test suite of 300+ tests with network access blocked at the socket layer.

### Fixed

Found by running the pipeline against live APIs rather than trusting documentation.

- **GDELT DOC query form.** `gdeltdoc`'s `keyword` means an *exact phrase* and OR-joins a
  list; a hand-written boolean string is quoted whole and rejected. Topics now carry
  `gdelt_keywords` as phrase lists.
- **GDELT language filter.** A single-element language list renders as
  `(sourcelang:English)` — parentheses around a non-OR'd term — which GDELT rejects with
  "Parentheses may only be used around OR'd statements". One language is now passed as a
  bare string; pinned by an offline test.
- **GDELT GKG parsing.** Raised the CSV field-size limit (GCAM/V2Themes exceed the 128KB
  default on valid data), added `csv.Error` to the wrapped exceptions so one malformed row
  cannot fail a run, and switched to `islice` so a busy drop is streamed rather than
  materialized — this changed a 15-minute hang into a run of seconds.
- **GDELT unavailable drops.** `lastupdate.txt` lists files that return 404; downloads are
  now checked for status, emptiness and zip magic, and no truncated artifact is left on
  disk to be "reused" identically forever.
- **HTML link extraction.** Anchor `href`s are read before tag stripping, because Mastodon
  truncates the visible link text and only the `href` holds the real destination.
- **Mention identity.** A bare `@colleague` in visible text no longer survives alongside
  the structured `colleague@instance.tld`, which would have split one account into two
  nodes in the Phase 2 coordination graph.
- **Domain resolution.** Non-HTTP schemes (`mailto:`, `tel:`) no longer yield a
  registrable domain.
- **Kaggle local path.** Reading an already-downloaded dump no longer requires Kaggle
  credentials, which had made the documented Academic Torrents workflow impossible.
- **Manifest checksums.** API responses are archived to `data/raw/<source>/` before
  parsing, so manifest entries hash bytes that actually exist.
- **Dead feeds.** Removed `feeds.reuters.com` (no longer resolves) and
  `apnews.com/index.rss` (returns zero entries); added a per-domain circuit breaker so
  paywalled outlets are tried three times, not sixty.

### Known limitations

- English-scoped by construction; not a random sample of any population.
- No X/Twitter, Telegram, Facebook, WhatsApp or TikTok.
- Reddit data is historical (ConvoKit snapshots); there is no live Reddit path.
- Coordination-graph work is valid only on ConvoKit Reddit, Mastodon and YouTube.
- `mastodon.social` returns nothing for the federated public timeline under a plain read
  token, so Mastodon coverage is hashtag-driven and topic-biased.
- GDELT enforces its rate limit with a stateful penalty window; a throttled topic is
  skipped for that run and picked up on the next.

[0.1.0]: https://github.com/TEJAS-SAI-PRASHAD-K/narrative-intelligence/releases/tag/v0.1.0
