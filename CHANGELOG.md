# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The corpus is retargeted from the United States to India, in English and Hindi.
Data layer only: the case, the sources and the language detection. The modeling
layer still scores English only, which is now a documented gap rather than a
decision.

### Added

- **Eighteen Indian RSS feeds in two roles.** Nine fact-checkers (BOOM, Alt News
  in English and Hindi, Factly, Newschecker, Quint WebQoof, Fact Crescendo,
  Vishvas News, DigitEye) carry a claim *and a verdict* and are the only
  supervision in the corpus; nine mainstream outlets supply the coverage a claim
  spreads against. Every URL probed live on 2026-10-09 and then exercised
  through the real adapter: 548 records, 16 domains, `en`/`hi`/`te`/`id` =
  454/90/3/1.
- **GDELT country scoping** (`gdelt.doc_api.countries`, default `[IN]`, FIPS
  2-letter). Without it the Indian topics return mostly US coverage.
- **The fourteen languages of India in `LANGUAGE_CODES`**, both spellings of
  Odia included. GDELT emits language *names*, so an unmapped name is a null
  `lang`, not a wrong one.
- **YouTube `regionCode`** (ISO 3166-1 alpha-2, unlike GDELT's FIPS on the same
  corpus) and optional `relevanceLanguage`, both omitted from the call entirely
  when unset because the client serializes an explicit `None`.
- **Script detection** — `script_profile`, `dominant_script`, `is_code_mixed` —
  over explicit Unicode ranges at 13.4 M chars/sec.
- **Romanized-Hindi detection** (`looks_romanized_hindi`), a function-word
  lexicon checked before langdetect runs.
- **Short-text resolution by alphabet.** langdetect abstains under 20 chars, but
  Tamil, Telugu, Kannada, Malayalam, Gujarati, Gurmukhi, Oriya and Sinhala are
  each used by one major language. Devanagari is excluded on purpose: Hindi,
  Marathi and Nepali share it, so short Devanagari stays an honest null.

### Fixed

- **Romanized Hindi was being assigned a confident wrong language.** langdetect
  has no class for Hindi in Latin script, so it cannot abstain — on ten
  hand-written sentences it returned Swahili ×5, Estonian ×2, Somali ×2,
  Turkish ×1 and English ×0. Because `modeling/config.py` gates scoring on
  `lang in languages`, a `sw` label silently routed the record out of the
  analysis rather than merely mislabelling it, which would have quietly dropped
  most of what the YouTube, Reddit and Mastodon adapters collect.
- **`gdeltdoc` silently accepts invalid country codes.** `"India"` and `"IND"`
  are emitted into the query and answered with zero rows — an empty corpus that
  reads as a quiet news week. `_country_filter` shape-checks and warns loudly
  while still passing the value through.
- **NewsAPI's hardcoded `language="en"`** is config-driven, with the reason it
  stays English recorded: NewsAPI's supported set has no Hindi.

### Changed

- `configs/topics.yaml` replaced: four Indian topic areas (communal, electoral,
  health, scams and synthetic media) with English and Hindi seeds, replacing the
  US vaccine/election/climate case.
- `news_rss.fulltext.max_articles` 60 → 120, and
  `youtube.discovery.max_searches_per_run` 4 → 12 so one run covers the case
  (1,200 of 10,000 daily quota units).
- `topics.yaml`'s `languages:` key is marked declarative, because nothing reads
  it. The three real enforcement points are named instead.

### Known gaps

- **The GDELT country filter is not verified live.** GDELT's stateful penalty
  window refused every attempt across ~30 minutes of spacing. The query form is
  pinned offline by tests; the end-to-end check is owed.
- **Romanized-Hindi recall is not a measured number.** Zero false positives on
  454 real English records is measured; the 20/20 on positives is circular,
  since the probe set and the lexicon share an author. Hindi/Urdu only —
  romanized Tamil, Telugu and Malayalam score zero by construction.
- **Hindi is ingested but not scored.** `modeling/config.py` still sets
  `languages = ("en",)`, so the scored tables cover a subset of the corpus and
  Hindi-vs-English volume comparisons are invalid until a multilingual
  checkpoint lands.
- **No WhatsApp, Telegram, X, ShareChat, Instagram or Facebook.** For an Indian
  case this is the largest gap, not a minor one. The corpus is public web
  discourse *about* Indian misinformation, not the misinformation itself.

## [0.4.0] — 2026-08-24

Phase 4: the corpus and the models now sit behind a persistent, queryable, async
backend. Five services under one `docker compose up`, 27 tables, 80 documented
API operations, and a fusion score that returns its own explanation.

### Added

- **The API contract, tagged `v0.1-contract`.** 69 paths / 80 operations, every
  one returning a schema-valid response under `DEMO_MODE=1` with no database and
  no Redis. Phase 5 was unblocked from that tag; everything after it changes
  values, not shapes. `openapi.json` is committed and `make openapi-check` fails
  the build if the code drifts from it.
- **Postgres 16 + pgvector schema**, 27 tables across the five pillars, in three
  reviewed Alembic migrations. `alembic downgrade base && alembic upgrade head`
  round-trips cleanly and a test asserts the models have not drifted from the
  migrations.
- **Parquet → Postgres ETL** (`app/etl/`). COPY into an unlogged staging table
  then one `INSERT ... SELECT ... ON CONFLICT`, because COPY cannot do
  ON CONFLICT and ORM inserts over this corpus would take hours. 4,188 posts,
  2,021 authors and 374 domains load in seconds; rerunning inserts zero rows.
- **Phase 2/3 scored-Parquet loader.** Those tables are a first-class input, not
  a fallback: the API serves real model output with no GPU and no checkpoint
  mounted. 4,188 post scores, 21 narratives, 1,959 coordination edges, 100 media
  verdicts and 2,474 embeddings.
- **Fusion scoring** (`app/scoring/`, `configs/fusion.yaml`). A plain documented
  function, not a model, that returns the score alongside its three components,
  their sub-signals, the applied weights and the config version.
- **Celery on four queues** split by what the work contends for, so a
  forty-minute deepfake job cannot starve fifteen-minute alert evaluation. One
  unified `jobs` table, so the frontend has one polling shape for every async
  operation.
- **API-key auth** with three scopes, a server-side pepper, prefix-narrowed
  constant-time verification and revocation tombstones.
- `docs/api.md`, `docs/data-model.md`, `docs/scoring.md` — the last with a worked
  example verified against the live API.

### Changed

- **API keys are hashed with HMAC-SHA256, not Argon2id.** Argon2 cost 81ms of CPU
  on every single request, measured. Memory-hard KDFs exist to make brute-forcing
  low-entropy human passwords expensive; an API key here is 32 bytes from
  `secrets.token_urlsafe`, so brute force is infeasible whatever the hash costs.
  What this actually needs is no plaintext at rest, a constant-time compare and a
  pepper — HMAC gives all three in 0.003ms. The stored string names its own
  algorithm, so keys minted under Argon2 keep verifying.
- **`scoring_version` is a digest, not a joined version string.** Phase 2's
  per-module map joins to 85 characters and would overflow again on the next
  module. The readable map stays in `model_versions`.
- Fusion weights live in `configs/fusion.yaml`, not `configs/scoring.yaml`: that
  name is already Phase 2's batch-stage config and overwriting it would break
  `modeling score --all`. Recorded in `docs/scoring.md`.

### Fixed

- **Alembic silently committed nothing.** Probing the connection before
  `context.begin_transaction()` opens an implicit transaction, so Alembic's own
  transaction degraded to a nested no-op and `close()` rolled the whole upgrade
  back. Every migration logged "Running upgrade", exited 0, and created no
  tables. One `rollback()` before `configure` fixes it.
- **Engine `connect_args={"options": …}` silently replaced any options in the
  URL** rather than merging, so anything scoping a connection through the URL was
  dropped and the connection quietly used the wrong schema. It now concatenates.
- **Net sentiment had opposite signs in two places on the same screen.**
  `/overview/kpis` computed negative-minus-positive and
  `/overview/kpis/sentiment` positive-minus-negative, so one corpus read +27.8
  and -27.8 depending on where you looked. Both are now positive-minus-negative,
  the definition text says so, and a test pins them together.
- **The async Redis client was cached across event loops.** An asyncio
  connection pool binds its connections to the loop that created them, so one
  global cache handing the same pool to a second loop raises
  `RuntimeError: Event loop is closed`. Under uvicorn there is one loop and it
  never appears; anywhere else /readyz flapped between `ok` and `down` on
  alternate probes, which would restart a healthy container. The cache is now
  keyed by loop.
- **`/readyz` re-read Parquet footers on every probe**, making a readiness check
  a 550ms call — slow enough to trip an orchestrator timeout. Capability probes
  now carry a 30-second TTL; checkpoints do not appear and disappear second to
  second.
- **The worker imported one model module in isolation** and got
  `NoReferencedTableError` on a foreign key — an import-order bug that reads like
  a schema bug. `import_all_models()` now runs once per entrypoint.

### Discovered in the data

Three column widths were wrong, and only loading the real corpus found them:

- `posts.id` reaches 219 characters; news and GDELT derive `native_id` from the
  article URL.
- `posts.author_handle` reaches 744; on news rows Phase 1 stores the article
  *byline* there. Now `TEXT` — there is no defensible number to pick.
- `authors.handle` mirrors it, and overflowed only when the newest post for an
  outlet happened to have a long byline, which made it non-deterministic.

Two artifact shapes also differed from the spec notes:

- **Phase 2's embedding cache is content-addressed** — `sha256(text)[:32] → row
  index` — not an index of post ids. 4,126 unique texts cover 4,188 posts because
  exact reposts share a vector, which is correct and which a positional zip would
  have silently mangled.
- **`Record` does not reject empty text or a null timestamp.** Phase 1's adapters
  drop those before a `Record` is built; at the Parquet boundary there is no
  adapter, so the loader applies the same rules with the same reason codes.

### Known gaps

- The inference tasks (`nlp.*`, `score.posts`, `score.authors`, `graph.*`,
  `compass.*`, `media.deepfake`, `domain.enrich`, `alerts.evaluate`,
  `reports.render`) are registered and routed but not implemented; they record a
  failure on the job row rather than hanging, and their routes return `501` with
  `code: not_implemented` outside demo mode. The data they would produce is
  already served from Phase 2/3's committed outputs — what is missing is scoring
  *new* posts without rerunning the Phase 2 pipeline.
- 1,714 of 4,188 posts have no cached embedding, because Phase 2 hashes the
  prepared text and its truncation policy needs the model's tokenizer to
  reproduce. Guessing at it would map posts to the wrong vectors.
- `docker compose` could not be exercised on this machine (the v2 plugin is not
  installed). Every service was verified against containers started directly:
  pgvector/pgvector:pg16, redis:7-alpine, a live Celery worker and uvicorn.

## [0.2.1] — 2026-08-15

Real benchmark data arrived. The bot classifier is trained on it, a better
stance corpus replaced the planned one, and three defects surfaced that only
real data could expose.

### Added

- **FNC-1 stance loader** (`modeling/datasets/fnc1.py`). The data supplied as
  "SemEval-2016" turned out to be the Fake News Challenge corpus, which is the
  better fit: its four labels map one-to-one onto the contract's
  support/deny/discuss/unrelated, where SemEval has no `unrelated` class at all
  and could never produce one. 75,385 pairs over 2,587 article bodies, versus
  SemEval's ~4k. `train_stance_classifier` now prefers it, falling back to
  SemEval — the same pattern the bot trainer uses for Cresci/TwiBot.
- Both stance corpora share `benchmarks/stance/`; each loader reads only what it
  recognises, and a test asserts they do not collide.

### Changed

- **Bot classifier trained on real Cresci-2017** (14,368 accounts, 8 bot
  campaigns). Pooled out-of-fold macro-F1 0.705 [0.696, 0.713], PR-AUC 0.908.
  Beats both baselines. Calibration improved Brier 0.4345 -> 0.1402; operating
  point 0.868 precision at 0.825 recall, meeting the 0.85 target.
- **Cresci grouping is now hybrid** — campaign for bots, account for humans.
  Campaign-only grouping is impossible on this dataset because the label is a
  deterministic function of the campaign, so every group is single-class and no
  fold can be stratified. Account-only grouping leaks the botnet template. Each
  test fold now holds out one or two entire campaigns.

### Fixed

- **A single-class training fold crashed the process with SIGSEGV and no
  message** — exit 139, empty output, nothing written. Two separate causes, both
  fixed: the hybrid grouping above, and an import-order constraint.
- **`xgboost` must be imported before `torch`.** On macOS both ship their own
  OpenMP runtime; torch-then-xgboost segfaults on the first `fit()`, while
  xgboost-then-torch is fine with full threading on both. Claimed at the package
  root in `modeling/__init__.py`. `OMP_NUM_THREADS=1` also avoids it but
  serialises transformer training, and `KMP_DUPLICATE_LIB_OK=TRUE` is documented
  by Intel as able to produce wrong results.
- **`modeling train` now reports failures instead of exiting silently.** Any
  exception is caught, named, and given a non-zero exit; a skipped run also
  exits non-zero. Silence is indistinguishable from success in a terminal.

### Known gaps

- **The misinformation classifier is still on its demo checkpoint.** Real
  training (26,777 rows, roberta-base) was started and stopped for machine
  load; it needs a GPU run. Every number in its model card remains marked
  DEMO FIXTURE until then.
- **Bot generalisation to unseen campaigns is weak, and the report says so.**
  Per-fold macro-F1 is 0.416 ± 0.187 with a worst fold of 0.249, against the
  pooled 0.705. The pooled figure averages away the folds where a held-out
  botnet looked nothing like the training ones; the per-fold mean is the number
  to quote.
- **Human recall is 0.608** at the chosen threshold — 39% of genuine accounts
  are flagged. That is the number to design the UI around.
- Cresci's `genuine_accounts/users.csv` has a different column order, so
  `created_at` fails to parse for exactly the human class — a textbook label
  proxy. It did not dominate (`account_age_is_missing` reaches the top-5 SHAP
  set for 4% of accounts, against `post_count` at 99%), but it is worth
  re-checking after any loader change.

## [0.2.0] — 2026-08-13

Phase 2: the modeling and scoring layer. Turns the Phase 1 corpus into scored Parquet
tables, plus the evaluation evidence that says what those scores are worth. No API, no
dashboard, and deliberately no fused risk score — the weighting is a Phase 4 product
decision, not a model output.

`ingest/` was not modified.

### Added

**Foundation** (`modeling/config.py`, `registry.py`, `io.py`)
- One seed for `random`, `numpy`, `torch` and every estimator; `run_fingerprint()` stamps
  seed, device, library versions and the input corpus manifest hash into every eval
  artifact.
- `registry.py` resolves checkpoints by name+version from a local cache, a private HF Hub
  repo or a Drive mount. `models/` is gitignored — the repo commits the pointer, never
  the blob.
- The scored output contract as explicit Arrow schemas, with joinability, idempotency and
  resumability enforced on write rather than asserted in a notebook.

**Group-aware splitting** (`modeling/datasets/splits.py`)
- The only splitter in the codebase. `tests/test_splits.py` proves the leakage detector
  fires on a post-level split, on a frame-level split, and on the second-order case where
  two rows carry different claim ids and the same sentence — then scans the source tree
  and fails if any module outside `splits.py` imports a scikit-learn splitter.
- Dedupe happens before splitting, because near-duplicates straddling the boundary leak
  even when the group keys differ.

**Benchmark loaders** (`modeling/datasets/`)
- Eight loaders (LIAR, FakeNewsNet, CoAID, SemEval stance, TwiBot-22, Cresci-2017,
  FaceForensics++, DFDC). None downloads: every benchmark is access-gated, so an absent
  dataset raises with the exact manual steps instead of returning an empty frame that
  would train a model on nothing.
- Each declares the group key that makes an honest split possible — LIAR by speaker,
  Cresci by *campaign* rather than account, FF++ and DFDC by source video with untied
  fakes dropped.
- Committed fixtures reproducing every real format, so parsing, label mapping and grouping
  run offline. Regenerate with `python scripts/make_fixtures.py`.

**Auxiliary scorers** (`modeling/aux/`)
- Toxicity (`unitary/toxic-bert`), sentiment (`cardiffnlp/twitter-roberta-base-sentiment-latest`),
  emotion (`j-hartmann/emotion-english-distilroberta-base`) and an IsolationForest
  behavioural anomaly rank. Batched, CPU-capable, cached by text hash, language-gated.
- Scored the 4190-record corpus in 9m37s on CPU: 96% coverage for the three transformer
  scorers, 54% for anomaly, every gap carrying a reason code.

**Text and narrative** (`modeling/text/`)
- Cached embeddings with the dimension read from the model rather than hardcoded, and a
  recorded truncation policy that keeps an article's lede whole.
- HDBSCAN clustering with near-duplicate collapse, so one repost swarm cannot become a
  narrative, and cross-run `narrative_id` carry-forward by centroid match with splits,
  merges and deaths logged.
- Misinformation classifier: fine-tune, calibrate on validation, report per-benchmark and
  cross-domain breakdowns.
- LLM narrative summarization, bounded to one call per cluster, cached, with a proven
  centroid fallback when no API key is present.

**Accounts and coordination** (`modeling/accounts/`)
- Feature tiers (universal / social-graph / threading) with an enforced intersection, so a
  model is never trained on features the target corpus cannot compute.
- Bot classifier with campaign-grouped CV, out-of-fold calibration, a precision-targeted
  operating point and per-account SHAP into the contract.
- Coordination: an evidence-typed co-behaviour graph with LSH bucketing, Louvain
  communities, and a within-author time-shuffled null model.

**Evaluation** (`modeling/eval/`)
- Metrics with bootstrap CIs (accuracy is banned from the report; PR-AUC leads),
  isotonic/Platt calibration with a documented fallback below 200 validation rows,
  baselines that run *before* the main model so the bar is fixed first, a counted error
  taxonomy, and report writers that regenerate from saved predictions without retraining.
- Module ablation with a provisional fusion labelled, in three places, as
  for-measurement-only.

**Interface**
- `modeling/cli.py`: `score`, `train`, `evaluate`, `report`, `cluster`, `ablate`,
  `datasets`, `registry`, `stats`, `warm-cache`, `sample-for-labelling`.
- Notebooks 02–05, generated from `notebooks/build_phase2_notebooks.py` so the diffs stay
  reviewable. Notebook 05 is the consolidated evaluation report.
- Model cards for all seven modules, error-analysis scaffolds, and a Phase 2 README
  section with the limitations stated plainly.

### Fixed

Six defects found by running the code against the real corpus rather than the plan:

- **Velocity divided microsecond timestamps by 1e9.** pandas 3 stores `datetime64` as
  microseconds, so every timeline compressed 1000×, one "hour" swallowed six weeks, and
  every narrative reported its entire size as its peak-hour velocity — a wrong number that
  looked perfectly plausible.
- **YAML 1.1 parsed the LIAR label map's bare `false:` as a boolean**, so it stopped
  matching the string label and silently dropped every `false` row. The map is now quoted
  and the loader refuses to run if it does not cover all six labels.
- **Severity as a 75th percentile scored 0.02 on a narrative that is a quarter alarming.**
  Replaced with an engagement-weighted mean of the top quartile, and the reasoning for
  rejecting the mean, the percentile and the maximum is written out.
- **pandas NA sentinels reached a string Arrow field and the language gate.** `str(NaN)`
  is the three-character string `"nan"`, which sailed through a length check and got
  scored as content.
- **Parquet list columns arrive as numpy arrays**, so `value or []` raised rather than
  defaulting. Every read now goes through `modeling.io.as_list`.
- **`anomaly_score` was fitted on the resumed subset.** It is a within-corpus percentile,
  so a record's score depended on how the previous run happened to die.

### Known gaps

Stated rather than hidden; each has a model card explaining what is missing.

- **Bot, stance and deepfake are not trained.** Every benchmark they need is access-gated.
  Each ships its complete training path, its split discipline and an honest null scoring
  path.
- **The misinformation fine-tune does not clear TF-IDF + logistic regression** on the demo
  fixture (macro-F1 0.908 vs 0.927, intervals overlapping). Reported, not tuned away.
- **Coordination modularity does not exceed the time-shuffled null on this corpus**
  (0.912 vs 0.979 ± 0.000). The mechanism works — it recovers a planted burst on the
  fixture — but this corpus does not contain the phenomenon at a detectable level.
- **The benchmark-to-corpus transfer gap is unmeasured.** Closing it needs a person to
  hand-label 100 corpus records via `modeling sample-for-labelling misinfo`.
- **Multi-label author cohorts** have a schema (`modeling.io.AUTHOR_COHORTS`) and are
  deliberately unpopulated.

### Notes

- `sklearn.cluster.HDBSCAN` is used rather than the standalone `hdbscan` package, and
  `networkx.community.louvain_communities` rather than `python-louvain` — same algorithms,
  two fewer build-fragile dependencies.
- Phase 1 observations found while consuming the corpus, not fixed here because `ingest/`
  is read-only to Phase 2: two duplicate `record_id`s across `date=` partitions in the
  news source, and no author roll-ups for GDELT/news (their `author_id` is an outlet
  domain). Both are handled defensively on the Phase 2 side with a logged warning.

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
