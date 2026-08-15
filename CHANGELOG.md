# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
