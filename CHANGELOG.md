# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The corpus is retargeted from the United States to India, in English and Hindi.
Data layer only: the case, the sources and the language detection. The modeling
layer still scores English only, which is now a documented gap rather than a
decision.

### Fixed

- **The claim dedup precision counts only pairs at or above the threshold.** The hand-check
  sample spans similarities below `claim_dedup.min_similarity`, because a row can join a
  cluster through a chain of pairs. `embedding_precision` counted all labelled pairs, so the
  number was too low for the threshold that the pass uses. It now reads the threshold of the
  run from `claim_links.json` (the config is the fallback, and the line says so). It counts
  the labelled pairs below the threshold apart, gives the share of same-claim pairs below it,
  and gives precision per similarity band with Wilson intervals. A pair below the threshold
  is in the links and is not stale.

### Added

- **The claim dedup hand-check sample records who labelled it.** The sample has the new
  columns `labeller_kind`, `labeller_1`, `labeller_2`, `label_1` and `label_2`.
  `embedding_precision` returns a Wilson interval, Cohen's kappa, raw agreement and the
  labeller kind. Only `human` labels give status `measured`. Other kinds and mixed kinds
  give `provisional` with the reason code `labeller_not_human`, and the CLI and the
  comparison report print `labeller: model, unreviewed by a person` for `model-unreviewed`.
  A filled row with no valid kind, an answer that is not yes or no, or one name for both
  labellers is refused. An older sample with only `is_same_claim` reads as kind `unknown`
  with the reason code `labeller_kind_missing`. `dedup-claims` no longer overwrites a sample
  that has a filled label. It writes the new blank sample to `claim_dedup_sample.new.csv`,
  warns with the reason code `sample_has_labels` and reads the precision from the kept file.
  The option `--overwrite-sample` replaces a labelled sample on purpose
  and deletes an older `claim_dedup_sample.new.csv`. The command refuses an invalid sample
  before it writes any file. The status is `measured` only for human labels with two names,
  one double-labelled pair and pairs that are in the current links. Stale pairs, a half-filled
  row, mixed kinds and a refused sample show on the rendered line. The Wilson interval treats
  pairs as independent, and the sample is systematic over the similarity range.

- **Per-domain count of the corpus rows that the fact-check comparison excludes.** The report
  JSON has `excluded_fact_check_publisher_by_domain` for `corpus_title` and `corpus_text` of
  each model. The Markdown report has a table with one row for each fact-checker domain, also
  a domain with 0 rows. The counts use the same domain function as the exclusion. The code
  fails if they do not add up to `n_excluded_fact_check_publisher`. A corpus row on a
  subdomain of a fact-checker, such as `hindi.boomlive.in`, stays in the flag rate. The report
  counts it under `fact_check_subdomain_not_excluded` and shows the count under the table.
- **`relabel-eval` reports what a sample with one class can still measure.** With only
  negatives, it keeps the refusal of ROC-AUC, precision and recall (reason code
  `one_class_only`, exit code 1). It now also reports the observed positive rate with a
  Wilson interval, the flag rate at the decision threshold (unweighted, with Wilson and
  bootstrap intervals, and reweighted), the flag rate per stratum, the mean predicted
  probability and the ECE. The flag rate on rows that report on a false claim against the
  other rows now carries Wilson intervals and a bootstrap interval of the difference, also
  when both classes exist. A null Laya threshold prints "no threshold configured".
- **Fact-check tier 4, `title_predicate`, for Alt News and Factly.** The verdict is read
  from the headline with the English and Hindi regular expressions and guards in
  `configs/factcheck.yaml`. A guard cancels a negated verdict, a headline that calls the
  media genuine, a quoted verdict word and a question. Rows carry the tier
  `title_predicate`. A headline with no match is counted as `title_predicate_no_match`, and a
  cancelled one as `title_predicate_guarded`. Tier 4 rows stay out of the primary evaluation,
  the frozen manifest and the Hindi publisher shares while
  `title_predicate.include_in_primary_evaluation` is false, which is the default. The
  comparison report shows them in a separate section and says that their claim text is the
  debunk headline. Claim deduplication keeps tier 4 rows apart from the other tiers, and
  with the switch on a tier 4 row collapses into the stated verdict. `altnews.in` and
  `factly.in` move from `other_fact_checkers` to `publishers`, and the eight-domain skip
  list is the same. Two commands measure the reader: `ingest.cli title-predicate-sample`
  draws 100 headlines (seed 20260813, stratified by language and by match) into a blind
  sheet and a key file outside git, and `ingest.cli title-predicate-score` writes weighted
  precision and recall with Wilson intervals, per language and per publisher, against the
  adjudicated label, against each labeller, and in a worst case. A result is a measured
  error rate only with `human` labels, adjudication on every row, a kappa of at least 0.6,
  30 rows per language and an unchanged pattern set. The error rate is not measured yet.
  See `docs/factcheck-tiers.md`.
- **Per-source coordination report.** `CoordinationDetector.detect` now returns observed
  modularity against the time-shuffled null for all eligible records and for each source.
  Each row has the null mean, a 95 percent interval, the empirical p value, its floor, the
  number of shuffles, the empty shuffles and the null mean edge count. `per_source=True`
  turns it on, and only `coordination-report` does that. A source below `min_source_authors` or `min_source_edges` gets a reason code
  (`too_few_authors`, `no_edges`, `too_few_edges`) and stays in the report. The seed is the settings seed
  (20260813). The new command `modeling.cli coordination-report` writes `report.md` and
  `report.json`. It takes `--sources`, which may name a source outside the production
  eligible set such as `telegram`, and `--null-shuffles` (default 200). It drops forwarded
  Telegram posts as `telegram_forward` and never writes to the scored tables.
- **Language purity in `compare-embedder`.** The output has the majority-language share of
  each cluster, the member-weighted mean and the count of mixed clusters for both embedders.
  It also gives each language's clustered and noise counts, and the share of its clustered
  records that sit in a cluster with each other language. Both embedders cluster the same
  sources and limit. The new option `--min-cluster-size 5,10,15,25` runs both embedders at
  each size. The narratives manifest records `min_cluster_size`.
- **Checkpoint backup and hash check.** `modeling.registry.resolve` pulls a checkpoint
  from the private repo in `HF_REPO` and checks its sha256 against the optional `sha256`
  map in `configs/models.yaml`, for local, Hub and Drive copies. A mismatch raises
  `checkpoint_sha256_mismatch` and deletes a pulled copy. The Laya stage turns it into
  the reason `laya:checkpoint_sha256_mismatch`. A pulled copy with no hash is refused and
  deleted with `checkpoint_unverified_pull`, unless `ALLOW_UNVERIFIED_PULL=1`. A local
  copy with no hash is used and counts `checkpoint_unverified`. A map with no value for
  the version counts `checkpoint_sha256_missing_key`. `HF_REVISION` or `hf_revision`
  pins the repo revision. The pull covers the Laya calibrator files and skips
  `checkpoint_latest/` at any depth. Other new reason codes: `checkpoint_broken_symlink`,
  `hub_token_missing`, `hub_not_installed`, `hub_pull_failed`, `hub_path_absent`.
  New commands: `python -m modeling.cli checkpoint-hash <module> [--version <folder>]`
  and `python -m modeling.cli checkpoint-pull <module> [--version <folder>]`. The model
  cards and `docs/laya-finetune.md` no longer say that a new clone must train again.
- **The checkpoint hash definition changed.** The hash now frames each path with a length
  prefix, follows symbolic links, skips `checkpoint_latest/` at any depth and `*.tmp`
  files, and reads `tokenizer/tokenizer_config.json` and `registry.json` in a normalized
  form. Every value that the old function produced is replaced, including the value
  `be9cc846...` in the misinfo model card. So the `checkpoint_stamp` suffix in
  `model_versions` changes, and the corpus is scored once again.
- **Weighted hand-labelling sample and two-labeller workflow (issues 30 and 31).**
  `modeling sample-for-labelling misinfo --lang en` and `--lang hi` now draw a stratified
  sample. The strata are posts that match a fact-check claim, one stratum for each of
  Reddit, Mastodon, YouTube and Telegram, and all other records. The shares are in
  `configs/labelling.yaml`. A match is a simhash within 3 bits of a false claim. An
  embedding match is also possible when `--embeddings` gives a Parquet file of record
  embeddings. Without it, the draw counts `embedding_unavailable` and says that only
  simhash matches count. The eight fact-checker domains from `configs/factcheck.yaml` never
  enter the sample (reason `fact_check_publisher`), and a Hindi row with no title is left
  out. A Hindi draw applies the skip logic of the `laya` stage to the title (reasons
  `no_title`, `text_too_short`, `unsupported_language`). `--lang` is required. The sheet rows
  are shuffled with the seed (default 20260813) and hold no stratum and no weight. Those go
  to `<sheet>.design.csv`, and `<sheet>.sampling.json` records the draw. Keep both files
  away from labellers. The command refuses to replace an existing sheet unless `--force`
  is set, and it refuses an `--n` that is too small to give each non-empty stratum a row.
  It prints how many records match a claim and warns when the count is 0. The matching
  thresholds are `claim_match` in `configs/labelling.yaml`, separate from the claim dedup.
  An embeddings file that names another model is refused (`embedding_model_mismatch`).
- **Label sheet columns and agreement report.** The sheet has `label_a`, `label_b`,
  `reports_on_false_claim_a`, `reports_on_false_claim_b`, the adjudicated columns,
  `labeller_kind` and `notes`. `modeling label-agreement <sheet>` prints Cohen's kappa with
  a bootstrap interval (seed 20260813), the raw agreement and the disagreement list.
  `--fill-agreed` fills the adjudicated columns where both labellers agree.
  `docs/labelling-guide.md` is the guide for labellers, with the rule for articles that
  report on a false claim and notes for Hindi.
- **`evaluate misinfo` reads `adjudicated_label`.** It refuses a sheet with labels from a
  model that no person reviewed (`labeller_kind` `model-unreviewed` or `llm-unreviewed`)
  unless `--allow-model-labels` is set. A labelled row with no valid `labeller_kind` is
  always refused. With the flag, every output carries `labeller: model, unreviewed by a
  person`. The transfer section also reports precision, recall, flag rate and positive
  rate reweighted to the corpus. The weight of a row is its stratum pool over the rows of
  that stratum that reach the metrics, so a dropped row does not skew the figures. The
  bootstrap resamples inside each stratum, and the Kish effective sample size appears next
  to each reweighted figure. The output lists dropped rows per stratum with their reason
  and warns when a stratum has no used row. It also reports the false-positive rate (flag
  rate among rows labelled 0) for rows that report on a false claim against the other
  rows, weighted and unweighted, with intervals (the input for issue 32). The transfer
  functions need a `ResolvedSheet` and enforce the labeller rule themselves.
- **`modeling evaluate-laya-hindi`.** It reads Laya values from `record_scores` and
  reports precision, recall, flag rate and ROC-AUC on Hindi titles, and the expected
  calibration error of the calibrated values, each with a bootstrap interval, for the
  sample and reweighted to the corpus. The ECE line states the bin count and `n`, and a
  caveat when `n` is below 200. With no `laya.decision_threshold`, the thresholded metrics
  are "not measured". The command runs no model.
- **`make relabel-eval`.** One command that reads the filled sheets and rewrites
  `artifacts/eval/labelling/<lang>/` (agreement, disagreement ids, transfer metrics). Every
  interval uses seed 20260813, and `transfer.json` records it.

- **Fact-check tier 3, `dom_block`, for Newschecker.** The rating is read from the
  visible result block and the claim from the `Claim` block, in English and Hindi, with
  the selectors in `configs/factcheck.yaml` rather than in code. A page where the verdict
  selector matches nothing is counted as `dom_block_not_found`, so a reskin shows up as a
  count. Other new reason codes: `dom_block_bad_config`, `html_parser_unavailable`. The
  rating map gains `Altered Photo/Video` and `Altered Media` (both `false`). The
  evaluation report now lists fact-check rows per extraction tier, including a tier that
  only appears after the evaluation set was frozen. Markup fixtures are in
  `fixtures/newschecker/`.
- **YouTube through yt-dlp, with transcripts.** The adapter no longer needs
  `YOUTUBE_API_KEY` and has no daily quota. The record text is title, description and
  subtitle text, so a video carries thousands of characters instead of a few hundred.
  A manual track in the video's language is preferred, then the original auto caption
  track; machine translations are never used. `raw` records `subtitle_lang` and
  `subtitle_kind` (`manual`, `auto` or `none`). The run summary counts
  `with_subtitles`, `subtitles_manual`, `subtitles_auto` and `no_subtitles`, and every
  skip has a reason code (`extract_failed`, `search_failed`, `subtitle_fetch_failed`,
  `outside_published_window`, `comments_unavailable`). Engagement mapping is unchanged:
  a field yt-dlp returns as null stays null. Tested against recorded yt-dlp output in
  `fixtures/youtube_ytdlp_info.json`.
- **Telegram adapter for public channels** (`ingest/sources/telegram.py`, Telethon behind
  a lazy import, in the `sources` extra). Views map to `views`, forwards to `shares`, the
  reaction total to `likes` and the reply count to `replies`; each is `None` when the
  channel does not expose it. Needs `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` and
  `TELEGRAM_SESSION` (a `StringSession`), all optional: without them the source is skipped
  with `missing_credentials`. Channels are listed under `telegram` in
  `configs/sources.yaml` (unverified seed list). `telegram` joins the `Source` literal, the
  API source enums and `openapi.json`.
- **`GET /keys/me`** returns the calling key's id, name, prefix and scopes
  (granted and effective, plus `can_read`/`can_write`/`can_admin`), never the
  secret or hash. It needs only `read`, so a read-only client can disable write
  actions up front. Under `DEMO_MODE` it answers as the `demo` principal
  (`read` and `write`, null prefix).
- **`GET /authors/{author_id}/cohorts` and `GET /authors/{author_id}/narratives`**,
  cursor paginated, returning `CohortOut` and `NarrativeSummary`. An unknown
  author is a 404 `author_not_found`. Both are declared ahead of the
  `:path` author detail route, which would otherwise match them.
- **Name lookups for filter chips.** `GET /cohorts/{cohort_id}` and
  `GET /author-groups/{group_id}` return `{id, name}` and a 404 (`cohort_not_found`,
  `author_group_not_found`) for an unknown id. Narratives already resolve through
  `GET /narratives/{narrative_id}`.
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
- **Read-side claim deduplication by multilingual embedding, with counts per pass.**
  `collapse_duplicate_claims` links two verdicts when their claim simhashes are within
  3 bits. A reworded claim, and the same claim debunked in English and in Hindi, sit
  outside that radius. `python -m modeling.cli dedup-claims` writes
  `data/factchecks/claim_links.json` with cluster ids, links and a collapse count for
  the simhash pass and for a second pass over the cosine similarity of
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` claim embeddings. It
  also writes a hand-check sample. `factchecks.parquet` is never modified. The
  evaluation report states how many claims each pass collapsed. The threshold is
  `claim_dedup.min_similarity` in `configs/factcheck.yaml`, provisionally 0.85 pending
  a hand check of the sample; the report records the precision once the sample has
  labels and `not_measured` until then. A missing `sentence-transformers` install is
  `sentence_transformers_not_installed` on the embedding pass, never a silent no-op.
- **Labels for the English corpus sample, with a one-class result.** An AI model labelled
  the 100-record sample from `modeling.cli sample-for-labelling misinfo --n 100`. A person
  did not check the labels (`labeller=llm-unreviewed`). Every label is 0. `misinfo` v0.2.0
  flags 42 of 100 rows at threshold 0.5, and all 42 are false positives. It flags all 20
  debunk articles. The transfer gap stays unmeasured, and `evaluate misinfo` refuses
  transfer metrics on one class.
- **BOOM Hindi and Fact Crescendo archive backfill.** BOOM Hindi lists only its newest
  day in its sitemaps, so the backfill reads the paginated `/fact-check` listing
  (`backfill.listing`) and walks about 600 articles. Fact Crescendo keeps its `rss_tag`
  daily path but opts into the ClaimReview backfill (`backfill.claimreview`), because
  ClaimReview is appearing on its recent posts. Pages without one are counted as
  `no_claimreview`. The frozen-set summary and the comparison report now print the Hindi
  claim share per publisher, so a set that still rests on one publisher is visible.
- **`coordinated_author_ratio` on the narrative scorecard and the authors KPI.** Fusion
  reads this ratio and not `bot_like`, so the API now serves the one it uses. The scorecard
  stores the value fusion read at score time (new nullable column, migration `c4d8a2f6e193`),
  and `/overview/kpis/authors` reports the same ratio over the filtered authors with the same
  threshold. Null means no author was scored by the coordination stage, never 0. Rows scored
  before the migration stay null until the next `score.fusion` run. `bot_like` is kept and
  documented as an input fusion does not read.
- **Cursor pagination on `/drivers/authors`, `/drivers/hashtags` and `/drivers/urls`.**
  `DriversResponse` gains `next_cursor`, `total` and `filters_applied`, with the same
  opaque cursor as the other list routes. Rows sort by post count and then by entity, so
  tied counts keep one order across pages.

### Fixed

- **Two `risk_score` fields declared no range.** `HighRiskPost.risk_score` and
  `AuthorScoreExplanation.risk_score` now carry `minimum: 0` and `maximum: 100` in
  `openapi.json`, like `AuthorOut.risk_score`. A client could only learn the 0 to 100
  scale from a sample response, and a dashboard once read 33.2 as 3320 percent. A
  contract test fails if any `risk_score` field loses its bounds. `GraphNode.bot_score`
  and `DomainOut.sharing_author_bot_ratio` now declare their 0 to 1 range the same way.

- **Fact-check articles had a misinformation score, and fusion counted it.** A debunk
  quotes the false claim, and `misinfo` v0.2.0 flagged all 20 debunks in the English
  sample. The `misinfo` and `laya` stages now give records from a fact-checker no value
  and the reason `fact_check_publisher`. The `misinfo` stage also clears a value it wrote
  on an earlier run when it now skips the record, and retracts its stamp.
- **The fact-check comparison left only four fact-checkers out of the flag rate.** It
  read the `publishers` block, which holds the publishers with an extraction tier. Alt
  News, Factly, Newschecker and DigitEye are in the corpus too. `other_fact_checkers` in
  `configs/factcheck.yaml` lists them, and `modeling.config.fact_check_domains` returns
  all eight for the comparison and the scoring stages. The committed report predates
  this and needs a rerun.
- **The fact-check backfill spent its URL budget on pages that it had already dropped.**
  Pages with a middle rating, no ClaimReview, or an excluded claim language wrote no
  verdict, so each run found and fetched them again. On Vishvas News about 299 of the 300
  URLs were such pages, and the older sitemaps were never reached. The backfill now keeps
  these pages in `data/factchecks/backfill_dropped.parquet`, skips them, and counts them
  as `previously_dropped`. Transient failures are not kept, so the next run retries them.
  A page with no ClaimReview that does not look like an article is counted as
  `no_claimreview_suspect` and is not kept. Each row holds a fingerprint of the rating map
  and `claim_languages`. A row from another config is `ledger_stale` and the page is
  fetched again. Only a middle rating or an excluded language depends on that fingerprint.
  A row for a 404 or 410 page, or for a robots.txt disallow, expires after
  `backfill.ledger_recheck_days_transient` days. A domain that answers 404 or 410
  `fetch.max_consecutive_gone_per_domain` times in a row is given up for the run. The
  option `--recheck-dropped` fetches the kept pages again and skips the HTML cache for
  them. A page is article-like when it has an `<article>` element or an article
  `og:type`, whatever else it holds. A failed ledger write is counted as
  `ledger_write_failed` and does not stop the run.
- **Sitemaps that held no new URL used the sitemap budget.** After a few runs, the newest
  sitemaps held only stored or dropped pages and used all of
  `backfill.max_sitemaps_per_publisher`, so the older sitemaps were never read. A child
  sitemap now uses the budget only when it holds a new URL. The new limit
  `backfill.max_sitemap_fetches_per_publisher` caps all sitemap requests. The report shows
  `sitemaps_exhausted`, and a sitemap that a bound skips is counted as unread.
- **Chart and drilldown series reported unmeasured engagement as 0.** The timeseries,
  the drivers list, the author timeline and the narrative timeline summed engagement
  with a per-row `coalesce(..., 0)`. A bucket where no post exposes likes, shares or
  replies now returns null. A bucket with one measured value returns the sum of the
  measured values. The engagement sort in the post list puts unmeasured posts last.
  The stored `narratives.engagement_total` column is now nullable. A migration
  recomputes it, and both writers use the same rule.

  These API fields can now be null. The dashboard must handle null in each field.
  - `TimeseriesPoint.value`
  - values in `TimeseriesPoint.series`
  - `AuthorTimelinePoint.engagement`
  - `TimelinePoint.engagement`
  - `NarrativeSummary.engagement_total`

  In a grouped timeseries, a missing series key means that the group has no posts
  in the bucket. A null value means that the group has posts with no measured
  engagement. Treat both as a gap. Regenerate the dashboard client.

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
- **`make seed` failed without a broker.** The graph step ran through a broker. The seed
  step now runs edges and communities in-process.
- **The loader gave 879 of 957 real posts no vector.** It matched cached embeddings only by
  text. The embedder now writes `{slug}.ids.json`, and the loader maps by record id. It
  checks a text hash and loads only the configured embedder's cache. It refuses an
  inconsistent cache, and it counts each miss by reason.
- **The embedding cache failed its own check and was discarded on every run.** It appended
  two rows when a batch repeated a text. It now appends one row per key. An existing cache
  with the extra row is rebuilt once.
- **The engagements KPI and its breakdown reported 0 when no post measured the metric.**
  They now report null.

### Changed

- **The first real load is rescored after the fact-checker skip.** The rescore ran on the
  957 records of the first load. 178 records lost a `misinfo_likelihood`. The stored fusion
  values were stale because domain risk was missing. With domain risk, narrative 1000 moves
  from 15.28 to 24.04. The rescore then moves 1001 from 36.05 to 34.22 and 1002 from 31.18
  to 30.71. Narrative 1000 does not change. All 3 stay low priority. See `docs/roadmap.md`.
- **The embedder comparison result is reported and the default stays.** The multilingual model
  merges 4216 records into one cluster at `min_cluster_size` 15. The cluster holds 3385
  English and 759 Hindi records and 72 records with other detector codes. The silhouette
  is 0.102 against 0.45. The cluster model card has the tables.
- **The coordination result is reported per source.** YouTube has an observed modularity of
  0.5334 against a null of 0.5331 (interval [0.5285, 0.5380], p 0.7512), so no signal above
  the null. News and GDELT are excluded (`no_threading`). Telegram is not measured. See
  `artifacts/model_cards/coordination.md`.
- **The Laya calibration diagnosis is recorded, and the clamp at 5 stays.** The optimum
  temperature on validation is 11.846 for the multilingual fine-tune and 11.408 for the
  English fine-tune, inside the search of 0.05 to 100. The cause is overconfident logits.
  `laya.temperature_refit` stays `false`. The isotonic calibrator lowers ECE prob on the
  stored scores from 0.159 to 0.009 and from 0.151 to 0.009 out of fold. The model card holds
  the full table. Both diagnosis reports are `valid: false`, so the card bases the decision
  on the stored scores. Issue 33.
- **The coordination null leaves out shuffles with no edges.** The coordination module
  is at `v0.2.1`. A shuffled graph with no edges was counted as a modularity of 0.0 and
  pulled the null down. It is now counted in `null_empty_shuffles` and left out. The
  threading exemption that was fixed to Mastodon in code is now the config list
  `threading_optional_sources`, with the code default `[mastodon]`. Telegram stays outside
  `eligible_sources`.
- **The fact-check manifest is refrozen at 4281 claims after the full Hindi archive
  read.** With the backfill fix of pull request 21, the Vishvas News sitemap backfill read
  the whole archive inside the 2022 window. The set grew from 1107 claims. 1106 old rows
  are unchanged. One claim is now kept under a different Vishvas News article with the
  same `claim_sha256`. The Hindi gate is measured again on 4180 Hindi claims. It still
  passes with a lift of 1.760 [1.286, 2.630]. The earlier result was 1.834
  [1.116, 3.981] on 1016 claims. 99.8 percent of the Hindi claims come from one
  publisher. On Hindi full text, the lift upper limit is still below 1 (0.476
  [0.401, 0.573]). The stage scores titles only. Ship decisions did not change.
- `configs/topics.yaml` replaced: four Indian topic areas (communal, electoral,
  health, scams and synthetic media) with English and Hindi seeds, replacing the
  US vaccine/election/climate case.
- `news_rss.fulltext.max_articles` 60 → 120, and
  `youtube.discovery.max_searches_per_run` 4 → 12 so one run covers the case
  (1,200 of 10,000 daily quota units).
- `topics.yaml`'s `languages:` key is marked declarative, because nothing reads
  it. The three real enforcement points are named instead.

### Removed

- `YOUTUBE_API_KEY`, `YOUTUBE_MAX_SEARCHES_PER_DAY`, `YOUTUBE_DAILY_QUOTA_UNITS` and the
  `google-api-python-client` dependency. `youtube.discovery.region_code` is now passed to
  yt-dlp as `geo_bypass_country`, which does not re-rank search results the way the
  Data API's `regionCode` did.
- **The deepfake module.** The code, the API, the table, the dependency tree, the
  training path and the evaluation artifacts are gone.
  - `POST /api/v1/media/check`, `GET /api/v1/media/check/{job_id}` and
    `GET /api/v1/media/checks` no longer exist. The API now has 66 paths and 77
    operations. The `media.deepfake` job kind is gone.
  - Migration `a91f3c5d7b24` deletes the `jobs` rows of kind `media.deepfake` and
    drops `media_checks`. The downgrade recreates the empty table. The deleted job
    rows do not come back. The database now has 26 tables and 5 migrations.
  - The `authenticity` pillar and its weight 0.20 stay. It now has two
    sub-signals, `domain_risk_agg` at 0.5833 and `anonymous_author_ratio` at
    0.4167. The config version is `fusion-v1.2.0`. A narrative that had no media
    checks keeps its score, apart from rounding below 0.01 points.
    The explanation payload changes for that narrative. The `missing` list of
    `components[authenticity]` no longer names `deepfake_hits`. The value of
    `weights_renormalized` is now false. The `deepfake_hits` input key is gone.
  - The `media` extra, `modeling/media/`, the DFDC and FaceForensics++ loaders,
    the `deepfake` trainer, the `media` scoring stage and the `media_scores` table
    are removed. A leftover `data/scored/media_scores` directory is ignored.
  - The `maint.purge_uploads` task, its beat schedule entry, the `uploads_dir` and
    `media_retention_hours` settings (`UPLOADS_DIR`, `MEDIA_RETENTION_HOURS`) and the
    compose `uploads` volume are removed. No deployment holds stored uploads.

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
