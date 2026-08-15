# Narrative Intelligence Platform — Phases 1–2: Ingestion, Modeling & Scoring

A reproducible, schema-normalized, multi-platform corpus for research on coordinated
misinformation narratives.

**Phase 1** is ingestion: a partitioned Parquet corpus in which every record — Reddit
comment, Mastodon toot, news article, YouTube comment — obeys one schema, plus the tooling
to rebuild it from scratch.

**Phase 2** is modeling and scoring: trained models, reproducible evaluation evidence, and
a batch pipeline that turns the corpus into scored Parquet tables. No API, no dashboard,
no fused risk score — those are Phases 4–6, and the fusion weighting is deliberately left
to Phase 4 because it is a product decision rather than a model output.

Two things are equally the Phase 2 deliverable: the models, and the evidence that they
work. A model without a group-split evaluation, a baseline comparison and an error
analysis is not done, and several modules here are honestly marked *not done* on exactly
that basis.

`ingest/schema.py` is the only interface Phase 2 depends on, and Phase 2 never writes to
`ingest/`.

---

## Quickstart

```bash
make setup
```

That creates `.venv`, installs everything, and copies `.env.example` to `.env`. **Every
credential is optional** — sources you cannot authenticate are skipped with a warning,
never a failure.

```bash
make data
```

Fetches every available source into `data/normalized/`, then:

```bash
make stats       # per-source summary table
make validate    # re-validate the corpus on disk against the schema
```

Single source, bounded:

```bash
python -m ingest.cli fetch mastodon --limit 500
```

What the pipeline can see, without printing any secret values:

```bash
python -m ingest.cli show-config
```

---

## The schema is the contract

Everything converges on one record type (`ingest/schema.py`). A downstream consumer must
never need to know which platform a record came from. Platform-specific fields live in
`raw` and nowhere else.

| field | notes |
|---|---|
| `id` | `f"{source}:{native_id}"` — namespaced, so a cross-source union cannot collide |
| `source` | `reddit \| mastodon \| news \| gdelt \| youtube` |
| `source_detail` | subreddit, instance domain, outlet domain, or channel id |
| `content_type` | `post \| comment \| article \| video \| video_comment` |
| `text` | cleaned plaintext, HTML stripped, **original surface form preserved** |
| `timestamp` | timezone-aware UTC, always |
| `parent_id` / `conversation_id` | namespaced; `None` when the source genuinely lacks threading |
| `engagement` | `{likes, shares, replies, views}` — keys always present |
| `urls` / `domains` | canonicalized links and their registrable domains |
| `simhash` | 64-bit, over word 3-grams, for Phase 2's near-duplicate work |
| `raw` | untouched source payload, stored as a JSON string |

Three rules are enforced by validation rather than convention, because each one silently
corrupts downstream analysis if it slips:

1. **Naive datetimes are rejected, never coerced.** A naive timestamp is an adapter bug.
   Stamping it UTC hides the bug until the velocity charts look wrong.
2. **`null` is not `0` in `engagement`.** `null` means the platform does not expose the
   metric; `0` means measured zero. Conflating them destroys the coordination signal.
   Compute engagement aggregates over the subset where the metric exists, and say so.
3. **No extra top-level fields.** Platform specifics go in `raw`, or the schema stops
   being a contract.

Storage is Parquet, partitioned `data/normalized/source=<source>/date=<YYYY-MM-DD>/`,
with an explicit Arrow schema. Not CSV: the schema has nested and typed fields and CSV
loses all of them.

---

## Sources: what you get, what it costs, what it cannot tell you

### Reddit — ConvoKit (primary)

**The Reddit API is not used anywhere in this project.** No PRAW, no `client_id`, no
Pushshift (shut down). Reddit data comes from static pre-collected corpora only.

- **Gives you:** threaded conversations, stable pseudonymous speaker ids, per-utterance
  score, subreddit, permalink. Threading and author identity come free, which is why this
  beats a flat dump — without threading, Phase 2's coordination graph has no edges.
- **Costs:** nothing. No key, no rate limit; just disk and a slow first download.
- **Cannot tell you:** anything about *current* Reddit. These are historical snapshots
  with a fixed end date, and deleted content was already tombstoned at collection time.
- Corpora are listed in `configs/sources.yaml`. Start with `reddit-corpus-small`; the
  `subreddit-<name>` corpora are hundreds of MB to GB, so uncomment deliberately.
- Downloads land in `~/.convokit/downloads` and are symlinked into
  `data/raw/reddit_convokit/` so the manifest can checksum them without duplicating GBs.
- **Deleted content:** a `[deleted]`/`[removed]` *body* is dropped (reason code
  `deleted_text`). A deleted *author* with surviving text is **kept** and flagged
  (`author_deleted`), with `author_id` set to the `__deleted__` sentinel — the text is
  still evidence; the author is simply unusable for coordination work.

### Reddit — Kaggle / Academic Torrents (secondary)

- **Gives you:** volume. Static CSV/JSON dumps, downloaded via the `kaggle` CLI.
- **Costs:** a Kaggle account (`~/.kaggle/kaggle.json`) for downloads only.
- **Cannot tell you:** anything structural. **Kaggle-sourced Reddit data is not usable
  for coordination-graph work** — these dumps almost never carry reply pointers, so
  `parent_id`/`conversation_id` are left `None` honestly and the record is flagged
  `no_threading_in_dataset`. Use ConvoKit (or Academic Torrents) for anything relational.
- Each dataset slug needs an **explicit column map** in `configs/sources.yaml`. One
  heuristic parser across many datasets half-works on all of them, which is worse than
  not working because it fails silently.
- **Academic Torrents:** torrent downloads are deliberately *not* automated. Download the
  dump yourself, then point the loader at it — no Kaggle credentials required:

  ```bash
  python -m ingest.cli fetch reddit_kaggle --path /path/to/downloaded/dump
  ```

  Line-delimited JSON is streamed, including zstd-compressed files, because
  `json.load()` on a 40GB file is not a strategy. Record the torrent hash and per-file
  checksums alongside the manifest entry; do not commit the data.

### Mastodon

- **Gives you:** real-time public posts, account age, follower/following counts and the
  `bot` flag — exactly the priors Phase 2's coordination classifier needs — plus
  federation structure via `acct` (`user@instance.tld`).
- **Costs:** 300 requests / 5 minutes on mastodon.social, enforced by the instance. The
  client sleeps until the reset rather than retrying; instances ban fast.
- **Cannot tell you:** anything about non-federating instances, and nothing about a
  thread's root — resolving one costs an extra API call per status, so `conversation_id`
  is `None` for replies rather than fabricated.
- **Boosts are emitted as their own records** whose `parent_id` points at the boosted
  status. Collapsing them into the original would delete the cross-instance amplification
  edge, which is the coordination signal this source exists for.
- Setup: `python -m ingest.cli mastodon-register --instance https://mastodon.social`,
  then create a read-scope token and put it in `.env`.
- Bounded live tail (safe for a demo, cannot hang a grading run):

  ```bash
  python -m ingest.cli mastodon-stream --minutes 2
  ```

- **Observed 2026-08:** mastodon.social returns nothing for the *federated public
  timeline* under a plain read token, while hashtag timelines work normally. Mastodon
  coverage is therefore hashtag-driven and topic-biased by construction. The adapter warns
  and flags `timeline_empty` rather than failing silently.

### GDELT

Two access paths, because they answer different questions.

- **DOC 2.0** (`gdeltdoc`): which outlets covered a claim, in which language, on which
  day. Driven by `configs/topics.yaml`.
- **Raw 15-minute drops**: GKG for themes, tone and named entities; `mentions` for
  propagation velocity.
- **Gives you:** publisher domain, language, timestamp, themes, tone — the baseline for
  the Domain Risk pillar and for narrative velocity over time.
- **Costs:** nothing (open data), but GDELT throttles hard — see below.
- **Cannot tell you:** what the article actually said. GDELT gives metadata, not full
  text, so `text` is the title (plus a short extract where GKG supplies one). Tone and
  themes are GDELT's inference, not ours. Coverage is biased toward the outlets GDELT
  monitors: overwhelmingly English-language and web-published.

Three things learned by running it, which contradict the documentation:

1. **`keyword` means an exact phrase.** `gdeltdoc` OR-joins a *list* into `("a" OR "b")`.
   A hand-written boolean string gets quoted as a single phrase and rejected. Topics
   therefore carry `gdelt_keywords` as a list of two-to-five word phrases.
2. **A single-element `language` list renders as `(sourcelang:English)`** — parentheses
   around a non-OR'd term — which GDELT rejects with *"Parentheses may only be used around
   OR'd statements."* One language must be passed as a bare string. Pinned by a test.
3. **`lastupdate.txt` lists files that 404.** GKG has been intermittently unavailable
   while `export`/`mentions` return 200. Downloads are checked for status, emptiness and
   the zip magic number, and a bad drop is skipped rather than failing the run.

Rate limiting: GDELT documents one query per five seconds but enforces it with a
*stateful penalty window* — once tripped, even 35-second spacing keeps getting refused for
a while. The adapter paces at one query per ten seconds with a single cool-off retry.

`mentions` rows are written as a **side Parquet artifact**, not as `Record`s: a mention
row has no text of its own, and inventing one would corrupt the corpus. Phase 2 joins them
on `GLOBALEVENTID`.

### News (RSS + optional NewsAPI)

- **Gives you:** headline, summary, publication time, outlet domain, and full text where
  the outlet publishes it openly.
- **Costs:** nothing for RSS. NewsAPI's free tier is ~100 requests/day, ~24h delayed, and
  licensed for development/non-commercial use only — which is why the pipeline never
  depends on it. Absent `NEWSAPI_KEY`, it degrades to RSS-only with a warning.
- **Cannot tell you:** anything about reach. No feed reports readership, so every
  engagement metric is `null`.
- Full-text extraction uses `trafilatura`, capped by a budget, delayed between requests,
  sent with a descriptive User-Agent, and gated on `robots.txt`. A host that fails
  extraction three times in a row is skipped for the rest of the run — paywalled outlets
  403 every article, and hammering them is rude and pointless. A failed extraction **keeps**
  the record with title+summary rather than dropping it; dropping would bias the corpus
  toward outlets with scraper-friendly markup.
- **Syndicated wire copy is kept, not deduplicated.** One AP story appears verbatim across
  dozens of outlets, and republication breadth is itself a spread signal. `simhash` is what
  lets Phase 2 collapse them when it wants to.
- Two feeds from the original list are gone and were removed: `feeds.reuters.com` no longer
  resolves, and `apnews.com/index.rss` returns zero entries.

### YouTube

- **Gives you:** view/like/comment counts (the only source here that reports views at
  all), channel identity, and threaded comments. `media_urls` is the watchable video URL —
  the hook for Phase 2's deepfake module.
- **Costs:** 10,000 quota units per UTC day, hard. The costs are wildly asymmetric:

  | call | units | use |
  |---|---|---|
  | `search.list` | 100 | discovery only, budgeted hard |
  | `videos.list` | 1 | hydration (50 ids per call) |
  | `commentThreads.list` | 1 | comments |

  Every call is charged to a ledger (`ingest/checkpoint.py`) **before** it is made, and the
  run stops cleanly when the budget is gone. Discovered video ids are checkpointed, so a
  later run can skip discovery entirely and spend its budget on the cheap calls. A verified
  run of 25 videos plus their comments produced 250 records for **105 units**.
- **Cannot tell you:** who watched. And nothing at all about videos with comments
  disabled — which correlates with exactly the political content of interest, so comment
  coverage is non-random. Counted as `comments_unavailable`.

---

## Reproducibility

Two things make the corpus reproducible, and neither is committed data:

1. **`data/manifest.json`** — every raw artifact with source URL, SHA256, byte size, row
   count and fetch time. API responses are archived to `data/raw/<source>/` before being
   parsed, so the checksum covers bytes that actually exist.
2. **One command:** `make data` (`python -m ingest.cli fetch-all`) rebuilds everything
   from scratch.

```bash
python -m ingest.cli manifest    # print the manifest as a table
```

**No raw data in git, ever.** `data/` is gitignored in full, including the manifest — it
is a build product that changes on every run, and a tracked copy would make `git status`
dirty after every fetch. Paste the manifest into a paper appendix instead.

**Resumability.** Every adapter checkpoints its cursor to `data/checkpoints/<source>.json`
after each page, written atomically. The store deduplicates on `id` against what is
already on disk. Killing a run mid-fetch and rerunning resumes rather than restarting, and
cannot double-count: a verified rerun reported 40 fetched, 0 written, 40 duplicates.

**Nothing is dropped silently.** Every dropped record is counted under a reason code
(`deleted_text`, `empty_text`, `missing_timestamp`, `missing_id`, `validation_error`,
`unsupported_type`) and every degraded-but-kept record under a flag (`author_deleted`,
`no_threading_in_dataset`, `fulltext_fetch_failed`, `title_from_url_slug`, `boost`, …).
Both are printed at the end of every run. Silent data loss is the failure mode that ruins
this kind of project: it surfaces three weeks later as an unexplainable metric.

---

## Configuration

| file | holds |
|---|---|
| `configs/sources.yaml` | which corpora, instances, feeds, dataset slugs and quotas |
| `configs/topics.yaml` | the case under study: seed keywords, GDELT phrases, hashtags, YouTube queries |
| `.env` | credentials only (see `.env.example`); never committed |

Changing the case under study — a different election, outbreak or conflict — should
require editing `configs/topics.yaml` and rerunning `make data`, and nothing else. If you
find yourself editing an adapter to add a source, the config is wrong.

---

## Testing

```bash
make test
```

`pytest` makes **no live network calls**: a fixture in `tests/conftest.py` patches
`socket.connect` to raise, so a test that tries to reach the network fails loudly instead
of passing slowly. Every adapter is tested against recorded payloads in `fixtures/`, which
is why each `fetch()` flattens client objects into plain dicts before `to_record()` sees
them.

---

## Known limitations

- **English-scoped by construction.** GDELT/NewsAPI queries, the RSS list and the seed
  hashtags are all English. Any claim from this corpus is a claim about English-language
  content.
- **Not a sample of anything.** Mastodon is hashtag- and instance-scoped, YouTube is
  discovery-query-scoped, GDELT covers only monitored outlets, and the feed list was
  hand-picked. The corpus cannot support prevalence or reach claims about any population.
- **No X/Twitter, Telegram, Facebook, WhatsApp or TikTok.** A large share of the
  phenomenon under study plausibly lives on platforms this project cannot access.
- **Reddit is historical.** No live Reddit path exists here.
- **Deleted content is absent**, so the corpus under-represents whatever moderators
  removed — plausibly correlated with the content of interest.
- **Different sources have different temporal shapes.** ConvoKit is a fixed historical
  snapshot; every other source is a rolling window anchored on the fetch date. Cross-source
  volume comparisons are meaningless; within-source trends are not.

`notebooks/01_corpus_eda.ipynb` reports all of this against the corpus you actually built,
and ends with an explicit coverage-and-bias statement. Regenerate the notebook skeleton
with `python notebooks/build_eda_notebook.py` (it is generated from a script so the diffs
stay reviewable), then run it.

---

## Phase 2 handoff

- **Read `ingest/schema.py` first.** It is the only interface Phase 2 should depend on.
- Load the corpus with `ParquetStore(...).read_all()` or straight from
  `data/normalized/` with any Parquet reader.
- `simhash` is already computed — near-duplicate clustering is cheap.
- Author roll-ups are in `data/authors/source=<source>/authors.parquet`, with account age,
  follower counts and the `bot` flag where the platform provides them.
- Benchmarks for training: `make benchmarks` fetches LIAR, CoAID and FakeNewsNet into
  `data/benchmarks/` with checksums. Note that FakeNewsNet ships ids and a crawler rather
  than article text, and CoAID's tweet content cannot be recollected in this project.
- **Coordination work is only valid on `reddit` (ConvoKit), `mastodon` and `youtube`.**
  `news` and `gdelt` have no threading, and Kaggle-sourced Reddit has none either.


---

# Phase 2 — Modeling & Scoring

## Quickstart

```bash
pip install -e ".[modeling]"          # add ".[media]" for deepfake, ".[llm]" for summaries
python -m modeling.cli warm-cache      # pre-download the auxiliary models, once
python -m modeling.cli score --all     # -> data/scored/
```

Everything runs with **no benchmarks, no API key and no GPU**. Modules that cannot run
write `null` with a reason code rather than a number:

```bash
python -m modeling.cli score --all --demo    # committed fixtures, ~12s, no network
```

| command | what it does |
|---|---|
| `score --all` | produce every scored table; resumable and idempotent |
| `train <module>` | misinfo \| bot — load, split, baseline, fit, calibrate, report |
| `evaluate <module>` | evaluate a trained checkpoint without retraining |
| `report [module]` | regenerate `artifacts/eval/**` from saved predictions |
| `cluster --audit 20` | re-cluster and print the manual coherence audit |
| `ablate` | the module-ablation table |
| `datasets` | what benchmark data is on disk, and how to get what is not |
| `registry` | local checkpoints (weights are never in git) |
| `show-config` | seed, device, corpus hash, library versions |
| `sample-for-labelling misinfo` | write the CSV for the corpus-transfer hand-label |

## What Phase 2 produces

Partitioned Parquet under `data/scored/`, plus a manifest mirroring Phase 1's pattern.

| table | grain | key columns |
|---|---|---|
| `record_scores` | one row per record | `misinfo_prob`, `stance`, `toxicity`, `sentiment`, `emotion`, `anomaly_score` |
| `narratives` | one row per narrative | `label`, `size`, `velocity`, `severity`, `coherence`, `centroid` |
| `narrative_membership` | record × narrative | `membership_prob`, `is_representative` |
| `author_scores` | one row per author | `bot_prob`, `bot_top_features`, `coordination_score`, `community_id` |
| `coordination_edges` | account pair × evidence | `weight`, `evidence`, `observations`, window |
| `media_scores` | record × media url | `deepfake_prob`, `face_detected`, `explanation` |

Four contract rules, enforced in `modeling/io.py` rather than by convention:

1. **Every table joins back to Phase 1.** Zero orphan `record_id` / `author_id`, asserted
   on write. This assertion has already caught one real gap (see *What the corpus told
   us*).
2. **`null` means "not assessed", never zero.** Every null carries a reason code. A
   fabricated `0.0` is indistinguishable from a confident negative once it reaches a
   dashboard.
3. **Every row carries `model_versions`** for the scorers that actually produced a value
   for *that row*, so a stale score is detectable after a retrain.
4. **Scoring is idempotent and resumable.** Rerunning over unchanged inputs with unchanged
   model versions writes zero rows — verified in `tests/test_scoring_io.py`.

## The one rule everything rests on

**Every split is group-aware, and `modeling/datasets/splits.py` is the only splitter.**

A random post-level split puts the same story in train and test, every metric goes up, and
nothing looks wrong. So the unit of a split is never a post:

| module | group key | why |
|---|---|---|
| misinfo | speaker, claim id, outlet | one politician's statements share phrasing and history |
| stance | claim/target | SemEval's unseen-target structure is the entire benchmark |
| bot | **campaign**, not account | each Cresci spambot directory is one botnet running one template |
| deepfake | **source video**, not frame | frames of one clip in both splits is *the* cause of fake 99% accuracy |

`tests/test_splits.py` proves the leakage detector fires on a post-level split, on a
frame-level split, and on the second-order case where two rows carry different claim ids
and the same sentence. It also **scans the source tree** and fails if any module outside
`splits.py` imports a scikit-learn splitter.

## How each score is meant to be read

| score | what it is | what it is **not** |
|---|---|---|
| `misinfo_prob` | calibrated similarity to claims fact-checkers rated false | a determination that a claim is false |
| `bot_prob` | calibrated similarity to accounts labelled automated on Twitter | a determination that an account is automated |
| `coordination_score` | a transparent 3-term formula over graph position | evidence about an individual |
| `toxicity` | an off-the-shelf classifier's output, **uncalibrated** | a judgement that an account is abusive |
| `anomaly_score` | a **within-corpus percentile rank** | a probability; it cannot be multiplied into one |
| `severity` | engagement-weighted mean of a narrative's top-quartile `misinfo_prob` | a mean, or a percentile — both fail on real narrative shape |
| `deepfake_prob` | top-k frame score for a detected face | evidence a video is authentic; `null` means "could not look" |

## Retraining a module

```bash
python -m modeling.cli datasets              # what is missing, and the exact steps
python -m modeling.cli train misinfo         # after the benchmark is on disk
python -m modeling.cli report misinfo        # regenerate charts without retraining
```

Training happens on Colab (T4); inference is CPU. Checkpoints save **every epoch**, then
`modeling/registry.py` resolves them by name+version from the local cache, a private HF Hub
repo (`HF_REPO`) or a Drive mount (`GDRIVE_DIR`). `models/` is gitignored — the repo
commits the pointer, never the blob.

## What the corpus told us that the plan did not

Trusting the data over the notes, as instructed. Each of these is a real property that
changed the code:

- **Two duplicate `record_id`s** across `date=` partitions in the news source. Phase 1
  dedupes *within* a partition, so an article whose timestamp shifts between fetches
  survives twice. `ingest/` is read-only to Phase 2, so `CorpusReader` collapses them
  loudly instead. Worth a Phase 1 fix if you want the row counts to reconcile exactly.
- **GDELT and news authors have no roll-up.** Their `author_id` is an outlet domain, and
  `data/authors/` only covers sources with real accounts. The joinability assertion caught
  115 of these; "exists in Phase 1" now means the union of the roll-up and the record
  author ids.
- **`lang` is null for 316 records** and arrives from Parquet as `NaN`, not `None`.
  `str(NaN)` is the three-character string `"nan"`, which sails through a length check and
  gets scored as if it were content.
- **Parquet list columns arrive as numpy arrays**, so `value or []` raises rather than
  defaulting. Every read of a list column now goes through `modeling.io.as_list`.
- **pandas 3 stores `datetime64` as microseconds**, not nanoseconds. The velocity metric
  divided the raw int64 by 1e9 and compressed every timeline 1000×, so every narrative
  reported its entire size as its peak-hour velocity — a wrong number that looked
  perfectly plausible.
- **YAML 1.1 parses a bare `false:` as a boolean.** The LIAR label map silently stopped
  matching the string label and dropped every `false` row with no error anywhere.

## Findings worth stating plainly

- **The bot classifier does not generalise to unseen botnets.** Trained on real
  Cresci-2017: pooled out-of-fold macro-F1 0.705, but the **mean across folds is 0.416 ±
  0.187 with a worst fold of 0.249**. The pooled figure averages away the folds where the
  held-out campaign looked nothing like the training ones. Quote the per-fold mean. Human
  recall is 0.608 — 39% of genuine accounts get flagged at the chosen threshold.
- **The misinformation fine-tune does not clear TF-IDF + logistic regression** on the demo
  fixture (macro-F1 0.908 vs 0.927, intervals overlapping). Reported in
  `artifacts/eval/misinfo/v0.1.0/report.md` rather than tuned away. **Still on the demo
  checkpoint** — real training on 26,777 rows needs a GPU. When it runs, if the transformer
  still fails to clear TF-IDF, it should not ship: it costs orders of magnitude more
  inference for no measured gain.
- **Coordination modularity does not exceed the time-shuffled null on this corpus**
  (0.912 observed vs 0.979 ± 0.000 shuffled). Any graph has communities; on this data the
  communities found are **not** evidence of coordination, and the report says so. The
  detector does recover a planted coordinated burst on the demo fixture, so the mechanism
  works — the corpus simply does not contain the phenomenon at a detectable level.
- **Stance is unblocked but untrained.** The corpus supplied as SemEval-2016 is actually
  **FNC-1**, which is the better fit: its four labels map one-to-one onto the contract,
  where SemEval has no `unrelated` class and could never predict one. 75,385 pairs over
  2,587 bodies. The loader ships; the 75k-pair fine-tune is GPU work.
- **Deepfake remains untrained.** FaceForensics++ is 17 GB behind a signed agreement, and
  the DFDC copy on disk is pre-extracted crops with no `metadata.json` — so a fake cannot
  be tied to its source video and the split cannot be made honest. Documented, not faked.

## Limitations

Everything in Phase 1's limitations still holds. Phase 2 adds:

- **Benchmark performance is not production performance.** LIAR is politicians'
  statements, FakeNewsNet is news headlines, CoAID is COVID-era health claims, and this
  corpus is Reddit comments, Mastodon toots and GDELT article metadata. **Expect a large
  drop.** The transfer gap is currently **unmeasured**; closing it needs a person to
  hand-label 100 corpus records (`sample-for-labelling misinfo`), and that single table is
  worth more than any hyperparameter sweep.
- **English-only, by decision.** Non-English text is skipped with a reason code, never
  scored by an English model. 316 records have no language tag at all and are admitted
  under a stated assumption.
- **The toxicity model carries a known demographic bias.** Jigsaw-trained classifiers
  over-flag African-American English and identity terms in non-pejorative use. `toxicity`
  must never be read as "this account is abusive" and nothing should be ranked by it alone.
- **`bot_prob` is trained on Twitter and applied to Mastodon.** Cross-platform transfer is
  unmeasured and should be assumed degraded — and given a worst fold of 0.249 on
  *in-platform* held-out campaigns, degraded from an already weak base.
- **Coordination cannot see across platforms.** Author identity does not survive between
  services, so one person coordinating from two accounts on two platforms appears as two
  unlinked nodes. It also excludes `news`, `gdelt` and Kaggle-flat Reddit entirely.
- **Narrative clusters are regions of embedding space, not claims.** Embedding similarity
  is topical, so posts taking *opposite positions* on one subject cluster together. The
  stance module that would separate them is the one that was descoped.
- **A high risk score is a prompt to look, not a conclusion.** It says a set of posts
  resembles patterns that have accompanied coordinated misinformation before.

**No output of this system is a determination that a person is a bot or that a claim is
false.** Not `bot_prob`, not `misinfo_prob`, not `coordination_score`, and not any
combination of them. They are research signals over public behaviour, every one of which
has innocent explanations, and they are reported with the intervals and domain caveats that
make that visible.

## Evidence

| artifact | what is in it |
|---|---|
| `artifacts/eval/<module>/<version>/` | `metrics.json`, `report.md`, `confusion.png`, `reliability.png`, `predictions.parquet` |
| `artifacts/model_cards/*.md` | one per module: data, label mapping, split, metrics with CIs, calibration, domain shift, intended and out-of-scope use |
| `artifacts/error_analysis/*.md` | counted failure taxonomies, plus the uncategorized examples that a human still has to read |
| `artifacts/eval/ablation/` | the module-ablation table |
| `notebooks/02`–`05` | **notebook 05 is the one to read end to end** |

Every eval artifact carries the run fingerprint — seed, device, library versions, corpus
manifest hash — so a rerun that disagrees can be diagnosed rather than argued about.

## Phase 4 handoff

- Load `data/scored/**` with any Parquet reader; `data/scored/manifest.json` reports rows,
  model versions and the input corpus hash per table.
- **Read the embedding dimension from `narratives.centroid`**, or call
  `modeling.text.embed.embedding_dim()`. It is 384 for the default model and 768 for the
  bge alternative; do not hardcode it in the pgvector column.
- **`null` means "not assessed".** Render it as such. Do not coalesce to zero anywhere.
- **The fused 0–100 risk score is yours.** `modeling/eval/ablation.py` contains a
  provisional equally-weighted fusion used *for measurement only*, and it is labelled as
  such in three places. The weighting is a documented product decision, not a model output,
  which is exactly why Phase 2 does not make it.
- `author_scores.bot_top_features` is per-account SHAP, ready for a "why is this flagged"
  panel. If an entry is suffixed `(global)`, SHAP was unavailable and the values are global
  importances — a different question, and it should not be presented as per-account.
- Multi-label author cohorts: the schema exists (`modeling.io.AUTHOR_COHORTS`) and is
  deliberately unpopulated. The 135-cohort taxonomy is a labelling project of its own.


## Troubleshooting

| symptom | cause |
|---|---|
| a source says `skipped` | its credential is absent; that is by design, check `show-config` |
| `convokit` / `Mastodon.py` import errors | run `pip install -e ".[sources]"` |
| GDELT returns `RateLimitError` for every topic | you are in its penalty window; wait several minutes |
| `fetch-all` exits 1 | a source *errored* (skips exit 0); the failing source is named in the summary |
