# Narrative Intelligence Platform — Phase 1: Data & Ingestion Layer

A reproducible, schema-normalized, multi-platform corpus for research on coordinated
misinformation narratives.

**Phase 1 scope is ingestion only.** No models, no API, no dashboard. The deliverable is a
partitioned Parquet corpus in which every record — Reddit comment, Mastodon toot, news
article, YouTube comment — obeys one schema, plus the tooling to rebuild it from scratch
and an EDA notebook that states honestly what the corpus can and cannot support.

Phases 2–6 (model training, backend API, dashboard, deployment) build on this. The only
contract they should depend on is `ingest/schema.py`.

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

## Troubleshooting

| symptom | cause |
|---|---|
| a source says `skipped` | its credential is absent; that is by design, check `show-config` |
| `convokit` / `Mastodon.py` import errors | run `pip install -e ".[sources]"` |
| GDELT returns `RateLimitError` for every topic | you are in its penalty window; wait several minutes |
| `fetch-all` exits 1 | a source *errored* (skips exit 0); the failing source is named in the summary |
