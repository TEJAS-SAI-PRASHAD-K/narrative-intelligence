# How It Works — the plain-English guide

This explains the whole system in simple language. Technical words are used only
where they are the real name of a thing, and each one is explained the first time
it appears.

If you want the same content as a designed, presentable page, see the
[Narrative Intelligence Field Guide](https://claude.ai/code/artifact/05c5949d-2c5c-4c3e-b306-f6100b78eb6c).

---

## Contents

1. [What this project does](#1-what-this-project-does)
2. [Where the data comes from](#2-where-the-data-comes-from)
3. [How the data is cleaned](#3-how-the-data-is-cleaned)
4. [Getting the data into the database](#4-getting-the-data-into-the-database)
5. [What models are used](#5-what-models-are-used)
6. [How the models are trained](#6-how-the-models-are-trained)
7. [How narratives are found](#7-how-narratives-are-found)
8. [Cohorts (grouping accounts)](#8-cohorts-grouping-accounts)
9. [Bot detection](#9-bot-detection)
10. [The network graph](#10-the-network-graph)
11. [The fusion score](#11-the-fusion-score)
12. [Compass — the AI fact-check helper](#12-compass--the-ai-fact-check-helper)
13. [The API](#13-the-api)
14. [Background jobs](#14-background-jobs)
15. [Docker and running the whole thing](#15-docker-and-running-the-whole-thing)
16. [What comes out, and how to use it](#16-what-comes-out-and-how-to-use-it)
17. [What this system cannot do](#17-what-this-system-cannot-do)

---

## 1. What this project does

The system collects public posts from five online platforms, cleans them into one
common format, runs machine learning models over them, groups similar posts into
"narratives" (stories that are spreading), draws a map of which accounts interact
with each other, and gives every narrative a risk score from 0 to 100.

The whole point is that **you can always take a score apart** and see exactly which
inputs produced it. Nothing is a black box number.

### The four phases

| Phase | What it is | Where the code lives |
|---|---|---|
| **1** | Ingestion — collect posts and save them in one format | `ingest/` |
| **2** | Modeling & scoring — train models, score the posts | `modeling/` |
| **4** | The product — database, API, graph, risk score | `app/` |
| **6** | Deployment — running it on cloud servers | `docker/`, compose |

These are also walls in the code. Phase 2 is only allowed to read one file from
Phase 1 (`ingest/schema.py`), and it never writes back into `ingest/`. Phase 2
deliberately does *not* produce the final risk score, because deciding how to weigh
the pieces is a product decision, not something a model can answer.

### The journey of one post

```
Mastodon post
   ↓  adapter fetches it
   ↓  converted into a standard "Record"
   ↓  saved as a Parquet file on disk
   ↓  loaded into a PostgreSQL database
   ↓  workers score it, cluster it, and add it to the graph
   ↓  the API serves it to whoever asks
```

**Parquet** is a file format for storing tables efficiently. **PostgreSQL** (or
Postgres) is the database. Both are explained more below.

---

## 2. Where the data comes from

Five platforms. Each one is included because it gives something the others don't —
and each one is documented with what it *can't* tell you, which is just as important.

| Source | What it gives you | What it costs | What it can't tell you |
|---|---|---|---|
| **Reddit** (ConvoKit) | Full conversation threads, stable user IDs, subreddit, scores | Free — just disk space and a slow first download | Anything about Reddit *today*. These are old snapshots. **The Reddit API is never used** — no PRAW, no Pushshift |
| **Reddit** (Kaggle dumps) | Lots of volume | A free Kaggle account | Who replied to whom. These dumps have no reply links, so they're **useless for the network graph** and are marked as such |
| **Mastodon** | Live posts, account age, follower counts, the platform's own "is this a bot" flag | 300 requests per 5 minutes, enforced by the server | Anything from servers that don't federate. Thread roots cost an extra API call, so that field is left empty rather than guessed |
| **GDELT** | Which news outlets covered a story, in what language, on what day, plus tone and themes | Free, but it throttles hard | What the article actually *said*. GDELT gives metadata only, so the text field is just the headline |
| **News** (RSS feeds) | Headlines, summaries, publication time, outlet name, full text when the outlet allows it | Free | How many people read it. No feed reports readership, so all engagement numbers are empty |
| **YouTube** | The only source that reports **view counts**. Channel identity, comment threads, video URLs | 10,000 "quota units" per day, hard limit | Who watched. And nothing about videos with comments turned off — which happens to correlate with the political content this project cares about |

### How each source is fetched

Every source has an **adapter** — a small piece of code that knows how to talk to
that platform. Each adapter only has to do two things:

- `fetch()` — get the raw data, handling paging and rate limits
- `to_record()` — convert one raw item into the standard format, or reject it with a reason

Everything else (saving files, removing duplicates, counting rejects, updating the
manifest) is shared code. That way six adapters can't drift apart and start
behaving differently.

**Every credential is optional.** If you don't have a YouTube key, the YouTube
adapter is skipped with a warning. The run doesn't fail.

### Budget control (YouTube example)

YouTube charges different amounts for different calls:

| Call | Cost | Used for |
|---|---|---|
| `search.list` | 100 units | Finding videos — expensive, so it's strictly budgeted |
| `videos.list` | 1 unit | Getting video details (50 at a time) |
| `commentThreads.list` | 1 unit | Getting comments |

Every call is charged to a ledger **before** it happens, and the run stops cleanly
when the budget runs out. A real test run collected 25 videos plus their comments —
250 records — for only 105 units.

---

## 3. How the data is cleaned

### One format for everything

No matter where a post came from, it becomes the same thing: a `Record`. Anyone
reading the data downstream should never need to know which platform it came from.
Platform-specific extras go into a `raw` field and nowhere else.

The important fields:

| Field | Rule |
|---|---|
| `id` | Always `platform:original_id`, like `reddit:t3_abc`. This means two platforms can never accidentally have the same ID |
| `timestamp` | Always UTC with a timezone attached. **A timestamp without a timezone is rejected, not fixed.** Quietly stamping it UTC would hide the bug until the timeline charts looked wrong |
| `engagement` | Likes, shares, replies, views — all four always present. **Empty is not the same as zero.** Empty means "this platform doesn't report it", zero means "we measured zero" |
| `text` | Cleaned plain text, HTML removed, but **the original wording is kept** — no lowercasing, no stemming |
| `parent_id` | Who this is replying to. Left empty when the source genuinely doesn't have it. Never invented |
| `simhash` | A 64-bit fingerprint of the text, used later to find near-identical posts |
| `urls` / `domains` | Links, with tracking junk (`utm_source`, `fbclid`, etc.) stripped off |

### Why the original wording is kept

The next step feeds this text into a language model that was trained on normal
human writing. "BREAKING!!!" and "breaking" carry different signals. Classic text
cleaning (lowercasing, removing punctuation, stemming) would destroy exactly the
information the model reads.

### What cleaning *is* done

- Normalize unicode so visually identical characters become the same character
- Remove invisible characters (zero-width spaces and similar) — these show up a lot
  in copy-paste bot networks and obfuscated spam
- Strip HTML, but turn paragraph breaks into newlines first, so `a</p><p>b` doesn't
  become `ab`
- Pull links out of the HTML `href` attribute **before** stripping tags, because
  Mastodon shows shortened link text — reading the visible text would capture
  `real/ur…` instead of the actual destination

### Nothing disappears silently

Every rejected record is counted under a reason code:

`deleted_text` · `empty_text` · `missing_timestamp` · `missing_id` ·
`validation_error` · `duplicate_id` · `unsupported_type` · `out_of_range`

Every record that was kept but is imperfect gets a flag:

`author_deleted` · `no_threading_in_dataset` · `fulltext_fetch_failed` ·
`timeline_empty` · `boost`

Both counts print at the end of every run. Silent data loss is the failure mode
that ruins this kind of project — it shows up three weeks later as a number nobody
can explain.

### A few judgment calls worth knowing

- **A deleted author with surviving text is kept.** The text is still evidence. The
  author ID is set to a `__deleted__` placeholder so it's obvious the account can't
  be used for network analysis.
- **Mastodon boosts (reshares) get their own record**, pointing at the original.
  Merging them into the original would delete the resharing link, which is the whole
  reason Mastodon is in this project.
- **Syndicated news copy is kept, not deduplicated.** One AP story appearing on
  forty sites *is* a spread signal. The simhash lets a later step collapse them if
  it wants to.

### How it's stored

Files go to `data/normalized/source=<platform>/date=<YYYY-MM-DD>/` as Parquet.

Why Parquet and not CSV: the data has nested fields (the engagement block), list
fields (URLs, hashtags) and timezone-aware timestamps. CSV loses all of that.

### Making it reproducible

Two things, and neither one is committed to git:

1. **A manifest** (`data/manifest.json`) recording every raw file with its source
   URL, checksum, size, row count and fetch time. Raw API responses are saved to
   disk *before* being parsed, so the checksum covers bytes that actually exist.
2. **One command** — `make data` — rebuilds everything from scratch.

Every adapter saves its position after each page, so if you kill a run halfway
through and restart it, it picks up where it left off and can't double-count.

---

## 4. Getting the data into the database

The Parquet files get loaded into PostgreSQL. Four rules govern this:

**1. Nothing is dropped silently.**
Every row is checked against the schema again before loading. Rejects are counted
and written to a file. The database has a constraint that refuses any run where
`records_in ≠ records_loaded + records_rejected`. The loader also checks this in
Python first, because failing there tells you *which source* and *what counts*,
while failing on the database constraint only tells you the constraint name.

**2. Running it twice changes nothing.**
Posts use "insert, or do nothing if it already exists". Author summaries use
"insert, or update", because those accumulate over time — and only the count
columns update, so a reload can't wipe out scores.

**3. Bulk loading, not row by row.**
Data is streamed with `COPY` into a temporary table, then moved across in one
statement. Loading a few million rows through an ORM one at a time takes hours;
this takes minutes.

**4. Empty stays empty.**
There is not a single `COALESCE` in the loading code, on purpose and permanently.
If Reddit doesn't report view counts, those stay `NULL`. Turning them into zeros
would silently halve "average views per post" the moment Reddit enters a filter.

### One clever detail: the simhash

The fingerprint is an unsigned 64-bit number. Postgres has no unsigned integer
type. Three options were considered:

- Store as `NUMERIC` — works, but loses the fast integer path
- Cast it naively — **wrong for half the range**, and near-duplicate detection would
  keep working *just well enough to look fine*
- Reinterpret the same 64 bits as a signed number — chosen

Comparing fingerprints uses XOR, which works bit by bit, so the sign convention
doesn't matter at all.

### Real bugs the data revealed

These are worth reading because they're the kind of thing that looks fine and isn't:

- **pandas 3 stores timestamps in microseconds, not nanoseconds.** The speed metric
  divided by the wrong number and squashed every timeline by 1000×. Every narrative
  reported its entire size as its peak-hour speed — a wrong number that looked
  completely plausible.
- **YAML reads a bare `false:` as a boolean, not the word "false".** The label map
  for one training dataset silently stopped matching and dropped a sixth of the
  rows, with no error anywhere.
- **Empty language codes come back from Parquet as `NaN`, not `None`.** Converting
  `NaN` to text gives you the three-letter string `"nan"`, which passes a
  "is this long enough?" check and gets scored as if it were real content.

---

## 5. What models are used

Eight modules. Two are trained here, several are used off the shelf, and two are
honestly marked as not done.

| What it does | Model used | Where it came from | Status |
|---|---|---|---|
| Turn text into numbers | `all-MiniLM-L6-v2` (384 numbers per post) | Pretrained, off the shelf | **In use** |
| Group posts into narratives | HDBSCAN | An algorithm, nothing to train | **In use** |
| Misinformation likelihood | `roberta-base` | Fine-tuned here on three fact-check datasets | **Demo checkpoint only** |
| Stance (does this post support or deny the claim?) | `roberta-base` | Data loader is ready — 75,385 pairs | **Not trained** |
| Bot / automation likelihood | XGBoost (with Random Forest as a comparison) | Trained here on the Cresci-2017 bot dataset | **Trained** |
| Coordination detection | Louvain community detection | An algorithm | **In use** |
| Toxicity | `unitary/toxic-bert` | Off the shelf, **not calibrated** | **In use, biased — see §17** |
| Sentiment | `twitter-roberta-base-sentiment-latest` | Off the shelf, trained on social media | **In use** |
| Emotion | `emotion-english-distilroberta-base` | Off the shelf, 7 emotions | **In use** |
| Unusual behaviour | Isolation Forest | Unsupervised, runs on behaviour not text | **In use** |
| Deepfake detection | `xception41` | Blocked — the training data is 17 GB behind a signed agreement, and the alternative copy has no metadata linking fakes to source videos | **Blocked** |
| Writing summaries | `claude-haiku-4-5` | API, pinned version, temperature 0 for repeatability | **In use** |

### Why some models are deliberately left off the shelf

Fine-tuning the toxicity, sentiment and emotion models isn't justified by any
measured problem, and every fine-tune creates a number somebody has to defend. The
stated policy is: one sensible configuration per model, tuned on the validation
set. Being rigorous about evaluation buys more than chasing the third decimal place
of an F1 score.

---

## 6. How the models are trained

Every trainable model follows the same six steps. The order is the whole point.

```
1. Load       the training data, or refuse with instructions on how to get it
2. Split      into train/test — using the one approved splitter, never anywhere else
3. Baseline   run a simple model first (TF-IDF + logistic regression), in seconds
4. Fit        train the real model
5. Calibrate  make the probabilities mean what they say
6. Report     metrics with error bars, baseline comparison, error analysis, model card
```

**Baselines run before the expensive model on purpose** — so the bar is set before
anyone has a stake in clearing it. If the real model doesn't beat the simple one,
that gets written into the report loudly. It is not treated as a reason to keep
tuning until it does.

### The most important rule: how data is split

If you split posts randomly into train and test, the same story ends up on both
sides. The model memorises it, every score goes up, and **nothing looks wrong**.

So the unit of a split is never a single post:

| Model | Split by | Why |
|---|---|---|
| Misinformation | Speaker, claim ID, outlet | One politician's statements share phrasing and history |
| Stance | Claim / target | Handling unseen targets *is* the benchmark |
| Bot | **Campaign**, not account | Each bot network runs one template — splitting by account puts siblings on both sides |
| Deepfake | **Source video**, not frame | Frames from one clip on both sides is *the* classic cause of fake 99% accuracy |

Two things happen in a fixed order: **remove near-duplicates, then split** (a
duplicate crossing the line leaks even when the group IDs differ), then **split by
group, then verify** — every split function checks for overlap and raises an error
if it finds any.

The test suite proves the leak detector fires in three different scenarios, and it
also **scans the source code and fails the build if any file outside the approved
splitter imports a scikit-learn splitter**.

### Calibration — making probabilities honest

The final risk score multiplies these outputs together. That only means anything if
`misinfo_prob = 0.7` genuinely means "about 70% of things scored 0.7 turn out to be
misinformation-like".

Raw model outputs fail this badly and in different ways, so combining them without
fixing it produces a number with no meaning at all.

The fix is **isotonic regression** fitted on held-out validation data, never on
training data (training-set probabilities are already overconfident by
construction). Below 200 validation rows it falls back to a simpler method, because
isotonic regression starts fitting noise at that size. The before-and-after quality
is always reported — and if calibration made things *worse*, that gets reported too
rather than hidden.

### Choosing the cutoff — not 0.5

In this product, wrongly flagging someone as a bot is an accusation about a real
person. That's worse than missing one. So the cutoff is picked from the
precision–recall curve at a target precision of 0.85, and **the recall it costs is
printed right next to it**.

### Where training happens

Training runs on Google Colab with a T4 GPU. Running the models (inference) happens
on CPU. Checkpoints save every epoch, and a registry resolves them by name and
version from a local cache, a private Hugging Face repo, or a Google Drive mount.

**Model weights are never in git.** The repo commits the pointer, never the file.

Every module has a `version` in `configs/models.yaml`, and that version gets written
onto every row it scores. That's how you can detect a stale score after a retrain
instead of just suspecting one.

### What evidence gets produced

- `artifacts/eval/<module>/<version>/` — metrics, a written report, confusion matrix,
  reliability chart, and the raw predictions
- `artifacts/model_cards/*.md` — one per model: what data, what split, what metrics,
  what it's for, what it's *not* for
- `artifacts/error_analysis/*.md` — counted lists of failure types, plus the examples
  nobody could categorise that a human still needs to read

Every one of these carries a fingerprint of the run — random seed, device, library
versions, corpus checksum — so if a rerun disagrees, you can diagnose it instead of
arguing about it.

### The results, reported honestly

> **The bot classifier does not generalise to bot networks it hasn't seen.**
> The pooled score looks like 0.705 macro-F1. But the **average across folds is
> 0.416 ± 0.187, with a worst fold of 0.249**. The pooled number averages away the
> folds where the held-out campaign looked nothing like the training ones.
> **Quote the per-fold average, not the pooled one.**
> Also: 39% of genuine human accounts get flagged at the chosen cutoff.

> **The misinformation model does not beat TF-IDF + logistic regression** on the
> demo data — 0.908 vs 0.927, with overlapping error bars. It is still on the demo
> checkpoint; real training on 26,777 rows needs a GPU. The stated position is that
> if it still doesn't beat the simple model after that, it shouldn't ship, because
> it costs orders of magnitude more to run for no measured gain.

---

## 7. How narratives are found

A "narrative" is a cluster of posts that are saying similar things. Here's how they
get found.

### Step 1 — turn text into numbers

Each post's text goes through MiniLM and comes out as a list of 384 numbers (an
**embedding**). Posts about similar things end up with similar numbers.

These are normalised to a standard length, which makes every "how similar are
these?" calculation downstream safe without each piece of code having to remember.

Embeddings are cached by the text's checksum, which is what makes re-clustering
cheap enough to run weekly.

The 384 is **read from the model, never hardcoded** — because the database column
width depends on it, and a hardcoded 384 that quietly disagrees with a 768-size
model is a migration failure you discover in production.

### Step 2 — collapse the copy-paste swarms first

One viral repost swarm is hundreds of nearly identical posts. Left alone, it forms
its own dense cluster and dominates every quality measurement.

So posts within a small distance of each other (by simhash) get collapsed to one
representative **for the clustering decision only**. The full member list is
restored afterwards, so the reported size and author count still reflect reality.

### Step 3 — cluster with HDBSCAN, not k-means

> **Why not k-means?** k-means makes you pick the number of clusters in advance, and
> then it assigns *every* post to one. That means it manufactures narratives out of
> background chatter.
>
> HDBSCAN figures out how many clusters there are on its own, and it has a genuine
> **"none of the above"** label — which is the honest answer for most of a corpus.

### Step 4 — keep the same IDs across runs

Narrative IDs are visible to users. The interface shows "Generated 48 days ago" and
an analyst may have renamed one. Making up fresh IDs on every run would orphan all
of that.

So on each rerun, new clusters are matched against the previous run's clusters by
similarity. If the match is above 0.85, the old ID carries over. Only genuinely new
clusters get new IDs. Splits, merges and deaths get logged rather than papered over.

### Step 5 — the numbers attached to each narrative

| Field | What it means |
|---|---|
| `coherence` | How tightly grouped the posts are |
| `velocity` | Peak posting rate — how fast it spread at its busiest |
| `severity` | **The engagement-weighted average of the worst 25% of member posts' misinformation scores.** Deliberately not a plain average and not a percentile — both hide the shape of a real narrative, where a big mild cluster and a small severe one are very different problems |
| `membership_prob` | How confidently each post belongs, so the UI can show shades of grey instead of in/out |
| `label` / `summary` | Written by an LLM over the 5 most representative posts. Always carries which model wrote it and when |

### How clustering is evaluated

There are no "correct answers" to compare against, so **there is no F1 score, and
none is invented**. What's reported instead:

- Silhouette score on a sample (a measure of cluster tightness)
- What fraction ended up as noise
- The distribution of cluster sizes
- **A hand-audit of 20 clusters, each rated coherent / mixed / junk by a person**

> ⚠️ **Important limitation:** clusters are regions of *meaning space*, not claims.
> Embedding similarity is about topic, so posts taking **opposite sides of the same
> argument cluster together**. The stance model that would separate them is the one
> that isn't trained.

---

## 8. Cohorts (grouping accounts)

A **cohort** is a model-guessed audience segment. One account can be in several at
once — "Micro Influencer" *and* "Crypto Fan" *and* "New Account". Nothing is
exclusive.

There is **no catch-all "General" bucket**. An account that matches nothing gets no
rows. A catch-all would put most of the corpus in one bar and make the whole chart
meaningless.

### Two kinds of rule

| Type | Based on | Confidence | Examples |
|---|---|---|---|
| **Metric** | Hard account facts — follower count, post count, account age, platform | 0.85–1.00 | Nano Account (<1k followers) · Micro Influencer (1k–50k) · Macro Influencer (>50k) · High Volume Poster (>20 posts) · New Account · Outlet |
| **Keyword** | How often certain words appear in the account's posts | 0.55–0.60 | Election Watchers · Crypto Fan · Health Sceptic |

Keyword confidence is **capped low on purpose**: a journalist writing about
cryptocurrency matches exactly the same words as a crypto promoter. Anything below
0.35 confidence isn't recorded at all.

### Two things this never does

**It never overwrites a person.** If an analyst assigned a cohort by hand, model
runs leave it alone. That's also why cohorts and "author groups" (analyst
watchlists) are separate tables — one is a machine's guess with a confidence score,
the other is a human's assertion, and an investigation depends on telling them apart.

**It never claims more than the rule can show.** The cohort is called "Health
Sceptic" rather than "anti-vax" because the rule matches *topic, not opinion*. The
label shouldn't assert more than the rule supports.

### Two details worth knowing

- **"Outlet"** matches news and GDELT sources, where the "author" is a publication
  rather than a person. It exists so that bot scoring on those accounts is
  *visibly skipped* rather than silently meaningless.
- **Account age is measured at the account's first post, not today.** "Registered
  three days before it started posting about the election" is a signal. "Registered
  in 2019" tells you nothing about a 2026 corpus.

---

## 9. Bot detection

### The problem this has to solve first

Follower counts exist on Mastodon and on the Twitter training data. They don't exist
on Reddit at all.

Training on 40 Twitter features and then scoring Reddit accounts with only 12 of
them present is a model applied to data it has never seen. So features are grouped
into **tiers**, and the model only trains on the tier available on *both* the
training data and the target data. Which tier was actually used gets recorded in the
model card.

| Tier | Needs | Features |
|---|---|---|
| **Universal** | Just the posts. Works everywhere | Posts per day · gap between posts (mean, spread, entropy) · hour-of-day entropy · burstiness · longest active streak · average text length · vocabulary variety · self-similarity · duplicate rate · URL/hashtag/mention rates · post count · active days |
| **Social graph** | Follower counts. Mastodon and Twitter data only | Followers · following · ratio between them · posts per follower · account age · posts per day of account life · plus "is this missing?" flags |
| **Threading** | Reply links. Reddit (ConvoKit) and YouTube | Distinct conversations · reply rate · distinct reply targets · reciprocity |

**Missing data is a feature, not something to fill in.** Every feature that could be
absent ships with an explicit "this was missing" indicator, carrying the
empty-is-not-zero rule all the way into the model.

### The model itself

**XGBoost** (400 trees, depth 5) is the main model. A **Random Forest** is trained
alongside and reported, so the choice of XGBoost is visible rather than assumed.

Cross-validation is 5-fold, grouped **by bot campaign** — not by account. Each
directory in the Cresci dataset is one bot network running one message template.
Splitting by account puts siblings from the same network on both sides and the model
scores nearly perfectly for having memorised seven signatures.

### Explaining a flag

`bot_top_features` stores the **top five reasons this specific account was scored the
way it was** (using SHAP, a technique for attributing a prediction to individual
inputs). The "why is this flagged?" panel reads it directly.

Every feature has a written explanation of what signal it's meant to capture. That's
not documentation tidiness — a feature nobody can explain becomes an accusation
nobody can defend.

If SHAP wasn't available, the entries are marked `(global)` and show overall feature
importance instead. That's a **different question** and must not be presented as
per-account.

### Recalibrating for a new platform

The calibration learned on Twitter doesn't survive the move to Mastodon. So it can
be refitted against a **weak label** — the platform's own self-declared bot flag.

Three safety rails:
1. Needs at least 100 rows, otherwise it's fitting noise
2. The weak label is **never a training input**, only a calibration target
3. If recalibration makes the quality score *worse*, it's rejected and logged rather
   than shipped

The weak label also undercounts — a bot that doesn't declare itself is a false
negative in the label — so recalibration is documented as **understating** the real
rate.

> ⚠️ **The headline number, stated properly:** 0.416 ± 0.187 average across folds,
> worst fold 0.249. The model has memorised campaign signatures more than it has
> learned what automation looks like. And it was trained on Twitter and is being
> applied to Mastodon, where transfer is unmeasured and should be assumed worse.

---

## 10. The network graph

> **Note on the word "constellation":** there is no component, config key, table or
> endpoint called *constellation* anywhere in this repository. What's described here
> is the **account interaction graph** — dots are accounts, lines are behavioural
> links — which is almost certainly the visualisation you have in mind.

### How the lines get drawn

Three types of connection, each built with one SQL query rather than by pulling
everything into Python. Loading 4,000 posts into memory to compute what one `GROUP
BY` answers is the slow, memory-hungry way to get the same number.

| Line type | Built from | Safety check |
|---|---|---|
| **Reply** | Post joined to the post it replies to | Skips self-replies — replying to yourself is a thread, not an interaction |
| **Mention** | Names in post text matched to accounts | Matched **within the same platform only** — a handle is only unique per platform, so matching across would merge two different people who share a name |
| **Co-posting similarity** | Two accounts posting near-identical text within the same time window | Uses exact fingerprint match, not "close enough" — the fuzzy version is a cross-product that no index can help with |

Every line is stamped with a **time bucket** when it's written, so the timeline
slider in the UI is an index lookup rather than a full scan.

Every write **updates instead of adding**, using a unique index. This matters more
than it sounds: *a coordination graph that looks more coordinated every time you
retry the build is a subtle bug with expensive consequences.*

### Finding communities

The **Louvain algorithm** groups accounts into communities based on who connects to
whom. It runs with a fixed random seed so results are repeatable.

Community IDs are **never used as a foreign key**. Communities are recomputed from
scratch each run and carry no identity between runs, so treating them as stable
would be a lie.

### Positioning the dots

Positions are computed by a worker using a spring layout and stored in the database.

Above a few thousand dots, running the layout in the browser drops frames until the
analyst concludes the tool is broken. Computing it once on the server turns that
into a simple fetch.

The random seed is fixed, so reloading the page shows the same shape — otherwise it
looks like the graph changed when it didn't.

If no layout has been computed yet, the fallback is a simple circle ordered by
connection count. It's not a good layout and isn't meant to be — it exists so the
browser is never handed dots with no positions.

### When the graph is too big

There's a server-side ceiling of 20,000 nodes. A caller can ask for fewer, never
more.

Connection counts are computed over the **whole** graph first, so the decision about
what to cut is made against reality rather than against an arbitrary page of it.

When the ceiling kicks in, the response says exactly what happened:

```json
"truncation": {
  "applied_max_nodes": 20000,
  "dropped_nodes": 4312,
  "dropped_edges_approx": 9077,
  "min_degree_kept": 15,
  "rule": "descending node degree"
}
```

> ⚠️ **Why this is never silent:** a graph that quietly loses its edges makes a
> coordinated cluster look **more isolated than it really is** — which for a
> coordination-detection tool is exactly the wrong direction to be wrong in.

### The legend is part of the data

Every colour and line style ships with its own definition *and its caveat*, in the
API response:

- **Co-posting similarity** — "suggestive of coordination, not proof of it:
  syndicated content and quote-tweet chains produce the same line"
- **Bot-like account** — "generalisation to unseen campaigns is weak; treat as a lead"

### The offline coordination detector

Separately from the graph you can look at, there's a deeper detector that builds
links from four kinds of evidence — and stores **which kind** on each link, so the
interface can say *why* two accounts are connected rather than just asserting that
they are:

| Evidence | Weight | Means |
|---|---|---|
| `near_dup` | 1.0 | Near-identical text within the window |
| `cotweet` | 0.8 | Same URL or domain within the window |
| `hashtag_seq` | 0.5 | Same hashtags in the same order |
| `temporal` | 0.3 | Replied to the same post within a tight window |

**Making it fast enough:** comparing every post to every other post is 20 billion
comparisons at 200,000 posts. Two tricks avoid it:

1. **Time bucketing** — only compare posts in the same 60-minute window. Each post is
   also placed in the *next* window, so a pair straddling a boundary isn't missed.
2. **Content bucketing** — inside a window, group by fingerprint prefix before
   computing any actual distances.

Plus a cap of 200,000 pairs per bucket, so one enormous viral burst can't blow up.

### The null model — the most important part

> **Any graph has communities.** Louvain will happily partition pure random noise and
> report a positive score. So "we found coordinated communities" means nothing on its
> own.
>
> The test: rerun the whole thing five times with **timestamps shuffled within each
> account**. That destroys coincidences *between* accounts while preserving each
> account's own volume and rhythm. If the real graph isn't more clustered than the
> shuffled one, there's no finding.
>
> **Result on this corpus: 0.912 real vs 0.979 shuffled.** It does not clear the bar.
> The communities found are **not** evidence of coordination, and the report says so.
>
> The detector *does* find a deliberately planted coordinated burst in the test data,
> so the machinery works. This corpus just doesn't contain the phenomenon at a
> detectable level.

---

## 11. The fusion score

This is the 0–100 number that ranks narratives. **It is a plain documented function,
not a model** — and that's the point.

```
score = 100 × ( 0.45 × how harmful  +  0.35 × how coordinated  +  0.20 × how authentic )
```

### The full breakdown

**Narrative severity (45%) — how harmful the content is**

| Sub-signal | Weight | What it is |
|---|---|---|
| `misinfo_likelihood_agg` | 0.40 | Engagement-weighted average of member posts' misinformation scores. Weighted by engagement because a false claim nobody saw isn't the same problem as one that reached 100,000 people |
| `compass_risk` | 0.25 | The fact-check verdict, converted to a number |
| `toxicity` | 0.20 | Average toxicity across scored posts |
| `negative_sentiment` | 0.15 | Share of posts classified negative |

**Coordination (35%) — how inauthentically it spread**

| Sub-signal | Weight | What it is |
|---|---|---|
| `bot_like_ratio` | 0.35 | Share of member accounts above the bot cutoff, **out of accounts that could actually be scored**. News outlets are excluded rather than counted as human |
| `co_post_similarity_density` | 0.25 | How densely the member accounts are linked by near-duplicate posting |
| `temporal_burstiness` | 0.25 | Share of volume in spikes above 3× the narrative's own average. Coordination looks like a spike |
| `cohort_concentration` | 0.15 | Share of posts from the single largest cohort. A narrative carried entirely by one segment is a different thing from one with broad reach |

**Authenticity (20%) — is the evidence what it claims to be**

| Sub-signal | Weight | What it is |
|---|---|---|
| `deepfake_hits` | 0.40 | Share of checked media flagged as manipulated |
| `domain_risk_agg` | 0.35 | Average risk of the websites linked |
| `anonymous_author_ratio` | 0.25 | Share of accounts that are deleted or have no handle. Lack of attribution isn't proof of bad faith, which is why it's a minority of one component rather than a flag on its own |

### Three rules built into the code

**1. Weights live in a config file, never in code.**
`configs/fusion.yaml` carries a version number that gets stamped on every row it
produces. It's re-read on each run, so editing a weight takes effect without a
restart.

**2. Missing inputs stay missing.**
If a component couldn't be computed, it's `null`, it's listed by name in the
response, and **the remaining weights get redistributed proportionally**. Treating
"not measured" as "measured zero" would systematically under-flag every narrative
where the deepfake check never ran.

**3. Everything normalises the same way.**
Mixing two normalisation methods produces scores that are each defensible on their
own and impossible to compare — and nobody notices until two numbers that should
agree don't.

### Why percentile ranking

Every sub-signal is converted to a 0–1 scale by **ranking it within the project**.

This is a triage tool: the question is *"which of these should I look at first?"* not
*"is this worse than something in a different investigation?"* Percentile ranking
answers the first one.

The cost is accepted and documented: a project where nothing is coordinated still
has a top 10%. And if there are fewer than 20 scored members there's no meaningful
distribution to rank against, so the raw value is used and **the response records
that it fell back**.

### Priority is derived, never assigned

`high` at 70 or above, `medium` at 40 or above. Two narratives with the same score
cannot end up with different priorities.

Thresholds have one definition system-wide: bot-like at 0.60, toxic at 0.60,
anomalous at 0.70, deepfake at 0.50, and a "burst" is 3× the narrative's own average.

### One value worth arguing about

Fact-check verdicts convert to risk like this:

| Verdict | Risk |
|---|---|
| Debunked | 1.0 |
| Unverified | 0.6 |
| **Insufficient evidence** | **0.5** |
| Partially substantiated | 0.35 |
| Substantiated | 0.1 |

"Insufficient evidence" sits in the middle deliberately, not at zero:
**failing to source a claim is not evidence that the claim is harmless.**

---

## 12. Compass — the AI fact-check helper

Compass writes a short, cited context note for a narrative's claim. The design is
built around one idea: **the AI is not trusted, the validator is.**

### The loop

```
retrieve sources ──▶ too few? ──────────────────▶ "insufficient evidence", stop
                 └─▶ generate ──▶ validate ──▶ pass ──▶ save
                                      │
                                      └──▶ fail ──▶ try once more
                                                      └──▶ fail ──▶ "insufficient evidence"
```

Two attempts, then stop. A third would just be the model talking itself into
something.

### Retrieval

The generator **never sees the open web and never answers from its own knowledge**.
It sees a numbered list of passages and is told to write only what those passages
support.

The sources are reputable news articles **already in the database**. An article from
Reuters that was already ingested is a better source than anything fetched live: it's
pinned, it's dated, and it doesn't depend on a network call at the moment of writing.

The list of acceptable outlets is a short explicit list rather than a computed
reputation score — because "our model rated this domain 0.82" is not an answer to
"where did this come from?"

### The validator — why this is safe to ship

| Rule | What it prevents |
|---|---|
| Every sentence must be covered by a citation | A note whose sentences aren't each backed by a source is an ungrounded generation wearing a bibliography |
| The claim is never stated as fact | Phrases like "this is true", "in fact,", "the truth is", "proves that", "definitively" are rejected. The note may say *"reporting indicates X"*, never *"X"* |
| Failure means saving nothing | Two failed attempts saves an **empty** note marked "insufficient evidence". An empty field is honest where a paragraph is not |

The sentence splitter is deliberately simple and biased toward splitting *too much*.
Over-splitting costs a failed check and a retry. The opposite error — merging two
sentences so one citation appears to cover both — would let an uncited claim through.

### Nothing is ever edited

Regenerating **inserts a new row** and marks the old one as superseded. An analyst
has to be able to say what the system claimed and when.

> The rules live in code, not in the prompt, because **a prompt is a request and a
> validator is a guarantee.**

---

## 13. The API

Built with **FastAPI**. It is deliberately thin: it validates, queries and queues
work. **It never computes anything.**

Anything that would take more than about half a second of CPU inside a request is
treated as a design bug and belongs in a background job. Requests over 500ms are
logged as warnings so they're impossible to miss during development.

### How you talk to it

| Thing | How it works |
|---|---|
| **Login** | An `X-API-Key` header. Reading needs the `read` permission, changing things needs `write`, managing keys needs `admin` |
| **Key storage** | Keys are hashed with a server-side secret and checked in constant time. The plain key exists exactly once — the moment it's created. Only the key's *ID* ever appears in logs |
| **Paging** | `?cursor=&limit=`, default 50, max 200. Every list response echoes back which filters were applied |
| **Slow work** | Returns `202 Accepted` with a job ID and a URL to poll |
| **Errors** | One consistent shape: `{"error": {"code", "message", "detail", "request_id"}}`. The request ID is also in a response header, so a screenshot of a failed call is enough to find the exact log line |
| **Rate limits** | A per-key budget in Redis, with the remaining amount in a response header |
| **Provenance** | Every score carries its breakdown and the scoring version. Every AI-written string carries which model wrote it and when |

### What's available

| Area | Endpoints |
|---|---|
| **Overview** | Top-level KPIs with drilldowns for posts, engagement, authors, emotions and sentiment · time series · extracted topics · high-risk posts |
| **Narratives** | Detail, timeline, posts, authors, cohorts, score breakdown, Compass note and regeneration · freshness · trigger re-clustering |
| **Accounts** | Authors with scores, timelines and posts · cohorts and their members · analyst-curated watchlists |
| **Network** | The graph · trigger a layout computation · saved side-by-side comparisons |
| **Domains** | Ranked by risk, with detail, linked narratives, and on-demand enrichment |
| **Posts** | Full-text search · find similar posts · reconstruct a thread |
| **Media** | Submit a deepfake check, see check history |
| **Operations** | Health checks · readiness with per-subsystem status · metrics · jobs · ingest history · API keys · alert rules |

### Two design choices worth noting

**Startup doesn't fail on a missing model.** A missing deepfake checkpoint must not
stop the narrative API from working. It gets marked "unavailable" in the readiness
check, and only the routes that need it return an error naming the missing file.

**Demo mode is a real feature.** Setting `DEMO_MODE=1` serves a built-in sample
dataset through the same schemas and the same validation — so you can demo the whole
API with no database, no models and no internet. And every fake job response says
`"DEMO_MODE: no work was enqueued."` rather than pretending it did something.

**Comparisons include the overlap.** When you compare two narratives, the response
includes which accounts appear in *both* — computed alongside the metrics rather than
needing a second call, because that overlap *is* the finding a comparison exists to
surface.

---

## 14. Background jobs

Slow work runs in **Celery**, a background job system, using **Redis** as the queue.
There are four queues, split by *what the work competes for*, not by feature.

| Queue | Competes for | Runs |
|---|---|---|
| `io` | Network | Fetching from platforms, looking up domain records. Many run at once |
| `cpu` | Processor cores | Scoring, graph building, alert checks, report rendering |
| `gpu` | The graphics card | Embedding, clustering, deepfake checks. Effectively one at a time |
| `llm` | The AI provider's rate limit | Writing summaries, Compass notes |

> **Why split them:** a 40-minute deepfake job must not be able to starve a
> 15-minute alert check. With one queue it can — and the first time it happens the
> alerts are simply late and nobody notices they were late.

### Settings that only make sense together

- **Late acknowledgement** means a job killed halfway through gets redelivered
  instead of lost. This is only safe *because* every job is written to be safely
  re-runnable. The two settings are a pair.
- **Prefetch of 1** keeps redelivery fast — a worker that dies must not also be
  holding a queue of jobs nobody else can see.
- **JSON-only messages** mean passing a database object or a dataframe through the
  queue is a startup error, rather than something that works until the worker and the
  API end up on different code versions.
- **Database locks** — long jobs take a lock keyed to the task and project, and
  return `{"skipped": true, "reason": ...}` if another run holds it, rather than
  racing.
- **Time limits on everything** — a job with no ceiling is a worker slot that can be
  lost forever to one stuck HTTP call.

### The schedule

| Every | What runs | Why |
|---|---|---|
| 15 min | Alert evaluation | Check alert rules against current scores |
| 1 hour | Domain enrichment | Look up registration and certificate info. Never blocks — a domain page must render regardless |
| 1 hour | Purge uploads | Uploaded media is imagery of real people. This is what makes the retention promise real rather than aspirational |
| 6 hours | Rebuild graph edges | Then automatically re-detect communities |
| 24 hours | Rescore narratives | Recompute every fusion score |
| 7 days | Re-cluster | Find narratives again, carrying IDs forward |

Scheduled jobs fan out to one job per project, so one slow project can't block the
others.

---

## 15. Docker and running the whole thing

Five services, one command.

| Service | Image | Notes |
|---|---|---|
| `db` | `pgvector/pgvector:pg16` | Postgres with vector search. Forced to UTC and a fixed sort order, so a timestamp never depends on which machine the container started on. Only reachable from localhost, so the dev database isn't on your network |
| `redis` | `redis:7-alpine` | Saves to disk — the results cache can afford to lose data, the job queue cannot |
| `api` | `Dockerfile.api` | The web server. **Runs the database migrations, and is the only service that does** |
| `worker` | `Dockerfile.worker` | Processes background jobs across all four queues |
| `beat` | `Dockerfile.worker` | Same image, just started in a different mode — runs the schedule |

### Why two images instead of one

The API image deliberately **does not install the machine learning libraries**. The
API never runs a model, so pulling in PyTorch would triple the image size and
startup time for zero benefit.

More practically: the API layer rebuilds every time you edit source code, and
rebuilding a PyTorch layer to change a URL route is a five-minute tax paid many times
a day.

Both images copy the dependency list first so editing source doesn't reinstall
everything, and both run as a non-root user.

### One startup script, four modes

`docker/entrypoint.sh` switches on its first argument — `api`, `worker`, `beat` or
`shell` — which is how the worker and beat share one image.

It pulls the host and port **out of the connection URLs** rather than requiring
separate environment variables, so there's exactly one source of truth per
datastore. It waits for each service with a **60-second limit**: a service that isn't
up after a minute is a misconfiguration, and waiting forever just hides it.

**Migrations run in the API only.** Running them from the worker too would race two
migration processes on a cold start. Database locks would survive it, but the logs
wouldn't tell you which one won. So the worker waits for the API to *start*, not to
be *healthy* — it needs the schema, not the API itself.

### Data is mounted, never baked in

The corpus and the model checkpoints are mounted from your disk, not copied into the
image. They're build products that change often, and a 4 GB image that goes stale on
the next `make data` is the wrong thing to ship. Config files mount read-only.

Everything comes through environment variables with nothing Docker-specific in the
code, which is what lets the same images run on cloud services with just a different
`.env` file and no code change.

Two secrets are **required rather than defaulted** — the database password and the
API key secret — so a missing value fails the whole stack loudly instead of silently
using something insecure.

---

## 16. What comes out, and how to use it

Three layers you can consume.

### Layer 1 — scored Parquet files

`data/scored/`, readable with any Parquet tool, with a manifest reporting row counts,
model versions and the input corpus checksum per table.

| Table | One row per | Key columns |
|---|---|---|
| `record_scores` | post | misinformation score, stance, toxicity, sentiment, emotion, anomaly rank |
| `narratives` | narrative | label, size, velocity, severity, coherence, centre point |
| `narrative_membership` | post × narrative | membership confidence, is-representative flag |
| `author_scores` | account | bot score, top reasons, coordination score, community |
| `coordination_edges` | account pair × evidence | weight, evidence type, observations, time window |
| `media_scores` | post × media file | deepfake score, was a face found, explanation |

**Four rules, checked when the file is written:**

1. **Every row links back to the original corpus.** Zero orphans, asserted on write.
   This check already caught a real gap: news and GDELT "authors" are website domains
   with no account summary, so the definition of "exists" had to be widened.
2. **Empty means "not assessed", never zero** — and every empty value carries a
   reason code.
3. **Every row records which model versions produced it**, for that specific row.
4. **Running it again changes nothing** if the inputs and model versions haven't
   changed. Verified by a test.

### Layer 2 — the database, through the API

27 tables, all scoped to a **project** (one case under investigation). Three
structural things shape how you query them:

- **Scores live in a separate table from posts.** The corpus is immutable; scores
  aren't. Rescoring touches one table and never the corpus — which is what makes
  "rerun scoring with new weights" a five-minute job instead of a migration.
- **Narratives record *which* fields a human edited.** So a re-clustering run can
  refresh an untouched summary while leaving a hand-written title alone. This is
  enforced in the database statement, not in a code path someone could forget.
- **The vector column is one fixed size per deployment.** It's literally named
  `post_embeddings_384`. Changing the embedding model is a migration, not a restart.

### Layer 3 — exports

`POST /reports` renders a **PDF**, **PowerPoint** or **CSV** from the narrative table
and records where it went. `GET /reports/{id}/download` fetches it. CSV is the path
for anyone who wants to take the numbers into their own tools.

### How to read each score

| Score | What it is | What it is **not** |
|---|---|---|
| `misinfo_prob` | Similarity to claims fact-checkers rated false | A determination that this claim is false |
| `bot_prob` | Similarity to accounts labelled automated on Twitter | A determination that this account is automated |
| `coordination_score` | A transparent 3-part formula about position in the graph | Evidence about an individual |
| `toxicity` | An off-the-shelf classifier's output, **not calibrated** | A judgement that an account is abusive |
| `anomaly_score` | A **percentile rank within this corpus** | A probability. It cannot be multiplied into one |
| `severity` | Engagement-weighted average of the worst 25% of member posts | An average, or a percentile |
| `deepfake_prob` | Score across the highest-scoring frames with a detected face | Evidence a video is real. Empty means "couldn't look" |

### Good uses

- **Triage** — rank narratives inside one investigation, then open the breakdown to
  see which component drove the score.
- **Lead generation** — find accounts appearing in two narratives at once via the
  comparison overlap, then check their explanation panel before believing anything.
- **Spread analysis** — use the time-bucketed graph to watch an amplification pattern
  form, and the co-posting links to find the shared template.
- **Research** — paste the manifest into a paper appendix. Every metric carries the
  seed, device, library versions and corpus checksum that produced it.

### Not good uses

Claims about how common something is, how many people saw it, or anything about a
named individual. See the next section.

---

## 17. What this system cannot do

These limits aren't disclaimers. They're stated in the code, the model cards, and the
API responses themselves.

> ## The governing statement
>
> **No output of this system is a determination that a person is a bot or that a
> claim is false.** Not the bot score, not the misinformation score, not the
> coordination score, and not any combination of them.
>
> They are research signals over public behaviour, and every one of them has innocent
> explanations. **A high score is a prompt to look, not a conclusion.**

### Coverage limits

- **English only, by construction.** Every search query, feed and hashtag is English.
  Any claim from this corpus is a claim about English-language content. Non-English
  text is skipped with a reason code rather than being scored by an English model.
- **This is not a sample of anything.** Mastodon is limited to certain hashtags and
  servers, YouTube to certain search queries, GDELT to the outlets it monitors, and
  the RSS feed list was hand-picked. **The corpus cannot support any claim about how
  common something is or how many people saw it.**
- **No X/Twitter, Telegram, Facebook, WhatsApp or TikTok.** A large share of the
  phenomenon being studied plausibly lives on platforms this project can't access.
- **Deleted content is absent**, so the corpus under-represents whatever moderators
  removed — which is plausibly correlated with the content of interest.
- **Sources have different time shapes.** The Reddit data is a fixed historical
  snapshot; everything else is a rolling window from the fetch date. Comparing volume
  *between* sources is meaningless. Trends *within* a source are fine.

### Measurement limits

- **Benchmark scores are not real-world accuracy.** The training data is politicians'
  statements, news headlines and COVID-era health claims. This corpus is Reddit
  comments, Mastodon posts and news metadata. **Expect a large drop.** The size of
  that gap is currently **unmeasured** — closing it needs a person to hand-label 100
  real corpus records, and that single table is worth more than any amount of
  hyperparameter tuning.
- **The toxicity model has a known demographic bias.** Classifiers trained on this
  data over-flag African-American English and identity terms used
  non-pejoratively. Nothing should ever be ranked by toxicity alone.
- **Coordination can't see across platforms.** Account identity doesn't survive
  between services, so one person coordinating from two accounts on two platforms
  appears as two unconnected dots. News, GDELT and flat Reddit dumps are excluded
  from it entirely.
- **Narrative clusters are topics, not claims** — and posts on opposite sides of the
  same argument end up in the same cluster.

---

## Where things live in the code

| What | Where |
|---|---|
| The record format (the contract) | `ingest/schema.py` |
| Source adapters | `ingest/sources/` |
| Text cleaning | `ingest/normalize.py` |
| Parquet writing and the manifest | `ingest/store.py` |
| Embeddings, clustering, misinformation | `modeling/text/` |
| Account features, bot model, coordination | `modeling/accounts/` |
| The one approved data splitter | `modeling/datasets/splits.py` |
| Training orchestration | `modeling/training.py` |
| Calibration | `modeling/eval/calibrate.py` |
| Parquet → Postgres | `app/etl/` |
| Fusion score, cohorts, normalisation | `app/scoring/` |
| Compass retrieval, generation, validation | `app/compass/` |
| Background jobs | `app/tasks/` |
| API routes | `app/routers/` |
| Database queries | `app/repositories/` |
| All tunable numbers | `configs/*.yaml` |
| Containers | `docker/`, `docker-compose.yml` |
| Evidence: metrics, model cards, error analysis | `artifacts/` |

### Related documents

- [`docs/data-model.md`](data-model.md) — the 27 database tables in detail
- [`docs/scoring.md`](scoring.md) — the fusion score reference
- [`docs/api.md`](api.md) — the API walkthrough
- [`README.md`](../README.md) — the full technical README for Phases 1 and 2

---

*Numbers quoted here (bot per-fold scores, misinformation vs. baseline, coordination
modularity vs. the null model) are the values recorded in this repository's own
evaluation artifacts and README. They were not re-measured for this document.*
