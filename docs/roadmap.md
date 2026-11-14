# State and remaining work

What is built, what is measured, what does not work, and what is left. Read
[CLAUDE.md](../CLAUDE.md) first for how the repo is laid out and what must not be
broken.

Every number below came from a run against real data or the committed fixtures.
Where a module does not work, the measurement saying so is included rather than
the module being quietly described as present.

## Status by component

| component | state | notes |
|---|---|---|
| canonical schema | done | `ingest/schema.py`, the contract everything converges on |
| source adapters | done | Reddit (ConvoKit, Kaggle), Mastodon, GDELT, news RSS, YouTube |
| India retarget | done | 18 Indian feeds, 4 topic areas, English and Hindi |
| script and language detection | done | Indic scripts, romanized Hindi, code-mix flag |
| fact-check verdicts | partial | 1108 verdicts, 2 of 4 extraction tiers, 4 of 8 publishers, Vishvas News backfill ran 7 times and the archive is not fully read |
| embeddings and clustering | done | MiniLM, HDBSCAN, narrative identity across runs |
| misinformation classifier | clears baseline | roberta-base fine-tune v0.2.0, see the comparison result |
| stance classifier | fine-tuned | macro-F1 0.797 on the FNC-1 competition test set, see the model card |
| bot classifier | does not generalize | see below |
| coordination detection | no signal on this corpus | see below |
| auxiliary scorers | done | toxicity, sentiment, emotion, anomaly, all off the shelf |
| Laya scorer | Hindi titles only | calibrated stage in modeling/scoring.py, English stays with roberta, see the model card |
| fusion scoring | done | `configs/fusion.yaml`, returns its own explanation, coordination input is the weak coordination score, not the bot ratio |
| API | done | 66 paths, 77 operations, contract tested under `DEMO_MODE` |
| database | done | 26 tables, 5 migrations, round-trip asserted in CI |
| deepfake module | removed | code, API, table and dependency tree are gone. `maint.purge_uploads` stays until every deployment has passed one retention window |
| first real load into Postgres | done | 957 news records, 3 narratives, scored by all four layers, fusion `fusion-v1.2.0` runs, see below |
| dashboard | done | separate repo, built against the committed contract |
| CI | done | lint, suite on 3.11 and 3.12, OpenAPI drift, Postgres suite |

## What works, with numbers

**Ingestion produces an Indian, bilingual corpus.** A bounded run over the 18
configured feeds gave 548 records across 16 domains, split 454 English, 90 Hindi,
3 Telugu and 1 misdetected. Every feed URL was probed live before being listed,
and rejected candidates are recorded in `configs/sources.yaml` with the reason,
so "why is this outlet missing" has an answer.

**Fact-check extraction produces bilingual supervision.** The file holds 1108 verdicts:
Vishvas News 1012, BOOM 53, The Quint WebQoof 38, Fact Crescendo 5. The claim language is
Hindi for 1017, English for 90 and one other. Every verdict has the label false. The
rating is `False` for 1105 verdicts and `Altered` for 3. Vishvas News and Fact Crescendo
supply claims in Devanagari, so this path yields Hindi labels and not merely Hindi text.
The corpus and the verdicts are not in the repo.

**The first real load into Postgres ran on real data.** The corpus has 957 news records:
English 716, Hindi 233, Telugu 4 and 4 in other languages. It has 17 authors (one per
outlet) and 124 domains. All 957 records have a score row, and fusion `fusion-v1.2.0`
ran. English toxicity, sentiment, emotion and misinfo cover all 716 records, and stance
covers 83. The Laya stage scored all 233 Hindi titles. Toxicity, sentiment, emotion and
stance scored no Hindi record, because those models are English only.

Clustering found 3 narratives, and 67 percent of records are noise. One narrative is
almost all Hindi (218 posts). The fusion scores are 26.6, 22.5 and 15.3, so all 3
narratives are low priority. The coordination detector found no edges, so the network
graph is empty. No news site exposes likes, shares, replies or views, so the engagements
KPI is null and not 0. Every GET endpoint that the check could fill returned without a
server error.

**The API serves the full contract without a database.** All contract tests pass
under `DEMO_MODE=1`, and `make openapi-check` holds the committed `openapi.json`
to what the code produces.

## What does not work

These are reported because a module that fails quietly is worse than one that
fails loudly. None of them is a blocker for the pipeline running end to end.

**The Hindi gate passes with clear limits.** The Laya multilingual fine-tune has a
Hindi title lift of 1.834 [1.116, 3.981] over 1016 claims. Recall is 0.104, and the
corpus has 211 Hindi titles. Almost all Hindi claims come from one publisher. On Hindi
full text, the lift is 0.367 [0.280, 0.487], and the upper limit is below 1. See
`artifacts/model_cards/laya.md`.

**The bot classifier does not generalize to unseen botnets.** Trained on
Cresci-2017: pooled out-of-fold macro-F1 0.705, but the mean across folds is
0.416 with a standard deviation of 0.187 and a worst fold of 0.249. The pooled
figure averages away the folds where the held-out campaign looked nothing like
the training ones. Quote the per-fold mean. Human recall is 0.608, so 39 percent
of genuine accounts are flagged at the chosen threshold. For this reason the fusion
formula does not read `bot_like_ratio`. It reads `coordinated_author_ratio`.

**Coordination modularity does not exceed the time-shuffled null.** Observed
0.912 against 0.979 for the null. Any graph has communities; on this corpus the
communities found are not evidence of coordination. The detector does recover a
planted coordinated burst on the demo fixture, so the mechanism works and the
corpus simply does not contain the phenomenon at a detectable level.

**The default embedder groups Hindi by language.** On a 636-record news corpus, the
default MiniLM model placed 147 of the 156 Hindi records in one cluster with 2 English
records. That cluster is a language group, not a narrative. The multilingual
candidate found no clusters at `min_cluster_size` 15 on the same records, so the
default stays. Repeat `modeling.cli compare-embedder` when the corpus is larger.

**Hindi is ingested and only partly scored.** Each module in `configs/models.yaml`
has its own `languages` list. The English-only modules skip Hindi with a reason code.

**Old Hindi hashes are stale.** The tokenizer now keeps the zero width joiner and
non-joiner inside a word, so Indic text with a conjunct such as क्‍ष gets a
different simhash than before. A corpus written before this change holds stale
simhash values for Indic text. Re-normalize it before you compare hashes across
the change.

## Comparison result

The three-way comparison ran on 2026-11-09 on a GPU host. The full report is
[results/factcheck-comparison.md](results/factcheck-comparison.md), and the runbook is
[laya-finetune.md](laya-finetune.md).

| model | test macro F1 at precision 0.85 | difference to TF-IDF | decision |
|---|---|---|---|
| roberta-base fine-tune | 0.753 [0.741, 0.764] | +0.150 [0.135, 0.164] | ship |
| Laya English fine-tune | 0.683 [0.670, 0.696] | +0.079 [0.065, 0.095] | passes the rule, not added |
| Laya multilingual fine-tune | 0.565 [0.552, 0.578] | -0.039 [-0.054, -0.022] | do not ship |
| TF-IDF + logreg | 0.603 [0.590, 0.616] | baseline | |

At the default threshold of 0.5, the gap between roberta and TF-IDF is 0.033. roberta model cannot score Hindi. Only the Laya multilingual fine-tune scores Hindi.

### How to report the fact-check evaluation

`data/factchecks/factchecks.parquet` is an **evaluation set, never a training set**. Published
fact-checks are single-class: the measured split is 1108 false and 0 true, because
debunking a viral falsehood is news and confirming a true claim is not. Nothing
can be trained on one class.

A model that answers "false" to everything scores **recall 1.0** on a
positives-only set. So recall from this set must never be reported alone. Report
it against the model's flag rate on the unlabelled corpus at the same threshold:

```
lift = recall on known-false claims / flag rate on the whole corpus
```

90 percent recall at a 10 percent flag rate is a finding. 90 percent recall at an
85 percent flag rate is a model that learned to say "false", and the recall
figure alone cannot distinguish them.

Report English and Hindi separately. Of the 1108 verdicts, 1017 claims are Hindi, 90 are
English and one is in another language. The frozen evaluation set has 1107 claims,
and 1016 of them are Hindi. The freeze collapsed one duplicate verdict. The Hindi estimate rests on those 1016 claims, so show the
interval and not the point estimate alone. The roberta fine-tune and TF-IDF cannot
score the Hindi subset at all. That asymmetry is the result, and "no score" is
the honest entry in that cell, not a zero.

## Open work, by area

### Ingestion

- **Hindi fact-check backfill.** The Vishvas News backfill ran 7 times on 2026-11-11.
  The new verdicts per run were 210, 164, 125, 96, 73, 49 and 37. About 170 URLs and 12
  sitemaps are still unread, so the archive is not fully read. Each run reads at most 300
  URLs (`backfill.max_urls_per_publisher`). Before the fix, pages that the backfill
  fetched and dropped used the URL budget on every run. About 299 pages of each 300 were
  such pages, so each run added only a few verdicts. The backfill now records these pages
  in `data/factchecks/backfill_dropped.parquet` and skips them. It counts them as
  `previously_dropped`. The option `--recheck-dropped` fetches them again. Rows for 404 pages and robots.txt
  disallows expire after 30 days. Sitemaps that
  hold no new URL no longer use the sitemap budget, so the older sitemaps can be reached.
  The unread sitemaps need new runs with this fix. The evaluation manifest
  `configs/factcheck_eval_manifest.csv` was refrozen at 1107 claims after this backfill.
  The reported comparison and the Hindi gate result include those claims. The unread URLs
  need another refreeze and a rerun of the comparison on a GPU host. The command
  `python -m ingest.cli factchecks --backfill` reads publisher sitemaps and sends each
  archive URL through the tier 1 ClaimReview path. The operator runs it on the data host.
  The budget is separate from `fetch.max_articles`. The Vishvas entry sets
  `claim_languages: [hi]`, so the backfill writes no English or Marathi row. The Hindi set
  therefore comes mostly from one publisher. A Hindi result reflects the claims that
  Vishvas News chose to check, not Hindi misinformation in general. Fact Crescendo and
  the BOOM Hindi site are not configured. Fact Crescendo shows a ClaimReview on only some
  pages. The BOOM Hindi site lists only the newest articles in its sitemaps. To add a
  publisher, add a `backfill` block to `configs/factcheck.yaml`. No code change is
  necessary.
- **Fact-check extraction tiers 3 and 4.** Newschecker publishes no JSON-LD but
  renders a visible `Claim` block in the article body. Alt News and Factly
  publish neither, and their verdict lives in the headline predicate. Tier 3
  needs a per-publisher CSS selector that will break on a reskin; tier 4 is
  inference rather than extraction and needs its own measured error rate before
  it sits next to structured data. Both would roughly double the evaluation set.
- **Telegram adapter.** Free official API, widely used for political forwards in
  the target domain, and the cheapest large source still missing.
- **Replace the YouTube Data API with yt-dlp.** Removes the only hard API-key
  dependency and the 10,000 unit daily quota, and additionally yields subtitles.
- **Cross-publisher claim deduplication over longer windows.** A 96-claim
  set measured earlier had zero near-duplicates at the project's own Hamming threshold.
  The set of 1108 verdicts is not measured. Several publishers often debunk the same claim
  within days, so a longer collection window will produce near-duplicates. Only
  `claim_simhash` is stored today, and no code compares claims yet.

### Modeling

- **Measure the benchmark-to-corpus transfer gap.** LIAR is politicians'
  statements, CoAID is COVID-era health claims, and this corpus is Indian
  recontextualised media. The gap is currently unmeasured. Closing it needs 100
  hand-labelled corpus records, and that single table is worth more than any
  hyperparameter sweep. `modeling.cli sample-for-labelling` writes the blank CSV.
  `modeling.cli sample-for-labelling misinfo [--lang en|hi]` writes the blank
  sheet with no model score in it. Label the `label` column by hand, then
  `modeling.cli evaluate misinfo` reports the transfer section. Blank labels are
  not 0, null scores are excluded with a reason, and no metric is computed below
  `MIN_LABELLED_ROWS` (50) labelled rows. The English sample sheet exists.
  `modeling.cli sample-for-labelling misinfo --n 100` wrote 100 English records on
  2026-11-11. The sample excludes 241 records that the misinfo stage cannot score. No row
  has a label yet.
- **Reframe bot detection as coordinated behaviour detection.** Supervised
  per-account classification trained on Twitter data does not transfer, and there
  are no labels for the platforms in this corpus. The unsupervised coordination
  path needs no labels and is validated against a time-shuffled null.

### Backend and frontend

- **Remove `maint.purge_uploads`.** The deepfake module is gone, together with its
  API, its table and its dependency tree. The upload sweep stays only to delete
  files that a deployment stored before the removal. Remove the task, its beat
  schedule, `uploads_dir` and `media_retention_hours` after every deployment has
  passed one retention window.

### Known gaps that are not scheduled

- **No WhatsApp, X, Facebook, Instagram or ShareChat.** For an Indian case this
  is the largest coverage gap, not a minor one. Treat the corpus as public web
  discourse *about* misinformation, not as the misinformation itself.
- **The news corpus is urban, English-weighted and nationally focused.** Every
  feed is a national outlet or national fact-checker. Regional-language and
  district-level reporting is absent, partly by scope and partly because most
  regional dailies' feeds are dead or blocked. The rejected candidates are listed
  in `configs/sources.yaml`.
- **Fact-checker coverage is a selection, not a census.** What gets debunked is
  driven by editorial priorities, funding and what is already viral. Narratives
  they ignore are invisible here, and nothing in this corpus distinguishes "not
  debunked" from "not false".
