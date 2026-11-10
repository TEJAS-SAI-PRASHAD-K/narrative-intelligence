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
| fact-check verdicts | partial | 2 of 4 extraction tiers, 4 of 8 publishers, sitemap backfill for Vishvas News |
| embeddings and clustering | done | MiniLM, HDBSCAN, narrative identity across runs |
| misinformation classifier | clears baseline | roberta-base fine-tune v0.2.0, see the comparison result |
| stance classifier | fine-tuned | macro-F1 0.797 on the FNC-1 competition test set, see the model card |
| bot classifier | does not generalize | see below |
| coordination detection | no signal on this corpus | see below |
| auxiliary scorers | done | toxicity, sentiment, emotion, anomaly, all off the shelf |
| Laya scorer | fine-tuned, not shipped | see the comparison result, no stage in modeling/scoring.py |
| fusion scoring | done | `configs/fusion.yaml`, returns its own explanation, coordination input is the weak coordination score, not the bot ratio |
| API | done | 69 paths, 80 operations, contract tested under `DEMO_MODE` |
| database | done | 27 tables, 4 migrations, round-trip asserted in CI |
| dashboard | done | separate repo, built against the committed contract |
| CI | done | lint, suite on 3.11 and 3.12, OpenAPI drift, Postgres suite |

## What works, with numbers

**Ingestion produces an Indian, bilingual corpus.** A bounded run over the 18
configured feeds gave 548 records across 16 domains, split 454 English, 90 Hindi,
3 Telugu and 1 misdetected. Every feed URL was probed live before being listed,
and rejected candidates are recorded in `configs/sources.yaml` with the reason,
so "why is this outlet missing" has an answer.

**Fact-check extraction produces bilingual supervision.** 96 verdicts from 111
eligible records: BOOM 55, The Quint WebQoof 29, Vishvas News 8, Fact Crescendo
4. Vishvas News and Fact Crescendo supply claims in Devanagari, so this path
yields Hindi labels and not merely Hindi text. The corpus and the verdicts are not
in the repo. RSS feeds hold only recent entries, so a new run does not produce the
same 96 verdicts.

**The API serves the full contract without a database.** All contract tests pass
under `DEMO_MODE=1`, and `make openapi-check` holds the committed `openapi.json`
to what the code produces.

## What does not work

These are reported because a module that fails quietly is worse than one that
fails loudly. None of them is a blocker for the pipeline running end to end.

**The Hindi gate passes with clear limits.** The Laya multilingual fine-tune has a
Hindi title lift of 2.233 [1.146, 6.605] over 261 claims. Recall is 0.115, and the
corpus has 136 Hindi titles. Almost all Hindi claims come from one publisher. See
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
fact-checks are single-class: the measured split is 96 false and 0 true, because
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

Report English and Hindi separately. Of the 96 claims, 83 are English and 12 are
Hindi, so the Hindi estimate rests on 12 items and the interval must be shown
rather than the point estimate alone. The roberta fine-tune and TF-IDF cannot
score the Hindi subset at all. That asymmetry is the result, and "no score" is
the honest entry in that cell, not a zero.

## Open work, by area

### Ingestion

- **Hindi fact-check backfill.** The Hindi evaluation gate needs at least 50 Hindi
  claims, and the corpus has 13. RSS feeds hold only recent items. The command
  `python -m ingest.cli factchecks --backfill` reads publisher sitemaps and sends each
  archive URL through the tier 1 ClaimReview path. Vishvas News is configured. The
  operator runs the backfill on the data host. Until that run, the Hindi count is
  not changed. The backfill budget is `backfill.max_urls_per_publisher`. It is separate
  from `fetch.max_articles`. The Vishvas entry sets `claim_languages: [hi]`, so the
  backfill writes no English or Marathi row. The Hindi set will then come mostly from one
  publisher. A Hindi result will therefore reflect the claims that Vishvas News chose to
  check, not Hindi misinformation in general. Fact Crescendo and the BOOM Hindi site are not configured. Fact Crescendo
  shows a ClaimReview on only some pages. The BOOM Hindi site lists only the newest
  articles in its sitemaps. To add a publisher, add a `backfill` block to
  `configs/factcheck.yaml`. No code change is necessary.
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
- **Cross-publisher claim deduplication over longer windows.** The current
  96-claim set has zero near-duplicates at the project's own Hamming threshold,
  but the same claim gets debunked by several publishers within days, so a longer
  collection window will produce them. Only `claim_simhash` is stored today,
  and no code compares claims yet.

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
  `MIN_LABELLED_ROWS` (50) labelled rows.
- **Reframe bot detection as coordinated behaviour detection.** Supervised
  per-account classification trained on Twitter data does not transfer, and there
  are no labels for the platforms in this corpus. The unsupervised coordination
  path needs no labels and is validated against a time-shuffled null.

### Backend and frontend

- **Remove the deepfake module.** It is out of scope, carries the heaviest
  dependency tree in the project, and its detector is uncalibrated with an
  operating point of 0.99 that belongs to one specific checkpoint.
- **Surface the language gate in the dashboard.** The scored tables cover a
  subset of the corpus while Hindi goes unscored, and nothing in the interface
  currently says so.

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
