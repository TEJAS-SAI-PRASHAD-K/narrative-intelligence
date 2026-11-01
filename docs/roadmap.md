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
| fact-check verdicts | partial | 2 of 4 extraction tiers, 4 of 8 publishers |
| embeddings and clustering | done | MiniLM, HDBSCAN, narrative identity across runs |
| misinformation classifier | does not clear baseline | see below |
| stance classifier | training path only | fine-tune is GPU work, zero-shot baseline exists |
| bot classifier | does not generalize | see below |
| coordination detection | no signal on this corpus | see below |
| auxiliary scorers | done | toxicity, sentiment, emotion, anomaly, all off the shelf |
| Laya scorer | code only | never run against real weights |
| fusion scoring | done | `configs/fusion.yaml`, returns its own explanation |
| API | done | 69 paths, 80 operations, contract tested under `DEMO_MODE` |
| database | done | 27 tables, 3 migrations, round-trip asserted in CI |
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
yields Hindi labels and not merely Hindi text.

**The API serves the full contract without a database.** All contract tests pass
under `DEMO_MODE=1`, and `make openapi-check` holds the committed `openapi.json`
to what the code produces.

## What does not work

These are reported because a module that fails quietly is worse than one that
fails loudly. None of them is a blocker for the pipeline running end to end.

**The misinformation classifier does not clear TF-IDF.** On the demo fixture,
macro-F1 0.908 against 0.927 for TF-IDF plus logistic regression, intervals
overlapping. A full-scale run on 26,777 rows recorded macro-F1 0.8062 with a
95 percent interval of 0.7956 to 0.8161 over 7,085 test rows, but **its weights
are no longer on disk**, only the config, tokenizer and calibrator. The number
cannot be reproduced from what is here. Retraining is the only way to close it.
The bar stands: if the transformer does not clear TF-IDF it should not ship,
because it costs orders of magnitude more inference for no measured gain.

**The bot classifier does not generalize to unseen botnets.** Trained on
Cresci-2017: pooled out-of-fold macro-F1 0.705, but the mean across folds is
0.416 with a standard deviation of 0.187 and a worst fold of 0.249. The pooled
figure averages away the folds where the held-out campaign looked nothing like
the training ones. Quote the per-fold mean. Human recall is 0.608, so 39 percent
of genuine accounts are flagged at the chosen threshold.

**Coordination modularity does not exceed the time-shuffled null.** Observed
0.912 against 0.979 for the null. Any graph has communities; on this corpus the
communities found are not evidence of coordination. The detector does recover a
planted coordinated burst on the demo fixture, so the mechanism works and the
corpus simply does not contain the phenomenon at a detectable level.

**Hindi is ingested but not scored.** `modeling/config.py` sets
`languages = ("en",)`, so Hindi records are collected, normalized, stored, and
then skipped by every scorer with a reason code. The scored tables therefore
cover a subset of the corpus, and any volume comparison between a Hindi and an
English narrative is invalid until a multilingual checkpoint lands.

## Critical path

The shortest route from here to a defensible result, in order. Each step unblocks
the next.

1. **Fetch the benchmarks.** `data/benchmarks/` is empty. `make benchmarks`
   pulls LIAR, CoAID and FakeNewsNet with checksums. FakeNewsNet ships ids and a
   crawler rather than article text, and CoAID's tweet content cannot be
   recollected, so budget for that. LIAR alone is enough for a first number and
   is what the current classifier was trained on, which is what makes the
   comparison like for like.

2. **Fine-tune Laya on a GPU.** Follow [laya-finetune.md](laya-finetune.md). The
   scorer and the training exporter are written and tested, but nothing has
   touched the real weights. Fine-tuning is mandatory rather than optional: the
   base checkpoint scores below its own majority-class baseline on the vendor's
   benchmark, so a zero-shot number is a baseline to report, never a model to
   ship.

3. **Build the three-way evaluation harness.** This is the deliverable and it
   does not exist yet. Score the fact-check evaluation set with Laya, with the
   roberta-base fine-tune and with TF-IDF plus logistic regression, on identical
   group-aware splits, and report them side by side. Details in step 6 of the
   fine-tune runbook.

4. **Widen the scoring language gate.** Once a multilingual checkpoint exists,
   `modeling/config.py` `Settings.languages` can include `hi`, and the Hindi
   corpus stops being collected-but-unscored.

### How to report the fact-check evaluation

`data/factchecks/` is an **evaluation set, never a training set**. Published
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
  collection window will produce them. The check exists; it needs to run over
  accumulated data.

### Modeling

- **Retrain the misinformation classifier** and either clear TF-IDF or drop the
  transformer. Its weights are gone, so the current state is neither.
- **Fine-tune stance.** The corpus supplied as SemEval-2016 is actually FNC-1,
  which is the better fit because its four labels map one to one onto the
  contract. 75,385 pairs over 2,587 bodies, all on disk. The fine-tune starts
  from an NLI and FEVER checkpoint rather than a bare encoder.
- **Measure the benchmark-to-corpus transfer gap.** LIAR is politicians'
  statements, CoAID is COVID-era health claims, and this corpus is Indian
  recontextualised media. The gap is currently unmeasured. Closing it needs 100
  hand-labelled corpus records, and that single table is worth more than any
  hyperparameter sweep. `modeling.cli sample-for-labelling` writes the blank CSV.
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
