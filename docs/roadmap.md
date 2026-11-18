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
| source adapters | done | Reddit (ConvoKit, Kaggle), Mastodon, GDELT, news RSS, YouTube (yt-dlp, with subtitles), Telegram |
| India retarget | done | 18 Indian feeds, 4 topic areas, English and Hindi |
| script and language detection | done | Indic scripts, romanized Hindi, code-mix flag |
| fact-check verdicts | partial | 4311 verdicts, tiers 1 and 2 run (tier 3 implemented, not yet run on the data host), 4 of 8 publishers, Vishvas News archive read inside the 2022 window |
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
| deepfake module | removed | code, API, table and dependency tree are gone |
| first real load into Postgres | done, rescored | 957 news records, 3 narratives, scored by all four layers, rescored after the fact-checker skip, fusion `fusion-v1.2.0` runs, see below |
| dashboard | done | separate repo, built against the committed contract |
| CI | done | lint, suite on 3.11 and 3.12, OpenAPI drift, Postgres suite |

## What works, with numbers

**Ingestion produces an Indian, bilingual corpus.** A bounded run over the 18
configured feeds gave 548 records across 16 domains, split 454 English, 90 Hindi,
3 Telugu and 1 misdetected. Every feed URL was probed live before being listed,
and rejected candidates are recorded in `configs/sources.yaml` with the reason,
so "why is this outlet missing" has an answer.

**Fact-check extraction produces bilingual supervision.** The file holds 4311 verdicts:
Vishvas News 4203, BOOM 55, The Quint WebQoof 46, Fact Crescendo 7. The claim language is
Hindi for 4210, English for 100 and one other. 4310 verdicts have the label false and one
has the label true. Vishvas News and Fact Crescendo
supply claims in Devanagari, so this path yields Hindi labels and not merely Hindi text.
The corpus and the verdicts are not in the repo.

**The first real load into Postgres ran on real data.** The corpus has 957 news records:
English 716, Hindi 233, Telugu 4 and 4 in other languages. It has 17 authors (one per
outlet) and 124 domains. All 957 records have a score row, and fusion `fusion-v1.2.0`
ran. In the load before the rescore, English toxicity, sentiment, emotion and misinfo
covered all 716 records, and stance covered 83. The Laya stage scored all 233 Hindi
titles. Toxicity, sentiment, emotion and stance scored no Hindi record, because those
models are English only. After the rescore below, 145 English and 33 Hindi `misinfo` and
Laya values are cleared, and 771 of 957 posts have a `misinfo_likelihood`.

Clustering found 3 narratives, and 67 percent of records are noise. One narrative is
almost all Hindi (218 posts). All 3 narratives are low priority. The fusion scores
changed after the rescore that this section describes below. The coordination detector
found no edges, so the network graph is empty. No news site exposes likes, shares,
replies or views, so the engagements KPI is null and not 0. Every GET endpoint that the
check could fill returned without a server error.

**The first load was rescored after the fact-checker skip.** The `misinfo` and `laya`
stages now give a record from a fact-checker no score. Both stages clear a value that
they wrote earlier. The first load held scores from before that change. We ran both
stages again on the 957 records of the first load. Then we ran a migration to head and
reloaded the scored tables into the Postgres database of the first load. Then we
recomputed fusion.

The scorer finds 182 fact-checker posts. Their domains are:

| domain | posts |
|---|---|
| altnews.in | 23 |
| boomlive.in | 62 |
| digiteye.in | 6 |
| factcrescendo.com | 10 |
| factly.in | 12 |
| newschecker.in | 15 |
| thequint.com | 42 |
| vishvasnews.com | 12 |

The table shows the counts before and after the rescore.

| count | before | after |
|---|---|---|
| fact-checker posts with a `misinfo_likelihood` | 178 | 0 |
| fact-checker posts with a `fact_check_publisher` skip reason | 0 | 182 |
| `post_scores` rows with a `misinfo_likelihood`, of 957 | 949 | 771 |

The `misinfo` stage skipped 182 records with the reason `fact_check_publisher`. It also
skipped 204 records with the reason `unsupported_language`. It cleared 145 earlier values.
The `laya` stage skipped 33 Hindi records with the reason `fact_check_publisher` and
cleared their values. It wrote no new score, because the other records already had a score
under the current stamp. In total, 178 records lost a value.

The fusion scores of the 3 narratives are in the next table. Fusion is `fusion-v1.2.0`.
The first column is the value that Postgres stored on 2026-11-11. The second column is
the value that we recomputed on the old scores with the current inputs. The third
column is the value after the rescore.

| narrative | posts | stored 2026-11-11 | recomputed on old scores | after rescore |
|---|---|---|---|---|
| 1000 | 19 | 15.28 | 24.04 | 24.04 |
| 1001 | 75 | 26.58 | 36.05 | 34.22 |
| 1002 | 218 | 22.48 | 31.18 | 30.71 |

Two effects explain the table. The first effect is that the stored values were stale. The
stored scorecard has a null `domain_risk_agg`, because the domain risk scores did not exist
on 2026-11-11. Now the value is 0.8117 for narrative 1001. With the domain risk present,
narrative 1000 moves from 15.28 to 24.04, narrative 1001 from 26.58 to 36.05 and narrative
1002 from 22.48 to 31.18. The fusion code did not change in this step. The git log shows no
commit to `app/scoring/fusion.py`, `app/scoring/signals.py` or `configs/fusion.yaml` after
the stored run of 2026-11-11.

The second effect is the rescore. It moves narrative 1001 from 36.05 to 34.22 and narrative
1002 from 31.18 to 30.71. Narrative 1000 does not change. The `misinfo_likelihood_agg`
after the rescore is 0.2926, 0.6617 and 0.4179 for narratives 1000, 1001 and 1002. For
narrative 1001 it was 0.7379 before. All 3 narratives stay low priority.

**The API serves the full contract without a database.** All contract tests pass
under `DEMO_MODE=1`, and `make openapi-check` holds the committed `openapi.json`
to what the code produces.

## What does not work

These are reported because a module that fails quietly is worse than one that
fails loudly. None of them is a blocker for the pipeline running end to end.

**The Hindi gate passes with clear limits.** The Laya multilingual fine-tune has a
Hindi title lift of 1.760 [1.286, 2.630] over 4180 claims. Recall is 0.122, and the
corpus has 431 Hindi titles. With 1016 claims the lift was 1.834 [1.116, 3.981], so the
lower limit moved further from 1 and the interval is much narrower. 99.8 percent of the
Hindi claims come from one publisher. On Hindi full text, the lift is 0.476
[0.401, 0.573], and the upper limit is below 1. See
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
fact-checks are single-class: the measured split is 4310 false and 1 true, because
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

Report English and Hindi separately. Of the 4311 verdicts, 4210 claims are Hindi, 100 are
English and one is in another language. The frozen evaluation set has 4281 claims,
and 4180 of them are Hindi. The freeze collapsed 29 duplicate false verdicts and left
out the one true verdict. The Hindi estimate rests on those 4180 claims, so show the
interval and not the point estimate alone. The roberta fine-tune and TF-IDF cannot
score the Hindi subset at all. That asymmetry is the result, and "no score" is
the honest entry in that cell, not a zero.

## Open work, by area

### Ingestion

- **Hindi fact-check backfill.** The Vishvas News sitemap backfill now reads the whole
  archive inside the `min_lastmod` window, which starts on 2022-01-01. The final loop
  stopped by itself. No URL bound and no sitemap budget was reached. Pull request 21 made
  this possible. Before it, pages that the backfill fetched and dropped used the URL
  budget on every run, so each run added only a few verdicts. The backfill now records
  these pages in `data/factchecks/backfill_dropped.parquet` and skips them. It counts
  them as `previously_dropped`. The ledger holds 2713 pages: `rating_is_middle` 2384,
  `backfill_language_excluded` 226, `no_claimreview` 102 and `robots_disallowed` 1. The
  option `--recheck-dropped` fetches them again. Rows for 404 pages and robots.txt
  disallows expire after 30 days. Sitemaps that hold no new URL do not use the sitemap
  budget. The evaluation manifest `configs/factcheck_eval_manifest.csv` was refrozen at
  4281 claims, and the comparison and the Hindi gate result include them. The archive before
  2022-01-01 is outside the window and stays unread. One page is unread because robots.txt
  disallows it. Each run reads at most
  300 URLs (`backfill.max_urls_per_publisher`). The command
  `python -m ingest.cli factchecks --backfill` reads publisher sitemaps and sends each
  archive URL through the tier 1 ClaimReview path. The operator runs it on the data host.
  The budget is separate from `fetch.max_articles`. The Vishvas entry sets
  `claim_languages: [hi]`, so the backfill writes no English or Marathi row. The Hindi set
  therefore comes almost all from one publisher (99.8 percent) in that freeze. A Hindi
  result reflects the claims that Vishvas News chose to check, not Hindi misinformation
  in general. The BOOM Hindi archive is now configured through its paginated
  `/fact-check` listing, because its sitemaps hold only the newest day, and Fact
  Crescendo opts into the ClaimReview path under its `rss_tag` tier. Both blocks are in
  `configs/factcheck.yaml`, and the listing path and the tier opt-in are covered by tests.
  The next backfill run on the data host adds their claims, and
  `modeling.cli freeze-factchecks` prints the Hindi share per publisher; the comparison
  report records the same shares. Refreezing the manifest and rerunning the Hindi gate
  remain a data host task.
- **Fact-check extraction tier 4, and a tier 3 run on the data host.** Tier 3 reads
  the Newschecker result and claim blocks with selectors from `configs/factcheck.yaml`.
  Tier 4 reads the verdict from the Alt News and Factly headline. Both are implemented.
  Run `python -m ingest.cli factchecks` on the data host to add their rows to
  `factchecks.parquet`, then run `python -m modeling.cli dedup-claims` again. Tier 4
  rows stay out of the frozen manifest. Draw and label the
  100 headline sheet, score it, and write the result into
  `title_predicate.measured_error_rate`. The commands are in `docs/factcheck-tiers.md`. The error
  rate is not measured yet. Tier 4 stays out of the primary evaluation until a person
  decides otherwise and writes the decision down.
- **A person must label the claim dedup sample before the threshold is a measured
  choice.** `python -m modeling.cli dedup-claims` links claims by simhash and by the
  cosine similarity of multilingual claim embeddings, and writes
  `data/factchecks/claim_links.json` plus `data/factchecks/claim_dedup_sample.csv`.
  The sample holds linked pairs across the similarity range with a blank
  `is_same_claim`. The threshold `claim_dedup.min_similarity` is 0.85, chosen from the
  distribution of those pairs and not from a hand check, so the comparison report
  records the embedding precision as `not_measured` until the sample is labelled. The
  links cover the claims stored today; a longer collection window needs the command
  rerun before the evaluation set is frozen again. Run the command before freezing: if
  the pass turns a frozen row into a duplicate, the harness refuses the stale manifest,
  and the manifest is rebuilt on purpose with the reason recorded.

### Modeling

- **Measure the benchmark-to-corpus transfer gap.** LIAR is politicians'
  statements, CoAID is COVID-era health claims, and this corpus is Indian
  recontextualised media. The gap is still unmeasured. `modeling.cli sample-for-labelling`
  writes the blank CSV. `modeling.cli sample-for-labelling misinfo [--lang en|hi]` writes
  the blank sheet with no model score in it. Then `modeling.cli evaluate misinfo` reports
  the transfer section. Blank labels are not 0, null scores are excluded with a reason,
  and no metric is computed below `MIN_LABELLED_ROWS` (50) labelled rows. The 100-record
  English sample from 2026-11-11 is now labelled, but an AI model wrote the labels. The
  sheet marks every row `labeller=llm-unreviewed`, and a person did not check them. Every
  label is 0. 20 rows are fact-check articles that debunk a claim, and 80 are news
  reports, features or opinion. The 95 percent Wilson interval for the positive rate is
  [0.000, 0.037]. At threshold 0.5, `misinfo` v0.2.0 flags 42 of 100 rows (Wilson
  [0.328, 0.518]). All 42 flags are false positives against these labels. The model flags
  all 20 debunks and 22 of the 80 other articles. It reacts to articles about false
  claims, not only to articles that push one. The command `evaluate misinfo` refuses
  transfer metrics here, because the labelled rows hold one class. The next sample needs
  more likely positives, for example social posts or posts that fact-checkers debunked.
  A person must label it. The `misinfo` and `laya` scoring stages give records from a
  fact-checker no score (reason `fact_check_publisher`), so fusion no longer reads a
  debunk as misinformation. The list of fact-checker domains is
  `configs/factcheck.yaml`, `publishers` plus `other_fact_checkers`. The label sheet and
  the corpus text are not in git.
- **Rerun the fact-check comparison report.** The report in
  [results/factcheck-comparison.md](results/factcheck-comparison.md) left only the four
  publishers with an extraction tier out of the corpus flag rate. Alt News, Factly,
  Newschecker and DigitEye articles stayed in it. `modeling.cli evaluate-factchecks`
  on the same score files and seed now leaves out all eight. It needs no GPU. The flag
  rates and lifts can change. The ship decisions use macro F1 on the benchmark split and
  do not change. The Hindi gate uses the Hindi title lift, and Alt News Hindi articles
  were in that flag rate, so the gate result has to be read again from the new report.
- **Reframe bot detection as coordinated behaviour detection.** Supervised
  per-account classification trained on Twitter data does not transfer, and there
  are no labels for the platforms in this corpus. The unsupervised coordination
  path needs no labels and is validated against a time-shuffled null.

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
