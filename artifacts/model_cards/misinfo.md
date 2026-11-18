# Model card — misinformation-likelihood classifier

**Module:** `modeling/text/misinfo_clf.py` · **Version:** `v0.2.0`
**Output:** `record_scores.misinfo_prob` — a calibrated probability in [0, 1]

---

## Status

**The transformer clears the TF-IDF baseline at the precision 0.85 operating point.**
This card reports the comparison run of 2026-11-09. The full report is
[docs/results/factcheck-comparison.md](../../docs/results/factcheck-comparison.md).

The checkpoint is a `roberta-base` fine-tune. Training saved the last epoch. The
checkpoint sha256 is `e6726079336c08cc7f22052f1c4222a25d7a600aaf22a25b53c68dec97c1adf0`.
A private Hugging Face copy holds the weights at `misinfo/v0.2.0`. See "Get the
weights on a fresh clone" below.

The weights of the older `v0.1.0` run are lost. That run recorded macro F1 0.8062
[0.7956, 0.8161] on the same split. It is not reproducible, and `v0.2.0` replaces it.

Test split at the default threshold of 0.5:

| metric | value | 95% CI |
|---|---|---|
| macro F1 | **0.7967** | [0.7862, 0.8078] |
| ROC-AUC | 0.8978 | [0.8903, 0.9049] |
| PR-AUC | 0.7872 | [0.7701, 0.8026] |
| Brier | 0.1151 | not computed |

Split: `grouped by group_id (30148 groups; train/val/test = 26777/3596/7085; seed=20260813)` · n_test 7085 · positive rate 0.2991

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| not-misinformation | 0.868 | 0.900 | 0.884 | 4966 |
| misinformation-like | 0.743 | 0.680 | 0.710 | 2119 |

```bash
python scripts/run_comparison.py all    # needs a GPU, see docs/laya-finetune.md
```

## Intended use

A component of a research scorecard for studying how narratives spread across
platforms. It scores how *similar a text is to claims that professional
fact-checkers have rated false*.

## Out-of-scope use

- **It is not a determination that a claim is false.** No output of this system
  is that, and the README says so as a project-level limitation.
- Not for moderation decisions, enforcement, or ranking anyone's account.
- Not for non-English text; those records are skipped, not scored.
- Not as a standalone signal — it is one of six fields in a scorecard.

---

## Training data and the label mapping

| dataset | what it is | access |
|---|---|---|
| LIAR | 12.8k PolitiFact statements by politicians, 6-way labels | open download |
| FakeNewsNet | PolitiFact + GossipCop article **titles** (not bodies) | crawler; titles only |
| CoAID | COVID-era health claims across 4 collection waves | git clone |

**The collapse is a modeling choice with consequences.** LIAR's six-way ordinal
scale becomes binary:

```
pants-fire, false, barely-true  ->  1 (misinformation-like)
half-true                       ->  DROPPED
mostly-true, true               ->  0
```

`half-true` is **dropped, not assigned**. Forcing it either way manufactures
label noise the metrics cannot see: a model that gets every half-true wrong
looks identical to one that gets them right, when they are split down the middle
by an arbitrary rule. Dropping costs roughly a sixth of LIAR's rows and buys a
target that means something.

The mapping lives in `configs/models.yaml`, with every key and value quoted —
YAML 1.1 parses a bare `false:` as a boolean, which silently stopped the mapping
matching LIAR's string labels and dropped every `false` row with no error
anywhere. The loader now also refuses to run if the map does not cover all six
labels.

**FakeNewsNet is titles only.** The repository ships ids and a crawler, not
article bodies, and the tweet half needs Twitter API keys this project does not
have. "Trained on FakeNewsNet" implies far more data than headlines; it is
headlines.

## Split strategy

Group-aware, through `modeling/datasets/splits.py`, which is the only splitter in
the codebase (enforced by a source scan in `tests/test_splits.py`).

| dataset | group key | why |
|---|---|---|
| LIAR | speaker | one politician's statements share phrasing, topic and fact-check history |
| FakeNewsNet | article id, plus outlet | one outlet's house style is memorizable |
| CoAID | normalized claim text | the same claim recurs across collection waves under different row ids |

Group ids are namespaced by dataset (`liar:…`, `coaid:…`) so two benchmarks
using small integer ids cannot collide into one pseudo-group.

Near-duplicates are collapsed **before** splitting. Two rows can carry different
claim ids and be the same sentence — a syndicated wire story republished under
two outlet ids is the canonical case — and that leaks even though the group keys
differ.

## Metrics

Reported with 95% bootstrap confidence intervals. **Accuracy is deliberately
absent**: on an imbalanced target it rewards predicting the majority class.

The ship rule compares macro F1 against TF-IDF plus logistic regression on the shared
test split. Each model uses its own threshold, from its own validation rows, at
precision 0.85. The 95% paired-bootstrap interval of the difference must be above 0.

| model | threshold | validation recall | test macro F1 | difference to TF-IDF |
|---|---|---|---|---|
| `misinfo` v0.2.0 | 0.7389 | 0.482 | 0.753 [0.741, 0.764] | **+0.150 [0.135, 0.164]** |
| TF-IDF + logreg | 0.8573 | 0.244 | 0.603 [0.590, 0.616] | baseline |

**Decision: ship.** The lower limit is 0.1354.

Read the size of this difference with care. At the default threshold, TF-IDF has
macro F1 0.764 [0.753, 0.773], and the transformer has 0.797. That gap is 0.033. The
gap is larger at precision 0.85 because TF-IDF reaches that precision only at a
recall of 0.244. The operating point is the decision rule, and the default-threshold
gap is the conservative figure.

### Fact-check set and the unlabelled corpus

The set has 4428 known-false claims (report of 2026-11-19), and the frozen manifest is
`configs/factcheck_eval_manifest.csv`. This model scores the 93 English claims and
skips the 4335 other claims with `unsupported_language`. The skipped claims are 4334
Hindi claims and one claim in another language. The report left all eight fact-checker
domains out of the corpus (265 rows). The earlier report, on 4281 claims and four domains
left out, gave a recall of 0.520 [0.420, 0.610], a flag rate of 0.081 [0.067, 0.096] on
1350 corpus titles and a lift of 6.382 [4.950, 8.297].

| measure, English titles | `misinfo` v0.2.0 | TF-IDF + logreg |
|---|---|---|
| recall on known-false claims | 0.516 [0.419, 0.613] | 0.032 [0.000, 0.075] |
| flag rate on 2205 scored corpus titles | 0.063 [0.053, 0.074] | 0.004 [0.002, 0.007] |
| lift | 8.188 [6.231, 10.538] | 7.903 [0.000, 28.452] |

A TF-IDF lift has no useful meaning here, because the model flags almost no corpus
title. The fact-check set holds only claims that publishers chose to debunk. Lift is
not accuracy on the corpus, and the corpus has no labels. These 2205 titles are the
English corpus rows that have a title (5706 English rows).

The operating threshold is a value of the comparison report. The scored tables hold
the calibrated probability, and `configs/models.yaml` does not store this threshold.

The files under `artifacts/eval/misinfo/v0.1.0/` are from the 55-row demo fixture.
They are not a result.

### Corpus sample labels

The 100-record English sample from `modeling.cli sample-for-labelling misinfo --n 100`
(2026-11-11) now has labels. An AI model wrote the labels, not a person. The sheet marks
every row `labeller=llm-unreviewed`. A person did not check the labels.

Every label is 0. Of the 100 rows, 20 are fact-check articles that debunk a claim
(DigitEye, Newschecker, Alt News, BOOM, The Quint). The other 80 are news reports,
features or opinion. No row pushes a false claim. The 95 percent Wilson interval for the
positive rate is [0.000, 0.037].

At its configured `decision_threshold` of 0.5, on the calibrated `record_scores`
values, `misinfo` v0.2.0 flags 42 of the 100 rows (Wilson [0.328, 0.518]). Against these
labels, every flag is a false positive. The model flags all 20 debunks (20 of 20,
[0.839, 1.000]) and 22 of the 80 other articles (0.275, [0.189, 0.381]). The model
reacts to articles about false claims, not only to articles that push one.

The scoring stage gives records from a fact-checker no value and the reason
`misinfo:fact_check_publisher`, the same domains that the fact-check comparison leaves
out of its corpus. Without that skip, fusion would read a debunk as misinformation with
the weight `misinfo_likelihood_agg` (0.40) in `configs/fusion.yaml`. The skip does not
fix the cause: the model still flags articles about false claims from outlets that are
not fact-checkers, 22 of the 80 in this sample.

`modeling.cli evaluate misinfo` refuses transfer metrics on this sheet, because the
labelled rows hold one class. The transfer gap stays unmeasured. A useful next sample
needs more likely positives, for example social posts or posts that fact-checkers
debunked, and a person must label it. The label sheet and the corpus text are not in git.

## Calibration

Isotonic by configuration, with an automatic fallback to Platt scaling below 200
validation rows — isotonic fits a step function to noise on small validation
sets and emits confident 0.0/1.0 outputs. The fallback fired on the demo run (31
validation rows) and is recorded in the report.

Demo run: Platt on 31 rows, Brier 0.0756 → 0.0570 (improved). The real run has a
test Brier of 0.1151.

**A checkpoint without a `calibrator.json` is refused at load time.**
`misinfo_prob` is contractually a calibrated probability that Phase 4 multiplies
into a fused score; serving raw softmax under that column name would be a silent
lie.

## Known domain shift — read before quoting any number

**Three benchmarks, three different problems.** LIAR is politicians' statements
fact-checked by journalists. FakeNewsNet is news headlines. CoAID is COVID-era
health claims. Training on their union produces a model good at none of them
individually and whose errors are hard to attribute. The union is used because
each alone is too small, and per-benchmark test metrics are reported so the
mixture is visible rather than averaged away.

**The corpus is none of the above.** This project's records are Reddit comments,
Mastodon toots and GDELT article metadata — a different register, a different
length distribution, and a different relationship between text and claim.
**Expect a large drop.**

**Cross-domain, inside one benchmark.** FakeNewsNet's PolitiFact → GossipCop
holdout is run and reported. That number is more honest than in-domain F1
because house style is memorizable and the underlying task is not.

**The transfer gap is currently UNMEASURED.** A first sample is labelled, but the labels
come from an AI model and hold one class (see "Corpus sample labels"). A person must close
the gap:

```bash
python -m modeling.cli sample-for-labelling misinfo --lang en --n 100
# two labellers fill in the sheet (docs/labelling-guide.md), adjudicate, set labeller_kind, then:
python -m modeling.cli evaluate misinfo   # or: make relabel-eval
```

That single table is worth more than any hyperparameter sweep, and until it
exists the honest statement is that production accuracy is unknown and expected
to be substantially below the benchmark numbers.

## Error analysis

`artifacts/error_analysis/misinfo.md`. The known failure modes, encoded as a
triage taxonomy: satire and parody read as sincere claims (the most common false
positive for this whole model family, because satire and disinformation share
surface form by design), sarcasm, posts quoting a false claim in order to debunk
it, text too short to carry a claim, and opinion with no checkable proposition.

The counts are automated; the analysis is not. The uncategorized examples are
where new categories come from.

## Fairness

Score distributions across language groups and source slices are reported in
notebook 05. English-only coverage means the fairness question is currently
about *register* rather than language: the model was trained on edited,
journalist-adjacent prose and is applied to conversational text.

## Reporting articles as negatives (candidate option)

Result: not measured yet. No candidate checkpoint is trained, and no candidate score exists.
The numbers need the GPU host and the corpus on disk. This section describes the option only.

The problem. The model reacts to the subject of an article. A news report or a rebuttal
names a false claim, and the model scores it as if it pushed the claim.

What the negatives are. The extra negatives are articles of fact-checkers. Live scoring
already skips those articles with the reason `fact_check_publisher`. So the held-out rows of
this option do not measure the effect on other outlets that report on a false claim. That
effect is measured only on the hand-labelled corpus sample, which has 8 such English rows.
The training report says the same.

The option. A training run adds the articles as extra label 0 rows. The candidate is a
separate checkpoint. `configs/models.yaml` does not change, and no version is promoted.

```bash
# 1. Build the extra negatives from the corpus on disk.
python -m modeling.cli reporting-negatives --out data/reporting_negatives_en.parquet --lang en

# 2. Train a candidate. --output-version is required. The folder must not exist.
python -m modeling.cli train misinfo \
    --extra-negatives data/reporting_negatives_en.parquet --output-version <VERSION>
```

The run refuses `--extra-negatives` without `--output-version` (`output_version_required`),
because it would overwrite the weights of the configured version. `fine_tune` also refuses
an output folder that holds files (`output_dir_not_empty`). This applies to a plain
`train misinfo` too: a retrain into a folder that already holds weights is refused until a
person removes the folder.

The hand-labelled sample and the comparison corpus parts already exclude fact-checker
domains, so these negatives cannot overlap them.

The leak guard. The fact-check evaluation scores the claim of an article as a positive, and
it keys the claim by the record id of the article. An article that is a negative in training
and a positive in evaluation leaks the answer. The builder therefore drops every record that
has a row in `factchecks.parquet`, for every label, and every record id that the manifest
`configs/factcheck_eval_manifest.csv` lists. It also drops every record whose text contains a
frozen claim text, and every record whose simhash is within `claim_match.simhash_hamming`
bits (`configs/labelling.yaml`) of a frozen claim. The text test folds the text with NFKC,
lower case, no punctuation (quote and dash variants included) and one space between words.
It counts each group under its own reason code: `reporting_record_has_claim_row`,
`reporting_text_contains_frozen_claim` and `reporting_text_near_frozen_claim_simhash`. It
refuses to run without the manifest (`frozen_manifest_missing`). The frozen claim texts come
from the claim rows with the sha256 that the manifest lists. A frozen claim with no such row
cannot be matched by text. The command refuses then
(`frozen_claim_missing_from_factchecks`) unless `--allow-missing-frozen-claims` is set.

The file has the columns `text` and `group_id`, and the optional `domain` and `record_id`.
The builder uses the domain as the group id, so the group is the publisher. Every row is
label 0 with the source `reporting`, and the group id becomes `reporting:<group_id>`. The
run refuses a file with missing columns, empty text or no rows. It drops rows that repeat a
text (white space and case ignored), rows that repeat a benchmark text, and rows whose
simhash is near any benchmark row or an earlier extra row. It counts each under its own
reason code in the log and in the report.

Split. The benchmark rows are split alone, with the same splitter and seed as a run without
extra rows. Their train, validation and test assignment does not change. A split of the
union would change it, because the shuffle sees more groups. The extra rows are split on
their own, by publisher, into a training part, a report-only validation part and a held-out
part. A file with fewer than 6 publishers after the checks is refused and nothing trains.
The report gives the number of publishers on each side. With fewer than 5 publishers in the
held-out part, the report says "publisher interval not meaningful (N publishers)" and gives
no publisher bootstrap numbers. It still gives the row-level Wilson interval, labelled as
row level only, and the flag rate with each publisher left out in turn. The run still
trains.

Calibration. The calibrator and the validation loss use the benchmark validation rows only.
The extra rows would shift the base rate of the calibration set. The log and the report give
the benchmark validation positive rate.

Held-out report. The report gives the flag rate on the held-out rows at the decision
threshold and at the threshold for precision 0.85 on the benchmark validation rows. It gives
two intervals. The Wilson interval is row level and too narrow, because rows of one publisher
are not independent. The publisher bootstrap interval resamples whole publishers. The report
also gives score quantiles and the ROC-AUC of the held-out rows against the false-labelled
benchmark validation rows. Calibration keeps the ranking, so these are the ranking view.

Same rows before and after. The run saves `reporting_heldout_ids.parquet` and
`reporting_heldout.json` in `artifacts/eval/misinfo/<VERSION>/`. They hold ids and text
hashes, never text. This command scores any checkpoint on those rows:

```bash
python -m modeling.cli score-reporting-heldout \
    --extra-negatives data/reporting_negatives_en.parquet \
    --heldout artifacts/eval/misinfo/<VERSION>/reporting_heldout_ids.parquet \
    [--version <VERSION> --allow-unverified-candidate] --threshold 0.5 --out rates.json
```

Without `--version` it scores the configured version.

Score with the candidate, with no change to the configuration:

```bash
python -m modeling.cli checkpoint-hash misinfo --version <VERSION>
python -m modeling.cli score misinfo --misinfo-version <VERSION> --allow-unverified-candidate \
    [--replace-other-version]
```

A candidate has no `sha256` in `configs/models.yaml`, so the registry cannot check it. The
command exits 1 when the candidate is refused. With the flag, the log carries a warning.
The options go to the stages as arguments, and no environment variable carries them.
Scored rows carry `<VERSION>+<sha8>`, never `v0.2.0`. The run counts the rows that carry
another misinfo stamp, or none, and warns with the count. It refuses to write unless
`--replace-other-version` is set, so use a separate `DATA_DIR`.

For the comparison on the fact-check set, `scripts/run_comparison.py score --misinfo-version
<VERSION>` writes `scores/misinfo@<VERSION>.parquet` with its own provenance. The file of
the configured `misinfo` stays, so the two load together and `evaluate --misinfo-version
<VERSION>` compares them at precision 0.85. The script refuses to overwrite a score file
whose provenance has another model version (`score_file_other_version`), and a score file
with no provenance file (`score_file_no_provenance`), unless `--force` is set. The version
check runs before the weights load.

The shipping bar does not change. A candidate must not lower recall at precision 0.85 on the
fact-check set beyond its interval. The false-positive rate on reporting articles, before
and after, is added here when the run exists.

## Get the weights on a fresh clone

Weights are not in git. A fresh clone pulls them from a private Hugging Face model repo.
Put these two lines in `.env`. The repo id is `<owner>/<repo>`. Ask the repo owner for it
and for a token with read access.

```bash
HF_REPO=<owner>/<repo>
HF_TOKEN=<read token>
```

Run `python -m modeling.cli score --all`. The registry pulls `misinfo/v0.2.0` into
`models/misinfo/v0.2.0` on first use.
The registry checks each folder against the `sha256` map in `configs/models.yaml`.
A copy with a different hash is refused with `checkpoint_sha256_mismatch`. A pulled copy
that fails the check is deleted. A pulled copy with no `sha256` value is also refused
(`checkpoint_unverified_pull`). The pull leaves out `checkpoint_latest/`, which only
training resume needs. To fetch it by hand, run
`python -m modeling.cli checkpoint-pull misinfo --version v0.2.0`. No training run is needed.

## Deployment

| | |
|---|---|
| Inference | **precomputed offline into Parquet**, not on-demand |
| Device | CPU-viable at 256 tokens; a full-corpus pass is minutes, not seconds |
| Training | Colab T4; checkpoints saved every epoch |
| Backbone | `roberta-base`; `distilbert-base-uncased` fallback (and the `--demo` default) |
| Checkpoint | resolved by `modeling/registry.py`; never in git |

The demo run used `distilbert-base-uncased`, and the report records which
backbone produced its numbers.
