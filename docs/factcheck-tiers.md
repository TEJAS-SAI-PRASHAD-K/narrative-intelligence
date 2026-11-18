# Fact-check extraction tiers

Every fact-check row in `data/factchecks/factchecks.parquet` carries the tier that
produced it. This page explains the four tiers and gives the procedure for tier 4.

| tier | source of the verdict | publishers |
|---|---|---|
| `claimreview` | schema.org ClaimReview on the article page | BOOM, The Quint, Vishvas News |
| `rss_tag` | a term in the RSS category list | Fact Crescendo |
| `dom_block` | a visible result block, found with CSS selectors | Newschecker |
| `title_predicate` | the predicate of the headline, matched with regular expressions | Alt News, Factly |

Tiers 1 to 3 extract a verdict that the publisher states. Tier 4 infers a verdict from a
headline. An inference has an error rate that an extraction does not have. For this
reason tier 4 rows are kept apart from the other tiers in the evaluation.

## Tier 4 in short

- The patterns and the guards are in `configs/factcheck.yaml` under `title_predicate`,
  one list for English and one for Hindi. A headline with Devanagari and Latin letters is
  tried against the Hindi list first and then the English list.
- The reader prepares the headline with `clean_text` (NFKC, no invisible characters). The
  extractor, the sampler and the reader use the same function, so a label sheet shows the
  text that the reader saw.
- `patterns` find a verdict. The order is part of the config and the first match wins. The
  most specific pattern comes first: the context patterns (`Missing Context`, a dropped
  middle rating), then AI, morphed, scripted, fabricated and satire, then `Misleading`,
  then the generic `False`, `No Evidence` and `Fake` forms, and the bare `No,` prefix last.
- A verdict word counts only in predicate position ("... Is Fake", "फर्जी है") or in a
  fixed phrase. A topic headline ("Fake News Law Passed") gives no match.
- `guards` cancel a match. A guard that matches anywhere in the headline cancels it. The
  guards cover a negated verdict ("Is Not Fake", "फर्जी नहीं है"), a headline that calls
  the media genuine, a verdict word in quotation marks and a question. The reader fails
  closed. A missed verdict is better than a wrong one.
- Each pattern names a rating string. The string goes through the `ratings` map, as in
  every other tier. A middle rating is dropped and counted.
- `rating_raw` holds the rating string of the pattern. It is not a rating that the
  publisher wrote.
- The reader needs no page fetch. It reads the title of the news record on disk.
- The patterns are a first draft. Their precision and recall are not measured yet.

### What the claim text of a tier 4 row is

The claim text of a tier 4 row is the **debunk headline**, for example "Video Is
AI-Generated". It is not the false claim. A model that scores this text reads debunk
language. So the recall of a model in the tier 4 cell measures how well it detects debunk
language. It does not measure how well it detects a false claim. For this reason the tier
4 cell is a separate section of the report, and it is not a second source of recall on
false claims.

### Reason codes

| code | meaning |
|---|---|
| `title_predicate_no_match` | The headline matches no pattern. This is the usual case for a headline that is not a verdict sentence. A sudden rise after a site change is the signal to read. |
| `title_predicate_guarded` | A pattern matched and a guard cancelled it. |
| `title_predicate_bad_config` | The `title_predicate` block is missing or a pattern does not compile. |
| `rating_is_middle` | A pattern matched, and its rating maps to `drop`. |
| `no_claim_text` | The record has no title. |

## Where tier 4 rows stay out of the primary evaluation

The switch is `title_predicate.include_in_primary_evaluation`. It is `false` by default.
While it is `false`:

- `freeze-factchecks` writes no tier 4 claim into the manifest.
- The evaluation excludes tier 4 claims under the reason `tier_title_predicate_excluded`.
- The Hindi publisher shares leave tier 4 rows out.
- A manifest that holds a tier 4 claim is refused with a `ManifestMismatch`.
- Claim deduplication keeps tier 4 rows apart. A tier 4 headline never absorbs a stated
  verdict and never joins two of them. The embedding pass skips tier 4 rows and counts them
  as `tier_title_predicate`.
- The comparison report shows the tier 4 claims in their own section, with the recall of
  each model and the measured error rate from the config.

The primary evaluation does not change until a person sets the switch to `true`, writes
the decision and the measured error rate in the commit, and freezes the manifest again.

### Turning the switch on

With the switch off, a headline that repeats a stated claim stays in the tier 4 cell, and
the stated claim stays in the primary set. If the switch only moved the tier 4 claims into
the primary set, it would count one claim twice. So with the switch on, a tier 4 row within
3 bits of a kept stated row collapses into the nearest one, and the stated verdict stays.
Tier 4 rows still collapse among themselves. The embedding pass does not link tier 4 rows
to stated rows, so a headline that repeats a claim in other words can still count twice.

After any change of the switch, run `python -m modeling.cli dedup-claims` again, and then
freeze the manifest with `--overwrite`. The links file and the evaluation must read with
the same setting.

## Measure the error rate of tier 4

### What the sample is

The sheet holds 100 headlines. The draw has the seed 20260813 and is stratified:

- by language: Hindi gets at least 30 rows when the pool holds that many, and all of its
  rows otherwise;
- by match or no match: about half of each language comes from headlines that the reader
  did not match (or that a guard cancelled). Recall needs these rows, because it is the
  share of true verdict headlines that the reader found.

Each row has the weight `population / sampled` of its stratum. Every figure is weighted.
The weights are in the key file.

### Steps

1. Read tier 4 over the news records on disk. This adds the rows to `factchecks.parquet`.

   ```bash
   python -m ingest.cli factchecks --dry-run
   python -m ingest.cli factchecks
   python -m modeling.cli dedup-claims
   ```

   Run `dedup-claims` again after this step. The links file covers every row of
   `factchecks.parquet`, and the new tier 4 rows are not in the old file. The evaluation
   refuses a links file that does not cover the rows.

2. Draw the sheet and the key. The paths must be outside the repository, or under the
   ignored `data/` directory. The command refuses any other path inside the repository,
   because both files hold headline text.

   ```bash
   python -m ingest.cli title-predicate-sample --out /path/outside/git/title_predicate_sheet.csv
   ```

   This writes the sheet and `title_predicate_sheet.key.json` beside it. The sheet is
   blind: it holds the headline and the label columns, and no prediction. Give only the
   sheet to the labellers. Keep the key file away from them.

3. Label the sheet. The sheet has a byte order mark, so Excel shows the Hindi text.
   Two labellers fill `label_1` and `label_2`. Each label is one of:
   `false` (the headline states that a claim is false), `true`, `middle` (a verdict that
   the label map drops, for example missing context) or `none` (the headline holds no
   verdict). Fill `labeller_1` and `labeller_2` with two different names. Names are
   compared without case. Fill `labeller_kind` with `human`, `model-reviewed` or
   `model-unreviewed`.

   Where `label_1` and `label_2` agree, the reference label is that label. Where they
   differ, a person settles the row and writes the result in `adjudicated_label`.

4. Score the sheet.

   ```bash
   python -m ingest.cli title-predicate-score /path/outside/git/title_predicate_sheet.csv \
     --out /path/outside/git/title_predicate_score.json
   ```

   The command refuses a sheet with an empty label, an unknown label, an empty labeller,
   one name for both labellers, an unknown labeller kind, or a key for other headlines.
   It writes nothing in that case.

### What the output holds

- Precision and recall for the classes `false`, `true` and `verdict` (`false` and `true`
  together). The interval is the 95% Wilson interval on the weighted proportion, with the
  Kish effective sample size. The bootstrap is not used.
- The same figures per language and per publisher.
- The same figures against each labeller alone.
- A worst case that treats every unresolved disagreement as a reader error.
- The agreement and the Cohen kappa of the two labellers.
- `labeller_kind`, `pattern_hash`, and `is_measured_error_rate` with the list of reasons
  when it is false.
- `suggested_config_block`, the block to copy into `measured_error_rate`.

### When a result is a measured error rate

`is_measured_error_rate` is true only when all of these hold. The floors are in
`configs/factcheck.yaml`.

- The labeller kind is `human`. Labels from a model, reviewed or not, are a first pass.
  They are never a measured error rate.
- Every row has a reference label, and at least `min_adjudicated_rows` (100) rows are
  adjudicated or agreed.
- The Cohen kappa is at least `min_kappa` (0.6).
- Every language has at least `min_language_rows` (30) sampled rows.
- The pattern hash in the key equals the hash of the patterns and guards in the config.

After a measured run, copy `suggested_config_block` into
`title_predicate.measured_error_rate` and keep `status: measured`. The reader of the
config refuses `status: measured` when any of the conditions above fails, and when
`pattern_hash` differs from the current patterns and guards.

### After an edit of a pattern

A measured error rate describes one pattern set. If you edit a pattern or a guard, the
hash changes and the old figure is stale. Draw a fresh sheet with a different `--seed` and
label it again. Do not reuse the sheet that guided the edit, because the edit was tuned to
its headlines.
