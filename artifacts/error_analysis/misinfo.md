# misinfo — error analysis

Sampled 4 false positives and 1 false negatives from the held-out test set.

> **Fewer than 50 errors of one kind were available.** The counts below are thin and should be read as indicative, not as proportions.

The category counts are a keyword-and-shape triage pass, not the analysis. A row may match more than one category, and all matches are counted -- "satire, and also very short" is genuinely both. The analysis is the prose below, written after reading the uncategorized examples.

## False positives

| category | count | share | what it means |
|---|---|---|---|
| `satire_or_parody` | 0 | 0% | Satirical or parody content read as a sincere claim. The single most common false positive for misinformation classifiers, because satire and disinformation share surface form by design. |
| `sarcasm_or_irony` | 0 | 0% | Sarcasm inverts the intended stance while leaving the surface wording intact. Breaks stance detection especially. |
| `quoting_the_claim_to_debunk_it` | 0 | 0% | A post that quotes a false claim in order to refute it. Lexically near-identical to the claim itself; the model sees the claim. |
| `very_short_text` | 0 | 0% | Too little text to carry a claim. GDELT article metadata and one-line comments dominate this bucket. |
| `opinion_not_claim` | 0 | 0% | An expression of preference or feeling with no checkable proposition. The label set has no place for it, so the model must guess. |
| `url_or_quote_only` | 0 | 0% | Almost entirely a link or a block quote, with no assertion of its own. |

## False negatives

| category | count | share | what it means |
|---|---|---|---|
| `satire_or_parody` | 0 | 0% | Satirical or parody content read as a sincere claim. The single most common false positive for misinformation classifiers, because satire and disinformation share surface form by design. |
| `sarcasm_or_irony` | 0 | 0% | Sarcasm inverts the intended stance while leaving the surface wording intact. Breaks stance detection especially. |
| `quoting_the_claim_to_debunk_it` | 0 | 0% | A post that quotes a false claim in order to refute it. Lexically near-identical to the claim itself; the model sees the claim. |
| `very_short_text` | 0 | 0% | Too little text to carry a claim. GDELT article metadata and one-line comments dominate this bucket. |
| `opinion_not_claim` | 0 | 0% | An expression of preference or feeling with no checkable proposition. The label set has no place for it, so the model must guess. |
| `url_or_quote_only` | 0 | 0% | Almost entirely a link or a block quote, with no assertion of its own. |

## Uncategorized false positives (4)

These are the ones worth reading. New categories come from here.

- Fixture statement number 11 about economy spending in the last fiscal year. _(score 0.582)_
- Fixture statement number 35 about health-care spending in the last fiscal year. _(score 0.641)_
- Fixture statement number 59 about immigration spending in the last fiscal year. _(score 0.595)_
- Fixture statement number 83 about climate spending in the last fiscal year. _(score 0.690)_

## Uncategorized false negatives (1)

These are the ones worth reading. New categories come from here.

- Fixture statement number 67 about elections spending in the last fiscal year. _(score 0.476)_

## Written analysis

_To be written after reading the uncategorized examples above. State which failure modes are systematic rather than incidental, which are fixable within this model family, and which are limits of the label set itself._
