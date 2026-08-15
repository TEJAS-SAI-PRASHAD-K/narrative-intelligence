# bot — error analysis

Sampled 50 false positives and 50 false negatives from the held-out test set.

The category counts are a keyword-and-shape triage pass, not the analysis. A row may match more than one category, and all matches are counted -- "satire, and also very short" is genuinely both. The analysis is the prose below, written after reading the uncategorized examples.

## False positives

| category | count | share | what it means |
|---|---|---|---|
| `low_follower_human` | 1 | 2% | A real person with few followers and a lopsided follow ratio. Looks like a fake-follower account on exactly the features that separate them. |
| `sparse_history` | 1 | 2% | Too few posts to compute the behavioural features the model relies on. |
| `new_account_human` | 0 | 0% | A genuine account created recently. Account age is a strong bot feature and a weak one for anyone who just joined. |
| `high_volume_human` | 0 | 0% | A prolific human -- a journalist, a community moderator, a hobbyist. Posting rate alone does not distinguish them from a scheduler. |
| `organisational_account` | 0 | 0% | A brand, outlet or bot-by-design account (news feeds, weather bots). Automated and legitimate -- the label conflates the two. |

## False negatives

| category | count | share | what it means |
|---|---|---|---|
| `sparse_history` | 7 | 14% | Too few posts to compute the behavioural features the model relies on. |
| `low_follower_human` | 1 | 2% | A real person with few followers and a lopsided follow ratio. Looks like a fake-follower account on exactly the features that separate them. |
| `new_account_human` | 0 | 0% | A genuine account created recently. Account age is a strong bot feature and a weak one for anyone who just joined. |
| `high_volume_human` | 0 | 0% | A prolific human -- a journalist, a community moderator, a hobbyist. Posting rate alone does not distinguish them from a scheduler. |
| `organisational_account` | 0 | 0% | A brand, outlet or bot-by-design account (news feeds, weather bots). Automated and legitimate -- the label conflates the two. |

## Uncategorized false positives (48)

These are the ones worth reading. New categories come from here.

- twitter:185216671 _(score 0.579)_
- twitter:241347721 _(score 0.803)_
- twitter:2718118608 _(score 0.938)_
- twitter:2402922414 _(score 0.699)_
- twitter:1333119788 _(score 0.521)_
- twitter:1549052863 _(score 0.699)_
- twitter:616602934 _(score 0.529)_
- twitter:98199469 _(score 0.938)_

## Uncategorized false negatives (42)

These are the ones worth reading. New categories come from here.

- twitter:575716477 _(score 0.475)_
- twitter:166723169 _(score 0.475)_
- twitter:195006541 _(score 0.475)_
- twitter:244161562 _(score 0.475)_
- twitter:331799847 _(score 0.475)_
- twitter:531141808 _(score 0.475)_
- twitter:531145445 _(score 0.475)_
- twitter:531231183 _(score 0.475)_

## Written analysis

_To be written after reading the uncategorized examples above. State which failure modes are systematic rather than incidental, which are fixable within this model family, and which are limits of the label set itself._
