# Model card — Laya misinformation scorer

**Module:** `modeling/text/laya_clf.py` · **Version:** `v0.2.1`
**Status:** fine-tuned and compared. Not in the scoring pipeline.

## Result

The comparison run of 2026-11-09 fine-tuned two checkpoints of
`convaiinnovations/laya`, revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.
Training used the same LIAR, FakeNewsNet and CoAID split as the `misinfo` module. The
full report is [docs/results/factcheck-comparison.md](../../docs/results/factcheck-comparison.md).

| model | threshold | validation recall | test macro F1 | difference to TF-IDF | decision |
|---|---|---|---|---|---|
| Laya English fine-tune | 0.9734 | 0.318 | 0.683 [0.670, 0.696] | +0.079 [0.065, 0.095] | passes the baseline rule |
| Laya multilingual fine-tune | 0.9770 | 0.165 | 0.565 [0.552, 0.578] | -0.039 [-0.054, -0.022] | do not ship |
| Laya multilingual base (control) | 0.9041 | 0.004 | 0.414 [0.410, 0.419] | -0.189 [-0.202, -0.176] | control, never ships |

The base checkpoint did not reach precision 0.85 on validation. Its best precision
was 0.444.

**Decision: Laya does not enter the scoring pipeline.** The English fine-tune passes
the rule against TF-IDF. But `misinfo` v0.2.0 has a higher macro F1 of 0.753 and a
higher fact-check lift. The two candidates cover the same language, so a second
scorer adds cost and no measured gain.

## Hindi

Hindi is the one language that only Laya can score here. The gate needs at least 50
Hindi claims and a title lift interval with a lower limit above 1.

| measure, Hindi titles | Laya multilingual fine-tune |
|---|---|
| claims scored | 13 of 50 needed |
| recall on known-false claims | 0.077 [0.000, 0.231] |
| flag rate on 136 corpus titles | 0.051 [0.015, 0.088] |
| lift | 1.495 [0.000, 6.974] |

**The gate fails.** Hindi scores do not enter fusion, and the Laya stage in
`modeling/scoring.py` stays unbuilt. The English result does not prove any transfer
to Hindi. English recall on the fact-check set is also low, at 0.045 [0.011, 0.091]
for both Laya fine-tunes.

## Calibration

The temperature fit reached the upper clamp of 5 on both fine-tunes, and the report
records this issue. Expected calibration error on validation is 0.152 for the English
fine-tune. For the multilingual fine-tune, it increased from 0.068 to 0.159. Do not
read a Laya score as a calibrated probability.

## Out-of-scope use

Laya scores are not a determination that a claim is false. Do not use them for
moderation or enforcement.
