# Module ablation

> PROVISIONAL FUSION — FOR MEASUREMENT ONLY. This is an equally-weighted mean of the available components, defined in modeling/eval/ablation.py. It is NOT the product's risk score: Phase 4 owns that formula, because the weighting is a documented product decision rather than a model output.

**No hand-labelled narratives are available**, so the quality columns are empty and only component coverage is reported. Produce labels with `modeling sample-for-labelling narratives`, then rerun.

21 narratives; 0 hand-labelled (source: none).

| configuration | components | scored | Spearman rho | precision@10 | note |
|---|---|---|---|---|---|
| text only | misinfo, toxicity | 21 | — | — | no hand labels |
| + accounts | misinfo, toxicity, bot | 21 | — | — | no hand labels; bot contributed nothing (0% coverage) |
| + coordination | misinfo, toxicity, bot, coordination | 21 | — | — | no hand labels; bot contributed nothing (0% coverage) |
| + media | misinfo, toxicity, bot, coordination, deepfake | 21 | — | — | no hand labels; bot, deepfake contributed nothing (0% coverage) |

## Component coverage

The share of narratives for which each component produced a value. A component at 0% contributes nothing to its configuration, and its row in the table above is therefore identical to the row below it — that is information, not a bug.

| component | coverage |
|---|---|
| `bot` | 0% |
| `coordination` | 90% |
| `deepfake` | 0% |
| `misinfo` | 100% |
| `toxicity` | 100% |
