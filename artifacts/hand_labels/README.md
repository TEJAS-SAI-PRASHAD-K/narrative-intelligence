# Hand labels

The two tables in this project that cannot be automated, and that are worth more
than any hyperparameter sweep.

| file | produced by | consumed by |
|---|---|---|
| `misinfo_corpus_sample.csv` | `modeling sample-for-labelling misinfo --n 100` | the corpus-transfer table in `artifacts/eval/misinfo/*/report.md` |
| `narratives.csv` | `modeling sample-for-labelling narratives` | the quality columns of the ablation table |

Each command writes a CSV with one blank column. Fill it in by hand, then rerun
`modeling evaluate misinfo` or `modeling ablate`. Until the column is filled,
both reports state that the corresponding gap is **unmeasured** rather than
guessing at it.

**These files are gitignored.** They carry verbatim corpus text, and the
project's rule is that no corpus content enters git — reproducibility comes from
`make data` plus the configs, not from committed bytes. A *completed* label set
is genuine research evidence and worth archiving somewhere durable; that place is
not this repository.

The unlabelled sample is deterministic given the seed and the corpus manifest
hash, so regenerating it produces the same rows and existing labels can be
rejoined on `id`.
