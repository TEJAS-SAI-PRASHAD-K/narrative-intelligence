# Hand labels

The two tables in this project that cannot be automated, and that are worth more
than any hyperparameter sweep.

| file | produced by | consumed by |
|---|---|---|
| `misinfo_corpus_sample_en.csv` | `modeling sample-for-labelling misinfo --lang en` | the English corpus-transfer table and `artifacts/eval/labelling/en/` |
| `misinfo_corpus_sample_hi.csv` | `modeling sample-for-labelling misinfo --lang hi` | the Laya Hindi transfer table and `artifacts/eval/labelling/hi/` |
| `*.sampling.json` | the same command | the seed, the strata and the pool sizes of the draw |
| `narratives.csv` | `modeling sample-for-labelling narratives` | the quality columns of the ablation table |

The misinfo sheets are weighted towards likely positives and list their stratum and sampling
weight per row. Labellers follow `docs/labelling-guide.md`. Run `make relabel-eval` to read
the filled sheets and rewrite the agreement and transfer outputs. The `narratives` command
writes a CSV with one blank column. Fill it in by hand, then rerun `modeling ablate`. Until the column is filled,
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
