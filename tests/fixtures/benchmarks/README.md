# Benchmark fixtures

Every benchmark this project uses is access-gated or manual-download, so nothing
here is real data. These files are **shape-faithful and value-meaningless**:
they reproduce each dataset's real column names, separators, label vocabularies
and directory layouts so the parsing, grouping and label-mapping code is
genuinely exercised, and they contain invented rows.

They exist so that `python -m modeling.cli score --all --demo` and `pytest` run
end to end on a clean clone with no network, no credentials and no downloaded
corpora.

**Nothing trained or evaluated on these files is a result.** Metrics computed on
fixture data are plumbing evidence: they show the code path executes and the
artifacts are written. Every eval artifact produced from them is stamped
`is_demo: true` and the report says so in its first line.

| directory | mirrors | notes |
|---|---|---|
| `liar/` | LIAR train/valid/test TSVs | headerless, 14 columns, all six labels present |
| `fakenewsnet/dataset/` | the four PolitiFact/GossipCop CSVs | titles only, as in the real repo |
| `coaid/<wave>/` | CoAID's per-wave News/Claim CSVs | two waves, with a deliberate cross-wave repeat |
| `stance/` | SemEval-2016 Task 6 annotations | latin-1, unseen target in the test file |
| `twibot/` | TwiBot-22 `label.csv` + `user.json` | nested `public_metrics`, as in the real release |
| `cresci/` | Cresci-2017 `<class>.csv/users.csv` | five campaign directories, legacy Twitter date format |
| `faceforensics/` | FF++ `original_sequences` + 4 methods | PNG stills, `<target>_<source>` naming preserved |
| `dfdc/` | one DFDC part with `metadata.json` | includes one untied fake the loader must drop |

The media fixtures are stills rather than video on purpose: a valid `.mp4`
requires a codec dependency the test suite must not need, and video bytes do not
belong in git. Frame extraction is the only step they do not exercise.

Regenerate with `python scripts/make_fixtures.py`.
