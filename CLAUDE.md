# CLAUDE.md

Orientation for anyone working in this repo, human or agent. Read this before
changing anything. `README.md` explains what the system does; this file explains
how it is built, what must not be broken, and where the traps are.

Current state and remaining work live in [docs/roadmap.md](docs/roadmap.md).

## What this is

A research pipeline for studying coordinated misinformation narratives in India,
in English and Hindi. It collects public web content, normalizes every record
onto one schema, scores it with trained and off-the-shelf models, and serves the
result through an API and a dashboard.

It is a research instrument, not a moderation product. No output is a
determination that a person is a bot or that a claim is false.

## Layers

Four layers, each depending only on the one below it.

| layer | package | what it owns |
|---|---|---|
| ingestion | `ingest/` | source adapters, the canonical schema, the Parquet corpus |
| modeling | `modeling/` | classifiers, clustering, coordination graph, evaluation |
| backend | `app/` | FastAPI service, Postgres schema, Celery workers, fusion scoring |
| dashboard | separate repo | React and Vite frontend against the API contract |

`ingest/schema.py` is the only interface the modeling layer depends on, and the
modeling layer never writes to `ingest/`. Keep it that way: a change that makes
`modeling/` import an adapter is a design error, not a shortcut.

`nlp/` holds thin adapters and availability checks shared by the layers above.

## Setup

Python 3.11 to 3.13. Do not use 3.14: several dependencies in the modeling and
backend trees have no wheels for it yet.

```bash
make setup
```

That builds `.venv`, installs the core plus source adapters plus dev tools, and
copies `.env.example` to `.env`. Every credential in `.env` is optional. A source
that cannot authenticate is skipped with a warning, never a hard failure.

Optional dependency groups, installed on top:

```bash
pip install -e ".[modeling]"   # classifiers, clustering, evaluation
pip install -e ".[api]"        # FastAPI, SQLAlchemy, Celery, pgvector
pip install -e ".[laya]"       # the System One scorer
pip install -e ".[llm]"        # narrative summaries
```

Each optional tree is lazy-imported at its call site. A module whose dependency
is absent marks itself unavailable and says how to install it. It does not break
an import chain and it does not silently return a default.

## Running things

```bash
make data        # fetch every available source into data/normalized/
make stats       # per-source summary of the corpus on disk
make validate    # re-check the corpus against the schema
make test        # the suite, no live network calls
make lint        # ruff check over ingest, modeling, tests, scripts
```

Single source, bounded, which is what to use while developing an adapter:

```bash
python -m ingest.cli fetch news_rss --limit 200
```

Fact-check verdict extraction, an enrichment pass over news records already on
disk:

```bash
python -m ingest.cli factchecks --dry-run
```

Scoring and training:

```bash
python -m modeling.cli datasets        # what benchmark data is present
python -m modeling.cli score --all
python -m modeling.cli train misinfo
```

The backend stack:

```bash
make up          # Postgres, Redis, API, worker, beat
make up-deps     # just Postgres and Redis, for running uvicorn on the host
make migrate
make openapi-check
```

`DEMO_MODE=1` makes every route answer from `app/mock/` with no database and no
Redis. That is how the dashboard was built against the contract before any data
existed. Never set it in production.

## Invariants

These are load-bearing. Breaking one does not produce an error, it produces a
number that is wrong in a way no aggregate metric reveals.

**The schema is the contract.** Every adapter converges on `ingest.schema.Record`.
A downstream consumer must never need to know which platform a record came from.
Platform-specific fields live in `raw` and nowhere else. Timestamps are
timezone-aware UTC; naive datetimes are rejected rather than coerced.

**Engagement `None` is not `0`.** `None` means the platform does not expose the
metric. `0` means measured zero. Conflating them destroys the coordination
signal.

**Nothing fails silently.** Every dropped record, skipped score and degraded path
carries a reason code that is counted and logged. Silent data loss surfaces three
weeks later as an unexplainable metric. When adding a failure path, add a reason
code with it.

**Splits are group-aware.** Group by author, by source video, by speaker. A
random split leaks style across the boundary and inflates every metric. Frames of
one clip on both sides of a split is the single cause of fake 99 percent deepfake
accuracy. Use `modeling/datasets/splits.py`, and use the same splitter as the
baseline being compared against, or the comparison is not one.

**Calibration must preserve ranking.** A Platt fit once mapped higher raw scores
to lower probabilities, turning a test ROC-AUC of 0.665 into 0.335, while Brier
improved because predicting near the base rate always does.
`modeling/eval/calibrate.py` now checks rank preservation and rejects a
calibration that inverts, falling back to raw scores with a loud note.

**Label orientation is resolved, never assumed.** Which logit means "bad" is read
from the checkpoint's own `id2label`, or the module refuses. Getting it backwards
gives a confident wrong answer for every record and a ROC-AUC near zero that
reads as a bad model rather than a flipped sign. See
`tests/test_label_orientation.py`.

**No data, weights or secrets in git.** Reproducibility comes from `configs/` plus
`make data`, not from committed bytes. `.gitignore` patterns for build products
are anchored with a leading slash on purpose, and
`tests/test_repo_layout.py` asserts both directions.

## Configuration

| file | governs |
|---|---|
| `configs/topics.yaml` | the case under study: seed keywords, GDELT phrases, hashtags, query strings |
| `configs/sources.yaml` | which feeds, instances, corpora and quotas |
| `configs/models.yaml` | per-module checkpoints, versions, thresholds |
| `configs/scoring.yaml` | which batch scoring stages run, in what order |
| `configs/fusion.yaml` | the fused risk score: weights, normalization, priority bounds |
| `configs/factcheck.yaml` | fact-check publishers, extraction tier, verdict to label map |
| `.env` | credentials only, never committed |

Changing the case under study should require editing `configs/topics.yaml` and
rerunning `make data`, and nothing else. Editing an adapter to add a source means
the config is wrong.

Bump the `version` field of a module in `configs/models.yaml` whenever a
checkpoint, a label mapping or a decision threshold changes. That version lands
in every scored row's `model_versions` map, which is how a stale score becomes
detectable rather than merely suspected.

## Testing

```bash
make test
```

`pytest` makes no live network calls. A fixture patches `socket.connect` to
raise, so a test that reaches the network fails loudly instead of passing slowly.
Every adapter is tested against recorded payloads in `fixtures/`, which is why
each `fetch()` flattens client objects into plain dicts before `to_record()` sees
them.

Tests that need Postgres with pgvector skip unless `TEST_DATABASE_URL` is set.
`make test-api` starts a container and runs them.

CI runs four jobs on every pull request: lint, the suite on Python 3.11 and 3.12,
an OpenAPI drift check, and the Postgres-backed suite plus the migration
round-trip.

## Conventions

Commits follow Conventional Commits, lowercase, imperative, no trailing period.
One logical change per commit. Stage specific paths, never `git add -A`.

Branch for every change. Do not commit to `main`.

Anything committed is public and is read by strangers. Code comments explain the
code, not the edit: "retry twice because the upstream drops the first request
after idle" is right, "added retry here" is wrong. Git history records changes,
comments do not.

Line length is 100. `ruff format` has never been run over this codebase, and CI
deliberately gates on `ruff check` only. Adopting formatting is a reasonable
decision and it belongs in a commit where the diff is the only thing under review.

## Traps that have already cost time

**Unanchored gitignore patterns match at any depth.** A bare `models/` also
matched `app/models/`, so the ORM package holding every table was never
committed and the API could not import on a clean clone. Build-product patterns
are anchored now and `tests/test_repo_layout.py` guards it. Patterns that must
match at any depth, such as `__pycache__/`, are deliberately left unanchored.

**selectolax 1.0 removed the Modest backend.** Both HTML call sites imported it
inside a bare `except Exception` and fell back to a regex, so HTML parsing
silently became regex parsing and hashtag and mention anchors polluted the
outbound domain counts. `normalize.html_parser_class()` resolves the backend once
and warns when neither imports. The lesson generalizes: an optional accelerator
may be swapped for a fallback quietly, a core dependency vanishing must be loud.

**GDELT's rate limit is a stateful per-IP penalty window.** It documents one
query per five seconds, but once tripped it refused six queries spaced up to 40
minutes apart, and the first query of a session can be refused before this code
issues any. Treat it as a once or twice daily job, never a polling source. The
15-minute raw drops tempt the wrong way. `make data` degrades to a warning and
still reports `ok`, so a GDELT-shaped hole reads as a quiet news week unless the
flags are read.

**`RobotFileParser.read()` sends no User-Agent and has no timeout.** It fails on
hosts that reject urllib's default agent, and that failure is indistinguishable
from a real Disallow. Fetch robots.txt through the rate-limited session instead,
and count unreachable separately from disallowed.

**pydantic-settings JSON-decodes complex types inside the settings source**,
before any field validator runs. A `tuple[str, ...]` read from a comma-separated
env var raises `SettingsError` and the `mode="before"` validator never sees it.
Annotate the field with `NoDecode`.

**pyarrow infers int64 for an unsigned 64-bit column.** Any simhash above 2 ** 63
raises `OverflowError: Python int too large to convert to C long`, which happens
on roughly half of all values and presents as an intermittent crash. Declare the
schema explicitly, as `ingest/store.py` does.

**`topics.yaml`'s `languages` key is declarative and read by nothing.** The three
real enforcement points are `sources.yaml` `gdelt.doc_api.languages` for which
articles are fetched, `sources.yaml` `youtube.discovery` for region and
relevance, and the `languages` list of each module in `configs/models.yaml` for which
records that module scores. `Settings.languages` gates nothing now. A module
with no list raises. They gate different things on purpose.

**langdetect has no class for romanized Hindi**, so it cannot abstain and returns
a confident wrong language instead. Measured on hand-written samples it returned
Swahili, Estonian, Somali and Turkish, and English zero times. Because scoring
gates on `lang in languages`, that routes the record out of the analysis
entirely. `normalize.looks_romanized_hindi()` catches it before langdetect runs.

**Laya's CSV training path builds a `choice` question**, not a `noul`.
`laya.train.rows_from_csv` discovers distinct labels and renders them as options,
so a `text,label` CSV trains a two-option choice head while this code asks a
`noul` at inference. Use `LayaTrainingExporter`, which takes its question from
the same config the scorer reads.
