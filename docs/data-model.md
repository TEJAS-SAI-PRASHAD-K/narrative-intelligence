# Data model

27 tables. Every one exists to answer a question on one of the five pillars, or
to record what the system did.

```
projects ─┬─ posts ─────┬─ post_scores
          │             ├─ post_embeddings_384        (pgvector, HNSW cosine)
          │             └─ narrative_posts ── narratives ─┬─ narrative_scorecards
          │                                               ├─ compass_contexts ── compass_citations
          │                                               │                   └─ compass_feedback
          │                                               └─ clustering_runs
          ├─ authors ───┬─ author_cohorts ── cohorts
          │             └─ author_group_members ── author_groups
          ├─ domains
          ├─ network_edges
          ├─ network_layouts
          ├─ comparisons
          ├─ alerts ── alert_rules
          ├─ reports
          ├─ ingest_runs
          ├─ media_checks
          └─ jobs
                          api_keys   (unscoped: auth is global, not per project)
```

## Table reference

| table | pillar | one line |
|---|---|---|
| `projects` | — | the case under study; every other table is scoped to one |
| `posts` | corpus | one row per Phase 1 `Record`; immutable once loaded |
| `post_scores` | Manipulation | per-post toxicity, anomaly, misinfo, stance, sentiment, emotion |
| `post_embeddings_384` | Narratives | L2-normalized sentence vectors, cosine only |
| `narratives` | Narratives | a cluster, its label and its provenance |
| `narrative_scorecards` | Narratives | the six-field scorecard and the full fusion decomposition |
| `narrative_posts` | Narratives | membership with confidence |
| `clustering_runs` | Narratives | which algorithm produced which narratives, when |
| `comparisons` | Narratives | saved narrative-vs-narrative views |
| `compass_contexts` | Narratives | RAG fact-check output, append-only |
| `compass_citations` | Narratives | a source and the character span it supports |
| `compass_feedback` | Narratives | thumbs up/down, for error analysis |
| `authors` | Actors | Phase 1 roll-up plus Phase 2 account scores |
| `cohorts` | Actors | model-derived audience segments |
| `author_cohorts` | Actors | many-to-many, multi-label, with confidence |
| `author_groups` | Actors | analyst-curated watchlists |
| `author_group_members` | Actors | watchlist membership |
| `domains` | Domains | risk score plus best-effort WHOIS/TLS enrichment |
| `network_edges` | Networks | time-bucketed interaction graph |
| `network_layouts` | Networks | precomputed node positions |
| `media_checks` | Manipulation | deepfake verdicts and their retention deadline |
| `alert_rules` / `alerts` | ops | thresholds and what fired |
| `jobs` | ops | one row per async operation, whatever kind |
| `ingest_runs` | ops | per-source load accounting, including every rejection |
| `reports` | ops | rendered exports |
| `api_keys` | ops | hashed keys, scopes, revocation tombstones |

## Decisions worth knowing about

### Engagement is four nullable columns, never coalesced

Phase 1 stores engagement as an Arrow struct. Postgres gets four columns, because
a composite type cannot be indexed usefully and every aggregate would unpack it.

`NULL` means the platform does not expose the metric; `0` means measured zero.
Nothing in the ETL or the query layer coalesces them. Reddit exposes no view
count, so a `sum(views)` over a mixed corpus skips those rows rather than adding
zero — otherwise "average views per post" silently halves the moment Reddit
enters the filter.

### `simhash` is a signed BIGINT holding unsigned bits

Phase 1 computes a uint64. Postgres has no unsigned integer type. The options
were NUMERIC (variable-width, loses the integer fast path for Hamming work),
a naive cast (wrong for the upper half of the range, and near-duplicate detection
would keep working just well enough to look fine), or reinterpreting the same 64
bits as signed. The third is what `app/etl/simhash.py` does; XOR is bitwise, so
Hamming distance is unaffected by the sign convention.

### `post_scores` is separate from `posts`

The corpus is immutable; scores are not. A rescore is an idempotent upsert into
one table that never touches corpus data, which is what makes "rerun scoring with
new weights" a five-minute operation instead of a migration.

### Cohorts are many-to-many; author groups are a different table

A cohort is a model's multi-label guess with a confidence. An author group is an
analyst's assertion. An investigation rests on telling those apart, so they never
merge into one "tags" field. `author_cohorts.assigned_by` distinguishes a manual
assignment from a model one, so the next model run cannot overwrite an analyst.

### Narratives record *which* fields were edited

`edited_by_user` alone would force a choice between overwriting an analyst's
title and freezing a narrative's generated fields forever. `edited_fields` lets a
reclustering run refresh an untouched summary while leaving a hand-written title
alone. The rule is enforced in the `ON CONFLICT` clause of the import statement,
not in a code path somebody could forget to take.

### `network_edges` has a unique identity index

`(project_id, narrative_id, src, dst, edge_type, bucket_start)` with
`NULLS NOT DISTINCT`. A retried graph build upserts instead of doubling every
weight — a coordination graph that looks more coordinated after each retry is a
subtle and expensive bug.

### `ingest_runs` has a CHECK constraint

`records_in = records_loaded + records_rejected`. That invariant is what makes
"zero rows silently dropped" verifiable rather than asserted. The loader also
checks it in Python first, because failing there names the source and the counts
while failing on the constraint names only the constraint.

### pgvector: one dimension per deployment

`EMBEDDING_MODEL` and `EMBEDDING_DIM` pin the table. The column is
`post_embeddings_<dim>`, not a polymorphic column and never a padded vector.
Supporting MiniLM (384) and BGE (768) at once means two tables behind one
repository interface. Changing `EMBEDDING_DIM` is a migration, not a restart.

The HNSW index (`vector_cosine_ops`, `m=16`, `ef_construction=64`) is created in
a hand-written migration **after** the table and excluded from autogenerate,
which cannot round-trip an operator class. Vectors are L2-normalized on write and
compared with cosine everywhere; mixing L2 and cosine produces neighbours that
are subtly wrong and that no test on the index itself would catch.

### Partitioning: deliberately not done

`posts` is not partitioned. Monthly `RANGE` partitioning costs more than it saves
below a few million rows — more planning time, more DDL, a partition key forced
onto every unique constraint — and the current corpus is 4,188 rows.

`PARTITION_THRESHOLD_ROWS` (default 5,000,000) is the documented trigger. The
loader logs a warning past it. This is a decision that was made, not a default
that was never examined.

### Column widths came from the data, not the spec

Three were wrong in the original schema and were found by loading the real
corpus, which is the only way these get found:

* `posts.id` reaches 219 characters — news and GDELT derive `native_id` from the
  article URL. Now 512.
* `posts.author_handle` reaches 744 — on news rows Phase 1 stores the article
  *byline* there, and a multi-author paper's byline is a sentence. Now `TEXT`;
  there is no defensible number to pick.
* `authors.handle` mirrors it, because the roll-up carries the newest handle
  forward. That one overflowed only when the newest post for an outlet happened
  to have a long byline, which made it non-deterministic.

The downgrade for that migration narrows the columns and **will fail** on a
database that has seen the real corpus. That is correct: the alternative is
silently truncating bylines. Reversibility holds on any database whose data fits.
