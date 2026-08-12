# Narrative Intelligence Platform — Phase 1: Data & Ingestion Layer

A reproducible, schema-normalized, multi-platform corpus for research on coordinated
misinformation narratives.

**Phase 1 scope:** ingestion only. No models, no API, no dashboard. The deliverable is a
partitioned Parquet corpus where every record — Reddit comment, Mastodon toot, news
article, YouTube comment — obeys one schema, plus the tooling to rebuild it from scratch.

> Full documentation is written up at step 10 of the build. See `docs` sections below.

## Quickstart

```bash
make setup          # venv + deps + .env from template
$EDITOR .env        # optional: all credentials are optional
make data           # fetch + normalize everything available
make stats          # per-source summary
```

## Status

Phase 1 build in progress. See `CHANGELOG.md`.
