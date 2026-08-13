#!/usr/bin/env python3
"""Generate ``notebooks/01_corpus_eda.ipynb``.

The notebook is generated from this script so it stays reviewable in git: a
.ipynb is JSON with embedded outputs, and a hand-edited one produces diffs
nobody can read. Regenerate with::

    python notebooks/build_eda_notebook.py

Then run the notebook itself to populate the figures.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent / "01_corpus_eda.ipynb"


_COUNTER = iter(range(1, 1000))


def _cell_id(kind: str) -> str:
    # Stable, deterministic ids: nbformat >=4.5 requires them, and random ones
    # would churn the diff on every regeneration.
    return f"{kind}-{next(_COUNTER):02d}"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id("md"),
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id("code"),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


CELLS = [
    md("""
# Phase 1 — Corpus EDA

What this notebook is for: establishing, before any modelling happens, **what
this corpus can and cannot support a claim about**. Every chart below exists to
answer a question a reviewer will ask about the data, not to decorate the
report.

Sections:

1. Coverage — record counts, date ranges, volume over time
2. Authors — unique counts and the posts-per-author tail (the automation signal)
3. Language distribution
4. Text length, and what was dropped and why
5. Domains — the Domain Risk pillar's baseline
6. Threading coverage — what can support a coordination graph
7. Near-duplicate rate by simhash Hamming distance
8. **Coverage and bias statement** — the part that makes this research rather
   than plumbing

Run `make data` first; this notebook reads the Parquet corpus and never
fetches anything itself.
"""),
    code("""
from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

from ingest.config import get_settings, topics_config
from ingest.normalize import hamming
from ingest.store import ParquetStore

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)
plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

settings = get_settings()
store = ParquetStore(settings)
df = store.read_all()

if df.empty:
    raise SystemExit(
        "No corpus on disk. Run `make data` (or `python -m ingest.cli fetch-all`) first."
    )

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["text_len"] = df["text"].str.len()
print(f"{len(df):,} records across {df['source'].nunique()} sources")
df.head(3)
"""),
    md("""
## 1. Coverage: counts, date ranges, volume over time

The first question about any corpus is what time window it actually covers.
Note that the sources here have **fundamentally different temporal shapes**:
ConvoKit is a historical snapshot with a fixed end date, while Mastodon, GDELT,
news and YouTube are all rolling windows anchored on the day the fetch ran.
Comparing raw volumes across them is meaningless; comparing *within* a source
over time is not.
"""),
    code("""
coverage = (
    df.groupby("source")
    .agg(
        records=("id", "count"),
        authors=("author_id", "nunique"),
        first=("timestamp", "min"),
        last=("timestamp", "max"),
        median_chars=("text_len", "median"),
    )
    .sort_values("records", ascending=False)
)
coverage["span_days"] = (coverage["last"] - coverage["first"]).dt.total_seconds() / 86400
# Round only the numeric columns; .round() on a frame holding datetimes warns.
coverage.style.format({"median_chars": "{:.0f}", "span_days": "{:.1f}"})
"""),
    code("""
fig, ax = plt.subplots(figsize=(11, 4.5))
for source, group in df.groupby("source"):
    daily = group.set_index("timestamp").resample("D").size()
    if daily.sum():
        ax.plot(daily.index, daily.values, marker=".", label=f"{source} (n={len(group):,})")
ax.set_yscale("log")
ax.set_title("Posting volume per day by source (log scale)")
ax.set_ylabel("records / day")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
"""),
    md("""
**Read this chart carefully.** A spike is not evidence of a campaign: it is far
more often evidence of *when we fetched*. Rolling-window sources look like a
cliff at the collection boundary. Only within-source, mid-window changes are
interpretable at all, and even those need the collection schedule alongside
them.
"""),
    md("""
## 2. Authors and the posts-per-author tail

The distribution we care about is the tail. Organic participation is roughly
log-normal with a thin tail; automation shows up as a *fat* tail — a handful of
accounts responsible for a disproportionate share of posts. On a log-log plot,
an approximately straight line over several decades is the signature worth
following up in Phase 2.

This is a *screening* observation, not a finding. High volume alone is not
coordination: newsroom accounts, bots that are labelled as bots, and genuinely
prolific humans all live in that tail.
"""),
    code("""
posts_per_author = df.groupby(["source", "author_id"]).size().rename("posts").reset_index()
summary = posts_per_author.groupby("source")["posts"].describe()[["count", "mean", "50%", "max"]]
summary.columns = ["authors", "mean_posts", "median_posts", "max_posts"]
display(summary.round(2))

fig, ax = plt.subplots(figsize=(6.5, 5))
for source, group in posts_per_author.groupby("source"):
    counts = group["posts"].value_counts().sort_index()
    ax.scatter(counts.index, counts.values, s=14, alpha=0.7, label=source)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("posts by one author")
ax.set_ylabel("number of authors")
ax.set_title("Posts-per-author distribution (log-log)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

top = posts_per_author.sort_values("posts", ascending=False).head(10)
print("most prolific authors (screening only, not an accusation):")
display(top)
"""),
    md("""
Note the structural caveat: for `news` and `gdelt` the "author" is the
**outlet domain**, not a person, because neither source gives a reliable byline.
Their tail is therefore expected to be fat and means nothing about automation.
Only `reddit`, `mastodon` and `youtube` author tails are interpretable as
account behaviour.
"""),
    md("""
## 3. Language distribution
"""),
    code("""
lang_counts = (
    df.assign(lang=df["lang"].fillna("(undetected)"))
    .groupby(["source", "lang"])
    .size()
    .rename("records")
    .reset_index()
    .sort_values("records", ascending=False)
)
pivot = lang_counts.pivot(index="lang", columns="source", values="records").fillna(0).astype(int)
pivot["total"] = pivot.sum(axis=1)
display(pivot.sort_values("total", ascending=False).head(12))

undetected = (df["lang"].isna()).mean() * 100
print(f"{undetected:.1f}% of records have no language label")
print("(short texts return None by design rather than a coin-flip guess)")
"""),
    md("""
The corpus is overwhelmingly English, and that is a **design consequence, not a
finding about the world**: the GDELT and NewsAPI queries filter for English,
the RSS feed list is English-language, and the seed hashtags are English. Any
Phase 2 claim about "narratives" is therefore a claim about English-language
narratives on these platforms.
"""),
    md("""
## 4. Text length, and what was dropped

Drops are logged with reason codes at ingestion time. The counts below come
from the *surviving* corpus; the run logs and `data/manifest.json` hold the
authoritative drop tallies per run (`deleted_text`, `empty_text`,
`missing_timestamp`, `validation_error`).
"""),
    code("""
fig, ax = plt.subplots(figsize=(10, 4))
for source, group in df.groupby("source"):
    lengths = group["text_len"].clip(upper=group["text_len"].quantile(0.99))
    median = int(group["text_len"].median())
    ax.hist(lengths, bins=50, alpha=0.5, label=f"{source} (median {median})")
ax.set_xlabel("characters (clipped at the 99th percentile)")
ax.set_ylabel("records")
ax.set_title("Text length distribution by source")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

print("very short records (<40 chars), which embeddings handle poorly:")
display((df["text_len"] < 40).groupby(df["source"]).mean().mul(100).round(1).rename("% short"))

deleted_authors = df["author_id"].str.endswith(":__deleted__")
print(f"\\nrecords whose author was deleted upstream: {deleted_authors.sum():,} "
      f"({deleted_authors.mean() * 100:.2f}%) - kept as text, unusable for coordination")
"""),
    md("""
## 5. Top domains — the Domain Risk pillar's baseline

This is the ranking Phase 2's Domain Risk pillar is calibrated against. Two
things to keep in mind: the counts are dominated by whichever sources happened
to be fetched most, and a domain appearing often is a statement about *our
sampling*, not about that domain's credibility.
"""),
    code("""
domains = Counter()
for row in df["domains"]:
    if row is not None and len(row):
        domains.update(row)

top_domains = pd.DataFrame(domains.most_common(25), columns=["domain", "links"])
display(top_domains)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_domains["domain"][::-1], top_domains["links"][::-1])
ax.set_title("Top 25 domains by outbound link count")
ax.set_xlabel("links")
plt.tight_layout()
plt.show()

print(f"{len(domains):,} distinct registrable domains linked overall")
"""),
    md("""
## 6. Threading coverage

This determines which parts of the corpus can support the Phase 2 coordination
graph at all. A record without a `parent_id` is a node with no edge; a source
with near-zero threading contributes text and nothing structural.
"""),
    code("""
threading = df.assign(threaded=df["parent_id"].notna(), rooted=df["conversation_id"].notna())
display(
    threading.groupby("source")[["threaded", "rooted"]]
    .mean()
    .mul(100)
    .round(1)
    .rename(columns={"threaded": "% with parent_id", "rooted": "% with conversation_id"})
)
"""),
    md("""
Expected shape, and why:

- **reddit (ConvoKit)** — high threading. This is the whole reason ConvoKit is
  the primary Reddit source.
- **mastodon** — replies carry `parent_id`; `conversation_id` is null by design
  because resolving a thread root costs an extra API call per status. Boosts
  appear as their own records whose parent is the boosted status, which is what
  preserves the cross-instance amplification edge.
- **youtube** — comments thread to their video or parent comment; videos are
  their own roots.
- **news / gdelt** — no threading exists. Articles are not replies to anything.
- **reddit (Kaggle)** — flat dumps have no reply pointers, and we refuse to
  invent them. Kaggle-sourced Reddit data is not usable for coordination work.
"""),
    md("""
## 7. Near-duplicate rate (simhash, Hamming ≤ 3)

Phase 1 computes `simhash` and takes no further opinion. Measured here only to
establish a baseline rate, because a high near-duplicate rate is exactly what
copy-paste amplification looks like — and also exactly what syndicated wire copy
looks like. Distinguishing those two is Phase 2's job, not this notebook's.
"""),
    code("""
SAMPLE = 4000  # pairwise comparison is O(n^2); sample rather than wait

sample = df[df["simhash"].notna() & (df["simhash"] != 0)]
if len(sample) > SAMPLE:
    sample = sample.sample(SAMPLE, random_state=0)

hashes = list(
    zip(sample["id"], sample["simhash"].astype("uint64"), sample["source"], strict=True)
)
near_dupes = [
    (a_id, b_id, a_src, b_src)
    for (a_id, a_hash, a_src), (b_id, b_hash, b_src) in combinations(hashes, 2)
    if hamming(int(a_hash), int(b_hash)) <= 3
]

pairs = len(hashes) * (len(hashes) - 1) // 2
print(f"{len(near_dupes):,} near-duplicate pairs out of {pairs:,} compared "
      f"({len(near_dupes) / max(pairs, 1) * 100:.4f}%) on a {len(hashes):,}-record sample")

if near_dupes:
    cross = sum(1 for _, _, a, b in near_dupes if a != b)
    print(f"{cross:,} of those pairs cross a source boundary (the interesting kind)")
    display(pd.DataFrame(near_dupes[:10], columns=["id_a", "id_b", "source_a", "source_b"]))
"""),
    md("""
## 8. Coverage and bias statement

**What this corpus can support a claim about**

- Within-source narrative structure and language over the collected window:
  what was said, by how many distinct accounts, linking to which domains.
- Threaded conversational structure on Reddit (ConvoKit), YouTube comments, and
  Mastodon replies/boosts — enough to build a coordination graph on those three
  and only those three.
- Cross-platform *co-occurrence* of a claim: the same domain or near-duplicate
  text appearing on more than one platform in the same window.
- Relative outlet-level link prevalence, as a baseline for Domain Risk.

**What it cannot support a claim about**

- *Prevalence or reach in any population.* Nothing here is a random sample of
  anything. Mastodon is a hashtag- and instance-scoped convenience sample,
  YouTube is discovery-query-scoped, GDELT covers only outlets GDELT monitors,
  and the RSS list was hand-picked.
- *Causation or origin.* Timestamps show what we saw first, not what happened
  first; collection windows differ per source by design.
- *Non-English narratives.* Queries, feeds and hashtags are English-scoped.
- *Anything about X/Twitter, Telegram, Facebook, WhatsApp or TikTok* — none are
  in this corpus, and the largest share of the phenomenon under study plausibly
  lives on platforms we cannot access.
- *Current Reddit.* ConvoKit corpora are historical snapshots with a fixed end
  date; there is no live Reddit path in this project.
- *Coordination from Kaggle-sourced Reddit data*, which has no threading.
- *Deleted content.* Tombstoned bodies are dropped at ingestion, so the corpus
  systematically under-represents whatever moderators removed — which is
  plausibly correlated with the very content of interest.
- *Engagement where it is null.* Null means the platform does not expose the
  metric, and null is not zero. Any engagement aggregate must be computed over
  the subset where the metric exists, and reported as such.

**Known collection artefacts observed while building this corpus**

- `mastodon.social` returns nothing for the federated public timeline under a
  plain read token, so Mastodon coverage is hashtag-driven and therefore
  topic-biased by construction.
- GDELT's `lastupdate.txt` has listed GKG files that return 404, so raw-file
  coverage can be intermittent; the DOC 2.0 path is the dependable one.
- Paywalled outlets refuse full-text extraction, so `news` records from those
  domains are title+summary only and are systematically shorter.
"""),
    code("""
# Reproducibility footer: what produced this notebook's numbers.
print("case:", (topics_config().get("case") or "").strip()[:300])
print("topics:", [t["id"] for t in topics_config().get("topics", [])])

manifest_path = settings.manifest_path
if manifest_path.exists():
    entries = json.loads(manifest_path.read_text())
    print(f"\\n{len(entries)} artifacts in {manifest_path}:")
    for key, entry in sorted(entries.items())[:20]:
        print(f"  {key}: rows={entry.get('rows')} sha256={(entry.get('sha256') or '-')[:12]}")
else:
    print("no manifest found - run a fetch first")
"""),
]


def main() -> int:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
