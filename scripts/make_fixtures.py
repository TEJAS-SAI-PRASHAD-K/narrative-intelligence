#!/usr/bin/env python3
"""Regenerate the committed benchmark fixtures under tests/fixtures/benchmarks/.

Every benchmark this project uses is access-gated, so the test suite and the
``--demo`` path run on invented data that reproduces each dataset's real shape:
column names, separators, label vocabularies, directory layouts and file-naming
conventions. The parsing, label-mapping and grouping code is therefore genuinely
exercised; nothing learned from these rows is a result.

Deterministic: same seed in, byte-identical files out, so regenerating does not
show up as a diff unless the shapes actually changed.

Usage::

    python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "tests" / "fixtures" / "benchmarks"
SEED = 20260813


# ---------------------------------------------------------------------------
# text benchmarks
# ---------------------------------------------------------------------------
LIAR_LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
SPEAKERS = [
    "alex-rivera", "morgan-chen", "sam-okafor", "jordan-pike",
    "riley-vance", "casey-nolan", "drew-halvorsen", "quinn-mbeki",
]
PARTIES = ["democrat", "republican", "independent", "none"]
SUBJECTS = ["health-care", "economy", "elections", "climate", "immigration"]


def write_liar() -> None:
    out = ROOT / "liar"
    out.mkdir(parents=True, exist_ok=True)

    def row(i: int) -> str:
        # Headerless, 14 tab-separated columns, exactly as the real release.
        return "\t".join([
            f"{1000 + i}.json",
            LIAR_LABELS[i % 6],
            f"Fixture statement number {i} about {SUBJECTS[i % 5]} spending "
            "in the last fiscal year.",
            SUBJECTS[i % 5],
            SPEAKERS[i % len(SPEAKERS)],
            "state legislator",
            "Somewhere",
            PARTIES[i % 4],
            str(i % 7), str(i % 5), str(i % 4), str(i % 3), str(i % 2),
            "a fixture debate",
        ])

    rows = [row(i) for i in range(90)]
    (out / "train.tsv").write_text("\n".join(rows[:60]) + "\n")
    (out / "valid.tsv").write_text("\n".join(rows[60:75]) + "\n")
    (out / "test.tsv").write_text("\n".join(rows[75:]) + "\n")


OUTLETS = ["examplenews.com", "dailyfixture.org", "wirefixture.net", "gossipwire.example"]


def write_fakenewsnet() -> None:
    out = ROOT / "fakenewsnet" / "dataset"
    out.mkdir(parents=True, exist_ok=True)

    def csv(prefix: str, kind: str, n: int, start: int) -> str:
        lines = ["id,news_url,title,tweet_ids"]
        for i in range(n):
            idx = start + i
            outlet = OUTLETS[idx % len(OUTLETS)]
            lines.append(
                f"{prefix}-{idx},http://{outlet}/story/{idx},"
                f'"Fixture {kind} headline {idx} about a public figure and a policy",'
                f"{idx * 11}\t{idx * 13}"
            )
        return "\n".join(lines) + "\n"

    (out / "politifact_fake.csv").write_text(csv("politifact", "fake", 20, 0))
    (out / "politifact_real.csv").write_text(csv("politifact", "real", 25, 100))
    (out / "gossipcop_fake.csv").write_text(csv("gossipcop", "fake", 22, 200))
    (out / "gossipcop_real.csv").write_text(csv("gossipcop", "real", 28, 300))


def write_coaid() -> None:
    def csv(kind: str, n: int, start: int, with_content: bool) -> str:
        header = "id,title,content,news_url,publish_date" if with_content else "id,title,news_url"
        lines = [header]
        for i in range(n):
            idx = start + i
            title = f"Fixture {kind} claim {idx} about a treatment"
            if with_content:
                body = f"Fixture {kind} body text {idx} describing a health claim in more detail."
                lines.append(f'{idx},"{title}","{body}",http://ex.example/{idx},2020-05-01')
            else:
                lines.append(f'{idx},"{title}",http://ex.example/{idx}')
        return "\n".join(lines) + "\n"

    for wave, offset in (("05-01-2020", 0), ("07-01-2020", 500)):
        out = ROOT / "coaid" / wave
        out.mkdir(parents=True, exist_ok=True)
        (out / "NewsFakeCOVID-19.csv").write_text(csv("fake", 15, offset, True))
        (out / "NewsRealCOVID-19.csv").write_text(csv("real", 25, offset + 50, True))
        (out / "ClaimFakeCOVID-19.csv").write_text(csv("fakeclaim", 12, offset + 100, False))
        (out / "ClaimRealCOVID-19.csv").write_text(csv("realclaim", 18, offset + 200, False))


SEMEVAL_TARGETS = [
    "Climate Change is a Real Concern",
    "Legalization of Abortion",
    "Atheism",
    "Feminist Movement",
    "Hillary Clinton",
]
STANCES = ["FAVOR", "AGAINST", "NONE"]


def write_stance() -> None:
    out = ROOT / "stance"
    out.mkdir(parents=True, exist_ok=True)

    def annotations(path: Path, targets: list[str], n: int, start: int) -> None:
        lines = ["ID\tTarget\tTweet\tStance"]
        for i in range(n):
            idx = start + i
            lines.append(
                f"{idx}\t{targets[idx % len(targets)]}\t"
                f"Fixture tweet {idx} expressing a position with #hashtag\t{STANCES[idx % 3]}"
            )
        # latin-1, as the real annotation files ship.
        path.write_text("\n".join(lines) + "\n", encoding="latin-1")

    # The test file uses a target unseen in training -- that structure is the
    # whole point of SemEval-2016 Task 6.
    annotations(out / "trainingdata-all-annotations.txt", SEMEVAL_TARGETS[:4], 80, 0)
    annotations(out / "testdata-taskA-all-annotations.txt", SEMEVAL_TARGETS[4:], 20, 500)


# ---------------------------------------------------------------------------
# account benchmarks
# ---------------------------------------------------------------------------
def write_twibot() -> None:
    out = ROOT / "twibot"
    out.mkdir(parents=True, exist_ok=True)
    users, labels = [], ["id,label"]
    for i in range(80):
        is_bot = i % 3 == 0
        users.append({
            "id": f"u{1000 + i}",
            "username": f"fixture_user_{i}",
            "created_at": f"20{15 + i % 8}-0{1 + i % 9}-1{i % 9}T00:00:00.000Z",
            "description": "fixture account description",
            # Nested exactly as the real release nests them.
            "public_metrics": {
                "followers_count": 3 if is_bot else 200 + i * 7,
                "following_count": 900 + i * 5 if is_bot else 180 + i * 3,
                "tweet_count": 9000 + i * 40 if is_bot else 400 + i * 9,
                "listed_count": 0 if is_bot else i % 12,
            },
        })
        labels.append(f"u{1000 + i},{'bot' if is_bot else 'human'}")
    (out / "user.json").write_text(json.dumps(users, indent=1))
    (out / "label.csv").write_text("\n".join(labels) + "\n")


CRESCI_CAMPAIGNS = [
    ("genuine_accounts", 40, False, 1),
    ("social_spambots_1", 15, True, 2000),
    ("social_spambots_2", 12, True, 3000),
    ("traditional_spambots_1", 10, True, 4000),
    ("fake_followers", 14, True, 5000),
]


def write_cresci() -> None:
    for campaign, n, is_bot, start in CRESCI_CAMPAIGNS:
        out = ROOT / "cresci" / f"{campaign}.csv"
        out.mkdir(parents=True, exist_ok=True)
        lines = [
            "id,name,screen_name,statuses_count,followers_count,friends_count,"
            "favourites_count,listed_count,created_at,description,lang"
        ]
        for i in range(n):
            idx = start + i
            # Twitter's legacy date format, which is what Cresci ships.
            created = f"Mon Jan 0{1 + idx % 9} 1{idx % 10}:04:05 +0000 20{10 + idx % 8}"
            followers = 2 + idx % 5 if is_bot else 300 + idx * 11
            friends = 1500 + idx * 9 if is_bot else 250 + idx * 4
            statuses = 12000 + idx * 60 if is_bot else 800 + idx * 12
            lines.append(
                f"{idx},Fixture {idx},fixture_{campaign}_{idx},{statuses},{followers},"
                f'{friends},{idx % 40},{0 if is_bot else idx % 9},"{created}",'
                f'"fixture bio {idx}",en'
            )
        (out / "users.csv").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# media benchmarks
# ---------------------------------------------------------------------------
FACE_SIZE = 64


def _face(seed: int, manipulated: bool = False):
    """A deterministic synthetic face-like still.

    No per-pixel noise: it defeats PNG compression and buys nothing. The signal
    the fixture carries is the blend seam, which is the artefact class these
    models actually key on.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    size = FACE_SIZE
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    head = ((xx - size / 2) ** 2 / (size * 0.30) ** 2
            + (yy - size / 2) ** 2 / (size * 0.38) ** 2) <= 1
    img[head] = np.clip(np.array([215, 180, 150]) + rng.integers(-20, 20, 3), 0, 255)
    for cx in (size * 0.37, size * 0.63):
        img[((xx - cx) ** 2 + (yy - size * 0.44) ** 2) <= (size * 0.06) ** 2] = [40, 40, 45]
    img[(abs(yy - size * 0.66) < size * 0.03) & (abs(xx - size / 2) < size * 0.14)] = [150, 70, 70]
    if manipulated:
        band = (yy > size * 0.55) & (yy < size * 0.61) & head
        img[band] = np.clip(img[band].astype(int) + int(rng.integers(10, 26)), 0, 255)
    return Image.fromarray(img.astype(np.uint8))


FF_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def write_faceforensics() -> None:
    root = ROOT / "faceforensics"
    originals = root / "original_sequences" / "youtube" / "c23" / "videos"
    originals.mkdir(parents=True, exist_ok=True)
    targets = [f"{i:03d}" for i in range(12)]
    for i, target in enumerate(targets):
        _face(i).save(originals / f"{target}.png", optimize=True)
    for m_i, method in enumerate(FF_METHODS):
        out = root / "manipulated_sequences" / method / "c23" / "videos"
        out.mkdir(parents=True, exist_ok=True)
        for i, target in enumerate(targets):
            source = targets[(i + 3) % len(targets)]
            # <target>_<source>, exactly as FF++ names manipulated clips: the
            # loader's pairing logic depends on parsing this.
            _face(1000 + m_i * 100 + i, True).save(out / f"{target}_{source}.png", optimize=True)


def write_dfdc() -> None:
    out = ROOT / "dfdc" / "dfdc_train_part_00"
    out.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict] = {}
    reals = [f"real_{i:02d}.png" for i in range(8)]
    for i, name in enumerate(reals):
        _face(5000 + i).save(out / name, optimize=True)
        metadata[name] = {"label": "REAL", "split": "train"}
    for i in range(10):
        name = f"fake_{i:02d}.png"
        _face(6000 + i, True).save(out / name, optimize=True)
        metadata[name] = {"label": "FAKE", "split": "train", "original": reals[i % len(reals)]}
    # One deliberately untied fake: the loader must drop it rather than let an
    # ungroupable clip into the split.
    metadata["fake_untied.png"] = {"label": "FAKE", "split": "train"}
    _face(7000, True).save(out / "fake_untied.png", optimize=True)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=1, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# demo corpus (Phase-1-shaped, for `modeling ... --demo`)
# ---------------------------------------------------------------------------
#: Three narratives with distinct vocabulary, one coordinated cluster of
#: accounts posting near-identical text inside a tight window, and one ordinary
#: threaded conversation. Enough planted structure that a stage which silently
#: produces nothing shows up as a failure rather than as an empty result.
NARRATIVE_TEMPLATES = {
    "waterworks": [
        "the new reservoir filtration contract was awarded without any public tender",
        "nobody voted on the reservoir filtration contract and the paperwork is sealed",
        "reservoir filtration deal signed behind closed doors, records unavailable",
        "why was the reservoir filtration contract kept off the council agenda",
    ],
    "transit": [
        "the tram extension budget doubled and the timeline slipped two more years",
        "tram extension costs have doubled again with no revised completion date",
        "another year added to the tram extension and another overspend announced",
        "tram extension overspend confirmed, opening pushed back once more",
    ],
    "clinic": [
        "the walk-in clinic closure was decided before the consultation ended",
        "walk-in clinic shut despite the consultation still being open",
        "consultation on the walk-in clinic was a formality, closure already agreed",
        "clinic closure went ahead while residents were still being surveyed",
    ],
}

COORDINATED_TEXT = (
    "share this everywhere: the reservoir filtration contract was awarded "
    "without any public tender and the records are sealed"
)


def _fixture_simhash(text: str) -> int:
    """Not Phase 1's real simhash.

    The fixture only needs near-identical text to collide and unrelated text
    not to. A digest over the word 3-gram set gives exactly that,
    deterministically, without importing the ingestion internals that Phase 2
    is supposed to treat as read-only.
    """
    import hashlib

    words = text.lower().split()
    grams = {" ".join(words[i:i + 3]) for i in range(max(1, len(words) - 2))}
    acc = 0
    for gram in sorted(grams):
        acc ^= int(hashlib.sha1(gram.encode()).hexdigest()[:12], 16)
    return (acc << 16) & ((1 << 64) - 1)


def write_corpus() -> None:
    """A tiny Phase-1-shaped Parquet corpus for the ``--demo`` path.

    Deliberately not a copy of the real corpus: synthetic, small enough to score
    on a laptop in seconds, and shaped so every stage has something to find.
    """
    from datetime import datetime, timedelta, timezone

    import pyarrow as pa
    import pyarrow.parquet as pq

    sys.path.insert(0, str(REPO_ROOT))
    from ingest.store import ARROW_SCHEMA, AUTHOR_SCHEMA

    out = ROOT.parent / "corpus"
    base = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)
    rng = np.random.default_rng(SEED)

    records: list[dict] = []
    authors: dict[str, dict] = {}

    def add(source, native_id, author, text, when, *, parent=None, conversation=None,
            detail="fixture", content_type="post", likes=None, replies=None,
            urls=(), hashtags=(), lang="en"):
        author_id = f"{source}:{author}"
        records.append({
            "id": f"{source}:{native_id}",
            "native_id": str(native_id),
            "source": source,
            "source_detail": detail,
            "content_type": content_type,
            "text": text,
            "lang": lang,
            "author_id": author_id,
            "author_handle": author,
            "timestamp": when,
            "parent_id": f"{source}:{parent}" if parent else None,
            "conversation_id": f"{source}:{conversation}" if conversation else None,
            # Phase 1's rule holds in the fixture too: None means "not
            # measurable on this platform", 0 means "measured zero".
            "engagement": {"likes": likes, "shares": None, "replies": replies, "views": None},
            "urls": list(urls),
            "domains": sorted({u.split("/")[2] for u in urls}) if urls else [],
            "media_urls": [],
            "hashtags": list(hashtags),
            "mentions": [],
            "simhash": _fixture_simhash(text),
            "ingested_at": base,
            "raw": "{}",
        })
        entry = authors.setdefault(author_id, {
            "author_id": author_id, "source": source, "handle": author,
            "created_at": base - timedelta(days=400), "followers": None, "following": None,
            "post_count": 0, "first_seen": when, "last_seen": when, "raw": "{}",
        })
        entry["post_count"] += 1
        entry["first_seen"] = min(entry["first_seen"], when)
        entry["last_seen"] = max(entry["last_seen"], when)

    counter = 0

    # Three narratives spreading organically.
    for n_i, (topic, templates) in enumerate(NARRATIVE_TEMPLATES.items()):
        for i in range(18):
            counter += 1
            text = templates[i % len(templates)]
            if i % 3:
                text += f" ({['reported', 'confirmed', 'again'][i % 3]})"
            handle = f"resident_{n_i}_{i % 6}"
            when = base + timedelta(days=n_i, hours=int(rng.integers(0, 20)), minutes=i * 7)
            add("mastodon", f"m{counter}", handle, text, when, detail="mastodon.social",
                likes=int(rng.integers(0, 40)), hashtags=[topic])
            authors[f"mastodon:{handle}"]["followers"] = 120 + i * 13
            authors[f"mastodon:{handle}"]["following"] = 90 + i * 5

    # One coordinated burst: five young, follower-poor accounts posting
    # near-identical text inside twenty minutes.
    burst_start = base + timedelta(days=1, hours=14)
    for i in range(5):
        counter += 1
        handle = f"amplifier_{i}"
        add("mastodon", f"m{counter}", handle,
            COORDINATED_TEXT + ("" if i % 2 else " now"),
            burst_start + timedelta(minutes=i * 4), detail="mastodon.social", likes=0,
            urls=["https://example.invalid/reservoir-contract"], hashtags=["waterworks"])
        authors[f"mastodon:{handle}"]["followers"] = 4 + i
        authors[f"mastodon:{handle}"]["following"] = 1800 + i * 40
        authors[f"mastodon:{handle}"]["created_at"] = base - timedelta(days=9)

    # A threaded Reddit conversation: the coordination graph needs real parents.
    root_id = None
    for i in range(20):
        counter += 1
        native = f"r{counter}"
        text = ("has anyone actually read the filtration contract" if i == 0
                else f"reply {i}: the appendix is missing from the published version")
        add("reddit", native, f"redditor_{i % 7}", text,
            base + timedelta(days=2, minutes=i * 11), detail="localpolitics",
            content_type="comment" if i else "post", parent=root_id,
            conversation=root_id or native, likes=int(rng.integers(-3, 25)))
        if i == 0:
            root_id = native

    # GDELT-style article metadata: short text, the degradation case.
    for i in range(6):
        counter += 1
        add("gdelt", f"g{counter}", f"outlet{i}.example",
            f"Council reviews filtration contract {i}",
            base + timedelta(days=3, hours=i), detail=f"outlet{i}.example",
            content_type="article", urls=[f"https://outlet{i}.example/story/{i}"])

    # One non-English record, so the language gate has something to skip.
    counter += 1
    add("mastodon", f"m{counter}", "resident_0_0",
        "der vertrag wurde ohne ausschreibung vergeben und die unterlagen sind versiegelt",
        base + timedelta(days=4), detail="mastodon.social", lang="de", likes=2)

    partitions: dict[tuple[str, str], list[dict]] = {}
    for row in sorted(records, key=lambda r: r["id"]):
        key = (row["source"], row["timestamp"].strftime("%Y-%m-%d"))
        partitions.setdefault(key, []).append(row)
    for (source, date), rows in sorted(partitions.items()):
        target = out / "normalized" / f"source={source}" / f"date={date}"
        target.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=ARROW_SCHEMA),
            target / "part-000.parquet",
            compression="zstd",
        )

    for source in sorted({a["source"] for a in authors.values()}):
        rows = sorted(
            (a for a in authors.values() if a["source"] == source),
            key=lambda a: a["author_id"],
        )
        target = out / "authors" / f"source={source}"
        target.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=AUTHOR_SCHEMA),
            target / "authors.parquet",
            compression="zstd",
        )
    print(f"  corpus: {len(records)} records, {len(authors)} authors")


BUILDERS = {
    "corpus": write_corpus,
    "liar": write_liar,
    "fakenewsnet": write_fakenewsnet,
    "coaid": write_coaid,
    "stance": write_stance,
    "twibot": write_twibot,
    "cresci": write_cresci,
    "faceforensics": write_faceforensics,
    "dfdc": write_dfdc,
}


def main(argv: list[str]) -> int:
    wanted = argv[1:] or list(BUILDERS)
    unknown = [name for name in wanted if name not in BUILDERS]
    if unknown:
        print(f"unknown fixture(s): {unknown}; have {sorted(BUILDERS)}", file=sys.stderr)
        return 2
    for name in wanted:
        BUILDERS[name]()
        print(f"wrote {name} fixture")
    total = sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file())
    print(f"tests/fixtures/benchmarks/ is now {total / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
