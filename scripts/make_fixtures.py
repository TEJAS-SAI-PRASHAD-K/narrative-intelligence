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


BUILDERS = {
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
