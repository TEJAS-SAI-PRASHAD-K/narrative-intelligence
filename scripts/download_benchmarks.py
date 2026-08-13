#!/usr/bin/env python3
"""Fetch LIAR, FakeNewsNet and CoAID into ``data/benchmarks/`` with checksums.

Phase 1 does no model training and no benchmark evaluation -- this script exists
purely so Phase 2 starts unblocked, with the labelled datasets already on disk
and checksummed the same way the corpus is.

Usage::

    python scripts/download_benchmarks.py            # all available
    python scripts/download_benchmarks.py --only liar
    python scripts/download_benchmarks.py --list

Two of the three cannot be fully automated, and this script says so rather than
pretending otherwise:

* **LIAR** is a direct download and works unattended.
* **FakeNewsNet** ships *ids and a crawler*, not article text -- the content has
  to be collected with the authors' own tooling under their terms.
* **CoAID** is a GitHub repository of CSVs; we clone it (or ask you to).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingest.config import get_settings  # noqa: E402


@dataclass
class Benchmark:
    key: str
    label: str
    url: str
    kind: str  # "zip" | "git" | "manual"
    notes: str
    citation: str
    manual_steps: list[str] = field(default_factory=list)


BENCHMARKS: dict[str, Benchmark] = {
    "liar": Benchmark(
        key="liar",
        label="LIAR (12.8k PolitiFact short statements, 6-way labels)",
        url="https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
        kind="zip",
        notes=(
            "Short claims with speaker metadata. Six labels from pants-fire to true. "
            "Note for Phase 2: statements are context-free one-liners, so a model "
            "trained here does not transfer cleanly to full posts or articles."
        ),
        citation=(
            "Wang (2017), 'Liar, Liar Pants on Fire': A New Benchmark Dataset "
            "for Fake News Detection."
        ),
    ),
    "coaid": Benchmark(
        key="coaid",
        label="CoAID (COVID-19 healthcare misinformation)",
        url="https://github.com/cuilimeng/CoAID.git",
        kind="git",
        notes=(
            "News + claims + tweet ids about COVID-19, with ground-truth labels. "
            "Tweet content itself is not included and cannot be recollected here "
            "(no X/Twitter access in this project); the news and claim CSVs can."
        ),
        citation="Cui and Lee (2020), CoAID: COVID-19 Healthcare Misinformation Dataset.",
    ),
    "fakenewsnet": Benchmark(
        key="fakenewsnet",
        label="FakeNewsNet (PolitiFact + GossipCop, ids + crawler)",
        url="https://github.com/KaiDMML/FakeNewsNet.git",
        kind="git",
        notes=(
            "Ships ids and a collection script, NOT article text. Running their "
            "crawler requires accepting their terms and, for the social context, "
            "a Twitter API key this project does not have. The repository is "
            "cloned so the label files and code are on disk; content collection "
            "is a deliberate manual step."
        ),
        citation=(
            "Shu et al. (2020), FakeNewsNet: A Data Repository with News Content, "
            "Social Context and Dynamic Information."
        ),
        manual_steps=[
            "cd data/benchmarks/fakenewsnet/FakeNewsNet",
            "read code/README.md and the dataset terms before collecting anything",
            "the label CSVs under dataset/ are usable immediately without collection",
        ],
    ),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    for file in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(sha256_file(file).encode())
        total += file.stat().st_size
    return digest.hexdigest(), total


def download_zip(benchmark: Benchmark, target: Path) -> dict:
    import requests

    target.mkdir(parents=True, exist_ok=True)
    archive = target / f"{benchmark.key}.zip"
    if not archive.exists():
        print(f"  downloading {benchmark.url}")
        response = requests.get(benchmark.url, timeout=120, stream=True)
        response.raise_for_status()
        with archive.open("wb") as fh:
            for chunk in response.iter_content(1 << 16):
                fh.write(chunk)
    else:
        print(f"  reusing {archive}")

    extracted = target / "extracted"
    extracted.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
    except zipfile.BadZipFile:
        # LIAR is occasionally served as a bare tarball despite the .zip name.
        shutil.unpack_archive(str(archive), str(extracted))

    return {
        "sha256": sha256_file(archive),
        "bytes": archive.stat().st_size,
        "path": str(archive),
        "extracted_to": str(extracted),
        "files": sorted(p.name for p in extracted.rglob("*") if p.is_file())[:20],
    }


def clone_repo(benchmark: Benchmark, target: Path) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    name = benchmark.url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    destination = target / name
    if destination.exists():
        print(f"  reusing {destination}")
    else:
        print(f"  cloning {benchmark.url}")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", benchmark.url, str(destination)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip()[:400])
    digest, size = sha256_tree(destination)
    return {
        "sha256": digest,
        "checksum_kind": "sha256-tree (excludes .git)",
        "bytes": size,
        "path": str(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated subset: liar,coaid,fakenewsnet")
    parser.add_argument("--list", action="store_true", help="describe the datasets and exit")
    args = parser.parse_args()

    if args.list:
        for benchmark in BENCHMARKS.values():
            print(f"\n{benchmark.key}: {benchmark.label}\n  {benchmark.url}\n  {benchmark.notes}")
            print(f"  cite: {benchmark.citation}")
        return 0

    settings = get_settings()
    root = settings.benchmarks_dir
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )

    keys = [k.strip() for k in args.only.split(",")] if args.only else list(BENCHMARKS)
    failures = 0
    for key in keys:
        benchmark = BENCHMARKS.get(key)
        if benchmark is None:
            print(f"unknown benchmark {key!r}; known: {', '.join(BENCHMARKS)}")
            failures += 1
            continue

        print(f"\n{benchmark.key}: {benchmark.label}")
        target = root / benchmark.key
        try:
            if benchmark.kind == "zip":
                entry = download_zip(benchmark, target)
            elif benchmark.kind == "git":
                entry = clone_repo(benchmark, target)
            else:  # pragma: no cover - no manual-only benchmarks today
                entry = {"path": str(target), "manual": True}
        except Exception as exc:
            print(f"  [failed] {type(exc).__name__}: {exc}")
            failures += 1
            continue

        entry.update(
            {
                "url": benchmark.url,
                "label": benchmark.label,
                "notes": benchmark.notes,
                "citation": benchmark.citation,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        manifest[benchmark.key] = entry
        print(f"  ok  sha256={entry.get('sha256', '-')[:16]}  bytes={entry.get('bytes', 0):,}")
        for step in benchmark.manual_steps:
            print(f"  next: {step}")

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nmanifest: {manifest_path}")
    print("note: Phase 1 does not train or evaluate anything on these. Phase 2 does.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
