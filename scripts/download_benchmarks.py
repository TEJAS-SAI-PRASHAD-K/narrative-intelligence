#!/usr/bin/env python3
"""Fetch LIAR, FakeNewsNet, CoAID and FNC-1 into ``data/benchmarks/``, pinned.

Phase 1 does no model training and no benchmark evaluation -- this script exists
purely so Phase 2 starts unblocked, with the labelled datasets already on disk
and checksummed the same way the corpus is.

Every download is pinned. A git benchmark is checked out at an exact commit, not
at whatever the default branch holds today, and its tree is compared with a pinned
``sha256-tree``. LIAR's zip is compared with a pinned sha256. On a mismatch the
script stops with a non-zero exit: the bad download is removed (or, if it was
already on disk, moved to ``_rejected/`` under the benchmarks directory) and no
manifest entry is written, so
a changed upstream never passes as the dataset the numbers were measured on. To
adopt a new upstream on purpose, change the pin in ``BENCHMARKS`` in the same
commit that explains why.

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
* **FNC-1** is the Fake News Challenge stance data (GitHub, CSVs). It is for the
  later stance task and is not part of the misinformation training set. It is
  placed in ``stance/``, the slot the stance loader reads.
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
    #: git: the exact commit to check out. zip: unused.
    commit: str = ""
    #: zip: sha256 of the archive. git: ``sha256_tree`` of the checkout.
    checksum: str = ""
    #: Directory under ``data/benchmarks/``; defaults to ``key``.
    slot: str = ""

    @property
    def directory(self) -> str:
        return self.slot or self.key


class ChecksumMismatch(RuntimeError):
    """A download does not match its pin. The data has been removed or quarantined."""


BENCHMARKS: dict[str, Benchmark] = {
    "liar": Benchmark(
        key="liar",
        label="LIAR (12.8k PolitiFact short statements, 6-way labels)",
        url="https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
        kind="zip",
        checksum="611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7",
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
        commit="d238224346781255e1e7e6ed8bc410c2b2e6329e",
        checksum="74e0aadfcbae89948d0e937a3c4290efef312a873f26fb27028946ca643d5f89",
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
        commit="654361e1c8d5baa751baf1dac5032df621652280",
        checksum="f1c75a1620ca497bda16ef4d7ef36da7d282c7b5c21b406afd3775359519a4e6",
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
    "fnc1": Benchmark(
        key="fnc1",
        label="FNC-1 (Fake News Challenge stance detection, 4 labels)",
        url="https://github.com/FakeNewsChallenge/fnc-1.git",
        kind="git",
        commit="29d473af2d15278f0464d5e41e4cbe7eb58231f2",
        checksum="1a79a083217d610eb84c7976875d4ea11d12168cb248369e95e291ae40810829",
        slot="stance",
        notes=(
            "Headline/body stance pairs (agree, disagree, discuss, unrelated) with "
            "train_*.csv and competition_test_*.csv. For the later stance task only: "
            "it is not part of the misinformation training frame. Cloned into "
            "stance/, where the stance loader looks."
        ),
        citation="Pomerleau and Rao (2017), the Fake News Challenge Stage 1 (FNC-1).",
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


REJECTED_DIRNAME = "_rejected"


def _reject(path: Path, root: Path, key: str) -> Path:
    """Move ``path`` out of its slot into ``<root>/_rejected/<key>-<timestamp>``.

    A rename inside the slot is not enough: loaders glob their slot recursively
    and would still read the rejected files as data. No loader reads ``_rejected/``
    because each one starts from its own slot under ``root``. A rejected file is
    placed inside the timestamped directory, a rejected directory becomes it.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / REJECTED_DIRNAME / f"{key}-{stamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.move(str(path), str(destination))
    else:
        destination.mkdir()
        shutil.move(str(path), str(destination / path.name))
    return destination


def _mismatch(what: str, expected: str, actual: str) -> ChecksumMismatch:
    return ChecksumMismatch(f"{what}: expected {expected}, got {actual}")


def download_zip(benchmark: Benchmark, target: Path) -> dict:
    import requests

    target.mkdir(parents=True, exist_ok=True)
    archive = target / f"{benchmark.key}.zip"
    if not archive.exists():
        print(f"  downloading {benchmark.url}")
        partial = archive.with_name(archive.name + ".part")
        response = requests.get(benchmark.url, timeout=120, stream=True)
        response.raise_for_status()
        with partial.open("wb") as fh:
            for chunk in response.iter_content(1 << 16):
                fh.write(chunk)
        partial.rename(archive)
    else:
        print(f"  reusing {archive}")

    actual = sha256_file(archive)
    if benchmark.checksum and actual != benchmark.checksum:
        rejected = _reject(archive, target.parent, benchmark.key)
        # An earlier extraction came from an archive that is no longer trusted.
        shutil.rmtree(target / "extracted", ignore_errors=True)
        raise _mismatch(f"{benchmark.key} archive sha256 (moved to {rejected})",
                        benchmark.checksum, actual)

    # Extract only after the archive is verified, and into an empty directory so a
    # stale extraction from an earlier archive cannot mix in.
    extracted = target / "extracted"
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir()
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
    except zipfile.BadZipFile:
        # LIAR is occasionally served as a bare tarball despite the .zip name.
        shutil.unpack_archive(str(archive), str(extracted))

    return {
        "sha256": actual,
        "checksum_kind": "sha256-file (zip archive)",
        "pinned": bool(benchmark.checksum),
        "bytes": archive.stat().st_size,
        "path": str(archive),
        "extracted_to": str(extracted),
        "files": sorted(p.name for p in extracted.rglob("*") if p.is_file())[:20],
    }


def _git(args: list[str], cwd: Path, timeout: int = 1800) -> str:
    result = subprocess.run(
        ["git", "-c", "core.autocrlf=false", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip()[:400])
    return result.stdout.strip()


def checkout_pinned(url: str, commit: str, destination: Path) -> None:
    """Clone ``url`` into ``destination`` with ``commit`` checked out, detached."""
    destination.mkdir(parents=True)
    _git(["init", "-q"], destination)
    _git(["remote", "add", "origin", url], destination)
    try:
        # A shallow fetch of one commit is cheap and GitHub serves it by sha.
        _git(["fetch", "-q", "--depth", "1", "origin", commit], destination)
        _git(["checkout", "-q", "--detach", "FETCH_HEAD"], destination)
    except RuntimeError:
        # A host that refuses a fetch by sha needs the full history instead.
        _git(["fetch", "-q", "origin"], destination)
        _git(["checkout", "-q", "--detach", commit], destination)
    head = _git(["rev-parse", "HEAD"], destination)
    if head != commit:
        raise _mismatch("checked-out commit", commit, head)


def clone_repo(benchmark: Benchmark, target: Path) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    name = benchmark.url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    destination = target / name
    if destination.exists():
        print(f"  reusing {destination}")
        candidate = destination
    else:
        print(f"  cloning {benchmark.url} at {benchmark.commit[:12] or 'default branch'}")
        candidate = destination.with_name(name + ".partial")
        if candidate.exists():
            shutil.rmtree(candidate)
        try:
            if benchmark.commit:
                checkout_pinned(benchmark.url, benchmark.commit, candidate)
            else:
                _git(["clone", "--depth", "1", benchmark.url, str(candidate)], target)
        except BaseException:
            shutil.rmtree(candidate, ignore_errors=True)
            raise

    problem = None
    if benchmark.commit and (candidate / ".git").exists():
        head = _git(["rev-parse", "HEAD"], candidate)
        if head != benchmark.commit:
            problem = _mismatch(f"{benchmark.key} commit", benchmark.commit, head)
    digest, size = sha256_tree(candidate)
    if problem is None and benchmark.checksum and digest != benchmark.checksum:
        problem = _mismatch(f"{benchmark.key} sha256-tree", benchmark.checksum, digest)
    if problem is not None:
        if candidate == destination:
            rejected = _reject(candidate, target.parent, benchmark.key)
        else:
            shutil.rmtree(candidate, ignore_errors=True)
            rejected = None
        raise ChecksumMismatch(
            f"{problem}" + (f" (moved to {rejected})" if rejected else " (download removed)")
        )
    if candidate != destination:
        candidate.rename(destination)
    return {
        "sha256": digest,
        "checksum_kind": "sha256-tree (excludes .git)",
        "pinned": bool(benchmark.checksum),
        "commit": benchmark.commit or None,
        "bytes": size,
        "path": str(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated subset: liar,coaid,fakenewsnet,fnc1")
    parser.add_argument("--list", action="store_true", help="describe the datasets and exit")
    args = parser.parse_args()

    if args.list:
        for benchmark in BENCHMARKS.values():
            print(f"\n{benchmark.key}: {benchmark.label}\n  {benchmark.url}\n  {benchmark.notes}")
            pin = benchmark.commit or benchmark.checksum
            print(f"  pin: {pin or 'none'}")
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
    mismatches = 0
    for key in keys:
        benchmark = BENCHMARKS.get(key)
        if benchmark is None:
            print(f"unknown benchmark {key!r}; known: {', '.join(BENCHMARKS)}")
            failures += 1
            continue

        print(f"\n{benchmark.key}: {benchmark.label}")
        target = root / benchmark.directory
        try:
            if benchmark.kind == "zip":
                entry = download_zip(benchmark, target)
            elif benchmark.kind == "git":
                entry = clone_repo(benchmark, target)
            else:  # pragma: no cover - no manual-only benchmarks today
                entry = {"path": str(target), "manual": True}
        except ChecksumMismatch as exc:
            # A manifest entry from an earlier good run no longer describes the disk.
            manifest.pop(benchmark.key, None)
            print(f"  [CHECKSUM MISMATCH] {exc}", file=sys.stderr)
            mismatches += 1
            failures += 1
            continue
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
    if mismatches:
        print(
            f"error: {mismatches} download(s) did not match their pin and were not "
            "accepted; see [CHECKSUM MISMATCH] above",
            file=sys.stderr,
        )
    print("note: Phase 1 does not train or evaluate anything on these. Phase 2 does.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
