"""The repository's own shape. Cheap to assert, expensive to get wrong.

These tests exist because of one character. `.gitignore` carried an unanchored
`models/`, written to exclude ML checkpoints at the repo root. Unanchored
patterns match at **any** depth, so it also matched `app/models/` -- the
SQLAlchemy ORM package holding all 27 tables -- and that package was therefore
never committed. On a clean clone `app.main` could not import, the API could
not boot, and the contract, migration and ETL suites errored out. The same
reach applied to `data/`, `uploads/`, `reports/` and `build/`, which between
them made `modeling/models/`, `modeling/data/` and `app/reports/` unreachable
by git.

Nothing in the test suite noticed, because every test that would have failed
was one that could not be collected.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_ignored(path: str) -> bool:
    """Whether git would ignore `path`, asked of git rather than reimplemented.

    `git check-ignore` exits 0 when the path IS ignored, 1 when it is not, and
    128 on error -- which must not be read as "not ignored".
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if result.returncode not in (0, 1):
        pytest.skip(f"git check-ignore unavailable: {result.stderr.decode()[:120]}")
    return result.returncode == 0


#: Import paths the code actually uses, written as files that may not exist
#: yet. `--no-index` means existence is irrelevant: we are asking about the
#: ignore rules, not about the working tree. `app/models` is the one that bit;
#: `modeling/models` is where a new model module would naturally go next.
MUST_BE_TRACKABLE = [
    "app/models/__init__.py",
    "app/models/core.py",
    "app/models/corpus.py",
    "app/models/narratives.py",
    "app/models/actors.py",
    "app/models/network.py",
    "app/models/domains.py",
    "app/models/embeddings.py",
    "app/models/compass.py",
    "app/models/ops.py",
    "app/reports/render.py",
    "app/data/seed.py",
    "modeling/models/any_new_model.py",
    "modeling/data/any_new_loader.py",
    "ingest/data/any_new_module.py",
]

#: The build products the anchored patterns still have to catch. Anchoring is
#: only correct if it did not quietly stop ignoring the things it was for.
MUST_STAY_IGNORED = [
    "data/normalized/source=news/date=2026-10-09/part-0.parquet",
    "data/manifest.json",
    "models/misinfo/v0.1.0/model.safetensors",
    "uploads/clip.mp4",
    "reports/report.pdf",
    "build/lib/app/main.py",
    "dist/narrative_intelligence-0.4.0.whl",
    ".env",
    "kaggle.json",
    "app/__pycache__/config.cpython-311.pyc",
    "narrative_intelligence.egg-info/PKG-INFO",
    ".pytest_cache/v/cache/lastfailed",
    ".ruff_cache/content",
]


@pytest.mark.parametrize("path", MUST_BE_TRACKABLE)
def test_source_paths_are_not_gitignored(path: str) -> None:
    assert not _is_ignored(path), (
        f"{path} is gitignored. A source file placed there would be invisible to "
        "git and absent from a clean clone -- exactly how app/models/ went "
        "missing. Anchor the offending pattern with a leading slash."
    )


@pytest.mark.parametrize("path", MUST_STAY_IGNORED)
def test_build_products_are_still_gitignored(path: str) -> None:
    assert _is_ignored(path), (
        f"{path} is NOT gitignored. Anchoring the patterns must not have "
        "stopped them catching what they were written for."
    )


def test_every_source_package_has_an_init() -> None:
    """A package dir without __init__.py is a namespace package by accident."""
    for package in ("ingest", "modeling", "app", "nlp"):
        root = REPO_ROOT / package
        if not root.is_dir():
            continue
        for directory in root.rglob("*"):
            if not directory.is_dir() or directory.name == "__pycache__":
                continue
            if not any(directory.glob("*.py")):
                continue  # data/prompt/fixture directories are fine
            assert (directory / "__init__.py").exists(), (
                f"{directory.relative_to(REPO_ROOT)} holds .py files but no "
                "__init__.py; it will import inconsistently depending on sys.path."
            )


def test_no_source_file_is_untracked_and_ignored() -> None:
    """Catch the live version of the bug: a .py on disk that git cannot see.

    This is the check that would have caught app/models/ at the moment it was
    written, rather than months later when someone tried a clean clone.
    """
    offenders: list[str] = []
    for package in ("ingest", "modeling", "app", "nlp", "tests"):
        root = REPO_ROOT / package
        if not root.is_dir():
            continue
        for source in root.rglob("*.py"):
            if "__pycache__" in source.parts:
                continue
            relative = source.relative_to(REPO_ROOT).as_posix()
            if _is_ignored(relative):
                offenders.append(relative)
    assert not offenders, (
        "These source files exist on disk but git is configured to ignore them, "
        "so they will not survive a clone: " + ", ".join(sorted(offenders))
    )
