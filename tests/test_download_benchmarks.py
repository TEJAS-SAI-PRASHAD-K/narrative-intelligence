"""Pinning and checksum logic of ``scripts/download_benchmarks.py``.

No test reaches the network: git sources are local repositories and the zip
download is a stub for ``requests.get``.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "download_benchmarks.py"
_spec = importlib.util.spec_from_file_location("download_benchmarks", SCRIPT)
db = importlib.util.module_from_spec(_spec)
sys.modules["download_benchmarks"] = db
_spec.loader.exec_module(db)

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def git(cwd: Path, *args: str) -> str:
    import os

    out = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **GIT_ENV},
    )
    return out.stdout.strip()


@pytest.fixture
def upstream(tmp_path):
    """A local repo with two commits. Returns (url, first_commit, tip_commit)."""
    repo = tmp_path / "upstream" / "data-repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "labels.csv").write_text("id,label\n1,a\n", encoding="utf-8")
    git(repo, "add", "labels.csv")
    git(repo, "commit", "-q", "-m", "first")
    first = git(repo, "rev-parse", "HEAD")
    (repo / "labels.csv").write_text("id,label\n1,b\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "second")
    tip = git(repo, "rev-parse", "HEAD")
    return repo.as_uri(), first, tip


def bench(url, commit="", checksum="", key="demo", slot=""):
    return db.Benchmark(
        key=key, label="demo", url=url, kind="git", notes="", citation="",
        commit=commit, checksum=checksum, slot=slot,
    )  # fmt: skip


def tree_checksum_at(url, commit, tmp_path):
    dest = tmp_path / "probe"
    db.checkout_pinned(url, commit, dest)
    return db.sha256_tree(dest)[0]


def test_every_pin_is_well_formed():
    for benchmark in db.BENCHMARKS.values():
        assert re.fullmatch(r"[0-9a-f]{64}", benchmark.checksum), benchmark.key
        if benchmark.kind == "git":
            assert re.fullmatch(r"[0-9a-f]{40}", benchmark.commit), benchmark.key


def test_fnc1_is_registered_in_the_stance_slot():
    fnc1 = db.BENCHMARKS["fnc1"]
    assert fnc1.url == "https://github.com/FakeNewsChallenge/fnc-1.git"
    assert fnc1.directory == "stance"


def test_fnc1_is_not_part_of_the_misinformation_training_frame():
    import inspect

    from modeling.text import misinfo_clf

    assert "fnc1" not in inspect.getsource(misinfo_clf.build_training_frame)


def test_checkout_is_at_the_pinned_commit_not_the_branch_tip(upstream, tmp_path):
    url, first, tip = upstream
    dest = tmp_path / "out"
    db.checkout_pinned(url, first, dest)
    assert git(dest, "rev-parse", "HEAD") == first != tip
    assert (dest / "labels.csv").read_text(encoding="utf-8") == "id,label\n1,a\n"


def test_clone_repo_accepts_a_matching_pin(upstream, tmp_path):
    url, first, _ = upstream
    checksum = tree_checksum_at(url, first, tmp_path)
    entry = db.clone_repo(bench(url, first, checksum), tmp_path / "bm")
    assert entry["sha256"] == checksum
    assert entry["commit"] == first
    assert (tmp_path / "bm" / "data-repo" / "labels.csv").exists()
    assert not (tmp_path / "bm" / "data-repo.partial").exists()


def test_clone_repo_rejects_a_checksum_mismatch_and_leaves_nothing(upstream, tmp_path):
    url, first, _ = upstream
    with pytest.raises(db.ChecksumMismatch, match="sha256-tree"):
        db.clone_repo(bench(url, first, "0" * 64), tmp_path / "bm")
    assert not (tmp_path / "bm" / "data-repo").exists()
    assert not (tmp_path / "bm" / "data-repo.partial").exists()


def test_clone_repo_of_an_unknown_commit_fails_and_leaves_nothing(upstream, tmp_path):
    url, _, _ = upstream
    with pytest.raises(Exception):  # noqa: B017 - git's own error is the contract
        db.clone_repo(bench(url, "1" * 40, "0" * 64), tmp_path / "bm")
    assert not (tmp_path / "bm" / "data-repo").exists()
    assert not (tmp_path / "bm" / "data-repo.partial").exists()


def test_existing_clone_that_was_edited_is_quarantined(upstream, tmp_path):
    url, first, _ = upstream
    checksum = tree_checksum_at(url, first, tmp_path)
    target = tmp_path / "bm"
    db.clone_repo(bench(url, first, checksum), target)
    (target / "data-repo" / "labels.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(db.ChecksumMismatch, match="rejected"):
        db.clone_repo(bench(url, first, checksum), target)
    assert not (target / "data-repo").exists()
    kept = list((tmp_path / "_rejected").glob("demo-*/labels.csv"))
    assert len(kept) == 1


def test_existing_clone_at_the_wrong_commit_is_quarantined(upstream, tmp_path):
    url, first, tip = upstream
    target = tmp_path / "bm"
    db.checkout_pinned(url, tip, target / "data-repo")
    with pytest.raises(db.ChecksumMismatch, match="commit"):
        db.clone_repo(bench(url, first, "0" * 64), target)
    assert not (target / "data-repo").exists()


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("train.tsv", "a\tb\n")
    return buffer.getvalue()


class _Response:
    def __init__(self, payload: bytes):
        self._payload = payload

    def raise_for_status(self):
        return None

    def iter_content(self, size):
        yield self._payload


@pytest.fixture
def stub_zip(monkeypatch):
    import requests

    payload = _zip_bytes()
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response(payload))
    return payload


def zip_bench(checksum):
    return db.Benchmark(
        key="z", label="z", url="https://example.invalid/z.zip", kind="zip",
        notes="", citation="", checksum=checksum,
    )  # fmt: skip


def test_zip_with_a_matching_pin_is_extracted(stub_zip, tmp_path):
    import hashlib

    entry = db.download_zip(zip_bench(hashlib.sha256(stub_zip).hexdigest()), tmp_path / "z")
    assert entry["pinned"] is True
    assert (tmp_path / "z" / "extracted" / "train.tsv").exists()


def test_zip_mismatch_is_removed_and_never_extracted(stub_zip, tmp_path):
    with pytest.raises(db.ChecksumMismatch, match="archive sha256"):
        db.download_zip(zip_bench("0" * 64), tmp_path / "z")
    assert not (tmp_path / "z" / "z.zip").exists()
    assert not (tmp_path / "z" / "extracted").exists()
    assert len(list((tmp_path / "_rejected").glob("z-*/z.zip"))) == 1


def test_main_exits_nonzero_on_mismatch_and_writes_no_manifest_entry(
    upstream, tmp_path, monkeypatch, capsys
):
    url, first, _ = upstream
    good = tree_checksum_at(url, first, tmp_path)

    class FakeSettings:
        benchmarks_dir = tmp_path / "benchmarks"

    monkeypatch.setattr(db, "get_settings", lambda: FakeSettings)
    monkeypatch.setattr(db, "BENCHMARKS", {"demo": bench(url, first, good)})
    monkeypatch.setattr(sys, "argv", ["download_benchmarks.py"])
    assert db.main() == 0
    manifest = json.loads((FakeSettings.benchmarks_dir / "manifest.json").read_text())
    assert manifest["demo"]["commit"] == first

    # Same data on disk, now pinned to a different checksum: upstream "changed".
    monkeypatch.setattr(db, "BENCHMARKS", {"demo": bench(url, first, "0" * 64)})
    assert db.main() == 1
    assert "CHECKSUM MISMATCH" in capsys.readouterr().err
    manifest = json.loads((FakeSettings.benchmarks_dir / "manifest.json").read_text())
    assert "demo" not in manifest
    assert not (FakeSettings.benchmarks_dir / "demo" / "data-repo").exists()


# --- a rejected download must not stay readable as data ------------------------


def test_rejected_coaid_clone_is_no_longer_found_by_its_loader(tmp_path, monkeypatch):
    from ingest.config import Settings
    from modeling.datasets.base import DatasetUnavailable
    from modeling.datasets.coaid import CoAID

    root = tmp_path / "benchmarks"
    slot = root / "coaid"
    clone = slot / "CoAID" / "05-01-2020"
    clone.mkdir(parents=True)
    (clone / "NewsFakeCOVID-19.csv").write_text("a\n1\n", encoding="utf-8")

    monkeypatch.setattr("modeling.datasets.base.get_settings", lambda: Settings(data_dir=tmp_path))
    dataset = CoAID()
    dataset.validate(dataset.default_path())  # present before the rejection

    db._reject(slot / "CoAID", root, "coaid")

    with pytest.raises(DatasetUnavailable):
        dataset.validate(dataset.default_path())
    assert list((root / "_rejected").glob("coaid-*/05-01-2020/NewsFakeCOVID-19.csv"))


def test_rejected_fnc1_clone_is_no_longer_resolved(tmp_path):
    from modeling.datasets.fnc1 import FNC1

    root = tmp_path / "benchmarks"
    slot = root / "stance"
    clone = slot / "fnc-1"
    clone.mkdir(parents=True)
    for name in ("train_stances.csv", "train_bodies.csv"):
        (clone / name).write_text("x\n", encoding="utf-8")
    dataset = FNC1()
    assert dataset._resolve_dir(slot) == clone

    db._reject(clone, root, "fnc1")

    assert not (dataset._resolve_dir(slot) / "train_stances.csv").exists()


def test_rejected_liar_archive_also_removes_the_earlier_extraction(stub_zip, tmp_path):
    import hashlib

    from modeling.datasets.base import DatasetUnavailable
    from modeling.datasets.liar import Liar

    slot = tmp_path / "benchmarks" / "liar"
    good = hashlib.sha256(stub_zip).hexdigest()
    bench_ok = db.Benchmark(
        key="liar", label="l", url="https://example.invalid/l.zip", kind="zip",
        notes="", citation="", checksum=good,
    )  # fmt: skip
    db.download_zip(bench_ok, slot)
    extracted = slot / "extracted"
    for name in ("train.tsv", "valid.tsv", "test.tsv"):
        (extracted / name).write_text("x\n", encoding="utf-8")
    Liar().validate(slot)  # the earlier extraction is readable

    bench_bad = db.Benchmark(
        key="liar", label="l", url=bench_ok.url, kind="zip", notes="", citation="",
        checksum="0" * 64,
    )  # fmt: skip
    with pytest.raises(db.ChecksumMismatch):
        db.download_zip(bench_bad, slot)

    assert not extracted.exists()
    with pytest.raises(DatasetUnavailable):
        Liar().validate(slot)
