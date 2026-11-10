"""The upload sweep: a file-system-only deletion of entries past the retention window."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from app.tasks import maintenance
from app.tasks.maintenance import sweep_uploads

HOURS = 24


def _age(path: Path, hours: float) -> None:
    stamp = time.time() - hours * 3600
    os.utime(path, (stamp, stamp), follow_symlinks=False)


@pytest.fixture
def uploads(tmp_path: Path) -> Path:
    root = tmp_path / "uploads"
    root.mkdir()
    return root


def test_an_old_entry_is_deleted_and_a_new_entry_is_kept(uploads: Path):
    old_file = uploads / "old.mp4"
    old_file.write_bytes(b"x" * 10)
    old_dir = uploads / "job-1"
    old_dir.mkdir()
    (old_dir / "clip.mp4").write_bytes(b"y" * 5)
    new_file = uploads / "new.mp4"
    new_file.write_bytes(b"z")
    _age(old_file, HOURS + 1)
    _age(old_dir, HOURS + 1)

    result = sweep_uploads(uploads, HOURS)

    assert not old_file.exists()
    assert not old_dir.exists()
    assert new_file.exists()
    assert result["deleted"] == 2
    assert result["failed"] == 0
    assert result["bytes_freed"] == 15


def test_a_missing_uploads_dir_is_not_an_error(tmp_path: Path):
    result = sweep_uploads(tmp_path / "absent", HOURS)

    assert result == {"deleted": 0, "failed": 0, "reasons": {}, "bytes_freed": 0}


def test_a_delete_failure_is_counted_with_a_reason_and_the_entry_stays(
    uploads: Path, monkeypatch: pytest.MonkeyPatch
):
    stuck = uploads / "stuck"
    stuck.mkdir()
    (stuck / "clip.mp4").write_bytes(b"x")
    _age(stuck, HOURS + 1)
    gone = uploads / "gone.mp4"
    gone.write_bytes(b"x")
    _age(gone, HOURS + 1)

    def refuse(path, *args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(shutil, "rmtree", refuse)

    result = sweep_uploads(uploads, HOURS)

    assert stuck.exists()
    assert not gone.exists()
    assert result["deleted"] == 1
    assert result["failed"] == 1
    assert result["reasons"] == {maintenance.REASON_DELETE_FAILED: 1}

    monkeypatch.undo()
    assert sweep_uploads(uploads, HOURS)["deleted"] == 1
    assert not stuck.exists()


def test_a_symlink_is_removed_and_its_target_is_not_followed(uploads: Path, tmp_path: Path):
    target = tmp_path / "outside"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    link = uploads / "link"
    link.symlink_to(target, target_is_directory=True)
    _age(link, HOURS + 1)

    result = sweep_uploads(uploads, HOURS)

    assert not link.is_symlink()
    assert (target / "keep.txt").exists()
    assert result["deleted"] == 1


def test_the_task_uses_no_database(uploads: Path, monkeypatch: pytest.MonkeyPatch):
    old = uploads / "old.mp4"
    old.write_bytes(b"x")
    _age(old, HOURS + 1)

    class Settings:
        uploads_dir = uploads
        media_retention_hours = HOURS
        data_dir = uploads.parent / "data"
        models_dir = uploads.parent / "models"
        reports_dir = uploads.parent / "reports"

    def no_database(*args, **kwargs):
        raise AssertionError("the sweep must not open a database session")

    monkeypatch.setattr(maintenance, "get_api_settings", lambda: Settings())
    monkeypatch.setattr("app.db.sync_session", no_database)
    monkeypatch.setattr("app.db.get_engine", no_database, raising=False)

    result = maintenance.purge_uploads.run()

    assert not old.exists()
    assert result["deleted"] == 1
    assert result["retention_hours"] == HOURS


def _forbid_deletion(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("a refused sweep must not delete")

    monkeypatch.setattr(shutil, "rmtree", boom)
    monkeypatch.setattr(Path, "unlink", boom)


def test_the_repo_root_is_refused(monkeypatch: pytest.MonkeyPatch):
    _forbid_deletion(monkeypatch)

    result = sweep_uploads(maintenance.REPO_ROOT, HOURS)

    assert result["deleted"] == 0
    assert result["reasons"] == {maintenance.REASON_UNSAFE_DIR: 1}


def test_the_filesystem_root_is_refused(monkeypatch: pytest.MonkeyPatch):
    _forbid_deletion(monkeypatch)

    result = sweep_uploads(Path("/"), HOURS)

    assert result["deleted"] == 0
    assert result["reasons"] == {maintenance.REASON_UNSAFE_DIR: 1}


def test_an_ancestor_of_a_protected_dir_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = tmp_path / "srv" / "data"
    data.mkdir(parents=True)
    old = tmp_path / "srv" / "old.bin"
    old.write_bytes(b"x")
    _age(old, HOURS + 1)

    result = sweep_uploads(tmp_path / "srv", HOURS, protected_dirs=[data])

    assert old.exists()
    assert result["reasons"] == {maintenance.REASON_UNSAFE_DIR: 1}


def test_a_protected_dir_itself_is_refused(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()

    result = sweep_uploads(models, HOURS, protected_dirs=[models])

    assert result["reasons"] == {maintenance.REASON_UNSAFE_DIR: 1}


def test_a_child_of_a_protected_dir_is_allowed(tmp_path: Path):
    data = tmp_path / "data"
    root = data / "uploads"
    root.mkdir(parents=True)
    old = root / "old.bin"
    old.write_bytes(b"x")
    _age(old, HOURS + 1)

    result = sweep_uploads(root, HOURS, protected_dirs=[data])

    assert not old.exists()
    assert result["deleted"] == 1


@pytest.mark.parametrize("hours", [0, -1])
def test_a_retention_below_one_hour_deletes_nothing(uploads: Path, hours: int):
    fresh = uploads / "fresh.bin"
    fresh.write_bytes(b"x")

    result = sweep_uploads(uploads, hours)

    assert fresh.exists()
    assert result["deleted"] == 0
    assert result["reasons"] == {maintenance.REASON_INVALID_RETENTION: 1}


def test_the_default_uploads_dir_passes_the_guard():
    from app.config import ApiSettings

    settings = ApiSettings()

    assert maintenance._refusal(
        settings.uploads_dir,
        settings.media_retention_hours,
        (settings.data_dir, settings.models_dir, settings.reports_dir),
    ) is None


def test_the_task_passes_the_settings_dirs_to_the_guard(
    uploads: Path, monkeypatch: pytest.MonkeyPatch
):
    old = uploads / "old.mp4"
    old.write_bytes(b"x")
    _age(old, HOURS + 1)

    class Settings:
        uploads_dir = uploads
        media_retention_hours = HOURS
        data_dir = uploads  # the uploads root equals a protected dir
        models_dir = uploads.parent / "models"
        reports_dir = uploads.parent / "reports"

    monkeypatch.setattr(maintenance, "get_api_settings", lambda: Settings())

    result = maintenance.purge_uploads.run()

    assert old.exists()
    assert result["reasons"] == {maintenance.REASON_UNSAFE_DIR: 1}
