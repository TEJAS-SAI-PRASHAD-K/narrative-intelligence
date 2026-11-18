"""Checkpoint resolution with a hash check. A fake hub replaces the network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import modeling.registry as registry
from modeling.registry import CheckpointIntegrityError, CheckpointNotFound, CheckpointReason

HUB_FILES = {
    "config.json": b'{"id2label": {"0": "real", "1": "fake"}}',
    "model.safetensors": b"weights",
    "tokenizer/tokenizer.json": b"vocab",
    "misinfo_prob_calibrator.json": b'{"kind": "isotonic"}',
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Settings, config and a fake ``snapshot_download`` that writes ``HUB_FILES``."""
    settings = SimpleNamespace(
        models_dir=tmp_path / "models", hf_repo="owner/repo", hf_token="t", gdrive_dir=None
    )
    config: dict = {"version": "v1", "sha256": {}}
    calls: list[dict] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("raises"):
            raise kwargs["raises"]
        root = settings.models_dir / ".hub" / "laya" / "v1"
        for name, data in HUB_FILES.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(data)
        (root / "checkpoint_latest").mkdir()
        (root / "checkpoint_latest" / "optimizer.pt").write_bytes(b"resume")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    monkeypatch.setattr(registry, "get_settings", lambda: settings)
    monkeypatch.setattr(registry, "module_config", lambda module: config)
    registry.REASON_COUNTS.clear()
    return SimpleNamespace(settings=settings, config=config, calls=calls, tmp=tmp_path)


def _good_hash(env) -> str:
    root = env.tmp / "ref"
    for name, data in HUB_FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    digest = registry._dir_sha256(root)
    assert digest
    return digest


def test_pull_with_good_hash_is_verified(env):
    env.config["sha256"] = {"v1": _good_hash(env)}
    got = registry.resolve("laya", "v1")
    assert got.origin == "hf" and got.verified
    assert (got.path / "misinfo_prob_calibrator.json").is_file()
    assert not (got.path / "checkpoint_latest").exists()
    assert "*/checkpoint_latest/*" in env.calls[0]["ignore_patterns"]
    assert env.calls[0]["allow_patterns"] == ["laya/v1/*"]
    assert not (env.settings.models_dir / ".hub" / "laya" / "v1").exists()


def test_local_copy_is_verified_on_every_resolve(env):
    env.config["sha256"] = {"v1": _good_hash(env)}
    registry.resolve("laya", "v1")
    again = registry.resolve("laya", "v1")
    assert again.origin == "local" and again.verified
    assert len(env.calls) == 1


def test_bad_hash_deletes_pulled_copy(env):
    env.config["sha256"] = {"v1": "0" * 64}
    with pytest.raises(CheckpointIntegrityError) as err:
        registry.resolve("laya", "v1")
    assert err.value.reason == "checkpoint_sha256_mismatch"
    assert not registry.local_dir("laya", "v1").exists()
    assert registry.REASON_COUNTS[CheckpointReason.SHA256_MISMATCH] == 1


def test_bad_hash_keeps_a_local_copy_and_still_raises(env):
    local = registry.local_dir("laya", "v1")
    local.mkdir(parents=True)
    (local / "model.safetensors").write_bytes(b"other")
    env.config["sha256"] = {"v1": "0" * 64}
    with pytest.raises(CheckpointIntegrityError):
        registry.resolve("laya", "v1")
    assert (local / "model.safetensors").exists()


def test_no_configured_hash_is_counted_unverified(env):
    got = registry.resolve("laya", "v1")
    assert not got.verified
    assert registry.REASON_COUNTS[CheckpointReason.UNVERIFIED] == 1


def test_hash_ignores_resume_state_and_registry_file(env):
    first = _good_hash(env)
    root = env.tmp / "ref"
    (root / "checkpoint_latest").mkdir()
    (root / "checkpoint_latest" / "optimizer.pt").write_bytes(b"x")
    (root / "registry.json").write_text("{}")
    assert registry._dir_sha256(root) == first
    (root / "misinfo_prob_calibrator.json").write_bytes(b"{}")
    assert registry._dir_sha256(root) != first


def test_plain_string_hash_applies_to_the_configured_version_only(env):
    env.config["sha256"] = "AB" * 32
    assert registry.expected_sha256("laya", "v1") == "ab" * 32
    assert registry.expected_sha256("laya", "v2") is None


def test_missing_repo_raises_not_found(env):
    env.settings.hf_repo = None
    with pytest.raises(CheckpointNotFound, match="HF_REPO and HF_TOKEN"):
        registry.resolve("laya", "v1")
    assert env.calls == []


def test_missing_token_is_counted_and_a_refused_pull_is_counted(env, monkeypatch):
    env.settings.hf_token = None
    import huggingface_hub

    def refuse(**kwargs):
        raise PermissionError("401 private repo")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)
    with pytest.raises(CheckpointNotFound):
        registry.resolve("laya", "v1")
    assert registry.REASON_COUNTS[CheckpointReason.HUB_TOKEN_MISSING] == 1
    assert registry.REASON_COUNTS[CheckpointReason.HUB_PULL_FAILED] == 1
    assert registry.resolve("laya", "v1", required=False) is None


def test_repo_without_the_folder_is_counted(env, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: None)
    assert registry.resolve("laya", "v1", required=False) is None
    assert registry.REASON_COUNTS[CheckpointReason.HUB_PATH_ABSENT] == 1


def test_checkpoint_hash_command_prints_the_digest(env, monkeypatch):
    from typer.testing import CliRunner

    from modeling.cli import app

    env.config["sha256"] = {"v1": _good_hash(env)}
    registry.resolve("laya", "v1")
    result = CliRunner().invoke(app, ["checkpoint-hash", "laya", "--version", "v1"])
    assert result.exit_code == 0, result.output
    assert env.config["sha256"]["v1"] in result.output
