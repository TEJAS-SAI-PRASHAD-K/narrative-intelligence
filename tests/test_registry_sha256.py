"""Checkpoint resolution with a hash check. A fake hub replaces the network."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import modeling.registry as registry
from modeling.registry import CheckpointIntegrityError, CheckpointNotFound, CheckpointReason

TOKENIZER_CONFIG = {"tokenizer_class": "TokenizersBackend", "backend": "tokenizers",
                    "is_local": True, "model_max_length": 512, "mask_token": "[MASK]",
                    "extra_special_tokens": ["<a>", "<b>"]}

HUB_FILES = {
    "config.json": b'{"id2label": {"0": "real", "1": "fake"}}',
    "model.safetensors": b"weights",
    "tokenizer/tokenizer.json": b"vocab",
    "tokenizer/tokenizer_config.json": json.dumps(TOKENIZER_CONFIG).encode(),
    "misinfo_prob_calibrator.json": b'{"kind": "isotonic"}',
    "checkpoint_latest/optimizer.pt": b"resume",
    "sub/checkpoint_latest/optimizer.pt": b"resume",
    "model.tmp": b"partial",
}


def _write(root: Path, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Settings, config and a fake ``snapshot_download`` that honors the patterns."""
    settings = SimpleNamespace(
        models_dir=tmp_path / "models", hf_repo="owner/repo", hf_token="t", gdrive_dir=None,
        hf_revision=None, allow_unverified_pull=False,
    )
    config: dict = {"version": "v1", "sha256": {}}
    calls: list[dict] = []

    def fake_download(**kwargs):
        from huggingface_hub.utils import filter_repo_objects

        calls.append(kwargs)
        kept = filter_repo_objects(
            [f"laya/v1/{name}" for name in HUB_FILES],
            allow_patterns=kwargs["allow_patterns"],
            ignore_patterns=kwargs["ignore_patterns"],
        )
        root = settings.models_dir / ".hub"
        for relative in kept:
            _write(root, {relative: HUB_FILES[relative.removeprefix("laya/v1/")]})

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    monkeypatch.setattr(registry, "get_settings", lambda: settings)
    monkeypatch.setattr(registry, "module_config", lambda module: config)
    monkeypatch.delenv("ALLOW_UNVERIFIED_PULL", raising=False)
    registry.REASON_COUNTS.clear()
    registry._LAST_REASON.clear()
    return SimpleNamespace(settings=settings, config=config, calls=calls, tmp=tmp_path)


def _good_hash(env) -> str:
    """The hash of the files that survive the pull, built from scratch."""
    root = env.tmp / "ref"
    shutil.rmtree(root, ignore_errors=True)
    _write(root, {k: v for k, v in HUB_FILES.items()
                  if "checkpoint_latest" not in k and not k.endswith(".tmp")})
    digest = registry._dir_sha256(root)
    assert digest
    return digest


def test_pull_with_good_hash_is_verified_and_skips_resume_state(env):
    env.config["sha256"] = {"v1": _good_hash(env)}
    got = registry.resolve("laya", "v1")
    assert got.origin == "hf" and got.verified
    assert (got.path / "misinfo_prob_calibrator.json").is_file()
    assert not (got.path / "checkpoint_latest").exists()
    assert not (got.path / "sub" / "checkpoint_latest").exists()
    assert not (got.path / "model.tmp").exists()
    assert env.calls[0]["allow_patterns"] == ["laya/v1/*"]
    assert not (env.settings.models_dir / ".hub" / "laya" / "v1").exists()


def test_the_revision_is_passed_from_config_or_settings(env):
    env.config["sha256"] = {"v1": _good_hash(env)}
    env.settings.hf_revision = "abc"
    registry.resolve("laya", "v1")
    assert env.calls[0]["revision"] == "abc"
    shutil.rmtree(registry.local_dir("laya", "v1"))
    env.config["hf_revision"] = "def"
    registry.resolve("laya", "v1")
    assert env.calls[1]["revision"] == "def"


def test_local_copy_is_verified_on_every_resolve(env):
    env.config["sha256"] = {"v1": _good_hash(env)}
    first = registry.resolve("laya", "v1")
    assert registry.resolve("laya", "v1").origin == "local"
    (first.path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(CheckpointIntegrityError) as err:
        registry.resolve("laya", "v1")
    assert err.value.reason == "checkpoint_sha256_mismatch"
    assert (first.path / "model.safetensors").exists()
    assert len(env.calls) == 1


def test_bad_hash_deletes_pulled_copy(env):
    env.config["sha256"] = {"v1": "0" * 64}
    with pytest.raises(CheckpointIntegrityError) as err:
        registry.resolve("laya", "v1", required=False)
    assert err.value.reason == "checkpoint_sha256_mismatch"
    assert not registry.local_dir("laya", "v1").exists()
    assert registry.REASON_COUNTS[CheckpointReason.SHA256_MISMATCH] == 1


def test_local_copy_without_hash_is_counted_unverified(env):
    del env.config["sha256"]
    _write(registry.local_dir("laya", "v1"), {"model.safetensors": b"w"})
    got = registry.resolve("laya", "v1")
    assert not got.verified
    assert registry.REASON_COUNTS[CheckpointReason.UNVERIFIED] == 1
    assert registry.last_reason("laya", "v1") == "checkpoint_unverified"


def test_pulled_copy_without_hash_is_refused_and_deleted(env):
    del env.config["sha256"]
    with pytest.raises(CheckpointIntegrityError) as err:
        registry.resolve("laya", "v1")
    assert err.value.reason == "checkpoint_unverified_pull"
    assert not registry.local_dir("laya", "v1").exists()


def test_the_override_keeps_an_unverified_pull(env, monkeypatch):
    del env.config["sha256"]
    monkeypatch.setenv("ALLOW_UNVERIFIED_PULL", "1")
    got = registry.resolve("laya", "v1")
    assert got.origin == "hf" and not got.verified


def test_a_map_without_the_version_has_its_own_code(env):
    env.config["sha256"] = {"other": "ab" * 32}
    with pytest.raises(CheckpointIntegrityError) as err:
        registry.resolve("laya", "v1")
    assert err.value.reason == "checkpoint_unverified_pull"
    assert registry.REASON_COUNTS[CheckpointReason.UNVERIFIED_PULL] == 1
    _write(registry.local_dir("laya", "v1"), {"model.safetensors": b"w"})
    assert not registry.resolve("laya", "v1").verified
    assert registry.REASON_COUNTS[CheckpointReason.SHA256_MISSING_KEY] == 1
    env.config["sha256"] = {"v1": ""}
    registry.resolve("laya", "v1")
    assert registry.REASON_COUNTS[CheckpointReason.SHA256_MISSING_KEY] == 2


def test_plain_string_hash_applies_to_the_configured_version_only(env):
    env.config["sha256"] = "AB" * 32
    assert registry.expected_sha256("laya", "v1") == "ab" * 32
    assert registry.expected_sha256("laya", "v2") is None


def test_drive_copy_is_verified_and_filtered(env, tmp_path):
    drive = tmp_path / "drive"
    _write(drive / "laya" / "v1", HUB_FILES)
    env.settings.hf_repo = None
    env.settings.gdrive_dir = str(drive)
    env.config["sha256"] = {"v1": _good_hash(env)}
    got = registry.resolve("laya", "v1")
    assert got.origin == "gdrive" and got.verified
    assert not (got.path / "sub" / "checkpoint_latest").exists()
    shutil.rmtree(got.path)
    env.config["sha256"] = {"v1": "0" * 64}
    with pytest.raises(CheckpointIntegrityError):
        registry.resolve("laya", "v1")
    assert not got.path.exists()


# --- the hash definition ---------------------------------------------------


def _tree(tmp_path, name="t", **overrides):
    root = tmp_path / name
    shutil.rmtree(root, ignore_errors=True)
    _write(root, {**{k: v for k, v in HUB_FILES.items() if "checkpoint_latest" not in k
                     and not k.endswith(".tmp")}, **overrides})
    return root


def test_hash_ignores_resume_state_at_any_depth_and_tmp_files(tmp_path):
    base = registry._dir_sha256(_tree(tmp_path))
    full = _tree(tmp_path, "full")
    extra = {"checkpoint_latest/a.pt": b"x", "a/b/checkpoint_latest/c.pt": b"y", "x.tmp": b"z"}
    _write(full, extra)
    assert registry._dir_sha256(full) == base


def test_loader_rewrite_of_the_tokenizer_config_keeps_the_hash(tmp_path):
    before = registry._dir_sha256(_tree(tmp_path))
    rewritten = dict(TOKENIZER_CONFIG, tokenizer_class="PreTrainedTokenizerFast",
                     extra_special_tokens={"extra_0": "<a>", "extra_1": "<b>"})
    del rewritten["backend"], rewritten["is_local"]
    root = _tree(tmp_path, "after",
                 **{"tokenizer/tokenizer_config.json": json.dumps(rewritten, indent=2).encode()})
    assert registry._dir_sha256(root) == before


@pytest.mark.parametrize("key,value", [("model_max_length", 256), ("mask_token", "<m>"),
                                       ("cls_token_id", 3)])
def test_any_other_tokenizer_config_change_alters_the_hash(tmp_path, key, value):
    before = registry._dir_sha256(_tree(tmp_path))
    changed = dict(TOKENIZER_CONFIG, **{key: value})
    root = _tree(tmp_path, "c", **{"tokenizer/tokenizer_config.json": json.dumps(changed).encode()})
    assert registry._dir_sha256(root) != before


def test_registry_json_counts_without_its_digest_key(tmp_path):
    a = _tree(tmp_path, "a", **{"registry.json": b'{"is_demo": false, "sha256": "x"}'})
    b = _tree(tmp_path, "b", **{"registry.json": b'{"sha256": "y", "is_demo": false}'})
    c = _tree(tmp_path, "c", **{"registry.json": b'{"is_demo": true}'})
    assert registry._dir_sha256(a) == registry._dir_sha256(b)
    assert registry._dir_sha256(a) != registry._dir_sha256(c)


def test_path_and_digest_framing_has_no_collision(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    _write(one, {"ab": b"1", "c": b"2"})
    _write(two, {"a": b"1", "bc": b"2"})
    assert registry._dir_sha256(one) != registry._dir_sha256(two)


def test_symlinks_are_followed_by_content(tmp_path):
    outside = tmp_path / "blob"
    outside.write_bytes(b"weights")
    real = tmp_path / "real"
    _write(real, {"config.json": b"{}", "model.bin": b"weights"})
    linked = tmp_path / "linked"
    _write(linked, {"config.json": b"{}"})
    os.symlink(outside, linked / "model.bin")
    assert registry._dir_sha256(linked) == registry._dir_sha256(real)
    outside.write_bytes(b"other")
    assert registry._dir_sha256(linked) != registry._dir_sha256(real)


def test_a_directory_cycle_terminates_and_a_broken_link_is_refused(tmp_path):
    root = tmp_path / "cyc"
    _write(root, {"config.json": b"{}", "d/w.bin": b"w"})
    os.symlink(root, root / "d" / "loop")
    assert registry._dir_sha256(root)
    os.symlink(tmp_path / "missing", root / "dead")
    with pytest.raises(CheckpointIntegrityError) as err:
        registry._dir_sha256(root)
    assert err.value.reason == "checkpoint_broken_symlink"


def test_register_writes_a_digest_that_matches_the_folder(env, tmp_path):
    src = _tree(tmp_path, "src")
    got = registry.register("laya", "v1", src, metadata={"is_demo": False})
    stored = json.loads((got.path / "registry.json").read_text())
    assert stored["sha256"] == registry._dir_sha256(got.path) == got.sha256


# --- hub failures ------------------------------------------------------------


def test_missing_repo_raises_not_found(env):
    env.settings.hf_repo = None
    with pytest.raises(CheckpointNotFound, match="HF_REPO and HF_TOKEN"):
        registry.resolve("laya", "v1")
    assert env.calls == []


def test_missing_token_is_counted_only_when_the_pull_fails(env, monkeypatch):
    env.settings.hf_token = None
    env.config["sha256"] = {"v1": _good_hash(env)}
    assert registry.resolve("laya", "v1").verified
    assert registry.REASON_COUNTS[CheckpointReason.HUB_TOKEN_MISSING] == 0
    shutil.rmtree(registry.local_dir("laya", "v1"))
    import huggingface_hub

    def refuse(**kwargs):
        raise PermissionError("401 private repo")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)
    with pytest.raises(CheckpointNotFound):
        registry.resolve("laya", "v1")
    assert registry.REASON_COUNTS[CheckpointReason.HUB_TOKEN_MISSING] == 1
    assert registry.REASON_COUNTS[CheckpointReason.HUB_PULL_FAILED] == 1
    assert registry.resolve("laya", "v1", required=False) is None
    assert registry.last_reason("laya", "v1") == "hub_pull_failed"


def test_repo_without_the_folder_is_counted(env, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: None)
    assert registry.resolve("laya", "v1", required=False) is None
    assert registry.REASON_COUNTS[CheckpointReason.HUB_PATH_ABSENT] == 1


# --- CLI -----------------------------------------------------------------------


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_checkpoint_hash_prints_the_digest_and_lists_folders_on_a_miss(env):
    from typer.testing import CliRunner

    from modeling.cli import app

    env.config["sha256"] = {"v1": _good_hash(env)}
    registry.resolve("laya", "v1")
    runner = CliRunner()
    ok = runner.invoke(app, ["checkpoint-hash", "laya", "--version", "v1"])
    assert ok.exit_code == 0, ok.output
    assert env.config["sha256"]["v1"] in ok.output
    miss = runner.invoke(app, ["checkpoint-hash", "laya", "--version", "v9"])
    assert miss.exit_code == 1
    assert "v1" in _plain(miss.output)


def test_checkpoint_pull_fetches_and_verifies(env):
    from typer.testing import CliRunner

    from modeling.cli import app

    env.config["sha256"] = {"v1": _good_hash(env)}
    result = CliRunner().invoke(app, ["checkpoint-pull", "laya", "--version", "v1"])
    assert result.exit_code == 0, result.output
    assert "origin=hf" in _plain(result.output) and "verified" in _plain(result.output)
    shutil.rmtree(registry.local_dir("laya", "v1"))
    env.config["sha256"] = {"v1": "0" * 64}
    bad = CliRunner().invoke(app, ["checkpoint-pull", "laya", "--version", "v1"])
    assert bad.exit_code == 2


# --- the Laya stage config -------------------------------------------------------


def _laya_settings(env, value="laya/v1"):
    return SimpleNamespace(models_dir=env.settings.models_dir, laya_hi_checkpoint=value)


@pytest.fixture
def laya_cfg(env, monkeypatch):
    import modeling.config as config_module

    monkeypatch.setattr(
        config_module, "module_config",
        lambda module: {"finetuned_checkpoints": {"hi": None}, **env.config},
    )


def test_laya_config_local_good_hash(env, laya_cfg):
    from modeling import scoring

    env.config["sha256"] = {"v1": _good_hash(env)}
    registry.resolve("laya", "v1")
    cfg = scoring._laya_stage_config(_laya_settings(env))
    assert cfg["finetuned_checkpoints"]["hi"] == str(registry.local_dir("laya", "v1"))
    assert "_registry_refusals" not in cfg


def test_laya_config_local_bad_hash_is_a_reason_coded_refusal(env, laya_cfg):
    from modeling import scoring

    env.config["sha256"] = {"v1": _good_hash(env)}
    got = registry.resolve("laya", "v1")
    (got.path / "model.safetensors").write_bytes(b"tampered")
    cfg = scoring._laya_stage_config(_laya_settings(env))
    reason, message = cfg["_registry_refusals"]["hi"]
    assert reason == "checkpoint_sha256_mismatch" and "expects" in message
    refused, usable = scoring._laya_refusals(
        SimpleNamespace(available=lambda: SimpleNamespace(ok=True)), ["hi"],
        cfg["_registry_refusals"],
    )
    assert refused["hi"][0] == "checkpoint_sha256_mismatch" and not usable


def test_laya_config_pulls_an_absent_folder(env, laya_cfg):
    from modeling import scoring

    env.config["sha256"] = {"v1": _good_hash(env)}
    cfg = scoring._laya_stage_config(_laya_settings(env))
    assert Path(cfg["finetuned_checkpoints"]["hi"]).is_dir()
    assert len(env.calls) == 1
