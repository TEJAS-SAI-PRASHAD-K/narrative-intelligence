"""Checkpoint resolution: name + version -> a local path.

``models/`` is gitignored. Weights never enter git; the repo commits the
*pointer* (name, version, sha256) in ``configs/models.yaml`` and this module
turns that pointer into a file on disk, pulling from the Hugging Face Hub or a
Google Drive mount if the local cache is cold.

Resolution order, first hit wins:

1. ``models/<module>/<version>/`` -- the local cache, populated by a prior
   resolve or by a Colab run that saved here.
2. Hugging Face Hub, private repo from ``HF_REPO``, path ``<module>/<version>``.
3. ``GDRIVE_DIR/<module>/<version>/`` -- the Colab escape hatch.

If none hit, :class:`CheckpointNotFound` is raised with the exact command that
would produce it. A missing checkpoint must never silently degrade into an
untrained model producing confident garbage.

Every origin is verified against the optional ``sha256`` map in ``configs/models.yaml``.
A mismatch raises :class:`CheckpointIntegrityError` with reason code
``checkpoint_sha256_mismatch``, and a copy that came from the Hub or Drive is deleted.
A local copy with no configured hash resolves with ``verified=False`` and counts
``checkpoint_unverified``. A pulled copy with no configured hash is refused and deleted
with ``checkpoint_unverified_pull``, unless ``ALLOW_UNVERIFIED_PULL=1``. Every reason code
is counted in :data:`REASON_COUNTS`, and :func:`last_reason` gives the latest one for a
checkpoint, so a stage note can name it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modeling.config import get_settings, module_config

log = logging.getLogger(__name__)


class CheckpointReason:
    """Counted reason codes of the checkpoint registry."""

    #: The directory hash differs from the ``sha256`` in ``configs/models.yaml``.
    SHA256_MISMATCH = "checkpoint_sha256_mismatch"
    #: A local copy has no ``sha256`` configured, so the weights are used without a check.
    UNVERIFIED = "checkpoint_unverified"
    #: A pulled copy has no ``sha256`` configured. It is deleted unless
    #: ``ALLOW_UNVERIFIED_PULL=1``.
    UNVERIFIED_PULL = "checkpoint_unverified_pull"
    #: A ``sha256`` map exists, and it has no value for the requested version.
    SHA256_MISSING_KEY = "checkpoint_sha256_missing_key"
    #: A symbolic link in the checkpoint points at nothing.
    BROKEN_SYMLINK = "checkpoint_broken_symlink"
    #: A directory symbolic link is inside the checkpoint. The hash would depend on link names.
    DIR_SYMLINK = "checkpoint_dir_symlink"
    #: The folder has no file that the hash covers. It is empty or holds only excluded entries.
    EMPTY = "checkpoint_empty"
    #: ``HF_REPO`` is set, ``HF_TOKEN`` is not, and the anonymous pull then failed.
    HUB_TOKEN_MISSING = "hub_token_missing"
    #: ``HF_REPO`` is set and ``huggingface_hub`` is not installed.
    HUB_NOT_INSTALLED = "hub_not_installed"
    #: The Hub call raised. Typical causes are a wrong repo id, a bad token and no network.
    HUB_PULL_FAILED = "hub_pull_failed"
    #: The Hub call worked and the repo has no ``<module>/<version>/`` folder.
    HUB_PATH_ABSENT = "hub_path_absent"


#: Reason code to count since the process started. Tests read it.
REASON_COUNTS: Counter[str] = Counter()
#: ``(module, version)`` to the latest reason code. Read through :func:`last_reason`.
_LAST_REASON: dict[tuple[str, str], str] = {}

#: Training resume state. Inference does not read it, so it is neither pulled nor hashed.
RESUME_DIR = "checkpoint_latest"
#: Written by ``register`` after the hash is taken. Its hash form leaves out the digest key.
REGISTRY_FILE = "registry.json"
#: The tokenizer config that the Laya loader rewrites in place on first load.
TOKENIZER_CONFIG = "tokenizer/tokenizer_config.json"
#: Keys of ``registry.json`` that hold a digest of the folder. They stay out of the hash.
_DIGEST_KEYS = ("sha256", "digest")
#: Hub patterns for a pull. ``*`` matches across ``/``, so a folder at any depth is covered.
HUB_IGNORE_PATTERNS = [f"{RESUME_DIR}/*", f"*/{RESUME_DIR}/*", "*.tmp"]


def excluded_name(name: str) -> bool:
    """The one exclusion rule: resume folders at any depth, and files and folders ``*.tmp``.

    The name test does not tell a file from a folder, so a folder called ``x.tmp`` is
    excluded too. The hash, the Hub pull, the local copy and the Drive copy all use it.
    """
    return name == RESUME_DIR or name.endswith(".tmp")


def _ignore_excluded(_directory: str, names: list[str]) -> set[str]:
    """``shutil.copytree`` ``ignore`` callback built on :func:`excluded_name`."""
    return {name for name in names if excluded_name(name)}


def _count(
    reason: str, detail: str = "", *, module: str | None = None, version: str | None = None
) -> None:
    REASON_COUNTS[reason] += 1
    if module and version:
        _LAST_REASON[(module, version)] = reason
    log.warning("registry reason=%s %s", reason, detail)


def last_reason(module: str, version: str) -> str | None:
    """The latest registry reason code for a checkpoint, or ``None``. For stage notes."""
    return _LAST_REASON.get((module, version))


class CheckpointNotFound(FileNotFoundError):
    """No local, Hub or Drive copy of a requested checkpoint."""


class CheckpointIntegrityError(RuntimeError):
    """A checkpoint exists and its hash differs from the configured one."""

    def __init__(self, message: str, reason: str = CheckpointReason.SHA256_MISMATCH) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Checkpoint:
    """A resolved checkpoint and everything a model card needs to cite it."""

    module: str
    version: str
    path: Path
    origin: str  # "local" | "hf" | "gdrive" | "fresh"
    sha256: str | None = None
    #: True when ``sha256`` matched the value configured in ``configs/models.yaml``.
    verified: bool = False

    @property
    def uri(self) -> str:
        """The citable location, for the model card's "checkpoint URI" line."""
        if self.origin == "hf":
            repo = get_settings().hf_repo or "<HF_REPO unset>"
            return f"hf://{repo}/{self.module}/{self.version}"
        if self.origin == "gdrive":
            return f"gdrive://{self.module}/{self.version}"
        return f"file://{self.path}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "version": self.version,
            "uri": self.uri,
            "origin": self.origin,
            "sha256": self.sha256,
            "verified": self.verified,
        }


def local_dir(module: str, version: str) -> Path:
    return get_settings().models_dir / module / version


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _normalized_tokenizer_config(data: bytes) -> bytes:
    """The tokenizer config after the Laya loader's idempotent rewrite, as canonical JSON.

    The loader (``laya.agent._fix_tokenizer_config``) sets ``tokenizer_class`` to
    ``PreTrainedTokenizerFast`` when it is absent or ``TokenizersBackend``, drops
    ``backend`` and ``is_local`` then, and turns a list ``extra_special_tokens`` into a
    mapping. This function applies the same rules in memory, so the hash is the same
    before and after the loader runs, and any other key change alters it. Bytes that are
    not a JSON object are returned unchanged.
    """
    try:
        config = json.loads(data)
    except ValueError:
        return data
    if not isinstance(config, dict):
        return data
    if config.get("tokenizer_class") in (None, "TokenizersBackend"):
        config["tokenizer_class"] = "PreTrainedTokenizerFast"
        config.pop("backend", None)
        config.pop("is_local", None)
    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        config["extra_special_tokens"] = {f"extra_{i}": t for i, t in enumerate(extra)}
    return _canonical_json(config)


def _registry_json_form(data: bytes) -> bytes:
    """``registry.json`` as canonical JSON without its digest keys.

    The file carries ``is_demo``, which gates scoring, so its content counts. The digest
    keys are removed because they hold a hash of the folder, written after the hash is taken.
    """
    try:
        meta = json.loads(data)
    except ValueError:
        return data
    if not isinstance(meta, dict):
        return data
    for key in _DIGEST_KEYS:
        meta.pop(key, None)
    return _canonical_json(meta)


def _file_bytes_digest(file: Path, relative: str) -> str:
    """sha256 of a file, in the normalized form for the two files that have one."""
    if relative in (TOKENIZER_CONFIG, REGISTRY_FILE):
        data = file.read_bytes()
        form = _normalized_tokenizer_config if relative == TOKENIZER_CONFIG else _registry_json_form
        return hashlib.sha256(form(data)).hexdigest()
    digest = hashlib.sha256()
    with file.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashed_files(path: Path) -> list[tuple[str, Path]]:
    """``(relative POSIX path, file)`` pairs that make up a checkpoint's hash, sorted.

    A symbolic link to a file counts by the content of its target. A link that points at
    nothing raises ``checkpoint_broken_symlink``. A symbolic link to a directory raises
    ``checkpoint_dir_symlink``, whatever its target. A sibling link and a link to an ancestor
    are both refused, so the hash never depends on link names and the walk cannot loop.
    Excluded by :func:`excluded_name`: ``checkpoint_latest/`` at any depth, and files and
    folders that end in ``.tmp``.
    """
    out: list[tuple[str, Path]] = []
    for root, dirs, names in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not excluded_name(d))
        for name in dirs:
            link = Path(root) / name
            if link.is_symlink():
                _count(CheckpointReason.DIR_SYMLINK, str(link))
                raise CheckpointIntegrityError(
                    f"{CheckpointReason.DIR_SYMLINK}: {link} is a link to a directory. "
                    "Copy the real folder into the checkpoint.",
                    CheckpointReason.DIR_SYMLINK,
                )
        for name in names:
            if excluded_name(name):
                continue
            file = Path(root) / name
            if not file.exists():
                _count(CheckpointReason.BROKEN_SYMLINK, str(file))
                raise CheckpointIntegrityError(
                    f"{CheckpointReason.BROKEN_SYMLINK}: {file} points at nothing.",
                    CheckpointReason.BROKEN_SYMLINK,
                )
            out.append((file.relative_to(path).as_posix(), file))
    return sorted(out)


def _dir_sha256(path: Path) -> str | None:
    """Checksum a checkpoint directory.

    For each file in :func:`_hashed_files` order, the hash takes the byte length of the
    relative POSIX path as 8 bytes, the path, a NUL byte and the file digest as 32 raw
    bytes. The length prefix makes the framing unambiguous. The file digest is the sha256 of
    the file bytes. ``tokenizer/tokenizer_config.json`` and ``registry.json`` use the
    normalized forms of :func:`_normalized_tokenizer_config` and :func:`_registry_json_form`.
    """
    if not path.exists():
        return None
    files = _hashed_files(path)
    if not files:
        return None
    digest = hashlib.sha256()
    for relative, file in files:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big") + encoded + b"\0")
        digest.update(bytes.fromhex(_file_bytes_digest(file, relative)))
    return digest.hexdigest()


def directory_sha256(module: str, version: str | None = None) -> tuple[Path, str | None]:
    """The local directory of a checkpoint and its hash. Never pulls. Backs ``checkpoint-hash``."""
    resolved_version = version or str(module_config(module).get("version", "v0.0.0-unset"))
    path = local_dir(module, resolved_version)
    return path, _dir_sha256(path)


def sibling_folders(module: str) -> list[str]:
    """Folder names under ``models/<module>/``. Lists the choices when a version is wrong."""
    root = get_settings().models_dir / module
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def sha256_versions(module: str) -> list[str]:
    """Folder names that have a ``sha256`` in ``configs/models.yaml``. Empty for no map."""
    value = module_config(module).get("sha256")
    return sorted(str(k) for k in value) if isinstance(value, dict) else []


def _expected(module: str, version: str) -> tuple[str | None, str | None]:
    """``(digest, reason)``. ``reason`` is set when the digest is missing.

    No ``sha256`` key gives ``checkpoint_unverified`` as the reason. A ``sha256`` map that
    has no value for ``version``, or an empty one, gives ``checkpoint_sha256_missing_key``.
    """
    config = module_config(module)
    if config.get("sha256") is None:
        return None, CheckpointReason.UNVERIFIED
    value = config["sha256"]
    if isinstance(value, dict):
        value = value.get(version)
    elif version != str(config.get("version")):
        value = None
    text = str(value).strip().lower() if value else ""
    return (text or None), (None if text else CheckpointReason.SHA256_MISSING_KEY)


def expected_sha256(module: str, version: str) -> str | None:
    """The ``sha256`` configured for ``module`` at ``version``, or ``None``.

    ``sha256`` in ``configs/models.yaml`` is a map from version to hex digest. A plain
    string is accepted and applies to the version configured for the module.
    """
    return _expected(module, version)[0]


def allow_unverified_pull() -> bool:
    """True when ``ALLOW_UNVERIFIED_PULL`` is ``1`` in the environment or in ``.env``."""
    settings_value = getattr(get_settings(), "allow_unverified_pull", False)
    return bool(settings_value) or os.environ.get("ALLOW_UNVERIFIED_PULL") == "1"


def _verify(
    module: str, version: str, target: Path, *, pulled: bool, origin: str = "local"
) -> Checkpoint:
    """Hash ``target`` and compare to the configured value. Raises on a refusal.

    A pulled copy that is refused is deleted. A local copy is kept, because it may be the
    only copy.
    """
    digest = _dir_sha256(target)
    expected, missing = _expected(module, version)
    if digest is None:
        if pulled:
            shutil.rmtree(target, ignore_errors=True)
        message = (
            f"{module}/{version} at {target} has no file that the hash covers. "
            "The folder is empty or holds only excluded entries."
        )
        _count(CheckpointReason.EMPTY, message, module=module, version=version)
        raise CheckpointIntegrityError(
            f"{CheckpointReason.EMPTY}: {message}", CheckpointReason.EMPTY
        )

    def refuse(reason: str, message: str) -> CheckpointIntegrityError:
        if pulled:
            shutil.rmtree(target, ignore_errors=True)
            message += " The pulled copy is deleted."
        _count(reason, message, module=module, version=version)
        return CheckpointIntegrityError(f"{reason}: {message}", reason)

    if expected is None:
        if pulled and not allow_unverified_pull():
            raise refuse(
                CheckpointReason.UNVERIFIED_PULL,
                f"{module}/{version} came from {origin} and has no sha256 ({missing}). "
                "Set it in configs/models.yaml, or set ALLOW_UNVERIFIED_PULL=1.",
            )
        _count(
            CheckpointReason.UNVERIFIED if missing == CheckpointReason.UNVERIFIED else missing,
            f"{module}/{version} has no usable sha256 in models.yaml",
            module=module,
            version=version,
        )
        return Checkpoint(module, version, target, origin, digest, verified=False)
    if digest != expected:
        raise refuse(
            CheckpointReason.SHA256_MISMATCH,
            f"{module}/{version} at {target} hashes to {digest}, "
            f"configs/models.yaml expects {expected}.",
        )
    return Checkpoint(module, version, target, origin, digest, verified=True)


def _install_copy(source: Path, target: Path) -> None:
    """Copy ``source`` to ``target`` through a staging folder, so ``target`` is all or nothing.

    A copy that stops half way leaves no ``target``. The next resolve then pulls again
    instead of refusing a partial folder as a hash mismatch.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        shutil.copytree(source, staging, ignore=_ignore_excluded)
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _try_hub(module: str, version: str, target: Path) -> bool:
    settings = get_settings()
    if not settings.hf_repo:
        return False
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _count(CheckpointReason.HUB_NOT_INSTALLED, "install huggingface_hub",
               module=module, version=version)
        return False
    revision = module_config(module).get("hf_revision") or getattr(settings, "hf_revision", None)
    try:
        snapshot_download(
            repo_id=settings.hf_repo,
            allow_patterns=[f"{module}/{version}/*"],
            ignore_patterns=HUB_IGNORE_PATTERNS,
            local_dir=str(settings.models_dir / ".hub"),
            token=settings.hf_token,
            revision=revision,
        )
    except Exception as exc:
        _count(CheckpointReason.HUB_PULL_FAILED, f"{module}/{version}: {exc}",
               module=module, version=version)
        # Counted last, so it is the reason a stage note shows. A missing token is the
        # cause that the user can act on.
        if not settings.hf_token:
            _count(CheckpointReason.HUB_TOKEN_MISSING, "HF_REPO is set, HF_TOKEN is not",
                   module=module, version=version)
        return False
    staged = settings.models_dir / ".hub" / module / version
    if not staged.exists():
        _count(CheckpointReason.HUB_PATH_ABSENT, f"{module}/{version} is not in the Hub repo",
               module=module, version=version)
        return False
    _install_copy(staged, target)
    # The staging copy is deleted so a failed check leaves nothing to reuse.
    shutil.rmtree(staged, ignore_errors=True)
    return True


def _try_gdrive(module: str, version: str, target: Path) -> bool:
    settings = get_settings()
    if not settings.gdrive_dir:
        return False
    source = Path(settings.gdrive_dir) / module / version
    if not source.exists():
        return False
    _install_copy(source, target)
    return True


def resolve(module: str, version: str | None = None, *, required: bool = True) -> Checkpoint | None:
    """Resolve a checkpoint, pulling it into the local cache if needed.

    ``required=False`` returns ``None`` instead of raising :class:`CheckpointNotFound` --
    that is the path the batch scorer takes, so a corpus can still be scored for the
    modules that *are* trained while an untrained module writes honest nulls. It does not
    change an integrity failure: :class:`CheckpointIntegrityError` is raised either way,
    because a checkpoint that fails its hash must never be scored with or skipped quietly.
    """
    resolved_version = version or str(module_config(module).get("version", "v0.0.0-unset"))
    target = local_dir(module, resolved_version)

    if target.exists() and any(target.iterdir()):
        # A local copy is never deleted on a mismatch. It may be the only copy.
        return _verify(module, resolved_version, target, pulled=False)

    for origin, fetch in (("hf", _try_hub), ("gdrive", _try_gdrive)):
        if fetch(module, resolved_version, target):
            log.info("pulled %s/%s from %s -> %s", module, resolved_version, origin, target)
            return _verify(module, resolved_version, target, pulled=True, origin=origin)

    if not required:
        log.warning(
            "no checkpoint for %s/%s; that module will write nulls. Train it with "
            "`python -m modeling.cli train %s`.",
            module,
            resolved_version,
            module,
        )
        return None

    raise CheckpointNotFound(
        f"no checkpoint for {module}/{resolved_version}.\n"
        f"  looked in: {target}, hf://{get_settings().hf_repo}, {get_settings().gdrive_dir}\n"
        f"  produce it with: python -m modeling.cli train {module}\n"
        f"  or set HF_REPO and HF_TOKEN in .env to pull it from the private Hub repo.\n"
        f"  last reason: {last_reason(module, resolved_version) or 'none'}"
    )


def register(
    module: str, version: str, source: Path, *, metadata: dict[str, Any] | None = None
) -> Checkpoint:
    """Install a freshly-trained checkpoint into the local cache.

    Writes a ``registry.json`` beside the weights holding the metadata a model
    card needs. Called at the end of every ``train`` command, before the Colab
    runtime has a chance to die.
    """
    target = local_dir(module, version)
    target.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    if source.resolve() != target.resolve():
        if source.is_dir():
            for item in source.iterdir():
                if excluded_name(item.name):
                    continue
                dest = target / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest, ignore=_ignore_excluded)
                else:
                    shutil.copy2(item, dest)
        else:
            shutil.copy2(source, target / source.name)

    # registry.json counts in the hash without its digest key, so it is written first,
    # the folder is hashed, and the digest is added.
    record = {"module": module, "version": version, **(metadata or {})}
    (target / REGISTRY_FILE).write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    digest = _dir_sha256(target)
    record["sha256"] = digest
    (target / REGISTRY_FILE).write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    log.info("registered %s/%s at %s (sha256=%s)", module, version, target, (digest or "")[:12])
    return Checkpoint(module, version, target, "fresh", digest)


def metadata(module: str, version: str | None = None) -> dict[str, Any]:
    """Read back a registered checkpoint's metadata (metrics, split, data hash)."""
    resolved_version = version or str(module_config(module).get("version", "v0.0.0-unset"))
    path = local_dir(module, resolved_version) / "registry.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def list_local() -> list[dict[str, Any]]:
    """Everything in the local cache. Backs ``modeling.cli registry``."""
    root = get_settings().models_dir
    if not root.exists():
        return []
    out = []
    modules = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    for module_dir in modules:
        versions = [p for p in module_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
        for version_dir in sorted(versions):
            meta = metadata(module_dir.name, version_dir.name)
            out.append(
                {
                    "module": module_dir.name,
                    "version": version_dir.name,
                    "path": str(version_dir),
                    "sha256": (meta.get("sha256") or "")[:12],
                    "trained_at": meta.get("trained_at"),
                }
            )
    return out
