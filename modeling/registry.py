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

Every origin is verified against the optional ``sha256`` in ``configs/models.yaml``.
A mismatch raises :class:`CheckpointIntegrityError` with reason code
``checkpoint_sha256_mismatch``, and a copy that came from the Hub or Drive is deleted.
A checkpoint with no configured hash resolves with ``verified=False`` and counts
``checkpoint_unverified``. Every reason code is counted in :data:`REASON_COUNTS`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modeling.checkpoint_identity import CHECKPOINT_LOADER_REWRITTEN
from modeling.config import get_settings, module_config

log = logging.getLogger(__name__)


class CheckpointReason:
    """Counted reason codes of the checkpoint registry."""

    #: The directory hash differs from the ``sha256`` in ``configs/models.yaml``.
    SHA256_MISMATCH = "checkpoint_sha256_mismatch"
    #: No ``sha256`` is configured, so the weights are used without a check.
    UNVERIFIED = "checkpoint_unverified"
    #: ``HF_REPO`` is set and ``HF_TOKEN`` is not. A private repo refuses the pull.
    HUB_TOKEN_MISSING = "hub_token_missing"
    #: ``HF_REPO`` is set and ``huggingface_hub`` is not installed.
    HUB_NOT_INSTALLED = "hub_not_installed"
    #: The Hub call raised. Typical causes are a wrong repo id, a bad token and no network.
    HUB_PULL_FAILED = "hub_pull_failed"
    #: The Hub call worked and the repo has no ``<module>/<version>/`` folder.
    HUB_PATH_ABSENT = "hub_path_absent"


#: Reason code to count since the process started. Read by tests and by the log summary.
REASON_COUNTS: Counter[str] = Counter()

#: Training resume state. Inference does not read it, so it is neither pulled nor hashed.
RESUME_DIR = "checkpoint_latest"
#: Written by ``register`` after the hash is taken, so it cannot be inside the hash.
REGISTRY_FILE = "registry.json"
#: Hub patterns for a pull. ``*`` matches across ``/``, so sub-folders such as
#: ``tokenizer/`` and the Laya calibrator files come with the weights.
HUB_IGNORE_PATTERNS = [f"*/{RESUME_DIR}/*", f"{RESUME_DIR}/*", "*.tmp"]


def _count(reason: str, detail: str = "") -> None:
    REASON_COUNTS[reason] += 1
    log.warning("registry reason=%s %s", reason, detail)


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


def _hashed_files(path: Path) -> list[Path]:
    """The files that make up a checkpoint's hash, sorted by relative path.

    Excluded: the ``checkpoint_latest/`` resume folder, ``registry.json`` (written after
    the hash is taken), ``*.tmp`` files, and the tokenizer config that the Laya loader
    rewrites in place on first load. Everything else counts, including the Laya
    calibrator files and the tokenizer folder.
    """
    skipped = CHECKPOINT_LOADER_REWRITTEN | {REGISTRY_FILE}
    out = []
    for file in path.rglob("*"):
        if not file.is_file() or file.name.endswith(".tmp"):
            continue
        relative = file.relative_to(path)
        if relative.parts[0] == RESUME_DIR or relative.as_posix() in skipped:
            continue
        out.append(file)
    return sorted(out, key=lambda f: f.relative_to(path).as_posix())


def _dir_sha256(path: Path) -> str | None:
    """Checksum a checkpoint directory: sha256 over (relative path, file digest) pairs.

    The relative path is POSIX style. The file digest is the sha256 of the file bytes.
    :func:`_hashed_files` lists which files count.
    """
    if not path.exists():
        return None
    files = _hashed_files(path)
    if not files:
        return None
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        file_digest = hashlib.sha256()
        with file.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                file_digest.update(block)
        digest.update(file_digest.hexdigest().encode("ascii"))
    return digest.hexdigest()


def directory_sha256(module: str, version: str | None = None) -> tuple[Path, str | None]:
    """The local directory of a checkpoint and its hash. Never pulls. Backs ``checkpoint-hash``."""
    resolved_version = version or str(module_config(module).get("version", "v0.0.0-unset"))
    path = local_dir(module, resolved_version)
    return path, _dir_sha256(path)


def expected_sha256(module: str, version: str) -> str | None:
    """The ``sha256`` configured for ``module`` at ``version``, or ``None``.

    ``sha256`` in ``configs/models.yaml`` is a map from version to hex digest. A plain
    string is accepted and applies to the version configured for the module.
    """
    config = module_config(module)
    value = config.get("sha256")
    if isinstance(value, dict):
        value = value.get(version)
    elif value is not None and version != str(config.get("version")):
        value = None
    return str(value).strip().lower() if value else None


def _verify(module: str, version: str, target: Path, *, discard_on_fail: bool) -> Checkpoint:
    """Hash ``target`` and compare to the configured value. Raises on a mismatch."""
    digest = _dir_sha256(target)
    expected = expected_sha256(module, version)
    if expected is None:
        _count(CheckpointReason.UNVERIFIED, f"{module}/{version} has no sha256 in models.yaml")
        return Checkpoint(module, version, target, "local", digest, verified=False)
    if digest != expected:
        message = (
            f"{CheckpointReason.SHA256_MISMATCH}: {module}/{version} at {target} hashes to "
            f"{digest}, configs/models.yaml expects {expected}."
        )
        if discard_on_fail:
            shutil.rmtree(target, ignore_errors=True)
            message += " The pulled copy is deleted."
        _count(CheckpointReason.SHA256_MISMATCH, message)
        raise CheckpointIntegrityError(message)
    return Checkpoint(module, version, target, "local", digest, verified=True)


def _try_hub(module: str, version: str, target: Path) -> bool:
    settings = get_settings()
    if not settings.hf_repo:
        return False
    if not settings.hf_token:
        _count(CheckpointReason.HUB_TOKEN_MISSING, "HF_REPO is set and HF_TOKEN is not")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _count(CheckpointReason.HUB_NOT_INSTALLED, "install huggingface_hub")
        return False
    try:
        snapshot_download(
            repo_id=settings.hf_repo,
            allow_patterns=[f"{module}/{version}/*"],
            ignore_patterns=HUB_IGNORE_PATTERNS,
            local_dir=str(settings.models_dir / ".hub"),
            token=settings.hf_token,
        )
    except Exception as exc:
        _count(CheckpointReason.HUB_PULL_FAILED, f"{module}/{version}: {exc}")
        return False
    staged = settings.models_dir / ".hub" / module / version
    if not staged.exists():
        _count(CheckpointReason.HUB_PATH_ABSENT, f"{module}/{version} is not in the Hub repo")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(staged, target, ignore=shutil.ignore_patterns(RESUME_DIR))
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
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return True


def resolve(module: str, version: str | None = None, *, required: bool = True) -> Checkpoint | None:
    """Resolve a checkpoint, pulling it into the local cache if needed.

    ``required=False`` returns ``None`` instead of raising -- that is the path
    the batch scorer takes, so a corpus can still be scored for the modules that
    *are* trained while an untrained module writes honest nulls.
    """
    resolved_version = version or str(module_config(module).get("version", "v0.0.0-unset"))
    target = local_dir(module, resolved_version)

    if target.exists() and any(target.iterdir()):
        # A local copy is never deleted on a mismatch. It may be the only copy.
        return _verify(module, resolved_version, target, discard_on_fail=False)

    for origin, fetch in (("hf", _try_hub), ("gdrive", _try_gdrive)):
        if fetch(module, resolved_version, target):
            log.info("pulled %s/%s from %s -> %s", module, resolved_version, origin, target)
            checked = _verify(module, resolved_version, target, discard_on_fail=True)
            return Checkpoint(
                module, resolved_version, target, origin, checked.sha256, checked.verified
            )

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
        f"  reasons counted so far: {dict(REASON_COUNTS) or 'none'}"
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
                dest = target / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        else:
            shutil.copy2(source, target / source.name)

    digest = _dir_sha256(target)
    (target / "registry.json").write_text(
        json.dumps(
            {"module": module, "version": version, "sha256": digest, **(metadata or {})},
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
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
        for version_dir in sorted(p for p in module_dir.iterdir() if p.is_dir()):
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
