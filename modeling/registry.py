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
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modeling.config import get_settings, module_config

log = logging.getLogger(__name__)


class CheckpointNotFound(FileNotFoundError):
    """No local, Hub or Drive copy of a requested checkpoint."""


@dataclass(frozen=True)
class Checkpoint:
    """A resolved checkpoint and everything a model card needs to cite it."""

    module: str
    version: str
    path: Path
    origin: str  # "local" | "hf" | "gdrive" | "fresh"
    sha256: str | None = None

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
        }


def local_dir(module: str, version: str) -> Path:
    return get_settings().models_dir / module / version


def _dir_sha256(path: Path) -> str | None:
    """Checksum a checkpoint directory, the same way Phase 1 checksums an
    artifact tree: sha256 over (relative path, file digest) pairs."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        return None
    for file in files:
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        file_digest = hashlib.sha256()
        with file.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                file_digest.update(block)
        digest.update(file_digest.hexdigest().encode("ascii"))
    return digest.hexdigest()


def _try_hub(module: str, version: str, target: Path) -> bool:
    settings = get_settings()
    if not settings.hf_repo:
        return False
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.warning("HF_REPO is set but huggingface_hub is not installed; skipping Hub lookup")
        return False
    try:
        snapshot_download(
            repo_id=settings.hf_repo,
            allow_patterns=[f"{module}/{version}/*"],
            local_dir=str(settings.models_dir / ".hub"),
            token=settings.hf_token,
        )
    except Exception as exc:
        log.warning("Hub lookup for %s/%s failed: %s", module, version, exc)
        return False
    staged = settings.models_dir / ".hub" / module / version
    if not staged.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(staged, target)
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
        return Checkpoint(module, resolved_version, target, "local", _dir_sha256(target))

    for origin, fetch in (("hf", _try_hub), ("gdrive", _try_gdrive)):
        if fetch(module, resolved_version, target):
            log.info("pulled %s/%s from %s -> %s", module, resolved_version, origin, target)
            return Checkpoint(module, resolved_version, target, origin, _dir_sha256(target))

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
        f"  or point HF_REPO / GDRIVE_DIR at an existing copy."
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
