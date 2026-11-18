"""Score with a candidate checkpoint without editing ``configs/models.yaml``.

A candidate is a checkpoint folder ``models/<module>/<VERSION>/`` that is not the configured
version. Two settings select it for one run:

``MISINFO_CANDIDATE_VERSION``
    The folder name. ``modeling.cli score --misinfo-version`` sets it.
``MISINFO_CANDIDATE_ALLOW_UNVERIFIED``
    ``1`` lets a candidate with no ``sha256`` in ``configs/models.yaml`` load.
    ``--allow-unverified-candidate`` sets it.
``MISINFO_CANDIDATE_REPLACE_OTHER_VERSION``
    ``1`` lets the stage overwrite ``misinfo_prob`` of rows that carry another misinfo stamp.
    ``--replace-other-version`` sets it.

``scoped_request`` sets the variables for one call and restores them after it.

The registry hash check still runs. A candidate with a configured hash must match it. A
candidate with no configured hash is refused unless the flag is set. With the flag, the load
works, the log carries a warning, and the registry counts ``checkpoint_unverified``. The
scored rows carry ``<VERSION>+<sha8>``, so no row looks like the configured version.

A candidate is never pulled from the Hub or Drive. It exists on local disk or it does not.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from collections.abc import Iterator

from modeling.config import module_config
from modeling.registry import (
    Checkpoint,
    CheckpointIntegrityError,
    _count,
    _dir_sha256,
    _expected,
    _verify,
    local_dir,
)

log = logging.getLogger(__name__)

VERSION_ENV = "MISINFO_CANDIDATE_VERSION"
ALLOW_UNVERIFIED_ENV = "MISINFO_CANDIDATE_ALLOW_UNVERIFIED"
REPLACE_ENV = "MISINFO_CANDIDATE_REPLACE_OTHER_VERSION"
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


class CandidateReason:
    """Counted reason codes of the candidate path."""

    INVALID_VERSION = "candidate_version_invalid"
    IS_CONFIGURED_VERSION = "candidate_is_configured_version"
    NOT_FOUND = "candidate_checkpoint_not_found"
    UNVERIFIED_REFUSED = "candidate_unverified_refused"
    #: Rows on disk carry another misinfo stamp, and the run did not ask to replace them.
    REPLACE_REQUIRED = "candidate_replace_other_version_required"


class CandidateRefused(RuntimeError):
    """A candidate cannot load. ``reason`` is the code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def requested_version() -> str | None:
    """The candidate folder name from the environment, or ``None``."""
    value = os.environ.get(VERSION_ENV, "").strip()
    return value or None


def allow_unverified() -> bool:
    return os.environ.get(ALLOW_UNVERIFIED_ENV) == "1"


def replace_other_version() -> bool:
    return os.environ.get(REPLACE_ENV) == "1"


@contextlib.contextmanager
def scoped_request(
    version: str | None, *, unverified: bool = False, replace_other: bool = False
) -> Iterator[None]:
    """Put the candidate options into the environment for the stages, and restore it after.

    The stages read the options from the environment. The call never leaves them set.
    """
    names = (VERSION_ENV, ALLOW_UNVERIFIED_ENV, REPLACE_ENV)
    saved = {name: os.environ.get(name) for name in names}
    try:
        if version:
            os.environ[VERSION_ENV] = version
        if unverified:
            os.environ[ALLOW_UNVERIFIED_ENV] = "1"
        if replace_other:
            os.environ[REPLACE_ENV] = "1"
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def resolve_candidate(
    module: str, version: str, *, unverified_ok: bool | None = None
) -> Checkpoint:
    """Find, verify and return the candidate checkpoint. Raise :class:`CandidateRefused`."""
    allowed = allow_unverified() if unverified_ok is None else unverified_ok
    if not _VERSION_PATTERN.fullmatch(version) or ".." in version:
        _count(CandidateReason.INVALID_VERSION, version, module=module, version=version)
        raise CandidateRefused(
            CandidateReason.INVALID_VERSION, f"{version!r} is not a checkpoint folder name"
        )
    configured = str(module_config(module).get("version", ""))
    if version == configured:
        _count(CandidateReason.IS_CONFIGURED_VERSION, version, module=module, version=version)
        raise CandidateRefused(
            CandidateReason.IS_CONFIGURED_VERSION,
            f"{version} is the configured version of {module}. Use the normal path.",
        )
    target = local_dir(module, version)
    if not target.is_dir() or not any(target.iterdir()):
        _count(CandidateReason.NOT_FOUND, str(target), module=module, version=version)
        raise CandidateRefused(
            CandidateReason.NOT_FOUND,
            f"no checkpoint at {target}. A candidate is never pulled. "
            f"Train it with `python -m modeling.cli train {module} --output-version {version}`.",
        )
    expected, _ = _expected(module, version)
    if expected is None:
        if not allowed:
            _count(
                CandidateReason.UNVERIFIED_REFUSED, str(target), module=module, version=version
            )
            raise CandidateRefused(
                CandidateReason.UNVERIFIED_REFUSED,
                f"{module}/{version} has no sha256 in configs/models.yaml. Run "
                f"`python -m modeling.cli checkpoint-hash {module} --version {version}` to read "
                f"the digest, or pass --allow-unverified-candidate to load it without a check.",
            )
        log.warning(
            "LOADING UNVERIFIED CANDIDATE %s/%s from %s (sha256 %s). No hash check ran. "
            "Do not ship a score from it.",
            module,
            version,
            target,
            (_dir_sha256(target) or "")[:12],
        )
    try:
        return _verify(module, version, target, pulled=False)
    except CheckpointIntegrityError as exc:
        raise CandidateRefused(exc.reason, str(exc)) from exc
