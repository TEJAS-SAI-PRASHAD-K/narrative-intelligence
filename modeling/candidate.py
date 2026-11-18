"""Score with a candidate checkpoint without editing ``configs/models.yaml``.

A candidate is a checkpoint folder ``models/<module>/<VERSION>/`` that is not the configured
version. :class:`CandidateOptions` selects it for one scoring run. ``modeling.cli score``
builds the options from ``--misinfo-version``, ``--allow-unverified-candidate`` and
``--replace-other-version``, and passes them to ``run_stages``. No environment variable
carries them, so an old value in the shell cannot select a candidate by accident.

The registry hash check still runs. A candidate with a configured hash must match it. A
candidate with no configured hash is refused unless ``unverified`` is set. With it, the load
works, the log carries a warning, and the registry counts ``checkpoint_unverified``. The
scored rows carry ``<VERSION>+<sha8>``, so no row looks like the configured version.

A candidate is never pulled from the Hub or Drive. It exists on local disk or it does not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

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


@dataclass(frozen=True)
class CandidateOptions:
    """The candidate settings of one scoring run."""

    version: str
    #: Load a candidate that has no ``sha256`` in ``configs/models.yaml``.
    unverified: bool = False
    #: Overwrite ``misinfo_prob`` of rows that carry another misinfo stamp.
    replace_other: bool = False


def resolve_candidate(module: str, version: str, *, unverified_ok: bool = False) -> Checkpoint:
    """Find, verify and return the candidate checkpoint. Raise :class:`CandidateRefused`."""
    allowed = unverified_ok
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
