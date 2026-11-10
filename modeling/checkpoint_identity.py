"""The identity of a checkpoint directory: a hash of what it holds, with no weights read.

The comparison script records it when it scores a checkpoint, and the Laya calibrator
records it when it is fitted. The scoring stage recomputes it from the served directory,
so a calibrator that was fitted for other weights is refused.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

#: Files whose bytes go into a checkpoint identity. All other files count by name and size.
CHECKPOINT_CONFIG_FILES = ("config.json", "rl_agent_config.json")
#: Laya's loader rewrites this file in place the first time it loads a saved checkpoint, so
#: its size differs between the scoring run and the check before evaluation. The rewrite
#: changes loader keys, not the vocabulary, and ``tokenizer/tokenizer.json`` stays in the listing.
CHECKPOINT_LOADER_REWRITTEN = frozenset({"tokenizer/tokenizer_config.json"})


def directory_identity(
    directory: Path, ignore: Collection[str] = frozenset()
) -> dict[str, Any] | None:
    """The name of a checkpoint directory and a hash of what it holds, or ``None`` if absent.

    The hash covers the file names with sizes and the bytes of the config files. It reads
    no weights, so it is fast on a large checkpoint. It leaves out the file that the Laya
    loader rewrites, because loading a checkpoint must not change its identity.
    ``ignore`` names more files, as paths relative to ``directory``, that stay out of the
    hash. The calibrator files use it, because they are added after the weights.
    """
    if not directory.is_dir():
        return None
    skipped = CHECKPOINT_LOADER_REWRITTEN | frozenset(ignore)
    listing = sorted(
        (relative, path.stat().st_size)
        for path in directory.rglob("*")
        if path.is_file()
        and not path.name.endswith(".tmp")
        and (relative := path.relative_to(directory).as_posix()) not in skipped
    )
    configs = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in CHECKPOINT_CONFIG_FILES
        if (directory / name).is_file() and name not in skipped
    }
    text = json.dumps({"files": listing, "configs": configs}, sort_keys=True)
    return {
        "kind": "directory",
        "name": directory.name,
        "files": len(listing),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
