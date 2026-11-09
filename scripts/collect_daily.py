"""Daily capture: news_rss, fact-check extraction, one GDELT pass.

The fact-check RSS feeds hold only 10 to 60 entries, so a day that is not
captured is lost. Each step runs in its own process. A failure in one step does
not stop the others, and the exit code reports it.

Exit codes: 0 all steps ok, 1 a step failed, 2 no step failed but a step
reported a degraded run (a throttle or a skipped source), 75 another run holds
the lock.

GDELT runs exactly once. Its rate limit is a stateful per-IP penalty window, so
a retry loop here would extend the penalty.
"""

from __future__ import annotations

import fcntl
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_DEGRADED = 2
EXIT_LOCKED = 75

#: Output markers of a run that exited 0 but lost data. The fetch commands exit 0
#: on a throttle and on a skipped source, so the exit code alone hides both.
DEGRADED_MARKERS = re.compile(r"rate_limited|\bSKIP\b")


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...]


def default_steps(python: str = sys.executable) -> list[Step]:
    cli = (python, "-m", "ingest.cli")
    return [
        Step("news_rss", (*cli, "fetch", "news_rss")),
        Step("factchecks", (*cli, "factchecks")),
        Step("gdelt", (*cli, "fetch", "gdelt")),
    ]


def run_step(step: Step, log, cwd: Path | None = None) -> str:
    """Run one step, copy its output to the log, and return ok, FAILED or DEGRADED."""
    log.write(f"=== {step.name}: {' '.join(step.command)}\n")
    log.flush()
    try:
        proc = subprocess.run(
            step.command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except OSError as exc:
        log.write(f"could not start: {exc}\n")
        return "FAILED"
    log.write(proc.stdout)
    log.write(f"=== {step.name}: exit {proc.returncode}\n")
    if proc.returncode != 0:
        return "FAILED"
    if DEGRADED_MARKERS.search(proc.stdout):
        return "DEGRADED"
    return "ok"


def collect(steps: Sequence[Step], data_dir: Path, cwd: Path | None = None, out=None) -> int:
    out = out or sys.stdout
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / "collect-daily.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"collect-daily: another run holds {lock_path}; exiting", file=out)
            return EXIT_LOCKED

        log_dir = data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc)
        log_path = log_dir / f"collect-daily-{stamp:%Y-%m-%d}.log"
        statuses: dict[str, str] = {}
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"##### collect-daily start {stamp.isoformat()}\n")
            for step in steps:
                status = run_step(step, log, cwd)
                statuses[step.name] = status
                print(f"collect-daily: {step.name}: {status}", file=out)
                out.flush()
            log.write(f"##### collect-daily end {datetime.now(timezone.utc).isoformat()}\n")
        print(f"collect-daily: log {log_path}", file=out)

    if "FAILED" in statuses.values():
        return EXIT_FAILED
    if "DEGRADED" in statuses.values():
        return EXIT_DEGRADED
    return EXIT_OK


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    sys.path.insert(0, str(root))
    from ingest.config import get_settings

    return collect(default_steps(), get_settings().data_dir, cwd=root)


if __name__ == "__main__":
    sys.exit(main())
