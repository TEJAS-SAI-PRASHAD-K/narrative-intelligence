"""The daily capture script: step isolation, exit codes, lock and log. No network."""

from __future__ import annotations

import fcntl
import importlib.util
import io
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "collect_daily.py"
spec = importlib.util.spec_from_file_location("collect_daily", SCRIPT)
cd = importlib.util.module_from_spec(spec)
sys.modules["collect_daily"] = cd
spec.loader.exec_module(cd)


def py(code: str) -> tuple[str, ...]:
    return (sys.executable, "-c", code)


def run(steps, data_dir):
    out = io.StringIO()
    return cd.collect(steps, data_dir, out=out), out.getvalue()


def test_all_ok(tmp_path):
    steps = [cd.Step("a", py("print('x')")), cd.Step("b", py("print('y')"))]
    code, out = run(steps, tmp_path)
    assert code == cd.EXIT_OK
    assert "a: ok" in out and "b: ok" in out


def test_failure_does_not_stop_later_steps(tmp_path):
    marker = tmp_path / "ran"
    steps = [
        cd.Step("news_rss", py("import sys; sys.exit(3)")),
        cd.Step("factchecks", py(f"open({str(marker)!r}, 'w').close()")),
        cd.Step("gdelt", py("pass")),
    ]
    code, out = run(steps, tmp_path)
    assert code == cd.EXIT_FAILED
    assert marker.exists()
    lines = [ln for ln in out.splitlines() if ln.startswith("collect-daily: ") and "log" not in ln]
    assert lines == [
        "collect-daily: news_rss: FAILED",
        "collect-daily: factchecks: ok",
        "collect-daily: gdelt: ok",
    ]


def test_a_hung_step_times_out_as_failed_and_later_steps_run(tmp_path):
    marker = tmp_path / "ran"
    steps = [
        cd.Step("hang", py("import time; time.sleep(30)")),
        cd.Step("after", py(f"open({str(marker)!r}, 'w').close()")),
    ]
    out = io.StringIO()
    code = cd.collect(steps, tmp_path, out=out, timeout=0.5)
    assert code == cd.EXIT_FAILED
    assert "hang: FAILED" in out.getvalue() and "after: ok" in out.getvalue()
    assert marker.exists()
    assert "timed out" in next((tmp_path / "logs").glob("collect-daily-*.log")).read_text()


def test_step_timeout_default_is_thirty_minutes():
    assert cd.STEP_TIMEOUT_SECONDS == 1800


def test_missing_command_is_a_failure(tmp_path):
    code, out = run([cd.Step("x", ("/nonexistent/binary",)), cd.Step("y", py("pass"))], tmp_path)
    assert code == cd.EXIT_FAILED
    assert "x: FAILED" in out and "y: ok" in out


def test_throttle_with_exit_zero_is_degraded_not_ok(tmp_path):
    steps = [cd.Step("gdelt", py("print(\"  gdelt flags: {'doc_rate_limited': 1}\")"))]
    code, out = run(steps, tmp_path)
    assert code == cd.EXIT_DEGRADED
    assert "gdelt: DEGRADED" in out


def test_failure_outranks_degraded(tmp_path):
    steps = [
        cd.Step("a", py("print('rate_limited')")),
        cd.Step("b", py("import sys; sys.exit(1)")),
    ]
    code, _ = run(steps, tmp_path)
    assert code == cd.EXIT_FAILED


def test_each_step_runs_once(tmp_path):
    counter = tmp_path / "count"
    code_str = f"open({str(counter)!r}, 'a').write('x')"
    steps = [cd.Step("gdelt", py(code_str + "; import sys; sys.exit(1)"))]
    run(steps, tmp_path)
    assert counter.read_text() == "x"


def test_log_is_dated_and_appended(tmp_path):
    steps = [cd.Step("a", py("print('hello' + '-log')"))]
    run(steps, tmp_path)
    run(steps, tmp_path)
    logs = list((tmp_path / "logs").glob("collect-daily-????-??-??.log"))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert text.count("hello-log") == 2
    assert text.count("collect-daily start") == 2


def test_lock_blocks_an_overlapping_run(tmp_path):
    marker = tmp_path / "ran"
    steps = [cd.Step("a", py(f"open({str(marker)!r}, 'w').close()"))]
    with (tmp_path / "collect-daily.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, out = run(steps, tmp_path)
    assert code == cd.EXIT_LOCKED
    assert "another run holds" in out
    assert not marker.exists()
    code, _ = run(steps, tmp_path)  # lock released: runs normally
    assert code == cd.EXIT_OK and marker.exists()


def test_default_steps_order_and_gdelt_once():
    names = [s.name for s in cd.default_steps("py")]
    assert names == ["news_rss", "factchecks", "gdelt"]
    assert cd.default_steps("py")[1].command[-1] == "factchecks"


def test_log_and_lock_paths_are_gitignored():
    root = Path(__file__).resolve().parent.parent
    for rel in ("data/logs/collect-daily-2026-01-01.log", "data/collect-daily.lock"):
        result = subprocess.run(["git", "check-ignore", "-q", rel], cwd=root)
        assert result.returncode == 0, rel
