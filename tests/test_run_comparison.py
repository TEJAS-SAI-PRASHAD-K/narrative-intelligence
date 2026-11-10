"""scripts/run_comparison.py: stages, resume, GPU gate, split guard and score files.

No model weights, no GPU and no network. ``laya``, the roberta classifier, ``finetune``
and ``nvidia-smi`` are all fakes. An autouse fixture makes any real call of
``nvidia-smi`` fail, so a test cannot reach the real GPU tool by mistake.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeling.config import module_config
from modeling.eval.factcheck_eval import PARTS, load_factcheck_publishers, load_score_dir
from modeling.training import TrainingResult

LAYA_V = module_config("laya")["version"]
MISINFO_V = module_config("misinfo")["version"]

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_comparison.py"
spec = importlib.util.spec_from_file_location("run_comparison", SCRIPT)
rc = importlib.util.module_from_spec(spec)
sys.modules["run_comparison"] = rc
spec.loader.exec_module(rc)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def no_real_nvidia_smi(monkeypatch):
    real_run = subprocess.run

    def guard(command, *args, **kwargs):
        if str(command[0]).endswith("nvidia-smi"):
            raise AssertionError("a test ran the real nvidia-smi")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guard)


@dataclass
class Gpu:
    """A fake ``nvidia-smi`` reader: one free-memory value per call, the last one repeats."""

    free: list[int]
    total: int = 40960
    used: int = 1000
    calls: list[int] = field(default_factory=list)

    def __call__(self, index: int):
        self.calls.append(index)
        free = self.free[min(len(self.calls) - 1, len(self.free) - 1)]
        return rc.GpuMemory(index, self.total, self.used, free)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_deps(free=(40000,), clock=None):
    clock = clock or Clock()
    gpu = Gpu(list(free))
    out = io.StringIO()
    return rc.Deps(gpu_reader=gpu, sleep=clock.sleep, clock=clock, out=out), gpu, clock, out


def run(argv, deps):
    return rc.main(argv, deps=deps)


@pytest.fixture(scope="module")
def prepared_template(tmp_path_factory):
    """One real ``prepare --demo`` on the committed fixtures. CPU only, about two seconds."""
    out = tmp_path_factory.mktemp("template") / "out"
    deps, *_ = make_deps()
    assert run(["prepare", "--demo", "--out", str(out)], deps) == 0
    return out


@pytest.fixture
def prepared(prepared_template, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(prepared_template, out)
    return out


class FakeLayaAgent:
    """Implements ``predict`` and ``predict_batch`` as ``tests/test_laya.py`` documents them."""

    def __init__(self, noul=0.8, fail=None):
        self.noul = noul
        self.fail = fail

    def _result(self, questions):
        return {"answers": {next(iter(questions)): {"noul": self.noul}}}

    def predict(self, state, questions, **kwargs):
        if self.fail:
            raise self.fail
        return self._result(questions)

    def predict_long(self, state, questions, **kwargs):
        return self.predict(state, questions)

    def predict_batch(self, states, questions, **kwargs):
        if self.fail:
            raise self.fail
        return [self._result(questions) for _ in states]


@pytest.fixture
def fake_laya(monkeypatch):
    loads: list[tuple[str, dict]] = []
    state = types.SimpleNamespace(loads=loads, fail=None)
    module = types.ModuleType("laya")
    module.__version__ = "0.4.30"

    def load(checkpoint, **kwargs):
        loads.append((checkpoint, kwargs))
        return FakeLayaAgent(fail=state.fail)

    module.load = load
    monkeypatch.setitem(sys.modules, "laya", module)
    return state


class FakeMisinfo:
    batch_size = 16

    def __init__(self, settings=None):
        self.settings = settings

    def load(self, model_dir):
        return True

    def predict(self, texts):
        return np.full(len(texts), 0.6)


@pytest.fixture
def fake_misinfo(monkeypatch):
    monkeypatch.setattr("modeling.text.misinfo_clf.MisinfoClassifier", FakeMisinfo)


def add_checkpoints(out: Path) -> None:
    """The files that the score stage looks for. The fakes never read them."""
    misinfo = out / "models" / "misinfo" / MISINFO_V
    misinfo.mkdir(parents=True)
    (misinfo / "calibrator.json").write_text("{}", encoding="utf-8")
    (misinfo / "config.json").write_text('{"model_type": "roberta"}', encoding="utf-8")
    for prefix in ("en", "multi"):
        directory = out / "models" / "laya" / f"{prefix}-{LAYA_V}"
        directory.mkdir(parents=True)
        (directory / "run_summary.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def scored(prepared, fake_laya, fake_misinfo):
    add_checkpoints(prepared)
    deps, *_ = make_deps()
    assert run(["score", "--demo", "--out", str(prepared)], deps) == 0
    return prepared


def score_frames(out: Path) -> dict[str, pd.DataFrame]:
    return {name: m.frame for name, m in load_score_dir(out / "scores").items()}


# ---------------------------------------------------------------------------
# stage order, failure and resume
# ---------------------------------------------------------------------------
def test_all_runs_the_stages_in_order(monkeypatch, tmp_path):
    seen: list[str] = []
    for name in ("stage_prepare", "stage_train_misinfo", "stage_score", "stage_evaluate"):
        monkeypatch.setattr(rc, name, lambda ctx, n=name: seen.append(n))
    monkeypatch.setattr(rc, "stage_train_laya", lambda ctx, which: seen.append(f"laya-{which}"))
    deps, _, _, out = make_deps()
    assert run(["all", "--demo", "--out", str(tmp_path / "o")], deps) == 0
    assert seen == [
        "stage_prepare",
        "stage_train_misinfo",
        "laya-en",
        "laya-multilingual",
        "stage_score",
        "stage_evaluate",
    ]
    headings = [line for line in out.getvalue().splitlines() if line.startswith("===")]
    assert headings == [
        "=== prepare ===",
        "=== train-misinfo ===",
        "=== train-laya-en ===",
        "=== train-laya-multilingual ===",
        "=== score ===",
        "=== evaluate ===",
    ]


def test_all_stops_at_the_first_failure(monkeypatch, tmp_path, capsys):
    seen: list[str] = []
    monkeypatch.setattr(rc, "stage_prepare", lambda ctx: seen.append("prepare"))

    def fail(ctx):
        seen.append("train-misinfo")
        raise rc.StageError(rc.EXIT_FAILED, "boom")

    monkeypatch.setattr(rc, "stage_train_misinfo", fail)
    monkeypatch.setattr(rc, "stage_train_laya", lambda ctx, which: seen.append("laya"))
    monkeypatch.setattr(rc, "stage_score", lambda ctx: seen.append("score"))
    deps, *_ = make_deps()
    assert run(["all", "--demo", "--out", str(tmp_path / "o")], deps) == rc.EXIT_FAILED
    assert seen == ["prepare", "train-misinfo"]
    assert "boom" in capsys.readouterr().err


def test_an_unexpected_exception_is_a_failure_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "stage_prepare", lambda ctx: 1 / 0)
    deps, *_ = make_deps()
    assert run(["prepare", "--out", str(tmp_path)], deps) == rc.EXIT_FAILED


def test_prepare_resumes_unless_forced(prepared, monkeypatch):
    marker = prepared / "laya" / "train.jsonl"
    marker.write_text("sentinel\n", encoding="utf-8")
    monkeypatch.setattr(rc, "load_training_frame", lambda demo: pytest.fail("rebuilt the frame"))
    deps, _, _, out = make_deps()
    assert run(["prepare", "--demo", "--out", str(prepared)], deps) == 0
    assert "skipped" in out.getvalue()
    assert marker.read_text(encoding="utf-8") == "sentinel\n"


def test_prepare_force_redoes_the_stage(prepared):
    marker = prepared / "laya" / "train.jsonl"
    marker.write_text("sentinel\n", encoding="utf-8")
    deps, *_ = make_deps()
    assert run(["prepare", "--demo", "--force", "--out", str(prepared)], deps) == 0
    assert marker.read_text(encoding="utf-8") != "sentinel\n"
    assert len(_states(marker)) > 100


def test_train_misinfo_resumes_before_it_reads_the_gpu(prepared, monkeypatch):
    (prepared / "summaries").mkdir()
    (prepared / "summaries" / "misinfo.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("modeling.training.train_module", lambda *a, **k: pytest.fail("trained"))
    deps, gpu, _, out = make_deps()
    assert run(["train-misinfo", "--demo", "--out", str(prepared)], deps) == 0
    assert "skipped" in out.getvalue()
    assert gpu.calls == []


def test_train_laya_resumes(prepared, monkeypatch):
    (prepared / "summaries").mkdir()
    (prepared / "summaries" / "laya_en.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rc, "resolve_base_checkpoint", lambda *a: pytest.fail("downloaded"))
    deps, gpu, _, out = make_deps()
    assert run(["train-laya", "--which", "en", "--demo", "--out", str(prepared)], deps) == 0
    assert "skipped" in out.getvalue() and gpu.calls == []


def test_score_resumes_per_model(scored, fake_laya):
    fake_laya.loads.clear()
    (scored / "scores" / "laya_multi.parquet").unlink()
    # Each file is written by an atomic replace, so a file that is written again has a new inode.
    done = {p.stem: p.stat().st_ino for p in (scored / "scores").glob("*.parquet")}
    deps, *_ = make_deps()
    assert run(["score", "--demo", "--out", str(scored)], deps) == 0
    after = {p.stem: p.stat().st_ino for p in (scored / "scores").glob("*.parquet")}
    assert set(after) == set(rc.MODELS)
    assert {name for name in done if after[name] == done[name]} == set(done)
    loaded = {checkpoint for checkpoint, _ in fake_laya.loads}
    assert loaded == {str(scored / "models" / "laya" / f"multi-{LAYA_V}")}


def test_score_stage_is_skipped_when_every_file_exists(scored):
    deps, gpu, _, out = make_deps()
    assert run(["score", "--demo", "--out", str(scored)], deps) == 0
    assert "skipped" in out.getvalue() and gpu.calls == []


def test_a_stage_before_prepare_says_so(tmp_path, capsys):
    deps, *_ = make_deps()
    assert run(["train-misinfo", "--demo", "--out", str(tmp_path)], deps) == rc.EXIT_FAILED
    assert "prepare" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# GPU gate
# ---------------------------------------------------------------------------
def test_low_gpu_memory_is_refused_with_a_distinct_code(prepared, monkeypatch, capsys):
    monkeypatch.setattr("modeling.training.train_module", lambda *a, **k: pytest.fail("trained"))
    deps, gpu, _, _ = make_deps(free=(3000,))
    code = run(["train-misinfo", "--demo", "--out", str(prepared)], deps)
    assert code == rc.EXIT_GPU_BUSY == 10
    err = capsys.readouterr().err
    assert "3000 MiB free" in err and "8000" in err and "does not stop other" in err
    assert gpu.calls == [0]


def test_default_thresholds_are_8000_to_train_and_6000_to_score(prepared, scored, capsys):
    # 7000 MiB is enough to score and not enough to train.
    deps, *_ = make_deps(free=(7000,))
    assert run(["train-misinfo", "--demo", "--out", str(prepared)], deps) == rc.EXIT_GPU_BUSY
    (scored / "scores" / "tfidf.parquet").unlink()
    assert run(["score", "--demo", "--out", str(scored), "--models", "tfidf"], deps) == 0
    deps, *_ = make_deps(free=(5000,))
    (scored / "scores" / "tfidf.parquet").unlink()
    assert run(["score", "--demo", "--out", str(scored), "--models", "tfidf"], deps) == 10
    assert "6000" in capsys.readouterr().err


def test_min_free_mib_option_changes_the_threshold(prepared, monkeypatch, capsys):
    monkeypatch.setattr("modeling.training.train_module", lambda *a, **k: pytest.fail("trained"))
    deps, *_ = make_deps(free=(3000,))
    argv = ["train-misinfo", "--demo", "--out", str(prepared), "--min-free-mib", "4000"]
    assert run(argv, deps) == rc.EXIT_GPU_BUSY
    assert "4000" in capsys.readouterr().err


def test_wait_for_gpu_polls_every_five_minutes_until_free(prepared, monkeypatch):
    calls = []
    result = misinfo_result(split_of(prepared))
    monkeypatch.setattr("modeling.training.train_module", lambda *a, **k: calls.append(1) or result)
    deps, gpu, clock, out = make_deps(free=(1000, 2000, 9000))
    argv = ["train-misinfo", "--demo", "--out", str(prepared), "--wait-for-gpu"]
    assert run(argv, deps) == 0
    assert clock.sleeps == [300, 300]
    assert len(calls) == 1
    assert out.getvalue().count("Next check in 5 minutes") == 2


def test_wait_for_gpu_gives_up_after_wait_hours(prepared, monkeypatch, capsys):
    monkeypatch.setattr("modeling.training.train_module", lambda *a, **k: pytest.fail("trained"))
    deps, gpu, clock, _ = make_deps(free=(1000,))
    argv = ["train-misinfo", "--demo", "--out", str(prepared), "--wait-for-gpu"]
    argv += ["--wait-hours", "1"]
    assert run(argv, deps) == rc.EXIT_GPU_TIMEOUT == 11
    assert sum(clock.sleeps) >= 3600
    assert "1 hours" in capsys.readouterr().err


def test_gpu_index_follows_cuda_visible_devices(prepared, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr("modeling.training.train_module", lambda *a, **k: pytest.fail("trained"))
    deps, gpu, _, _ = make_deps(free=(10,))
    run(["train-misinfo", "--demo", "--out", str(prepared)], deps)
    assert gpu.calls == [3]
    run(["train-misinfo", "--demo", "--out", str(prepared), "--gpu-index", "1"], deps)
    assert gpu.calls[-1] == 1


def _fake_smi(monkeypatch, stdout="", returncode=0, error=None):
    def fake(command, **kwargs):
        assert command[0] == "nvidia-smi"
        if error:
            raise error
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="no driver")

    monkeypatch.setattr(subprocess, "run", fake)


def test_read_gpu_memory_parses_nvidia_smi_output(monkeypatch):
    _fake_smi(monkeypatch, "0, 40960, 31000, 9960\n1, 40960, 100, 40860\n")
    assert rc.read_gpu_memory(1) == rc.GpuMemory(1, 40960, 100, 40860)
    assert rc.read_gpu_memory(0).free_mib == 9960


@pytest.mark.parametrize(
    "kwargs",
    [
        {"error": FileNotFoundError("nvidia-smi")},
        {"stdout": "", "returncode": 9},
        {"stdout": "0, [N/A], [N/A], [N/A]\n"},
        {"stdout": "5, 1, 1, 1\n"},
    ],
)
def test_unreadable_gpu_memory_is_refused(monkeypatch, kwargs):
    _fake_smi(monkeypatch, **kwargs)
    with pytest.raises(rc.StageError) as caught:
        rc.read_gpu_memory(0)
    assert caught.value.code == rc.EXIT_NO_NVIDIA_SMI == 12


def test_the_script_never_signals_a_process():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("os.kill", "killpg", "terminate(", ".kill(", "pkill", "SIGKILL", "SIGTERM"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# end-of-stage report and out-of-memory
# ---------------------------------------------------------------------------
def split_of(out: Path) -> str:
    return json.loads((out / "split.json").read_text())["description"]


@functools.cache
def _report():
    from modeling.eval.metrics import classification_report

    y = np.array([0, 1] * 10)
    return classification_report(
        y,
        y,
        y_score=y.astype(float),
        module="misinfo",
        split_description="placeholder",
        class_names=["a", "b"],
        seed=1,
        is_demo=True,
    )


def misinfo_result(split, notes=None, skipped=False):
    report = dataclasses.replace(_report(), split_description=split)
    return TrainingResult(
        "misinfo", MISINFO_V, None if skipped else report, notes=notes or [], skipped=skipped
    )


def test_train_misinfo_saves_macro_f1_and_reports_time_and_memory(prepared, monkeypatch):
    deps, gpu, clock, out = make_deps()
    seen = {}

    def fake_train(module, *, demo, epochs, settings, **kwargs):
        seen.update(module=module, demo=demo, epochs=epochs, models_dir=settings.models_dir)
        clock.now += 125
        return misinfo_result(split_of(prepared))

    monkeypatch.setattr("modeling.training.train_module", fake_train)
    assert run(["train-misinfo", "--demo", "--out", str(prepared)], deps) == 0
    assert seen["module"] == "misinfo" and seen["demo"] is True and seen["epochs"] == 1
    assert seen["models_dir"] == prepared / "models"
    summary = json.loads((prepared / "summaries" / "misinfo.json").read_text())
    assert summary["macro_f1_test"]["value"] == 1.0
    assert summary["split_matches_prepare"] is True
    text = out.getvalue()
    assert "done in 0:02:05" in text
    assert "1000 MiB of 40960 MiB in use on GPU 0" in text


def test_train_misinfo_restores_settings_and_the_class(prepared, monkeypatch):
    from modeling.config import get_settings
    from modeling.text.misinfo_clf import MisinfoClassifier

    before = (get_settings().models_dir, os.environ.get("MODELS_DIR"))
    init = MisinfoClassifier.__init__
    sizes = []

    def fake_train(module, **kwargs):
        sizes.append(MisinfoClassifier().batch_size)
        return misinfo_result(split_of(prepared))

    monkeypatch.setattr("modeling.training.train_module", fake_train)
    deps, *_ = make_deps()
    argv = ["train-misinfo", "--demo", "--out", str(prepared), "--batch-size", "4"]
    assert run(argv, deps) == 0
    assert sizes == [4]
    assert MisinfoClassifier.__init__ is init
    assert (get_settings().models_dir, os.environ.get("MODELS_DIR")) == before


def test_train_misinfo_refuses_a_different_split(prepared, monkeypatch, capsys):
    other = "grouped by group_id (10 groups; train/val/test = 6/2/2; seed=1)"
    monkeypatch.setattr("modeling.training.train_module", lambda *a, **k: misinfo_result(other))
    deps, *_ = make_deps()
    assert run(["train-misinfo", "--demo", "--out", str(prepared)], deps) == rc.EXIT_SPLIT
    assert "another split" in capsys.readouterr().err
    assert not (prepared / "summaries" / "misinfo.json").exists()


def test_train_misinfo_refuses_when_a_benchmark_was_dropped(prepared, monkeypatch, capsys):
    notes = ["coaid not available; excluded from training"]
    monkeypatch.setattr(
        "modeling.training.train_module",
        lambda *a, **k: misinfo_result(split_of(prepared), notes=notes),
    )
    deps, *_ = make_deps()
    assert run(["train-misinfo", "--demo", "--out", str(prepared)], deps) == rc.EXIT_SPLIT
    assert "smaller union" in capsys.readouterr().err


def test_train_misinfo_reports_a_skipped_training(prepared, monkeypatch, capsys):
    monkeypatch.setattr(
        "modeling.training.train_module",
        lambda *a, **k: misinfo_result(split_of(prepared), notes=["no data"], skipped=True),
    )
    deps, *_ = make_deps()
    assert run(["train-misinfo", "--demo", "--out", str(prepared)], deps) == rc.EXIT_FAILED
    assert "no data" in capsys.readouterr().err


def test_oom_in_train_misinfo_names_the_option_to_lower(prepared, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr("modeling.training.train_module", boom)
    deps, *_ = make_deps()
    assert run(["train-misinfo", "--demo", "--out", str(prepared)], deps) == rc.EXIT_GPU_OOM
    err = capsys.readouterr().err
    assert "--batch-size" in err and "ran out of memory" in err


def test_oom_is_found_through_a_cause_chain():
    inner = RuntimeError("CUDA out of memory")
    try:
        try:
            raise inner
        except RuntimeError as exc:
            raise ValueError("wrapped") from exc
    except ValueError as outer:
        assert rc.is_oom(outer)
    assert not rc.is_oom(ValueError("disk full"))
    OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    assert rc.is_oom(OutOfMemoryError("x"))


# ---------------------------------------------------------------------------
# train-laya
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_finetune(monkeypatch):
    @dataclass
    class TrainConfig:
        epochs: int = 4
        micro_batch: int = 8
        grad_accum: int = 8
        encoder_lr: float = 0.0
        head_lr: float = 0.0
        weight_decay: float = 0.0
        grad_clip: float = 0.0
        loss: str = ""
        rl_samples: int = 4
        calib_frac: float = 0.0
        calib_max: int = 0
        calib_seed: int = 0
        eval_data: str | None = None

    calls = []
    state = types.SimpleNamespace(calls=calls, eval_mode="held_out", error=None)

    def finetune(data, model_dir, output_dir, config=None, device="auto"):
        calls.append((data, model_dir, output_dir, config, device))
        if state.error:
            raise state.error
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {"train_items": 10, "train_report": {"eval_mode": state.eval_mode}}

    train = types.ModuleType("laya.train")
    train.TrainConfig = TrainConfig
    train.finetune = finetune
    package = types.ModuleType("laya")
    package.__version__ = "0.4.30"
    package.train = train
    monkeypatch.setitem(sys.modules, "laya", package)
    monkeypatch.setitem(sys.modules, "laya.train", train)
    monkeypatch.setattr(rc, "resolve_base_checkpoint", lambda which, cfg: f"/base/{which}")
    return state


def test_train_laya_calls_finetune_as_the_runbook_does(prepared, fake_finetune):
    deps, _, _, out = make_deps()
    assert run(["train-laya", "--which", "en", "--demo", "--out", str(prepared)], deps) == 0
    (data, model_dir, output_dir, config, device), = fake_finetune.calls
    cfg = module_config("laya")["train"]
    assert data == str(prepared / "laya" / "train.jsonl")
    assert config.eval_data == str(prepared / "laya" / "val.jsonl")
    assert model_dir == "/base/en"
    assert output_dir == str(prepared / "models" / "laya" / f"en-{LAYA_V}")
    assert (config.micro_batch, config.grad_accum) == (cfg["micro_batch"], cfg["grad_accum"])
    assert config.encoder_lr == cfg["encoder_lr"] and config.loss == cfg["loss"]
    assert config.epochs == 1  # a demo run uses one epoch
    assert device == "auto"
    summary = json.loads((prepared / "summaries" / "laya_en.json").read_text())
    assert summary["finetune"]["train_items"] == 10
    assert (Path(output_dir) / "run_summary.json").exists()
    assert "done in" in out.getvalue() and "MiB" in out.getvalue()


def test_train_laya_multilingual_uses_its_own_directory(prepared, fake_finetune):
    deps, *_ = make_deps()
    argv = ["train-laya", "--which", "multilingual", "--demo", "--out", str(prepared)]
    assert run(argv, deps) == 0
    assert fake_finetune.calls[0][1] == "/base/multilingual"
    assert fake_finetune.calls[0][2].endswith(f"multi-{LAYA_V}")


def test_train_laya_keeps_the_effective_batch_when_micro_batch_is_lowered(prepared, fake_finetune):
    deps, *_ = make_deps()
    argv = ["train-laya", "--which", "en", "--demo", "--out", str(prepared), "--micro-batch", "2"]
    argv += ["--epochs", "3"]
    assert run(argv, deps) == 0
    config = fake_finetune.calls[0][3]
    assert (config.micro_batch, config.grad_accum, config.epochs) == (2, 32, 3)
    assert config.micro_batch * config.grad_accum == 64


def test_train_laya_warns_when_validation_is_not_held_out(prepared, fake_finetune):
    fake_finetune.eval_mode = "overlapping_eval"
    deps, _, _, out = make_deps()
    assert run(["train-laya", "--which", "en", "--demo", "--out", str(prepared)], deps) == 0
    assert "WARNING: validation is not strictly held out" in out.getvalue()
    summary = json.loads((prepared / "summaries" / "laya_en.json").read_text())
    assert summary["warnings"]


def test_oom_in_train_laya_names_micro_batch(prepared, fake_finetune, capsys):
    fake_finetune.error = RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
    deps, *_ = make_deps()
    code = run(["train-laya", "--which", "en", "--demo", "--out", str(prepared)], deps)
    assert code == rc.EXIT_GPU_OOM
    assert "--micro-batch" in capsys.readouterr().err
    assert not (prepared / "summaries" / "laya_en.json").exists()


def test_the_base_checkpoint_is_downloaded_at_the_pinned_revision(monkeypatch, tmp_path):
    calls = []

    def snapshot_download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path / "snap")

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    cfg = module_config("laya")
    assert rc.resolve_base_checkpoint("en", cfg) == str(tmp_path / "snap")
    multilingual = rc.resolve_base_checkpoint("multilingual", cfg)
    assert multilingual == str(tmp_path / "snap" / "multilingual")
    for _, kwargs in calls:
        assert kwargs["revision"] == cfg["revision"]
    assert "multilingual/model.safetensors" in calls[1][1]["allow_patterns"]
    assert "model.safetensors" in calls[0][1]["allow_patterns"]
    with pytest.raises(rc.StageError):
        rc.resolve_base_checkpoint("en", {**cfg, "revision": None})


# ---------------------------------------------------------------------------
# prepare: the split guard and the test split
# ---------------------------------------------------------------------------
def synthetic_frame(n_groups=60, per_group=3) -> pd.DataFrame:
    rows = []
    for g in range(n_groups):
        for k in range(per_group):
            bad = (g + k) % 2 == 0
            rows.append(
                {
                    "text": f"{'hoax' if bad else 'budget'} claim number {g} variant {k} wording",
                    "label": int(bad),
                    "source_dataset": "liar" if g % 2 else "coaid",
                    "group_id": f"grp{g}",
                    "domain": "x",
                }
            )
    return pd.DataFrame(rows)


def test_prepare_refuses_a_split_that_differs_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rc, "load_training_frame", lambda demo: synthetic_frame())
    deps, *_ = make_deps()
    out = tmp_path / "out"
    assert run(["prepare", "--out", str(out)], deps) == rc.EXIT_SPLIT == 14
    err = capsys.readouterr().err
    assert rc.EXPECTED_SPLIT in err and "grouped by group_id (60 groups" in err
    assert not out.exists() or not any(out.rglob("*.jsonl"))
    assert not (out / "split.json").exists()


def test_the_required_split_string_is_the_documented_one():
    assert rc.EXPECTED_SPLIT == (
        "grouped by group_id (30148 groups; train/val/test = 26777/3596/7085; seed=20260813)"
    )


def test_prepare_accepts_a_split_that_matches(tmp_path, monkeypatch):
    from modeling.datasets.splits import group_train_val_test

    frame = synthetic_frame()
    _, split = group_train_val_test(frame, group_col="group_id", label_col="label")
    monkeypatch.setattr(rc, "EXPECTED_SPLIT", split.describe())
    monkeypatch.setattr(rc, "load_training_frame", lambda demo: frame)
    deps, *_ = make_deps()
    out = tmp_path / "out"
    assert run(["prepare", "--out", str(out)], deps) == 0
    info = json.loads((out / "split.json").read_text())
    assert info["verified_against"] == split.describe() and info["demo"] is False
    assert (out / "run.json").exists()


def test_demo_waives_the_split_guard_and_stays_out_of_the_real_directory(prepared_template):
    info = json.loads((prepared_template / "split.json").read_text())
    assert info["demo"] is True and info["verified_against"] is None
    assert info["description"] != rc.EXPECTED_SPLIT
    ctx = rc.Context(args=rc.build_parser().parse_args(["prepare", "--demo"]))
    real = rc.Context(args=rc.build_parser().parse_args(["prepare"]))
    assert ctx.out != real.out
    assert ctx.out == rc.REPO_ROOT / "data" / "comparison_demo"
    assert real.out == rc.REPO_ROOT / "data" / "comparison"


def test_a_demo_refuses_a_directory_that_holds_a_real_run(tmp_path, capsys):
    (tmp_path / "run.json").write_text(json.dumps({"demo": False}), encoding="utf-8")
    deps, *_ = make_deps()
    assert run(["prepare", "--demo", "--out", str(tmp_path)], deps) == rc.EXIT_FAILED
    assert "holds a real run" in capsys.readouterr().err
    (tmp_path / "run.json").write_text(json.dumps({"demo": True}), encoding="utf-8")
    assert run(["prepare", "--out", str(tmp_path)], deps) == rc.EXIT_FAILED


def _states(path: Path) -> list[str]:
    return [json.loads(line)["state"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_test_split_ids_never_reach_a_training_file(prepared):
    info = json.loads((prepared / "split.json").read_text())
    ids = {name: set(values) for name, values in info["ids"].items()}
    assert ids["train"].isdisjoint(ids["test"]) and ids["val"].isdisjoint(ids["test"])
    assert ids["train"].isdisjoint(ids["val"])
    assert {n: len(v) for n, v in ids.items()} == info["sizes"]

    rows = pd.read_parquet(prepared / "eval_rows.parquet")
    test_texts = set(rows.loc[rows["split"] == "test", "text"].str.strip())
    val_texts = set(rows.loc[rows["split"] == "val", "text"].str.strip())
    train_states = _states(prepared / "laya" / "train.jsonl")
    val_states = _states(prepared / "laya" / "val.jsonl")
    assert test_texts and not test_texts & set(train_states)
    assert not test_texts & set(val_states)
    assert set(val_states) == val_texts  # val.jsonl is exactly the val split
    assert len(train_states) == info["jsonl"]["train"]["rows"]
    # eval_rows holds val and test only. No train row is in it.
    assert set(rows["id"]) == ids["val"] | ids["test"]


def test_a_test_text_in_a_training_file_is_refused(prepared):
    rows = pd.read_parquet(prepared / "eval_rows.parquet")
    leaked = rows.loc[rows["split"] == "test", "text"].iloc[0]
    with (prepared / "laya" / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"state": leaked}) + "\n")
    argv = ["prepare", "--demo", "--out", str(prepared)]
    ctx = rc.Context(args=rc.build_parser().parse_args(argv))
    work = pd.DataFrame({"text": rows.loc[rows["split"] == "test", "text"].tolist()})
    split = types.SimpleNamespace(test=np.arange(len(work)))
    with pytest.raises(rc.StageError) as caught:
        rc._check_no_text_overlap(ctx, work, split)
    assert caught.value.code == rc.EXIT_SPLIT


def test_prepare_prints_the_exporter_skip_counts(tmp_path):
    deps, _, _, out = make_deps()
    assert run(["prepare", "--demo", "--out", str(tmp_path / "o")], deps) == 0
    text = out.getvalue()
    assert re.search(r"train\.jsonl: \d+ rows, skipped: none", text)
    assert re.search(r"val\.jsonl: \d+ rows, skipped: none", text)


def test_train_module_reproduces_the_prepared_split(prepared, tmp_path, monkeypatch):
    """The roberta path builds its own frame. It must see the same rows in the same order."""
    import modeling.config as config
    from modeling.text.misinfo_clf import TrainedMisinfoModel

    captured = {}

    class Recorder:
        fallback_model = "distilbert-base-uncased"
        base_model = "roberta-base"
        batch_size = 16

        def __init__(self, settings=None):
            self.settings = settings

        def fine_tune(self, train_texts, train_labels, val_texts, val_labels, **kwargs):
            captured.update(train=list(train_texts), val=list(val_texts))
            Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
            return TrainedMisinfoModel(
                model_dir=kwargs["output_dir"],
                base_model="fake",
                calibrator=None,
                threshold=0.5,
                metadata={},
            )

        def predict(self, texts):
            return np.linspace(0.05, 0.95, len(texts))

    monkeypatch.setattr("modeling.text.misinfo_clf.MisinfoClassifier", Recorder)
    deps, *_ = make_deps()
    code = run(["train-misinfo", "--demo", "--out", str(prepared)], deps)
    assert code == 0, "train_module did not complete on the fixtures"
    summary = json.loads((prepared / "summaries" / "misinfo.json").read_text())
    info = json.loads((prepared / "split.json").read_text())
    assert summary["split"] == info["description"]

    rows = pd.read_parquet(prepared / "eval_rows.parquet")
    assert captured["val"] == rows.loc[rows["split"] == "val", "text"].tolist()
    assert [s.strip() for s in captured["train"]] == _states(prepared / "laya" / "train.jsonl")
    # train_module must not leave the real output directories behind
    assert (prepared / "models" / "misinfo").exists()
    assert config.get_settings().models_dir != prepared / "models"


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def test_every_model_and_part_is_present_and_accepted_by_the_harness(scored):
    models = load_score_dir(scored / "scores")  # validates the schema of every file
    assert set(models) == set(rc.MODELS)
    for name, model in models.items():
        assert set(model.frame["part"]) == set(PARTS), name
        assert list(model.frame.columns) == [
            "part", "id", "lang", "label", "score", "source_detail", "skip_reason"
        ]


def test_all_models_score_the_same_val_and_test_ids(scored):
    frames = score_frames(scored)
    info = json.loads((scored / "split.json").read_text())
    for part in ("val", "test"):
        reference = set(info["ids"][part])
        for name, frame in frames.items():
            ids = set(frame.loc[frame["part"] == part, "id"])
            assert ids == reference, (name, part)
    for frame in frames.values():
        labels = frame[frame["part"].isin(["val", "test"])]["label"]
        assert labels.isin([0, 1]).all()


def test_output_is_exhaustive_every_row_has_a_score_or_a_reason(scored):
    ctx = rc.Context(args=rc.build_parser().parse_args(["score", "--demo", "--out", str(scored)]))
    parts = rc.build_parts(ctx)
    for name, frame in score_frames(scored).items():
        for part in PARTS:
            sub = frame[frame["part"] == part]
            assert sorted(sub["id"]) == sorted(parts[part]["id"]), (name, part)
            has_score = sub["score"].notna()
            has_reason = sub["skip_reason"].notna()
            assert (has_score ^ has_reason).all(), (name, part)
    summary = json.loads((scored / "summaries" / "score.json").read_text())
    assert set(summary) == set(rc.MODELS)


def test_hindi_is_handled_per_model(scored):
    frames = score_frames(scored)
    for name in ("tfidf", "misinfo"):
        hindi = frames[name].query("lang == 'hi'")
        assert len(hindi) >= 3
        assert hindi["score"].isna().all()
        assert set(hindi["skip_reason"]) <= {"unsupported_language", "no_title"}
        assert "unsupported_language" in set(hindi["skip_reason"])
    hindi = frames["laya_en"].query("lang == 'hi'")
    assert hindi["score"].isna().all()
    # LayaScorer checks the text length before it routes, so a short title can be too short
    assert set(hindi["skip_reason"]) <= {"unrouted_language", "no_title", "text_too_short"}
    assert "unrouted_language" in set(hindi["skip_reason"])
    for name in ("laya_multi", "laya_multi_base"):
        hindi = frames[name].query("lang == 'hi' and part == 'claims'")
        assert len(hindi) == 2 and hindi["score"].notna().all()
        assert frames[name].query("lang == 'hi'")["score"].notna().any()


def test_english_claims_score_for_every_model(scored):
    for name, frame in score_frames(scored).items():
        english = frame.query("part == 'claims' and lang == 'en'")
        assert len(english) == 4 and english["score"].notna().all(), name


def test_a_record_with_no_title_gets_a_reason_and_never_an_empty_score(scored):
    for name, frame in score_frames(scored).items():
        titles = frame[frame["part"] == "corpus_title"]
        none_titled = titles[titles["skip_reason"] == "no_title"]
        # the demo has two records with a title missing, and the fixture rows have none
        assert len(none_titled) >= 2, name
        assert none_titled["score"].isna().all()
        assert titles["score"].dropna().between(0, 1).all()
        text = frame[frame["part"] == "corpus_text"]
        assert "no_title" not in set(text["skip_reason"].dropna())


def test_corpus_rows_carry_source_detail(scored):
    for name, frame in score_frames(scored).items():
        for part in ("corpus_title", "corpus_text"):
            sub = frame[frame["part"] == part]
            assert sub["source_detail"].notna().all(), (name, part)
        publishers = frame.query("part == 'corpus_text'")["source_detail"]
        assert sorted(load_factcheck_publishers())[0] in set(publishers)


def test_corpus_holds_only_english_and_hindi_records(scored):
    for frame in score_frames(scored).values():
        corpus = frame[frame["part"].isin(["corpus_title", "corpus_text"])]
        assert set(corpus["lang"]) <= {"en", "hi"}


def test_laya_models_load_the_checkpoints_the_task_names(scored, fake_laya):
    # `scored` already ran once with the fake. Score the three Laya models again.
    for model in ("laya_en", "laya_multi", "laya_multi_base"):
        (scored / "scores" / f"{model}.parquet").unlink()
    fake_laya.loads.clear()
    deps, *_ = make_deps()
    assert run(["score", "--demo", "--out", str(scored)], deps) == 0
    revision = module_config("laya")["revision"]
    by_path = {}
    for checkpoint, kwargs in fake_laya.loads:
        by_path.setdefault(checkpoint, []).append(kwargs)
    en_dir = str(scored / "models" / "laya" / f"en-{LAYA_V}")
    multi_dir = str(scored / "models" / "laya" / f"multi-{LAYA_V}")
    assert en_dir in by_path and multi_dir in by_path
    assert all("revision" not in kw for kw in by_path[en_dir] + by_path[multi_dir])
    base = by_path["convaiinnovations/laya"]
    assert all(kw["revision"] == revision and kw["subfolder"] == "multilingual" for kw in base)


def test_laya_configs_differ_only_in_routing_and_checkpoint():
    base = module_config("laya")
    args = rc.build_parser().parse_args(["score", "--demo", "--out", "/x", "--batch-size", "3"])
    ctx = rc.Context(args=args)
    en = rc.laya_model_config("laya_en", ctx)
    multi = rc.laya_model_config("laya_multi", ctx)
    multi_base = rc.laya_model_config("laya_multi_base", ctx)
    assert en["route"] == {"en": "checkpoint"}
    assert en["finetuned_checkpoints"]["hi"] is None
    assert multi["route"] == multi_base["route"] == {
        "en": "multilingual_checkpoint",
        "hi": "multilingual_checkpoint",
    }
    assert multi["finetuned_checkpoints"]["en"] == multi["finetuned_checkpoints"]["hi"]
    assert multi_base["finetuned_checkpoints"] == {"en": None, "hi": None}
    assert multi_base["revision"] == base["revision"]
    assert en["batch_size"] == 3
    assert module_config("laya")["finetuned_checkpoints"] == base["finetuned_checkpoints"]
    assert base["finetuned_checkpoints"] == {"en": None, "hi": None}  # the shared config is intact


def test_score_refuses_when_a_checkpoint_is_missing_before_it_reads_the_gpu(
    prepared, fake_laya, fake_misinfo, capsys
):
    deps, gpu, *_ = make_deps()
    assert run(["score", "--demo", "--out", str(prepared)], deps) == rc.EXIT_FAILED
    err = capsys.readouterr().err
    assert "checkpoints are missing" in err and "laya_en" in err
    assert gpu.calls == []
    assert not (prepared / "scores").exists()


def test_oom_during_laya_scoring_stops_the_stage_and_writes_no_partial_file(
    prepared, fake_laya, fake_misinfo, capsys
):
    add_checkpoints(prepared)
    fake_laya.fail = RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB")
    deps, *_ = make_deps()
    assert run(["score", "--demo", "--out", str(prepared)], deps) == rc.EXIT_GPU_OOM
    err = capsys.readouterr().err
    assert "--batch-size" in err
    scores = {p.stem for p in (prepared / "scores").glob("*.parquet")}
    assert scores == {"tfidf", "misinfo"}  # finished models are kept, laya_en is not written
    # a rerun with the memory problem gone resumes at laya_en
    fake_laya.fail = None
    deps, *_ = make_deps()
    assert run(["score", "--demo", "--out", str(prepared)], deps) == 0
    assert {p.stem for p in (prepared / "scores").glob("*.parquet")} == set(rc.MODELS)


def test_other_laya_errors_are_counted_not_hidden(prepared, fake_laya, fake_misinfo):
    add_checkpoints(prepared)
    fake_laya.fail = RuntimeError("tokenizer exploded")
    deps, *_ = make_deps()
    # every Laya record fails with inference_error; the model scored no val rows, so no file
    code = run(["score", "--demo", "--out", str(prepared), "--models", "laya_en"], deps)
    assert code == rc.EXIT_FAILED
    assert not (prepared / "scores" / "laya_en.parquet").exists()


def test_score_part_returns_one_row_per_input_with_a_score_or_reason():
    from modeling.aux.base import ScoringOutcome

    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "text": ["fine text", "", None, "another fine text", "third"],
            "lang": ["en", "en", "hi", "hi", None],
            "label": None,
            "source_detail": ["s1", None, "s3", "", "s5"],
        }
    )

    def scorer(sub):
        outcome = ScoringOutcome()
        for record_id in sub["id"]:
            if record_id == "a":
                outcome.values[record_id] = 0.4
            elif record_id == "d":
                outcome.skipped[record_id] = "unsupported_language"
            # "e" is dropped on purpose
        return outcome

    rows = {r["id"]: r for r in rc.score_part(scorer, "claims", frame)}
    assert list(rows) == ["a", "b", "c", "d", "e"]
    assert rows["a"]["score"] == 0.4 and rows["a"]["skip_reason"] is None
    assert rows["b"]["skip_reason"] == "empty_text" and rows["b"]["score"] is None
    assert rows["c"]["skip_reason"] == "empty_text"
    assert rows["d"]["skip_reason"] == "unsupported_language"
    assert rows["e"]["skip_reason"] == "scorer_dropped_record"
    assert rows["d"]["source_detail"] is None and rows["a"]["source_detail"] == "s1"
    titles = {r["id"]: r for r in rc.score_part(scorer, "corpus_title", frame)}
    assert titles["b"]["skip_reason"] == "no_title" and titles["c"]["skip_reason"] == "no_title"


def test_score_part_records_an_out_of_range_score_as_invalid_not_as_a_score():
    from modeling.aux.base import ScoringOutcome

    frame = pd.DataFrame(
        {"id": ["a", "b"], "text": ["x y z", "p q r"], "lang": ["en", "en"],
         "label": None, "source_detail": None}
    )

    def scorer(sub):
        return ScoringOutcome(values={"a": float("nan"), "b": 1.5})

    rows = rc.score_part(scorer, "claims", frame)
    assert [r["skip_reason"] for r in rows] == ["invalid_score", "invalid_score"]
    assert all(r["score"] is None for r in rows)


def test_score_part_refuses_duplicate_ids():
    frame = pd.DataFrame({"id": ["a", "a"], "text": ["x", "y"], "lang": "en",
                          "label": None, "source_detail": None})
    with pytest.raises(rc.StageError):
        rc.score_part(lambda sub: None, "claims", frame)


def test_misinfo_and_tfidf_skip_an_unknown_language_rather_than_assume_english(
    prepared, fake_misinfo
):
    ctx = rc.Context(args=rc.build_parser().parse_args(["score", "--demo", "--out", str(prepared)]))
    frame = pd.DataFrame({"id": ["a", "b"], "text": ["some english words", "more words"],
                          "lang": [None, "en"]})
    for make in (rc.make_misinfo_scorer, rc.make_tfidf_scorer):
        (scorer, _) = make(ctx)
        outcome = scorer(frame)
        assert outcome.skipped == {"a": "unknown_language"}
        assert "b" in outcome.values


def test_a_laya_scorer_skips_an_unknown_language(prepared, fake_laya):
    add_checkpoints(prepared)
    ctx = rc.Context(args=rc.build_parser().parse_args(["score", "--demo", "--out", str(prepared)]))
    scorer, version = rc.make_laya_scorer("laya_multi", ctx)
    texts = ["a claim long enough to score", "another long claim here"]
    # pandas 3 stores the missing language as NaN. score_part must hand the scorer a None.
    frame = pd.DataFrame(
        {"id": ["a", "b"], "text": texts, "lang": [None, "hi"], "label": None, "source_detail": ""}
    )
    rows = {r["id"]: r for r in rc.score_part(scorer, "claims", frame)}
    assert rows["a"]["skip_reason"] == "unknown_language" and rows["a"]["score"] is None
    assert rows["a"]["lang"] is None
    assert rows["b"]["score"] is not None
    assert version == f"{LAYA_V}+finetuned:multi-{LAYA_V}"
    base_scorer, base_version = rc.make_laya_scorer("laya_multi_base", ctx)
    assert base_version.startswith(f"{LAYA_V}+base@")


def test_laya_that_is_not_installed_refuses_instead_of_writing_empty_scores(prepared, monkeypatch):
    add_checkpoints(prepared)
    monkeypatch.setitem(sys.modules, "laya", None)  # makes `import laya` raise ImportError
    ctx = rc.Context(args=rc.build_parser().parse_args(["score", "--demo", "--out", str(prepared)]))
    with pytest.raises(rc.StageError, match="laya is not installed"):
        rc.make_laya_scorer("laya_en", ctx)


def test_corpus_read_falls_back_when_the_dataset_read_fails(prepared, monkeypatch, caplog):
    ctx = rc.Context(args=rc.build_parser().parse_args(["score", "--demo", "--out", str(prepared)]))
    expected, _ = rc.read_corpus(ctx)

    def broken(self, *a, **k):
        raise ValueError("Unable to merge: Field source has incompatible types")

    monkeypatch.setattr("ingest.store.ParquetStore.iter_records", broken)
    with caplog.at_level(logging.WARNING):
        fallback, _ = rc.read_corpus(ctx)
    assert "reading file by file" in caplog.text
    assert sorted(fallback["id"]) == sorted(expected["id"])


def test_corpus_read_counts_other_languages_and_titles(prepared):
    ctx = rc.Context(args=rc.build_parser().parse_args(["score", "--demo", "--out", str(prepared)]))
    frame, counts = rc.read_corpus(ctx)
    assert counts["other_language"] >= 1  # the fixture has one German record
    assert counts["kept"] == len(frame)
    assert set(frame["lang"]) == {"en", "hi"}
    assert frame["title"].notna().sum() == 4 and frame["id"].is_unique


# ---------------------------------------------------------------------------
# frozen inputs, provenance and the selected models
# ---------------------------------------------------------------------------
def score_argv(out, *extra):
    return ["score", "--demo", "--out", str(out), *extra]


def patch_corpus(monkeypatch, *, add=None, drop=None, retitle=None):
    """Make the next corpus read return other rows than the first read did."""
    real = rc.read_corpus

    def changed(ctx):
        frame, counts = real(ctx)
        if drop is not None:
            frame = frame[frame["id"] != drop]
        if retitle is not None:
            frame = frame.copy()
            frame.loc[frame["id"] == retitle, "text"] = "this text is different now"
        if add is not None:
            extra = pd.DataFrame(
                [{"id": add, "text": "a new article body", "title": "A new headline",
                  "lang": "en", "source_detail": "news.example"}]
            )
            frame = pd.concat([frame, extra], ignore_index=True)
        return frame, counts

    monkeypatch.setattr(rc, "read_corpus", changed)


def rows_of(frame: pd.DataFrame) -> list[dict]:
    """Score-file rows from a frame that was read back, with nulls as None and labels as int."""
    rows = []
    for record in frame.astype(object).where(frame.notna(), None).to_dict("records"):
        if record["label"] is not None:
            record["label"] = int(record["label"])
        rows.append(record)
    return rows


def a_corpus_id(out) -> str:
    return str(load_score_dir(out / "scores")["tfidf"].part("corpus_title")["id"].iloc[0])


def test_the_first_score_run_freezes_the_claim_and_corpus_ids(scored):
    frozen = pd.read_parquet(scored / "score_inputs.parquet")
    frames = score_frames(scored)
    for part in rc.FROZEN_PARTS:
        assert sorted(frozen.loc[frozen["part"] == part, "id"]) == sorted(
            frames["tfidf"].loc[frames["tfidf"]["part"] == part, "id"]
        )
    assert set(frozen.columns) == {"part", "id", "lang", "text_sha256"}


def test_a_later_score_run_reuses_the_frozen_ids_and_leaves_new_rows_out(
    scored, monkeypatch
):
    patch_corpus(monkeypatch, add="brand-new-record")
    (scored / "scores" / "tfidf.parquet").unlink()
    deps, _, _, out = make_deps()
    assert run(score_argv(scored, "--models", "tfidf"), deps) == 0
    assert "new since the freeze" in out.getvalue()
    frames = score_frames(scored)
    for part in ("corpus_title", "corpus_text"):
        ids = {name: set(f.loc[f["part"] == part, "id"]) for name, f in frames.items()}
        assert "brand-new-record" not in ids["tfidf"]
        assert ids["tfidf"] == ids["misinfo"] == ids["laya_en"]


def test_a_later_score_run_refuses_when_a_frozen_id_is_gone(scored, monkeypatch, capsys):
    victim = a_corpus_id(scored)
    patch_corpus(monkeypatch, drop=victim)
    (scored / "scores" / "tfidf.parquet").unlink()
    deps, gpu, *_ = make_deps()
    assert run(score_argv(scored, "--models", "tfidf"), deps) == rc.EXIT_FAILED
    err = capsys.readouterr().err
    assert "gone from the data" in err and victim in err and "--refreeze" in err
    assert not (scored / "scores" / "tfidf.parquet").exists()


def test_a_later_score_run_refuses_when_a_frozen_text_changed(scored, monkeypatch, capsys):
    victim = a_corpus_id(scored)
    patch_corpus(monkeypatch, retitle=victim)
    (scored / "scores" / "tfidf.parquet").unlink()
    deps, *_ = make_deps()
    assert run(score_argv(scored, "--models", "tfidf"), deps) == rc.EXIT_FAILED
    assert "changed text or language" in capsys.readouterr().err


def test_refreeze_scores_the_chosen_models_again_and_evaluate_refuses_the_others(
    scored, monkeypatch, capsys
):
    patch_corpus(monkeypatch, add="brand-new-record")
    deps, *_ = make_deps()
    assert run(score_argv(scored, "--models", "tfidf", "--refreeze"), deps) == 0
    ids = score_frames(scored)["tfidf"].query("part == 'corpus_title'")["id"]
    assert "brand-new-record" in set(ids)
    # The other four files were made on the old rows. The provenance check stops them.
    capsys.readouterr()
    assert run(["evaluate", "--demo", "--out", str(scored)], deps) == rc.EXIT_FAILED
    err = capsys.readouterr().err
    assert "misinfo" in err and "other claim or corpus rows" in err


def read_provenance(out, model) -> dict:
    return json.loads((out / "scores" / f"{model}.provenance.json").read_text())


def test_every_score_file_has_a_provenance_sidecar(scored):
    split_info = json.loads((scored / "split.json").read_text())
    frozen_hash = rc.input_manifest_hash(pd.read_parquet(scored / "score_inputs.parquet"))
    for model in rc.MODELS:
        recorded = read_provenance(scored, model)
        assert recorded["model"] == model
        assert recorded["split_sha256"] == rc.split_fingerprint(split_info)
        assert recorded["inputs_sha256"] == frozen_hash
        parquet = scored / "scores" / f"{model}.parquet"
        assert recorded["score_file_sha256"] == rc.file_sha256(parquet)
        assert recorded["model_version"]
    misinfo = read_provenance(scored, "misinfo")["checkpoint"]
    assert misinfo["kind"] == "directory" and misinfo["name"] == MISINFO_V
    assert misinfo["backbone"] == "roberta" and len(misinfo["sha256"]) == 64
    assert read_provenance(scored, "laya_en")["checkpoint"]["name"] == f"en-{LAYA_V}"
    assert read_provenance(scored, "laya_multi")["checkpoint"]["name"] == f"multi-{LAYA_V}"
    base = read_provenance(scored, "laya_multi_base")["checkpoint"]
    assert base["kind"] == "pinned_revision"
    assert base["revision"] == module_config("laya")["revision"]
    assert read_provenance(scored, "tfidf")["checkpoint"]["kind"] == "file"


def test_score_scores_again_a_model_whose_checkpoint_changed_and_only_that_one(
    scored, fake_laya
):
    inodes = {p.stem: p.stat().st_ino for p in (scored / "scores").glob("*.parquet")}
    (scored / "models" / "misinfo" / MISINFO_V / "pytorch_model.bin").write_bytes(b"new weights")
    deps, _, _, out = make_deps()
    assert run(score_argv(scored), deps) == 0
    assert "misinfo: the score file is not current (the checkpoint changed" in out.getvalue()
    after = {p.stem: p.stat().st_ino for p in (scored / "scores").glob("*.parquet")}
    assert {m for m in inodes if inodes[m] != after[m]} == {"misinfo"}


def test_score_scores_again_when_the_split_changed(scored, fake_laya):
    path = scored / "split.json"
    info = json.loads(path.read_text())
    info["seed"] = info["seed"] + 1
    path.write_text(json.dumps(info), encoding="utf-8")
    inodes = {p.stem: p.stat().st_ino for p in (scored / "scores").glob("*.parquet")}
    deps, _, _, out = make_deps()
    assert run(score_argv(scored), deps) == 0
    assert "another split" in out.getvalue()
    after = {p.stem: p.stat().st_ino for p in (scored / "scores").glob("*.parquet")}
    assert all(after[m] != inodes[m] for m in inodes)


def test_score_does_not_reuse_a_file_that_has_no_provenance(scored, fake_laya):
    (scored / "scores" / "tfidf.provenance.json").unlink()
    inode = (scored / "scores" / "tfidf.parquet").stat().st_ino
    deps, _, _, out = make_deps()
    assert run(score_argv(scored), deps) == 0
    assert "no provenance file" in out.getvalue()
    assert (scored / "scores" / "tfidf.provenance.json").exists()
    assert (scored / "scores" / "tfidf.parquet").stat().st_ino != inode


def test_evaluate_refuses_a_score_file_without_provenance(scored, capsys):
    (scored / "scores" / "tfidf.provenance.json").unlink()
    deps, *_ = make_deps()
    assert run(["evaluate", "--demo", "--out", str(scored)], deps) == rc.EXIT_FAILED
    err = capsys.readouterr().err
    assert "tfidf" in err and "no provenance file" in err and "--force" in err


def test_evaluate_refuses_after_the_checkpoint_changed(scored, capsys):
    (scored / "models" / "laya" / f"en-{LAYA_V}" / "extra.bin").write_bytes(b"x")
    deps, *_ = make_deps()
    assert run(["evaluate", "--demo", "--out", str(scored)], deps) == rc.EXIT_FAILED
    err = capsys.readouterr().err
    assert "laya_en" in err and "checkpoint changed since scoring" in err


def test_evaluate_refuses_after_the_split_changed(scored, capsys):
    path = scored / "split.json"
    info = json.loads(path.read_text())
    info["ids"]["test"] = info["ids"]["test"][1:]
    path.write_text(json.dumps(info), encoding="utf-8")
    deps, *_ = make_deps()
    assert run(["evaluate", "--demo", "--out", str(scored)], deps) == rc.EXIT_SPLIT
    assert "another split" in capsys.readouterr().err


def test_evaluate_refuses_score_files_whose_ids_differ_from_split_json(scored, capsys):
    # A file with a valid sidecar and a test row missing: the ids check is the one that stops it.
    path = scored / "scores" / "tfidf.parquet"
    frame = pd.read_parquet(path)
    victim = frame.loc[frame["part"] == "test", "id"].iloc[0]
    rc.write_score_file(path, rows_of(frame[frame["id"] != victim]))
    sidecar = scored / "scores" / "tfidf.provenance.json"
    recorded = json.loads(sidecar.read_text())
    recorded["score_file_sha256"] = rc.file_sha256(path)
    sidecar.write_text(json.dumps(recorded), encoding="utf-8")
    deps, *_ = make_deps()
    assert run(["evaluate", "--demo", "--out", str(scored)], deps) == rc.EXIT_SPLIT
    err = capsys.readouterr().err
    assert "tfidf has other test ids than split.json" in err and victim in err


def test_check_scores_match_split_compares_val_and_test():
    info = {"ids": {"val": ["a", "b"], "test": ["c"]}}

    def scores(val, test):
        rows = [{"part": "val", "id": i} for i in val] + [{"part": "test", "id": i} for i in test]
        frame = pd.DataFrame(rows)
        return {"m": types.SimpleNamespace(part=lambda n, f=frame: f[f["part"] == n])}

    rc.check_scores_match_split(scores(["b", "a"], ["c"]), info)
    with pytest.raises(rc.StageError, match="other val ids"):
        rc.check_scores_match_split(scores(["a"], ["c"]), info)
    with pytest.raises(rc.StageError, match="other test ids"):
        rc.check_scores_match_split(scores(["a", "b"], ["c", "d"]), info)


def test_evaluate_leaves_a_leftover_score_file_out_of_the_comparison(scored):
    frame = pd.read_parquet(scored / "scores" / "tfidf.parquet")
    leftover = frame[frame["id"] != frame.loc[frame["part"] == "corpus_title", "id"].iloc[0]]
    rc.write_score_file(scored / "scores" / "old_run.parquet", rows_of(leftover))
    deps, _, _, out = make_deps()
    assert run(["evaluate", "--demo", "--out", str(scored)], deps) == 0
    report = next(line.split("report: ", 1)[1] for line in out.getvalue().splitlines()
                  if line.endswith("factcheck_eval.json"))
    assert set(json.loads(Path(report).read_text())["models"]) == set(rc.MODELS)


def test_evaluate_with_models_compares_only_those_models(scored):
    deps, _, _, out = make_deps()
    assert run(["evaluate", "--demo", "--out", str(scored), "--models", "tfidf,misinfo"], deps) == 0
    report = next(line.split("report: ", 1)[1] for line in out.getvalue().splitlines()
                  if line.endswith("factcheck_eval.json"))
    assert set(json.loads(Path(report).read_text())["models"]) == {"tfidf", "misinfo"}


def test_evaluate_reports_the_version_recorded_at_scoring_not_the_current_config(scored):
    sidecar = scored / "scores" / "misinfo.provenance.json"
    recorded = json.loads(sidecar.read_text())
    recorded["model_version"] = "v9.9.9-recorded"
    sidecar.write_text(json.dumps(recorded), encoding="utf-8")
    deps, _, _, out = make_deps()
    assert run(["evaluate", "--demo", "--out", str(scored)], deps) == 0
    report = next(line.split("report: ", 1)[1] for line in out.getvalue().splitlines()
                  if line.endswith("factcheck_eval.json"))
    models = json.loads(Path(report).read_text())["models"]
    assert models["misinfo"]["version"] == "v9.9.9-recorded"


def test_the_zero_shot_control_is_listed_among_the_models_and_fails_the_gate(scored):
    from modeling.eval.factcheck_eval import CONTROL_MODELS

    assert CONTROL_MODELS <= set(rc.MODELS)
    deps, _, _, out = make_deps()
    assert run(["evaluate", "--demo", "--out", str(scored)], deps) == 0
    text = out.getvalue()
    assert "laya_multi_base: do not ship" in text
    assert "laya_multi_base: Hindi gate FAIL (control)" in text


# ---------------------------------------------------------------------------
# the misinfo backbone
# ---------------------------------------------------------------------------
def test_score_refuses_a_misinfo_checkpoint_that_fell_back_to_distilbert(
    prepared, fake_laya, fake_misinfo, capsys
):
    add_checkpoints(prepared)
    config = prepared / "models" / "misinfo" / MISINFO_V / "config.json"
    config.write_text('{"model_type": "distilbert"}', encoding="utf-8")
    deps, gpu, *_ = make_deps()
    assert run(score_argv(prepared), deps) == rc.EXIT_FAILED
    err = capsys.readouterr().err
    assert "model_type 'distilbert'" in err and "roberta" in err and "--force" in err
    assert gpu.calls == [] and not (prepared / "scores").exists()


def test_score_refuses_a_misinfo_checkpoint_with_no_config_json(
    prepared, fake_laya, fake_misinfo, capsys
):
    add_checkpoints(prepared)
    (prepared / "models" / "misinfo" / MISINFO_V / "config.json").unlink()
    deps, *_ = make_deps()
    assert run(score_argv(prepared), deps) == rc.EXIT_FAILED
    assert "backbone cannot be checked" in capsys.readouterr().err


def test_evaluate_refuses_after_the_misinfo_backbone_changed(scored, capsys):
    config = scored / "models" / "misinfo" / MISINFO_V / "config.json"
    config.write_text('{"model_type": "distilbert"}', encoding="utf-8")
    deps, *_ = make_deps()
    assert run(["evaluate", "--demo", "--out", str(scored)], deps) == rc.EXIT_FAILED
    assert "model_type 'distilbert'" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def test_evaluate_refuses_when_a_score_file_is_missing(scored, capsys):
    (scored / "scores" / "laya_en.parquet").unlink()
    deps, *_ = make_deps()
    assert run(["evaluate", "--demo", "--out", str(scored)], deps) == rc.EXIT_FAILED
    assert "laya_en" in capsys.readouterr().err


def test_evaluate_runs_the_harness_and_prints_the_report_path(scored):
    deps, _, _, out = make_deps()
    assert run(["evaluate", "--demo", "--out", str(scored)], deps) == 0
    text = out.getvalue()
    reports = [line.split("report: ", 1)[1] for line in text.splitlines() if "report: " in line]
    assert len(reports) == 2
    for report in reports:
        assert Path(report).exists()
        assert Path(report).is_relative_to(scored)  # nothing leaves the output directory
    payload = json.loads(Path(reports[0]).read_text())
    assert payload["split"]["description"].startswith("[DEMO] grouped by group_id")
    assert set(payload["models"]) == set(rc.MODELS)
    versions = {name: model["version"] for name, model in payload["models"].items()}
    assert versions["misinfo"] == MISINFO_V
    assert versions["laya_en"] == f"{LAYA_V}+finetuned:en-{LAYA_V}"
    assert versions["laya_multi"] == f"{LAYA_V}+finetuned:multi-{LAYA_V}"
    revision = module_config("laya")["revision"][:12]
    assert versions["laya_multi_base"] == f"{LAYA_V}+base@{revision}"
    assert versions["tfidf"].startswith("tfidf-logreg+sklearn-")


def test_the_full_demo_flow_runs_end_to_end_with_fakes(
    tmp_path, monkeypatch, fake_misinfo, fake_finetune
):
    # fake_finetune installs fake `laya` and `laya.train` modules. Give `laya` a loader too.
    sys.modules["laya"].load = lambda checkpoint, **kwargs: FakeLayaAgent()
    out = tmp_path / "demo"

    def fake_train(module, *, demo, epochs, settings, **kwargs):
        directory = Path(settings.models_dir) / "misinfo" / MISINFO_V
        directory.mkdir(parents=True)
        (directory / "calibrator.json").write_text("{}", encoding="utf-8")
        (directory / "config.json").write_text('{"model_type": "roberta"}', encoding="utf-8")
        return misinfo_result(split_of(out))

    monkeypatch.setattr("modeling.training.train_module", fake_train)
    deps, gpu, *_ = make_deps()
    assert run(["all", "--demo", "--out", str(out)], deps) == 0
    assert len(gpu.calls) == 8  # a gate and an end report for each of the four GPU stages
    assert {p.stem for p in (out / "scores").glob("*.parquet")} == set(rc.MODELS)
    assert list(out.rglob("factcheck_eval.md"))
    assert not (rc.REPO_ROOT / "data" / "comparison_demo").exists()
    assert not (rc.REPO_ROOT / "data" / "comparison").exists()


def test_a_loader_rewrite_of_the_tokenizer_config_keeps_the_checkpoint_identity(tmp_path):
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "tokenizer" / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer" / "tokenizer_config.json").write_text('{"backend": "tokenizers"}')
    before = rc.directory_identity(tmp_path)

    (tmp_path / "tokenizer" / "tokenizer_config.json").write_text("{}")
    assert rc.directory_identity(tmp_path) == before

    (tmp_path / "tokenizer" / "tokenizer.json").write_text('{"vocab": 1}')
    assert rc.directory_identity(tmp_path) != before


def test_a_root_level_tokenizer_config_stays_in_the_checkpoint_identity(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{}")
    before = rc.directory_identity(tmp_path)

    (tmp_path / "tokenizer_config.json").write_text('{"model_max_length": 512}')
    assert rc.directory_identity(tmp_path) != before


# ---------------------------------------------------------------------------
# calibrate-laya
# ---------------------------------------------------------------------------
def _synthetic_laya_scores(
    out: Path, test_scores=None, invert: bool = False, null_rows: int = 0
) -> None:
    """Replace the laya_multi score file with synthetic validation rows and its provenance."""
    from tests.test_laya_stage import synthetic_validation

    path = out / "scores" / "laya_multi.parquet"
    frame = pd.read_parquet(path).astype(object)
    rows = [
        {k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
        for r in frame.to_dict(orient="records")
        if r["part"] != "val"
    ]
    scores, labels = synthetic_validation()
    if invert:
        scores = 1.0 - scores
    for i, (score, label) in enumerate(zip(scores, labels, strict=True)):
        rows.append({"part": "val", "id": f"syn{i}", "lang": "en", "label": int(label),
                     "score": float(score), "source_detail": None, "skip_reason": None})
    for r in [r for r in rows if r["id"].startswith("syn")][:null_rows]:
        r["score"] = None
    if test_scores is not None:
        for r in rows:
            if r["part"] == "test":
                r["score"] = test_scores
    rc.write_score_file(path, rows)
    sidecar = out / "scores" / "laya_multi.provenance.json"
    recorded = json.loads(sidecar.read_text())
    recorded["score_file_sha256"] = rc.file_sha256(path)
    sidecar.write_text(json.dumps(recorded))


@pytest.fixture
def promoted(scored, tmp_path):
    """A copy of the comparison checkpoint outside the comparison output, as the runbook says."""
    directory = tmp_path / "promoted" / f"multi-{LAYA_V}"
    shutil.copytree(scored / "models" / "laya" / f"multi-{LAYA_V}", directory)
    return directory


def _calibrate(out, directory):
    deps, _, _, text = make_deps()
    code = run(["calibrate-laya", "--demo", "--out", str(out), "--checkpoint-dir", str(directory)],
               deps)
    return code, text.getvalue()


def test_calibrate_laya_writes_the_state_and_the_report(scored, promoted):
    from modeling.text.laya_calibration import (
        CALIBRATOR_FILE,
        REPORT_FILE,
        LayaCalibratorError,
        load_calibrator,
    )

    _synthetic_laya_scores(scored)
    own = scored / "models" / "laya" / f"multi-{LAYA_V}"
    identity = rc.directory_identity(own)
    code, text = _calibrate(scored, promoted)
    assert code == 0, text
    report = json.loads((promoted / REPORT_FILE).read_text())
    assert report["rows"] == 400 and report["fit_part"] == "val"
    assert report["demo"] is True  # the demo flow fits, and the stage refuses the result
    assert report["dropped_null_rows"] == 0
    assert report["checkpoint_identity"]["sha256"] == identity["sha256"]
    assert json.loads((promoted / CALIBRATOR_FILE).read_text())["demo"] is True
    assert {"method", "brier_before", "brier_after", "ece_before", "ece_after"} <= set(report)
    assert "English" in report["label_statement"] and "not measured" in report["label_statement"]
    with pytest.raises(LayaCalibratorError) as info:
        load_calibrator(promoted)
    assert info.value.reason == "calibrator_demo"
    assert (promoted / CALIBRATOR_FILE).exists()
    assert rc.directory_identity(own) == identity, "the comparison checkpoint must not change"
    assert not (own / CALIBRATOR_FILE).exists()


def test_calibrate_laya_never_reads_test_rows(scored, promoted, tmp_path):
    from modeling.text.laya_calibration import CALIBRATOR_FILE, fit_calibrator
    from tests.test_laya_stage import synthetic_validation

    _synthetic_laya_scores(scored, test_scores=0.0)
    code, text = _calibrate(scored, promoted)
    assert code == 0, text
    expected, _ = fit_calibrator(*synthetic_validation())
    state = json.loads((promoted / CALIBRATOR_FILE).read_text())
    assert {k: state[k] for k in expected.state()} == json.loads(json.dumps(expected.state()))


def test_calibrate_laya_writes_nothing_when_the_fit_is_rejected(scored, promoted, monkeypatch):
    import modeling.text.laya_calibration as LC

    _synthetic_laya_scores(scored)

    def reject(scores, labels, method="isotonic"):
        raise LC.LayaCalibratorError(LC.REJECTED, "inverted ranking")

    monkeypatch.setattr(LC, "fit_calibrator", reject)
    code, _ = _calibrate(scored, promoted)
    assert code != 0
    assert sorted(p.name for p in promoted.iterdir()) == ["run_summary.json"]


@pytest.mark.parametrize("where", ["checkpoint", "models_root", "inside"])
def test_calibrate_laya_refuses_a_directory_in_the_comparison_models(scored, where):
    own = scored / "models" / "laya" / f"multi-{LAYA_V}"
    _synthetic_laya_scores(scored)
    target = {
        "checkpoint": own,
        "models_root": scored / "models",
        "inside": scored / "models" / "laya" / "copy",
    }[where]
    if where == "inside":
        shutil.copytree(own, target)
    code, _ = _calibrate(scored, target)
    assert code != 0
    assert not (target / "misinfo_prob_calibrator.json").exists()


def test_calibrate_laya_refuses_a_directory_with_other_weights(scored, promoted):
    _synthetic_laya_scores(scored)
    (promoted / "run_summary.json").write_text('{"other": true}')
    code, _ = _calibrate(scored, promoted)
    assert code != 0
    assert not (promoted / "misinfo_prob_calibrator.json").exists()


def test_calibrate_laya_ignores_calibrator_files_already_in_the_directory(scored, promoted):
    _synthetic_laya_scores(scored)
    assert _calibrate(scored, promoted)[0] == 0
    assert _calibrate(scored, promoted)[0] == 0


def test_calibrate_laya_refuses_inverted_scores_and_writes_nothing(scored, promoted):
    _synthetic_laya_scores(scored, invert=True)
    code, _ = _calibrate(scored, promoted)
    assert code != 0
    assert not (promoted / "misinfo_prob_calibrator.json").exists()
    assert not (promoted / "misinfo_prob_calibration_report.json").exists()


def test_calibrate_laya_counts_the_dropped_null_rows(scored, promoted):
    from modeling.text.laya_calibration import REPORT_FILE

    _synthetic_laya_scores(scored, null_rows=7)
    assert _calibrate(scored, promoted)[0] == 0
    report = json.loads((promoted / REPORT_FILE).read_text())
    assert report["dropped_null_rows"] == 7 and report["rows"] == 393


def test_calibrate_laya_refuses_a_stale_score_file(scored, promoted):
    _synthetic_laya_scores(scored)
    sidecar = scored / "scores" / "laya_multi.provenance.json"
    recorded = json.loads(sidecar.read_text())
    recorded["score_file_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(recorded))
    code, _ = _calibrate(scored, promoted)
    assert code != 0
    assert sorted(p.name for p in promoted.iterdir()) == ["run_summary.json"]
