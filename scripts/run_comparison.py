"""Train and score the models for the three-way comparison, then run the harness.

Stages, in order (``all`` runs them and stops at the first failure)::

    prepare        LIAR + FakeNewsNet + CoAID -> grouped split, Laya JSONL, TF-IDF
    train-misinfo  roberta fine-tune through modeling.training.train_module
    train-laya     laya.train.finetune, English and multilingual
    score          one <model>.parquet per model, in the harness input schema
    evaluate       modeling.eval.factcheck_eval.run_comparison

Everything is written under one directory (``--out``, default ``data/comparison``,
which ``/data/`` in ``.gitignore`` covers). ``--demo`` runs the same flow on the
committed fixtures, with a different default directory, and refuses to reuse a
directory that holds a real run.

The three stages that use the GPU (``train-misinfo``, ``train-laya``, ``score``)
read free memory from ``nvidia-smi`` first. The GPU host is shared, so this script
never stops another process: it refuses, or with ``--wait-for-gpu`` it waits.

Exit codes::

    0   ok
    1   a stage failed
    2   bad command line
    10  not enough free GPU memory
    11  waited ``--wait-hours`` and the GPU stayed busy
    12  ``nvidia-smi`` is missing or unreadable
    13  the GPU ran out of memory during a stage
    14  the split is not the split that the comparison requires
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # `python scripts/run_comparison.py` has no package root
    sys.path.insert(0, str(REPO_ROOT))

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_GPU_BUSY = 10
EXIT_GPU_TIMEOUT = 11
EXIT_NO_NVIDIA_SMI = 12
EXIT_GPU_OOM = 13
EXIT_SPLIT = 14

#: The split that the roberta baseline used. Any other split is a different
#: comparison, so ``prepare`` stops unless ``--demo`` is set.
EXPECTED_SPLIT = (
    "grouped by group_id (30148 groups; train/val/test = 26777/3596/7085; seed=20260813)"
)

DEFAULT_OUT = Path("data") / "comparison"
DEFAULT_DEMO_OUT = Path("data") / "comparison_demo"

#: Free GPU memory that each kind of stage needs, in MiB.
DEFAULT_MIN_FREE_TRAIN_MIB = 8000
DEFAULT_MIN_FREE_SCORE_MIB = 6000
GPU_POLL_SECONDS = 300
DEFAULT_WAIT_HOURS = 12.0

BENCHMARKS = ("liar", "fakenewsnet", "coaid")
MODELS = ("tfidf", "misinfo", "laya_en", "laya_multi", "laya_multi_base")
SCORE_PARTS = ("val", "test", "claims", "corpus_title", "corpus_text")
CORPUS_LANGUAGES = ("en", "hi")
LAYA_WHICH = ("en", "multilingual")

#: Skip reason codes that this script adds to the ones in ``modeling.aux.base``.
NO_TITLE = "no_title"
UNKNOWN_LANGUAGE = "unknown_language"
SCORER_DROPPED_RECORD = "scorer_dropped_record"
INVALID_SCORE = "invalid_score"
INFERENCE_ERROR = "inference_error"
EMPTY_TEXT = "empty_text"
UNSUPPORTED_LANGUAGE = "unsupported_language"

log = logging.getLogger("run_comparison")


class StageError(RuntimeError):
    """A stage cannot continue. ``code`` is the process exit code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class GpuOutOfMemory(RuntimeError):
    """The GPU ran out of memory while a Laya checkpoint was scoring."""


# ---------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GpuMemory:
    index: int
    total_mib: int
    used_mib: int
    free_mib: int


def read_gpu_memory(index: int) -> GpuMemory:
    """Read one GPU's memory from ``nvidia-smi``. Never loads a model or uses the GPU."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise StageError(
            EXIT_NO_NVIDIA_SMI, f"cannot run nvidia-smi ({exc}). Free GPU memory is unknown."
        ) from exc
    if result.returncode != 0:
        raise StageError(
            EXIT_NO_NVIDIA_SMI,
            f"nvidia-smi exited {result.returncode}: {result.stderr.strip()[:200]}",
        )
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            values = [int(part) for part in parts]
        except ValueError:
            continue
        if values[0] == index:
            return GpuMemory(index, values[1], values[2], values[3])
    raise StageError(
        EXIT_NO_NVIDIA_SMI,
        f"nvidia-smi has no readable row for GPU {index}. "
        f"Output was: {result.stdout.strip()[:200]!r}",
    )


def default_gpu_index() -> int:
    """The first entry of CUDA_VISIBLE_DEVICES when it is an integer, else 0."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(visible) if visible.isdigit() else 0


@dataclass
class Deps:
    """Everything the stages take from outside the process. Tests replace these."""

    gpu_reader: Callable[[int], GpuMemory] = read_gpu_memory
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    out: TextIO = field(default_factory=lambda: sys.stdout)


def is_oom(exc: BaseException | None) -> bool:
    """True when ``exc``, or anything it was raised from, is a CUDA out-of-memory error."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class _OomWatch(logging.Handler):
    """Raise at once when the Laya scorer logs a CUDA out-of-memory failure.

    ``LayaScorer`` catches every inference exception, logs a warning and turns the
    record into an ``inference_error`` skip. An out-of-memory error would then
    become thousands of skipped rows and a score file that looks complete. A
    handler that raises propagates out of the logging call, which is inside the
    scorer's own ``except`` block, so the exception leaves the scorer.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if "out of memory" in record.getMessage().lower():
            raise GpuOutOfMemory(record.getMessage())


@contextlib.contextmanager
def watch_laya_oom() -> Iterator[None]:
    logger = logging.getLogger("modeling.text.laya_clf")
    handler = _OomWatch(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
@dataclass
class Context:
    args: argparse.Namespace
    deps: Deps = field(default_factory=Deps)

    # --- paths -----------------------------------------------------------
    @property
    def demo(self) -> bool:
        return bool(getattr(self.args, "demo", False))

    @property
    def force(self) -> bool:
        return bool(getattr(self.args, "force", False))

    @property
    def out(self) -> Path:
        value = getattr(self.args, "out", None)
        if value is None:
            value = DEFAULT_DEMO_OUT if self.demo else DEFAULT_OUT
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def run_file(self) -> Path:
        return self.out / "run.json"

    @property
    def split_file(self) -> Path:
        return self.out / "split.json"

    @property
    def eval_rows_file(self) -> Path:
        return self.out / "eval_rows.parquet"

    @property
    def laya_dir(self) -> Path:
        return self.out / "laya"

    @property
    def train_jsonl(self) -> Path:
        return self.laya_dir / "train.jsonl"

    @property
    def val_jsonl(self) -> Path:
        return self.laya_dir / "val.jsonl"

    @property
    def tfidf_file(self) -> Path:
        return self.out / "tfidf_misinfo.joblib"

    @property
    def summaries_dir(self) -> Path:
        return self.out / "summaries"

    @property
    def scores_dir(self) -> Path:
        return self.out / "scores"

    @property
    def models_dir(self) -> Path:
        return self.out / "models"

    @property
    def demo_data_dir(self) -> Path:
        return self.out / "demo_data"

    def misinfo_dir(self) -> Path:
        from modeling.config import module_config

        return self.models_dir / "misinfo" / str(module_config("misinfo").get("version", "v0.0.0"))

    def laya_checkpoint_dir(self, which: str) -> Path:
        from modeling.config import module_config

        version = str(module_config("laya").get("version", "v0.0.0"))
        prefix = "en" if which == "en" else "multi"
        return self.models_dir / "laya" / f"{prefix}-{version}"

    def say(self, message: str = "") -> None:
        print(message, file=self.deps.out, flush=True)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically, so an interrupted run leaves no half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_run_marker(ctx: Context) -> None:
    """Refuse to mix a demo run and a real run in one directory."""
    if not ctx.run_file.exists():
        return
    marker = read_json(ctx.run_file)
    if bool(marker.get("demo")) != ctx.demo:
        kind = "a demo run" if marker.get("demo") else "a real run"
        raise StageError(
            EXIT_FAILED,
            f"{ctx.out} holds {kind}, and this command is "
            f"{'a demo' if ctx.demo else 'not a demo'}. Use another --out.",
        )


def require_prepared(ctx: Context) -> dict[str, Any]:
    check_run_marker(ctx)
    if not ctx.split_file.exists():
        raise StageError(EXIT_FAILED, f"no {ctx.split_file}. Run the prepare stage first.")
    return read_json(ctx.split_file)


@contextlib.contextmanager
def scoped_settings(ctx: Context) -> Iterator[Any]:
    """Point ``modeling`` settings at the output directory, then restore them.

    ``train_module`` and ``register`` write checkpoints, reports and a registry file
    under ``models_dir`` and ``artifacts_dir``. In a demo, ``data_dir`` moves too, so
    nothing under the real ``data/`` changes.
    """
    from modeling.config import get_settings

    overrides = {
        "MODELS_DIR": str(ctx.models_dir),
        "ARTIFACTS_DIR": str(ctx.out / "artifacts"),
    }
    if ctx.demo:
        overrides["DATA_DIR"] = str(ctx.demo_data_dir)
    saved = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    get_settings.cache_clear()
    try:
        yield get_settings()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# GPU gate and stage wrapper
# ---------------------------------------------------------------------------
def ensure_gpu_memory(ctx: Context, name: str, default_min_mib: int) -> GpuMemory:
    """Return once the GPU has enough free memory. Refuse, or wait, when it has not."""
    args = ctx.args
    index = args.gpu_index if getattr(args, "gpu_index", None) is not None else default_gpu_index()
    need = args.min_free_mib if getattr(args, "min_free_mib", None) is not None else default_min_mib
    wait = bool(getattr(args, "wait_for_gpu", False))
    deadline = ctx.deps.clock() + float(getattr(args, "wait_hours", DEFAULT_WAIT_HOURS)) * 3600
    while True:
        memory = ctx.deps.gpu_reader(index)
        if memory.free_mib >= need:
            ctx.say(
                f"[{name}] GPU {index}: {memory.free_mib} MiB free of {memory.total_mib} MiB, "
                f"need {need} MiB"
            )
            return memory
        if not wait:
            raise StageError(
                EXIT_GPU_BUSY,
                f"[{name}] GPU {index} has {memory.free_mib} MiB free and the stage needs "
                f"{need} MiB. Another job holds the rest. This script does not stop other "
                "processes. Wait, add --wait-for-gpu, or lower --min-free-mib with care.",
            )
        if ctx.deps.clock() >= deadline:
            raise StageError(
                EXIT_GPU_TIMEOUT,
                f"[{name}] GPU {index} stayed below {need} MiB free for "
                f"{getattr(args, 'wait_hours', DEFAULT_WAIT_HOURS):g} hours.",
            )
        ctx.say(
            f"[{name}] GPU {index} has {memory.free_mib} MiB free, need {need} MiB. "
            f"Next check in {GPU_POLL_SECONDS // 60} minutes."
        )
        ctx.deps.sleep(GPU_POLL_SECONDS)


def format_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def oom_message(name: str, option: str) -> str:
    return (
        f"[{name}] the GPU ran out of memory. Lower the batch size with {option} and run "
        "the stage again. Stages that finished are kept and are not repeated."
    )


def run_gpu_stage(
    ctx: Context, name: str, default_min_mib: int, option: str, body: Callable[[], Any]
) -> Any:
    """Gate on free GPU memory, run ``body``, translate an OOM, report time and memory."""
    ensure_gpu_memory(ctx, name, default_min_mib)
    started = ctx.deps.clock()
    try:
        result = body()
    except StageError:
        raise
    except Exception as exc:
        if is_oom(exc):
            raise StageError(EXIT_GPU_OOM, oom_message(name, option)) from exc
        raise
    elapsed = ctx.deps.clock() - started
    index = getattr(ctx.args, "gpu_index", None)
    index = index if index is not None else default_gpu_index()
    try:
        memory = ctx.deps.gpu_reader(index)
        used = f"{memory.used_mib} MiB of {memory.total_mib} MiB in use on GPU {index} (all jobs)"
    except StageError as exc:
        used = f"GPU memory unreadable ({exc})"
    peak = _torch_peak_mib()
    peak_text = f"; this process peak {peak} MiB" if peak is not None else ""
    ctx.say(f"[{name}] done in {format_elapsed(elapsed)}; {used}{peak_text}")
    return result


def _torch_peak_mib() -> int | None:
    """Peak allocation of this process, only when torch already initialised CUDA."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            return int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:  # noqa: BLE001 - a diagnostic must not fail a stage
        return None
    return None


def release_gpu() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def load_training_frame(demo: bool) -> pd.DataFrame:
    """The union of LIAR, FakeNewsNet and CoAID, built as ``modeling.training`` builds it."""
    from modeling.datasets import DatasetUnavailable, get_dataset
    from modeling.text.misinfo_clf import build_training_frame

    loaded: dict[str, pd.DataFrame] = {}
    for key in BENCHMARKS:
        try:
            loaded[key] = get_dataset(key).load(demo=demo).frame
        except DatasetUnavailable as exc:
            raise StageError(
                EXIT_FAILED,
                f"benchmark {key!r} is not available: {exc}. The comparison needs all of "
                f"{', '.join(BENCHMARKS)}. Run `make benchmarks`.",
            ) from exc
    return build_training_frame(loaded)


def frame_fingerprint(work: pd.DataFrame) -> str:
    columns = work[["text", "label", "group_id"]]
    digest = hashlib.sha256(pd.util.hash_pandas_object(columns, index=False).values.tobytes())
    return digest.hexdigest()


def bench_id(source_dataset: str, position: int) -> str:
    return f"{source_dataset}:{position}"


def stage_prepare(ctx: Context) -> None:
    check_run_marker(ctx)
    if ctx.split_file.exists() and not ctx.force:
        ctx.say("[prepare] skipped: split.json exists (use --force to redo)")
        return

    from modeling.config import get_settings
    from modeling.datasets.splits import group_train_val_test
    from modeling.eval.tfidf_artifact import fit_tfidf_on_split
    from modeling.text.laya_clf import LayaTrainingExporter

    seed = get_settings().seed
    frame = load_training_frame(ctx.demo)
    work, split = group_train_val_test(frame, group_col="group_id", label_col="label", seed=seed)
    description = split.describe()
    ctx.say(f"[prepare] split: {description}")
    if description != EXPECTED_SPLIT and not ctx.demo:
        raise StageError(
            EXIT_SPLIT,
            "the split is not the one the roberta baseline used.\n"
            f"  got:      {description}\n"
            f"  required: {EXPECTED_SPLIT}\n"
            "The benchmarks on disk differ from the ones that `misinfo` used, so the "
            "comparison would not be like for like. Nothing was written.",
        )

    exporter = LayaTrainingExporter()
    counts: dict[str, Any] = {}
    for name, idx, path in (
        ("train", split.train, ctx.train_jsonl),
        ("val", split.val, ctx.val_jsonl),
    ):
        written = exporter.export(work.iloc[idx], path, label_column="label")
        counts[name] = {"rows": written, "skipped": dict(exporter.skipped)}
        ctx.say(f"[prepare] {name}.jsonl: {written} rows, skipped: {exporter.skipped or 'none'}")

    _check_no_text_overlap(ctx, work, split)

    fit_tfidf_on_split(work, split, path=ctx.tfidf_file)
    ctx.say(f"[prepare] TF-IDF fitted on the {len(split.train)} training rows: {ctx.tfidf_file}")

    ids = [bench_id(s, i) for i, s in enumerate(work["source_dataset"].tolist())]
    eval_positions = [*split.val.tolist(), *split.test.tolist()]
    eval_split = ["val"] * len(split.val) + ["test"] * len(split.test)
    eval_rows = pd.DataFrame(
        {
            "split": eval_split,
            "id": [ids[i] for i in eval_positions],
            "text": work["text"].iloc[eval_positions].astype(str).to_numpy(),
            "label": work["label"].iloc[eval_positions].astype(int).to_numpy(),
            "source_dataset": work["source_dataset"].iloc[eval_positions].to_numpy(),
            "group_id": work["group_id"].iloc[eval_positions].to_numpy(),
        }
    )
    ctx.out.mkdir(parents=True, exist_ok=True)
    eval_rows.to_parquet(ctx.eval_rows_file, index=False)

    write_json(ctx.run_file, {"demo": ctx.demo, "created_at": datetime.now(timezone.utc)})
    # The marker that the stage finished is written last.
    write_json(
        ctx.split_file,
        {
            "description": description,
            "verified_against": None if ctx.demo else EXPECTED_SPLIT,
            "demo": ctx.demo,
            "seed": seed,
            "group_col": "group_id",
            "n_rows": len(work),
            "sizes": split.sizes,
            "dropped": split.dropped,
            "frame_sha256": frame_fingerprint(work),
            "jsonl": counts,
            "ids": {
                "train": [ids[i] for i in split.train.tolist()],
                "val": [ids[i] for i in split.val.tolist()],
                "test": [ids[i] for i in split.test.tolist()],
            },
        },
    )
    ctx.say(f"[prepare] done: {ctx.out}")


def _check_no_text_overlap(ctx: Context, work: pd.DataFrame, split: Any) -> None:
    """Refuse when a test text is in a training file. The test split is never exported."""
    exported: set[str] = set()
    for path in (ctx.train_jsonl, ctx.val_jsonl):
        with path.open(encoding="utf-8") as handle:
            exported.update(json.loads(line)["state"].strip() for line in handle)
    test_texts = {str(t).strip() for t in work["text"].iloc[split.test]}
    leaked = exported & test_texts
    if leaked:
        raise StageError(
            EXIT_SPLIT,
            f"{len(leaked)} test text(s) are also in train.jsonl or val.jsonl, for example "
            f"{sorted(leaked)[0][:80]!r}. The test split must never reach training.",
        )


# ---------------------------------------------------------------------------
# train-misinfo
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def misinfo_batch_size(batch_size: int | None) -> Iterator[None]:
    """Apply ``--batch-size`` to ``MisinfoClassifier``, which reads only the config.

    ``train_module`` takes no batch size. The override lasts for the one call.
    """
    if batch_size is None:
        yield
        return
    import modeling.text.misinfo_clf as misinfo_module

    cls = misinfo_module.MisinfoClassifier
    original = cls.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        self.batch_size = batch_size

    cls.__init__ = init
    try:
        yield
    finally:
        cls.__init__ = original


def stage_train_misinfo(ctx: Context) -> None:
    split_info = require_prepared(ctx)
    summary_path = ctx.summaries_dir / "misinfo.json"
    if summary_path.exists() and not ctx.force:
        ctx.say("[train-misinfo] skipped: summaries/misinfo.json exists (use --force)")
        return
    epochs = ctx.args.epochs if ctx.args.epochs is not None else (1 if ctx.demo else None)

    def body() -> dict[str, Any]:
        import modeling.training as training

        with scoped_settings(ctx) as settings, misinfo_batch_size(ctx.args.batch_size):
            result = training.train_module(
                "misinfo", demo=ctx.demo, epochs=epochs, settings=settings
            )
        if result.skipped or result.report is None:
            raise StageError(
                EXIT_FAILED, f"misinfo training did not run: {'; '.join(result.notes)}"
            )
        excluded = [note for note in result.notes if "excluded from training" in note]
        if excluded:
            raise StageError(
                EXIT_SPLIT, f"misinfo trained on a smaller union: {'; '.join(excluded)}"
            )
        # train_module builds its own frame and splits it. The split string carries the
        # group column, the group count, the three sizes and the seed.
        reported = result.report.split_description
        if reported != split_info["description"]:
            raise StageError(
                EXIT_SPLIT,
                "misinfo trained on another split than prepare saved.\n"
                f"  train_module: {reported}\n  prepare:      {split_info['description']}",
            )
        report = result.report
        return {
            "module": "misinfo",
            "version": result.version,
            "demo": ctx.demo,
            "epochs": epochs,
            "batch_size": ctx.args.batch_size,
            "split": reported,
            "split_matches_prepare": True,
            "test_split": "test",
            "n_test": report.n_test,
            "macro_f1_test": report.macro_f1.as_dict(),
            "checkpoint_dir": str(ctx.misinfo_dir()),
            "notes": list(result.notes),
        }

    summary = run_gpu_stage(
        ctx, "train-misinfo", DEFAULT_MIN_FREE_TRAIN_MIB, "--batch-size", body
    )
    write_json(summary_path, summary)
    ctx.say(f"[train-misinfo] macro-F1 on test: {summary['macro_f1_test']}")
    ctx.say(f"[train-misinfo] summary: {summary_path}")


# ---------------------------------------------------------------------------
# train-laya
# ---------------------------------------------------------------------------
def resolve_base_checkpoint(which: str, config: dict[str, Any]) -> str:
    """Download the base checkpoint at the pinned revision and return its local directory.

    ``finetune`` resolves an alias such as ``"multilingual"`` to the newest files
    of the Hub repository. The revision in ``configs/models.yaml`` is the one that
    was reviewed, so this call downloads that revision itself and hands a directory
    to ``finetune``.
    """
    from huggingface_hub import snapshot_download

    revision = config.get("revision")
    if not revision:
        raise StageError(EXIT_FAILED, "configs/models.yaml laya.revision is not set")
    if which == "en":
        repo, subfolder = config["checkpoint"], None
    else:
        repo, subfolder = config["multilingual_checkpoint"], config.get("multilingual_subfolder")
    prefix = f"{subfolder}/" if subfolder else ""
    snapshot = snapshot_download(
        repo,
        revision=str(revision),
        allow_patterns=[
            prefix + name
            for name in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
        ],
    )
    return os.path.join(snapshot, subfolder) if subfolder else snapshot


def laya_train_settings(ctx: Context) -> tuple[dict[str, Any], dict[str, Any]]:
    """The ``TrainConfig`` keyword arguments, and a record of what was overridden."""
    from modeling.config import module_config

    cfg = module_config("laya")["train"]
    kwargs = {
        key: cfg[key]
        for key in (
            "epochs",
            "micro_batch",
            "grad_accum",
            "encoder_lr",
            "head_lr",
            "weight_decay",
            "grad_clip",
            "loss",
            "rl_samples",
            "calib_frac",
            "calib_max",
            "calib_seed",
        )
    }
    overrides: dict[str, Any] = {}
    effective = int(cfg["micro_batch"]) * int(cfg["grad_accum"])
    micro = getattr(ctx.args, "micro_batch", None)
    accum = getattr(ctx.args, "grad_accum", None)
    if micro is not None:
        kwargs["micro_batch"] = micro
        # Keep the effective batch of the runbook unless the operator sets both.
        kwargs["grad_accum"] = accum if accum is not None else max(1, math.ceil(effective / micro))
        overrides["micro_batch"] = micro
        overrides["grad_accum"] = kwargs["grad_accum"]
    elif accum is not None:
        kwargs["grad_accum"] = accum
        overrides["grad_accum"] = accum
    epochs = ctx.args.epochs if ctx.args.epochs is not None else (1 if ctx.demo else None)
    if epochs is not None:
        kwargs["epochs"] = epochs
        overrides["epochs"] = epochs
    return kwargs, overrides


def stage_train_laya(ctx: Context, which: str) -> None:
    require_prepared(ctx)
    name = f"train-laya-{which}"
    summary_path = ctx.summaries_dir / f"laya_{which}.json"
    if summary_path.exists() and not ctx.force:
        ctx.say(f"[{name}] skipped: {summary_path.name} exists (use --force)")
        return
    for path in (ctx.train_jsonl, ctx.val_jsonl):
        if not path.exists():
            raise StageError(EXIT_FAILED, f"{path} is missing. Run the prepare stage first.")

    from modeling.config import module_config

    config = module_config("laya")
    output_dir = ctx.laya_checkpoint_dir(which)

    def body() -> dict[str, Any]:
        from laya.train import TrainConfig, finetune

        model_dir = resolve_base_checkpoint(which, config)
        kwargs, overrides = laya_train_settings(ctx)
        ctx.say(f"[{name}] base checkpoint {model_dir}")
        ctx.say(f"[{name}] train settings {kwargs}")
        summary = finetune(
            str(ctx.train_jsonl),
            model_dir,
            str(output_dir),
            TrainConfig(eval_data=str(ctx.val_jsonl), **kwargs),
            getattr(ctx.args, "device", None) or config.get("device", "auto"),
        )
        report = (summary or {}).get("train_report") or {}
        warnings = []
        if report.get("eval_mode") != "held_out":
            warnings.append(
                f"validation is not strictly held out (eval_mode={report.get('eval_mode')!r}); "
                "do not report it as validation"
            )
            ctx.say(f"[{name}] WARNING: {warnings[-1]}")
        return {
            "model": f"laya_{which}",
            "demo": ctx.demo,
            "base_checkpoint": model_dir,
            "revision": config.get("revision"),
            "output_dir": str(output_dir),
            "train_jsonl": str(ctx.train_jsonl),
            "eval_data": str(ctx.val_jsonl),
            "train_config": kwargs,
            "overrides": overrides,
            "warnings": warnings,
            "finetune": summary,
        }

    summary = run_gpu_stage(ctx, name, DEFAULT_MIN_FREE_TRAIN_MIB, "--micro-batch", body)
    write_json(output_dir / "run_summary.json", summary["finetune"])
    write_json(summary_path, summary)
    ctx.say(f"[{name}] summary: {summary_path}")


# ---------------------------------------------------------------------------
# score: inputs
# ---------------------------------------------------------------------------
def demo_corpus_extras(publisher: str) -> list[dict[str, Any]]:
    """Corpus rows that the committed fixture lacks: Hindi, titles, and a missing title."""
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        ("demo:c1", "outlet-a.example", "en",
         "The minister resigned today, a viral post says.", "Minister resigns"),
        ("demo:c2", "outlet-a.example", "en",
         "A viral video shows a flood that never happened.", "Old flood video"),
        ("demo:c3", "outlet-b.example", "hi",
         "वायरल पोस्ट में दावा किया गया है कि मंत्री ने आज इस्तीफा दे दिया।", "मंत्री का इस्तीफा"),
        ("demo:c4", "outlet-b.example", "hi",
         "एक पुराना वीडियो बाढ़ की ताज़ा तस्वीर बताकर फैलाया जा रहा है।", None),
        ("demo:c5", "outlet-c.example", "en",
         "Council publishes the quarterly budget schedule for review.", None),
        ("demo:c6", publisher, "en",
         "Fact check: the viral claim about the minister is false.", "Fact check"),
    ]
    return [
        {
            "id": rid,
            "source_detail": detail,
            "text": text,
            "lang": lang,
            "raw": json.dumps({"title": title} if title else {}),
            "ingested_at": stamp,
        }
        for rid, detail, lang, text, title in rows
    ]


def demo_claims() -> list[Any]:
    from ingest.factcheck import FactCheck
    from ingest.normalize import simhash

    claims = [
        ("demo:fc1", "en", "A video shows the prime minister resigning today."),
        ("demo:fc2", "en", "Old flood footage is shared as a new disaster in the capital."),
        ("demo:fc3", "en", "A bank will freeze every account that has no linked phone number."),
        ("demo:fc4", "en", "A photograph shows a crowd at a rally that took place last year."),
        ("demo:fc5", "hi", "प्रधानमंत्री के इस्तीफे का वीडियो आज का बताया जा रहा है।"),
        ("demo:fc6", "hi", "पुरानी बाढ़ का वीडियो राजधानी की नई तबाही बताकर फैलाया गया।"),
    ]
    return [
        FactCheck(
            record_id=rid,
            url=f"https://factcheck.example/{rid}",
            domain="factcheck.example",
            publisher="Demo",
            tier="claimreview",
            claim=claim,
            claim_lang=lang,
            rating_raw="False",
            label="false",
            claim_simhash=simhash(claim),
            reviewed_at=None,
        )
        for rid, lang, claim in claims
    ]


def corpus_root(ctx: Context) -> Path:
    from modeling.scoring import DEMO_CORPUS

    if ctx.demo:
        return DEMO_CORPUS / "normalized"
    from modeling.config import get_settings

    return get_settings().normalized_dir


def read_corpus(ctx: Context) -> tuple[pd.DataFrame, dict[str, int]]:
    """Corpus rows with ``lang`` in en or hi: id, text, title, lang, source_detail.

    Reads the Parquet store through ``ingest.store.ParquetStore``. The title is in the
    ``raw`` JSON blob and is null when the record has none.
    """
    from ingest.config import Settings
    from ingest.store import ParquetStore

    root = corpus_root(ctx)
    data_dir = ctx.demo_data_dir if ctx.demo else None
    settings = Settings(data_dir=data_dir, _env_file=None) if data_dir else Settings(_env_file=None)
    try:
        rows = list(ParquetStore(settings, root=root).iter_records())
    except Exception as exc:  # noqa: BLE001
        # Some partitions store `source` as a dictionary column and others as a string,
        # and a dataset read refuses to merge them. The corpus reader in modeling reads
        # file by file, so use it and say so.
        log.warning("ParquetStore read of %s failed (%s); reading file by file instead", root, exc)
        from modeling.io import CorpusReader

        frame = CorpusReader(root=root).records(
            columns=["id", "source_detail", "text", "lang", "raw", "ingested_at"]
        )
        rows = frame.to_dict("records")
    if ctx.demo:
        from modeling.eval.factcheck_eval import load_factcheck_publishers

        rows = rows + demo_corpus_extras(sorted(load_factcheck_publishers())[0])

    counts = {"rows_read": len(rows), "other_language": 0, "duplicate_id": 0}
    newest: dict[str, tuple[Any, dict[str, Any]]] = {}
    for row in rows:
        lang = str(row.get("lang") or "").strip().lower()
        if lang not in CORPUS_LANGUAGES:
            counts["other_language"] += 1
            continue
        record_id = str(row["id"])
        stamp = row.get("ingested_at")
        held = newest.get(record_id)
        if held is not None:
            counts["duplicate_id"] += 1
            if not (stamp is not None and (held[0] is None or stamp > held[0])):
                continue
        newest[record_id] = (stamp, {**row, "lang": lang})

    records = []
    for record_id, (_, row) in newest.items():
        records.append(
            {
                "id": record_id,
                "text": row.get("text"),
                "title": _title_of(row.get("raw")),
                "lang": row["lang"],
                "source_detail": row.get("source_detail"),
            }
        )
    frame = pd.DataFrame(records, columns=["id", "text", "title", "lang", "source_detail"])
    counts["kept"] = len(frame)
    return frame, counts


def _title_of(raw: Any) -> str | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    title = raw.get("title") if isinstance(raw, dict) else None
    title = str(title).strip() if title is not None else ""
    return title or None


def read_claims(ctx: Context) -> pd.DataFrame:
    """Claims the harness counts: the kept rows of ``read_deduplicated_factchecks``."""
    from ingest.config import Settings
    from ingest.factcheck import read_deduplicated_factchecks, write_factchecks
    from modeling.config import get_settings

    data_dir = ctx.demo_data_dir if ctx.demo else get_settings().data_dir
    settings = Settings(data_dir=data_dir, _env_file=None)
    if ctx.demo and not (data_dir / "factchecks" / "factchecks.parquet").exists():
        write_factchecks(demo_claims(), settings)
    result = read_deduplicated_factchecks(settings)
    if not result.kept:
        raise StageError(
            EXIT_FAILED,
            f"no fact-check claims under {data_dir / 'factchecks'}. "
            "Run `python -m ingest.cli factchecks` first.",
        )
    log.info("claims: %d kept, %d collapsed", len(result.kept), len(result.collapsed))
    return pd.DataFrame(
        {
            "id": [str(r["record_id"]) for r in result.kept],
            "text": [r.get("claim") for r in result.kept],
            "lang": [_lang_or_none(r.get("claim_lang")) for r in result.kept],
            "label": None,
            "source_detail": [r.get("domain") for r in result.kept],
        }
    )


def _lang_or_none(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip().lower()
    return text or None


def build_parts(ctx: Context) -> dict[str, pd.DataFrame]:
    """The five input tables. Each has id, text, lang, label, source_detail."""
    eval_rows = pd.read_parquet(ctx.eval_rows_file)
    parts: dict[str, pd.DataFrame] = {}
    for name in ("val", "test"):
        chosen = eval_rows[eval_rows["split"] == name]
        parts[name] = pd.DataFrame(
            {
                "id": chosen["id"].astype(str).to_numpy(),
                "text": chosen["text"].to_numpy(),
                "lang": "en",
                "label": chosen["label"].astype(int).to_numpy(),
                "source_detail": chosen["source_dataset"].to_numpy(),
            }
        )
    parts["claims"] = read_claims(ctx)
    corpus, counts = read_corpus(ctx)
    log.info("corpus: %s", counts)
    for part, column in (("corpus_title", "title"), ("corpus_text", "text")):
        parts[part] = pd.DataFrame(
            {
                "id": corpus["id"].to_numpy(),
                "text": corpus[column].to_numpy(),
                "lang": corpus["lang"].to_numpy(),
                "label": None,
                "source_detail": corpus["source_detail"].to_numpy(),
            }
        )
    return parts


# ---------------------------------------------------------------------------
# score: models
# ---------------------------------------------------------------------------
Scorer = Callable[[pd.DataFrame], Any]  # frame of id, text, lang -> ScoringOutcome


def laya_model_config(name: str, ctx: Context) -> dict[str, Any]:
    """The ``laya`` config for one of the three Laya models. The scorer code is unchanged."""
    from modeling.config import module_config

    cfg = copy.deepcopy(module_config("laya"))
    cfg["finetuned_checkpoints"] = {"en": None, "hi": None}
    multilingual = {"en": "multilingual_checkpoint", "hi": "multilingual_checkpoint"}
    if name == "laya_en":
        cfg["route"] = {"en": "checkpoint"}
        cfg["finetuned_checkpoints"]["en"] = str(ctx.laya_checkpoint_dir("en"))
    elif name == "laya_multi":
        cfg["route"] = multilingual
        directory = str(ctx.laya_checkpoint_dir("multilingual"))
        cfg["finetuned_checkpoints"] = {"en": directory, "hi": directory}
    elif name == "laya_multi_base":
        if not cfg.get("revision"):
            raise StageError(EXIT_FAILED, "configs/models.yaml laya.revision is not set")
        cfg["route"] = multilingual
    else:  # pragma: no cover - guarded by the caller
        raise KeyError(name)
    batch = getattr(ctx.args, "batch_size", None)
    if batch is not None:
        cfg["batch_size"] = batch
    return cfg


def make_laya_scorer(name: str, ctx: Context) -> tuple[Scorer, str]:
    from modeling.config import get_settings
    from modeling.text.laya_clf import LayaScorer

    # An unknown language is a skip, never an assumed English, as in the TF-IDF scorer.
    settings = get_settings().model_copy(update={"score_unknown_language": False})
    scorer = LayaScorer(settings=settings, _config=laya_model_config(name, ctx))
    availability = scorer.available()
    if not availability.ok:
        raise StageError(EXIT_FAILED, f"[{name}] {availability.reason}")
    version = scorer.model_version_for("hi" if name != "laya_en" else "en")
    return (lambda frame: scorer.score(frame)), version


def make_tfidf_scorer(ctx: Context) -> tuple[Scorer, str]:
    from modeling.eval.tfidf_artifact import load_tfidf, score_tfidf

    model = load_tfidf(ctx.tfidf_file)

    def score(frame: pd.DataFrame) -> Any:
        ids, texts, langs = (frame[c].tolist() for c in ("id", "text", "lang"))
        return score_tfidf(model, ids, texts, langs)

    return score, f"tfidf-logreg+sklearn-{model.metadata.get('sklearn_version')}"


MISINFO_CHUNK = 1024


def make_misinfo_scorer(ctx: Context) -> tuple[Scorer, str]:
    from modeling.aux.base import ScoringOutcome
    from modeling.config import get_settings, module_config
    from modeling.text.misinfo_clf import MisinfoClassifier

    directory = ctx.misinfo_dir()
    classifier = MisinfoClassifier(get_settings())
    if getattr(ctx.args, "batch_size", None) is not None:
        classifier.batch_size = ctx.args.batch_size
    if not classifier.load(directory):
        raise StageError(
            EXIT_FAILED, f"misinfo checkpoint at {directory} did not load (see the log above)"
        )

    def score(frame: pd.DataFrame) -> Any:
        outcome = ScoringOutcome()
        todo: list[tuple[str, str]] = []
        for record_id, text, lang in zip(frame["id"], frame["text"], frame["lang"], strict=True):
            if lang is None or (isinstance(lang, float) and lang != lang) or not str(lang).strip():
                outcome.skipped[record_id] = UNKNOWN_LANGUAGE
            elif str(lang).strip().lower() != "en":
                outcome.skipped[record_id] = UNSUPPORTED_LANGUAGE
            elif not str(text or "").strip():
                outcome.skipped[record_id] = EMPTY_TEXT
            else:
                todo.append((record_id, str(text)))
        for start in range(0, len(todo), MISINFO_CHUNK):
            chunk = todo[start : start + MISINFO_CHUNK]
            try:
                values = classifier.predict([text for _, text in chunk])
            except Exception as exc:  # noqa: BLE001
                if is_oom(exc):
                    raise
                log.warning("misinfo inference failed for %d records: %s", len(chunk), exc)
                for record_id, _ in chunk:
                    outcome.skipped[record_id] = INFERENCE_ERROR
                continue
            for (record_id, _), value in zip(chunk, values, strict=True):
                outcome.values[record_id] = float(value)
        return outcome

    return score, str(module_config("misinfo").get("version", "v0.0.0-unset"))


def make_scorer(name: str, ctx: Context) -> tuple[Scorer, str]:
    if name == "tfidf":
        return make_tfidf_scorer(ctx)
    if name == "misinfo":
        return make_misinfo_scorer(ctx)
    return make_laya_scorer(name, ctx)


def model_versions(ctx: Context) -> dict[str, str]:
    """Versions for the report: configs/models.yaml, and the checkpoint directory names."""
    from modeling.config import get_settings, module_config
    from modeling.text.laya_clf import LayaScorer

    versions: dict[str, str] = {"misinfo": str(module_config("misinfo").get("version", "v0.0.0"))}
    for name in ("laya_en", "laya_multi", "laya_multi_base"):
        scorer = LayaScorer(settings=get_settings(), _config=laya_model_config(name, ctx))
        versions[name] = scorer.model_version_for("en" if name == "laya_en" else "hi")
    tfidf_meta = _tfidf_metadata(ctx)
    versions["tfidf"] = f"tfidf-logreg+sklearn-{tfidf_meta.get('sklearn_version')}"
    return versions


def _tfidf_metadata(ctx: Context) -> dict[str, Any]:
    if not ctx.tfidf_file.exists():
        return {}
    from modeling.eval.tfidf_artifact import load_tfidf

    return dict(load_tfidf(ctx.tfidf_file).metadata)


# ---------------------------------------------------------------------------
# score: one part, one model
# ---------------------------------------------------------------------------
def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value) or not str(value).strip()


def score_part(scorer: Scorer, part: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Score one part. Every input row comes back with a score or a ``skip_reason``."""
    from modeling.aux.base import ScoringOutcome

    ids = [str(i) for i in frame["id"]]
    if len(set(ids)) != len(ids):
        raise StageError(EXIT_FAILED, f"part {part!r} has duplicate ids; refusing to hide rows")
    blank_reason = NO_TITLE if part == "corpus_title" else EMPTY_TEXT

    pre: dict[str, str] = {}
    keep: list[int] = []
    for position, (record_id, text) in enumerate(zip(ids, frame["text"], strict=True)):
        if _blank(text):
            pre[record_id] = blank_reason
        else:
            keep.append(position)
    # object dtype keeps a missing language as None. pandas 3 would turn it into NaN, and
    # LayaScorer would then read "nan" as a language code instead of an unknown language.
    sub = pd.DataFrame(
        {
            "id": [ids[p] for p in keep],
            "text": [str(frame["text"].iloc[p]) for p in keep],
            "lang": pd.Series([_lang_or_none(frame["lang"].iloc[p]) for p in keep], dtype=object),
        }
    )
    outcome = scorer(sub) if len(sub) else ScoringOutcome()

    rows: list[dict[str, Any]] = []
    for position, record_id in enumerate(ids):
        score: float | None = None
        reason: str | None = pre.get(record_id)
        if reason is None:
            if record_id in outcome.values:
                value = float(outcome.values[record_id])
                if math.isfinite(value) and 0.0 <= value <= 1.0:
                    score = value
                else:
                    reason = INVALID_SCORE
                    log.warning("model gave %r for %s; recorded as %s", value, record_id, reason)
            else:
                reason = outcome.skipped.get(record_id)
                if reason is None:
                    reason = SCORER_DROPPED_RECORD
                    log.warning("scorer returned nothing for %s in part %s", record_id, part)
        label = frame["label"].iloc[position]
        rows.append(
            {
                "part": part,
                "id": record_id,
                "lang": _lang_or_none(frame["lang"].iloc[position]),
                "label": None if label is None or label != label else int(label),
                "score": score,
                "source_detail": None
                if _blank(frame["source_detail"].iloc[position])
                else str(frame["source_detail"].iloc[position]),
                "skip_reason": reason,
            }
        )
    return rows


def write_score_file(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``<model>.parquet`` with the column types that the harness requires."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("part", pa.string()),
            ("id", pa.string()),
            ("lang", pa.string()),
            ("label", pa.int64()),
            ("score", pa.float64()),
            ("source_detail", pa.string()),
            ("skip_reason", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def tally(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = out.setdefault(row["part"], {"rows": 0, "scored": 0, "skipped": {}})
        entry["rows"] += 1
        if row["score"] is None:
            entry["skipped"][row["skip_reason"]] = entry["skipped"].get(row["skip_reason"], 0) + 1
        else:
            entry["scored"] += 1
    return out


def checkpoints_missing(ctx: Context, models: list[str]) -> list[str]:
    missing = []
    if "tfidf" in models and not ctx.tfidf_file.exists():
        missing.append(f"tfidf: {ctx.tfidf_file}")
    if "misinfo" in models and not (ctx.misinfo_dir() / "calibrator.json").exists():
        missing.append(f"misinfo: {ctx.misinfo_dir()} (no calibrator.json)")
    if "laya_en" in models and not (ctx.laya_checkpoint_dir("en") / "run_summary.json").exists():
        missing.append(f"laya_en: {ctx.laya_checkpoint_dir('en')}")
    if "laya_multi" in models and not (
        ctx.laya_checkpoint_dir("multilingual") / "run_summary.json"
    ).exists():
        missing.append(f"laya_multi: {ctx.laya_checkpoint_dir('multilingual')}")
    return missing


def score_file(ctx: Context, model: str) -> Path:
    return ctx.scores_dir / f"{model}.parquet"


def selected_models(ctx: Context) -> list[str]:
    value = getattr(ctx.args, "models", None)
    if not value:
        return list(MODELS)
    chosen = [m.strip() for m in value.split(",") if m.strip()]
    unknown = [m for m in chosen if m not in MODELS]
    if unknown:
        raise StageError(EXIT_USAGE, f"unknown model(s) {unknown}; choose from {list(MODELS)}")
    return chosen


def stage_score(ctx: Context) -> None:
    require_prepared(ctx)
    chosen = selected_models(ctx)
    todo = [m for m in chosen if ctx.force or not score_file(ctx, m).exists()]
    if not todo:
        ctx.say("[score] skipped: every selected model has a score file (use --force)")
        return
    missing = checkpoints_missing(ctx, todo)
    if missing:
        raise StageError(
            EXIT_FAILED, "[score] checkpoints are missing, so nothing was scored:\n  "
            + "\n  ".join(missing),
        )

    def body() -> dict[str, Any]:
        parts = build_parts(ctx)
        for part, frame in parts.items():
            ctx.say(f"[score] input {part}: {len(frame)} rows")
        counts: dict[str, Any] = {}
        for model in todo:
            counts[model] = score_model(ctx, model, parts)
            release_gpu()
        return counts

    counts = run_gpu_stage(ctx, "score", DEFAULT_MIN_FREE_SCORE_MIB, "--batch-size", body)
    summary_path = ctx.summaries_dir / "score.json"
    merged = read_json(summary_path) if summary_path.exists() else {}
    merged.update(counts)
    write_json(summary_path, merged)


def score_model(ctx: Context, model: str, parts: dict[str, pd.DataFrame]) -> dict[str, Any]:
    started = ctx.deps.clock()
    with watch_laya_oom():
        scorer, version = make_scorer(model, ctx)
        rows: list[dict[str, Any]] = []
        for part in SCORE_PARTS:
            rows.extend(score_part(scorer, part, parts[part]))
    counts = tally(rows)
    for part in ("val", "test"):
        if counts.get(part, {}).get("scored", 0) == 0:
            raise StageError(
                EXIT_FAILED,
                f"[score] {model} scored no {part} rows ({counts.get(part)}). "
                "Not writing a score file with no scores.",
            )
    write_score_file(score_file(ctx, model), rows)
    ctx.say(f"[score] {model} ({version}): {score_file(ctx, model)}")
    for part, entry in counts.items():
        skipped = entry["skipped"] or "none"
        ctx.say(f"[score]   {part}: {entry['scored']}/{entry['rows']} scored, skipped: {skipped}")
    return {
        "version": version,
        "seconds": round(ctx.deps.clock() - started, 1),
        "parts": counts,
    }


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def stage_evaluate(ctx: Context) -> None:
    split_info = require_prepared(ctx)
    models = selected_models(ctx) if getattr(ctx.args, "models", None) else list(MODELS)
    missing = [m for m in models if not score_file(ctx, m).exists()]
    if missing:
        raise StageError(
            EXIT_FAILED,
            f"[evaluate] no score file for {missing} in {ctx.scores_dir}. Run the score stage.",
        )
    from modeling.eval.factcheck_eval import EvalInputError, gate_text, run_comparison

    description = split_info["description"]
    split_text = f"[DEMO] {description}" if ctx.demo else description
    with scoped_settings(ctx) as settings:
        versions = model_versions(ctx)
        manifest = ctx.out / "no_manifest.csv" if ctx.demo else None
        try:
            result, paths = run_comparison(
                ctx.scores_dir,
                data_dir=ctx.demo_data_dir if ctx.demo else None,
                manifest_path=manifest,
                versions=versions,
                split_description=split_text,
                label="comparison",
                settings=settings,
            )
        except EvalInputError as exc:
            raise StageError(EXIT_FAILED, f"[evaluate] the harness refused: {exc}") from exc
    if not result["evaluation_set"]["frozen"]:
        ctx.say("[evaluate] evaluation set not frozen (see the report)")
    for name, comparison in result["comparisons"].items():
        verdict = "ship" if comparison["ship"] else "do not ship"
        ctx.say(f"[evaluate] {name}: {verdict} - {comparison['reason']}")
    for name, gate in result["hindi_gate"].items():
        ctx.say(f"[evaluate] {name}: Hindi gate {gate_text(gate)}")
    for path in paths:
        ctx.say(f"[evaluate] report: {path}")


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def stage_plan() -> list[tuple[str, Callable[[Context], None]]]:
    """The stages of ``all``, in order. Names are looked up at call time."""
    return [
        ("prepare", stage_prepare),
        ("train-misinfo", stage_train_misinfo),
        ("train-laya-en", lambda ctx: stage_train_laya(ctx, "en")),
        ("train-laya-multilingual", lambda ctx: stage_train_laya(ctx, "multilingual")),
        ("score", stage_score),
        ("evaluate", stage_evaluate),
    ]


def run_all(ctx: Context) -> None:
    for name, stage in stage_plan():
        ctx.say(f"=== {name} ===")
        stage(ctx)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output directory, relative to the repository root. Default {DEFAULT_OUT}, "
        f"or {DEFAULT_DEMO_OUT} with --demo.",
    )
    parser.add_argument("--demo", action="store_true", help="Run on the committed fixtures.")
    parser.add_argument("--force", action="store_true", help="Redo a stage that has output.")
    parser.add_argument("-v", "--verbose", action="store_true")


def _gpu(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=None,
        help=f"Free GPU memory required. Default {DEFAULT_MIN_FREE_TRAIN_MIB} to train, "
        f"{DEFAULT_MIN_FREE_SCORE_MIB} to score.",
    )
    parser.add_argument(
        "--wait-for-gpu",
        action="store_true",
        help=f"Poll every {GPU_POLL_SECONDS // 60} minutes until the memory is free.",
    )
    parser.add_argument("--wait-hours", type=float, default=DEFAULT_WAIT_HOURS)
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="GPU to read. Default: first CUDA_VISIBLE_DEVICES entry, else 0.",
    )


def _misinfo_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=None, help="Demo default is 1.")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size. Lower it on a shared GPU."
    )


def _laya_opts(parser: argparse.ArgumentParser, *, epochs: bool = True) -> None:
    if epochs:
        parser.add_argument("--epochs", type=int, default=None, help="Demo default is 1.")
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=None,
        help="Laya micro batch. --grad-accum rises to keep the effective batch.",
    )
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--device", default=None, help="Laya device. Default from the config.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_comparison.py",
        description="Train and score the models for the three-way comparison.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Build the frame, split, JSONL files and TF-IDF.")
    _common(p)

    p = sub.add_parser("train-misinfo", help="Fine-tune the roberta classifier.")
    _common(p)
    _gpu(p)
    _misinfo_opts(p)

    p = sub.add_parser("train-laya", help="Fine-tune a Laya checkpoint.")
    _common(p)
    _gpu(p)
    _laya_opts(p)
    p.add_argument("--which", choices=LAYA_WHICH, required=True)

    p = sub.add_parser("score", help="Write one score file per model.")
    _common(p)
    _gpu(p)
    p.add_argument("--batch-size", type=int, default=None, help="Lower it on a shared GPU.")
    p.add_argument("--models", default=None, help=f"Comma list. Default all: {','.join(MODELS)}.")

    p = sub.add_parser("evaluate", help="Run the harness on the score files.")
    _common(p)
    p.add_argument("--models", default=None)

    p = sub.add_parser("all", help="Run every stage in order. Stops at the first failure.")
    _common(p)
    _gpu(p)
    _misinfo_opts(p)
    _laya_opts(p, epochs=False)
    p.add_argument("--models", default=None)
    return parser


def main(argv: list[str] | None = None, *, deps: Deps | None = None) -> int:
    args = build_parser().parse_args(argv)
    ctx = Context(args=args, deps=deps or Deps())
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "filelock", "huggingface_hub", "transformers", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if ctx.demo:
        ctx.say("DEMO RUN on committed fixtures: nothing here is a result.")

    try:
        if args.command == "prepare":
            stage_prepare(ctx)
        elif args.command == "train-misinfo":
            stage_train_misinfo(ctx)
        elif args.command == "train-laya":
            stage_train_laya(ctx, args.which)
        elif args.command == "score":
            stage_score(ctx)
        elif args.command == "evaluate":
            stage_evaluate(ctx)
        else:
            run_all(ctx)
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code
    except Exception as exc:  # noqa: BLE001 - last resort: report and exit non-zero
        traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_GPU_OOM if is_oom(exc) else EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
