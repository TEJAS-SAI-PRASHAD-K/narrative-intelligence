"""Find out where the Laya temperature optimum lies, on the validation rows of one fine-tune.

The fine-tune does not save its validation logits. This script computes them once from the
checkpoint with ``laya.calibrate.records_from_labeled``, which returns the raw logit rows that
the decoder divides by its temperature. It stores the logit gap in ``<out>.logits.npz`` and
writes the diagnosis to ``--out``.

Run it once for each fine-tune::

    python scripts/diagnose_laya_calibration.py --model laya_en    --out reports/laya_en.json
    python scripts/diagnose_laya_calibration.py --model laya_multi --out reports/laya_multi.json

Add ``--wait-for-gpu`` on a shared host. The script reads free GPU memory first, as the stages
of ``run_comparison.py`` do, and never stops another process. Add ``--reuse-logits`` to compute
the diagnosis again from the stored gap with no GPU.

Only validation rows are read. The test rows are never loaded.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_comparison as rc  # noqa: E402

from modeling.text import laya_temperature as lt  # noqa: E402

log = logging.getLogger("diagnose_laya_calibration")

MODELS = ("laya_en", "laya_multi")
#: The ``noul`` head has two options, so its temperature bucket is ``noul:2``.
BUCKET = "noul:2"
#: A stored score and a gap rebuilt from logits agree when they differ by at most this.
SCORE_TOLERANCE = 1.5e-4


def checkpoint_temperature(agent: Any) -> dict[str, Any]:
    """The temperature that the checkpoint applies to a ``noul``, and where it comes from."""
    by_options = dict(getattr(agent, "temperature_by_options", {}) or {})
    per_type = list(getattr(agent, "temperature", [1.0, 1.0, 1.0]))
    applied = float(by_options.get(BUCKET, per_type[2]))
    return {
        "applied": applied,
        "source": BUCKET if BUCKET in by_options else "per_type_noul",
        "per_type": [float(t) for t in per_type],
        "per_type_raw": [float(t) for t in getattr(agent, "temperature_raw", per_type)],
        "by_options": {k: float(v) for k, v in by_options.items()},
        "binning_map_installed": bool(getattr(agent, "binning_map", None)),
    }


def collect_gaps(ctx: rc.Context, model: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Logit gap per validation row, from the checkpoint. This is the GPU part."""
    from laya.calibrate import records_from_labeled

    from modeling.config import get_settings
    from modeling.text.laya_clf import LayaScorer

    settings = get_settings().model_copy(update={"score_unknown_language": False})
    scorer = LayaScorer(settings=settings, _config=rc.laya_model_config(model, ctx))
    availability = scorer.available()
    if not availability.ok:
        raise rc.StageError(rc.EXIT_FAILED, f"[{model}] {availability.reason}")
    agent = scorer.agent_for("en")
    questions = scorer.question()
    qid = scorer.question_id

    pairs, kept = [], []
    for position, (text, label) in enumerate(zip(frame["text"], frame["label"], strict=True)):
        if rc._blank(text):
            continue
        y = int(label)
        pairs.append((str(text), questions, {qid: [1.0 - y, float(y)]}))
        kept.append(position)
    records = records_from_labeled(agent, pairs)
    if len(records) != len(pairs):
        raise rc.StageError(
            rc.EXIT_FAILED,
            f"[{model}] {len(pairs)} states gave {len(records)} records. Refusing to guess "
            "which rows were dropped.",
        )
    gaps = np.empty(len(records))
    for i, (_qtype, logits, _target, k) in enumerate(records):
        if k != 2:
            raise rc.StageError(rc.EXIT_FAILED, f"[{model}] row {i} has {k} options, not 2")
        gaps[i] = float(logits[1] - logits[0])
    return {
        "gap": gaps,
        "positions": np.asarray(kept),
        "temperature": checkpoint_temperature(agent),
        "encode_max_len": int(agent.cfg.get("max_len", 0) or 0),
    }


def vendor_fit_on_validation(gap: np.ndarray, labels: np.ndarray) -> dict[str, Any] | None:
    """Run the vendor fitter itself on all validation rows, with its clamp and its report."""
    try:
        from laya.calibrate import fit_temperature_map
    except ImportError:
        return None
    records = [
        (2, np.array([0.0, g], dtype=np.float32), np.array([1.0 - y, float(y)]), 2)
        for g, y in zip(gap, labels, strict=True)
    ]
    result = fit_temperature_map(records)
    return {
        "temperature_noul": float(result["temperature"][2]),
        "temperature_clamped": result.get("temperature_clamped"),
        "rows": len(records),
    }


def read_run_summary(model: str, ctx: rc.Context) -> dict[str, Any] | None:
    which = "en" if model == "laya_en" else "multilingual"
    path = ctx.laya_checkpoint_dir(which) / "run_summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    report = summary.get("train_report") or {}
    return {
        "calibration": summary.get("calibration"),
        "eval_mode": report.get("eval_mode"),
        "before": report.get("before"),
        "after": report.get("after"),
    }


def compare_with_score_file(
    ctx: rc.Context, model: str, ids: np.ndarray, gap: np.ndarray, temperature: float
) -> dict[str, Any] | None:
    """Do the rebuilt scores match the stored validation scores of the score stage?"""
    path = rc.score_file(ctx, model)
    if not path.is_file():
        return None
    stored = pd.read_parquet(
        path, columns=["part", "id", "score"], filters=[("part", "==", "val")]
    ).set_index("id")["score"]
    rebuilt = pd.Series(lt._round_served(gap, temperature), index=ids)
    common = rebuilt.index.intersection(stored.dropna().index)
    if not len(common):
        return {"compared": 0}
    diff = (rebuilt.loc[common] - stored.loc[common]).abs()
    return {
        "compared": int(len(common)),
        "max_abs_diff": float(diff.max()),
        "share_above_tolerance": float((diff > SCORE_TOLERANCE).mean()),
        "note": "A difference comes from long states, which the scorer reads with predict_long.",
    }


def run(args: argparse.Namespace) -> int:
    ns = argparse.Namespace(
        out=args.comparison_dir,
        demo=args.demo,
        wait_for_gpu=args.wait_for_gpu,
        wait_hours=args.wait_hours,
        min_free_mib=args.min_free_mib,
        gpu_index=args.gpu_index,
        batch_size=None,
    )
    ctx = rc.Context(args=ns)
    split_info = rc.require_prepared(ctx)
    rows = pd.read_parquet(ctx.eval_rows_file)
    val = rows[rows["split"] == "val"].reset_index(drop=True)
    if sorted(val["id"].astype(str)) != sorted(split_info["ids"]["val"]):
        raise rc.StageError(rc.EXIT_SPLIT, "eval_rows.parquet and split.json differ on val ids")

    cache = Path(args.logits) if args.logits else args.out.with_suffix(".logits.npz")
    if args.reuse_logits:
        if not cache.is_file():
            raise rc.StageError(rc.EXIT_FAILED, f"--reuse-logits: {cache} does not exist")
        stored = np.load(cache, allow_pickle=False)
        gap = stored["gap"]
        positions = stored["positions"]
        temperature = json.loads(str(stored["temperature"]))
        encode_max_len = int(stored["encode_max_len"])
        if int(stored["n_val"]) != len(val):
            raise rc.StageError(rc.EXIT_FAILED, f"{cache} was made for another validation set")
    else:
        collected = rc.run_gpu_stage(
            ctx,
            f"diagnose-{args.model}",
            rc.DEFAULT_MIN_FREE_SCORE_MIB,
            "--min-free-mib",
            lambda: collect_gaps(ctx, args.model, val),
        )
        gap, positions = collected["gap"], collected["positions"]
        temperature, encode_max_len = collected["temperature"], collected["encode_max_len"]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache,
            gap=gap,
            positions=positions,
            temperature=json.dumps(temperature),
            encode_max_len=encode_max_len,
            n_val=len(val),
        )
        ctx.say(f"[diagnose] stored the logit gaps: {cache}")

    kept = val.iloc[positions]
    labels = kept["label"].astype(int).to_numpy()
    groups = kept["group_id"].astype(str).to_numpy()
    sources = kept["source_dataset"].astype(str).to_numpy()
    t_applied = float(temperature["applied"])

    report = lt.diagnose(
        gap, labels, groups, checkpoint_temperature=t_applied, sources=sources, n_boot=args.n_boot
    )
    report.update(
        {
            "model": args.model,
            "validation_rows_total": int(len(val)),
            "validation_rows_used": int(len(kept)),
            "dropped_blank_text": int(len(val) - len(kept)),
            "checkpoint": temperature,
            "encode_max_len": encode_max_len,
            "vendor_fit_on_validation": vendor_fit_on_validation(gap, labels),
            "run_summary": read_run_summary(args.model, ctx),
            "score_file_check": compare_with_score_file(
                ctx, args.model, kept["id"].astype(str).to_numpy(), gap, t_applied
            ),
            "split": split_info["description"],
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")

    fit = report["temperature_fit"]
    ctx.say(f"[diagnose] {args.model}: {report['rows']} validation rows")
    ctx.say(
        f"[diagnose] checkpoint T {t_applied:.4g}; wide-range optimum {fit['value']:.4g}"
        f" (at bound: {fit['at_bound']}); NLL at 1: {fit['nll_at_one']:.4f},"
        f" at optimum {fit['nll']:.4f}"
    )
    for name, block in report["ece"].items():
        e = block["ece_prob"]
        ctx.say(
            f"[diagnose] ECE {name}: {e['value']:.4f} [{e['low']:.4f}, {e['high']:.4f}]"
        )
    for note in report["findings"]:
        ctx.say(f"[diagnose] finding: {note}")
    ctx.say(f"[diagnose] wrote {args.out}")
    return rc.EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--out", type=Path, required=True, help="The JSON report to write.")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=rc.DEFAULT_OUT,
        help="The run directory of run_comparison.py. Default %(default)s.",
    )
    parser.add_argument(
        "--logits", type=Path, default=None, help="The logit file. Default: <out>.logits.npz."
    )
    parser.add_argument("--reuse-logits", action="store_true", help="Read the stored gaps.")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=None)
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--wait-hours", type=float, default=rc.DEFAULT_WAIT_HOURS)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    try:
        return run(args)
    except rc.StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
