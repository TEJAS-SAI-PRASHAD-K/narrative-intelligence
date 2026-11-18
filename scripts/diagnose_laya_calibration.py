"""Find out where the Laya temperature optimum lies, on the validation rows of one fine-tune.

The fine-tune does not save its validation logits. This script computes them once from the
checkpoint with ``laya.calibrate.records_from_labeled``, which returns the raw logit rows that
the decoder divides by its temperature. It stores the logit gap in ``<out>.logits.npz`` and
writes the diagnosis to ``--out``.

Run it once for each fine-tune::

    python scripts/diagnose_laya_calibration.py --model laya_en    --out reports/laya_en.json
    python scripts/diagnose_laya_calibration.py --model laya_multi --out reports/laya_multi.json

Add ``--wait-for-gpu`` on a shared host. The script reads free GPU memory first, as the stages
of ``run_comparison.py`` do, and never stops another process. ``--device`` defaults to
``cuda``, and the script refuses to go on if the agent loads on another device. Add
``--reuse-logits`` to compute the diagnosis again from the stored gap with no GPU. The stored
file holds the model name, the checkpoint hash and a hash of the validation ids, and a reuse
refuses a file that does not match all three.

Only validation rows are read. The test rows are never loaded. A text that the scorer reads
with ``predict_long`` is left out, because ``predict_long`` exposes no logits. The report
counts these rows.

The report is marked ``valid: false``, with the reasons, when it does not describe the
scorer: a binning map is installed, a language has its own temperature, or too many rebuilt
scores differ from the stored score file.
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
#: A stored score and a gap rebuilt from logits agree when they differ by at most this.
SCORE_TOLERANCE = 1.5e-4
#: The report is invalid when more than this share of the compared rows differ by more than
#: ``SCORE_TOLERANCE``. A few rows can differ by a float rounding at the fourth decimal. A
#: larger share means that the gaps are not the ones the scorer used.
SCORE_MISMATCH_LIMIT = 0.02


def checkpoint_temperature(agent: Any) -> dict[str, Any]:
    """The temperature that the checkpoint applies to a ``noul``, and where it comes from."""
    by_options = dict(getattr(agent, "temperature_by_options", {}) or {})
    per_type = list(getattr(agent, "temperature", [1.0, 1.0, 1.0]))
    bucket = lt.NOUL_BUCKET
    return {
        "applied": float(by_options.get(bucket, per_type[lt.NOUL_TYPE])),
        "source": bucket if bucket in by_options else "per_type_noul",
        "per_type": [float(t) for t in per_type],
        "per_type_raw": [float(t) for t in getattr(agent, "temperature_raw", per_type)],
        "by_options": {k: float(v) for k, v in by_options.items()},
        "binning_map_installed": bool(getattr(agent, "binning_map", None)),
        "lang_temperatures": sorted(getattr(agent, "lang_temperatures", {}) or {}),
    }


def collect_gaps(ctx: rc.Context, model: str, frame: pd.DataFrame, device: str) -> dict[str, Any]:
    """Logit gap per validation row, from the checkpoint. This is the GPU part."""
    from laya.calibrate import records_from_labeled

    from modeling.config import get_settings
    from modeling.text.laya_clf import LayaScorer

    settings = get_settings().model_copy(update={"score_unknown_language": False})
    config = rc.laya_model_config(model, ctx)
    config["device"] = device
    scorer = LayaScorer(settings=settings, _config=config)
    availability = scorer.available()
    if not availability.ok:
        raise rc.StageError(rc.EXIT_FAILED, f"[{model}] {availability.reason}")
    agent = scorer.agent_for("en")
    loaded_on = str(getattr(agent, "device", "unknown"))
    if not loaded_on.startswith(device.split(":")[0]):
        raise rc.StageError(
            rc.EXIT_FAILED,
            f"[{model}] --device {device} was asked, and the agent loaded on {loaded_on}. "
            "Nothing was computed.",
        )
    questions = scorer.question()
    qid = scorer.question_id

    pairs, kept, long_rows, blank = [], [], [], 0
    for position, (text, label) in enumerate(zip(frame["text"], frame["label"], strict=True)):
        if rc._blank(text):
            blank += 1
            continue
        if scorer._is_long(str(text), "en"):
            long_rows.append(position)
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
        "positions": np.asarray(kept, dtype=int),
        "long_positions": np.asarray(long_rows, dtype=int),
        "blank_rows": blank,
        "temperature": checkpoint_temperature(agent),
        "encode_max_len": int(agent.cfg.get("max_len", 0) or 0),
        "device": loaded_on,
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
    rebuilt = pd.Series(lt.served_scores(gap, temperature), index=ids)
    common = rebuilt.index.intersection(stored.dropna().index)
    if not len(common):
        return {"compared": 0}
    diff = (rebuilt.loc[common] - stored.loc[common]).abs()
    return {
        "compared": int(len(common)),
        "max_abs_diff": float(diff.max()),
        "share_above_tolerance": float((diff > SCORE_TOLERANCE).mean()),
        "tolerance": SCORE_TOLERANCE,
        "limit": SCORE_MISMATCH_LIMIT,
    }


def invalid_reasons(
    temperature: dict[str, Any], score_check: dict[str, Any] | None
) -> list[str]:
    """Why the report does not describe the scorer. An empty list means that it does."""
    reasons = []
    if temperature.get("binning_map_installed"):
        reasons.append("a binning map is installed on the agent, so the scores are re-mapped")
    if temperature.get("lang_temperatures"):
        reasons.append(
            f"languages {temperature['lang_temperatures']} have their own temperature"
        )
    if score_check and score_check.get("compared"):
        if score_check["share_above_tolerance"] > SCORE_MISMATCH_LIMIT:
            reasons.append(
                f"{score_check['share_above_tolerance']:.1%} of the rebuilt scores differ from "
                f"the score file by more than {SCORE_TOLERANCE}, above the limit "
                f"{SCORE_MISMATCH_LIMIT:.0%}"
            )
    return reasons


def run(args: argparse.Namespace, deps: rc.Deps | None = None) -> int:
    ns = argparse.Namespace(
        out=args.comparison_dir,
        demo=args.demo,
        wait_for_gpu=args.wait_for_gpu,
        wait_hours=args.wait_hours,
        min_free_mib=args.min_free_mib,
        gpu_index=args.gpu_index,
        batch_size=None,
    )
    ctx = rc.Context(args=ns, deps=deps or rc.Deps())
    split_info = rc.require_prepared(ctx)
    rows = pd.read_parquet(ctx.eval_rows_file)
    val = rows[rows["split"] == "val"].reset_index(drop=True)
    if sorted(val["id"].astype(str)) != sorted(split_info["ids"]["val"]):
        raise rc.StageError(rc.EXIT_SPLIT, "eval_rows.parquet and split.json differ on val ids")
    val_hash = lt.ids_hash(val["id"].astype(str))
    identity = rc.checkpoint_identity(ctx, args.model)
    if identity is None or "sha256" not in identity:
        raise rc.StageError(rc.EXIT_FAILED, f"the checkpoint of {args.model} is not present")

    cache = lt.gap_cache_path(args.logits or args.out.with_name(args.out.stem + ".logits"))
    if args.reuse_logits:
        try:
            stored = lt.read_gap_cache(
                cache,
                model=args.model,
                checkpoint_sha256=identity["sha256"],
                validation_ids_hash=val_hash,
            )
        except ValueError as exc:
            raise rc.StageError(rc.EXIT_FAILED, f"--reuse-logits: {exc}") from exc
        gap, positions = stored["gap"], stored["positions"]
        temperature = json.loads(str(stored["temperature"]))
        encode_max_len = int(stored["encode_max_len"])
        long_rows = int(len(stored["long_positions"]))
        blank_rows = int(stored["blank_rows"])
        device = str(stored["device"])
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        collected = rc.run_gpu_stage(
            ctx,
            f"diagnose-{args.model}",
            rc.DEFAULT_MIN_FREE_SCORE_MIB,
            "--min-free-mib",
            lambda: collect_gaps(ctx, args.model, val, args.device),
        )
        gap, positions = collected["gap"], collected["positions"]
        temperature, encode_max_len = collected["temperature"], collected["encode_max_len"]
        long_rows, blank_rows = int(len(collected["long_positions"])), collected["blank_rows"]
        device = collected["device"]
        lt.write_gap_cache(
            cache,
            gap=gap,
            positions=positions,
            ids=val["id"].astype(str).to_numpy()[positions].astype(str),
            long_positions=collected["long_positions"],
            blank_rows=blank_rows,
            model=args.model,
            checkpoint_sha256=identity["sha256"],
            validation_ids_hash=val_hash,
            temperature=json.dumps(temperature),
            encode_max_len=encode_max_len,
            device=device,
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
    score_check = compare_with_score_file(
        ctx, args.model, kept["id"].astype(str).to_numpy(), gap, t_applied
    )
    reasons = invalid_reasons(temperature, score_check)
    report.update(
        {
            "model": args.model,
            "valid": not reasons,
            "invalid_reasons": reasons,
            "device": device,
            "validation_rows_total": int(len(val)),
            "validation_rows_used": int(len(kept)),
            "excluded_blank_text": blank_rows,
            "excluded_long_text_predict_long": long_rows,
            "checkpoint": temperature,
            "checkpoint_sha256": identity["sha256"],
            "encode_max_len": encode_max_len,
            "vendor_fit_on_validation": vendor_fit_on_validation(gap, labels),
            "run_summary": read_run_summary(args.model, ctx),
            "score_file_check": score_check,
            "split": split_info["description"],
            "ece_definitions": {
                "ece_prob": f"{lt.ECE_BINS_PROB}-bin ECE of P(true), as modeling.eval.calibrate",
                "ece_top1": f"{lt.ECE_BINS_TOP1}-bin ECE of top-1 confidence, as the vendor",
            },
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")

    fit = report["temperature_fit"]
    ctx.say(f"[diagnose] {args.model}: {report['rows']} validation rows on {device}")
    if reasons:
        for reason in reasons:
            ctx.say(f"[diagnose] REPORT INVALID: {reason}")
    ctx.say(
        f"[diagnose] checkpoint T {t_applied:.4g}; wide-range optimum {fit['value']:.4g}"
        f" (at bound: {fit['at_bound']}); NLL at 1: {fit['nll_at_one']:.4f},"
        f" at optimum {fit['nll']:.4f}"
    )
    for name, block in report["ece"].items():
        p, t = block["ece_prob"], block["ece_top1"]
        ctx.say(
            f"[diagnose] ECE {name}: prob {p['value']:.4f} [{p['low']:.4f}, {p['high']:.4f}]"
            f", top1 {t['value']:.4f} [{t['low']:.4f}, {t['high']:.4f}]"
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
    parser.add_argument(
        "--device", default="cuda", help="Default %(default)s. The agent must load there."
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=None)
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--wait-hours", type=float, default=rc.DEFAULT_WAIT_HOURS)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, deps: rc.Deps | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    try:
        return run(args, deps)
    except rc.StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
