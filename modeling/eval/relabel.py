"""One run that reads the filled label sheets and rewrites every labelling output.

For each sheet that exists, this writes into the output directory:

* ``<lang>/agreement.md`` and ``<lang>/disagreements.csv`` (ids and labels, no text),
* ``<lang>/transfer.md`` and ``<lang>/transfer.json``: ``misinfo`` for English, Laya on
  titles for Hindi.

A sheet that is refused still gets its files, and they say why. When labels come from a
model and the caller opts in, every file carries the line ``labeller: model, unreviewed
by a person``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.eval.labelling import (
    agreement_report,
    read_sheet,
    render_agreement,
    resolve_sheet_file,
)
from modeling.eval.metrics import Interval
from modeling.eval.weighted import DEFAULT_SEED, StratumWeights, stratum_warning

log = logging.getLogger(__name__)

ScoresReader = Callable[[], "pd.DataFrame | None"]


@dataclass
class Section:
    lines: list[str]
    data: dict[str, Any]
    ok: bool


@dataclass
class RelabelOutcome:
    written: list[Path] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    ok: bool = True


def _default_scores() -> pd.DataFrame | None:
    from modeling.config import get_settings
    from modeling.io import ScoredStore

    return ScoredStore(get_settings()).read("record_scores")


def _jsonable(value: Any) -> Any:
    """JSON-safe form of a result. A NaN, as in an interval that has too few draws, is null."""
    if isinstance(value, Interval):
        return _jsonable(value.as_dict())
    if isinstance(value, StratumWeights):
        return {
            "per_stratum": value.per_stratum,
            "empty": value.empty,
            "small": value.small,
            "single": value.single,
            "interval_reliable": value.interval_reliable,
            "warnings": stratum_warning(value),
            "n_eff": value.n_eff,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def misinfo_english_section(
    sheet_path: Path,
    *,
    allow_model_labels: bool,
    scores: ScoresReader | None = None,
    min_labelled: int | None = None,
    n_bootstrap: int = 1000,
) -> Section:
    from modeling.config import get_settings, module_config
    from modeling.eval.corpus_transfer import evaluate_transfer

    resolved = resolve_sheet_file(sheet_path, allow_model_labels=allow_model_labels)
    settings = get_settings()
    config = module_config("misinfo")
    result = evaluate_transfer(
        resolved,
        (scores or _default_scores)(),
        language_allowed=lambda lang: settings.language_allowed("misinfo", lang),
        model_version=str(config.get("version", "v0.0.0-unset")),
        threshold=config.get("decision_threshold"),
        min_labelled=min_labelled,
        n_bootstrap=n_bootstrap,
        seed=DEFAULT_SEED,
    )
    return Section(result.render(), _jsonable(result), result.computed)


def laya_hindi_section(
    sheet_path: Path,
    *,
    allow_model_labels: bool,
    threshold: float | None = None,
    scores: ScoresReader | None = None,
    min_labelled: int | None = None,
    n_bootstrap: int = 1000,
) -> Section:
    from modeling.config import module_config
    from modeling.eval.laya_transfer import evaluate_laya_transfer

    if not sheet_path.exists():
        return Section(
            [f"laya hindi transfer: NOT MEASURED - no labelled sample at {sheet_path}"],
            {"not_measured": True},
            False,
        )
    resolved = resolve_sheet_file(sheet_path, allow_model_labels=allow_model_labels)
    config = module_config("laya")
    configured = config.get("decision_threshold")
    chosen = threshold if threshold is not None else configured
    source = (
        "given on the command line, not the configured threshold"
        if threshold is not None
        else "laya.decision_threshold"
    )
    result = evaluate_laya_transfer(
        resolved,
        (scores or _default_scores)(),
        model_version=str(config.get("version", "v0.0.0-unset")),
        threshold=chosen,
        threshold_source=source if chosen is not None else "unset",
        min_labelled=min_labelled,
        n_bootstrap=n_bootstrap,
        seed=DEFAULT_SEED,
    )
    return Section(result.render(), _jsonable(result), result.computed)


def _write(path: Path, text: str, written: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    written.append(path)


def _fenced(title: str, lines: list[str]) -> str:
    return f"# {title}\n\n```text\n" + "\n".join(lines) + "\n```\n"


def run_relabel_eval(
    *,
    en_sheet: Path,
    hi_sheet: Path,
    out_dir: Path,
    allow_model_labels: bool = False,
    laya_threshold: float | None = None,
    scores: ScoresReader | None = None,
    min_labelled: int | None = None,
    n_bootstrap: int = 1000,
) -> RelabelOutcome:
    """Rewrite every labelling output from the sheets on disk."""
    outcome = RelabelOutcome()
    for lang, path in (("en", en_sheet), ("hi", hi_sheet)):
        if not path.exists():
            outcome.lines.append(f"{lang}: NOT MEASURED - no sheet at {path}")
            outcome.ok = False
            continue
        sheet = read_sheet(path)
        resolved = resolve_sheet_file(path, allow_model_labels=allow_model_labels)
        note = [resolved.labeller_note] if resolved.labeller_note else []

        report = agreement_report(sheet, seed=DEFAULT_SEED)
        agreement_lines = note + render_agreement(report, DEFAULT_SEED) + note
        _write(
            out_dir / lang / "agreement.md",
            _fenced(f"Agreement, {lang}", agreement_lines),
            outcome.written,
        )
        rows = [{"column": col, **d} for col, item in report.items() for d in item.disagreements]
        disagreements = pd.DataFrame(
            rows,
            columns=[
                "column",
                "id",
                "label_a",
                "label_b",
                "reports_on_false_claim_a",
                "reports_on_false_claim_b",
            ],
        )
        csv_path = out_dir / lang / "disagreements.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        disagreements.to_csv(csv_path, index=False)
        outcome.written.append(csv_path)

        if lang == "en":
            section = misinfo_english_section(
                path,
                allow_model_labels=allow_model_labels,
                scores=scores,
                min_labelled=min_labelled,
                n_bootstrap=n_bootstrap,
            )
        else:
            section = laya_hindi_section(
                path,
                allow_model_labels=allow_model_labels,
                threshold=laya_threshold,
                scores=scores,
                min_labelled=min_labelled,
                n_bootstrap=n_bootstrap,
            )
        stamp = {"labeller": "model, unreviewed by a person"} if note else {}
        payload = {
            **stamp,
            "sheet": path.name,
            "seed": DEFAULT_SEED,
            "computed": section.ok,
            **section.data,
        }
        _write(
            out_dir / lang / "transfer.md",
            _fenced(f"Transfer, {lang}", section.lines),
            outcome.written,
        )
        _write(
            out_dir / lang / "transfer.json",
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            outcome.written,
        )
        outcome.lines += [f"[{lang}]", *section.lines, *render_agreement(report, DEFAULT_SEED)[:1]]
        outcome.ok = outcome.ok and section.ok
    return outcome
