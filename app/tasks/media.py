"""Deepfake inference over uploaded media."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.tasks.base import TrackedTask, report_progress
from app.tasks.celery_app import celery

log = logging.getLogger(__name__)


@celery.task(name="media.deepfake", base=TrackedTask, bind=True)
def deepfake(self, *, job_id: str | None = None, project_id: str = "", **params: Any) -> dict:
    """Analyse one uploaded file and write a MediaCheck row.

    The row is written whatever happens, including when no model is available,
    because the client is polling `/media/check/{job_id}` and a missing row
    reads as "still working" forever.
    """
    from datetime import timedelta

    from app.config import get_api_settings
    from app.db import utcnow
    from nlp.adapters import get_deepfake_detector

    settings = get_api_settings()
    path = params.get("path")
    check_id = uuid.UUID(params["job_id"]) if params.get("job_id") else uuid.uuid4()
    deletes_at = utcnow() + timedelta(hours=settings.media_retention_hours)

    base = {
        "id": uuid.uuid4(),
        "job_id": check_id,
        "project_id": None,
        "post_id": None,
        "filename": params.get("filename"),
        "media_type": params.get("media_type"),
        "size_bytes": params.get("size_bytes"),
        "storage_path": path,
        "submitted_at": utcnow(),
        "deletes_at": deletes_at,
    }

    detector = get_deepfake_detector()
    if detector is None:
        _write(session=None, row={**base, **_unavailable_row()})
        return {
            "verdict": "inconclusive",
            "reason": (
                "No deepfake checkpoint is mounted. The upload is recorded and will be "
                "purged on schedule; no inference was run."
            ),
        }

    report_progress(job_id, 0.3)
    try:
        analysis = detector.analyze(str(path))
    except Exception as exc:
        log.exception("deepfake inference failed")
        _write(session=None, row={**base, **_failed_row(exc)})
        raise

    report_progress(job_id, 0.85)
    probability = analysis.get("deepfake_prob")
    face_detected = analysis.get("face_detected")
    row = {
        **base,
        "verdict": _verdict(probability, face_detected),
        "confidence": probability,
        "manipulation_type": analysis.get("manipulation_type"),
        "frames_analyzed": analysis.get("frames_analyzed"),
        "face_detected": face_detected,
        "explanation": analysis.get("explanation") or _explain(analysis),
        "limitations": _limitations(analysis),
        "model": detector.info.name,
        "model_version": detector.info.version,
        "completed_at": utcnow(),
    }
    _write(session=None, row=row)

    result = {
        "verdict": row["verdict"],
        "confidence": probability,
        "frames_analyzed": row["frames_analyzed"],
        "deletes_at": deletes_at.isoformat(),
    }
    report_progress(job_id, 1.0, result=result)
    return result


def _write(session, row: dict) -> None:
    from sqlalchemy import text

    from app.db import sync_session

    with sync_session() as active:
        active.execute(
            text(
                """
                INSERT INTO media_checks (
                    id, job_id, project_id, post_id, filename, media_type, size_bytes,
                    storage_path, verdict, confidence, manipulation_type, frames_analyzed,
                    face_detected, explanation, limitations, model, model_version,
                    submitted_at, completed_at, deletes_at
                ) VALUES (
                    :id, :job_id, :project_id, :post_id, :filename, :media_type, :size_bytes,
                    :storage_path, :verdict, :confidence, :manipulation_type, :frames_analyzed,
                    :face_detected, :explanation, :limitations, :model, :model_version,
                    :submitted_at, :completed_at, :deletes_at
                )
                """
            ),
            {
                "verdict": None,
                "confidence": None,
                "manipulation_type": None,
                "frames_analyzed": None,
                "face_detected": None,
                "explanation": None,
                "limitations": [],
                "model": None,
                "model_version": None,
                "completed_at": None,
                **row,
            },
        )


def _unavailable_row() -> dict:
    return {
        "verdict": "inconclusive",
        "explanation": (
            "No deepfake model is available in this deployment, so this file was not "
            "analysed. This is not a finding about the file: it is the absence of one. "
            "The upload is still deleted on the retention schedule."
        ),
        "limitations": ["No deepfake checkpoint was mounted when this file was submitted."],
        "completed_at": None,
    }


def _failed_row(exc: Exception) -> dict:
    return {
        "verdict": "inconclusive",
        "explanation": (
            f"Analysis failed ({type(exc).__name__}). No verdict can be given. A failed "
            "analysis says nothing about whether the media is authentic."
        ),
        "limitations": [f"Inference error: {type(exc).__name__}"],
    }


def _verdict(probability: float | None, face_detected: bool | None) -> str:
    """Probability -> the four-value verdict.

    "No face found" is `inconclusive`, never `likely_authentic`. A frame with no
    face is a frame the model could not assess, and reporting that as authentic
    is the single most misleading thing this module could do.
    """
    if face_detected is False:
        return "inconclusive"
    if probability is None:
        return "inconclusive"
    if probability >= 0.75:
        return "likely_manipulated"
    if probability >= 0.5:
        return "possibly_manipulated"
    return "likely_authentic"


def _explain(analysis: dict) -> str:
    """A plain-language account, because a bare number is not an answer.

    The UI spec is explicit that "0.68" is not an acceptable response to "is
    this video fake". This is what an analyst quotes in a report.
    """
    frames = analysis.get("frames_analyzed") or 0
    probability = analysis.get("deepfake_prob")
    if probability is None:
        return "The model produced no score for this file, so there is no verdict."
    if analysis.get("face_detected") is False:
        return (
            f"{frames} frame(s) were sampled and no face was found in any of them. This "
            "model only assesses faces, so it has no opinion on this file. That is not "
            "evidence the media is authentic."
        )
    confidence = (
        "well above" if probability >= 0.75 else ("around" if probability >= 0.5 else "below")
    )
    return (
        f"{frames} frame(s) were sampled and scored, and the aggregate manipulation "
        f"score is {probability:.2f}, {confidence} the 0.5 decision threshold. "
        "Compression and low resolution produce artefacts similar to manipulation, so "
        "treat this as a lead for manual review rather than a determination."
    )


def _limitations(analysis: dict) -> list[str]:
    out = [
        "Trained on FaceForensics++ and DFDC; generators outside those sets are out "
        "of distribution.",
        "Heavy compression produces artefacts similar to manipulation.",
    ]
    if analysis.get("face_detected") is False:
        out.append(
            "No face was detected, so no verdict is possible. This is not evidence "
            "that the media is authentic."
        )
    frames = analysis.get("frames_analyzed") or 0
    if frames and frames < 10:
        out.append(f"Only {frames} frame(s) were analysed; confidence is correspondingly weak.")
    return out
