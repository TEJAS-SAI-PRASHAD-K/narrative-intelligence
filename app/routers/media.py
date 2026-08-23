"""Deepfake checks.

Three things this router is careful about, all of them because the payload is
imagery of real people:

* **MIME is validated by magic bytes, not by the extension or the client's
  Content-Type.** Both are attacker-controlled and neither says what the file is.
* **The size cap is enforced while streaming, not after.** Reading a 4GB body
  into memory to discover it is too large is the denial of service, not the
  protection against it.
* **Uploads are deleted on a retention timer** and the API says when, in every
  response. Indefinite retention of faces in a research project is not
  defensible.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_api_settings
from app.db import get_session
from app.deps import Pagination, RequireRead, RequireWrite, rate_limit
from app.errors import PayloadTooLarge, UnsupportedMedia
from app.mock import responses as mock
from app.schemas.common import JobAccepted, PageResponse
from app.schemas.media import MediaCheckOut

log = logging.getLogger(__name__)

router = APIRouter(prefix="/media", tags=["media"], dependencies=[Depends(rate_limit)])

#: Magic-byte signatures for the formats the deepfake module can actually read.
#: An extension is a claim; these are evidence.
MAGIC_SIGNATURES: tuple[tuple[bytes, int, str], ...] = (
    (b"\xff\xd8\xff", 0, "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", 0, "image/png"),
    (b"RIFF", 0, "image/webp"),  # narrowed further below by the WEBP tag at 8
    (b"GIF87a", 0, "image/gif"),
    (b"GIF89a", 0, "image/gif"),
    (b"ftyp", 4, "video/mp4"),
    (b"\x1a\x45\xdf\xa3", 0, "video/webm"),
)

CHUNK = 1 << 20


def sniff_media_type(head: bytes) -> str | None:
    """Identify a file from its first bytes. Returns None for anything unknown."""
    for signature, offset, media_type in MAGIC_SIGNATURES:
        if head[offset : offset + len(signature)] == signature:
            if media_type == "image/webp" and head[8:12] != b"WEBP":
                continue
            return media_type
    return None


@router.post(
    "/check",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit media for a deepfake check",
)
async def submit_media_check(
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
    file: Annotated[UploadFile, File(description="Image or video. 100MB ceiling.")],
) -> JobAccepted:
    settings = get_api_settings()
    job_id = str(uuid.uuid4())
    target_dir = Path(settings.uploads_dir) / job_id
    target_dir.mkdir(parents=True, exist_ok=True)
    # The client's filename is untrusted input and never becomes a path
    # component: `../../etc/passwd` is a perfectly valid string in that field.
    safe_name = Path(file.filename or "upload").name or "upload"
    target = target_dir / safe_name

    head = await file.read(64)
    media_type = sniff_media_type(head)
    if media_type is None:
        target_dir.rmdir()
        raise UnsupportedMedia(
            "Could not identify the file from its magic bytes. Supported: JPEG, PNG, "
            "WebP, GIF, MP4, WebM. The extension and Content-Type header are ignored "
            "on purpose -- both are client-supplied and neither says what the file is.",
            detail={"detected": None, "first_bytes_hex": head[:8].hex()},
        )

    written = len(head)
    try:
        with target.open("wb") as sink:
            sink.write(head)
            while chunk := await file.read(CHUNK):
                written += len(chunk)
                # Enforced while streaming. Buffering the whole body first to
                # measure it *is* the denial of service this cap exists to stop.
                if written > settings.media_max_bytes:
                    raise PayloadTooLarge(
                        f"Upload exceeds the {settings.media_max_bytes // (1024 * 1024)}MB limit.",
                        detail={"limit_bytes": settings.media_max_bytes},
                    )
                sink.write(chunk)
    except PayloadTooLarge:
        target.unlink(missing_ok=True)
        target_dir.rmdir()
        raise

    if mock.demo_mode():
        target.unlink(missing_ok=True)
        target_dir.rmdir()
        return JobAccepted(
            job_id=job_id,
            status="pending",
            status_url=f"/api/v1/media/check/{job_id}",
            kind="media.deepfake",
            message="DEMO_MODE: the upload was discarded and no inference was run.",
        )

    from app.services.jobs import enqueue

    return await enqueue(
        session,
        kind="media.deepfake",
        project_id=None,
        params={
            "job_id": job_id,
            "path": str(target),
            "media_type": media_type,
            "filename": safe_name,
            "size_bytes": written,
        },
        job_id=job_id,
        status_url=f"/api/v1/media/check/{job_id}",
    )


@router.get("/checks", response_model=PageResponse[MediaCheckOut], summary="Deepfake check history")
async def list_media_checks(
    principal: RequireRead,
    page: Pagination,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PageResponse[MediaCheckOut]:
    if mock.demo_mode():
        items = [mock.media_check("demo-job-1"), mock.media_check("demo-job-2", "running")]
        return mock.paginate(items, page, {})
    from app.repositories.media import list_checks

    return await list_checks(session, page)


@router.get(
    "/check/{job_id}",
    response_model=MediaCheckOut,
    summary="Deepfake check result",
    description=(
        "Carries a plain-language `explanation` alongside the confidence. A bare "
        "number is not an acceptable answer to 'is this video fake', and the UI spec "
        "says so explicitly."
    ),
)
async def get_media_check(
    job_id: str,
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> MediaCheckOut:
    if mock.demo_mode():
        return mock.media_check(job_id)
    from app.repositories.media import get_check

    return await get_check(session, job_id)
