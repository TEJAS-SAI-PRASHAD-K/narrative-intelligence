"""Compass Context queries.

Contexts are append-only: the live one for a narrative is the row nothing
supersedes. Reading the "latest" is therefore a filter, not an ORDER BY LIMIT 1,
which matters because a regeneration inserts before it supersedes and the two
orderings disagree for the moment in between.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import NotFound
from app.schemas.compass import (
    Citation,
    CompassContext,
    CompassFeedback,
    CompassFeedbackResult,
)

log = logging.getLogger(__name__)


def to_context_out(row, citations) -> CompassContext:
    return CompassContext(
        id=str(row.id),
        narrative_id=str(row.narrative_id),
        claim=row.claim,
        context=row.context,
        verification_status=row.verification_status,
        risk=row.risk,
        caution_note=row.caution_note,
        citations=[
            Citation(
                id=str(c.id),
                url=c.url,
                title=c.title,
                publisher=c.publisher,
                domain=c.domain,
                retrieved_at=c.retrieved_at,
                snippet=c.snippet,
                char_start=c.char_start,
                char_end=c.char_end,
            )
            for c in citations
        ],
        model=row.model,
        prompt_version=row.prompt_version,
        generated_at=row.generated_at,
        superseded_by=str(row.superseded_by) if row.superseded_by else None,
        attempts=row.attempts,
        retrieved_document_count=row.retrieved_document_count,
    )


async def latest_context(session: AsyncSession, narrative_id: str) -> CompassContext:
    from app.models.compass import CompassCitation
    from app.models.compass import CompassContext as ContextModel

    try:
        parsed = uuid.UUID(narrative_id)
    except ValueError:
        raise NotFound(
            f"No narrative with id {narrative_id}.", code="narrative_not_found"
        ) from None

    row = (
        await session.execute(
            select(ContextModel).where(
                ContextModel.narrative_id == parsed,
                ContextModel.superseded_by.is_(None),
            )
        )
    ).scalar_one_or_none()

    if row is None:
        raise NotFound(
            "No Compass Context has been generated for this narrative yet.",
            code="compass_context_not_found",
            detail={
                "narrative_id": narrative_id,
                "hint": f"POST /api/v1/narratives/{narrative_id}/compass/regenerate",
            },
        )

    citations = (
        (
            await session.execute(
                select(CompassCitation)
                .where(CompassCitation.context_id == row.id)
                .order_by(CompassCitation.char_start.nullslast())
            )
        )
        .scalars()
        .all()
    )
    return to_context_out(row, citations)


async def record_feedback(
    session: AsyncSession, context_id: str, body: CompassFeedback, *, actor: str
) -> CompassFeedbackResult:
    """Store a rating for the writeup's error analysis.

    Every rating is kept, not just the latest per user: the interesting signal
    is how often analysts disagree with a context, and deduplicating would erase
    exactly that.
    """
    from app.models.compass import CompassContext as ContextModel
    from app.models.compass import CompassFeedback as FeedbackModel

    context = await session.get(ContextModel, context_id)
    if context is None:
        raise NotFound(
            f"No Compass Context with id {context_id}.", code="compass_context_not_found"
        )

    session.add(
        FeedbackModel(
            context_id=context.id,
            helpful=body.helpful,
            reason_code=body.reason_code,
            reason=body.reason,
            submitted_by=actor,
        )
    )
    await session.commit()

    counts = dict(
        (
            await session.execute(
                select(FeedbackModel.helpful, func.count())
                .where(FeedbackModel.context_id == context.id)
                .group_by(FeedbackModel.helpful)
            )
        ).all()
    )
    return CompassFeedbackResult(
        context_id=str(context.id),
        recorded=True,
        helpful_count=counts.get(True, 0),
        unhelpful_count=counts.get(False, 0),
    )
