"""Projects and setup.

The project is the scope for everything else in the API: every list endpoint
takes ``project_id``, and there is no cross-project query anywhere. That is not
multi-tenancy -- it is the UI's project selector, and keeping it mandatory means
no endpoint can accidentally aggregate two unrelated investigations together.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import RequireRead, RequireWrite, rate_limit
from app.errors import BadRequest, NotFound
from app.mock import responses as mock
from app.schemas.common import PageResponse
from app.schemas.projects import (
    ProjectCreate,
    ProjectOut,
    ProjectStats,
    ProjectUpdate,
    SourceHealth,
    SourceTestResult,
)

router = APIRouter(prefix="/projects", tags=["projects"], dependencies=[Depends(rate_limit)])


def _to_out(row, stats: ProjectStats | None = None) -> ProjectOut:
    return ProjectOut(
        id=str(row.id),
        slug=row.slug,
        name=row.name,
        description=row.description,
        date_start=row.date_start,
        date_end=row.date_end,
        seed_config=row.seed_config,
        created_at=row.created_at,
        updated_at=row.updated_at,
        stats=stats or ProjectStats(),
    )


async def resolve_project(session: AsyncSession, identifier: str):
    """Accept a uuid or a slug.

    The CLI and the docs both use slugs because nobody can hold a uuid in their
    head, while the UI stores ids. Supporting both in one resolver means neither
    side has to translate, and there is exactly one place that knows the rule.
    """
    from app.models.core import Project

    stmt = select(Project).where(Project.slug == identifier)
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is not None:
        return row
    try:
        import uuid as _uuid

        _uuid.UUID(identifier)
    except ValueError:
        raise NotFound(
            f"No project with slug or id {identifier!r}.", code="project_not_found"
        ) from None
    row = await session.get(Project, identifier)
    if row is None:
        raise NotFound(f"No project with id {identifier}.", code="project_not_found")
    return row


@router.get("", response_model=PageResponse[ProjectOut], summary="List projects")
async def list_projects(
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PageResponse[ProjectOut]:
    if mock.demo_mode():
        items = mock.projects_list()
        return PageResponse[ProjectOut](items=items, total=len(items), filters_applied={})

    from app.models.core import Project

    rows = (await session.execute(select(Project).order_by(Project.created_at.desc()))).scalars()
    items = [_to_out(row) for row in rows]
    return PageResponse[ProjectOut](items=items, total=len(items), filters_applied={})


@router.post(
    "", response_model=ProjectOut, status_code=status.HTTP_201_CREATED, summary="Create a project"
)
async def create_project(
    body: ProjectCreate,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ProjectOut:
    if mock.demo_mode():
        return mock.projects_list()[0]

    from sqlalchemy.exc import IntegrityError

    from app.models.core import Project

    row = Project(**body.model_dump())
    session.add(row)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise BadRequest(
            f"A project with slug {body.slug!r} already exists.", code="project_slug_taken"
        ) from None
    await session.refresh(row)
    return _to_out(row)


@router.get("/{project_id}", response_model=ProjectOut, summary="Get a project")
async def get_project(
    project_id: str,
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ProjectOut:
    if mock.demo_mode():
        return mock.projects_list()[0]
    row = await resolve_project(session, project_id)
    return _to_out(row, await _project_stats(session, row.id))


@router.patch("/{project_id}", response_model=ProjectOut, summary="Update a project")
async def update_project(
    project_id: str,
    body: ProjectUpdate,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ProjectOut:
    if mock.demo_mode():
        return mock.projects_list()[0]
    row = await resolve_project(session, project_id)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(row, field, value)
    await session.commit()
    await session.refresh(row)
    return _to_out(row)


@router.get(
    "/{project_id}/sources",
    response_model=list[SourceHealth],
    summary="Per-source pipeline health",
    description=(
        "Credential state, last sync, record counts, resume checkpoint and quota. "
        "A source with no credentials reports `skipped`, not `error`: Phase 1 skips it "
        "deliberately and a red light for a configuration choice would be misleading."
    ),
)
async def project_sources(
    project_id: str,
    principal: RequireRead,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[SourceHealth]:
    if mock.demo_mode():
        return mock.project_sources()

    from app.services.sources import source_health

    row = await resolve_project(session, project_id)
    return await source_health(session, row.id)


@router.post(
    "/{project_id}/sources/{source}/test",
    response_model=SourceTestResult,
    summary="Test a source's credentials",
    description=(
        "The one route in this API that makes an outbound network call. It is a POST "
        "rather than part of the health GET precisely because it has that side effect."
    ),
)
async def test_source(
    project_id: str,
    source: str,
    principal: RequireWrite,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> SourceTestResult:
    if mock.demo_mode():
        return mock.source_test(source)

    from app.services.sources import test_source_credentials

    await resolve_project(session, project_id)
    return await test_source_credentials(source)


async def _project_stats(session: AsyncSession, project_id) -> ProjectStats:
    """Cheap counts for the project card.

    Deliberately separate from the row fetch and deliberately not a join: five
    indexed COUNTs are boring and fast, while one query with four LEFT JOINs
    multiplies rows and needs DISTINCT on every aggregate to be correct.
    """
    from sqlalchemy import func, text

    from app.models.corpus import Post

    result = await session.execute(
        select(
            func.count(Post.id),
            func.count(func.distinct(Post.author_id)),
            func.min(Post.timestamp),
            func.max(Post.timestamp),
        ).where(Post.project_id == project_id)
    )
    posts, authors, first, last = result.one()

    narratives = (
        await session.execute(
            text("SELECT count(*) FROM narratives WHERE project_id = :p"), {"p": str(project_id)}
        )
    ).scalar() or 0
    domains = (
        await session.execute(
            text("SELECT count(*) FROM domains WHERE project_id = :p"), {"p": str(project_id)}
        )
    ).scalar() or 0

    return ProjectStats(
        post_count=posts or 0,
        author_count=authors or 0,
        narrative_count=narratives,
        domain_count=domains,
        first_post_at=first,
        last_post_at=last,
    )
