"""The risk sort and the score filters both need post_scores.

Needs Postgres: the failure is the database refusing a query that repeats a
table name, which SQLite-style compilation checks do not reproduce.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from tests.conftest_api import requires_postgres


@pytest.fixture
def scored_posts(migrated_db):
    from app.db import sync_session
    from app.models.core import Project
    from app.models.corpus import Post, PostScore

    project_id = uuid.uuid4()
    rows = [
        ("a", 0.9, 0.0),
        ("b", 0.2, 0.5),
        ("c", 0.6, None),  # toxicity not measured
        ("d", None, 0.7),  # misinfo not measured
    ]
    with sync_session() as session:
        session.add(Project(id=project_id, slug="risk-project", name="Risk"))
        session.flush()
        for name, _misinfo, _toxicity in rows:
            session.add(
                Post(
                    id=f"reddit:{name}",
                    project_id=project_id,
                    native_id=name,
                    source="reddit",
                    source_detail="r/test",
                    content_type="post",
                    text_=f"post {name}",
                    author_id=f"reddit:{name}",
                    timestamp=datetime(2026, 5, 1, tzinfo=UTC),
                )
            )
        session.flush()
        for name, misinfo, toxicity in rows:
            session.add(
                PostScore(
                    post_id=f"reddit:{name}", misinfo_likelihood=misinfo, toxicity=toxicity
                )
            )
    return "risk-project"


def _ids(response) -> list[str]:
    assert response.status_code == 200, response.text
    return [item["id"] for item in response.json()["items"]]


@requires_postgres
def test_risk_sort_with_a_score_filter_joins_post_scores_once(client, read_key, scored_posts):
    response = client.get(
        "/api/v1/posts",
        params={"project_id": scored_posts, "sort": "risk", "min_toxicity": 0.0},
        headers={"X-API-Key": read_key},
    )
    # min_toxicity=0.0 keeps measured zeros and drops nulls; the risk sort drops
    # posts with no misinfo score and orders the rest high to low.
    assert _ids(response) == ["reddit:a", "reddit:b"]


@requires_postgres
def test_risk_sort_alone_still_works(client, read_key, scored_posts):
    response = client.get(
        "/api/v1/posts",
        params={"project_id": scored_posts, "sort": "risk"},
        headers={"X-API-Key": read_key},
    )
    assert _ids(response) == ["reddit:a", "reddit:c", "reddit:b"]
