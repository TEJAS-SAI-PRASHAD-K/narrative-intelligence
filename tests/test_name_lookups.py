"""Name lookups for cohorts and author groups against a real schema.

Needs Postgres: the repository parses the id as a uuid and queries one column,
and a malformed id must be a 404 rather than a database error.
"""

from __future__ import annotations

import uuid

import pytest

from tests.conftest_api import requires_postgres


@pytest.fixture
def named_entities(migrated_db):
    from app.db import sync_session
    from app.models.actors import AuthorGroup, Cohort
    from app.models.core import Project

    project_id = uuid.uuid4()
    cohort_id = uuid.uuid4()
    group_id = uuid.uuid4()
    with sync_session() as session:
        session.add(Project(id=project_id, slug="names-project", name="Names"))
        session.flush()
        session.add(
            Cohort(
                id=cohort_id,
                project_id=project_id,
                name="Verified journalists",
                category="influence_tier",
            )
        )
        session.add(AuthorGroup(id=group_id, project_id=project_id, name="Watchlist A"))
    return {"cohort": str(cohort_id), "group": str(group_id)}


@requires_postgres
def test_cohort_lookup_returns_the_name(client, read_key, named_entities):
    response = client.get(
        f"/api/v1/cohorts/{named_entities['cohort']}", headers={"X-API-Key": read_key}
    )
    assert response.status_code == 200
    assert response.json() == {"id": named_entities["cohort"], "name": "Verified journalists"}


@requires_postgres
def test_author_group_lookup_returns_the_name(client, read_key, named_entities):
    response = client.get(
        f"/api/v1/author-groups/{named_entities['group']}", headers={"X-API-Key": read_key}
    )
    assert response.status_code == 200
    assert response.json() == {"id": named_entities["group"], "name": "Watchlist A"}


@requires_postgres
@pytest.mark.parametrize(
    ("route", "code"),
    [("cohorts", "cohort_not_found"), ("author-groups", "author_group_not_found")],
)
@pytest.mark.parametrize("missing", [str(uuid.uuid4()), "not-a-uuid"])
def test_unknown_id_is_404(client, read_key, named_entities, route, code, missing):
    response = client.get(f"/api/v1/{route}/{missing}", headers={"X-API-Key": read_key})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == code
