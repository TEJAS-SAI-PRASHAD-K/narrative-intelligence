"""Chart and drilldown series report unmeasured engagement as null, not 0.

Needs Postgres. The per-row COALESCE that used to sit inside these SUMs made an
all-NULL bucket read as a measured zero.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from tests.conftest_api import requires_postgres
from tests.test_routers.test_overview import _seed_engagement

# One post per hour, in this order. Hour 0 and 3 measure nothing, hour 1 measures
# one input, hour 2 is a measured zero.
ROWS = [
    (None, None, None),
    (5, None, 2),
    (0, None, None),
    (None, None, None),
]


def _get(client, admin_key, path, project_id, **params):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    response = client.get(
        f"/api/v1{path}?project_id={project_id}&{query}", headers={"X-API-Key": admin_key}
    )
    assert response.status_code == 200, response.text
    return response.json()


@requires_postgres
def test_timeseries_engagement_is_null_for_an_unmeasured_bucket(client, migrated_db, admin_key):
    project_id = uuid.uuid4()
    _seed_engagement(project_id, ROWS)
    body = _get(
        client, admin_key, "/overview/timeseries", project_id, metric="engagements", interval="1h"
    )

    assert [p["value"] for p in body["points"]] == [None, 7, 0, None]


@requires_postgres
def test_timeseries_grouped_series_keeps_null_per_series(client, migrated_db, admin_key):
    project_id = uuid.uuid4()
    _seed_engagement(project_id, ROWS)
    body = _get(
        client,
        admin_key,
        "/overview/timeseries",
        project_id,
        metric="engagements",
        interval="1h",
        group_by="platform",
    )

    assert [p["series"]["news_rss"] for p in body["points"]] == [None, 7, 0, None]


@requires_postgres
def test_timeseries_posts_metric_still_reports_counts(client, migrated_db, admin_key):
    project_id = uuid.uuid4()
    _seed_engagement(project_id, ROWS)
    body = _get(
        client, admin_key, "/overview/timeseries", project_id, metric="posts", interval="1h"
    )

    assert [p["value"] for p in body["points"]] == [1, 1, 1, 1]


@requires_postgres
def test_drivers_engagement_total_is_null_when_no_post_measured_it(
    client, migrated_db, admin_key
):
    project_id = uuid.uuid4()
    _seed_engagement(project_id, ROWS)
    body = _get(client, admin_key, "/drivers/authors", project_id)

    totals = {item["entity"]: item["engagement_total"] for item in body["items"]}
    assert totals["news:author0"] is None
    assert totals["news:author1"] == 7
    assert totals["news:author2"] == 0
    assert totals["news:author3"] is None


@requires_postgres
def test_engagement_sort_ranks_unmeasured_posts_last(client, migrated_db, admin_key):
    project_id = uuid.uuid4()
    _seed_engagement(project_id, ROWS)
    body = _get(client, admin_key, "/posts", project_id, sort="engagement")

    suffixes = [item["id"].rsplit(":", 1)[1] for item in body["items"]]
    assert suffixes[:2] == ["1", "2"]
    assert set(suffixes[2:]) == {"0", "3"}


def _seed_author(project_id, author_id):
    from app.db import sync_session
    from app.models.actors import Author

    with sync_session() as session:
        session.add(
            Author(author_id=author_id, project_id=project_id, source="news_rss", handle="h")
        )


def _seed_narrative(project_id, post_ids):
    from app.db import sync_session
    from app.models.narratives import Narrative, NarrativePost

    narrative_id = uuid.uuid4()
    with sync_session() as session:
        session.add(Narrative(id=narrative_id, project_id=project_id))
        session.flush()
        for post_id in post_ids:
            session.add(NarrativePost(narrative_id=narrative_id, post_id=post_id))
    return narrative_id


def _seed_posts(project_id, rows):
    """rows: (id, source, hour, likes, shares, replies, author_id)."""
    from app.db import sync_session
    from app.models.core import Project
    from app.models.corpus import Post

    base = datetime(2026, 5, 1, tzinfo=UTC)
    with sync_session() as session:
        session.add(Project(id=project_id, slug=f"ser-{project_id.hex[:8]}", name="Series"))
        session.flush()
        for post_id, source, hour, likes, shares, replies, author_id in rows:
            session.add(
                Post(
                    id=post_id,
                    project_id=project_id,
                    native_id=post_id,
                    source=source,
                    source_detail="x",
                    content_type="post",
                    text_=post_id,
                    author_id=author_id,
                    timestamp=base + timedelta(hours=hour),
                    likes=likes,
                    shares=shares,
                    replies=replies,
                )
            )


@requires_postgres
def test_author_timeline_engagement_is_null_for_an_unmeasured_bucket(
    client, migrated_db, admin_key
):
    project_id = uuid.uuid4()
    rows = [
        (f"a:{i}", "news_rss", i, *measures, "a:author")
        for i, measures in enumerate(ROWS)
    ]
    _seed_posts(project_id, rows)
    _seed_author(project_id, "a:author")
    response = client.get(
        "/api/v1/authors/a:author/timeline?interval=1h", headers={"X-API-Key": admin_key}
    )
    assert response.status_code == 200, response.text

    assert [p["engagement"] for p in response.json()["points"]] == [None, 7, 0, None]


@requires_postgres
def test_narrative_timeline_merges_sources_by_measured_values_only(
    client, migrated_db, admin_key
):
    project_id = uuid.uuid4()
    rows = [
        # hour 0: both sources unmeasured
        ("n:0a", "news_rss", 0, None, None, None, "n:x"),
        ("n:0b", "reddit", 0, None, None, None, "n:y"),
        # hour 1: an unmeasured source plus a measured one sums only the measured one
        ("n:1a", "news_rss", 1, None, None, None, "n:x"),
        ("n:1b", "reddit", 1, 4, None, 3, "n:y"),
        # hour 2: a measured zero
        ("n:2a", "reddit", 2, 0, None, None, "n:y"),
    ]
    _seed_posts(project_id, rows)
    narrative_id = _seed_narrative(project_id, [r[0] for r in rows])
    response = client.get(
        f"/api/v1/narratives/{narrative_id}/timeline?interval=1h",
        headers={"X-API-Key": admin_key},
    )
    assert response.status_code == 200, response.text

    assert [p["engagement"] for p in response.json()["points"]] == [None, 7, 0]


@requires_postgres
def test_high_risk_engagement_is_null_when_no_scored_post_measured_it(
    client, migrated_db, admin_key
):
    from app.db import sync_session
    from app.models.corpus import PostScore

    project_id = uuid.uuid4()
    rows = [
        ("h:0", "news_rss", 0, None, None, 9, "h:a"),  # replies are not in the definition
        ("h:1", "news_rss", 1, 3, None, None, "h:b"),
        ("h:2", "news_rss", 2, 100, 100, None, "h:c"),  # measured but not high risk
    ]
    _seed_posts(project_id, rows)
    with sync_session() as session:
        for post_id, risk in (("h:0", 0.9), ("h:1", 0.9), ("h:2", 0.1)):
            session.add(PostScore(post_id=post_id, misinfo_likelihood=risk))
    headers = {"X-API-Key": admin_key}
    body = client.get(
        f"/api/v1/overview/kpis/engagements?project_id={project_id}", headers=headers
    ).json()
    assert body["on_high_risk"]["value"] == 3

    only_replies = uuid.uuid4()
    _seed_posts(only_replies, [("h2:0", "news_rss", 0, None, None, 9, "h2:a")])
    with sync_session() as session:
        session.add(PostScore(post_id="h2:0", misinfo_likelihood=0.9))
    body = client.get(
        f"/api/v1/overview/kpis/engagements?project_id={only_replies}", headers=headers
    ).json()
    assert body["on_high_risk"]["value"] is None


@requires_postgres
def test_engagement_sort_ranks_a_replies_only_post_last(client, migrated_db, admin_key):
    project_id = uuid.uuid4()
    rows = [
        ("s:0", "news_rss", 0, None, None, 50, "s:a"),
        ("s:1", "news_rss", 1, 0, None, None, "s:b"),
        ("s:2", "news_rss", 2, 2, None, None, "s:c"),
    ]
    _seed_posts(project_id, rows)
    body = _get(client, admin_key, "/posts", project_id, sort="engagement")

    assert [item["id"] for item in body["items"]] == ["s:2", "s:1", "s:0"]


def _two_narratives(project_id):
    """One narrative where no post measures engagement, one with a mixed set."""
    rows = [
        ("w:0", "news_rss", 0, None, None, None, "w:a"),
        ("w:1", "news_rss", 1, None, None, None, "w:b"),
        ("w:2", "news_rss", 2, 5, None, 2, "w:c"),
        ("w:3", "news_rss", 3, None, None, None, "w:d"),
    ]
    _seed_posts(project_id, rows)
    return rows


@requires_postgres
def test_cluster_writer_stores_null_for_an_unmeasured_narrative(migrated_db):
    from app.db import sync_session
    from app.tasks.nlp import _persist_clusters

    project_id = uuid.uuid4()
    _two_narratives(project_id)
    run_id = _seed_run(project_id)
    members = {0: [("w:0", 0.9), ("w:1", 0.8)], 1: [("w:2", 0.9), ("w:3", 0.8)]}
    with sync_session() as session:
        _persist_clusters(session, project_id, run_id, members)
        session.flush()
        from sqlalchemy import text

        stored = {
            row.cluster_id: row.engagement_total
            for row in session.execute(
                text("SELECT cluster_id, engagement_total FROM narratives WHERE project_id = :p"),
                {"p": project_id},
            )
        }
    assert stored == {0: None, 1: 7}


def _seed_run(project_id):
    from app.db import sync_session
    from app.models.narratives import ClusteringRun

    run_id = uuid.uuid4()
    with sync_session() as session:
        session.add(ClusteringRun(id=run_id, project_id=project_id, algorithm="hdbscan"))
    return run_id


@requires_postgres
def test_scored_loader_stores_null_for_an_unmeasured_narrative(migrated_db, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sqlalchemy import text

    from app.db import sync_session
    from app.etl.scored_loader import load_narratives

    project_id = uuid.uuid4()
    _two_narratives(project_id)
    narratives = tmp_path / "narratives"
    membership = tmp_path / "narrative_membership"
    narratives.mkdir()
    membership.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"narrative_id": "quiet", "label": "Quiet", "size": 2},
                {"narrative_id": "loud", "label": "Loud", "size": 2},
            ]
        ),
        narratives / "part.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"narrative_id": "quiet", "record_id": "w:0"},
                {"narrative_id": "quiet", "record_id": "w:1"},
                {"narrative_id": "loud", "record_id": "w:2"},
                {"narrative_id": "loud", "record_id": "w:3"},
            ]
        ),
        membership / "part.parquet",
    )
    with sync_session() as session:
        load_narratives(session, tmp_path, project_id)
        session.flush()
        stored = {
            row.title: row.engagement_total
            for row in session.execute(
                text("SELECT title, engagement_total FROM narratives WHERE project_id = :p"),
                {"p": project_id},
            )
        }
    assert stored == {"Quiet": None, "Loud": 7}


@requires_postgres
def test_narrative_list_reports_null_engagement_total_and_sorts_it_last(
    client, migrated_db, admin_key
):
    from app.db import sync_session
    from app.models.narratives import Narrative

    project_id = uuid.uuid4()
    _two_narratives(project_id)
    quiet = _seed_narrative(project_id, ["w:0", "w:1"])
    loud = _seed_narrative(project_id, ["w:2", "w:3"])
    with sync_session() as session:
        session.get(Narrative, loud).engagement_total = 7
    body = _get(client, admin_key, "/narratives", project_id, sort="engagement")

    assert [(i["id"], i["engagement_total"]) for i in body["items"]] == [
        (str(loud), 7),
        (str(quiet), None),
    ]


@pytest.fixture
def demo_client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app.config import get_api_settings

    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("API_KEY_PEPPER", "contract-test-pepper")
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("REDIS_URL", "")
    get_api_settings.cache_clear()

    from app.main import create_app

    with TestClient(create_app()) as demo:
        yield demo
    get_api_settings.cache_clear()


def test_demo_timeseries_reports_none_for_an_unmeasured_engagement_bucket(demo_client):
    from app.mock.corpus import corpus

    project_id = corpus().project_id
    body = demo_client.get(
        f"/api/v1/overview/timeseries?project_id={project_id}"
        "&metric=engagements&group_by=none&interval=1h"
    ).json()
    values = [p["value"] for p in body["points"]]
    assert None in values
    assert any(v for v in values if v is not None)
