"""Overview aggregations against real rows.

These are the numbers on the dashboard's top row, so the tests are about them
agreeing with each other and with the corpus -- not about the endpoints
returning 200.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest_api import requires_postgres


@pytest.fixture
def seeded(migrated_db):
    """A small corpus with deliberately uneven scoring coverage."""
    from app.db import sync_session
    from app.models.core import Project
    from app.models.corpus import Post, PostScore

    project_id = uuid.uuid4()
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)

    with sync_session() as session:
        session.add(Project(id=project_id, slug="overview-test", name="Overview"))
        for index in range(10):
            session.add(
                Post(
                    id=f"reddit:{index}",
                    project_id=project_id,
                    native_id=str(index),
                    source="reddit",
                    source_detail="r/news",
                    content_type="post",
                    text_=f"post {index}",
                    author_id=f"reddit:author{index % 3}",
                    author_handle=f"author{index % 3}",
                    timestamp=base + timedelta(hours=index),
                    likes=index,
                    shares=None,  # not measured on this platform
                    replies=0,  # measured zero
                    views=None,
                )
            )
        # Seven of ten scored: four negative, two positive, one neutral.
        labels = ["negative"] * 4 + ["positive"] * 2 + ["neutral"]
        for index, label in enumerate(labels):
            session.add(
                PostScore(
                    post_id=f"reddit:{index}",
                    sentiment=label,
                    sentiment_score=-0.5 if label == "negative" else 0.5,
                    emotion="anger" if label == "negative" else "joy",
                    toxicity=0.8 if index == 0 else 0.1,
                    is_toxic=index == 0,
                    misinfo_likelihood=0.9 if index == 0 else 0.2,
                )
            )
    return project_id


@requires_postgres
def test_engagement_sums_skip_nulls_rather_than_treating_them_as_zero(client, seeded, admin_key):
    """sum(likes) over 0..9 is 45; shares and views contribute nothing at all."""
    response = client.get(
        f"/api/v1/overview/kpis?project_id={seeded}", headers={"X-API-Key": admin_key}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["posts"]["value"] == 10
    # 45 from likes, 0 from the measured-zero replies, nothing from the nulls.
    assert body["engagements"]["value"] == 45
    assert "views" in body["engagements"]["nullable_fields"]
    assert body["authors"]["value"] == 3


@requires_postgres
def test_net_sentiment_agrees_between_the_kpi_and_the_drilldown(client, seeded, admin_key):
    """Two numbers on one screen must not use opposite sign conventions.

    They did, once: the KPI computed negative-minus-positive and the breakdown
    positive-minus-negative, so the same corpus read +27.8 in one place and
    -27.8 in the other.
    """
    headers = {"X-API-Key": admin_key}
    kpi = client.get(f"/api/v1/overview/kpis?project_id={seeded}", headers=headers).json()
    drilldown = client.get(
        f"/api/v1/overview/kpis/sentiment?project_id={seeded}", headers=headers
    ).json()

    # 2 positive, 4 negative, over 7 scored: (2-4)/7 = -28.57
    assert kpi["sentiment"]["value"] == pytest.approx(-28.6, abs=0.1)
    assert drilldown["net_sentiment"] == pytest.approx(-28.57, abs=0.1)
    assert kpi["sentiment"]["value"] == pytest.approx(drilldown["net_sentiment"], abs=0.1)


@requires_postgres
def test_unscored_posts_are_counted_apart_from_the_distribution(client, seeded, admin_key):
    """Percentages are over what was measured; the rest is reported separately."""
    headers = {"X-API-Key": admin_key}
    emotions = client.get(
        f"/api/v1/overview/kpis/emotions?project_id={seeded}", headers=headers
    ).json()

    assert emotions["total_scored"] == 7
    assert emotions["unscored"] == 3
    assert sum(item["pct"] for item in emotions["items"]) == pytest.approx(100.0, abs=0.1)
    # Phase 2 emits `joy`; the UI vocabulary is `happiness`. Translated once,
    # server-side, so two clients cannot disagree about the label.
    happiness = next(i for i in emotions["items"] if i["emotion"] == "happiness")
    assert happiness["count"] == 3


@requires_postgres
def test_a_toxicity_floor_excludes_unscored_posts_rather_than_sweeping_them_in(
    client, seeded, admin_key
):
    """`min_toxicity=0.0` means "measured at 0.0 or above", not "everything"."""
    headers = {"X-API-Key": admin_key}
    unfiltered = client.get(f"/api/v1/posts?project_id={seeded}", headers=headers).json()
    floored = client.get(
        f"/api/v1/posts?project_id={seeded}&min_toxicity=0.0", headers=headers
    ).json()

    assert unfiltered["total"] == 10
    # Only the seven scored posts qualify; the three unscored ones are not
    # "measured as harmless" and must not appear.
    assert floored["total"] == 7
