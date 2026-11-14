"""Chart and drilldown series report unmeasured engagement as null, not 0.

Needs Postgres. The per-row COALESCE that used to sit inside these SUMs made an
all-NULL bucket read as a measured zero.
"""

from __future__ import annotations

import uuid

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
