"""model_versions reaches the API from the loader and from the live scorer.

The Postgres tests run the real loader and the real upsert, then read the post back through
the route, so a stale or coerced map shows up on the wire.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.conftest_api import requires_postgres

LAYA_STAMP = "v0.2.1+finetuned:multi-v0.2.1+isotonic@5c0de1ab"
MAP_TYPE = pa.map_(pa.string(), pa.string())


@pytest.fixture
def two_posts(migrated_db):
    from app.db import sync_session
    from app.models.core import Project
    from app.models.corpus import Post

    project_id = uuid.uuid4()
    with sync_session() as session:
        session.add(Project(id=project_id, slug="versions-project", name="Versions"))
        session.flush()
        for name, lang in (("hindi", "hi"), ("bare", "en")):
            session.add(
                Post(
                    id=f"news:{name}",
                    project_id=project_id,
                    native_id=name,
                    source="news",
                    source_detail="example.org",
                    content_type="article",
                    text_=f"post {name}",
                    lang=lang,
                    author_id=f"news:{name}",
                    timestamp=datetime(2026, 5, 1, tzinfo=UTC),
                )
            )
    return project_id


def _get(client, key, post_id):
    response = client.get(
        f"/api/v1/posts/{post_id}",
        params={"project_id": "versions-project"},
        headers={"X-API-Key": key},
    )
    assert response.status_code == 200, response.text
    return response.json()["scores"]


@requires_postgres
def test_the_loader_carries_a_laya_map_and_an_empty_map_reads_as_null(
    client, read_key, two_posts, tmp_path
):
    from app.db import sync_session
    from app.etl.scored_loader import load_post_scores

    table = pa.table(
        {
            "record_id": ["news:hindi", "news:bare"],
            "misinfo_prob": [0.7, None],
            "model_versions": pa.array([[("laya", LAYA_STAMP)], []], type=MAP_TYPE),
        }
    )
    target = tmp_path / "record_scores"
    target.mkdir()
    pq.write_table(table, target / "part.parquet")

    with sync_session() as session:
        assert load_post_scores(session, tmp_path, two_posts) == 2

    hindi = _get(client, read_key, "news:hindi")
    assert hindi["model_versions"] == {"laya": LAYA_STAMP}
    assert _get(client, read_key, "news:bare")["model_versions"] is None


@requires_postgres
def test_a_live_rescore_replaces_a_stale_map(client, read_key, two_posts):
    from sqlalchemy import text

    from app.db import sync_session
    from app.tasks.scoring import _UPSERT_POST_SCORES, _score_row

    stale = {"misinfo": "v0.2.0+3f9a1c2e", "toxicity": "v0.0.1"}
    with sync_session() as session:
        for post_id in ("news:hindi", "news:bare"):
            session.execute(
                text(
                    "INSERT INTO post_scores (post_id, skip_reasons, model_versions) "
                    "VALUES (:p, '{}', cast(:m AS jsonb))"
                ),
                {"p": post_id, "m": json.dumps(stale)},
            )
        rows = [
            _score_row("news:hindi", {"toxicity": 0.1}, "phase2:x", {"toxicity": "v0.1.0"}),
            _score_row("news:bare", {"skip_reason": "text_too_short"}, "phase2:x", {}),
        ]
        session.execute(text(_UPSERT_POST_SCORES), rows)

    assert _get(client, read_key, "news:hindi")["model_versions"] == {"toxicity": "v0.1.0"}
    assert _get(client, read_key, "news:bare")["model_versions"] is None


def test_the_live_score_row_stamps_only_modules_that_produced_a_value():
    from app.tasks.scoring import _UPSERT_POST_SCORES, _score_row

    versions = {"toxicity": "v1", "sentiment": "v2", "emotion": "v3", "anomaly": "v4"}
    row = _score_row("p", {"toxicity": 0.0, "sentiment": "neutral"}, "phase2:x", versions)
    assert json.loads(row["model_versions"]) == {"toxicity": "v1", "sentiment": "v2"}
    assert _score_row("p", {}, "phase2:x", versions)["model_versions"] is None
    assert _score_row("p", {"toxicity": 0.2}, "phase2:x")["model_versions"] is None
    assert "model_versions = EXCLUDED.model_versions" in _UPSERT_POST_SCORES
