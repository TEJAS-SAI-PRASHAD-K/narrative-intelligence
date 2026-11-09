"""The author-level fusion signals, read back from Postgres.

The ratio is a SQL expression, so it is tested against real rows. The cases
that matter are the denominator (scored authors only), the strict boundary and
NULL when no author has a score.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from tests.conftest_api import requires_postgres

THRESHOLD = 0.5


@pytest.fixture
def seeded(migrated_db):
    """Return a function that seeds one narrative and returns its id.

    Each author is ``(coordination_score, bot_score)``.
    """
    from app.db import sync_session
    from app.models.actors import Author
    from app.models.core import Project
    from app.models.corpus import Post
    from app.models.narratives import Narrative, NarrativePost

    project_id = uuid.uuid4()
    with sync_session() as session:
        session.add(Project(id=project_id, slug="signals", name="Signals"))

    def seed(authors: list[tuple[float | None, float | None]]) -> str:
        narrative_id = uuid.uuid4()
        tag = narrative_id.hex[:8]
        with sync_session() as session:
            session.add(Narrative(id=narrative_id, project_id=project_id))
            session.flush()
            for index, (coordination, bot) in enumerate(authors):
                author_id = f"reddit:{tag}-{index}"
                post_id = f"reddit:{tag}-p{index}"
                session.add(
                    Author(
                        author_id=author_id,
                        project_id=project_id,
                        source="reddit",
                        handle=f"user{index}",
                        coordination_score=coordination,
                        bot_score=bot,
                    )
                )
                session.add(
                    Post(
                        id=post_id,
                        project_id=project_id,
                        native_id=f"{tag}-p{index}",
                        source="reddit",
                        source_detail="news",
                        content_type="post",
                        text_="text",
                        author_id=author_id,
                        timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
                    )
                )
                session.flush()
                session.add(NarrativePost(narrative_id=narrative_id, post_id=post_id))
        return str(narrative_id)

    seed.project_id = project_id
    return seed


def _collect(project_id, config=None):
    from app.db import sync_session
    from app.scoring import signals
    from app.scoring.fusion import load_config

    with sync_session() as session:
        return signals.collect(session, project_id, config or load_config())


def test_the_threshold_comes_from_the_fusion_config():
    from app.scoring.fusion import load_config

    assert load_config()["thresholds"]["coordination_score_threshold"] == THRESHOLD


@requires_postgres
def test_the_ratio_is_over_scored_authors_only(seeded):
    """Two of three scored authors are above the threshold. Unscored authors do not count."""
    narrative = seeded([(0.9, None), (0.8, None), (0.1, None), (None, None), (None, None)])
    item = _collect(seeded.project_id)[narrative]

    assert item.coordinated_author_ratio == pytest.approx(2 / 3)
    assert item.inputs["member_authors"] == 5
    assert item.inputs["coordination_scored_authors"] == 3


@requires_postgres
def test_a_score_equal_to_the_threshold_is_not_above_it(seeded):
    narrative = seeded([(THRESHOLD, None), (0.5000001, None), (0.4999999, None)])
    item = _collect(seeded.project_id)[narrative]

    assert item.coordinated_author_ratio == pytest.approx(1 / 3)


@requires_postgres
def test_no_scored_author_gives_null_not_zero(seeded):
    """Bot scores alone do not produce a coordination signal."""
    narrative = seeded([(None, 0.9), (None, 0.1)])
    item = _collect(seeded.project_id)[narrative]

    assert item.coordinated_author_ratio is None
    assert item.inputs["coordination_scored_authors"] == 0
    # The bot ratio is still collected and stored.
    assert item.bot_like_ratio == pytest.approx(0.5)


@requires_postgres
def test_all_scored_authors_below_the_threshold_is_a_measured_zero(seeded):
    narrative = seeded([(0.1, None), (0.2, None)])
    item = _collect(seeded.project_id)[narrative]

    assert item.coordinated_author_ratio == 0.0


@requires_postgres
def test_the_threshold_is_read_from_config(seeded):
    from app.scoring.fusion import load_config

    narrative = seeded([(0.3, None), (0.6, None)])
    config = load_config()
    config["thresholds"] = {**config["thresholds"], "coordination_score_threshold": 0.2}
    item = _collect(seeded.project_id, config)[narrative]

    assert item.coordinated_author_ratio == 1.0


@requires_postgres
def test_populations_rank_the_new_signal_and_not_the_bot_ratio(seeded):
    from app.scoring import signals

    seeded([(0.9, 0.9)])
    population = signals.populations(_collect(seeded.project_id))

    assert "coordinated_author_ratio" in population
    assert "bot_like_ratio" not in population
