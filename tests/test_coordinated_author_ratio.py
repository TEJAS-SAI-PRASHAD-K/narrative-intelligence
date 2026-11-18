"""``coordinated_author_ratio`` on the narrative scorecard and the authors KPI.

Fusion reads this ratio and does not read ``bot_like``, so the API has to show
the one it uses. The rule under test is null versus zero: no coordination score
means null, a measured zero stays zero.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from tests.conftest_api import requires_postgres
from tests.test_contract import demo_client, demo_ids  # noqa: F401  (fixtures)

BASE = "/api/v1"


def _scorecard_row(**overrides):
    fields = {
        "priority": "medium",
        "bot_like": None,
        "coordinated_author_ratio": None,
        "anomalous": None,
        "toxicity": None,
        "compass_risk": None,
        "negative_sentiment": None,
        "fusion_score": 42.0,
        "scoring_version": "fusion-test",
        "computed_at": None,
        "components": {"missing": []},
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_stored_null_stays_null_and_stored_zero_stays_zero():
    from app.repositories.narratives import to_scorecard

    assert to_scorecard(_scorecard_row()).coordinated_author_ratio is None
    zero = to_scorecard(_scorecard_row(coordinated_author_ratio=0.0))
    assert zero.coordinated_author_ratio == 0.0
    assert to_scorecard(_scorecard_row(coordinated_author_ratio=0.25)).coordinated_author_ratio == (
        0.25
    )


def test_a_narrative_with_no_scorecard_has_a_null_ratio():
    from app.repositories.narratives import to_scorecard

    card = to_scorecard(None)
    assert card.coordinated_author_ratio is None
    assert card.missing == ["not_scored"]


def test_the_ratio_is_bounded_to_the_unit_interval():
    from pydantic import ValidationError

    from app.schemas.narratives import NarrativeScorecard

    for bad in (-0.01, 1.01):
        with pytest.raises(ValidationError):
            NarrativeScorecard(priority="low", coordinated_author_ratio=bad)


def test_openapi_says_fusion_reads_one_field_and_not_the_other():
    from app.openapi_export import build_schema

    props = build_schema()["components"]["schemas"]["NarrativeScorecard"]["properties"]
    assert "fusion reads" in props["coordinated_author_ratio"]["description"].lower()
    assert "fusion does not read" in props["bot_like"]["description"].lower()
    assert "null" in props["coordinated_author_ratio"]["description"].lower()


def test_a_narrative_with_no_coordination_score_returns_null_not_zero(demo_client, demo_ids):  # noqa: F811
    narrative_id = demo_ids["missing_component_narrative_id"]
    detail = demo_client.get(f"{BASE}/narratives/{narrative_id}").json()
    assert "coordinated_author_ratio" in detail["scorecard"]
    assert detail["scorecard"]["coordinated_author_ratio"] is None

    listing = demo_client.get(f"{BASE}/narratives?project_id={demo_ids['project_id']}").json()
    card = next(i for i in listing["items"] if i["id"] == narrative_id)["scorecard"]
    assert card["coordinated_author_ratio"] is None


def test_scored_narratives_return_a_ratio_in_range(demo_client, demo_ids):  # noqa: F811
    listing = demo_client.get(f"{BASE}/narratives?project_id={demo_ids['project_id']}").json()
    ratios = {
        item["id"]: item["scorecard"]["coordinated_author_ratio"] for item in listing["items"]
    }
    measured = [r for r in ratios.values() if r is not None]
    assert measured, "the fixture must include narratives with a measured ratio"
    assert all(0.0 <= r <= 1.0 for r in measured)
    assert demo_ids["missing_component_narrative_id"] in ratios


def test_the_ratio_on_the_card_is_the_one_in_the_fusion_explanation(demo_client, demo_ids):  # noqa: F811
    for narrative_id in (demo_ids["narrative_id"], demo_ids["missing_component_narrative_id"]):
        card = demo_client.get(f"{BASE}/narratives/{narrative_id}").json()["scorecard"]
        score = demo_client.get(f"{BASE}/narratives/{narrative_id}/score").json()
        coordination = next(c for c in score["components"] if c["name"] == "coordination")
        fusion_input = coordination["inputs"]["coordinated_author_ratio"]
        assert fusion_input == card["coordinated_author_ratio"]


def test_bot_like_is_still_served(demo_client, demo_ids):  # noqa: F811
    detail = demo_client.get(f"{BASE}/narratives/{demo_ids['narrative_id']}").json()
    assert "bot_like" in detail["scorecard"]


def test_the_authors_kpi_carries_the_coordinated_author_ratio(demo_client, demo_ids):  # noqa: F811
    from app.schemas.overview import AuthorsBreakdown

    body = demo_client.get(
        f"{BASE}/overview/kpis/authors?project_id={demo_ids['project_id']}"
    ).json()
    breakdown = AuthorsBreakdown.model_validate(body)
    metric = breakdown.coordinated_author_ratio
    assert metric.value is not None and 0.0 <= metric.value <= 1.0
    assert "fusion" in metric.definition.lower()
    assert "coordination_score" in metric.nullable_fields
    assert metric.sample_size and metric.sample_size > 0
    assert breakdown.bot_like.value is not None, "bot_like stays alongside"


def test_the_authors_kpi_ratio_is_null_when_no_author_is_scored(monkeypatch):
    from app.deps import FilterSpec
    from app.mock import responses
    from app.mock.corpus import corpus

    for author in corpus().authors:
        monkeypatch.setattr(author, "coordination_score", None)
    spec = FilterSpec(project_id=corpus().project_id)
    metric = responses.overview_authors_breakdown(spec).coordinated_author_ratio
    assert metric.value is None
    assert metric.sample_size == 0


@requires_postgres
def test_the_scorecard_column_round_trips_null_and_zero(migrated_db):
    from app.db import sync_session
    from app.models.core import Project
    from app.models.narratives import Narrative, NarrativeScorecard
    from app.repositories.narratives import to_scorecard

    project_id = uuid.uuid4()
    ids = {"null": uuid.uuid4(), "zero": uuid.uuid4()}
    with sync_session() as session:
        session.add(Project(id=project_id, slug="ratio-roundtrip", name="Ratio"))
        for narrative_id in ids.values():
            session.add(Narrative(id=narrative_id, project_id=project_id))
        session.flush()
        session.add(NarrativeScorecard(narrative_id=ids["null"], coordinated_author_ratio=None))
        session.add(NarrativeScorecard(narrative_id=ids["zero"], coordinated_author_ratio=0.0))

    with sync_session() as session:
        null_row = session.get(NarrativeScorecard, ids["null"])
        zero_row = session.get(NarrativeScorecard, ids["zero"])
        assert to_scorecard(null_row).coordinated_author_ratio is None
        assert to_scorecard(zero_row).coordinated_author_ratio == 0.0
