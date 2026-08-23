"""The contract test.

Every route in the API surface is exercised under ``DEMO_MODE=1`` and its
response is validated against the very response model the OpenAPI schema
publishes. This is what makes the tag meaningful: after it, values change and
shapes do not.

Needs no database and no Redis -- DEMO_MODE short-circuits both -- so it runs on
any machine and is the first thing to break if somebody changes a payload shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


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

    with TestClient(create_app()) as client:
        yield client
    get_api_settings.cache_clear()


@pytest.fixture
def demo_ids(demo_client):
    """Real ids from the fixture corpus, so path routes are exercised for real."""
    from app.mock.corpus import corpus

    c = corpus()
    return {
        "project_id": c.project_id,
        "narrative_id": c.narratives[0].id,
        "unsourceable_narrative_id": c.narratives[7].id,
        "missing_component_narrative_id": c.narratives[4].id,
        "author_id": c.authors[0].author_id,
        "post_id": c.posts[0].id,
        "cohort_id": c.cohorts[0]["id"],
        "domain": c.domains[0]["domain"],
    }


def _routes(app) -> list[tuple[str, str]]:
    spec = app.openapi()
    return [
        (method.upper(), path)
        for path, ops in spec["paths"].items()
        for method in ops
        if method in {"get", "post", "patch", "put", "delete"}
    ]


def test_committed_openapi_matches_the_code():
    """The published contract must not drift from what the app serves.

    This is the guard that stops a route rename shipping silently and breaking
    the dashboard at runtime instead of in CI.
    """
    from app.openapi_export import build_schema

    committed = json.loads((REPO_ROOT / "openapi.json").read_text(encoding="utf-8"))
    assert build_schema() == committed, "openapi.json has drifted; run `make openapi`"


def test_every_documented_route_is_reachable(demo_client):
    """No route is documented but unroutable.

    A path parameter shadowed by a later literal segment produces a schema entry
    that 404s forever, and that failure is otherwise invisible until a frontend
    developer reports it.
    """
    documented = set(_routes(demo_client.app))
    assert len(documented) >= 78, f"expected the full surface, found {len(documented)}"
    for _method, path in documented:
        assert path.startswith(("/api/v1/", "/healthz", "/readyz"))


# ---------------------------------------------------------------------------
# every route, exercised
# ---------------------------------------------------------------------------
def test_probes(demo_client):
    from app.schemas.ops import HealthResponse, ReadinessResponse

    HealthResponse.model_validate(demo_client.get("/healthz").json())
    ready = demo_client.get("/readyz")
    ReadinessResponse.model_validate(ready.json())


def test_projects(demo_client, demo_ids):
    from app.schemas.common import PageResponse
    from app.schemas.projects import ProjectOut, SourceHealth, SourceTestResult

    listing = demo_client.get("/api/v1/projects")
    assert listing.status_code == 200
    PageResponse[ProjectOut].model_validate(listing.json())

    project_id = demo_ids["project_id"]
    ProjectOut.model_validate(demo_client.get(f"/api/v1/projects/{project_id}").json())
    ProjectOut.model_validate(
        demo_client.patch(f"/api/v1/projects/{project_id}", json={"name": "renamed"}).json()
    )
    ProjectOut.model_validate(
        demo_client.post("/api/v1/projects", json={"slug": "new-case", "name": "New case"}).json()
    )

    sources = demo_client.get(f"/api/v1/projects/{project_id}/sources")
    assert sources.status_code == 200
    rows = [SourceHealth.model_validate(row) for row in sources.json()]
    # The fixture must include unhappy paths: a setup page where every light is
    # green teaches nobody how the degraded states render.
    assert {row.status for row in rows} & {"quota_exhausted", "skipped"}

    SourceTestResult.model_validate(
        demo_client.post(f"/api/v1/projects/{project_id}/sources/mastodon/test").json()
    )


def test_ingest(demo_client, demo_ids):
    from app.schemas.common import JobAccepted, PageResponse
    from app.schemas.ingest import IngestRunOut

    accepted = demo_client.post(
        "/api/v1/ingest", json={"project_id": demo_ids["project_id"], "mode": "both"}
    )
    assert accepted.status_code == 202
    job = JobAccepted.model_validate(accepted.json())
    assert job.status_url.endswith(job.job_id)

    runs = demo_client.get(f"/api/v1/ingest/runs?project_id={demo_ids['project_id']}")
    page = PageResponse[IngestRunOut].model_validate(runs.json())
    # Every rejected row is accounted for by reason code, not just counted.
    rejecting = [r for r in page.items if r.records_rejected]
    assert rejecting and all(
        sum(r.rejection_reasons.values()) == r.records_rejected for r in rejecting
    )
    IngestRunOut.model_validate(demo_client.get(f"/api/v1/ingest/runs/{page.items[0].id}").json())


def test_overview(demo_client, demo_ids):
    from app.schemas.common import PageResponse
    from app.schemas.overview import (
        AuthorsBreakdown,
        ConceptsResponse,
        EmotionsBreakdown,
        EngagementsBreakdown,
        KpiBundle,
        PostsBreakdown,
        SentimentBreakdown,
        TimeseriesResponse,
    )
    from app.schemas.posts import HighRiskPost

    base = f"?project_id={demo_ids['project_id']}"
    kpis = KpiBundle.model_validate(demo_client.get(f"/api/v1/overview/kpis{base}").json())
    # Every KPI ships its own definition. The `(?)` tooltip is the backend's job.
    for metric in (kpis.posts, kpis.engagements, kpis.authors, kpis.sentiment, kpis.emotions):
        assert metric.definition and len(metric.definition) > 20
    # Engagements must declare that it skipped nulls rather than zeroing them.
    assert "views" in kpis.engagements.nullable_fields

    TimeseriesResponse.model_validate(
        demo_client.get(
            f"/api/v1/overview/timeseries{base}&metric=posts&group_by=platform&interval=1d"
        ).json()
    )
    ConceptsResponse.model_validate(demo_client.get(f"/api/v1/overview/concepts{base}").json())
    PageResponse[HighRiskPost].model_validate(
        demo_client.get(f"/api/v1/overview/high-risk-posts{base}").json()
    )
    PostsBreakdown.model_validate(demo_client.get(f"/api/v1/overview/kpis/posts{base}").json())
    EngagementsBreakdown.model_validate(
        demo_client.get(f"/api/v1/overview/kpis/engagements{base}").json()
    )
    AuthorsBreakdown.model_validate(demo_client.get(f"/api/v1/overview/kpis/authors{base}").json())
    emotions = EmotionsBreakdown.model_validate(
        demo_client.get(f"/api/v1/overview/kpis/emotions{base}").json()
    )
    # Percentages are over what was measured, and the unmeasured are counted
    # separately rather than folded into 'neutral'.
    assert abs(sum(item.pct for item in emotions.items) - 100.0) < 0.5
    assert emotions.unscored > 0
    SentimentBreakdown.model_validate(
        demo_client.get(f"/api/v1/overview/kpis/sentiment{base}").json()
    )


def test_narratives(demo_client, demo_ids):
    from app.schemas.actors import AuthorOut
    from app.schemas.common import JobAccepted, PageResponse
    from app.schemas.narratives import (
        NarrativeCohortShare,
        NarrativeDetail,
        NarrativeFreshness,
        NarrativeScoreExplanation,
        NarrativeSummary,
        NarrativeTimeline,
    )
    from app.schemas.posts import PostOut

    project_id = demo_ids["project_id"]
    narrative_id = demo_ids["narrative_id"]
    base = f"?project_id={project_id}"

    feed = PageResponse[NarrativeSummary].model_validate(
        demo_client.get(f"/api/v1/narratives{base}&sort=priority").json()
    )
    assert feed.items
    # Every AI-written string carries its provenance so the UI never guesses.
    for item in feed.items:
        assert item.title.provenance.generated_by in {"ai", "human", "heuristic"}
        if item.title.provenance.generated_by == "ai":
            assert item.title.provenance.model and item.title.provenance.generated_at
    assert any(item.title.provenance.edited_by_user for item in feed.items)

    NarrativeDetail.model_validate(demo_client.get(f"/api/v1/narratives/{narrative_id}").json())
    NarrativeDetail.model_validate(
        demo_client.patch(
            f"/api/v1/narratives/{narrative_id}", json={"title": "Analyst title"}
        ).json()
    )
    NarrativeDetail.model_validate(
        demo_client.post(
            "/api/v1/narratives",
            json={"project_id": project_id, "title": "Hand-built narrative"},
        ).json()
    )
    NarrativeTimeline.model_validate(
        demo_client.get(
            f"/api/v1/narratives/{narrative_id}/timeline?interval=1d&cross_platform=true"
        ).json()
    )
    PageResponse[PostOut].model_validate(
        demo_client.get(f"/api/v1/narratives/{narrative_id}/posts{base}").json()
    )
    PageResponse[AuthorOut].model_validate(
        demo_client.get(f"/api/v1/narratives/{narrative_id}/authors{base}").json()
    )
    cohorts = demo_client.get(f"/api/v1/narratives/{narrative_id}/cohorts").json()
    [NarrativeCohortShare.model_validate(row) for row in cohorts]

    NarrativeFreshness.model_validate(demo_client.get(f"/api/v1/narratives/freshness{base}").json())
    accepted = demo_client.post(f"/api/v1/narratives/recluster{base}")
    assert accepted.status_code == 202
    JobAccepted.model_validate(accepted.json())

    explanation = NarrativeScoreExplanation.model_validate(
        demo_client.get(f"/api/v1/narratives/{narrative_id}/score").json()
    )
    assert len(explanation.components) == 3
    assert explanation.formula and explanation.normalization and explanation.scoring_version


def test_a_missing_component_renormalizes_rather_than_scoring_zero(demo_client, demo_ids):
    """The acceptance criterion, asserted on the contract fixture.

    Treating "not measured" as "measured zero" would systematically under-flag
    every narrative the deepfake module never saw, which is the same null-versus-
    zero discipline the corpus enforces on engagement metrics.
    """
    from app.schemas.narratives import NarrativeScoreExplanation

    narrative_id = demo_ids["missing_component_narrative_id"]
    explanation = NarrativeScoreExplanation.model_validate(
        demo_client.get(f"/api/v1/narratives/{narrative_id}/score").json()
    )
    assert explanation.missing == ["authenticity"]
    assert explanation.weights_renormalized is True

    authenticity = next(c for c in explanation.components if c.name == "authenticity")
    assert authenticity.value is None, "a missing component must be null, never 0.0"
    assert authenticity.contribution is None

    live = [c for c in explanation.components if c.value is not None]
    assert abs(sum(c.weight for c in live) - 1.0) < 1e-6, "live weights must renormalize to 1"


def test_compass(demo_client, demo_ids):
    from app.schemas.common import JobAccepted
    from app.schemas.compass import CompassContext, CompassFeedbackResult

    narrative_id = demo_ids["narrative_id"]
    context = CompassContext.model_validate(
        demo_client.get(f"/api/v1/narratives/{narrative_id}/compass").json()
    )
    assert context.generated_by == "ai" and context.model and context.prompt_version
    # Mandatory citation: a non-empty context must carry citations with spans.
    assert context.citations
    for citation in context.citations:
        assert citation.url and citation.char_start is not None and citation.char_end is not None

    accepted = demo_client.post(f"/api/v1/narratives/{narrative_id}/compass/regenerate")
    assert accepted.status_code == 202
    JobAccepted.model_validate(accepted.json())

    CompassFeedbackResult.model_validate(
        demo_client.post(
            f"/api/v1/compass/{context.id}/feedback",
            json={"helpful": False, "reason_code": "bad_sources"},
        ).json()
    )


def test_an_unsourceable_claim_persists_no_uncited_paragraph(demo_client, demo_ids):
    """The worst failure this system has, asserted against.

    Persisting an uncited AI fact-check into a misinformation product is not a
    degraded result -- it is the thing the product exists to argue against. The
    honest output is an empty context and an explicit status.
    """
    from app.schemas.compass import CompassContext

    narrative_id = demo_ids["unsourceable_narrative_id"]
    context = CompassContext.model_validate(
        demo_client.get(f"/api/v1/narratives/{narrative_id}/compass").json()
    )
    assert context.verification_status == "insufficient_evidence"
    assert context.context == ""
    assert context.citations == []
    assert context.caution_note


def test_actors(demo_client, demo_ids):
    from app.schemas.actors import (
        AuthorGroupMembersResult,
        AuthorGroupOut,
        AuthorOut,
        AuthorScoreExplanation,
        AuthorTimeline,
        CohortOut,
    )
    from app.schemas.common import PageResponse
    from app.schemas.posts import PostOut

    project_id = demo_ids["project_id"]
    author_id = demo_ids["author_id"]
    base = f"?project_id={project_id}"

    authors = PageResponse[AuthorOut].model_validate(
        demo_client.get(f"/api/v1/authors{base}&sort=bot_score").json()
    )
    assert authors.items
    # An author can be in several cohorts at once. A fixture with one cohort per
    # author would let the frontend get away with rendering a single chip.
    assert any(len(item.cohorts) > 1 for item in authors.items)

    AuthorOut.model_validate(demo_client.get(f"/api/v1/authors/{author_id}").json())
    AuthorScoreExplanation.model_validate(
        demo_client.get(f"/api/v1/authors/{author_id}/score").json()
    )
    AuthorTimeline.model_validate(demo_client.get(f"/api/v1/authors/{author_id}/timeline").json())
    PageResponse[PostOut].model_validate(
        demo_client.get(f"/api/v1/authors/{author_id}/posts{base}").json()
    )

    PageResponse[CohortOut].model_validate(demo_client.get(f"/api/v1/cohorts{base}").json())
    PageResponse[AuthorOut].model_validate(
        demo_client.get(f"/api/v1/cohorts/{demo_ids['cohort_id']}/authors{base}").json()
    )

    PageResponse[AuthorGroupOut].model_validate(
        demo_client.get(f"/api/v1/author-groups{base}").json()
    )
    group = AuthorGroupOut.model_validate(
        demo_client.post(
            "/api/v1/author-groups", json={"project_id": project_id, "name": "Watchlist"}
        ).json()
    )
    result = AuthorGroupMembersResult.model_validate(
        demo_client.post(
            f"/api/v1/author-groups/{group.id}/members",
            json={"author_ids": [author_id, "reddit:does-not-exist"]},
        ).json()
    )
    # Unknown ids are reported, never silently dropped.
    assert result.unknown == ["reddit:does-not-exist"]


def test_drivers(demo_client, demo_ids):
    from app.schemas.drivers import DriversResponse

    base = f"?project_id={demo_ids['project_id']}"
    for tab in ("authors", "hashtags", "urls"):
        response = demo_client.get(f"/api/v1/drivers/{tab}{base}")
        assert response.status_code == 200
        payload = DriversResponse.model_validate(response.json())
        assert payload.definition


def test_domains(demo_client, demo_ids):
    from app.schemas.common import JobAccepted, PageResponse
    from app.schemas.domains import DomainNarrativeLink, DomainOut

    project_id = demo_ids["project_id"]
    base = f"?project_id={project_id}"
    page = PageResponse[DomainOut].model_validate(
        demo_client.get(f"/api/v1/domains{base}&sort=risk").json()
    )
    # An un-enriched domain still carries a risk score from in-corpus signals.
    unenriched = [d for d in page.items if d.enrichment_status == "unavailable"]
    assert unenriched, "the fixture must include a domain WHOIS could not resolve"
    for domain in unenriched:
        assert domain.risk_score is not None
        assert domain.whois_created_at is None
        assert domain.enrichment_detail

    domain = demo_ids["domain"]
    DomainOut.model_validate(demo_client.get(f"/api/v1/domains/{domain}{base}").json())
    links = demo_client.get(f"/api/v1/domains/{domain}/narratives{base}").json()
    [DomainNarrativeLink.model_validate(row) for row in links]

    accepted = demo_client.post(f"/api/v1/domains/{domain}/enrich{base}")
    assert accepted.status_code == 202
    JobAccepted.model_validate(accepted.json())


def test_network(demo_client, demo_ids):
    from app.schemas.common import JobAccepted, PageResponse
    from app.schemas.network import ComparisonOut, GraphResponse

    project_id = demo_ids["project_id"]
    base = f"?project_id={project_id}"

    graph = GraphResponse.model_validate(
        demo_client.get(f"/api/v1/network/graph{base}&bucket=12h").json()
    )
    assert graph.stats.node_count == len(graph.nodes)
    # Positions are precomputed server-side; the browser is never handed a
    # position-less graph and asked to lay it out.
    assert all(node.x is not None and node.y is not None for node in graph.nodes)
    assert graph.legend and all(entry.definition for entry in graph.legend)

    JobAccepted.model_validate(
        demo_client.post("/api/v1/network/layout", json={"project_id": project_id}).json()
    )
    comparisons = PageResponse[ComparisonOut].model_validate(
        demo_client.get(f"/api/v1/comparisons{base}").json()
    )
    comparison = comparisons.items[0]
    ComparisonOut.model_validate(demo_client.get(f"/api/v1/comparisons/{comparison.id}").json())
    ComparisonOut.model_validate(
        demo_client.post(
            "/api/v1/comparisons",
            json={
                "project_id": project_id,
                "name": "Three-way",
                "narrative_ids": comparison.narrative_ids[:2],
            },
        ).json()
    )
    assert demo_client.delete(f"/api/v1/comparisons/{comparison.id}").status_code == 200


def test_graph_truncation_is_never_silent(demo_client, demo_ids):
    """A graph that quietly drops its periphery makes a coordinated cluster look
    more isolated than it is -- the wrong error for this product to make."""
    from app.schemas.network import GraphResponse

    base = f"?project_id={demo_ids['project_id']}"
    graph = GraphResponse.model_validate(
        demo_client.get(f"/api/v1/network/graph{base}&max_nodes=25").json()
    )
    assert graph.stats.truncated is True
    assert graph.stats.truncation is not None
    assert graph.stats.truncation["applied_max_nodes"] == 25
    assert graph.stats.truncation["dropped_nodes"] > 0
    assert graph.stats.truncation["rule"] == "descending node degree"
    assert len(graph.nodes) == 25


def test_posts(demo_client, demo_ids):
    from app.schemas.common import PageResponse
    from app.schemas.posts import PostOut, SimilarResponse, ThreadResponse

    project_id = demo_ids["project_id"]
    post_id = demo_ids["post_id"]
    base = f"?project_id={project_id}"

    page = PageResponse[PostOut].model_validate(demo_client.get(f"/api/v1/posts{base}").json())
    assert page.filters_applied["project_id"] == project_id
    # Null engagement survives to the wire as null, not as zero.
    assert any(item.engagement.views is None for item in page.items)

    PostOut.model_validate(demo_client.get(f"/api/v1/posts/{post_id}").json())
    ThreadResponse.model_validate(demo_client.get(f"/api/v1/posts/{post_id}/thread").json())
    PageResponse[PostOut].model_validate(
        demo_client.get(f"/api/v1/posts/search{base}&q=ballot").json()
    )

    similar = SimilarResponse.model_validate(
        demo_client.post(
            "/api/v1/posts/similar",
            json={"project_id": project_id, "post_id": post_id, "limit": 5},
        ).json()
    )
    assert similar.metric == "cosine"
    assert len(similar.items) == 5
    # Ranked by descending similarity, and distance is consistent with it.
    scores = [item.similarity for item in similar.items]
    assert scores == sorted(scores, reverse=True)
    for item in similar.items:
        assert abs(item.similarity + item.distance - 1.0) < 1e-6


def test_similar_rejects_an_ambiguous_query(demo_client, demo_ids):
    response = demo_client.post(
        "/api/v1/posts/similar",
        json={"project_id": demo_ids["project_id"], "post_id": demo_ids["post_id"], "text": "x"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "ambiguous_query"


def test_media(demo_client):
    from app.schemas.common import JobAccepted, PageResponse
    from app.schemas.media import MediaCheckOut

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 128
    accepted = demo_client.post(
        "/api/v1/media/check", files={"file": ("frame.png", png, "image/png")}
    )
    assert accepted.status_code == 202
    job = JobAccepted.model_validate(accepted.json())

    check = MediaCheckOut.model_validate(
        demo_client.get(f"/api/v1/media/check/{job.job_id}").json()
    )
    # A bare number is not an acceptable answer to "is this video fake".
    assert check.explanation and len(check.explanation) > 80
    assert check.limitations
    assert check.retention["retention_hours"]

    PageResponse[MediaCheckOut].model_validate(demo_client.get("/api/v1/media/checks").json())


def test_media_rejects_a_file_by_magic_bytes_not_extension(demo_client):
    """The extension and the Content-Type header are both client-supplied."""
    response = demo_client.post(
        "/api/v1/media/check",
        files={"file": ("payload.png", b"#!/bin/sh\nrm -rf /\n", "image/png")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_alerts(demo_client, demo_ids):
    from app.schemas.alerts import AlertAcknowledged, AlertOut, AlertRuleOut
    from app.schemas.common import PageResponse

    project_id = demo_ids["project_id"]
    base = f"?project_id={project_id}"

    alerts = PageResponse[AlertOut].model_validate(demo_client.get(f"/api/v1/alerts{base}").json())
    alert = alerts.items[0]
    # An alert is self-explaining: it carries the observed value and the rule.
    assert {"metric", "op", "threshold", "observed"} <= set(alert.payload)
    AlertAcknowledged.model_validate(
        demo_client.post(f"/api/v1/alerts/{alert.id}/acknowledge").json()
    )

    rules = PageResponse[AlertRuleOut].model_validate(
        demo_client.get(f"/api/v1/alert-rules{base}").json()
    )
    rule = rules.items[0]
    AlertRuleOut.model_validate(
        demo_client.post(
            "/api/v1/alert-rules",
            json={
                "project_id": project_id,
                "name": "New rule",
                "condition": {"metric": "fusion_score", "op": ">", "value": 80},
            },
        ).json()
    )
    AlertRuleOut.model_validate(
        demo_client.patch(f"/api/v1/alert-rules/{rule.id}", json={"enabled": False}).json()
    )
    assert demo_client.delete(f"/api/v1/alert-rules/{rule.id}").status_code == 200


def test_a_rule_with_no_channels_is_rejected(demo_client, demo_ids):
    response = demo_client.post(
        "/api/v1/alert-rules",
        json={
            "project_id": demo_ids["project_id"],
            "name": "Silent rule",
            "condition": {"metric": "fusion_score", "op": ">", "value": 80},
            "channels": [],
        },
    )
    assert response.status_code == 422


def test_reports(demo_client, demo_ids):
    from app.schemas.common import JobAccepted, PageResponse
    from app.schemas.reports import ReportOut

    project_id = demo_ids["project_id"]
    accepted = demo_client.post(
        "/api/v1/reports",
        json={"project_id": project_id, "template": "exec_summary", "format": "pdf"},
    )
    assert accepted.status_code == 202
    JobAccepted.model_validate(accepted.json())

    page = PageResponse[ReportOut].model_validate(
        demo_client.get(f"/api/v1/reports?project_id={project_id}").json()
    )
    download = demo_client.get(f"/api/v1/reports/{page.items[0].id}/download")
    assert download.status_code == 200
    assert "attachment" in download.headers["content-disposition"]


def test_jobs(demo_client):
    from app.schemas.common import PageResponse
    from app.schemas.jobs import JobCancelled, JobOut

    page = PageResponse[JobOut].model_validate(demo_client.get("/api/v1/jobs").json())
    job = page.items[0]
    JobOut.model_validate(demo_client.get(f"/api/v1/jobs/{job.id}").json())
    JobCancelled.model_validate(demo_client.post(f"/api/v1/jobs/{job.id}/cancel").json())


# ---------------------------------------------------------------------------
# cross-cutting conventions
# ---------------------------------------------------------------------------
def test_every_list_response_echoes_the_filters_that_produced_it(demo_client, demo_ids):
    """A surprising number is only debuggable if its query travels with it."""
    project_id = demo_ids["project_id"]
    for path in (
        f"/api/v1/posts?project_id={project_id}&min_toxicity=0.5",
        f"/api/v1/narratives?project_id={project_id}",
        f"/api/v1/authors?project_id={project_id}&is_bot_like=true",
        f"/api/v1/domains?project_id={project_id}",
    ):
        payload = demo_client.get(path).json()
        assert payload["filters_applied"]["project_id"] == project_id
        assert "include_shared" in payload["filters_applied"]


def test_cursor_pagination_walks_without_repeating(demo_client, demo_ids):
    project_id = demo_ids["project_id"]
    first = demo_client.get(f"/api/v1/posts?project_id={project_id}&limit=10").json()
    assert len(first["items"]) == 10
    assert first["next_cursor"]

    second = demo_client.get(
        f"/api/v1/posts?project_id={project_id}&limit=10&cursor={first['next_cursor']}"
    ).json()
    assert not {i["id"] for i in first["items"]} & {i["id"] for i in second["items"]}


def test_limit_is_capped(demo_client, demo_ids):
    response = demo_client.get(f"/api/v1/posts?project_id={demo_ids['project_id']}&limit=5000")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_errors_share_one_envelope(demo_client, demo_ids):
    response = demo_client.get(f"/api/v1/narratives/{demo_ids['project_id']}")
    assert response.status_code == 404
    error = response.json()["error"]
    assert set(error) == {"code", "message", "detail", "request_id"}
    assert error["code"] == "narrative_not_found"
    assert error["request_id"]


def test_a_missing_required_project_id_is_a_422_not_a_500(demo_client):
    response = demo_client.get("/api/v1/posts")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
