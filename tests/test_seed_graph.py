"""The seed graph step runs in-process: no broker, and both graph tasks still run."""

from __future__ import annotations

import uuid

import pytest

from tests.conftest_api import requires_postgres


@pytest.fixture
def project(migrated_db):
    from app.db import sync_session
    from app.models.core import Project

    with sync_session() as session:
        session.add(Project(id=uuid.uuid4(), slug="seed-graph", name="Seed graph"))
    return "seed-graph"


def _forbid_broker(monkeypatch):
    """Make any broker contact raise, the way a missing Redis does."""
    from app.tasks.celery_app import celery

    def refuse(*args, **kwargs):
        raise ConnectionRefusedError("[Errno 111] Connection refused")

    monkeypatch.setattr(celery, "send_task", refuse)
    monkeypatch.setattr(celery, "connection_for_write", refuse)
    return celery


def _spy(monkeypatch, task) -> list[dict]:
    calls: list[dict] = []
    original = task.run

    def run(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(task, "run", run)
    return calls


@requires_postgres
def test_seed_graph_step_runs_edges_and_communities_without_a_broker(project, monkeypatch):
    from sqlalchemy import select

    from app.db import sync_session
    from app.etl.seed import _build_graph
    from app.models.core import Project
    from app.tasks.graph import build_edges, communities

    _forbid_broker(monkeypatch)
    edge_calls = _spy(monkeypatch, build_edges)
    community_calls = _spy(monkeypatch, communities)

    with sync_session() as session:
        project_id = session.execute(select(Project.id).where(Project.slug == project)).scalar_one()
        result = _build_graph(session, project_id, project)

    assert len(edge_calls) == 1 and len(community_calls) == 1
    assert "written" in result["edges"]
    assert result["edges"]["communities_dispatched"] is False
    # An empty graph is a counted skip with a reason, not a broker error.
    assert "skipped" in result["communities"] or "written" in result["communities"]


@requires_postgres
def test_worker_path_still_chains_communities(project, monkeypatch):
    from app.tasks.celery_app import celery
    from app.tasks.graph import build_edges

    sent: list[tuple] = []
    monkeypatch.setattr(celery, "send_task", lambda *a, **k: sent.append((a, k)))

    result = build_edges.run(job_id=None, project_id=project)

    assert result["communities_dispatched"] is True
    assert sent == [(("graph.communities",), {"kwargs": {"project_id": project}, "queue": "cpu"})]
