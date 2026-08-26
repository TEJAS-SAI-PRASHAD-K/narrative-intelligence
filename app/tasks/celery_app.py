"""The Celery application.

Queue design is the load-bearing decision here. Four queues, split by what the
work *contends* for rather than by feature:

* ``io``   -- network-bound: Phase 1 adapters, WHOIS. Many can run concurrently.
* ``cpu``  -- scoring, graph building, report rendering.
* ``gpu``  -- embedding, clustering, deepfake inference. Serialised in practice.
* ``llm``  -- summarization and Compass. Rate-limited by the provider, not by us.

The point is isolation: a forty-minute deepfake job must not be able to starve
fifteen-minute alert evaluation. With one queue it can, and the first time it
happens the alerts are simply late and nobody notices they were late.

Every task is idempotent and resumable. ``acks_late`` means a task killed
mid-flight is redelivered rather than lost, which is only safe *because* they
are idempotent -- the two settings are a pair and neither makes sense alone.
"""

from __future__ import annotations

import logging

from celery import Celery
from celery.signals import setup_logging as celery_setup_logging

from app.config import get_api_settings
from app.models import import_all_models

log = logging.getLogger(__name__)

settings = get_api_settings()

# Import every model before any task runs. A task that touches one model module
# in isolation gets a NoReferencedTableError the moment SQLAlchemy tries to
# resolve a foreign key into a table nobody imported -- a failure that reads
# like a schema bug and is really an import-order bug. Doing it once here makes
# it impossible for a task to hit.
import_all_models()

celery = Celery(
    "narrative_intelligence",
    broker=settings.celery_broker_url or settings.redis_url,
    backend=settings.celery_result_backend or settings.redis_url,
    include=[
        "app.tasks.ingest",
        "app.tasks.nlp",
        "app.tasks.scoring",
        "app.tasks.graph",
        "app.tasks.compass",
        "app.tasks.media",
        "app.tasks.domains",
        "app.tasks.alerts",
        "app.tasks.reports",
        "app.tasks.maintenance",
    ],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Never pass an ORM object or a DataFrame through the broker. JSON-only
    # serialization makes that a startup error instead of a pickle that works
    # until the worker and the API are on different code versions.
    task_track_started=True,
    task_acks_late=True,
    # With acks_late, a worker that dies mid-task must not also hold a reserved
    # queue of tasks nobody else can see. Prefetching one keeps redelivery fast.
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    result_expires=60 * 60 * 24 * 3,
    broker_connection_retry_on_startup=True,
    task_default_queue="cpu",
    task_routes={
        "ingest.*": {"queue": "io"},
        "domain.*": {"queue": "io"},
        "nlp.embed": {"queue": "gpu"},
        "nlp.cluster": {"queue": "gpu"},
        "media.*": {"queue": "gpu"},
        "nlp.summarize": {"queue": "llm"},
        "compass.*": {"queue": "llm"},
        "score.*": {"queue": "cpu"},
        "graph.*": {"queue": "cpu"},
        "alerts.*": {"queue": "cpu"},
        "reports.*": {"queue": "cpu"},
        "maint.*": {"queue": "cpu"},
    },
    # Per-queue time limits. A task with no ceiling is a worker slot that can be
    # lost forever to one wedged HTTP call.
    task_annotations={
        "*": {"max_retries": 3, "default_retry_delay": 30},
        "media.deepfake": {"time_limit": 3600, "soft_time_limit": 3300},
        "nlp.embed": {"time_limit": 7200, "soft_time_limit": 7000},
        "nlp.cluster": {"time_limit": 3600, "soft_time_limit": 3400},
        "ingest.fetch": {"time_limit": 3600, "soft_time_limit": 3400},
        "compass.generate": {"time_limit": 600, "soft_time_limit": 540},
        "alerts.evaluate": {"time_limit": 300, "soft_time_limit": 270},
    },
    beat_schedule={
        "recluster-weekly": {
            "task": "nlp.cluster",
            "schedule": 7 * 24 * 3600.0,
            "kwargs": {"project_id": None, "all_projects": True},
            "options": {"queue": "gpu"},
        },
        "rescore-daily": {
            "task": "score.fusion",
            "schedule": 24 * 3600.0,
            "kwargs": {"project_id": None, "all_projects": True},
            "options": {"queue": "cpu"},
        },
        "evaluate-alerts": {
            "task": "alerts.evaluate",
            "schedule": 15 * 60.0,
            "options": {"queue": "cpu"},
        },
        "refresh-graph": {
            "task": "graph.build_edges",
            "schedule": 6 * 3600.0,
            "kwargs": {"project_id": None, "all_projects": True},
            "options": {"queue": "cpu"},
        },
        "enrich-domains": {
            "task": "domain.enrich",
            "schedule": 3600.0,
            "kwargs": {"batch": True},
            "options": {"queue": "io"},
        },
        # Uploaded media is imagery of real people. This is the task that makes
        # the retention promise in the API real rather than aspirational.
        "purge-uploads": {
            "task": "maint.purge_uploads",
            "schedule": 3600.0,
            "options": {"queue": "cpu"},
        },
    },
)


@celery_setup_logging.connect
def _configure_logging(**_kwargs):
    """Use the application's log format in the worker too.

    Celery replaces the root logger by default, which means every log line from
    app code inside a task comes out in a different format from the same line
    logged by the API. Grepping across both is then needlessly hard.
    """
    from app.main import configure_logging

    configure_logging(get_api_settings().log_level)
