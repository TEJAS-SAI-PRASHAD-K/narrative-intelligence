"""Phase 2/3's scored Parquet -> Postgres.

Phase 2/3 already ran every model over the corpus and committed the results
under ``data/scored/``. Those tables are a first-class input to Phase 4, not a
fallback: serving them means the API returns real model output with no GPU, no
checkpoint mounted and no inference on the request path.

The Celery scoring tasks (``score.posts``, ``score.authors``) exist to score
*new* posts as they arrive. This loader exists to bring the existing corpus's
scores in. Both write the same tables, and both are idempotent upserts keyed on
the model versions, so running one after the other is safe in either order.

Phase 2/3's column names differ from Phase 4's in several places, and the
mapping is written out explicitly below rather than inferred, because a silently
mismapped column produces a number that is wrong but plausible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


@dataclass
class ScoredLoadReport:
    post_scores: int = 0
    author_scores: int = 0
    narratives: int = 0
    narrative_posts: int = 0
    network_edges: int = 0
    media_checks: int = 0
    embeddings: int = 0
    skipped: dict[str, str] = field(default_factory=dict)


def _read(scored_dir: Path, table: str) -> list[dict[str, Any]] | None:
    """Read one scored table, or None when Phase 2/3 never produced it."""
    import pyarrow.dataset as ds

    path = scored_dir / table
    if not path.exists():
        return None
    try:
        return ds.dataset(str(path), format="parquet", partitioning="hive").to_table().to_pylist()
    except Exception as exc:  # pragma: no cover - a corrupt table is not fatal
        log.error("could not read data/scored/%s: %s", table, exc)
        return None


def _bulk_upsert(session: Session, sql: str, rows: list[dict[str, Any]], *, chunk: int = 5000):
    """Chunked executemany.

    Chunked because a single 4,000-row executemany builds one enormous
    parameter list and psycopg pipelines it as one round trip that either
    entirely succeeds or entirely fails; at chunk boundaries a failure names a
    range instead of "the load".
    """
    total = 0
    for start in range(0, len(rows), chunk):
        batch = rows[start : start + chunk]
        session.execute(text(sql), batch)
        total += len(batch)
    return total


def load_post_scores(session: Session, scored_dir: Path, project_id: uuid.UUID) -> int:
    """``record_scores`` -> ``post_scores``.

    Phase 2 stores ``emotion`` as a struct of seven float scores; Phase 4 stores
    the dominant label in ``emotion`` and the whole distribution in
    ``emotion_scores``. The dominant label is derived here, once, rather than in
    every query that needs it.
    """
    rows = _read(scored_dir, "record_scores")
    if not rows:
        return 0

    payload = []
    for row in rows:
        emotion_scores = row.get("emotion") or {}
        if isinstance(emotion_scores, dict) and emotion_scores:
            usable = {k: v for k, v in emotion_scores.items() if v is not None}
            dominant = max(usable, key=usable.get) if usable else None
        else:
            emotion_scores, dominant = {}, None

        toxicity = row.get("toxicity")
        anomaly = row.get("anomaly_score")
        payload.append(
            {
                "post_id": row["record_id"],
                "toxicity": toxicity,
                # Thresholds live with the scores rather than being reapplied by
                # every consumer. 0.6 is Phase 2's documented operating point.
                "is_toxic": None if toxicity is None else toxicity > 0.6,
                "anomaly": anomaly,
                "is_anomalous": None if anomaly is None else anomaly > 0.7,
                "misinfo_likelihood": row.get("misinfo_prob"),
                "stance": row.get("stance"),
                "stance_confidence": row.get("stance_conf"),
                "sentiment": row.get("sentiment"),
                "sentiment_score": row.get("sentiment_score"),
                "emotion": dominant,
                "emotion_scores": json.dumps(emotion_scores) if emotion_scores else None,
                "skip_reasons": list(row.get("skip_reasons") or ()),
                "model_versions": json.dumps(dict(row.get("model_versions") or {})),
                "scoring_version": _version_of(row.get("model_versions")),
                "scored_at": row.get("scored_at"),
            }
        )

    sql = """
    INSERT INTO post_scores (
        post_id, toxicity, is_toxic, anomaly, is_anomalous, misinfo_likelihood,
        stance, stance_confidence, sentiment, sentiment_score, emotion, emotion_scores,
        skip_reasons, model_versions, scoring_version, scored_at
    )
    -- Every parameter is cast explicitly. INSERT ... SELECT does not give
    -- Postgres a target column to infer a placeholder's type from, and a
    -- parameter that appears in both the SELECT list and a WHERE comparison
    -- fails with "inconsistent types deduced" rather than picking one.
    SELECT
        cast(:post_id AS text), cast(:toxicity AS real), cast(:is_toxic AS boolean),
        cast(:anomaly AS real), cast(:is_anomalous AS boolean),
        cast(:misinfo_likelihood AS real), cast(:stance AS text),
        cast(:stance_confidence AS real), cast(:sentiment AS text),
        cast(:sentiment_score AS real), cast(:emotion AS text),
        cast(:emotion_scores AS jsonb), cast(:skip_reasons AS varchar[]),
        cast(:model_versions AS jsonb), cast(:scoring_version AS text),
        cast(:scored_at AS timestamptz)
    -- Scores for a post that is not in this project are silently irrelevant,
    -- not an error: data/scored/ covers the whole corpus and a project may hold
    -- a subset of it. The EXISTS makes that explicit instead of relying on a
    -- foreign key violation to filter.
    WHERE EXISTS (
        SELECT 1 FROM posts
        WHERE id = cast(:post_id AS text) AND project_id = cast(:project_id AS uuid)
    )
    ON CONFLICT (post_id) DO UPDATE SET
        toxicity = EXCLUDED.toxicity,
        is_toxic = EXCLUDED.is_toxic,
        anomaly = EXCLUDED.anomaly,
        is_anomalous = EXCLUDED.is_anomalous,
        misinfo_likelihood = EXCLUDED.misinfo_likelihood,
        stance = EXCLUDED.stance,
        stance_confidence = EXCLUDED.stance_confidence,
        sentiment = EXCLUDED.sentiment,
        sentiment_score = EXCLUDED.sentiment_score,
        emotion = EXCLUDED.emotion,
        emotion_scores = EXCLUDED.emotion_scores,
        skip_reasons = EXCLUDED.skip_reasons,
        model_versions = EXCLUDED.model_versions,
        scoring_version = EXCLUDED.scoring_version,
        scored_at = EXCLUDED.scored_at
    """
    for row in payload:
        row["project_id"] = project_id
    _bulk_upsert(session, sql, payload)
    loaded = session.execute(
        text(
            "SELECT count(*) FROM post_scores s JOIN posts p ON p.id = s.post_id "
            "WHERE p.project_id = :p"
        ),
        {"p": project_id},
    ).scalar()
    log.info("post_scores: %d rows in this project", loaded)
    return loaded or 0


def load_author_scores(session: Session, scored_dir: Path, project_id: uuid.UUID) -> int:
    """``author_scores`` -> columns on ``authors``.

    An UPDATE, not an upsert: an author row must already exist from the corpus
    roll-up. A score for an author with no posts in this project is a score for
    somebody else's corpus and is dropped, which the row count reflects.
    """
    rows = _read(scored_dir, "author_scores")
    if not rows:
        return 0

    payload = [
        {
            "author_id": row["author_id"],
            "bot_score": row.get("bot_prob"),
            "coordination_score": row.get("coordination_score"),
            "anomalous_score": row.get("anomalous"),
            "toxicity_score": row.get("toxicity_mean"),
            "dominant_sentiment": row.get("dominant_sentiment"),
            "dominant_emotion": row.get("dominant_emotion"),
            "community_id": row.get("community_id"),
            "community_size": row.get("community_size"),
            "skip_reasons": list(row.get("skip_reasons") or ()),
            # The per-feature attribution Phase 2 computed with SHAP. This is
            # what /authors/{id}/score serves, and it is the difference between
            # an explainable score and the black box this product argues against.
            "score_components": json.dumps(
                {
                    "top_features": [
                        {"name": f.get("name"), "contribution": f.get("contribution")}
                        for f in (row.get("bot_top_features") or [])
                    ],
                    "narratives_touched": list(row.get("narratives_touched") or ()),
                },
                default=str,
            ),
            "scoring_version": _version_of(row.get("model_versions")),
            "project_id": project_id,
        }
        for row in rows
    ]

    # COALESCE on every score, and it is load-bearing. Phase 2's author_scores
    # covers a subset of the corpus and carries NULL for the rest; a plain
    # assignment would blank a score that `score.authors` had just computed with
    # a live model. Importing the committed scores must add to what is known,
    # never erase it.
    #
    # The non-score columns (community, dominant sentiment) are assigned
    # directly: those come only from Phase 2, so there is nothing to preserve.
    sql = """
    UPDATE authors SET
        bot_score = COALESCE(:bot_score, authors.bot_score),
        coordination_score = COALESCE(:coordination_score, authors.coordination_score),
        anomalous_score = COALESCE(:anomalous_score, authors.anomalous_score),
        toxicity_score = COALESCE(:toxicity_score, authors.toxicity_score),
        dominant_sentiment = COALESCE(:dominant_sentiment, authors.dominant_sentiment),
        dominant_emotion = COALESCE(:dominant_emotion, authors.dominant_emotion),
        community_id = COALESCE(:community_id, authors.community_id),
        community_size = COALESCE(:community_size, authors.community_size),
        -- An empty skip_reasons from phase 2 does not clear a reason a live
        -- run recorded; a non-empty one replaces it.
        skip_reasons = CASE
            WHEN cardinality(cast(:skip_reasons AS varchar[])) > 0
            THEN cast(:skip_reasons AS varchar[])
            ELSE authors.skip_reasons END,
        score_components = COALESCE(
            cast(:score_components AS jsonb), authors.score_components
        ),
        scoring_version = COALESCE(:scoring_version, authors.scoring_version)
    WHERE author_id = :author_id AND project_id = :project_id
    """
    _bulk_upsert(session, sql, payload)
    updated = session.execute(
        text("SELECT count(*) FROM authors WHERE project_id = :p AND scoring_version IS NOT NULL"),
        {"p": project_id},
    ).scalar()
    log.info("author_scores: %d authors carry scores", updated)
    return updated or 0


def load_narratives(session: Session, scored_dir: Path, project_id: uuid.UUID) -> tuple[int, int]:
    """``narratives`` + ``narrative_membership`` -> ``narratives`` + ``narrative_posts``.

    Phase 2's ``narrative_id`` is a stable *string* carried across clustering
    runs by centroid matching. Phase 4's primary key is a uuid. The two are
    bridged by a deterministic uuid5 over (project, phase-2 id), so a
    re-import maps to the same row and the analyst's edits survive.

    **User-edited titles are preserved.** That is the acceptance criterion, and
    it is enforced in the ON CONFLICT clause rather than by remembering to check.
    """
    rows = _read(scored_dir, "narratives")
    if not rows:
        return 0, 0

    run_id = uuid.uuid4()
    session.execute(
        text(
            """
            INSERT INTO clustering_runs (
                id, project_id, algorithm, params, embedding_model,
                post_count, narrative_count, status, started_at, finished_at
            ) VALUES (
                :id, :project_id, 'hdbscan', cast(:params AS jsonb), :embedding_model,
                :post_count, :narrative_count, 'succeeded', :started_at, :finished_at
            )
            """
        ),
        {
            "id": run_id,
            "project_id": project_id,
            "params": json.dumps({"imported_from": "data/scored/narratives"}),
            "embedding_model": _first_model(rows, "embed"),
            "post_count": None,
            "narrative_count": len(rows),
            "started_at": min(
                (r.get("generated_at") for r in rows if r.get("generated_at")), default=None
            ),
            "finished_at": max(
                (r.get("generated_at") for r in rows if r.get("generated_at")), default=None
            ),
        },
    )

    payload = []
    for row in rows:
        payload.append(
            {
                "id": narrative_uuid(project_id, row["narrative_id"]),
                "project_id": project_id,
                "title": row.get("label"),
                # Phase 2 records whether the label came from an LLM or from
                # centroid keywords. Passed through verbatim: the UI's AI pill
                # must not be rendered over a heuristic label.
                "title_generated_by": _label_source(row.get("label_source")),
                "summary": row.get("summary"),
                "summary_generated_by": _label_source(row.get("label_source")),
                "summary_model": _first_model([row], "summarize"),
                "summary_generated_at": row.get("generated_at"),
                "cluster_id": None,
                "clustering_run_id": run_id,
                "date_start": row.get("first_seen"),
                "date_end": row.get("last_seen"),
                "post_count": row.get("size") or 0,
                "author_count": row.get("author_count") or 0,
                "platforms": list(row.get("platforms") or ()),
                "top_domains": list(row.get("top_domains") or ()),
                "top_hashtags": list(row.get("top_hashtags") or ()),
                "coherence": row.get("coherence"),
                "velocity": row.get("velocity"),
            }
        )

    sql = """
    INSERT INTO narratives (
        id, project_id, title, title_generated_by, summary, summary_generated_by,
        summary_model, summary_generated_at, cluster_id, clustering_run_id,
        date_start, date_end, post_count, author_count, platforms, top_domains,
        top_hashtags, coherence, velocity
    ) VALUES (
        :id, :project_id, :title, :title_generated_by, :summary, :summary_generated_by,
        :summary_model, :summary_generated_at, :cluster_id, :clustering_run_id,
        :date_start, :date_end, :post_count, :author_count, :platforms, :top_domains,
        :top_hashtags, :coherence, :velocity
    )
    ON CONFLICT (id) DO UPDATE SET
        -- Membership, counts and freshness always refresh. The *text* refreshes
        -- only where an analyst has not touched it. This is the rule that keeps
        -- a rename from being silently reverted by the next clustering import,
        -- and it lives in the statement rather than in a code path somebody
        -- could forget to take.
        title = CASE
            WHEN narratives.edited_by_user AND 'title' = ANY(narratives.edited_fields)
            THEN narratives.title ELSE EXCLUDED.title END,
        title_generated_by = CASE
            WHEN narratives.edited_by_user AND 'title' = ANY(narratives.edited_fields)
            THEN narratives.title_generated_by ELSE EXCLUDED.title_generated_by END,
        summary = CASE
            WHEN narratives.edited_by_user AND 'summary' = ANY(narratives.edited_fields)
            THEN narratives.summary ELSE EXCLUDED.summary END,
        summary_generated_by = CASE
            WHEN narratives.edited_by_user AND 'summary' = ANY(narratives.edited_fields)
            THEN narratives.summary_generated_by ELSE EXCLUDED.summary_generated_by END,
        summary_model = EXCLUDED.summary_model,
        summary_generated_at = EXCLUDED.summary_generated_at,
        clustering_run_id = EXCLUDED.clustering_run_id,
        date_start = EXCLUDED.date_start,
        date_end = EXCLUDED.date_end,
        post_count = EXCLUDED.post_count,
        author_count = EXCLUDED.author_count,
        platforms = EXCLUDED.platforms,
        top_domains = EXCLUDED.top_domains,
        top_hashtags = EXCLUDED.top_hashtags,
        coherence = EXCLUDED.coherence,
        velocity = EXCLUDED.velocity,
        updated_at = now()
    -- A hand-built narrative is never touched by an import at all.
    WHERE NOT narratives.is_manual
    """
    _bulk_upsert(session, sql, payload)

    membership = _read(scored_dir, "narrative_membership") or []
    member_payload = [
        {
            "narrative_id": narrative_uuid(project_id, row["narrative_id"]),
            "post_id": row["record_id"],
            "membership_score": row.get("membership_prob"),
            "is_representative": bool(row.get("is_representative")),
            "project_id": project_id,
        }
        for row in membership
    ]
    if member_payload:
        _bulk_upsert(
            session,
            """
            INSERT INTO narrative_posts (
                narrative_id, post_id, membership_score, is_representative
            )
            SELECT cast(:narrative_id AS uuid), cast(:post_id AS text),
                   cast(:membership_score AS real), cast(:is_representative AS boolean)
            WHERE EXISTS (
                SELECT 1 FROM posts
                WHERE id = cast(:post_id AS text) AND project_id = cast(:project_id AS uuid)
            )
              AND EXISTS (SELECT 1 FROM narratives WHERE id = cast(:narrative_id AS uuid))
            ON CONFLICT (narrative_id, post_id) DO UPDATE SET
                membership_score = EXCLUDED.membership_score,
                is_representative = EXCLUDED.is_representative
            """,
            member_payload,
        )

    narrative_count = session.execute(
        text("SELECT count(*) FROM narratives WHERE project_id = :p"), {"p": project_id}
    ).scalar()
    member_count = session.execute(
        text(
            "SELECT count(*) FROM narrative_posts np JOIN narratives n ON n.id = np.narrative_id "
            "WHERE n.project_id = :p"
        ),
        {"p": project_id},
    ).scalar()

    # Engagement is a corpus fact, not a model output, so it is recomputed from
    # posts rather than trusted from the scored table. Nulls contribute nothing.
    session.execute(
        text(
            """
            UPDATE narratives n SET engagement_total = COALESCE(sub.total, 0)
            FROM (
                SELECT np.narrative_id,
                       sum(COALESCE(p.likes,0) + COALESCE(p.shares,0)
                           + COALESCE(p.replies,0)) AS total
                FROM narrative_posts np JOIN posts p ON p.id = np.post_id
                WHERE p.project_id = :p
                GROUP BY np.narrative_id
            ) sub
            WHERE n.id = sub.narrative_id
            """
        ),
        {"p": project_id},
    )

    log.info("narratives: %d rows, %d memberships", narrative_count, member_count)
    return narrative_count or 0, member_count or 0


def load_network_edges(session: Session, scored_dir: Path, project_id: uuid.UUID) -> int:
    """``coordination_edges`` -> ``network_edges``.

    Phase 2 emits one edge per observed coordination window with an ``evidence``
    string. Phase 4 buckets by time for the UI's scrubber, so the window start
    is snapped to the configured bucket width on the way in.
    """
    from app.config import get_api_settings

    rows = _read(scored_dir, "coordination_edges")
    if not rows:
        return 0

    hours = get_api_settings().graph_bucket_hours
    payload = []
    for row in rows:
        window_start = row.get("window_start") or row.get("generated_at")
        if window_start is None:
            continue
        payload.append(
            {
                "project_id": project_id,
                "src_author_id": row["src_author_id"],
                "dst_author_id": row["dst_author_id"],
                # Phase 2's `evidence` names the mechanism; it maps onto the
                # edge_type vocabulary the UI legend is written against.
                "edge_type": _edge_type(row.get("evidence")),
                "weight": row.get("weight") or 1.0,
                "bucket_start": _snap(window_start, hours),
                "first_ts": row.get("window_start"),
                "last_ts": row.get("window_end"),
                "observations": row.get("observations") or 1,
            }
        )

    _bulk_upsert(
        session,
        """
        INSERT INTO network_edges (
            project_id, narrative_id, src_author_id, dst_author_id, edge_type,
            weight, bucket_start, first_ts, last_ts, observations
        )
        SELECT cast(:project_id AS uuid), NULL, cast(:src_author_id AS text),
               cast(:dst_author_id AS text), cast(:edge_type AS text),
               cast(:weight AS real), cast(:bucket_start AS timestamptz),
               cast(:first_ts AS timestamptz), cast(:last_ts AS timestamptz),
               cast(:observations AS integer)
        WHERE EXISTS (
            SELECT 1 FROM authors
            WHERE author_id = cast(:src_author_id AS text)
              AND project_id = cast(:project_id AS uuid)
        )
          AND EXISTS (
            SELECT 1 FROM authors
            WHERE author_id = cast(:dst_author_id AS text)
              AND project_id = cast(:project_id AS uuid)
        )
        -- Upsert on the identity index, so a rerun refreshes weights instead of
        -- doubling them. A coordination graph whose weights double on every
        -- retry looks more coordinated each time it is rebuilt.
        ON CONFLICT
            (project_id, narrative_id, src_author_id, dst_author_id, edge_type,
             bucket_start)
        DO UPDATE SET
            weight = EXCLUDED.weight,
            last_ts = GREATEST(network_edges.last_ts, EXCLUDED.last_ts),
            observations = EXCLUDED.observations
        """,
        payload,
    )
    count = session.execute(
        text("SELECT count(*) FROM network_edges WHERE project_id = :p"), {"p": project_id}
    ).scalar()
    log.info("network_edges: %d rows", count)
    return count or 0


def load_media_scores(session: Session, scored_dir: Path, project_id: uuid.UUID) -> int:
    """``media_scores`` -> ``media_checks``.

    These are checks of media already in the corpus, so there is no uploaded
    file and nothing to purge: ``storage_path`` stays null and the retention
    task passes over them.
    """
    rows = _read(scored_dir, "media_scores")
    if not rows:
        return 0

    payload = []
    for row in rows:
        probability = row.get("deepfake_prob")
        payload.append(
            {
                "id": uuid.uuid5(
                    uuid.NAMESPACE_URL, f"{project_id}:{row['record_id']}:{row['media_url']}"
                ),
                "project_id": project_id,
                "post_id": row["record_id"],
                "filename": row["media_url"].rsplit("/", 1)[-1][:512],
                "verdict": _verdict(probability, row.get("face_detected")),
                "confidence": probability,
                "manipulation_type": row.get("manipulation_type"),
                "frames_analyzed": row.get("frames_analyzed"),
                "face_detected": row.get("face_detected"),
                "explanation": row.get("explanation"),
                "limitations": _deepfake_limitations(row),
                "model": "xception-deepfake",
                "model_version": _version_of(row.get("model_versions")),
                "submitted_at": row.get("scored_at"),
                "completed_at": row.get("scored_at"),
            }
        )

    _bulk_upsert(
        session,
        """
        INSERT INTO media_checks (
            id, project_id, post_id, filename, verdict, confidence, manipulation_type,
            frames_analyzed, face_detected, explanation, limitations, model,
            model_version, submitted_at, completed_at
        ) VALUES (
            :id, :project_id, :post_id, :filename, :verdict, :confidence, :manipulation_type,
            :frames_analyzed, :face_detected, :explanation, :limitations, :model,
            :model_version, :submitted_at, :completed_at
        )
        ON CONFLICT (id) DO UPDATE SET
            verdict = EXCLUDED.verdict,
            confidence = EXCLUDED.confidence,
            manipulation_type = EXCLUDED.manipulation_type,
            explanation = EXCLUDED.explanation,
            completed_at = EXCLUDED.completed_at
        """,
        payload,
    )
    count = session.execute(
        text("SELECT count(*) FROM media_checks WHERE project_id = :p"), {"p": project_id}
    ).scalar()
    log.info("media_checks: %d rows", count)
    return count or 0


def load_embeddings(session: Session, data_dir: Path, project_id: uuid.UUID) -> int:
    """Phase 2's embedding cache -> the pgvector table.

    The artifact is **content-addressed**, which is not what the file names
    suggest. ``<model>.keys.json`` is a dict of ``sha256(text)[:32] -> row index``
    into ``<model>.npy``; it is a cache keyed by what was embedded, not an index
    of post ids. In this corpus 4,126 unique texts cover 4,188 posts, because
    exact reposts share text and therefore share a vector -- which is correct,
    and which a naive positional zip would have silently mangled.

    So the mapping is recomputed here: read each post's text, hash it the same
    way Phase 2 does, and look the row up. ``modeling.text.embed.EmbeddingCache.key``
    is the source of truth for that hash, and it is imported rather than
    reimplemented so the two cannot drift.

    Vectors are re-normalized on write even though Phase 2 normalizes them
    already. It costs one pass and it is the only way to be sure: everything
    downstream compares these with cosine distance, and one un-normalized batch
    produces neighbours that are subtly wrong in a way no test on the index
    itself would catch.
    """
    import hashlib as _hashlib

    import numpy as np

    from app.config import get_api_settings

    settings = get_api_settings()
    embeddings_dir = data_dir / "embeddings"
    if not embeddings_dir.exists():
        log.info("no embedding cache at %s", embeddings_dir)
        return 0

    def cache_key(text_value: str) -> str:
        """Phase 2's key function.

        Imported when modeling/ is installed so the definition cannot drift;
        the inline fallback exists because the API image deliberately does not
        carry the modeling tree, and a test asserts the two agree.
        """
        try:
            from modeling.text.embed import EmbeddingCache

            return EmbeddingCache.key(text_value)
        except ImportError:
            return _hashlib.sha256(text_value.encode("utf-8")).hexdigest()[:32]

    posts = session.execute(
        text('SELECT id, "text" FROM posts WHERE project_id = :p'), {"p": project_id}
    ).all()
    if not posts:
        return 0

    for keys_path in sorted(embeddings_dir.glob("*.keys.json")):
        matrix_path = keys_path.with_name(keys_path.name.replace(".keys.json", ".npy"))
        if not matrix_path.exists():
            log.warning("embedding index %s has no matrix beside it; skipping", keys_path.name)
            continue

        index = json.loads(keys_path.read_text(encoding="utf-8"))
        if not isinstance(index, dict):
            log.error("%s is not a text-hash index; refusing to guess its layout", keys_path.name)
            continue

        matrix = np.load(matrix_path)
        if matrix.shape[1] != settings.embedding_dim:
            # Never pad and never truncate. A pgvector column is fixed-width and
            # a padded vector is a wrong answer that looks like a right one.
            log.error(
                "%s is %d-dimensional but EMBEDDING_DIM is %d. Change EMBEDDING_DIM "
                "and migrate rather than reshaping the vectors.",
                keys_path.name,
                matrix.shape[1],
                settings.embedding_dim,
            )
            continue

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = (matrix / norms).astype("float32")

        model_name = keys_path.name.replace(".keys.json", "")
        payload, misses = [], 0
        for post_id, post_text in posts:
            row = index.get(cache_key(post_text or ""))
            if row is None or row >= normalized.shape[0]:
                misses += 1
                continue
            vector = normalized[row]
            payload.append(
                {
                    "post_id": post_id,
                    "model": model_name,
                    "dim": settings.embedding_dim,
                    "embedding": "[" + ",".join(f"{v:.6f}" for v in vector) + "]",
                    "project_id": project_id,
                }
            )

        if misses:
            # Not fatal and not silent. Two causes, both benign:
            #
            # 1. Phase 2 hashes the *prepared* text, not the raw text -- long
            #    articles are truncated with a lede-preserving policy that needs
            #    the model's tokenizer to reproduce. Reproducing it by guesswork
            #    would map some posts to the wrong vector, which is far worse
            #    than not mapping them, so short posts match and long ones do
            #    not.
            # 2. A post ingested after the cache was written is simply not in it.
            #
            # Either way nlp.embed computes what is missing, and until it runs
            # those posts are absent from kNN rather than wrongly present.
            log.info(
                "%s: %d of %d post(s) have no cached vector (long texts are hashed "
                "post-truncation by phase 2); nlp.embed will compute them",
                model_name,
                misses,
                len(posts),
            )

        _bulk_upsert(
            session,
            f"""
            INSERT INTO {settings.embedding_table} (post_id, model, dim, embedding)
            SELECT cast(:post_id AS text), cast(:model AS text), cast(:dim AS integer),
                   cast(:embedding AS vector)
            WHERE EXISTS (
                SELECT 1 FROM posts
                WHERE id = cast(:post_id AS text) AND project_id = cast(:project_id AS uuid)
            )
            ON CONFLICT (post_id) DO UPDATE SET
                model = EXCLUDED.model, dim = EXCLUDED.dim, embedding = EXCLUDED.embedding
            """,
            payload,
            chunk=1000,
        )

    total = session.execute(
        text(
            f"SELECT count(*) FROM {settings.embedding_table} e "  # noqa: S608 - name from config
            "JOIN posts p ON p.id = e.post_id WHERE p.project_id = :p"
        ),
        {"p": project_id},
    ).scalar()
    log.info("embeddings: %d vectors", total)
    return total or 0


def load_all(session: Session, *, project: str) -> ScoredLoadReport:
    """Load every Phase 2/3 output that exists on disk."""
    from app.config import get_api_settings
    from app.etl.derive import refresh_domain_narrative_counts
    from app.etl.service import resolve_project

    settings = get_api_settings()
    project_id = resolve_project(session, project)
    report = ScoredLoadReport()

    report.post_scores = load_post_scores(session, settings.scored_dir, project_id)
    report.author_scores = load_author_scores(session, settings.scored_dir, project_id)
    report.narratives, report.narrative_posts = load_narratives(
        session, settings.scored_dir, project_id
    )
    report.network_edges = load_network_edges(session, settings.scored_dir, project_id)
    report.media_checks = load_media_scores(session, settings.scored_dir, project_id)
    report.embeddings = load_embeddings(session, settings.data_dir, project_id)

    # Narrative counts on domains need narratives, which only exist now.
    refresh_domain_narrative_counts(session, project_id)
    return report


# ---------------------------------------------------------------------------
# mapping helpers
# ---------------------------------------------------------------------------
def narrative_uuid(project_id: uuid.UUID, phase2_id: str) -> uuid.UUID:
    """Phase 2's string narrative id -> a stable Phase 4 uuid.

    Deterministic so a re-import lands on the same row and an analyst's rename
    survives. Namespaced by project so two projects that both clustered into
    ``narrative_003`` do not collide.
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, f"narrative:{project_id}:{phase2_id}")


def _version_of(model_versions: Any) -> str | None:
    """Collapse Phase 2's per-module version map into one comparable identifier.

    A digest over the sorted map, not a join of it. Joining is the obvious thing
    and it is wrong: the string grows with the number of scoring modules, and
    the real corpus already produces
    ``anomaly=v0.1.0,emotion=v0.1.0,misinfo=v0.1.0+08e4a6ff,...`` at 85
    characters. It would overflow again the next time a module is added.

    A digest is fixed-width and has the property that actually matters -- two
    rows scored by different module versions get different identifiers, so a
    stale row is detectable. The human-readable map is one column over in
    ``model_versions``, which is where anybody debugging should look anyway.
    """
    if not model_versions:
        return None
    mapping = dict(model_versions)
    if not mapping:
        return None
    canonical = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return "phase2:" + hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _first_model(rows: list[dict[str, Any]], key: str) -> str | None:
    for row in rows:
        versions = dict(row.get("model_versions") or {})
        if key in versions:
            return f"{key}:{versions[key]}"
    return None


def _label_source(source: str | None) -> str:
    """Phase 2's label_source -> the API's generated_by vocabulary.

    An LLM-written label is `ai` and gets the pill. A centroid-keyword label is
    `heuristic` and must not: claiming a model wrote something it did not is the
    same failure as hiding that one did.
    """
    if not source:
        return "heuristic"
    lowered = source.lower()
    if "llm" in lowered or "claude" in lowered or "anthropic" in lowered:
        return "ai"
    if "human" in lowered or "analyst" in lowered:
        return "human"
    return "heuristic"


def _edge_type(evidence: str | None) -> str:
    if not evidence:
        return "co_post_similarity"
    lowered = evidence.lower()
    for candidate in ("reply", "repost", "mention"):
        if candidate in lowered:
            return candidate
    return "co_post_similarity"


def _snap(value, hours: int):
    """Snap a timestamp down to the bucket grid the scrubber uses."""
    from datetime import timedelta, timezone

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    epoch = value.replace(hour=0, minute=0, second=0, microsecond=0)
    offset = int((value - epoch).total_seconds() // (hours * 3600))
    return epoch + timedelta(hours=hours * offset)


def _verdict(probability: float | None, face_detected: bool | None) -> str | None:
    """Probability -> the four-value verdict the UI renders.

    "No face found" is `inconclusive`, never `likely_authentic`. A frame with no
    face is a frame the model could not assess, and reporting that as authentic
    is the single most misleading thing this module could do.
    """
    if face_detected is False:
        return "inconclusive"
    if probability is None:
        return None
    if probability >= 0.75:
        return "likely_manipulated"
    if probability >= 0.5:
        return "possibly_manipulated"
    return "likely_authentic"


def _deepfake_limitations(row: dict[str, Any]) -> list[str]:
    limitations = [
        "Trained on FaceForensics++ and DFDC; generators outside those sets are "
        "out of distribution.",
        "Heavy compression produces artefacts similar to manipulation.",
    ]
    if row.get("face_detected") is False:
        limitations.append(
            "No face was detected, so no verdict is possible. This is not evidence "
            "that the media is authentic."
        )
    if (row.get("frames_analyzed") or 0) < 10:
        limitations.append(
            f"Only {row.get('frames_analyzed')} frame(s) were analysed; the confidence "
            "is correspondingly weak."
        )
    return limitations
