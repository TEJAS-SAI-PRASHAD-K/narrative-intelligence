"""The generated demo corpus.

Built once, lazily, from a fixed seed. Every id here is stable across restarts
so a frontend developer can bookmark ``/narratives/<id>`` and have it keep
working.

The generated shape is deliberately awkward in the same ways real data is:
some engagement metrics are null rather than zero, some posts carry no scores at
all, one narrative has been renamed by an analyst, and one has no Compass
context. A demo corpus in which everything is populated teaches the frontend to
assume fields exist, and that assumption fails on the first real query.
"""

from __future__ import annotations

import hashlib
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

#: Everything stochastic here derives from this. Changing it changes every
#: fixture id, so it is a deliberate act.
SEED = 20260820

#: A fixed "now" so the fixtures never drift with the wall clock. Relative
#: language in the UI ("2 hours ago") is computed from the timestamps, so a
#: moving now would make every snapshot test flaky.
DEMO_NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
DEMO_WINDOW_DAYS = 60

DEMO_PROJECT_SLUG = "demo-election-integrity"

PLATFORMS = ("reddit", "mastodon", "news", "gdelt", "youtube")
CONTENT_TYPES = {
    "reddit": ("post", "comment"),
    "mastodon": ("post",),
    "news": ("article",),
    "gdelt": ("article",),
    "youtube": ("video", "video_comment"),
}
SENTIMENTS = ("positive", "neutral", "negative")
EMOTIONS = ("fear", "anger", "disgust", "happiness", "surprise", "sadness", "neutral")
STANCES = ("support", "deny", "discuss", "unrelated")

NARRATIVE_SEEDS: tuple[tuple[str, str, str], ...] = (
    (
        "Mail-in ballots claimed to be counted twice",
        "A cluster alleging that mail-in ballots in three counties were scanned more "
        "than once. The claim circulates as a screenshot of a tabulator log with no "
        "county identifier.",
        "Mail-in ballots in three counties were counted twice.",
    ),
    (
        "Voting machine vendor tied to foreign ownership",
        "Posts asserting that a voting-machine vendor is majority-owned abroad, "
        "amplified by a tight cluster of accounts created within the same week.",
        "The voting machine vendor is majority foreign-owned.",
    ),
    (
        "Poll worker footage presented as ballot stuffing",
        "A short clip of a poll worker handling a ballot box, reposted with captions "
        "asserting fraud. Reverse image search places the footage two elections earlier.",
        "The footage shows ballot stuffing during this election.",
    ),
    (
        "Voter roll purge framed as targeted suppression",
        "Coverage and commentary on a routine voter-roll maintenance action, reframed "
        "as deliberate suppression of one demographic.",
        "The voter roll purge deliberately targeted one demographic.",
    ),
    (
        "Signature verification software alleged to auto-reject",
        "Claims that signature verification software rejects ballots automatically "
        "without human review.",
        "Signature verification software auto-rejects ballots with no human review.",
    ),
    (
        "Overnight vote spike described as impossible",
        "A cluster built on a chart of reported vote totals, arguing an overnight "
        "increase is mathematically impossible.",
        "The overnight vote increase was mathematically impossible.",
    ),
    (
        "Election observers claimed to be barred from counting rooms",
        "Accounts of observers being denied entry, amplified across two platforms "
        "within a four-hour window.",
        "Election observers were barred from the counting rooms.",
    ),
    (
        "Ballot paper watermark conspiracy",
        "A recurring claim that authentic ballots carry a covert watermark, used to "
        "argue that unwatermarked ballots are fraudulent.",
        "Authentic ballots carry a covert watermark.",
    ),
)

COHORT_SEEDS: tuple[tuple[str, str, str], ...] = (
    ("Right Wing", "political", "Accounts whose posting history leans consistently right."),
    ("Left Wing", "political", "Accounts whose posting history leans consistently left."),
    ("Crypto Fan", "interest", "Accounts posting heavily about cryptocurrency."),
    ("Election Watchers", "interest", "Accounts focused on electoral process and administration."),
    ("Micro Influencer", "influence_tier", "1k-50k followers with above-median engagement."),
    ("Macro Influencer", "influence_tier", "50k+ followers."),
    ("Nano Account", "influence_tier", "Under 1k followers; the bulk of the corpus."),
    (
        "US Domestic",
        "geopolitical",
        "Accounts geolocating to or posting about US domestic politics.",
    ),
    ("Eastern Europe", "geopolitical", "Accounts geolocating to Eastern Europe."),
)

DOMAIN_SEEDS: tuple[tuple[str, str], ...] = (
    ("truthpatriotdaily.com", "high"),
    ("electionfactsnow.net", "high"),
    ("libertysignal.org", "medium"),
    ("reuters.com", "low"),
    ("apnews.com", "low"),
    ("bbc.co.uk", "low"),
    ("statecountyclerk.gov", "low"),
    ("vote-integrity-watch.info", "high"),
    ("mediumpost.example", "medium"),
    ("youtube.com", "low"),
)

HASHTAG_POOL = (
    "#election2026",
    "#ballotfraud",
    "#voterintegrity",
    "#stopthecount",
    "#auditthevote",
    "#mailinvoting",
    "#pollwatch",
    "#certifyresults",
    "#electionday",
    "#votingmachines",
)


def _stable_uuid(namespace: str, key: str) -> str:
    """A uuid that depends only on its inputs.

    ``uuid4`` here would give a frontend developer a different narrative id on
    every container restart, which makes bookmarking a page impossible and any
    recorded demo unreproducible.
    """
    digest = hashlib.sha256(f"{SEED}:{namespace}:{key}".encode()).digest()
    return str(uuid.UUID(bytes=digest[:16], version=5))


@dataclass
class MockNarrative:
    id: str
    display_id: int
    title: str
    summary: str
    claim: str
    title_generated_by: str
    summary_generated_by: str
    edited_by_user: bool
    is_manual: bool
    cluster_id: int
    date_start: datetime
    date_end: datetime
    post_ids: list[str] = field(default_factory=list)
    author_ids: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    top_domains: list[str] = field(default_factory=list)
    top_hashtags: list[str] = field(default_factory=list)
    engagement_total: int = 0
    scorecard: dict[str, Any] = field(default_factory=dict)
    coherence: float = 0.0
    velocity: float = 0.0
    has_compass: bool = True


@dataclass
class MockAuthor:
    author_id: str
    source: str
    handle: str
    created_at_source: datetime | None
    followers: int | None
    following: int | None
    post_count: int
    first_seen: datetime
    last_seen: datetime
    is_bot_flagged: bool | None
    bot_score: float | None
    risk_score: float | None
    partisan_lean: float | None
    dominant_sentiment: str | None
    dominant_emotion: str | None
    anomalous_score: float | None
    toxicity_score: float | None
    coordination_score: float | None
    community_id: str | None
    community_size: int | None
    cohort_ids: list[str] = field(default_factory=list)
    narrative_ids: list[str] = field(default_factory=list)
    skip_reasons: list[str] = field(default_factory=list)


@dataclass
class MockPost:
    id: str
    native_id: str
    source: str
    source_detail: str
    content_type: str
    text: str
    lang: str | None
    author_id: str
    author_handle: str
    timestamp: datetime
    parent_id: str | None
    conversation_id: str
    likes: int | None
    shares: int | None
    replies: int | None
    views: int | None
    urls: list[str]
    domains: list[str]
    media_urls: list[str]
    hashtags: list[str]
    mentions: list[str]
    scores: dict[str, Any] | None
    narrative_ids: list[str] = field(default_factory=list)


@dataclass
class MockCorpus:
    project_id: str
    narratives: list[MockNarrative]
    authors: list[MockAuthor]
    posts: list[MockPost]
    cohorts: list[dict[str, Any]]
    domains: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    # --- indexes, built once ---
    def __post_init__(self) -> None:
        self.by_narrative = {n.id: n for n in self.narratives}
        self.by_author = {a.author_id: a for a in self.authors}
        self.by_post = {p.id: p for p in self.posts}
        self.by_cohort = {c["id"]: c for c in self.cohorts}
        self.by_domain = {d["domain"]: d for d in self.domains}


def _make_text(rng: random.Random, narrative: tuple[str, str, str], platform: str) -> str:
    openers = (
        "Seeing more of this today:",
        "Can anyone verify this?",
        "This keeps getting reposted.",
        "Sharing because nobody else will:",
        "Filed a records request about this.",
        "Local outlet is covering it now.",
        "Third account I've seen post the same screenshot.",
    )
    closers = (
        "Sources in the replies.",
        "Screenshot attached.",
        "Do your own research.",
        "The timeline does not add up.",
        "Waiting on an official statement.",
        "",
    )
    if platform in {"news", "gdelt"}:
        return f"{narrative[0]}: {narrative[1][:140]}"
    return " ".join(
        part for part in (rng.choice(openers), narrative[2], rng.choice(closers)) if part
    )


@lru_cache(maxsize=1)
def corpus() -> MockCorpus:
    """Build the demo corpus. Cached, so the ids are stable for the process."""
    rng = random.Random(SEED)
    project_id = _stable_uuid("project", DEMO_PROJECT_SLUG)
    window_start = DEMO_NOW - timedelta(days=DEMO_WINDOW_DAYS)

    # --- cohorts ---------------------------------------------------------
    cohorts: list[dict[str, Any]] = []
    for name, category, description in COHORT_SEEDS:
        cohorts.append(
            {
                "id": _stable_uuid("cohort", name),
                "project_id": project_id,
                "name": name,
                "description": description,
                "category": category,
                "author_count": 0,
                "post_count": 0,
            }
        )

    # --- authors ---------------------------------------------------------
    authors: list[MockAuthor] = []
    community_ids = [_stable_uuid("community", str(i)) for i in range(6)]
    for index in range(240):
        source = rng.choices(PLATFORMS, weights=(35, 30, 10, 10, 15))[0]
        handle = (
            f"{rng.choice(('patriot', 'watcher', 'clerk', 'daily', 'signal', 'anon'))}_{index:03d}"
        )
        author_id = f"{source}:{handle}"
        created = (
            None
            if source in {"news", "gdelt"}
            else window_start - timedelta(days=rng.randint(1, 2400))
        )
        # News and GDELT "authors" are outlets, not people. Phase 2 skips
        # account-level bot scoring for them with a reason code rather than
        # producing a number, and the fixture reproduces that honestly.
        is_outlet = source in {"news", "gdelt"}
        bot_score = None if is_outlet else round(rng.betavariate(2, 5), 3)
        # A deliberate minority of accounts look coordinated: recently created,
        # high bot score, tight community. This is what the UI must render well.
        if not is_outlet and index % 17 == 0:
            bot_score = round(rng.uniform(0.78, 0.96), 3)
            created = window_start - timedelta(days=rng.randint(3, 21))

        first_seen = window_start + timedelta(hours=rng.randint(0, 24 * DEMO_WINDOW_DAYS - 48))
        authors.append(
            MockAuthor(
                author_id=author_id,
                source=source,
                handle=handle,
                created_at_source=created,
                followers=None if is_outlet else rng.randint(3, 90_000),
                following=None if is_outlet else rng.randint(1, 4000),
                post_count=0,
                first_seen=first_seen,
                last_seen=first_seen + timedelta(hours=rng.randint(1, 600)),
                is_bot_flagged=(None if source != "mastodon" else rng.random() < 0.06),
                bot_score=bot_score,
                risk_score=None if is_outlet else round(rng.uniform(5, 95), 1),
                partisan_lean=None if is_outlet else round(rng.uniform(-1, 1), 2),
                dominant_sentiment=rng.choice(SENTIMENTS),
                dominant_emotion=rng.choice(EMOTIONS),
                anomalous_score=round(rng.betavariate(2, 6), 3),
                toxicity_score=round(rng.betavariate(2, 8), 3),
                coordination_score=None if is_outlet else round(rng.betavariate(2, 6), 3),
                community_id=None if is_outlet else rng.choice(community_ids),
                community_size=None if is_outlet else rng.randint(4, 90),
                skip_reasons=["author_is_outlet"] if is_outlet else [],
            )
        )

    # Multi-label cohort assignment. An author can be Right Wing *and* Crypto
    # Fan *and* Micro Influencer at once -- the UI spec is explicit about this,
    # and a fixture with one cohort per author would let the frontend get away
    # with rendering a single chip.
    for author in authors:
        picks = rng.sample(cohorts, k=rng.randint(1, 3))
        author.cohort_ids = [c["id"] for c in picks]
        for cohort in picks:
            cohort["author_count"] += 1

    # --- narratives ------------------------------------------------------
    narratives: list[MockNarrative] = []
    for index, seed in enumerate(NARRATIVE_SEEDS):
        start = window_start + timedelta(days=rng.randint(0, DEMO_WINDOW_DAYS - 10))
        # One narrative is analyst-renamed and one is hand-built, because both
        # states have to survive a reclustering run and the frontend needs to
        # render the difference.
        edited = index == 2
        manual = index == 7
        narratives.append(
            MockNarrative(
                id=_stable_uuid("narrative", seed[0]),
                display_id=1000 + index,
                title=("Reposted 2022 clip framed as 2026 fraud" if edited else seed[0]),
                summary=seed[1],
                claim=seed[2],
                title_generated_by="human" if edited else ("human" if manual else "ai"),
                summary_generated_by="human" if manual else "ai",
                edited_by_user=edited,
                is_manual=manual,
                cluster_id=index,
                date_start=start,
                date_end=start + timedelta(days=rng.randint(3, 20)),
                coherence=round(rng.uniform(0.42, 0.91), 3),
                velocity=round(rng.uniform(0.4, 34.0), 2),
                has_compass=index != 5,
            )
        )

    # --- posts -----------------------------------------------------------
    posts: list[MockPost] = []
    for index in range(1400):
        author = rng.choice(authors)
        source = author.source
        narrative = rng.choice(narratives) if rng.random() < 0.81 else None
        seed = NARRATIVE_SEEDS[narrative.cluster_id] if narrative else NARRATIVE_SEEDS[0]
        timestamp = window_start + timedelta(minutes=rng.randint(0, 60 * 24 * DEMO_WINDOW_DAYS))
        content_type = rng.choice(CONTENT_TYPES[source])
        native_id = f"{index:06d}"
        domains = [rng.choice(DOMAIN_SEEDS)[0]] if rng.random() < 0.42 else []
        # Engagement nulls are per-platform, exactly as Phase 1 leaves them:
        # Reddit exposes no view count, news exposes almost nothing. Zero would
        # be a lie and would quietly halve every per-post average.
        likes = None if source in {"news", "gdelt"} else rng.randint(0, 900)
        shares = None if source in {"news", "gdelt"} else rng.randint(0, 300)
        replies = None if source in {"news", "gdelt"} else rng.randint(0, 120)
        views = rng.randint(200, 90_000) if source == "youtube" else None

        # ~7% of posts carry no scores at all: the scorer skipped them. The
        # frontend must render a post with null scores without falling over.
        scored = rng.random() > 0.07
        scores = None
        if scored:
            emotion_scores = {e: round(rng.random(), 3) for e in EMOTIONS}
            total = sum(emotion_scores.values()) or 1.0
            emotion_scores = {k: round(v / total, 4) for k, v in emotion_scores.items()}
            toxicity = round(rng.betavariate(2, 8), 3)
            anomaly = round(rng.betavariate(2, 7), 3)
            scores = {
                "toxicity": toxicity,
                "is_toxic": toxicity > 0.6,
                "anomaly": anomaly,
                "is_anomalous": anomaly > 0.7,
                "misinfo_likelihood": round(rng.betavariate(3, 4), 3),
                "stance": rng.choice(STANCES),
                "sentiment": rng.choice(SENTIMENTS),
                "sentiment_score": round(rng.uniform(-1, 1), 3),
                "emotion": max(emotion_scores, key=emotion_scores.get),
                "emotion_scores": emotion_scores,
                "scoring_version": "phase2-v0.1.0",
                "scored_at": DEMO_NOW - timedelta(hours=6),
                "skip_reasons": [],
            }
        else:
            scores = {
                "scoring_version": None,
                "skip_reasons": [rng.choice(["non_english", "text_too_short"])],
            }

        post = MockPost(
            id=f"{source}:{native_id}",
            native_id=native_id,
            source=source,
            source_detail={
                "reddit": rng.choice(("r/politics", "r/news", "r/Conservative")),
                "mastodon": "mastodon.social",
                "news": rng.choice(("reuters.com", "bbc.co.uk")),
                "gdelt": "gdelt-doc",
                "youtube": "UC" + f"{rng.randint(10**9, 10**10 - 1)}",
            }[source],
            content_type=content_type,
            text=_make_text(rng, seed, source),
            lang="en" if rng.random() > 0.06 else rng.choice(("es", "de", None)),
            author_id=author.author_id,
            author_handle=author.handle,
            timestamp=timestamp,
            parent_id=None,
            conversation_id=f"{source}:conv-{index // 7:05d}",
            likes=likes,
            shares=shares,
            replies=replies,
            views=views,
            urls=[f"https://{d}/story/{index}" for d in domains],
            domains=domains,
            media_urls=(
                [f"https://cdn.example/{source}/{index}.jpg"] if rng.random() < 0.12 else []
            ),
            hashtags=rng.sample(HASHTAG_POOL, k=rng.randint(0, 3)),
            mentions=[],
            scores=scores,
            narrative_ids=[narrative.id] if narrative else [],
        )
        posts.append(post)
        author.post_count += 1
        if narrative:
            narrative.post_ids.append(post.id)
            if author.author_id not in narrative.author_ids:
                narrative.author_ids.append(author.author_id)
            if narrative.id not in author.narrative_ids:
                author.narrative_ids.append(narrative.id)
            narrative.engagement_total += sum(v for v in (likes, shares, replies) if v is not None)
            for domain in domains:
                if domain not in narrative.top_domains:
                    narrative.top_domains.append(domain)
            for tag in post.hashtags:
                if tag not in narrative.top_hashtags:
                    narrative.top_hashtags.append(tag)
            if source not in narrative.platforms:
                narrative.platforms.append(source)

    # Threading: give some replies a real parent inside their conversation, so
    # /posts/{id}/thread has something to reconstruct.
    by_conversation: dict[str, list[MockPost]] = {}
    for post in posts:
        by_conversation.setdefault(post.conversation_id, []).append(post)
    for thread in by_conversation.values():
        thread.sort(key=lambda p: p.timestamp)
        for child in thread[1:]:
            if child.source == "reddit" and rng.random() < 0.75:
                child.parent_id = rng.choice(thread[: thread.index(child)]).id

    # --- narrative scorecards -------------------------------------------
    for narrative in narratives:
        members = [
            corpus_post for corpus_post in posts if narrative.id in corpus_post.narrative_ids
        ]
        scored_members = [m for m in members if m.scores and m.scores.get("toxicity") is not None]
        member_authors = [a for a in authors if a.author_id in narrative.author_ids]
        bot_scores = [a.bot_score for a in member_authors if a.bot_score is not None]

        bot_like = (
            round(sum(1 for b in bot_scores if b > 0.6) / len(bot_scores), 3)
            if bot_scores
            else None
        )
        toxicity = (
            round(sum(m.scores["toxicity"] for m in scored_members) / len(scored_members), 3)
            if scored_members
            else None
        )
        anomalous = (
            round(sum(m.scores["anomaly"] for m in scored_members) / len(scored_members), 3)
            if scored_members
            else None
        )
        negative = (
            round(
                sum(1 for m in scored_members if m.scores["sentiment"] == "negative")
                / len(scored_members),
                3,
            )
            if scored_members
            else None
        )
        # Narrative 4 deliberately has no authenticity component: the deepfake
        # module never ran over it. Its weights renormalize over the remaining
        # two rather than treating authenticity as zero, and the frontend has
        # to render `missing` rather than a suspiciously low score.
        missing = ["authenticity"] if narrative.cluster_id == 4 else []
        fusion = round(rng.uniform(12, 94), 1)
        narrative.scorecard = {
            "priority": "high" if fusion >= 70 else ("medium" if fusion >= 40 else "low"),
            "bot_like": bot_like,
            "anomalous": anomalous,
            "toxicity": toxicity,
            "compass_risk": rng.choice(("high", "medium", "low")),
            "negative_sentiment": negative,
            "fusion_score": fusion,
            "scoring_version": "fusion-v1.0.0",
            "computed_at": DEMO_NOW - timedelta(hours=3),
            "missing": missing,
        }

    # --- domains ---------------------------------------------------------
    domains_out: list[dict[str, Any]] = []
    for domain, band in DOMAIN_SEEDS:
        linking = [p for p in posts if domain in p.domains]
        linking_authors = {p.author_id for p in linking}
        # Two domains are deliberately un-enriched: WHOIS is best-effort and
        # rate-limited, and a domain page must render without it.
        enriched = domain not in {"vote-integrity-watch.info", "mediumpost.example"}
        domains_out.append(
            {
                "domain": domain,
                "project_id": project_id,
                "risk_score": {
                    "high": rng.uniform(72, 96),
                    "medium": rng.uniform(38, 68),
                    "low": rng.uniform(4, 30),
                }[band],
                "risk_band": band,
                "first_seen": min((p.timestamp for p in linking), default=window_start),
                "last_seen": max((p.timestamp for p in linking), default=DEMO_NOW),
                "post_count": len(linking),
                "author_count": len(linking_authors),
                "narrative_count": len({n for p in linking for n in p.narrative_ids}),
                "whois_created_at": (
                    window_start - timedelta(days=rng.randint(20, 6000)) if enriched else None
                ),
                "hosting_country": rng.choice(("US", "NL", "RU", "DE")) if enriched else None,
                "tls_cert_age_days": rng.randint(3, 900) if enriched else None,
                "registrar": rng.choice(("NameCheap", "GoDaddy", "Cloudflare"))
                if enriched
                else None,
                "enrichment_status": "enriched" if enriched else "unavailable",
                "enriched_at": DEMO_NOW - timedelta(hours=9) if enriched else None,
                "enrichment_detail": (
                    None
                    if enriched
                    else (
                        "WHOIS query returned no record; risk computed from in-corpus signals only."
                    )
                ),
                "link_velocity": round(rng.uniform(0.1, 22.0), 2),
                "sharing_author_bot_ratio": round(rng.uniform(0.02, 0.7), 3),
            }
        )

    # --- network edges ---------------------------------------------------
    edges: list[dict[str, Any]] = []
    interactive = [a for a in authors if a.source in {"reddit", "mastodon"}]
    for _ in range(2600):
        src, dst = rng.sample(interactive, 2)
        bucket = window_start + timedelta(
            hours=12 * rng.randint(0, (DEMO_WINDOW_DAYS * 24) // 12 - 1)
        )
        edges.append(
            {
                "source": src.author_id,
                "target": dst.author_id,
                "edge_type": rng.choice(("reply", "repost", "mention", "co_post_similarity")),
                "weight": round(rng.uniform(0.1, 1.0), 3),
                "bucket_start": bucket,
                "first_ts": bucket,
                "last_ts": bucket + timedelta(hours=rng.randint(1, 11)),
            }
        )

    for cohort in cohorts:
        cohort["post_count"] = sum(a.post_count for a in authors if cohort["id"] in a.cohort_ids)

    return MockCorpus(
        project_id=project_id,
        narratives=narratives,
        authors=authors,
        posts=posts,
        cohorts=cohorts,
        domains=domains_out,
        edges=edges,
    )


def demo_project_id() -> str:
    return corpus().project_id


def demo_project_slug() -> str:
    return DEMO_PROJECT_SLUG
