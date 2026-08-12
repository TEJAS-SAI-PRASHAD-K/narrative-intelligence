"""Source adapters. Each one is dumb on purpose: fetch raw dicts, map one dict
to one validated :class:`~ingest.schema.Record`, and nothing else.

The registry is lazy -- importing ``ingest.sources`` must not import convokit,
Mastodon.py or googleapiclient, so the pipeline still runs when the optional
extras are not installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ingest.sources.base import BaseSource

#: adapter name -> "module:class". Order is the order ``fetch-all`` runs them,
#: cheapest and most reliable first so a partial run still yields a usable corpus.
REGISTRY: dict[str, str] = {
    "reddit_convokit": "ingest.sources.reddit_convokit:ConvoKitSource",
    "gdelt": "ingest.sources.gdelt:GdeltSource",
    "news_rss": "ingest.sources.news_rss:NewsRssSource",
    "mastodon": "ingest.sources.mastodon:MastodonSource",
    "youtube": "ingest.sources.youtube:YouTubeSource",
    "reddit_kaggle": "ingest.sources.reddit_kaggle:RedditKaggleSource",
}


def get_source_class(name: str) -> type[BaseSource]:
    try:
        target = REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown source {name!r}; known: {', '.join(sorted(REGISTRY))}") from None
    module_name, class_name = target.split(":")
    return getattr(import_module(module_name), class_name)


def available_sources() -> list[str]:
    return list(REGISTRY)
