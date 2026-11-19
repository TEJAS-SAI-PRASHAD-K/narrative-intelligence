"""The coordination -> accounts handoff survives running the stages separately.

The defect this pins: ``ctx.coordination`` was set in memory by the coordination
stage and read by the accounts stage, and nothing rehydrated it. Scoring the two
stages in separate invocations left every author with ``coordination:not_run``
and a null ``coordination_score``, which reaches a dashboard as "no coordination"
rather than "not measured". It held for three months over a real corpus.

The distinction these tests defend is the one in CLAUDE.md: an account the
detector saw but linked to nobody is a measured ``0.0``; an account it never saw
is ``None``. ``coordination_edges`` cannot tell them apart, because neither
appears in any edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from modeling.io import TABLES, ScoredStore
from modeling.scoring import (
    StageContext,
    _coordination_accounts_frame,
    _coordination_from_store,
    _write_coordination_accounts,
)


@dataclass
class FakeResult:
    """The parts of ``CoordinationResult`` the handoff reads."""

    scores: dict[str, float] = field(default_factory=dict)
    communities: dict[str, str] = field(default_factory=dict)
    community_sizes: dict[str, int] = field(default_factory=dict)
    seen_accounts: frozenset[str] = frozenset()

    def score_for(self, account: str) -> float | None:
        if account in self.scores:
            return self.scores[account]
        return 0.0 if account in self.seen_accounts else None


@pytest.fixture
def store(tmp_path, monkeypatch):
    from modeling import config as C

    C.get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    settings = C.get_settings()
    settings.ensure_dirs()
    yield ScoredStore(settings)
    C.get_settings.cache_clear()


def make_ctx(store, **kwargs):
    import pandas as pd

    from modeling import config as C

    return StageContext(
        settings=C.get_settings(),
        reader=None,
        store=store,
        records=pd.DataFrame(),
        authors=pd.DataFrame(),
        demo=False,
        dry_run=False,
        **kwargs,
    )


#: linked:  in an edge, carries a real score
#: lonely:  the detector saw it and linked it to nobody -> measured 0.0
#: unseen:  never in the detector's input -> no measurement at all
RESULT = FakeResult(
    scores={"mastodon:linked": 0.62},
    communities={"mastodon:linked": "c1"},
    community_sizes={"c1": 4},
    seen_accounts=frozenset({"mastodon:linked", "mastodon:lonely"}),
)


def test_the_table_is_part_of_the_output_contract():
    assert "coordination_accounts" in TABLES
    table = TABLES["coordination_accounts"]
    assert table.keys == ("account_id",)
    assert not table.schema.field("account_id").nullable


def test_an_account_linked_to_nobody_is_written_as_a_measured_zero():
    frame = _coordination_accounts_frame(RESULT, _now())
    rows = frame.set_index("account_id")["coordination_score"].to_dict()

    assert rows["mastodon:lonely"] == 0.0
    assert rows["mastodon:linked"] == pytest.approx(0.62)
    # The account the detector never saw must not acquire a row, because a row
    # is what makes a score a measurement.
    assert "mastodon:unseen" not in rows


def test_the_round_trip_preserves_measured_zero_against_never_measured(store):
    ctx = make_ctx(store)
    _write_coordination_accounts(ctx, RESULT, "v0.1.0")

    recovered = _coordination_from_store(make_ctx(store))

    assert recovered is not None
    assert recovered.score_for("mastodon:linked") == pytest.approx(0.62)
    # The whole point: 0.0, not None, and not absent.
    assert recovered.score_for("mastodon:lonely") == 0.0
    assert recovered.score_for("mastodon:unseen") is None


def test_communities_and_their_sizes_survive_the_round_trip(store):
    ctx = make_ctx(store)
    _write_coordination_accounts(ctx, RESULT, "v0.1.0")

    recovered = _coordination_from_store(make_ctx(store))

    assert recovered.communities["mastodon:linked"] == "c1"
    assert recovered.community_sizes["c1"] == 4
    # An account with no community must not fabricate one.
    assert "mastodon:lonely" not in recovered.communities


def test_no_table_is_a_real_not_run_and_not_a_table_of_zeros(store):
    """Absent must stay absent, or every author silently becomes uncoordinated."""
    assert _coordination_from_store(make_ctx(store)) is None


def test_an_edgeless_run_still_persists_the_accounts_it_measured(store):
    """A run with no edges measured every account it saw; all of them are 0.0."""
    edgeless = FakeResult(seen_accounts=frozenset({"reddit:a", "reddit:b"}))
    ctx = make_ctx(store)
    _write_coordination_accounts(ctx, edgeless, "v0.1.0")

    recovered = _coordination_from_store(make_ctx(store))

    assert recovered.score_for("reddit:a") == 0.0
    assert recovered.score_for("reddit:b") == 0.0
    assert recovered.score_for("reddit:never") is None


def test_a_stale_graph_warns_rather_than_failing_or_passing_quietly(store, caplog, monkeypatch):
    ctx = make_ctx(store)
    _write_coordination_accounts(ctx, RESULT, "v0.1.0")

    monkeypatch.setattr("modeling.config.manifest_hash", lambda: "a-different-corpus")
    with caplog.at_level("WARNING"):
        recovered = _coordination_from_store(make_ctx(store))

    assert recovered is not None, "a stale graph is still a measurement, not a failure"
    assert any(
        "coordination_accounts was built against corpus" in record.message
        for record in caplog.records
    )


def _now():
    from modeling.io import utcnow

    return utcnow()
