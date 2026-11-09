"""The stance module: headline-grouped splits, label orientation, the scoring stage."""

from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeling import config as C
from modeling import scoring as S
from modeling.datasets.fnc1 import FNC1
from modeling.io import ScoredStore
from modeling.registry import Checkpoint
from modeling.text.stance_clf import StanceClassifier, resolve_stance_labels
from modeling.training import stance_split

LABELS = ["agree", "disagree", "discuss", "unrelated"]


def _write_fnc1(root: Path) -> Path:
    """Few headlines, many bodies each, like the real corpus, plus test overlap."""
    root.mkdir(parents=True)
    body_text = "An article body with enough words to pass the length floor, number {}."
    train_headlines = [f"Train headline number {i} about a public matter" for i in range(12)]
    # The last train headline also occurs in the test file.
    shared = "Shared headline that occurs in both files"
    train_headlines[-1] = shared
    with (root / "train_stances.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Headline", "Body ID", "Stance"])
        for h, headline in enumerate(train_headlines):
            for k in range(8):
                writer.writerow([headline, h * 8 + k, LABELS[(h + k) % 4]])
    with (root / "train_bodies.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Body ID", "articleBody"])
        for b in range(96):
            writer.writerow([b, body_text.format(b)])
    test_headlines = [shared] + [f"Test headline number {i} about a matter" for i in range(3)]
    with (root / "competition_test_stances.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Headline", "Body ID", "Stance"])
        for h, headline in enumerate(test_headlines):
            for k in range(5):
                writer.writerow([headline, h * 5 + k, LABELS[(h + k) % 4]])
    with (root / "competition_test_bodies.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Body ID", "articleBody"])
        for b in range(20):
            writer.writerow([b, f"A held-out article body with enough words, number {b}."])
    return root


def test_stance_split_has_no_headline_on_both_sides(tmp_path):
    root = _write_fnc1(tmp_path / "fnc")
    frame = FNC1().load(root).frame
    work, split, description, notes = stance_split(frame, C.get_settings())

    sides = {
        name: set(work["claim_id"].iloc[idx])
        for name, idx in (("train", split.train), ("val", split.val), ("test", split.test))
    }
    assert sides["val"], "the validation side must not be empty"
    assert not sides["train"] & sides["val"]
    assert not sides["train"] & sides["test"]
    assert not sides["val"] & sides["test"]
    assert "headline" in description


def test_stance_split_keeps_the_competition_test_set_whole(tmp_path):
    root = _write_fnc1(tmp_path / "fnc")
    frame = FNC1().load(root).frame
    work, split, _, notes = stance_split(frame, C.get_settings())

    assert (work["official_split"].iloc[split.test] == "competition_test").all()
    assert len(split.test) == int((frame["official_split"] == "competition_test").sum())
    # The one shared headline is removed from train, never from test.
    assert any(n.startswith("purged_overlap_with_test: 8 ") for n in notes)
    assert split.dropped["purged_overlap_with_test"] == 8


# ---------------------------------------------------------------------------
# label orientation of a fine-tuned head
# ---------------------------------------------------------------------------
class _FakeConfig:
    def __init__(self, id2label):
        self.id2label = id2label


class _FakeModel:
    def __init__(self, id2label):
        self.config = _FakeConfig(id2label)

    def to(self, device):
        return self

    def eval(self):
        return self


def _install_fake_transformers(monkeypatch, id2label):
    """A stand-in for `transformers` that loads no weights."""
    module = types.ModuleType("transformers")
    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: object())
    module.AutoModelForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda path: _FakeModel(id2label)
    )
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_stance_head_is_resolved_by_name_not_by_position(monkeypatch, tmp_path):
    permuted = {0: "unrelated", 1: "disagree", 2: "agree", 3: "discuss"}
    assert resolve_stance_labels(permuted) == ["unrelated", "deny", "support", "discuss"]

    _install_fake_transformers(monkeypatch, permuted)
    classifier = StanceClassifier()
    assert classifier.load(tmp_path)

    # Logit 2 is the largest, and in this head position 2 means "agree".
    monkeypatch.setattr(
        classifier, "_raw_probabilities", lambda *a, **k: np.array([[0.1, 0.1, 0.7, 0.1]])
    )
    (prediction,) = classifier.predict([("claim", "post")])
    assert prediction.label == "support"
    assert prediction.confidence == pytest.approx(0.7)


@pytest.mark.parametrize(
    "id2label",
    [
        {},
        None,
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2", 3: "LABEL_3"},  # the Hub default
        {0: "support", 1: "deny", 2: "discuss"},  # a class is missing
        {0: "support", 1: "support", 2: "discuss", 3: "unrelated"},  # a class repeats
        {0: "entailment", 1: "neutral", 2: "contradiction", 3: "unrelated"},
        {0: "support", 1: "deny", 2: "discuss", 4: "unrelated"},  # a gap in the indices
    ],
)
def test_a_stance_head_with_unreadable_labels_is_refused(monkeypatch, tmp_path, id2label):
    assert resolve_stance_labels(id2label) is None

    _install_fake_transformers(monkeypatch, id2label)
    classifier = StanceClassifier()
    assert classifier.load(tmp_path) is False
    assert classifier.trained is False
    assert classifier.load_error == "label_orientation_unresolved"


# ---------------------------------------------------------------------------
# the scoring stage
# ---------------------------------------------------------------------------
CLAIM = "the representative claim of this narrative, long enough to score"
POST = "a post about that claim, comfortably longer than twenty characters"
HINDI = "यह एक दावा है जो बीस अक्षरों से काफी लंबा है"


class _FakeStance:
    """Stands in for StanceClassifier; loads nothing and records its inputs."""

    seen: list[tuple[str, str]] = []
    refuse_with: str | None = None

    def __init__(self, settings):
        self.load_error = None

    def load(self, path):
        if type(self).refuse_with:
            self.load_error = type(self).refuse_with
            return False
        return True

    def predict(self, pairs):
        from modeling.text.stance_clf import StancePrediction

        type(self).seen = list(pairs)
        return [StancePrediction("deny", 0.9) for _ in pairs]


@pytest.fixture
def settings(monkeypatch):
    C.get_settings.cache_clear()
    monkeypatch.delenv("SCORE_UNKNOWN_LANGUAGE", raising=False)
    yield C.get_settings()
    C.get_settings.cache_clear()


def _records() -> pd.DataFrame:
    rows = [
        {"id": "r:claim", "source": "news", "text": CLAIM, "lang": "en"},
        {"id": "r:en", "source": "news", "text": POST, "lang": "en"},
        {"id": "r:hi", "source": "news", "text": HINDI, "lang": "hi"},
        {"id": "r:noise", "source": "news", "text": POST + " unclustered", "lang": "en"},
        {"id": "r:short", "source": "news", "text": "too short", "lang": "en"},
    ]
    return pd.DataFrame(rows)


def _ctx(tmp_path, settings, *, checkpoint, monkeypatch, dry_run=False):
    import modeling.registry as registry
    import modeling.text.stance_clf as clf

    _FakeStance.seen = []
    _FakeStance.refuse_with = None
    monkeypatch.setattr(
        registry,
        "resolve",
        lambda module, version, required=False: Checkpoint(
            module="stance", version=version, path=tmp_path, origin="local", sha256="cd" * 32
        )
        if checkpoint
        else None,
    )
    monkeypatch.setattr(clf, "StanceClassifier", _FakeStance)
    records = _records()
    store = ScoredStore(settings, root=tmp_path / "scored")
    now = pd.Timestamp.now(tz="UTC")
    store.write(
        "narrative_membership",
        pd.DataFrame(
            [
                {"record_id": "r:claim", "narrative_id": "n1", "membership_prob": 0.99,
                 "is_representative": True, "generated_at": now},
                {"record_id": "r:en", "narrative_id": "n1", "membership_prob": 0.8,
                 "is_representative": False, "generated_at": now},
                {"record_id": "r:hi", "narrative_id": "n1", "membership_prob": 0.7,
                 "is_representative": False, "generated_at": now},
                {"record_id": "r:short", "narrative_id": "n1", "membership_prob": 0.6,
                 "is_representative": False, "generated_at": now},
            ]
        ),
    )
    return S.StageContext(
        settings=settings,
        reader=None,
        store=store,
        records=records,
        authors=pd.DataFrame(),
        demo=True,
        dry_run=dry_run,
        known_record_ids=set(records["id"]),
    )


def test_stance_stage_scores_toward_the_narrative_claim_with_a_checkpoint(
    settings, monkeypatch, tmp_path
):
    ctx = _ctx(tmp_path, settings, checkpoint=True, monkeypatch=monkeypatch)
    S._stance_stage(ctx)

    assert _FakeStance.seen == [(CLAIM, POST)]
    table = ctx.store.read("record_scores").set_index("record_id")
    assert table.loc["r:en", "stance"] == "deny"
    assert table.loc["r:en", "stance_conf"] == pytest.approx(0.9)
    assert dict(table.loc["r:en", "model_versions"])["stance"].startswith("v0.3.0+")
    assert not list(table.loc["r:en", "skip_reasons"])


def test_stance_stage_stays_null_without_a_checkpoint(settings, monkeypatch, tmp_path):
    ctx = _ctx(tmp_path, settings, checkpoint=False, monkeypatch=monkeypatch)
    out = S._stance_stage(ctx)

    assert _FakeStance.seen == []
    assert "not trained" in out[0]["note"]
    table = ctx.store.read("record_scores")
    assert not len(table) or table["stance"].isna().all()


def test_stance_stage_skips_hindi_and_says_why(settings, monkeypatch, tmp_path):
    ctx = _ctx(tmp_path, settings, checkpoint=True, monkeypatch=monkeypatch)
    S._stance_stage(ctx)

    assert all(HINDI not in pair for pair in _FakeStance.seen)
    table = ctx.store.read("record_scores").set_index("record_id")
    assert pd.isna(table.loc["r:hi", "stance"])
    assert list(table.loc["r:hi", "skip_reasons"]) == ["stance:unsupported_language"]


def test_stance_stage_gives_every_other_skip_a_reason_code(settings, monkeypatch, tmp_path):
    ctx = _ctx(tmp_path, settings, checkpoint=True, monkeypatch=monkeypatch)
    S._stance_stage(ctx)

    table = ctx.store.read("record_scores").set_index("record_id")
    assert list(table.loc["r:noise", "skip_reasons"]) == ["stance:no_narrative"]
    assert list(table.loc["r:claim", "skip_reasons"]) == ["stance:is_claim"]
    assert list(table.loc["r:short", "skip_reasons"]) == ["stance:text_too_short"]
    assert table["stance"].drop("r:en").isna().all()


def test_stance_stage_clears_a_stale_stance_and_keeps_other_reasons(
    settings, monkeypatch, tmp_path
):
    ctx = _ctx(tmp_path, settings, checkpoint=True, monkeypatch=monkeypatch)
    ctx.store.write(
        "record_scores",
        pd.DataFrame(
            [
                {"record_id": "r:noise", "source": "news", "stance": "support",
                 "stance_conf": 0.5, "skip_reasons": ["toxicity:unsupported_language"]},
            ]
        ),
    )
    S._stance_stage(ctx)
    S._stance_stage(ctx)

    table = ctx.store.read("record_scores").set_index("record_id")
    assert pd.isna(table.loc["r:noise", "stance"])
    assert list(table.loc["r:noise", "skip_reasons"]) == [
        "toxicity:unsupported_language",
        "stance:no_narrative",
    ]


def test_stance_stage_refuses_a_checkpoint_with_unresolved_labels(settings, monkeypatch, tmp_path):
    ctx = _ctx(tmp_path, settings, checkpoint=True, monkeypatch=monkeypatch)
    _FakeStance.refuse_with = "label_orientation_unresolved"
    out = S._stance_stage(ctx)

    assert "label_orientation_unresolved" in out[0]["note"]
    assert _FakeStance.seen == []
    table = ctx.store.read("record_scores")
    assert not len(table) or table["stance"].isna().all()


# ---------------------------------------------------------------------------
# which epoch the fine-tune keeps
# ---------------------------------------------------------------------------
class _TinyHead:
    """A real torch module small enough to train in milliseconds."""

    @staticmethod
    def build():
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                self.embed = torch.nn.Embedding(16, 4)
                self.head = torch.nn.Linear(4, 4)
                self.config = types.SimpleNamespace()

            def forward(self, input_ids, attention_mask):
                pooled = self.embed(input_ids).mean(dim=1)
                return types.SimpleNamespace(logits=self.head(pooled))

            def save_pretrained(self, path):
                torch.save(self.state_dict(), Path(path) / "weights.pt")

        return Tiny()


class _TinyTokenizer:
    def __call__(self, claims, posts, **kwargs):
        import torch

        n = len(claims)
        return {
            "input_ids": torch.arange(n * 3).reshape(n, 3) % 16,
            "attention_mask": torch.ones(n, 3, dtype=torch.long),
        }

    def save_pretrained(self, path):
        pass


def _fine_tune_with_val_losses(monkeypatch, tmp_path, favoured_probability):
    """Run three epochs where validation quality follows ``favoured_probability``.

    Returns the classifier, the model, the weights seen at the end of each
    epoch, and the output directory.
    """
    classifier = StanceClassifier()
    classifier.gradient_checkpointing = False
    classifier.batch_size = 2
    model = _TinyHead.build()
    monkeypatch.setattr(
        classifier, "_load_base", lambda name, num_labels=None: (_TinyTokenizer(), model)
    )

    snapshots: list[dict] = []
    calls = {"n": 0}

    def fake_probabilities(model_, tokenizer, pairs, device):
        snapshots.append({k: v.detach().clone() for k, v in model_.state_dict().items()})
        p = favoured_probability[calls["n"]]
        calls["n"] += 1
        # Probability p on the true class (always class 0 in this test).
        row = [p] + [(1 - p) / 3] * 3
        return np.tile(row, (len(pairs), 1))

    monkeypatch.setattr(classifier, "_raw_probabilities", fake_probabilities)

    pairs = [(f"claim {i}", f"post {i}") for i in range(6)]
    labels = np.zeros(6, dtype=int)
    out = tmp_path / "ckpt"
    classifier.fine_tune(pairs, labels, pairs[:2], labels[:2], output_dir=out, epochs=3)
    return classifier, model, snapshots, out


def test_fine_tune_saves_the_best_validation_epoch_not_the_last(monkeypatch, tmp_path):
    import json

    import torch

    # Epoch 2 has the lowest validation loss; epoch 3 is worse.
    classifier, model, snapshots, out = _fine_tune_with_val_losses(
        monkeypatch, tmp_path, [0.5, 0.9, 0.3]
    )
    assert len(snapshots) == 3

    assert json.loads((out / "best_epoch.json").read_text())["epoch"] == 2
    saved = torch.load(out / "weights.pt")
    for name, tensor in snapshots[1].items():
        assert torch.equal(saved[name], tensor), name
    assert any(not torch.equal(saved[n], t) for n, t in snapshots[2].items())

    # The model scored in the rest of this run carries the saved weights too.
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, snapshots[1][name]), name
    assert classifier._model is model
    assert json.loads((out / "training.json").read_text())["best_epoch"] == 2


def test_fine_tune_leaves_a_checkpoint_after_the_first_epoch(monkeypatch, tmp_path):
    import torch

    # Validation never improves after epoch one, so epoch one is what is saved.
    _, _, snapshots, out = _fine_tune_with_val_losses(monkeypatch, tmp_path, [0.9, 0.6, 0.3])
    saved = torch.load(out / "weights.pt")
    for name, tensor in snapshots[0].items():
        assert torch.equal(saved[name], tensor), name


def test_fine_tune_refuses_an_empty_validation_set(monkeypatch, tmp_path):
    classifier = StanceClassifier()
    monkeypatch.setattr(
        classifier, "_load_base", lambda *a, **k: pytest.fail("loaded a model before refusing")
    )
    pairs = [("claim", "post")] * 4
    out = tmp_path / "ckpt"
    with pytest.raises(ValueError, match="validation"):
        classifier.fine_tune(pairs, np.zeros(4, dtype=int), [], np.zeros(0, dtype=int),
                             output_dir=out, epochs=1)
    assert not out.exists()


def test_fine_tune_never_picks_a_non_finite_epoch_as_best(monkeypatch, tmp_path):
    import json

    # Epoch one is NaN; epochs two and three are finite and three is worse.
    classifier = StanceClassifier()
    classifier.gradient_checkpointing = False
    classifier.batch_size = 2
    model = _TinyHead.build()
    monkeypatch.setattr(
        classifier, "_load_base", lambda name, num_labels=None: (_TinyTokenizer(), model)
    )
    rows = iter([np.nan, 0.5, 0.0])

    def fake_probabilities(model_, tokenizer, pairs, device):
        p = next(rows)
        return np.tile([p] + [(1 - p) / 3] * 3, (len(pairs), 1))

    monkeypatch.setattr(classifier, "_raw_probabilities", fake_probabilities)
    pairs = [(f"claim {i}", f"post {i}") for i in range(6)]
    labels = np.zeros(6, dtype=int)
    out = tmp_path / "ckpt"
    result = classifier.fine_tune(pairs, labels, pairs[:2], labels[:2], output_dir=out, epochs=3)
    assert result.metadata["best_epoch"] == 2
    assert json.loads((out / "best_epoch.json").read_text())["epoch"] == 2


def test_fine_tune_refuses_when_no_epoch_has_a_finite_loss(monkeypatch, tmp_path):
    classifier = StanceClassifier()
    classifier.gradient_checkpointing = False
    classifier.batch_size = 2
    model = _TinyHead.build()
    monkeypatch.setattr(
        classifier, "_load_base", lambda name, num_labels=None: (_TinyTokenizer(), model)
    )
    monkeypatch.setattr(
        classifier,
        "_raw_probabilities",
        lambda m, t, pairs, d: np.full((len(pairs), 4), np.nan),
    )
    pairs = [(f"claim {i}", f"post {i}") for i in range(6)]
    labels = np.zeros(6, dtype=int)
    out = tmp_path / "ckpt"
    with pytest.raises(RuntimeError, match="finite"):
        classifier.fine_tune(pairs, labels, pairs[:2], labels[:2], output_dir=out, epochs=2)
    assert not (out / "best_epoch.json").exists()
    assert not (out / "training.json").exists()


def test_a_record_skipped_after_being_scored_loses_its_stance_version(
    settings, monkeypatch, tmp_path
):
    ctx = _ctx(tmp_path, settings, checkpoint=True, monkeypatch=monkeypatch)
    S._stance_stage(ctx)
    scored = ctx.store.read("record_scores").set_index("record_id")
    stamp = dict(scored.loc["r:en", "model_versions"])
    assert "stance" in stamp
    # Another scorer's stamp on the same row must survive the retraction.
    ctx.store.write(
        "record_scores",
        pd.DataFrame(
            [{"record_id": "r:en", "source": "news", "model_versions": {"toxicity": "v0.1.0"}}]
        ),
    )
    assert ("r:en",) in ctx.store.already_scored("record_scores", {"stance": stamp["stance"]})

    # The record leaves its narrative, so the next run skips it.
    ctx.store.write(
        "narrative_membership",
        ctx.store.read("narrative_membership").query("record_id != 'r:en'"),
        merge=False,
    )
    S._stance_stage(ctx)

    table = ctx.store.read("record_scores").set_index("record_id")
    versions = dict(table.loc["r:en", "model_versions"])
    assert "stance" not in versions
    assert versions["toxicity"] == "v0.1.0"
    assert list(table.loc["r:en", "skip_reasons"]) == ["stance:no_narrative"]
    assert ("r:en",) not in ctx.store.already_scored("record_scores", {"stance": stamp["stance"]})


def _with_short_first_representative(ctx, *, second_representative: bool):
    """Make ``r:claim`` a too-short premise, optionally with a usable runner-up."""
    ctx.records.loc[ctx.records["id"] == "r:claim", "text"] = "ok"
    membership = ctx.store.read("narrative_membership")
    if second_representative:
        # A second representative of the same narrative with usable text.
        ctx.records = pd.concat(
            [
                ctx.records,
                pd.DataFrame(
                    [{"id": "r:alt", "source": "news", "text": CLAIM + " (alt)", "lang": "en"}]
                ),
            ],
            ignore_index=True,
        )
        ctx.known_record_ids = set(ctx.records["id"])
        extra = membership.iloc[[0]].copy()
        extra["record_id"] = "r:alt"
        extra["membership_prob"] = 0.5
        membership = pd.concat([membership, extra], ignore_index=True)
    ctx.store.write("narrative_membership", membership, merge=False,
                    known_keys=ctx.known_record_ids)


def test_a_too_short_claim_is_not_used_and_the_skip_says_so(settings, monkeypatch, tmp_path):
    ctx = _ctx(tmp_path, settings, checkpoint=True, monkeypatch=monkeypatch)
    _with_short_first_representative(ctx, second_representative=False)
    S._stance_stage(ctx)

    assert _FakeStance.seen == []
    table = ctx.store.read("record_scores").set_index("record_id")
    assert list(table.loc["r:en", "skip_reasons"]) == ["stance:claim_text_too_short"]
    assert pd.isna(table.loc["r:en", "stance"])


def test_a_narrative_falls_back_to_its_next_qualifying_representative(
    settings, monkeypatch, tmp_path
):
    ctx = _ctx(tmp_path, settings, checkpoint=True, monkeypatch=monkeypatch)
    _with_short_first_representative(ctx, second_representative=True)
    S._stance_stage(ctx)

    assert (CLAIM + " (alt)", POST) in _FakeStance.seen
    assert all(claim != "ok" for claim, _ in _FakeStance.seen)
    table = ctx.store.read("record_scores").set_index("record_id")
    assert table.loc["r:en", "stance"] == "deny"
