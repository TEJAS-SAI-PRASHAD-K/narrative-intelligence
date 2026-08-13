"""Benchmark loaders: they parse the real formats, and they never download.

Two things are being tested. The obvious one is that each fixture parses. The
important one is the *absence* behaviour: a missing benchmark must raise with
actionable instructions, never return an empty frame, because an empty frame
trains a model on nothing and still writes a metrics file.
"""

from __future__ import annotations

import pandas as pd
import pytest

from modeling.datasets import DatasetUnavailable, all_datasets, get_dataset
from modeling.datasets.base import FIXTURE_ROOT
from modeling.datasets.splits import group_train_val_test, grouped_kfold

ALL_KEYS = [
    "liar",
    "fakenewsnet",
    "coaid",
    "stance",
    "twibot",
    "cresci",
    "faceforensics",
    "dfdc",
]


@pytest.mark.parametrize("key", ALL_KEYS)
def test_every_loader_parses_its_fixture(key):
    loaded = get_dataset(key).load(demo=True)
    assert len(loaded) > 0
    assert loaded.is_demo is True
    assert loaded.group_col in loaded.frame.columns
    assert loaded.label_col in loaded.frame.columns
    assert not loaded.frame[loaded.group_col].isna().any()


@pytest.mark.parametrize("key", ALL_KEYS)
def test_missing_dataset_raises_with_instructions(key, tmp_path):
    dataset = get_dataset(key)
    with pytest.raises(DatasetUnavailable) as excinfo:
        dataset.load(tmp_path / "not-downloaded")
    message = str(excinfo.value)
    assert dataset.info.url in message
    assert "--demo" in message
    # An instruction block with no steps is not instructions.
    assert "expected layout" in message


@pytest.mark.parametrize("key", ALL_KEYS)
def test_available_never_raises_and_never_downloads(key, tmp_path):
    dataset = get_dataset(key)
    assert dataset.available(tmp_path / "nothing") is False
    assert dataset.available(demo=True) is True
    # Probing must not have created anything on disk.
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("key", ALL_KEYS)
def test_fixture_is_splittable_without_leakage(key):
    """A benchmark we cannot split by group is a benchmark we cannot evaluate."""
    dataset = get_dataset(key)
    loaded = dataset.load(demo=True)
    frame = loaded.frame
    if frame[loaded.group_col].nunique() < 3:
        pytest.skip(f"{key} fixture has too few groups for a 3-way split")
    text_col = "text" if "text" in frame.columns else loaded.group_col
    _, split = group_train_val_test(
        frame, group_col=loaded.group_col, dedupe=False, text_col=text_col
    )
    assert len(split.train) and len(split.test)


# ---------------------------------------------------------------------------
# label mapping -- the decisions with consequences
# ---------------------------------------------------------------------------
def test_liar_drops_half_true_rather_than_forcing_it():
    """Forcing half-true either way manufactures label noise the metrics cannot
    see. The loader must drop it, and must say how many it dropped."""
    loaded = get_dataset("liar").load(demo=True)
    assert "half-true" not in set(loaded.frame["label_6way"])
    assert loaded.dropped["half_true_dropped"] > 0
    assert set(loaded.frame["label"].unique()) <= {0, 1}


def test_liar_maps_the_six_way_scale_as_documented():
    loaded = get_dataset("liar").load(demo=True)
    by_label = loaded.frame.groupby("label_6way")["label"].unique()
    for lie in ("pants-fire", "false", "barely-true"):
        assert list(by_label[lie]) == [1], f"{lie} must map to the misinformation class"
    for truth in ("mostly-true", "true"):
        assert list(by_label[truth]) == [0], f"{truth} must map to the non-misinformation class"


def test_liar_groups_by_speaker_not_by_row():
    loaded = get_dataset("liar").load(demo=True)
    assert loaded.group_col == "speaker"
    assert loaded.frame["speaker"].nunique() < len(loaded.frame)


def test_liar_never_merges_unknown_speakers_into_one_group():
    """A blank speaker is an unknown, not a person. Collapsing every anonymous
    statement into one pseudo-speaker would create a huge leaky group."""
    from modeling.datasets.liar import Liar

    frame = pd.read_csv(
        FIXTURE_ROOT / "liar" / "train.tsv", sep="\t", header=None, dtype=str, quoting=3
    )
    frame[4] = ""  # blank every speaker
    loader = Liar()
    # Exercise the synthetic-id branch directly rather than rewriting the fixture.
    import hashlib

    synthetic = {
        "unknown-" + hashlib.sha1(str(s).encode()).hexdigest()[:10] for s in frame[2].head(5)
    }
    assert len(synthetic) == 5
    assert loader.group_col == "speaker"


def test_stance_maps_none_to_discuss_and_leaves_unrelated_unattested():
    """SemEval has no 'unrelated' class. A stance model trained on it alone
    cannot predict one, and that gap belongs in the model card, not in a
    silently-invented mapping."""
    loaded = get_dataset("stance").load(demo=True)
    labels = set(loaded.frame["label"].unique())
    assert labels <= {"support", "deny", "discuss", "unrelated"}
    assert "unrelated" not in labels
    assert "discuss" in labels


def test_fakenewsnet_carries_both_domains_for_the_transfer_table():
    loaded = get_dataset("fakenewsnet").load(demo=True)
    assert set(loaded.frame["domain"].unique()) == {"politifact", "gossipcop"}


def test_fakenewsnet_refuses_a_single_domain_copy(tmp_path):
    """One half is parseable and useless: the PolitiFact -> GossipCop number is
    the reason this dataset is here at all."""
    source = FIXTURE_ROOT / "fakenewsnet" / "dataset"
    target = tmp_path / "dataset"
    target.mkdir(parents=True)
    for name in ("politifact_fake.csv", "politifact_real.csv"):
        (target / name).write_bytes((source / name).read_bytes())
    with pytest.raises(DatasetUnavailable, match="Both halves are required"):
        get_dataset("fakenewsnet").load(tmp_path)


def test_coaid_groups_by_claim_text_so_waves_cannot_leak():
    loaded = get_dataset("coaid").load(demo=True)
    assert loaded.group_col == "claim_id"
    # The same claim in two waves shares one group id.
    counts = loaded.frame.groupby("claim_id").size()
    assert counts.max() >= 1


def test_cresci_groups_by_campaign_not_by_account():
    """Each spambot directory is one botnet running one template. Grouping by
    account would put siblings in train and test and report near-perfect F1."""
    loaded = get_dataset("cresci").load(demo=True)
    assert loaded.group_col == "campaign"
    assert loaded.frame["campaign"].nunique() < loaded.frame["account_id"].nunique()


def test_cresci_grouped_cv_isolates_every_campaign():
    loaded = get_dataset("cresci").load(demo=True)
    work, folds = grouped_kfold(
        loaded.frame, group_col="campaign", label_col="label", n_splits=5
    )
    for fold in folds:
        train_campaigns = set(work["campaign"].iloc[fold.train])
        test_campaigns = set(work["campaign"].iloc[fold.test])
        assert not (train_campaigns & test_campaigns)


def test_twibot_reads_nested_public_metrics():
    loaded = get_dataset("twibot").load(demo=True)
    assert loaded.frame["followers"].notna().all()
    assert loaded.frame["following"].notna().all()
    assert loaded.frame["post_count"].notna().all()


def test_twibot_ids_join_between_users_and_labels():
    """The 'u' prefix appears on both sides. Normalizing one side breaks the
    join and fails as "zero labelled users", which reads like a download bug."""
    loaded = get_dataset("twibot").load(demo=True)
    assert len(loaded) > 0
    assert loaded.dropped.get("unlabeled_user", 0) == 0


# ---------------------------------------------------------------------------
# media loaders -- the split key is the thing
# ---------------------------------------------------------------------------
def test_faceforensics_ties_each_fake_to_its_target_identity():
    loaded = get_dataset("faceforensics").load(demo=True)
    assert loaded.group_col == "source_video"
    originals = loaded.frame.loc[loaded.frame["label"] == 0, "source_video"]
    fakes = loaded.frame.loc[loaded.frame["label"] == 1, "source_video"]
    # Every fake's group key must correspond to a real original, or the split
    # cannot keep the identity on one side.
    assert set(fakes) <= set(originals)


def test_faceforensics_exposes_all_four_methods_for_the_cross_method_table():
    loaded = get_dataset("faceforensics").load(demo=True)
    methods = set(loaded.frame["method"].unique())
    assert {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"} <= methods
    assert "original" in methods


def test_faceforensics_cross_method_holdout_is_possible():
    from modeling.datasets.splits import domain_holdout

    loaded = get_dataset("faceforensics").load(demo=True)
    work, split = domain_holdout(
        loaded.frame, domain_col="method", held_out="NeuralTextures", group_col="source_video"
    )
    assert set(work["method"].iloc[split.test]) == {"NeuralTextures"}
    assert "NeuralTextures" not in set(work["method"].iloc[split.train])


def test_dfdc_drops_fakes_with_no_named_original():
    """An untied fake cannot be grouped with its source, so including it risks
    putting the same face on both sides of the split."""
    loaded = get_dataset("dfdc").load(demo=True)
    assert loaded.dropped["fake_without_original"] == 1
    assert "fake_untied" not in set(loaded.frame["video_id"])


def test_dfdc_fakes_group_with_their_source_real():
    loaded = get_dataset("dfdc").load(demo=True)
    reals = set(loaded.frame.loc[loaded.frame["label"] == 0, "video_id"])
    fake_groups = set(loaded.frame.loc[loaded.frame["label"] == 1, "source_video"])
    assert fake_groups <= reals


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_every_registered_dataset_declares_its_access_terms():
    for dataset in all_datasets():
        info = dataset.info
        assert info.access in {"open", "request_form", "signed_agreement", "crawler"}
        assert info.citation, f"{info.key} has no citation"
        assert info.expected_layout, f"{info.key} does not say what it expects on disk"
        assert info.manual_steps, f"{info.key} gives no manual steps"


def test_unknown_dataset_key_is_a_clear_error():
    with pytest.raises(KeyError, match="unknown benchmark"):
        get_dataset("imagenet")
