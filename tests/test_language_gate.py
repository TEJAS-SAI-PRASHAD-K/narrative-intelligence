"""The per-module language gate.

Which languages a module may score is the `languages` list under that module in
`configs/models.yaml`. These tests protect three properties: a module with no
list fails loudly, an English-only module never admits Hindi, and every text
module declares a list at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from modeling import config as C
from modeling.config import ModuleLanguagesMissing, module_languages


@pytest.fixture
def settings(monkeypatch):
    C.get_settings.cache_clear()
    monkeypatch.delenv("SCORE_UNKNOWN_LANGUAGE", raising=False)
    yield C.get_settings()
    C.get_settings.cache_clear()


@pytest.fixture
def fake_models_yaml(monkeypatch):
    """Replace configs/models.yaml with a dict the test controls."""

    def install(body: dict):
        monkeypatch.setattr(C, "models_config", lambda: body)

    return install


def test_english_only_module_rejects_hindi_and_admits_english(settings):
    assert settings.language_allowed("toxicity", "en")
    assert settings.language_allowed("toxicity", "EN")
    assert not settings.language_allowed("toxicity", "hi")


def test_laya_admits_english_and_hindi_only(settings):
    assert settings.language_allowed("laya", "en")
    assert settings.language_allowed("laya", "hi")
    assert not settings.language_allowed("laya", "de")


def test_null_language_follows_score_unknown_language(settings, monkeypatch):
    for null in (None, float("nan"), np.nan):
        assert settings.language_allowed("misinfo", null) is True
    C.get_settings.cache_clear()
    monkeypatch.setenv("SCORE_UNKNOWN_LANGUAGE", "false")
    strict = C.get_settings()
    assert strict.language_allowed("misinfo", None) is False
    assert strict.language_allowed("misinfo", float("nan")) is False


def test_wildcard_module_admits_every_record_even_with_strict_unknown_policy(
    settings, monkeypatch
):
    """`embed` embeds every record today, so its entry must not drop nulls."""
    C.get_settings.cache_clear()
    monkeypatch.setenv("SCORE_UNKNOWN_LANGUAGE", "false")
    strict = C.get_settings()
    for lang in ("en", "hi", "ta", None, float("nan")):
        assert strict.language_allowed("embed", lang) is True


def test_module_without_a_list_raises_instead_of_scoring_everything(settings, fake_models_yaml):
    fake_models_yaml({"newmodel": {"version": "v0.1.0"}})
    with pytest.raises(ModuleLanguagesMissing, match="newmodel"):
        settings.language_allowed("newmodel", "en")
    with pytest.raises(ModuleLanguagesMissing):
        settings.language_allowed("newmodel", "hi")


def test_unknown_module_name_raises(settings):
    with pytest.raises(ModuleLanguagesMissing, match="no_such_module"):
        settings.language_allowed("no_such_module", "en")


@pytest.mark.parametrize("bad", [[], "en", None])
def test_empty_or_malformed_list_raises(settings, fake_models_yaml, bad):
    fake_models_yaml({"m": {"languages": bad}})
    with pytest.raises(ModuleLanguagesMissing):
        module_languages("m")


def test_every_module_in_models_yaml_declares_languages():
    """A new module cannot be added without taking a stand on language."""
    body = C.models_config()
    modules = [n for n, v in body.items() if n != "defaults" and isinstance(v, dict)]
    missing = [name for name in modules if "languages" not in body[name]]
    assert modules and not missing, f"modules without `languages`: {missing}"


def test_english_only_text_modules_list_only_english():
    for name in ("misinfo", "stance", "toxicity", "sentiment", "emotion"):
        assert module_languages(name) == ("en",), name


# ---------------------------------------------------------------------------
# acceptance: no English-only module writes a score for a Hindi record
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402

from modeling import scoring as S  # noqa: E402
from modeling.aux import run_aux_pass  # noqa: E402
from modeling.aux.base import AuxScorer  # noqa: E402
from modeling.io import ScoredStore  # noqa: E402
from modeling.registry import Checkpoint  # noqa: E402

TEXT = "a claim that is comfortably longer than twenty characters"
HINDI = "यह एक दावा है जो बीस अक्षरों से काफी लंबा है"
FRAME_ROWS = [
    {"id": "r:en", "source": "news", "text": TEXT, "lang": "en"},
    {"id": "r:hi", "source": "news", "text": HINDI, "lang": "hi"},
    {"id": "r:null", "source": "news", "text": TEXT + " again", "lang": None},
]


def _frame() -> pd.DataFrame:
    return pd.DataFrame(FRAME_ROWS)


def _fake_aux(monkeypatch):
    """Replace every aux scorer's model with one that returns a fixed value."""
    values = {
        "toxicity": lambda t: 0.1,
        "sentiment": lambda t: {"sentiment": "neutral", "sentiment_score": 0.0},
        "emotion": lambda t: {"neutral": 1.0},
    }
    from modeling.aux import TEXT_SCORERS

    for cls in TEXT_SCORERS:
        fn = values[cls.module]
        monkeypatch.setattr(cls, "load", lambda self: True)
        monkeypatch.setattr(
            cls, "score_texts", lambda self, texts, fn=fn: [fn(t) for t in texts]
        )
    assert issubclass(TEXT_SCORERS[0], AuxScorer)


def test_aux_pass_writes_no_score_for_hindi_and_says_why(settings, monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    C.get_settings.cache_clear()
    _fake_aux(monkeypatch)
    out = run_aux_pass(
        _frame(), None, settings=C.get_settings(), scorers=["toxicity", "sentiment", "emotion"]
    )
    by_id = out.set_index("record_id")

    for module in ("toxicity", "sentiment", "emotion"):
        assert module in by_id.loc["r:en", "model_versions"]
        assert module not in by_id.loc["r:hi", "model_versions"]
        assert f"{module}:unsupported_language" in by_id.loc["r:hi", "skip_reasons"]
        # null lang follows score_unknown_language, which defaults to true
        assert module in by_id.loc["r:null", "model_versions"]
    assert pd.isna(by_id.loc["r:hi", "toxicity"])
    assert pd.isna(by_id.loc["r:hi", "sentiment_score"])
    assert not any((by_id.loc["r:hi", "emotion"] or {}).values())
    assert by_id.loc["r:en", "skip_reasons"] == []


class _FakeMisinfo:
    seen: list[str] = []

    def __init__(self, settings):
        pass

    def load(self, path):
        return True

    def predict(self, texts):
        type(self).seen = list(texts)
        return [0.5] * len(texts)


def _misinfo_ctx(tmp_path, settings, monkeypatch, *, dry_run=False):
    import modeling.registry as registry
    import modeling.text.misinfo_clf as clf

    monkeypatch.setattr(
        registry, "resolve",
        lambda module, version, required=False: Checkpoint(
            module="misinfo", version=version, path=tmp_path, origin="local", sha256="ab" * 32
        ),
    )
    monkeypatch.setattr(clf, "MisinfoClassifier", _FakeMisinfo)
    records = _frame()
    return S.StageContext(
        settings=settings,
        reader=None,
        store=ScoredStore(settings, root=tmp_path / "scored"),
        records=records,
        authors=pd.DataFrame(),
        demo=True,
        dry_run=dry_run,
        known_record_ids=set(records["id"]),
    )


def test_misinfo_stage_scores_en_and_null_but_not_hindi_and_records_the_reason(
    settings, monkeypatch, tmp_path
):
    ctx = _misinfo_ctx(tmp_path, settings, monkeypatch)
    S._misinfo_stage(ctx)

    assert HINDI not in _FakeMisinfo.seen
    assert len(_FakeMisinfo.seen) == 2
    table = ctx.store.read("record_scores").set_index("record_id")
    assert table.loc["r:en", "misinfo_prob"] == pytest.approx(0.5)
    assert table.loc["r:null", "misinfo_prob"] == pytest.approx(0.5)
    assert pd.isna(table.loc["r:hi", "misinfo_prob"])
    assert list(table.loc["r:hi", "skip_reasons"]) == ["misinfo:unsupported_language"]
    assert not table.loc["r:en", "skip_reasons"]


def test_misinfo_skip_reason_keeps_the_aux_reasons_and_is_idempotent(
    settings, monkeypatch, tmp_path
):
    ctx = _misinfo_ctx(tmp_path, settings, monkeypatch)
    ctx.store.write(
        "record_scores",
        pd.DataFrame(
            [
                {
                    "record_id": "r:hi",
                    "source": "news",
                    "skip_reasons": ["toxicity:unsupported_language"],
                }
            ]
        ),
    )
    S._misinfo_stage(ctx)
    S._misinfo_stage(ctx)
    table = ctx.store.read("record_scores").set_index("record_id")
    assert list(table.loc["r:hi", "skip_reasons"]) == [
        "toxicity:unsupported_language",
        "misinfo:unsupported_language",
    ]


def test_rerunning_the_aux_stage_keeps_the_misinfo_skip_reasons(settings, monkeypatch, tmp_path):
    ctx = _misinfo_ctx(tmp_path, settings, monkeypatch)
    _fake_aux(monkeypatch)
    import modeling.aux as aux

    real = aux.run_aux_pass
    monkeypatch.setattr(
        aux,
        "run_aux_pass",
        lambda records, authors, settings=None: real(
            records, authors, settings=settings, scorers=["toxicity", "sentiment", "emotion"]
        ),
    )
    S._misinfo_stage(ctx)
    before = ctx.store.read("record_scores").set_index("record_id")
    assert list(before.loc["r:hi", "skip_reasons"]) == ["misinfo:unsupported_language"]

    S._aux_stage(ctx)

    after = ctx.store.read("record_scores").set_index("record_id")
    assert list(after.loc["r:hi", "skip_reasons"]) == [
        "toxicity:unsupported_language",
        "sentiment:unsupported_language",
        "emotion:unsupported_language",
        "misinfo:unsupported_language",
    ]
    assert not after.loc["r:en", "skip_reasons"].tolist()


def test_misinfo_stage_dry_run_writes_nothing_but_reports_the_skip(
    settings, monkeypatch, tmp_path
):
    ctx = _misinfo_ctx(tmp_path, settings, monkeypatch, dry_run=True)
    out = S._misinfo_stage(ctx)
    assert not len(ctx.store.read("record_scores"))
    assert "1 unsupported_language" in out[0]["note"]


class _RecordingEmbedder:
    seen: list[str] = []

    def __init__(self, settings=None, model_name=None):
        pass

    def load(self):
        return True

    def embed_records(self, records, **kwargs):
        type(self).seen = list(records["id"])
        return []


def _cluster_ctx(tmp_path, settings, monkeypatch):
    import modeling.text.embed as embed

    monkeypatch.setattr(embed, "Embedder", _RecordingEmbedder)
    records = _frame()
    return S.StageContext(
        settings=settings,
        reader=None,
        store=ScoredStore(settings, root=tmp_path / "scored"),
        records=records,
        authors=pd.DataFrame(),
        demo=True,
        dry_run=True,
    )


def test_cluster_stage_embeds_every_language_while_embed_lists_wildcard(
    settings, monkeypatch, tmp_path
):
    """Clustering behaviour is unchanged by the gate: Hindi is still embedded."""
    ctx = _cluster_ctx(tmp_path, settings, monkeypatch)
    S._cluster_stage(ctx)
    assert _RecordingEmbedder.seen == ["r:en", "r:hi", "r:null"]


def test_cluster_stage_follows_a_narrowed_embed_list(settings, monkeypatch, tmp_path):
    body = {**C.models_config(), "embed": {**C.models_config()["embed"], "languages": ["en"]}}
    monkeypatch.setattr(C, "models_config", lambda: body)
    ctx = _cluster_ctx(tmp_path, settings, monkeypatch)
    S._cluster_stage(ctx)
    assert _RecordingEmbedder.seen == ["r:en", "r:null"]
