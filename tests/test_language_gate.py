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
