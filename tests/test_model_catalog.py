"""Tests for the provisionable model catalog."""

from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

import backend.config as config_module
from backend.model_catalog import (
    MODEL_CONTEXT_LIMITS,
    MODEL_COSTS,
    MODEL_DEFAULTS,
    load_model_catalog,
)


class TestLoadModelCatalog:
    """Tests for loading the bundled catalog and overrides."""

    def test_bundled_defaults_reference_known_models(self, tmp_path):
        """Every provider default should have pricing and a context window."""
        catalog = load_model_catalog(tmp_path / "missing.yaml")

        for provider in ("anthropic", "openai", "gemini"):
            defaults = catalog.providers[provider]
            assert defaults.analysis in catalog.models
            assert defaults.generation in catalog.models

    def test_override_is_deep_merged(self, tmp_path):
        """Override file should change only the keys it sets."""
        override = tmp_path / "models.yaml"
        override.write_text(yaml.dump({
            "providers": {"anthropic": {"generation": "claude-haiku-4-5"}},
            "models": {"my-model": {"input_cost": 0.5, "output_cost": 1.5, "context_window": 64000}},
        }))

        bundled = load_model_catalog(tmp_path / "missing.yaml")
        catalog = load_model_catalog(override)

        assert catalog.providers["anthropic"].generation == "claude-haiku-4-5"
        assert catalog.providers["anthropic"].analysis == bundled.providers["anthropic"].analysis
        assert catalog.providers["openai"] == bundled.providers["openai"]
        assert catalog.models["my-model"].context_window == 64000
        assert set(bundled.models) < set(catalog.models)

    def test_override_path_from_env(self, tmp_path, monkeypatch):
        """MEDIASAGE_MODELS_FILE should select the override file."""
        override = tmp_path / "custom.yaml"
        override.write_text(yaml.dump({"providers": {"gemini": {"analysis": "gemini-3.1-pro-preview"}}}))
        monkeypatch.setenv("MEDIASAGE_MODELS_FILE", str(override))

        assert load_model_catalog().providers["gemini"].analysis == "gemini-3.1-pro-preview"

    def test_invalid_model_entry_rejected(self, tmp_path):
        """Incomplete model entries should fail validation."""
        override = tmp_path / "models.yaml"
        override.write_text(yaml.dump({"models": {"bad": {"input_cost": 1.0}}}))

        with pytest.raises(ValidationError):
            load_model_catalog(override)

    def test_lookup_tables_built_from_catalog(self):
        """Legacy lookup tables should expose catalog data."""
        analysis = MODEL_DEFAULTS["anthropic"]["analysis"]
        assert analysis in MODEL_COSTS
        assert MODEL_CONTEXT_LIMITS[analysis] >= 200_000
        assert MODEL_DEFAULTS["ollama"] == {"analysis": "", "generation": ""}


class TestProviderSwitchPersistence:
    """Catalog defaults should not be frozen into config.user.yaml."""

    def test_default_models_not_persisted(self, tmp_path, monkeypatch):
        for var in ["LLM_PROVIDER", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)
        user_config = tmp_path / "config.user.yaml"
        user_config.write_text(yaml.dump({"llm": {"provider": "gemini", "model_analysis": "old-model"}}))
        monkeypatch.setattr(config_module, "USER_CONFIG_PATH", user_config)
        monkeypatch.setattr(config_module, "_config", None)

        with patch("backend.config.load_yaml_config", return_value={}):
            config = config_module.update_config_values({"llm_provider": "anthropic"})

        assert config.llm.model_analysis == MODEL_DEFAULTS["anthropic"]["analysis"]
        saved = yaml.safe_load(user_config.read_text())
        assert saved["llm"]["provider"] == "anthropic"
        assert "model_analysis" not in saved["llm"]
        assert "model_generation" not in saved["llm"]

    def test_explicit_models_persisted(self, tmp_path, monkeypatch):
        for var in ["LLM_PROVIDER", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)
        user_config = tmp_path / "config.user.yaml"
        monkeypatch.setattr(config_module, "USER_CONFIG_PATH", user_config)
        monkeypatch.setattr(config_module, "_config", None)

        with patch("backend.config.load_yaml_config", return_value={}):
            config_module.update_config_values(
                {"llm_provider": "openai", "model_generation": "gpt-5.6-luna"}
            )

        saved = yaml.safe_load(user_config.read_text())
        assert saved["llm"]["model_generation"] == "gpt-5.6-luna"
        assert "model_analysis" not in saved["llm"]
