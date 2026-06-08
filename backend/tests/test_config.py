"""
tests/test_config.py
─────────────────────
Tests for app/core/config.py — Settings loading, validation and computed properties.
"""

from __future__ import annotations

import os

import pytest

from app.core.config import Settings, get_settings


class TestSettingsDefaults:
    """Verify default values when no environment variables are set."""

    def test_default_app_name(self) -> None:
        s = Settings()
        assert s.app_name == "open-deep-research"

    def test_default_app_env(self) -> None:
        s = Settings()
        assert s.app_env == "development"

    def test_default_port(self) -> None:
        s = Settings()
        assert s.port == 8000

    def test_default_ollama_base_url(self) -> None:
        s = Settings()
        assert s.ollama_base_url in ("http://localhost:11434", "http://ollama:11434")

    def test_default_ollama_model(self) -> None:
        s = Settings()
        assert s.ollama_model == "llama3.2:3b"

    def test_default_log_level(self) -> None:
        s = Settings()
        assert s.log_level == "INFO"

    def test_default_cors_origins(self) -> None:
        s = Settings()
        assert isinstance(s.cors_origins, list)
        assert len(s.cors_origins) > 0

    def test_default_max_retries(self) -> None:
        s = Settings()
        assert s.ollama_max_retries == 3


class TestSettingsFromEnvVars:
    """Verify settings can be overridden via environment variables."""

    def test_app_name_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_NAME", "my-research-app")
        s = Settings()
        assert s.app_name == "my-research-app"

    def test_port_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PORT", "9000")
        s = Settings()
        assert s.port == 9000

    def test_ollama_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-ollama:11434")
        s = Settings()
        assert s.ollama_base_url == "http://remote-ollama:11434"

    def test_log_level_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.log_level == "DEBUG"

    def test_cors_origins_json_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "CORS_ORIGINS", '["http://example.com", "http://app.example.com"]'
        )
        s = Settings()
        assert "http://example.com" in s.cors_origins
        assert "http://app.example.com" in s.cors_origins


class TestSettingsValidation:
    """Verify validators reject invalid configurations."""

    def test_invalid_port_too_low(self) -> None:
        with pytest.raises(Exception):
            Settings(PORT=0)

    def test_invalid_port_too_high(self) -> None:
        with pytest.raises(Exception):
            Settings(PORT=99999)

    def test_invalid_log_level(self) -> None:
        with pytest.raises(Exception):
            Settings(LOG_LEVEL="VERBOSE")

    def test_invalid_app_env(self) -> None:
        with pytest.raises(Exception):
            Settings(APP_ENV="staging")

    def test_debug_true_in_production_raises(self) -> None:
        with pytest.raises(Exception, match="DEBUG must be False"):
            Settings(APP_ENV="production", LOG_FORMAT="json", DEBUG=True)


class TestSettingsComputedProperties:
    """Verify computed properties return correct values."""

    def test_is_development_true(self) -> None:
        s = Settings(APP_ENV="development")
        assert s.is_development is True
        assert s.is_production is False
        assert s.is_testing is False

    def test_is_production_true(self) -> None:
        s = Settings(APP_ENV="production", LOG_FORMAT="json", DEBUG=False)
        assert s.is_production is True
        assert s.is_development is False

    def test_is_testing_true(self) -> None:
        s = Settings(APP_ENV="testing")
        assert s.is_testing is True

    def test_ollama_api_url_strips_trailing_slash(self) -> None:
        s = Settings(OLLAMA_BASE_URL="http://localhost:11434/")
        assert not s.ollama_api_url.endswith("/")
        assert s.ollama_api_url == "http://localhost:11434"

    def test_ollama_api_url_no_trailing_slash(self) -> None:
        s = Settings(OLLAMA_BASE_URL="http://localhost:11434")
        assert s.ollama_api_url == "http://localhost:11434"


class TestGetSettings:
    """Verify the cached settings singleton behaves correctly."""

    def test_get_settings_returns_settings_instance(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_is_cached(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
