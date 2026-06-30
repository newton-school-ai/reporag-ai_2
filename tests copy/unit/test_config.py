"""Unit tests for the configuration module (Issue 4).

These tests construct ``Settings`` with ``_env_file=None`` so they never read a
developer's local ``.env`` -- the behavior under test is fully driven by the
arguments/environment each test sets, which keeps them hermetic in CI.
"""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from reporag.config import Settings, get_settings, settings

REPO_ROOT = Path(__file__).resolve().parents[2]

# Fields that hold sensitive values and must be SecretStr.
SECRET_FIELDS = [
    "secret_key",
    "neo4j_password",
    "openai_api_key",
    "anthropic_api_key",
    "google_client_secret",
    "jwt_secret_key",
]


# ---------------------------------------------------------------------------
# Import / singleton
# ---------------------------------------------------------------------------


def test_settings_importable():
    """The module-level ``settings`` singleton is importable and validated."""
    assert isinstance(settings, Settings)


def test_get_settings_is_cached():
    """``get_settings`` returns the same cached instance on every call."""
    assert get_settings() is get_settings()


def test_defaults_are_dev_friendly():
    """With no configuration the app imports with safe local defaults."""
    s = Settings(_env_file=None)
    assert s.app_env == "development"
    assert s.is_production is False
    assert s.database_url == "sqlite:///./reporag.db"
    assert s.llm_provider == "openai"


# ---------------------------------------------------------------------------
# Secrets are SecretStr and do not leak
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", SECRET_FIELDS)
def test_sensitive_fields_are_secretstr(field):
    """Every sensitive field is typed as SecretStr."""
    s = Settings(_env_file=None)
    assert isinstance(getattr(s, field), SecretStr)


def test_secret_value_is_not_leaked_in_repr_or_dump():
    """SecretStr keeps the real value out of str()/repr and JSON dumps."""
    s = Settings(_env_file=None, secret_key="topsecret-value")
    assert s.secret_key.get_secret_value() == "topsecret-value"
    assert "topsecret-value" not in str(s)
    assert "topsecret-value" not in repr(s)
    assert "topsecret-value" not in s.model_dump_json()


# ---------------------------------------------------------------------------
# Type / value validation
# ---------------------------------------------------------------------------


def test_invalid_app_env_raises():
    """An unknown APP_ENV is rejected by the Literal type."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="bogus")


def test_invalid_llm_provider_raises():
    """An unsupported LLM provider is rejected by the Literal type."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="foo")


def test_invalid_int_field_raises():
    """Non-integer values for int fields raise a ValidationError."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_port="not-a-number")


# ---------------------------------------------------------------------------
# Production secret requirements (the "required vars" acceptance criterion)
# ---------------------------------------------------------------------------


def test_production_missing_secrets_raises():
    """Production with placeholder secrets fails validation, naming each var."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            app_env="production",
            secret_key="change-me",
            jwt_secret_key="change-me-to-a-random-string",
            openai_api_key="",
            anthropic_api_key="",
        )
    message = str(exc_info.value)
    assert "SECRET_KEY" in message
    assert "JWT_SECRET_KEY" in message
    assert "OPENAI_API_KEY" in message


def test_production_with_real_secrets_ok():
    """Production validates once the required secrets are real values."""
    s = Settings(
        _env_file=None,
        app_env="production",
        secret_key="a-real-app-secret",
        jwt_secret_key="a-real-jwt-secret",
        openai_api_key="sk-a-real-openai-key",
    )
    assert s.is_production is True
    assert s.active_llm_api_key.get_secret_value() == "sk-a-real-openai-key"


def test_production_anthropic_requires_anthropic_key():
    """Switching provider changes which API key is required in production."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            app_env="production",
            llm_provider="anthropic",
            secret_key="a-real-app-secret",
            jwt_secret_key="a-real-jwt-secret",
            openai_api_key="sk-a-real-openai-key",
            anthropic_api_key="",
        )
    message = str(exc_info.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "OPENAI_API_KEY" not in message


def test_non_production_allows_placeholder_secrets():
    """Staging/dev/test never require real secrets, so import stays zero-config."""
    s = Settings(_env_file=None, app_env="staging")
    assert s.is_production is False


# ---------------------------------------------------------------------------
# Environment overrides and helpers
# ---------------------------------------------------------------------------


def test_env_var_overrides_default(monkeypatch):
    """Environment variables override defaults (case-insensitive names)."""
    monkeypatch.setenv("APP_PORT", "9999")
    monkeypatch.setenv("app_host", "127.0.0.1")
    s = Settings(_env_file=None)
    assert s.app_port == 9999
    assert s.app_host == "127.0.0.1"


def test_supported_languages_list():
    """The comma-separated languages string parses into a clean list."""
    s = Settings(_env_file=None, supported_languages="python, javascript ,, typescript")
    assert s.supported_languages_list == ["python", "javascript", "typescript"]


def test_active_llm_api_key_follows_provider():
    """``active_llm_api_key`` returns the key for the configured provider."""
    s = Settings(
        _env_file=None,
        llm_provider="anthropic",
        openai_api_key="sk-openai",
        anthropic_api_key="sk-anthropic",
    )
    assert s.active_llm_api_key.get_secret_value() == "sk-anthropic"


# ---------------------------------------------------------------------------
# .env.example documents every setting
# ---------------------------------------------------------------------------


def test_env_example_documents_every_field():
    """Every Settings field has a matching variable documented in .env.example."""
    env_example = (REPO_ROOT / ".env.example").read_text()
    for field_name in Settings.model_fields:
        assert (
            field_name.upper() in env_example
        ), f"{field_name.upper()} is missing from .env.example"
