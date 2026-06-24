import pytest
from pydantic import ValidationError

from reporag.config import Settings


def test_invalid_app_env():
    with pytest.raises(ValidationError):
        Settings(app_env="invalid")


def test_invalid_llm_provider():
    with pytest.raises(ValidationError):
        Settings(llm_provider="unsupported")


def test_type_coercion():
    settings = Settings(app_port="9000")
    assert settings.app_port == 9000

    with pytest.raises(ValidationError):
        Settings(app_port="not-a-number")


def test_production_missing_secrets():
    # Production without setting a real secret key should fail
    with pytest.raises(ValueError) as exc:
        Settings(
            app_env="production",
            secret_key="change-me",
            jwt_secret_key="super-jwt-key",
            llm_provider="openai",
            openai_api_key="",
        )
    assert "SECRET_KEY must be set" in str(exc.value)
    assert "JWT_SECRET_KEY must be set" in str(exc.value)
    assert "OPENAI_API_KEY must be set" in str(exc.value)


def test_production_valid_secrets():
    # Production with valid secrets should succeed
    settings = Settings(
        app_env="production",
        secret_key="a-real-secure-key",
        jwt_secret_key="a-real-jwt-secure-key",
        llm_provider="openai",
        openai_api_key="sk-real-key",
    )
    assert settings.app_env == "production"
    assert settings.openai_api_key.get_secret_value() == "sk-real-key"


def test_production_anthropic_validation():
    with pytest.raises(ValueError) as exc:
        Settings(
            app_env="production",
            secret_key="a-real-secure-key",
            jwt_secret_key="a-real-jwt-secure-key",
            llm_provider="anthropic",
            anthropic_api_key="",
        )
    assert "ANTHROPIC_API_KEY must be set" in str(exc.value)


def test_env_file_documents_all_fields():
    # Ensure .env.example contains all variables
    with open(".env.example") as f:
        env_content = f.read()

    for field in Settings.model_fields:
        assert (
            field.upper() in env_content
        ), f"Field {field.upper()} missing in .env.example"


def test_secretstr_hides_value():
    settings = Settings(neo4j_password="my-password")
    assert "my-password" not in str(settings.neo4j_password)
    assert settings.neo4j_password.get_secret_value() == "my-password"
