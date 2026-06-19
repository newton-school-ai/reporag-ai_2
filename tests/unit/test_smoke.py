"""Smoke tests for the RepoRAG scaffold.

Verifies the package is importable and configuration loads with sane defaults.
Real per-module unit tests are added in their corresponding issues.
"""

from reporag.config import Settings, settings


def test_settings_singleton_is_loaded():
    assert isinstance(settings, Settings)


def test_settings_defaults():
    fresh = Settings()
    assert fresh.app_port == 8000
    assert fresh.app_host == "0.0.0.0"
    assert fresh.vector_search_top_k > 0
    assert fresh.rrf_constant == 60
