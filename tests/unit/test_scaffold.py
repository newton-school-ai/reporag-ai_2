"""Scaffold smoke tests for Issue 1.

Validates that the project directory structure is correct, all packages
are importable, configuration loads without errors, and required
infrastructure files exist.
"""

import importlib
from pathlib import Path

# Root of the project (two levels up from tests/unit/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestPackageImports:
    """Verify every src/reporag subpackage is importable."""

    def test_top_level_package(self) -> None:
        """Top-level reporag package imports without error."""
        mod = importlib.import_module("reporag")
        assert mod is not None

    def test_ingestion_package(self) -> None:
        """Ingestion subpackage imports without error."""
        mod = importlib.import_module("reporag.ingestion")
        assert mod is not None

    def test_graph_package(self) -> None:
        """Graph subpackage imports without error."""
        mod = importlib.import_module("reporag.graph")
        assert mod is not None

    def test_embedding_package(self) -> None:
        """Embedding subpackage imports without error."""
        mod = importlib.import_module("reporag.embedding")
        assert mod is not None

    def test_retrieval_package(self) -> None:
        """Retrieval subpackage imports without error."""
        mod = importlib.import_module("reporag.retrieval")
        assert mod is not None

    def test_agent_package(self) -> None:
        """Agent subpackage imports without error."""
        mod = importlib.import_module("reporag.agent")
        assert mod is not None

    def test_generation_package(self) -> None:
        """Generation subpackage imports without error."""
        mod = importlib.import_module("reporag.generation")
        assert mod is not None

    def test_evaluation_package(self) -> None:
        """Evaluation subpackage imports without error."""
        mod = importlib.import_module("reporag.evaluation")
        assert mod is not None

    def test_api_package(self) -> None:
        """API subpackage imports without error."""
        mod = importlib.import_module("reporag.api")
        assert mod is not None

    def test_api_routes_package(self) -> None:
        """API routes subpackage imports without error."""
        mod = importlib.import_module("reporag.api.routes")
        assert mod is not None

    def test_api_middleware_package(self) -> None:
        """API middleware subpackage imports without error."""
        mod = importlib.import_module("reporag.api.middleware")
        assert mod is not None


class TestInitFiles:
    """Verify __init__.py files exist in every package directory."""

    EXPECTED_PACKAGES = [
        "src/reporag",
        "src/reporag/ingestion",
        "src/reporag/graph",
        "src/reporag/embedding",
        "src/reporag/retrieval",
        "src/reporag/agent",
        "src/reporag/generation",
        "src/reporag/evaluation",
        "src/reporag/api",
        "src/reporag/api/routes",
        "src/reporag/api/middleware",
    ]

    def test_init_files_exist(self) -> None:
        """Every package directory has an __init__.py file."""
        missing = []
        for pkg_path in self.EXPECTED_PACKAGES:
            init_file = PROJECT_ROOT / pkg_path / "__init__.py"
            if not init_file.exists():
                missing.append(str(pkg_path))
        assert missing == [], f"Missing __init__.py in: {missing}"


class TestConfiguration:
    """Verify the Settings configuration loads correctly."""

    def test_settings_loads(self) -> None:
        """Settings object can be instantiated without error."""
        from reporag.config import Settings

        s = Settings()
        assert s is not None

    def test_settings_defaults(self) -> None:
        """Default settings have expected values."""
        from reporag.config import Settings

        s = Settings()
        assert s.app_env == "development"
        assert s.app_port == 8000
        assert s.neo4j_uri == "bolt://localhost:7687"
        assert s.qdrant_url == "http://localhost:6333"
        assert s.llm_provider == "openai"

    def test_settings_singleton(self) -> None:
        """Module-level settings singleton is available."""
        from reporag.config import settings

        assert settings is not None
        assert settings.app_port == 8000


class TestInfrastructureFiles:
    """Verify required infrastructure files exist at the project root."""

    def test_dockerfile_exists(self) -> None:
        """Dockerfile exists at project root."""
        assert (PROJECT_ROOT / "Dockerfile").is_file()

    def test_docker_compose_exists(self) -> None:
        """docker-compose.yml exists at project root."""
        assert (PROJECT_ROOT / "docker-compose.yml").is_file()

    def test_ci_workflow_exists(self) -> None:
        """CI workflow file exists."""
        assert (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").is_file()

    def test_requirements_exists(self) -> None:
        """requirements.txt exists at project root."""
        assert (PROJECT_ROOT / "requirements.txt").is_file()

    def test_pyproject_exists(self) -> None:
        """pyproject.toml exists at project root."""
        assert (PROJECT_ROOT / "pyproject.toml").is_file()

    def test_env_example_exists(self) -> None:
        """.env.example exists for developer onboarding."""
        assert (PROJECT_ROOT / ".env.example").is_file()

    def test_gitignore_exists(self) -> None:
        """.gitignore exists at project root."""
        assert (PROJECT_ROOT / ".gitignore").is_file()


class TestDockerComposeServices:
    """Verify docker-compose.yml defines the required services."""

    def test_services_defined(self) -> None:
        """docker-compose.yml defines api, neo4j, qdrant, postgres."""
        import yaml

        compose_path = PROJECT_ROOT / "docker-compose.yml"
        with open(compose_path) as f:
            compose = yaml.safe_load(f)

        services = set(compose.get("services", {}).keys())
        required = {"api", "neo4j", "qdrant", "postgres"}
        missing = required - services
        assert missing == set(), f"Missing services: {missing}"


class TestCIWorkflow:
    """Verify CI workflow has the required jobs."""

    def test_ci_jobs_defined(self) -> None:
        """CI workflow defines ascii-guard, lint, and test jobs."""
        import yaml

        ci_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        with open(ci_path) as f:
            ci = yaml.safe_load(f)

        jobs = set(ci.get("jobs", {}).keys())
        required = {"ascii-guard", "lint", "test"}
        missing = required - jobs
        assert missing == set(), f"Missing CI jobs: {missing}"

    def test_ci_triggers_on_dev(self) -> None:
        """CI triggers on push and PR to dev branch."""
        import yaml

        ci_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        with open(ci_path) as f:
            ci = yaml.safe_load(f)

        on = ci.get("on", ci.get(True, {}))
        assert "dev" in on.get("push", {}).get("branches", [])
        assert "dev" in on.get("pull_request", {}).get("branches", [])


class TestModuleDocstrings:
    """Verify all source modules have module-level docstrings."""

    SOURCE_MODULES = [
        "reporag.config",
        "reporag.ingestion.cloner",
        "reporag.ingestion.parser",
        "reporag.ingestion.symbol_extractor",
        "reporag.ingestion.chunker",
        "reporag.graph.call_graph",
        "reporag.graph.dependency_graph",
        "reporag.graph.symbol_table",
        "reporag.embedding.code_embedder",
        "reporag.embedding.doc_embedder",
        "reporag.embedding.index_builder",
        "reporag.retrieval.vector_search",
        "reporag.retrieval.bm25_search",
        "reporag.retrieval.graph_traversal",
        "reporag.retrieval.fusion",
        "reporag.retrieval.reranker",
        "reporag.agent.planner",
        "reporag.agent.router",
        "reporag.agent.executor",
        "reporag.agent.synthesizer",
        "reporag.api.main",
        "reporag.api.routes.health",
    ]

    def test_all_modules_have_docstrings(self) -> None:
        """Every source module has a non-empty module docstring."""
        missing = []
        for mod_name in self.SOURCE_MODULES:
            mod = importlib.import_module(mod_name)
            if not mod.__doc__ or not mod.__doc__.strip():
                missing.append(mod_name)
        assert missing == [], f"Modules missing docstrings: {missing}"
