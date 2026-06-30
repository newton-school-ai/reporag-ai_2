"""Unit tests for the repository cloner and file discovery service (Issue 5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from git import Repo
from pydantic import SecretStr

from reporag.config import Settings
from reporag.ingestion.cloner import (
    CloneError,
    FileInfo,
    RepoCloner,
    normalize_repo_url,
)


def _make_settings(**overrides: object) -> Settings:
    """Build isolated Settings for cloner tests."""
    base = {
        "app_env": "development",
        "secret_key": SecretStr("test-secret"),
        "jwt_secret_key": SecretStr("test-jwt"),
        "openai_api_key": SecretStr("sk-test"),
        "max_repo_size_mb": 500,
        "clone_depth": 1,
        "supported_languages": "python,javascript,typescript",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _init_local_repo(
    path: Path,
    files: dict[str, str],
    *,
    branch: str = "main",
    extra_branches: dict[str, dict[str, str]] | None = None,
) -> Path:
    """Create a local Git repository with committed files."""
    path.mkdir(parents=True, exist_ok=True)
    repo = Repo.init(path, initial_branch=branch)
    repo.config_writer().set_value("user", "email", "test@example.com").release()
    repo.config_writer().set_value("user", "name", "Test User").release()

    def write_files(file_map: dict[str, str]) -> None:
        for relative_path, content in file_map.items():
            target = path / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    write_files(files)
    repo.index.add([str(path / rel) for rel in files])
    repo.index.commit("initial commit")

    if extra_branches:
        for branch_name, branch_files in extra_branches.items():
            repo.git.checkout("-b", branch_name)
            write_files(branch_files)
            if branch_files:
                repo.index.add([str(path / rel) for rel in branch_files])
                repo.index.commit(f"add {branch_name} files")

    return path


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """A small local repository with mixed source and ignored files."""
    return _init_local_repo(
        tmp_path / "sample",
        {
            "src/main.py": "print('hello')\n",
            "src/util.js": "export const x = 1;\n",
            "README.md": "# sample\n",
            "node_modules/pkg/index.js": "module.exports = {};\n",
        },
    )


@pytest.fixture
def cloner() -> RepoCloner:
    """RepoCloner backed by default test settings."""
    return RepoCloner(settings=_make_settings())


class TestExtensionConfiguration:
    def test_extension_map_loaded_from_settings(self, cloner: RepoCloner):
        mapping = cloner.extension_map()
        assert mapping["python"] == (".py",)
        assert ".ts" in mapping["typescript"]
        assert ".js" in mapping["javascript"]

    def test_supported_language_detection(self, cloner: RepoCloner, sample_repo: Path):
        manifest = cloner.discover(sample_repo)
        languages = {item.language for item in manifest}
        assert languages == {"python", "javascript"}

    def test_custom_extension_override(self, sample_repo: Path):
        cloner = RepoCloner(
            settings=_make_settings(supported_languages="python"),
            extensions={"python": (".py", ".pyw")},
        )
        (sample_repo / "launcher.pyw").write_text("pass\n", encoding="utf-8")
        manifest = cloner.discover(sample_repo)
        paths = {item.path for item in manifest}
        assert "launcher.pyw" in paths


class TestFileDiscovery:
    def test_ignored_directories_excluded(self, cloner: RepoCloner, sample_repo: Path):
        manifest = cloner.discover(sample_repo)
        paths = {item.path for item in manifest}
        assert "node_modules/pkg/index.js" not in paths
        assert "src/main.py" in paths

    def test_manifest_contains_path_language_and_size(
        self, cloner: RepoCloner, sample_repo: Path
    ):
        manifest = cloner.discover(sample_repo)
        main_py = next(item for item in manifest if item.path == "src/main.py")
        assert isinstance(main_py, FileInfo)
        assert main_py.language == "python"
        assert main_py.size_bytes == len("print('hello')\n")

    def test_manifest_sorted_by_path(self, cloner: RepoCloner, sample_repo: Path):
        manifest = cloner.discover(sample_repo)
        paths = [item.path for item in manifest]
        assert paths == sorted(paths)


class TestCloning:
    def test_clone_local_repository(
        self, cloner: RepoCloner, sample_repo: Path, tmp_path: Path
    ):
        clone_root = cloner.clone(str(sample_repo))
        try:
            assert clone_root.exists()
            assert (clone_root / "src" / "main.py").is_file()
            assert cloner.last_clone_path == clone_root
        finally:
            cloner.cleanup()

    def test_local_path_normalized_to_file_uri(self, sample_repo: Path):
        normalized = normalize_repo_url(str(sample_repo))
        assert normalized.startswith("file://")
        assert sample_repo.name in normalized

    def test_uses_configured_clone_depth_by_default(
        self, sample_repo: Path, tmp_path: Path
    ):
        cloner = RepoCloner(settings=_make_settings(clone_depth=1))
        clone_root = cloner.clone(str(sample_repo))
        try:
            assert (clone_root / ".git" / "shallow").exists()
        finally:
            cloner.cleanup()

    def test_explicit_depth_override_performs_full_clone(
        self, sample_repo: Path, tmp_path: Path
    ):
        cloner = RepoCloner(settings=_make_settings(clone_depth=1))
        clone_root = cloner.clone(str(sample_repo), depth=0)
        try:
            assert not (clone_root / ".git" / "shallow").exists()
        finally:
            cloner.cleanup()

    def test_branch_selection_checks_out_correct_branch(self, tmp_path: Path):
        repo_path = _init_local_repo(
            tmp_path / "branched",
            {"main.py": "on main\n"},
            branch="main",
            extra_branches={"feature": {"main.py": "on feature\n"}},
        )
        cloner = RepoCloner(settings=_make_settings())
        clone_root = cloner.clone(str(repo_path), branch="feature")
        try:
            assert (clone_root / "main.py").read_text(
                encoding="utf-8"
            ) == "on feature\n"
        finally:
            cloner.cleanup()

    def test_shallow_clone_depth_validation(self, sample_repo: Path):
        cloner = RepoCloner(settings=_make_settings(clone_depth=1))
        clone_root = cloner.clone(str(sample_repo), depth=1)
        try:
            assert (clone_root / ".git" / "shallow").is_file()
        finally:
            cloner.cleanup()

    def test_last_clone_path_tracked_after_clone(
        self, cloner: RepoCloner, sample_repo: Path
    ):
        assert cloner.last_clone_path is None
        clone_root = cloner.clone(str(sample_repo))
        try:
            assert cloner.last_clone_path == clone_root
        finally:
            cloner.cleanup()
            assert cloner.last_clone_path is None


class TestErrorHandlingAndCleanup:
    def test_clone_invalid_repo_raises_clone_error(self, cloner: RepoCloner):
        with pytest.raises(CloneError, match="Failed to clone"):
            cloner.clone("/path/that/does/not/exist")

        assert cloner.last_clone_path is None

    def test_cleanup_removes_cloned_directory(
        self, cloner: RepoCloner, sample_repo: Path
    ):
        clone_root = cloner.clone(str(sample_repo))
        assert clone_root.exists()
        cloner.cleanup()
        assert not clone_root.exists()
        assert cloner.last_clone_path is None

    def test_repo_size_limit_raises_clone_error(self, tmp_path: Path):
        large_content = "x" * 2048
        repo_path = _init_local_repo(
            tmp_path / "large",
            {"big.py": large_content},
        )
        cloner = RepoCloner(settings=_make_settings(max_repo_size_mb=0))
        with pytest.raises(CloneError, match="exceeds limit"):
            cloner.clone(str(repo_path))

        assert cloner.last_clone_path is None

    def test_clone_and_discover_propagates_clone_error(self, cloner: RepoCloner):
        with pytest.raises(CloneError):
            cloner.clone_and_discover("/definitely/missing/repo")

        assert cloner.last_clone_path is None

    def test_clone_and_discover_returns_manifest(
        self, cloner: RepoCloner, sample_repo: Path
    ):
        manifest = cloner.clone_and_discover(str(sample_repo))
        try:
            assert len(manifest) >= 2
            assert all(isinstance(item, FileInfo) for item in manifest)
        finally:
            cloner.cleanup()
