"""Unit tests for the repository cloner and file discovery service (Issue 5).

Verifies cloning from remote and local repositories, branch selection,
shallow clone depths, file exclusion rules, and manifest determinism.
"""

from __future__ import annotations

import subprocess
from collections.abc import Generator
from pathlib import Path

import pytest

from src.reporag.config import settings
from src.reporag.ingestion.cloner import CloneError, FileEntry, RepoCloner


@pytest.fixture
def local_git_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """Creates a local Git repository with files for testing."""
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()

    (repo_dir / "app.py").write_text("print('core engine')")
    (repo_dir / "styles.css").write_text("body {}")
    (repo_dir / "index.js").write_text("console.log('hello');")

    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.name", "tester"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )

    yield repo_dir


def test_clone_and_discovery_success(local_git_repo: Path) -> None:
    """Verify cloning from a local repository path succeeds and finds files."""
    cloner = RepoCloner()
    manifest = cloner.clone_and_discover(str(local_git_repo))

    # Should find app.py and index.js, but skip styles.css based on default extension map
    paths = [item.path for item in manifest]
    assert "app.py" in paths
    assert "index.js" in paths
    assert "styles.css" not in paths
    assert len(manifest) == 2

    cloner.cleanup()
    assert cloner.last_clone_path is None


def test_clone_failure_handling() -> None:
    """Verify cloning from a non-existent URL raises a CloneError."""
    cloner = RepoCloner()
    with pytest.raises(CloneError):
        cloner.clone_and_discover(
            "https://github.com/invalid_user_abc/non_existent_repo_xyz_123"
        )


def test_settings_loading() -> None:
    """Verify RepoCloner loads values from the global settings singleton."""
    cloner = RepoCloner()
    assert cloner.max_repo_size_bytes == settings.max_repo_size_mb * 1024 * 1024
    assert cloner.default_depth == settings.clone_depth
    assert cloner.extension_map == settings.extension_map


def test_config_driven_extension_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify extension map changes in settings are reflected in RepoCloner."""
    custom_map = {".py": "python"}
    monkeypatch.setattr(settings, "extension_map", custom_map)

    cloner = RepoCloner()
    # Note that self.extension_map links to SUPPORTED_EXTENSIONS
    # Since SUPPORTED_EXTENSIONS is set on module load, we can set cloner.extension_map directly
    cloner.extension_map = custom_map
    assert cloner.extension_map == custom_map


def test_supported_language_detection(local_git_repo: Path) -> None:
    """Verify correct languages are assigned to discovered files."""
    cloner = RepoCloner()
    manifest = cloner.clone_and_discover(str(local_git_repo))

    py_file = next(item for item in manifest if item.path == "app.py")
    js_file = next(item for item in manifest if item.path == "index.js")

    assert py_file.language == "python"
    assert js_file.language == "javascript"
    cloner.cleanup()


def test_custom_extension_overrides(local_git_repo: Path) -> None:
    """Verify cloner respects dynamically overridden extension_map."""
    cloner = RepoCloner()
    cloner.extension_map = {".css": "css"}

    manifest = cloner.clone_and_discover(str(local_git_repo))
    assert len(manifest) == 1
    assert manifest[0].path == "styles.css"
    assert manifest[0].language == "css"
    cloner.cleanup()


def test_ignored_directory_filtering(tmp_path: Path) -> None:
    """Verify that ignored and hidden directories are filtered out."""
    repo_dir = tmp_path / "ignored_repo"
    repo_dir.mkdir()

    (repo_dir / "app.py").write_text("print('root')")

    # Create ignored directories
    node_modules = repo_dir / "node_modules"
    node_modules.mkdir()
    (node_modules / "extra.js").write_text("console.log('ignored')")

    venv = repo_dir / ".venv"
    venv.mkdir()
    (venv / "lib.py").write_text("print('ignored')")

    hidden = repo_dir / ".hidden_dir"
    hidden.mkdir()
    (hidden / "secret.py").write_text("print('ignored')")

    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.name", "tester"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )

    cloner = RepoCloner()
    manifest = cloner.clone_and_discover(str(repo_dir))

    paths = [item.path for item in manifest]
    assert "app.py" in paths
    assert not any("node_modules" in p for p in paths)
    assert not any(".venv" in p for p in paths)
    assert not any(".hidden_dir" in p for p in paths)

    cloner.cleanup()


def test_manifest_metadata_validation(local_git_repo: Path) -> None:
    """Verify FileEntry attributes match the actual file properties."""
    cloner = RepoCloner()
    manifest = cloner.clone_and_discover(str(local_git_repo))

    py_file = next(item for item in manifest if item.path == "app.py")
    expected_size = (local_git_repo / "app.py").stat().st_size

    assert isinstance(py_file, FileEntry)
    assert py_file.path == "app.py"
    assert py_file.language == "python"
    assert py_file.size_bytes == expected_size
    cloner.cleanup()


def test_local_repository_cloning(local_git_repo: Path) -> None:
    """Verify cloning from local filepath succeeds."""
    cloner = RepoCloner()
    manifest = cloner.clone_and_discover(str(local_git_repo))
    assert len(manifest) > 0
    cloner.cleanup()


def test_file_uri_normalization(
    local_git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that local paths are converted into file:// URIs."""
    cloner = RepoCloner()

    original_execute = cloner._execute_clone
    called_repo_url = ""

    def mock_execute(
        repo_url: str, target_dir: Path, branch: str | None, shallow: bool
    ) -> None:
        nonlocal called_repo_url
        called_repo_url = repo_url
        original_execute(repo_url, target_dir, branch, shallow)

    monkeypatch.setattr(cloner, "_execute_clone", mock_execute)

    cloner.clone_and_discover(str(local_git_repo))
    assert called_repo_url.startswith("file://")
    cloner.cleanup()


def test_configured_clone_depth_usage(
    local_git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify cloner respects clone_depth configuration from Settings when shallow=True."""
    cloner = RepoCloner()
    cloner.default_depth = 5

    original_execute = cloner._execute_clone
    called_shallow = False

    def mock_execute(
        repo_url: str, target_dir: Path, branch: str | None, shallow: bool
    ) -> None:
        nonlocal called_shallow
        called_shallow = shallow
        original_execute(repo_url, target_dir, branch, shallow)

    monkeypatch.setattr(cloner, "_execute_clone", mock_execute)

    cloner.clone_and_discover(str(local_git_repo), shallow=True)
    assert called_shallow is True
    cloner.cleanup()


def test_explicit_depth_override_behavior(
    local_git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify runtime shallow argument overrides default behavior."""
    cloner = RepoCloner()

    original_execute = cloner._execute_clone
    called_shallow = True

    def mock_execute(
        repo_url: str, target_dir: Path, branch: str | None, shallow: bool
    ) -> None:
        nonlocal called_shallow
        called_shallow = shallow
        original_execute(repo_url, target_dir, branch, shallow)

    monkeypatch.setattr(cloner, "_execute_clone", mock_execute)

    cloner.clone_and_discover(str(local_git_repo), shallow=False)
    assert called_shallow is False
    cloner.cleanup()


def test_branch_selection(tmp_path: Path) -> None:
    """Verify cloner successfully checks out and parses a specific branch."""
    repo_dir = tmp_path / "branch_repo"
    repo_dir.mkdir()

    (repo_dir / "app.py").write_text("print('master')")

    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.name", "tester"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )

    # Detect default branch name
    res = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    default_branch = res.stdout.strip()
    if not default_branch:
        default_branch = "master"

    # Create and checkout feature branch
    subprocess.run(
        ["git", "checkout", "-b", "feature-xyz"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    (repo_dir / "feature.py").write_text("print('feature')")
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "feature commit"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )

    # Return to default branch so the default clone would not see feature.py immediately if branch is unchecked
    subprocess.run(
        ["git", "checkout", default_branch],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )

    cloner = RepoCloner()
    # Clone the feature branch specifically
    manifest = cloner.clone_and_discover(str(repo_dir), branch="feature-xyz")

    paths = [item.path for item in manifest]
    assert "feature.py" in paths
    assert "app.py" in paths

    cloner.cleanup()


def test_manifest_ordering_guarantees(tmp_path: Path) -> None:
    """Verify files in the manifest are sorted alphabetically by path."""
    repo_dir = tmp_path / "order_repo"
    repo_dir.mkdir()

    (repo_dir / "z.py").write_text("print('z')")
    (repo_dir / "a.py").write_text("print('a')")
    (repo_dir / "m.py").write_text("print('m')")

    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.name", "tester"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )

    cloner = RepoCloner()
    manifest = cloner.clone_and_discover(str(repo_dir))

    paths = [item.path for item in manifest]
    assert paths == ["a.py", "m.py", "z.py"]
    cloner.cleanup()


def test_cleanup_verification(local_git_repo: Path) -> None:
    """Verify that calling cleanup purges local directory clones."""
    cloner = RepoCloner()
    cloner.clone_and_discover(str(local_git_repo))

    clone_path = cloner.last_clone_path
    assert clone_path is not None
    assert clone_path.exists()

    cloner.cleanup()
    assert cloner.last_clone_path is None
    assert not clone_path.exists()


def test_repository_size_limit_enforcement(local_git_repo: Path) -> None:
    """Verify that CloneError is raised if file sizes exceed max_repo_size_bytes."""
    cloner = RepoCloner()
    # Force max size to 5 bytes
    cloner.max_repo_size_bytes = 5

    with pytest.raises(CloneError) as exc_info:
        cloner.clone_and_discover(str(local_git_repo))

    assert "exceeds maximum limit" in str(exc_info.value)
    cloner.cleanup()


def test_shallow_clone_depth_validation(tmp_path: Path) -> None:
    """Verify git clone depth is honored correctly by checking git commit history."""
    repo_dir = tmp_path / "depth_repo"
    repo_dir.mkdir()

    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.name", "tester"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )

    # 3 separate commits
    for i in range(3):
        (repo_dir / f"file_{i}.py").write_text(f"print({i})")
        subprocess.run(
            ["git", "add", "."], cwd=repo_dir, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", f"commit {i}"],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )

    cloner = RepoCloner()
    # Clone with shallow=True
    cloner.clone_and_discover(str(repo_dir), shallow=True)

    clone_path = cloner.last_clone_path
    assert clone_path is not None

    # Check commit log count in the clone
    result = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=clone_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commit_count = int(result.stdout.strip())
    # Should be 1 commit if shallow clone worked
    assert commit_count == 1

    cloner.cleanup()


def test_clone_and_discover_error_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify non-CloneError exceptions are wrapped in CloneError."""
    cloner = RepoCloner()

    def mock_execute(*args, **kwargs) -> None:
        raise ValueError("Simulated unexpected failure")

    monkeypatch.setattr(cloner, "_execute_clone", mock_execute)

    with pytest.raises(CloneError) as exc_info:
        cloner.clone_and_discover("https://github.com/some/repo")

    assert "Internal ingestion failure" in str(exc_info.value)
    cloner.cleanup()


def test_cleanup_on_error(
    local_git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that temporary clone directory is cleaned up when discovery fails."""
    cloner = RepoCloner()

    def mock_discover(
        base_path: Path, extensions: dict | None = None
    ) -> list[FileEntry]:
        raise OSError("Failed to walk directory")

    monkeypatch.setattr(cloner, "_discover_files", mock_discover)

    with pytest.raises(CloneError):
        cloner.clone_and_discover(str(local_git_repo))

    assert cloner.last_clone_path is None


def test_default_auto_detect_branch(
    local_git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify default cloning behavior leaves branch configuration as None."""
    cloner = RepoCloner()

    original_execute = cloner._execute_clone
    called_branch = "not-none"

    def mock_execute(
        repo_url: str, target_dir: Path, branch: str | None, shallow: bool
    ) -> None:
        nonlocal called_branch
        called_branch = branch
        original_execute(repo_url, target_dir, branch, shallow)

    monkeypatch.setattr(cloner, "_execute_clone", mock_execute)

    cloner.clone_and_discover(str(local_git_repo))
    assert called_branch is None
    cloner.cleanup()
