"""Unit tests for the Git repository cloner and file discovery (Issue 5).

These tests never touch the network: a small local Git repository is created on
disk and cloned via a ``file://`` URL, which exercises the real clone code path
(branch selection, shallow depth, cleanup) hermetically in CI.
"""

import tempfile
from pathlib import Path

import pytest
from git import Repo

from reporag.ingestion.cloner import CloneError, FileInfo, RepoCloner

# Source files the fixture repo contains, by repo-relative path -> language.
EXPECTED_SOURCES = {
    "app.py": "python",
    "pkg/util.py": "python",
    "web/index.js": "javascript",
    "web/types.ts": "typescript",
}


def _commit_all(repo: Repo, message: str) -> None:
    """Stage everything and commit using the repo-local identity."""
    repo.git.add(A=True)
    repo.git.commit("-m", message)


def _uri(path: Path) -> str:
    return path.resolve().as_uri()


def _spy_mkdtemp(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every temp dir the cloner creates so cleanup can be asserted."""
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def spy(*args: object, **kwargs: object) -> str:
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", spy)
    return created


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a local Git repo on ``main`` with two commits and mixed files."""
    root = tmp_path / "origin"
    root.mkdir()
    repo = Repo.init(root)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test User")
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("commit", "gpgsign", "false")

    # Source files that should be discovered.
    (root / "app.py").write_text("def main():\n    return 0\n")
    (root / "pkg").mkdir()
    (root / "pkg" / "util.py").write_text("def helper():\n    return 1\n")
    (root / "web").mkdir()
    (root / "web" / "index.js").write_text("export const x = 1;\n")
    (root / "web" / "types.ts").write_text("export type T = number;\n")

    # Files and dirs that should be filtered or ignored.
    (root / "README.md").write_text("# Sample\n")
    (root / "data.txt").write_text("not source\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.js").write_text("module.exports = {};\n")

    _commit_all(repo, "initial commit")
    repo.git.branch("-M", "main")

    # A second commit so a depth-1 clone is observably shallower than a full one.
    (root / "app.py").write_text("def main():\n    return 42\n")
    _commit_all(repo, "second commit")
    return root


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def test_discover_files_filters_by_language(sample_repo: Path) -> None:
    """Only files whose extension maps to a language are in the manifest."""
    manifest = RepoCloner().discover_files(sample_repo)
    found = {info.path: info.language for info in manifest}
    assert found == EXPECTED_SOURCES


def test_discover_ignores_vendored_and_vcs_dirs(sample_repo: Path) -> None:
    """node_modules and .git are never walked into."""
    paths = [info.path for info in RepoCloner().discover_files(sample_repo)]
    assert not any(p.startswith("node_modules/") for p in paths)
    assert not any(p.startswith(".git/") for p in paths)


def test_manifest_entries_carry_path_language_and_size(sample_repo: Path) -> None:
    """Each manifest entry reports path, language, and real size in bytes."""
    manifest = RepoCloner().discover_files(sample_repo)
    by_path = {info.path: info for info in manifest}
    app = by_path["app.py"]
    assert isinstance(app, FileInfo)
    assert app.language == "python"
    assert app.size_bytes == (sample_repo / "app.py").stat().st_size
    assert app.size_bytes > 0


def test_manifest_is_sorted_by_path(sample_repo: Path) -> None:
    """The manifest is deterministically ordered by path."""
    paths = [info.path for info in RepoCloner().discover_files(sample_repo)]
    assert paths == sorted(paths)


def test_explicit_extensions_override(sample_repo: Path) -> None:
    """An explicit extension map makes discovery fully configurable."""
    manifest = RepoCloner(extensions={".md": "markdown"}).discover_files(sample_repo)
    assert [info.path for info in manifest] == ["README.md"]
    assert manifest[0].language == "markdown"


def test_languages_override_narrows_discovery(sample_repo: Path) -> None:
    """Restricting languages restricts which extensions are discovered."""
    cloner = RepoCloner(languages=["python"])
    languages = {info.language for info in cloner.discover_files(sample_repo)}
    assert languages == {"python"}


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------


def test_clone_and_discover_local_repo(sample_repo: Path) -> None:
    """Cloning a repo and discovering its files yields the expected manifest."""
    cloner = RepoCloner()
    manifest = cloner.clone_and_discover(_uri(sample_repo))
    found = {info.path: info.language for info in manifest}
    assert found == EXPECTED_SOURCES
    cloner.cleanup()


def test_clone_returns_path_and_tracks_last_clone(sample_repo: Path) -> None:
    """clone() returns a real working tree and records it for cleanup."""
    cloner = RepoCloner()
    path = cloner.clone(_uri(sample_repo))
    try:
        assert path.exists()
        assert cloner.last_clone_path == path
        assert (path / "app.py").exists()
    finally:
        cloner.cleanup()
    assert not path.exists()
    assert cloner.last_clone_path is None


def test_default_branch_is_autodetected(sample_repo: Path) -> None:
    """With no branch given, the remote's default branch is checked out."""
    cloner = RepoCloner()
    path = cloner.clone(_uri(sample_repo))
    try:
        assert Repo(path).active_branch.name == "main"
    finally:
        cloner.cleanup()


def test_branch_selection(sample_repo: Path) -> None:
    """A requested branch is cloned, and other branches' files are absent."""
    origin = Repo(sample_repo)
    origin.git.checkout("-b", "feature")
    (sample_repo / "feature_only.py").write_text("FEATURE = True\n")
    _commit_all(origin, "feature commit")
    origin.git.checkout("main")

    cloner = RepoCloner()
    feature = cloner.clone_and_discover(_uri(sample_repo), branch="feature")
    assert any(info.path == "feature_only.py" for info in feature)
    cloner.cleanup()

    main = cloner.clone_and_discover(_uri(sample_repo), branch="main")
    assert not any(info.path == "feature_only.py" for info in main)
    cloner.cleanup()


def test_shallow_clone_truncates_history(sample_repo: Path) -> None:
    """A shallow (depth-1) clone has a single commit; a full clone has more."""
    cloner = RepoCloner()
    shallow = cloner.clone(_uri(sample_repo), shallow=True)
    try:
        assert len(list(Repo(shallow).iter_commits())) == 1
    finally:
        cloner.cleanup()

    full = cloner.clone(_uri(sample_repo), shallow=False)
    try:
        assert len(list(Repo(full).iter_commits())) >= 2
    finally:
        cloner.cleanup()


# ---------------------------------------------------------------------------
# Failure handling and cleanup
# ---------------------------------------------------------------------------


def test_clone_error_cleans_up_temp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed clone raises CloneError and leaves no temp directory behind."""
    created = _spy_mkdtemp(monkeypatch)
    cloner = RepoCloner()

    with pytest.raises(CloneError):
        cloner.clone(str(tmp_path / "does-not-exist"))

    assert created, "a temp directory should have been created"
    assert not Path(created[0]).exists(), "temp dir must be removed on error"


def test_size_limit_rejects_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, sample_repo: Path
) -> None:
    """A repo over the size limit is rejected and its temp dir is removed."""
    created = _spy_mkdtemp(monkeypatch)
    cloner = RepoCloner(max_repo_size_mb=1e-6)

    with pytest.raises(CloneError, match="exceeds the configured"):
        cloner.clone(_uri(sample_repo))

    assert created
    assert not Path(created[0]).exists()
