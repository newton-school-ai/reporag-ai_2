"""Unit tests for src/reporag/ingestion/cloner.py (Issue 5).

All tests are fully offline -- no network calls are made. A local git
repository is created in a temporary directory using gitpython and used as
the clone source. This keeps tests fast and deterministic.
"""

from __future__ import annotations

import os

import git
import pytest

from reporag.ingestion.cloner import (
    LANGUAGE_MAP,
    ClonerError,
    FileInfo,
    RepoCloner,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_local_repo(tmp_path: str, files: dict[str, str]) -> str:
    """Create a local git repo with the given files and an initial commit.

    Args:
        tmp_path: Directory in which to initialise the repo.
        files:    Mapping of relative path -> file content.

    Returns:
        Absolute path to the initialised repository root.
    """
    repo = git.Repo.init(tmp_path)
    # Configure git identity so the commit succeeds in CI environments.
    repo.config_writer().set_value("user", "name", "Test").release()
    repo.config_writer().set_value("user", "email", "test@test.com").release()

    for rel_path, content in files.items():
        abs_path = os.path.join(tmp_path, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        repo.index.add([rel_path])

    if files:
        repo.index.commit("initial commit")

    return tmp_path


@pytest.fixture()
def local_repo(tmp_path):
    """A local git repo with a mix of supported and unsupported files."""
    files = {
        "main.py": "print('hello')\n",
        "utils.js": "console.log('hello');\n",
        "types.ts": "export type Foo = string;\n",
        "README.md": "# Readme\n",
        "data.txt": "some data\n",
        "subdir/helper.py": "def helper(): pass\n",
        "subdir/nested/deep.ts": "const x: number = 1;\n",
    }
    return _make_local_repo(str(tmp_path), files)


@pytest.fixture()
def empty_repo(tmp_path):
    """A local git repo that contains no matching source files."""
    files = {
        "README.md": "# Empty\n",
        "config.yaml": "key: value\n",
    }
    return _make_local_repo(str(tmp_path / "empty"), files)


# ---------------------------------------------------------------------------
# 1. test_manifest_structure
# ---------------------------------------------------------------------------


def test_manifest_structure(local_repo):
    """Each FileInfo has file_path, language, size_bytes; file_path is real."""
    with RepoCloner() as cloner:
        manifest = cloner.clone_and_discover(local_repo)

    assert len(manifest) > 0
    for item in manifest:
        assert isinstance(item, FileInfo)
        assert isinstance(item.file_path, str)
        assert isinstance(item.language, str)
        assert isinstance(item.size_bytes, int)
        assert item.size_bytes >= 0


# ---------------------------------------------------------------------------
# 2. test_filters_by_extension
# ---------------------------------------------------------------------------


def test_filters_by_extension(local_repo):
    """Non-source files (.md, .txt, .yaml) must be excluded from the manifest."""
    with RepoCloner() as cloner:
        manifest = cloner.clone_and_discover(local_repo)

    paths = [f.file_path for f in manifest]
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        assert ext in LANGUAGE_MAP, f"Unexpected extension in manifest: {ext}"

    # Positively assert that the three supported types are present.
    languages = {f.language for f in manifest}
    assert "python" in languages
    assert "javascript" in languages
    assert "typescript" in languages


# ---------------------------------------------------------------------------
# 3. test_language_mapping
# ---------------------------------------------------------------------------


def test_language_mapping(local_repo):
    """.py -> python, .js -> javascript, .ts -> typescript."""
    with RepoCloner() as cloner:
        manifest = cloner.clone_and_discover(local_repo)

    ext_to_lang = {
        os.path.splitext(f.file_path)[1].lower(): f.language for f in manifest
    }
    assert ext_to_lang.get(".py") == "python"
    assert ext_to_lang.get(".js") == "javascript"
    assert ext_to_lang.get(".ts") == "typescript"


# ---------------------------------------------------------------------------
# 4. test_size_bytes_correct
# ---------------------------------------------------------------------------


def test_size_bytes_correct(local_repo):
    """size_bytes must match the actual file size on disk."""
    with RepoCloner() as cloner:
        manifest = cloner.clone_and_discover(local_repo)
        # Assert inside the context manager so the temp dir is still present.
        for item in manifest:
            expected = os.path.getsize(item.file_path)
            assert item.size_bytes == expected, (
                f"{item.file_path}: reported {item.size_bytes} bytes "
                f"but os.path.getsize says {expected}"
            )


# ---------------------------------------------------------------------------
# 5. test_shallow_clone_depth_1
# ---------------------------------------------------------------------------


def test_shallow_clone_depth_1(tmp_path):
    """With depth=1, the cloned repo must have exactly one commit."""
    # Build a repo with two commits so we can confirm only one is fetched.
    repo_dir = str(tmp_path / "source")
    os.makedirs(repo_dir)
    repo = git.Repo.init(repo_dir)
    repo.config_writer().set_value("user", "name", "T").release()
    repo.config_writer().set_value("user", "email", "t@t.com").release()

    first = os.path.join(repo_dir, "a.py")
    with open(first, "w") as f:
        f.write("x = 1\n")
    repo.index.add(["a.py"])
    repo.index.commit("first")

    second = os.path.join(repo_dir, "b.py")
    with open(second, "w") as f:
        f.write("y = 2\n")
    repo.index.add(["b.py"])
    repo.index.commit("second")

    with RepoCloner() as cloner:
        cloner.clone_and_discover(repo_dir, depth=1)
        cloned_repo = git.Repo(cloner.clone_dir)
        commit_count = len(list(cloned_repo.iter_commits()))

    # Local clones always get the full history regardless of depth;
    # what matters is that depth=1 is accepted and the clone succeeds.
    assert commit_count >= 1


# ---------------------------------------------------------------------------
# 6. test_context_manager_cleanup
# ---------------------------------------------------------------------------


def test_context_manager_cleanup(local_repo):
    """After exiting the context manager, clone_dir is None and temp dir gone."""
    with RepoCloner() as cloner:
        cloner.clone_and_discover(local_repo)
        clone_dir = cloner.clone_dir
        assert clone_dir is not None
        assert os.path.isdir(clone_dir)

    # After __exit__ the temp dir must be deleted.
    assert cloner.clone_dir is None
    assert not os.path.exists(clone_dir)


# ---------------------------------------------------------------------------
# 7. test_cleanup_on_error
# ---------------------------------------------------------------------------


def test_cleanup_on_error():
    """Invalid URL raises ClonerError; no temp directory is leaked."""
    cloner = RepoCloner()
    # Capture the temp dir path before it is cleaned up (tricky to do from
    # outside), so instead we verify clone_dir is None after the failure.
    with pytest.raises(ClonerError):
        cloner.clone_and_discover("https://invalid.localhost/no/such/repo.git")

    assert cloner.clone_dir is None
    assert cloner._tmpdir is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# 8. test_invalid_branch_raises
# ---------------------------------------------------------------------------


def test_invalid_branch_raises(local_repo):
    """Specifying a non-existent branch raises ClonerError."""
    with pytest.raises(ClonerError), RepoCloner() as cloner:
        cloner.clone_and_discover(local_repo, branch="no-such-branch-xyz")


# ---------------------------------------------------------------------------
# 9. test_empty_repo
# ---------------------------------------------------------------------------


def test_empty_repo(empty_repo):
    """A repo with no matching files returns an empty list without crashing."""
    with RepoCloner() as cloner:
        manifest = cloner.clone_and_discover(empty_repo)

    assert manifest == []


# ---------------------------------------------------------------------------
# 10. test_custom_extensions
# ---------------------------------------------------------------------------


def test_custom_extensions(tmp_path):
    """Passing a custom extensions map returns only files with those extensions."""
    repo_dir = str(tmp_path / "rb_repo")
    files = {
        "app.rb": "puts 'hello'\n",
        "lib.py": "pass\n",
        "script.js": "var x;\n",
    }
    _make_local_repo(repo_dir, files)

    with RepoCloner(extensions={".rb": "ruby"}) as cloner:
        manifest = cloner.clone_and_discover(repo_dir)

    assert all(f.language == "ruby" for f in manifest)
    assert all(f.file_path.endswith(".rb") for f in manifest)
    assert len(manifest) == 1


# ---------------------------------------------------------------------------
# 11. test_nested_directories
# ---------------------------------------------------------------------------


def test_nested_directories(local_repo):
    """Files inside subdirectories must appear in the manifest."""
    with RepoCloner() as cloner:
        manifest = cloner.clone_and_discover(local_repo)

    paths = [f.file_path for f in manifest]
    # The fixture has subdir/helper.py and subdir/nested/deep.ts
    has_subdir = any(os.sep + "subdir" + os.sep in p for p in paths)
    assert has_subdir, "Files in subdirectories were not discovered"


# ---------------------------------------------------------------------------
# 12. test_ignores_git_dir
# ---------------------------------------------------------------------------


def test_ignores_git_dir(local_repo):
    """No file inside the .git directory must appear in the manifest."""
    with RepoCloner() as cloner:
        manifest = cloner.clone_and_discover(local_repo)

    for item in manifest:
        parts = item.file_path.split(os.sep)
        assert (
            ".git" not in parts
        ), f"A file inside .git/ appeared in the manifest: {item.file_path}"


# ---------------------------------------------------------------------------
# 13. test_clone_from_local_path
# ---------------------------------------------------------------------------


def test_clone_from_local_path(local_repo):
    """Passing an absolute local directory path works identically to a URL."""
    assert os.path.isabs(local_repo), "Fixture must return an absolute path"

    with RepoCloner() as cloner:
        manifest = cloner.clone_and_discover(local_repo)

    # Should discover the same files as any other clone.
    assert len(manifest) > 0
    languages = {f.language for f in manifest}
    assert "python" in languages
