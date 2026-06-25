import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from git import Repo

from reporag.config import settings
from src.reporag.ingestion.cloner import CloneError, RepoCloner


def test_repo_cloner_uses_settings_extensions():
    cloner = RepoCloner()

    assert cloner.extensions == settings.language_extensions


def test_repo_cloner_has_supported_extensions():
    cloner = RepoCloner()

    assert ".py" in cloner.extensions
    assert ".js" in cloner.extensions
    assert ".ts" in cloner.extensions


def test_discover_files_filters_supported_languages(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "app.js").write_text("console.log('hello')")
    (tmp_path / "image.png").write_text("fake")

    files = RepoCloner().discover_files(str(tmp_path))

    assert len(files) == 2

    languages = {file.language for file in files}

    assert "python" in languages
    assert "javascript" in languages


def test_discover_files_with_custom_extensions(tmp_path):
    (tmp_path / "main.go").write_text("package main")
    (tmp_path / "main.py").write_text("print('hi')")

    files = RepoCloner().discover_files(
        str(tmp_path),
        extensions={".go": "go"},
    )

    assert len(files) == 1
    assert files[0].language == "go"


def test_discover_files_ignores_known_directories(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')")

    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "bad.js").write_text("console.log('bad')")

    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "cached.py").write_text("print('cached')")

    files = RepoCloner().discover_files(str(tmp_path))

    assert len(files) == 1
    assert Path(files[0].path).name == "main.py"


def test_manifest_contains_path_language_and_size(tmp_path):
    file_path = tmp_path / "main.py"
    file_path.write_text("print('hello')")

    files = RepoCloner().discover_files(str(tmp_path))

    assert len(files) == 1

    file_info = files[0]

    assert file_info.path.endswith("main.py")
    assert file_info.language == "python"
    assert file_info.size_bytes > 0


def test_clone_and_discover_local_repository():
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        repo_dir.mkdir()

        (repo_dir / "main.py").write_text("print('hello')")
        (repo_dir / "app.js").write_text("console.log('hello')")

        repo = Repo.init(repo_dir)
        repo.index.add(["main.py", "app.js"])
        repo.index.commit("initial commit")

        manifest = RepoCloner().clone_and_discover(str(repo_dir))

        assert len(manifest) == 2

        languages = {file.language for file in manifest}

        assert "python" in languages
        assert "javascript" in languages


def test_local_repository_is_cloned_via_file_uri(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    repo = Repo.init(repo_dir)

    (repo_dir / "main.py").write_text("print('hello')")

    repo.index.add(["main.py"])
    repo.index.commit("initial commit")

    with patch("src.reporag.ingestion.cloner.Repo.clone_from") as mock_clone:
        RepoCloner().clone_repository(str(repo_dir))

    source = mock_clone.call_args.args[0]

    assert source.startswith("file://")


def test_clone_repository_uses_configured_depth(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    repo = Repo.init(repo_dir)

    (repo_dir / "main.py").write_text("print('hello')")

    repo.index.add(["main.py"])
    repo.index.commit("initial commit")

    monkeypatch.setattr(settings, "clone_depth", 5)

    with patch("src.reporag.ingestion.cloner.Repo.clone_from") as mock_clone:
        RepoCloner().clone_repository(str(repo_dir))

    assert mock_clone.call_args.kwargs["depth"] == 5


def test_explicit_depth_overrides_settings(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    repo = Repo.init(repo_dir)

    (repo_dir / "main.py").write_text("print('hello')")

    repo.index.add(["main.py"])
    repo.index.commit("initial commit")

    monkeypatch.setattr(settings, "clone_depth", 5)

    with patch("src.reporag.ingestion.cloner.Repo.clone_from") as mock_clone:
        RepoCloner().clone_repository(
            str(repo_dir),
            depth=2,
        )

    assert mock_clone.call_args.kwargs["depth"] == 2


def test_clone_repository_respects_branch():
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"
        repo_dir.mkdir()

        repo = Repo.init(repo_dir)

        (repo_dir / "main.py").write_text("print('main')")
        repo.index.add(["main.py"])
        repo.index.commit("main commit")

        feature_branch = repo.create_head("feature")
        feature_branch.checkout()

        (repo_dir / "feature.py").write_text("print('feature')")
        repo.index.add(["feature.py"])
        repo.index.commit("feature commit")

        repo.active_branch.checkout()

        cloner = RepoCloner()

        cloned_path = cloner.clone_repository(
            str(repo_dir),
            branch="feature",
        )

        assert (Path(cloned_path) / "feature.py").exists()


def test_manifest_is_sorted(tmp_path):
    (tmp_path / "z.py").write_text("print('z')")
    (tmp_path / "a.py").write_text("print('a')")
    (tmp_path / "m.py").write_text("print('m')")

    files = RepoCloner().discover_files(str(tmp_path))

    paths = [Path(file.path).name for file in files]

    assert paths == sorted(paths)


def test_clone_repository_cleans_up_when_git_clone_fails():
    with (
        patch(
            "src.reporag.ingestion.cloner.Repo.clone_from",
            side_effect=Exception("boom"),
        ),
        pytest.raises(CloneError),
    ):
        RepoCloner().clone_repository("fake-repository")


def test_clone_and_discover_raises_clone_error():
    with (
        patch(
            "src.reporag.ingestion.cloner.Repo.clone_from",
            side_effect=Exception("boom"),
        ),
        pytest.raises(CloneError),
    ):
        RepoCloner().clone_and_discover("fake-repository")


def test_shallow_clone_respects_depth(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    repo = Repo.init(repo_dir)

    (repo_dir / "a.py").write_text("a")
    repo.index.add(["a.py"])
    repo.index.commit("commit1")

    (repo_dir / "b.py").write_text("b")
    repo.index.add(["b.py"])
    repo.index.commit("commit2")

    cloner = RepoCloner()

    clone_path = cloner.clone_repository(
        str(repo_dir),
        depth=1,
    )

    cloned_repo = Repo(clone_path)

    commits = list(cloned_repo.iter_commits())

    assert len(commits) == 1


def test_cleanup_resets_last_clone_path():
    cloner = RepoCloner()

    with tempfile.TemporaryDirectory() as temp_dir:
        cloner.last_clone_path = temp_dir

        cloner.cleanup()

        assert cloner.last_clone_path is None


def test_repository_size_limit_cleans_up(
    tmp_path,
    monkeypatch,
):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    (repo_dir / "main.py").write_text("print('hello')")

    repo = Repo.init(repo_dir)
    repo.index.add(["main.py"])
    repo.index.commit("init")

    monkeypatch.setattr(
        settings,
        "max_repo_size_mb",
        0,
    )

    cloner = RepoCloner()

    with pytest.raises(CloneError):
        cloner.clone_repository(str(repo_dir))

    if cloner.last_clone_path:
        assert not Path(cloner.last_clone_path).exists()


def test_shallow_clone_truncates_history(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    repo = Repo.init(repo_dir)

    (repo_dir / "main.py").write_text("v1")
    repo.index.add(["main.py"])
    repo.index.commit("commit1")

    (repo_dir / "main.py").write_text("v2")
    repo.index.add(["main.py"])
    repo.index.commit("commit2")

    cloner = RepoCloner()

    cloned_path = cloner.clone_repository(
        str(repo_dir),
        depth=1,
    )

    cloned_repo = Repo(cloned_path)

    commits = list(cloned_repo.iter_commits())

    assert len(commits) == 1

    cloner.cleanup()
