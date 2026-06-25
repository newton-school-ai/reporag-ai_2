import tempfile
from pathlib import Path

from git import Repo

from src.reporag.ingestion.cloner import RepoCloner


def test_repo_cloner_has_supported_extensions():
    cloner = RepoCloner()

    assert ".py" in cloner.extensions
    assert ".js" in cloner.extensions
    assert ".ts" in cloner.extensions


def test_discover_files_filters_supported_languages(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "app.js").write_text("console.log('hello')")
    (tmp_path / "image.png").write_text("fake")

    cloner = RepoCloner()

    files = cloner.discover_files(str(tmp_path))

    assert len(files) == 2

    languages = {file.language for file in files}

    assert "python" in languages
    assert "javascript" in languages


def test_clone_and_discover_local_repository():
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir) / "repo"

        repo_dir.mkdir()

        (repo_dir / "main.py").write_text("print('hello')")
        (repo_dir / "app.js").write_text("console.log('hello')")

        repo = Repo.init(repo_dir)
        repo.index.add(["main.py", "app.js"])
        repo.index.commit("initial commit")

        cloner = RepoCloner()

        manifest = cloner.clone_and_discover(str(repo_dir))

        assert len(manifest) == 2

        languages = {file.language for file in manifest}

        assert "python" in languages
        assert "javascript" in languages
