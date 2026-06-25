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
