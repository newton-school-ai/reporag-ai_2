from src.reporag.ingestion.cloner import RepoCloner


def test_repo_cloner_has_supported_extensions():
    cloner = RepoCloner()

    assert ".py" in cloner.extensions
    assert ".js" in cloner.extensions
    assert ".ts" in cloner.extensions
