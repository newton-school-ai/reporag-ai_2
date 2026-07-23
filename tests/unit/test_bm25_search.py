"""Unit tests for the BM25 sparse keyword search retrieval engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from reporag.embedding.index_builder import BM25Index
from reporag.retrieval.bm25_search import BM25Search


@pytest.fixture
def temp_bm25_index_path(tmp_path: Path) -> Path:
    """Create a temporary BM25 index file populated with test code chunks."""
    index = BM25Index()

    # 1. Define a function definition for 'authenticate_user'
    index.add(
        doc_id="point_1",
        text="def authenticate_user(user, password): login",
        metadata={
            "file_path": "src/auth.py",
            "symbol": "auth.authenticate_user",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "repo_id": "test-repo",
            "start_line": 10,
            "end_line": 15,
            "content": "def authenticate_user(user, password): login",
        },
    )

    # 2. A continuation chunk that merely mentions 'authenticate_user' multiple times.
    # It has a higher natural BM25 score than point_1.
    index.add(
        doc_id="point_2",
        text="authenticate_user authenticate_user authenticate_user authenticate_user",
        metadata={
            "file_path": "src/utils.py",
            "symbol": "utils.login_helper",
            "symbol_type": "function",
            "chunk_kind": "continuation",
            "repo_id": "test-repo",
            "start_line": 20,
            "end_line": 25,
            "content": "authenticate_user authenticate_user authenticate_user authenticate_user",
        },
    )

    # 3. A JavaScript file containing helper rendering login screen
    index.add(
        doc_id="point_3",
        text="function render_login() { console.log('login ui'); }",
        metadata={
            "file_path": "static/app.js",
            "symbol": "render_login",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "repo_id": "test-repo",
            "start_line": 1,
            "end_line": 3,
            "content": "function render_login() { console.log('login ui'); }",
        },
    )

    # 4. A Python class definition
    index.add(
        doc_id="point_4",
        text="class UserSession: pass",
        metadata={
            "file_path": "src/session.py",
            "symbol": "session.UserSession",
            "symbol_type": "class",
            "chunk_kind": "definition",
            "repo_id": "test-repo",
            "start_line": 5,
            "end_line": 10,
            "content": "class UserSession: pass",
        },
    )

    # Add 10 unrelated filler documents to keep document statistics positive (avoid BM25 score <= 0 drop)
    for i in range(10):
        index.add(
            doc_id=f"filler_{i}",
            text=f"some completely unrelated text block {i}",
            metadata={
                "file_path": f"src/filler_{i}.py",
                "symbol": f"filler_{i}",
                "symbol_type": "variable",
                "chunk_kind": "definition",
                "repo_id": "test-repo",
                "start_line": 1,
                "end_line": 2,
                "content": f"some completely unrelated text block {i}",
            },
        )

    index_file = tmp_path / "bm25.pkl"
    index.save(index_file)
    return index_file


@pytest.fixture
def bm25_searcher(temp_bm25_index_path: Path) -> BM25Search:
    """Return a BM25Search instance with the test index loaded."""
    searcher = BM25Search(index_path=temp_bm25_index_path)
    searcher.load_index()
    return searcher


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestBM25SearchBasic:
    """Verify loading, basic search and return values of BM25Search."""

    def test_load_index_non_existent_raises(self) -> None:
        """Loading a non-existent index file path raises FileNotFoundError."""
        searcher = BM25Search(index_path="non_existent_path.pkl")
        with pytest.raises(FileNotFoundError, match="BM25 index file not found"):
            searcher.load_index()

    def test_search_without_load_raises(self, temp_bm25_index_path: Path) -> None:
        """Calling search() before loading the index raises RuntimeError."""
        searcher = BM25Search(index_path=temp_bm25_index_path)
        with pytest.raises(RuntimeError, match="BM25 index is not loaded"):
            searcher.search("test")

    def test_basic_search_returns_schema(self, bm25_searcher: BM25Search) -> None:
        """BM25 search returns RetrievalResult objects matching the schema."""
        results = bm25_searcher.search("render_login", top_k=5)
        assert len(results) >= 1
        r = results[0]
        assert r.score > 0.0
        assert r.file_path == "static/app.js"
        assert r.start_line == 1
        assert r.end_line == 3
        assert r.symbol_name == "render_login"
        assert r.chunk_text == "function render_login() { console.log('login ui'); }"
        assert isinstance(r.metadata, dict)


class TestBM25Boosting:
    """Verify boosting behavior for exact symbol name definitions."""

    def test_boosting_exact_identifier_definition_wins(
        self, bm25_searcher: BM25Search
    ) -> None:
        """Exact identifier search returns the defining function as top-1.

        Without boosting, the continuation/mention chunk (point_2) ranks
        higher because it is short and has a high term frequency of 'authenticate_user'.
        With boosting (boost_factor=2.0), the definition chunk (point_1) must
        win and return as rank 1.
        """
        # Run search with default boost
        results = bm25_searcher.search("authenticate_user", top_k=5)
        assert len(results) >= 2

        # Defining function (src/auth.py) must be top-1
        top_result = results[0]
        assert top_result.file_path == "src/auth.py"
        assert top_result.symbol_name == "auth.authenticate_user"
        assert top_result.metadata.get("chunk_kind") == "definition"

        # Mention chunk (src/utils.py) should be second
        second_result = results[1]
        assert second_result.file_path == "src/utils.py"
        assert second_result.symbol_name == "utils.login_helper"

    def test_configurable_boost_factor(self, bm25_searcher: BM25Search) -> None:
        """The boost factor can be customized to adjust the ranking or scores."""
        # 1. Search with no boost (boost_factor=1.0)
        # Point 2 mentions 'authenticate_user' multiple times in a short doc, so it should score higher without boost.
        results_no_boost = bm25_searcher.search(
            "authenticate_user", top_k=5, boost_factor=1.0
        )
        assert results_no_boost[0].file_path == "src/utils.py"

        # 2. Search with default boost (2.0)
        results_boosted = bm25_searcher.search("authenticate_user", top_k=5)
        assert results_boosted[0].file_path == "src/auth.py"


class TestBM25Filtering:
    """Verify optional query filters on language, file_path, and symbol_type."""

    def test_language_filter(self, bm25_searcher: BM25Search) -> None:
        """Language filter matches extension mappings (inferred language)."""
        # App.js is javascript, auth.py is python. Querying "login" matches both.
        results_js = bm25_searcher.search("login", language="javascript")
        assert len(results_js) == 1
        assert results_js[0].file_path == "static/app.js"

        results_py = bm25_searcher.search("login", language="python")
        assert len(results_py) == 1
        assert results_py[0].file_path == "src/auth.py"

    def test_file_path_glob_filter(self, bm25_searcher: BM25Search) -> None:
        """File path filter supports exact matching and glob patterns."""
        # Glob pattern matching Python source files in src/
        results_glob = bm25_searcher.search("login", file_path="src/*.py")
        assert len(results_glob) == 1
        assert results_glob[0].file_path == "src/auth.py"

        # Exact path filter
        results_exact = bm25_searcher.search("login", file_path="static/app.js")
        assert len(results_exact) == 1
        assert results_exact[0].file_path == "static/app.js"

        # Non-matching path
        results_none = bm25_searcher.search("login", file_path="nonexistent/*.py")
        assert len(results_none) == 0

    def test_symbol_type_filter(self, bm25_searcher: BM25Search) -> None:
        """Filters results based on the metadata symbol_type."""
        # session.UserSession is a class, auth.authenticate_user is a function
        results_class = bm25_searcher.search("user", symbol_type="class")
        assert len(results_class) == 1
        assert results_class[0].symbol_name == "session.UserSession"

        results_func = bm25_searcher.search("user", symbol_type="function")
        # Matches auth.authenticate_user (function definition) and utils.login_helper (function mention)
        assert len(results_func) >= 2
        assert all(r.metadata.get("symbol_type") == "function" for r in results_func)


class TestBM25EdgeCases:
    """Verify input validation and boundary cases."""

    def test_top_k_invalid_raises(self, bm25_searcher: BM25Search) -> None:
        """ValueError is raised when top_k is less than 1."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            bm25_searcher.search("test", top_k=0)

    def test_empty_query_returns_empty(self, bm25_searcher: BM25Search) -> None:
        """Whitespace or empty query returns an empty list immediately."""
        assert bm25_searcher.search("") == []
        assert bm25_searcher.search("   ") == []
