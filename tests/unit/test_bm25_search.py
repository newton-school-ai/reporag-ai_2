"""Unit tests for BM25 sparse keyword search.

Uses a **real** BM25Index built in-memory with the real ``tokenize_code``
tokenizer, so the full pipeline (tokenization → BM25 scoring → boosting →
filtering → RetrievalResult) is validated end-to-end without network,
GPU, or pickle files.

Payload schema follows the Issue 15 / HybridIndexBuilder canonical layout:
  BM25 metadata: file_path, symbol, symbol_type, chunk_kind,
                 repo_id, start_line, end_line, content
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reporag.config import settings
from reporag.embedding.index_builder import BM25Index
from reporag.retrieval.bm25_search import BM25Search
from reporag.retrieval.vector_search import RetrievalResult

# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------

# Each tuple is (doc_id, text_to_index, metadata_dict).
# The text is what BM25 tokenizes; metadata is what search returns.
_CORPUS = [
    # Doc 1: DEFINES authenticate_user — should rank #1 for that query
    (
        "id-1",
        "def authenticate_user(username, password): validate(username) check_password(password)",
        {
            "file_path": "src/auth.py",
            "symbol": "auth.authenticate_user",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "repo_id": "org/repo",
            "start_line": 10,
            "end_line": 25,
            "content": "def authenticate_user(username, password): validate(username) check_password(password)",
        },
    ),
    # Doc 2: CALLS authenticate_user but defines login_handler
    (
        "id-2",
        "def login_handler(request): result = authenticate_user(u, p) return result",
        {
            "file_path": "src/api.py",
            "symbol": "api.login_handler",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "repo_id": "org/repo",
            "start_line": 1,
            "end_line": 10,
            "content": "def login_handler(request): result = authenticate_user(u, p) return result",
        },
    ),
    # Doc 3: Unrelated function
    (
        "id-3",
        "def format_date(d): return d.strftime('%Y-%m-%d')",
        {
            "file_path": "src/utils.py",
            "symbol": "utils.format_date",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "repo_id": "org/repo",
            "start_line": 1,
            "end_line": 5,
            "content": "def format_date(d): return d.strftime('%Y-%m-%d')",
        },
    ),
    # Doc 4: A class with authenticate in it
    (
        "id-4",
        "class AuthManager: def authenticate(self, user): pass",
        {
            "file_path": "src/manager.py",
            "symbol": "AuthManager",
            "symbol_type": "class",
            "chunk_kind": "definition",
            "repo_id": "org/repo",
            "start_line": 1,
            "end_line": 20,
            "content": "class AuthManager: def authenticate(self, user): pass",
        },
    ),
    # Doc 5: Same function name, different repo
    (
        "id-5",
        "def authenticate_user(u): pass",
        {
            "file_path": "src/auth.py",
            "symbol": "auth.authenticate_user",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "repo_id": "other-org/other-repo",
            "start_line": 1,
            "end_line": 5,
            "content": "def authenticate_user(u): pass",
        },
    ),
]


@pytest.fixture()
def bm25_index() -> BM25Index:
    """Build a BM25Index in-memory from the test corpus."""
    index = BM25Index()
    for doc_id, text, metadata in _CORPUS:
        index.add(doc_id, text, metadata)
    return index


@pytest.fixture()
def searcher(bm25_index: BM25Index) -> BM25Search:
    """A BM25Search wired to the in-memory test index."""
    return BM25Search(bm25_index=bm25_index)


# ---------------------------------------------------------------------------
# TestBM25SearchBasic: edge cases
# ---------------------------------------------------------------------------


class TestBM25SearchBasic:
    """Edge cases and error handling."""

    def test_no_index_raises(self) -> None:
        """Searching without a loaded index raises RuntimeError."""
        s = BM25Search()
        with pytest.raises(RuntimeError, match="No BM25 index loaded"):
            s.search("test")

    def test_empty_query_returns_empty(self, searcher: BM25Search) -> None:
        assert searcher.search("") == []
        assert searcher.search("   ") == []

    def test_no_matches_returns_empty(self, searcher: BM25Search) -> None:
        """A query with zero lexical overlap returns nothing."""
        results = searcher.search("zzzznonexistentzzzzz")
        assert results == []

    def test_invalid_top_k_raises(self, searcher: BM25Search) -> None:
        """top_k < 1 raises ValueError, matching VectorSearch behaviour."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            searcher.search("authenticate", top_k=0)
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            searcher.search("authenticate", top_k=-1)


# ---------------------------------------------------------------------------
# TestBM25SearchResults: schema and output format
# ---------------------------------------------------------------------------


class TestBM25SearchResults:
    """Result type, field alignment, and sorting."""

    def test_returns_retrieval_results(self, searcher: BM25Search) -> None:
        results = searcher.search("authenticate_user")
        assert len(results) > 0
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_field_mapping_matches_schema(self, searcher: BM25Search) -> None:
        """Verify each field maps to the correct BM25 metadata key."""
        results = searcher.search("authenticate_user")
        r = results[0]
        # All fields from the merged RetrievalResult are present
        assert isinstance(r.score, float)
        assert isinstance(r.file_path, str)
        assert r.file_path != ""
        assert r.start_line is not None
        assert r.end_line is not None
        assert r.symbol_name is not None
        assert isinstance(r.chunk_text, str)
        assert r.chunk_text != ""
        assert isinstance(r.metadata, dict)

    def test_sorted_by_score_descending(self, searcher: BM25Search) -> None:
        results = searcher.search("authenticate", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_metadata_is_full_dict(self, searcher: BM25Search) -> None:
        """The full BM25 metadata dict is passed through."""
        results = searcher.search("authenticate_user")
        meta = results[0].metadata
        # All 8 canonical BM25 metadata keys should be present
        assert "file_path" in meta
        assert "symbol" in meta
        assert "symbol_type" in meta
        assert "chunk_kind" in meta
        assert "repo_id" in meta
        assert "start_line" in meta
        assert "end_line" in meta
        assert "content" in meta


# ---------------------------------------------------------------------------
# TestNameBoosting: exact-name boosting
# ---------------------------------------------------------------------------


class TestNameBoosting:
    """Verify the name_boost parameter works correctly."""

    def test_defining_function_ranks_first(self, searcher: BM25Search) -> None:
        """'authenticate_user' returns the defining function as top-1,
        not a file that merely mentions it.

        Doc 1 (symbol='auth.authenticate_user') must rank above
        Doc 2 (symbol='api.login_handler', which calls authenticate_user).
        """
        results = searcher.search("authenticate_user", top_k=10)
        assert len(results) >= 2
        # The defining function must be #1
        assert "authenticate_user" in results[0].symbol_name
        # Its symbol_name is the qualified name from BM25 metadata
        assert results[0].symbol_name == "auth.authenticate_user"

    def test_boost_disabled_when_one(self, searcher: BM25Search) -> None:
        """name_boost=1.0 effectively disables boosting (multiply by 1)."""
        boosted = searcher.search("authenticate_user", name_boost=2.0)
        unboosted = searcher.search("authenticate_user", name_boost=1.0)
        # Both should return results, but scores differ
        assert len(boosted) > 0
        assert len(unboosted) > 0
        # With boost=1.0, no score multiplication happens
        # The boosted top-1 score should be >= unboosted top-1
        assert boosted[0].score >= unboosted[0].score

    def test_custom_boost_factor(self, searcher: BM25Search) -> None:
        """A higher boost factor produces a higher score for matching symbols."""
        results_2x = searcher.search("authenticate_user", name_boost=2.0)
        results_5x = searcher.search("authenticate_user", name_boost=5.0)
        # Both should have the defining function as top-1
        assert results_2x[0].symbol_name == "auth.authenticate_user"
        assert results_5x[0].symbol_name == "auth.authenticate_user"
        # The 5x boost should produce a higher score
        assert results_5x[0].score > results_2x[0].score


# ---------------------------------------------------------------------------
# TestFiltering: post-filter on metadata
# ---------------------------------------------------------------------------


class TestFiltering:
    """All post-filters work correctly."""

    def test_symbol_type_filter(self, searcher: BM25Search) -> None:
        """Filter to classes only — should exclude functions."""
        results = searcher.search("authenticate", symbol_type="class")
        assert len(results) > 0
        for r in results:
            assert r.metadata["symbol_type"] == "class"

    def test_repo_id_filter(self, searcher: BM25Search) -> None:
        """Filter to a specific repo."""
        results = searcher.search("authenticate_user", repo_id="other-org/other-repo")
        assert len(results) > 0
        for r in results:
            assert r.metadata["repo_id"] == "other-org/other-repo"

    def test_file_path_exact_filter(self, searcher: BM25Search) -> None:
        """Exact file path match."""
        results = searcher.search("authenticate_user", file_path="src/api.py")
        assert len(results) > 0
        for r in results:
            assert r.file_path == "src/api.py"

    def test_file_path_glob_filter(self, searcher: BM25Search) -> None:
        """Glob pattern file path filter."""
        results = searcher.search("authenticate", file_path="src/*.py")
        assert len(results) > 0
        for r in results:
            assert r.file_path.startswith("src/")
            assert r.file_path.endswith(".py")

    def test_combined_filters(self, searcher: BM25Search) -> None:
        """Multiple filters applied simultaneously."""
        results = searcher.search(
            "authenticate_user",
            symbol_type="function",
            repo_id="org/repo",
        )
        assert len(results) > 0
        for r in results:
            assert r.metadata["symbol_type"] == "function"
            assert r.metadata["repo_id"] == "org/repo"


# ---------------------------------------------------------------------------
# TestTopK: result limiting
# ---------------------------------------------------------------------------


class TestTopK:
    """Top-k limit and config default."""

    def test_top_k_limits_output(self, searcher: BM25Search) -> None:
        results = searcher.search("authenticate", top_k=1)
        assert len(results) == 1

    def test_default_top_k(self, searcher: BM25Search) -> None:
        """When top_k=None, uses settings.bm25_search_top_k (default 20)."""
        results = searcher.search("authenticate")
        # We have fewer docs than 20, so all matching docs return
        assert len(results) <= settings.bm25_search_top_k


# ---------------------------------------------------------------------------
# TestLoadIndex: persistence round-trip
# ---------------------------------------------------------------------------


class TestLoadIndex:
    """Index load from disk."""

    def test_load_and_search(self, bm25_index: BM25Index, tmp_path: Path) -> None:
        """Save an index, load it via BM25Search.load_index, and search."""
        pkl_path = tmp_path / "test.bm25.pkl"
        bm25_index.save(pkl_path)

        searcher = BM25Search()
        searcher.load_index(pkl_path)

        results = searcher.search("authenticate_user", top_k=5)
        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)
        assert "authenticate_user" in results[0].symbol_name
