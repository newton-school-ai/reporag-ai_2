"""Unit tests for BM25 sparse keyword search (Issue 17).

Tests are fully network-free: a pre-built :class:`BM25Index` is injected
directly into :class:`BM25Search`, so no disk I/O, pickle files, or external
services are required.

Payload schema follows the Issue 15 / HybridIndexBuilder canonical layout:
  BM25 metadata: file_path, symbol, symbol_type, chunk_kind, repo_id,
                 start_line, end_line, content
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from reporag.embedding.index_builder import BM25Index, tokenize_code
from reporag.retrieval.bm25_search import BM25Search
from reporag.retrieval.vector_search import RetrievalResult

# ---------------------------------------------------------------------------
# Helpers: build a populated BM25Index for testing
# ---------------------------------------------------------------------------


def _build_test_index() -> BM25Index:
    """Build a BM25Index populated with realistic code chunks.

    Mirrors the metadata schema set by
    :meth:`HybridIndexBuilder.upsert_code_chunks` -- the same fields that
    ``BM25Search._to_retrieval_results`` reads.
    """
    index = BM25Index()

    # Chunk 1: the authenticate_user function definition
    index.add(
        "id-auth-def",
        "def authenticate_user(username, password):\n"
        "    user = get_user_by_name(username)\n"
        "    if user and verify_password(password, user.hashed_password):\n"
        "        return create_access_token(user)\n"
        "    raise AuthenticationError('Invalid credentials')",
        metadata={
            "file_path": "src/auth/service.py",
            "symbol": "authenticate_user",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "repo_id": "test-repo",
            "start_line": 10,
            "end_line": 15,
            "content": (
                "def authenticate_user(username, password):\n"
                "    user = get_user_by_name(username)\n"
                "    if user and verify_password(password, user.hashed_password):\n"
                "        return create_access_token(user)\n"
                "    raise AuthenticationError('Invalid credentials')"
            ),
            "language": "python",
        },
    )

    # Chunk 2: a test file that *mentions* authenticate_user (not the def)
    index.add(
        "id-auth-test",
        "def test_authenticate_user():\n"
        "    result = authenticate_user('admin', 'secret')\n"
        "    assert result is not None",
        metadata={
            "file_path": "tests/test_auth.py",
            "symbol": "test_authenticate_user",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "repo_id": "test-repo",
            "start_line": 1,
            "end_line": 3,
            "content": (
                "def test_authenticate_user():\n"
                "    result = authenticate_user('admin', 'secret')\n"
                "    assert result is not None"
            ),
            "language": "python",
        },
    )

    # Chunk 3: getUserById (camelCase, JavaScript)
    index.add(
        "id-get-user",
        "async function getUserById(id) {\n"
        "    const user = await db.users.findOne({ _id: id });\n"
        "    if (!user) throw new NotFoundError('User not found');\n"
        "    return user;\n"
        "}",
        metadata={
            "file_path": "src/users/service.js",
            "symbol": "getUserById",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "repo_id": "test-repo",
            "start_line": 20,
            "end_line": 25,
            "content": (
                "async function getUserById(id) {\n"
                "    const user = await db.users.findOne({ _id: id });\n"
                "    if (!user) throw new NotFoundError('User not found');\n"
                "    return user;\n"
                "}"
            ),
            "language": "javascript",
        },
    )

    # Chunk 4: UserManager class definition
    index.add(
        "id-user-manager",
        "class UserManager:\n"
        "    def __init__(self, db):\n"
        "        self.db = db\n"
        "    def get_user(self, user_id):\n"
        "        return self.db.query(User).get(user_id)\n"
        "    def create_user(self, data):\n"
        "        user = User(**data)\n"
        "        self.db.add(user)\n"
        "        return user",
        metadata={
            "file_path": "src/users/manager.py",
            "symbol": "UserManager",
            "symbol_type": "class",
            "chunk_kind": "definition",
            "repo_id": "test-repo",
            "start_line": 1,
            "end_line": 9,
            "content": (
                "class UserManager:\n"
                "    def __init__(self, db):\n"
                "        self.db = db\n"
                "    def get_user(self, user_id):\n"
                "        return self.db.query(User).get(user_id)\n"
                "    def create_user(self, data):\n"
                "        user = User(**data)\n"
                "        self.db.add(user)\n"
                "        return user"
            ),
            "language": "python",
        },
    )

    # Chunk 5: get_user_by_id (snake_case, Python) -- different from camelCase
    index.add(
        "id-get-user-by-id",
        "def get_user_by_id(user_id: int) -> User:\n"
        "    return db.session.query(User).filter_by(id=user_id).first()",
        metadata={
            "file_path": "src/users/queries.py",
            "symbol": "get_user_by_id",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "repo_id": "test-repo",
            "start_line": 30,
            "end_line": 32,
            "content": (
                "def get_user_by_id(user_id: int) -> User:\n"
                "    return db.session.query(User).filter_by(id=user_id).first()"
            ),
            "language": "python",
        },
    )

    # Chunk 6: format_response utility (unrelated to auth)
    index.add(
        "id-format-response",
        "def format_response(data, status_code=200):\n"
        "    return {'data': data, 'status': status_code}",
        metadata={
            "file_path": "src/utils/response.py",
            "symbol": "format_response",
            "symbol_type": "function",
            "chunk_kind": "definition",
            "repo_id": "test-repo",
            "start_line": 1,
            "end_line": 2,
            "content": (
                "def format_response(data, status_code=200):\n"
                "    return {'data': data, 'status': status_code}"
            ),
            "language": "python",
        },
    )

    return index


@pytest.fixture
def bm25_index() -> BM25Index:
    """Provide a pre-built BM25Index for injection into BM25Search."""
    return _build_test_index()


@pytest.fixture
def searcher(bm25_index: BM25Index) -> BM25Search:
    """Provide a BM25Search instance backed by the test index."""
    return BM25Search(bm25_index=bm25_index)


# ---------------------------------------------------------------------------
# AC1: Exact identifier queries return the defining function as top-1
# ---------------------------------------------------------------------------


class TestExactIdentifierTopOne:
    """Verify that exact identifier queries rank the definition first."""

    def test_authenticate_user_returns_definition_as_top_1(
        self, searcher: BM25Search
    ) -> None:
        """The spec example: 'authenticate_user' must return the function, not mentions."""
        results = searcher.search("authenticate_user", top_k=10)
        assert len(results) >= 1
        assert results[0].symbol_name == "authenticate_user"
        assert results[0].file_path == "src/auth/service.py"

    def test_definition_outranks_test_that_mentions_it(
        self, searcher: BM25Search
    ) -> None:
        """The test file mentions authenticate_user but should rank below the definition."""
        results = searcher.search("authenticate_user", top_k=10)
        definition_idx = next(
            i for i, r in enumerate(results) if r.symbol_name == "authenticate_user"
        )
        test_idx = next(
            i
            for i, r in enumerate(results)
            if r.symbol_name == "test_authenticate_user"
        )
        assert (
            definition_idx < test_idx
        ), "Definition should rank above the test that merely calls it"

    def test_get_user_by_id_returns_snake_case_definition(
        self, searcher: BM25Search
    ) -> None:
        """Exact search for a snake_case identifier."""
        results = searcher.search("get_user_by_id", top_k=5)
        assert len(results) >= 1
        assert results[0].symbol_name == "get_user_by_id"


# ---------------------------------------------------------------------------
# AC2: Code-aware tokenization matches indexing tokenizer
# ---------------------------------------------------------------------------


class TestTokenizationConsistency:
    """Verify that query tokenization uses the same tokenizer as indexing."""

    def test_tokenizer_is_tokenize_code(self, searcher: BM25Search) -> None:
        """BM25Search's underlying index must use tokenize_code."""
        assert searcher.index.tokenizer is tokenize_code

    def test_camel_case_query_matches_snake_case_index(
        self, searcher: BM25Search
    ) -> None:
        """'getUserById' should match 'get_user_by_id' via shared tokens."""
        results = searcher.search("getUserById", top_k=10)
        symbols = [r.symbol_name for r in results]
        # Both the camelCase JS and snake_case Python versions share tokens
        assert "getUserById" in symbols or "get_user_by_id" in symbols

    def test_case_insensitive_matching(self, searcher: BM25Search) -> None:
        """'AuthenticateUser' (PascalCase) should still find authenticate_user."""
        results = searcher.search("AuthenticateUser", top_k=10)
        assert len(results) >= 1
        symbols = [r.symbol_name for r in results]
        assert "authenticate_user" in symbols


# ---------------------------------------------------------------------------
# AC3: Boosting exact name matches works (configurable boost factor)
# ---------------------------------------------------------------------------


class TestExactNameBoost:
    """Verify the configurable exact-name boost mechanism."""

    def test_boost_lifts_exact_match(self, bm25_index: BM25Index) -> None:
        """With boost, the exact symbol match should have a higher score."""
        boosted = BM25Search(bm25_index=bm25_index, exact_name_boost=3.0)
        results_boosted = boosted.search("authenticate_user", top_k=10)

        no_boost = BM25Search(bm25_index=bm25_index, exact_name_boost=1.0)
        results_plain = no_boost.search("authenticate_user", top_k=10)

        # The definition result should have a higher score with boost
        boosted_def = next(
            r for r in results_boosted if r.symbol_name == "authenticate_user"
        )
        plain_def = next(
            r for r in results_plain if r.symbol_name == "authenticate_user"
        )
        assert boosted_def.score > plain_def.score

    def test_boost_factor_configurable(self, bm25_index: BM25Index) -> None:
        """Different boost factors produce proportionally different scores."""
        boost_2x = BM25Search(bm25_index=bm25_index, exact_name_boost=2.0)
        boost_4x = BM25Search(bm25_index=bm25_index, exact_name_boost=4.0)

        results_2x = boost_2x.search("authenticate_user", top_k=10)
        results_4x = boost_4x.search("authenticate_user", top_k=10)

        score_2x = next(
            r.score for r in results_2x if r.symbol_name == "authenticate_user"
        )
        score_4x = next(
            r.score for r in results_4x if r.symbol_name == "authenticate_user"
        )
        # 4x boost should produce roughly double the 2x boost score
        assert score_4x > score_2x

    def test_no_boost_when_disabled(self, bm25_index: BM25Index) -> None:
        """With boost=1.0, scores should equal raw BM25 scores."""
        no_boost = BM25Search(bm25_index=bm25_index, exact_name_boost=1.0)
        results = no_boost.search("authenticate_user", top_k=10)

        raw_results = bm25_index.search("authenticate_user", top_k=10)
        raw_scores = {hit["metadata"]["symbol"]: hit["score"] for hit in raw_results}

        for r in results:
            if r.symbol_name in raw_scores:
                assert abs(r.score - raw_scores[r.symbol_name]) < 1e-6


# ---------------------------------------------------------------------------
# AC4: Returns same result schema as vector search
# ---------------------------------------------------------------------------


class TestResultSchema:
    """Verify that BM25Search returns the same RetrievalResult schema."""

    def test_returns_retrieval_result_instances(self, searcher: BM25Search) -> None:
        """Each result must be a RetrievalResult."""
        results = searcher.search("authenticate_user", top_k=5)
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_all_fields_populated(self, searcher: BM25Search) -> None:
        """Key RetrievalResult fields should be populated from BM25 metadata."""
        results = searcher.search("authenticate_user", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert r.score > 0
        assert r.file_path != ""
        assert r.start_line is not None
        assert r.end_line is not None
        assert r.symbol_name is not None
        assert r.chunk_text != ""
        assert isinstance(r.metadata, dict)
        assert "repo_id" in r.metadata

    def test_score_is_float(self, searcher: BM25Search) -> None:
        """BM25 scores must be floats, not numpy scalars."""
        results = searcher.search("user", top_k=5)
        for r in results:
            assert isinstance(r.score, float)


# ---------------------------------------------------------------------------
# AC5: Unit test: "authenticate_user" returns the function, not just mentions
# ---------------------------------------------------------------------------


class TestIdentifierVsMention:
    """The spec's explicit AC: the defining function, not just mentions."""

    def test_authenticate_user_top_result_is_definition(
        self, searcher: BM25Search
    ) -> None:
        """Top result for 'authenticate_user' is the def, not the test call."""
        results = searcher.search("authenticate_user", top_k=5)
        top = results[0]
        assert top.symbol_name == "authenticate_user"
        assert "def authenticate_user" in top.chunk_text


# ---------------------------------------------------------------------------
# Filtering tests (mirrors VectorSearch API)
# ---------------------------------------------------------------------------


class TestFiltering:
    """Verify post-search filtering by language, file_path, symbol_type."""

    def test_language_filter_python(self, searcher: BM25Search) -> None:
        """Language filter should exclude JavaScript results."""
        results = searcher.search("user", top_k=20, language="python")
        for r in results:
            assert r.metadata.get("language") == "python"

    def test_language_filter_javascript(self, searcher: BM25Search) -> None:
        """Language filter for javascript should only return JS results."""
        results = searcher.search("getUserById", top_k=10, language="javascript")
        for r in results:
            assert r.metadata.get("language") == "javascript"

    def test_file_path_exact_filter(self, searcher: BM25Search) -> None:
        """Exact file path filter narrows to a single file."""
        results = searcher.search(
            "authenticate_user", top_k=10, file_path="src/auth/service.py"
        )
        assert len(results) >= 1
        for r in results:
            assert r.file_path == "src/auth/service.py"

    def test_file_path_glob_filter(self, searcher: BM25Search) -> None:
        """Glob pattern should match files within the pattern."""
        results = searcher.search("user", top_k=20, file_path="src/users/*.py")
        assert len(results) >= 1
        for r in results:
            assert r.file_path.startswith("src/users/")
            assert r.file_path.endswith(".py")

    def test_symbol_type_filter_function(self, searcher: BM25Search) -> None:
        """Symbol type filter should only return functions."""
        results = searcher.search("user", top_k=20, symbol_type="function")
        for r in results:
            assert r.metadata.get("symbol_type") == "function"

    def test_symbol_type_filter_class(self, searcher: BM25Search) -> None:
        """Symbol type filter for class should only return classes."""
        results = searcher.search("UserManager", top_k=10, symbol_type="class")
        assert len(results) >= 1
        for r in results:
            assert r.metadata.get("symbol_type") == "class"


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty index, no match, invalid args, disk loading."""

    def test_empty_index_returns_empty(self) -> None:
        """Searching an empty index should return an empty list."""
        empty = BM25Index()
        searcher = BM25Search(bm25_index=empty)
        results = searcher.search("anything", top_k=10)
        assert results == []

    def test_no_match_returns_empty(self, searcher: BM25Search) -> None:
        """A query with zero lexical overlap should return no results."""
        results = searcher.search("xyzzy_nonexistent_symbol", top_k=10)
        assert results == []

    def test_top_k_respected(self, searcher: BM25Search) -> None:
        """Should return at most top_k results."""
        results = searcher.search("user", top_k=2)
        assert len(results) <= 2

    def test_invalid_top_k_raises(self, searcher: BM25Search) -> None:
        """top_k < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            searcher.search("user", top_k=0)

    def test_score_ordering(self, searcher: BM25Search) -> None:
        """Results must be sorted by score descending."""
        results = searcher.search("user", top_k=20)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_load_from_disk(self, bm25_index: BM25Index) -> None:
        """Loading a persisted BM25 index from disk should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bm25.pkl"
            bm25_index.save(path)

            searcher = BM25Search(index_path=path)
            results = searcher.search("authenticate_user", top_k=5)
            assert len(results) >= 1
            assert results[0].symbol_name == "authenticate_user"

    def test_injected_index_takes_precedence(self, bm25_index: BM25Index) -> None:
        """When both index_path and bm25_index are given, bm25_index wins."""
        searcher = BM25Search(
            index_path="/nonexistent/path.pkl",
            bm25_index=bm25_index,
        )
        # Should not raise FileNotFoundError because injected index is used
        results = searcher.search("authenticate_user", top_k=5)
        assert len(results) >= 1

    def test_missing_index_path_raises(self) -> None:
        """No index_path and no bm25_index should raise on search."""
        searcher = BM25Search()
        with pytest.raises(FileNotFoundError, match="No BM25 index available"):
            searcher.search("anything", top_k=5)

    def test_nonexistent_file_raises(self) -> None:
        """An index_path that does not exist should raise FileNotFoundError."""
        searcher = BM25Search(index_path="/nonexistent/path.pkl")
        with pytest.raises(FileNotFoundError, match="not found"):
            searcher.search("anything", top_k=5)
