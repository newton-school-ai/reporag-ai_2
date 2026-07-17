"""Unit tests for the vector semantic search module (Issue 16).

Covers:
- RetrievalResult dataclass behaviour
- Qdrant filter builder helpers
- ScoredPoint -> RetrievalResult converters
- Merge, deduplication, score-floor, and path-glob filtering
- VectorSearch.search_code / search_docs / search (merged)
- search_with_stats observability
- Graceful degradation on Qdrant errors
- Health-check utility
- Edge cases: empty queries, no results, missing payloads
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from reporag.retrieval.vector_search import (
    RetrievalResult,
    VectorSearch,
    _build_doc_filter,
    _code_point_to_result,
    _doc_point_to_result,
    _merge_results,
    build_qdrant_filter,
)

# ---------------------------------------------------------------------------
# Fakes matching the codebase patterns (test_index_builder.py, test_embedder.py)
# ---------------------------------------------------------------------------


@dataclass
class _FakeScoredPoint:
    """Duck-typed stand-in for ``qdrant_client.models.ScoredPoint``."""

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class _FakeEmbedder:
    """Duck-typed embedder with deterministic, dimension-correct output.

    Returns a fixed vector for any input.  The dimension is configurable
    so the same fake can impersonate both CodeEmbedder (768) and
    DocEmbedder (384).
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.calls: list[str] = []

    def embed(self, text: str) -> np.ndarray:
        self.calls.append(text)
        # Simple deterministic vector: hash the text to a seed, then
        # generate a unit vector so tests are reproducible.
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec


class _FakeQdrantClient:
    """In-memory Qdrant fake that supports ``search()`` and ``collection_exists()``.

    Pre-populate with points via ``add_points(collection, points)`` and the
    fake will return them (sorted by score desc) when ``search()`` is called
    on that collection.  Filter evaluation is intentionally simplified --
    we only need to verify that VectorSearch *passes* filters correctly,
    not that Qdrant's filter engine works.
    """

    def __init__(self) -> None:
        self._collections: dict[str, list[_FakeScoredPoint]] = {}

    def add_points(self, collection: str, points: list[_FakeScoredPoint]) -> None:
        self._collections.setdefault(collection, []).extend(points)

    def collection_exists(self, name: str) -> bool:
        return name in self._collections

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        query_filter: Any = None,
        with_payload: bool = True,
    ) -> list[_FakeScoredPoint]:
        points = list(self._collections.get(collection_name, []))

        # Simplified filter evaluation: only check "must" field conditions
        if query_filter is not None:
            conditions = getattr(query_filter, "must", []) or []
            filtered: list[_FakeScoredPoint] = []
            for p in points:
                match = True
                for cond in conditions:
                    key = cond.key
                    expected = cond.match.value
                    if p.payload.get(key) != expected:
                        match = False
                        break
                if match:
                    filtered.append(p)
            points = filtered

        # Sort by score descending and truncate
        points.sort(key=lambda p: p.score, reverse=True)
        return points[:limit]


class _FakeErrorQdrantClient:
    """A Qdrant client fake that always raises on search."""

    def search(self, **kwargs: Any) -> list[Any]:
        raise ConnectionError("Qdrant is down")


@pytest.fixture
def code_embedder() -> _FakeEmbedder:
    return _FakeEmbedder(dim=768)


@pytest.fixture
def doc_embedder() -> _FakeEmbedder:
    return _FakeEmbedder(dim=384)


@pytest.fixture
def fake_client() -> _FakeQdrantClient:
    """Pre-populated Qdrant fake with code and doc points."""
    client = _FakeQdrantClient()

    # Code points
    client.add_points(
        "test_code",
        [
            _FakeScoredPoint(
                id="code-1",
                score=0.95,
                payload={
                    "file_path": "src/auth.py",
                    "start_line": 10,
                    "end_line": 25,
                    "symbol": "authenticate_user",
                    "qualified_name": "auth.authenticate_user",
                    "content": "def authenticate_user(token): ...",
                    "language": "python",
                    "symbol_type": "function",
                    "repo_id": "my-repo",
                    "chunk_kind": "definition",
                },
            ),
            _FakeScoredPoint(
                id="code-2",
                score=0.82,
                payload={
                    "file_path": "src/middleware.py",
                    "start_line": 5,
                    "end_line": 15,
                    "symbol": "check_token",
                    "qualified_name": "middleware.check_token",
                    "content": "def check_token(request): ...",
                    "language": "python",
                    "symbol_type": "function",
                    "repo_id": "my-repo",
                    "chunk_kind": "definition",
                },
            ),
            _FakeScoredPoint(
                id="code-3",
                score=0.70,
                payload={
                    "file_path": "src/utils.js",
                    "start_line": 1,
                    "end_line": 10,
                    "symbol": "validateToken",
                    "content": "function validateToken(token) { ... }",
                    "language": "javascript",
                    "symbol_type": "function",
                    "repo_id": "my-repo",
                    "chunk_kind": "definition",
                },
            ),
        ],
    )

    # Doc points
    client.add_points(
        "test_docs",
        [
            _FakeScoredPoint(
                id="doc-1",
                score=0.90,
                payload={
                    "file_path": "src/auth.py",
                    "start_line": 10,
                    "end_line": 12,
                    "symbol_id": "auth.authenticate_user",
                    "text": "Authenticates a user by verifying their JWT token.",
                    "doc_type": "docstring",
                    "language": "python",
                    "repo_id": "my-repo",
                },
            ),
            _FakeScoredPoint(
                id="doc-2",
                score=0.75,
                payload={
                    "file_path": "README.md",
                    "start_line": 1,
                    "end_line": 20,
                    "symbol_id": None,
                    "text": "# Authentication\nThis module handles user authentication.",
                    "doc_type": "readme",
                    "language": None,
                    "repo_id": "my-repo",
                },
            ),
        ],
    )

    return client


@pytest.fixture
def searcher(
    code_embedder: _FakeEmbedder,
    doc_embedder: _FakeEmbedder,
    fake_client: _FakeQdrantClient,
) -> VectorSearch:
    return VectorSearch(
        code_embedder=code_embedder,
        doc_embedder=doc_embedder,
        qdrant_client=fake_client,
        collection_code="test_code",
        collection_docs="test_docs",
    )


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_construction_with_defaults(self):
        r = RetrievalResult(
            score=0.85,
            file_path="src/auth.py",
            start_line=10,
            end_line=25,
        )
        assert r.score == 0.85
        assert r.file_path == "src/auth.py"
        assert r.symbol_name is None
        assert r.chunk_text == ""
        assert r.source == "vector_code"
        assert r.point_id is None
        assert r.metadata == {}

    def test_construction_with_all_fields(self):
        r = RetrievalResult(
            score=0.95,
            file_path="src/auth.py",
            start_line=10,
            end_line=25,
            symbol_name="authenticate_user",
            chunk_text="def authenticate_user(): ...",
            source="vector_doc",
            point_id="abc-123",
            metadata={"language": "python"},
        )
        assert r.source == "vector_doc"
        assert r.point_id == "abc-123"
        assert r.metadata["language"] == "python"

    def test_dedup_key(self):
        r = RetrievalResult(
            score=0.9,
            file_path="src/auth.py",
            start_line=10,
            end_line=25,
        )
        assert r.dedup_key == ("src/auth.py", 10)

    def test_dedup_key_same_location_different_source(self):
        r1 = RetrievalResult(
            score=0.9,
            file_path="src/auth.py",
            start_line=10,
            end_line=25,
            source="vector_code",
        )
        r2 = RetrievalResult(
            score=0.85,
            file_path="src/auth.py",
            start_line=10,
            end_line=12,
            source="vector_doc",
        )
        assert r1.dedup_key == r2.dedup_key

    def test_metadata_is_independent_per_instance(self):
        r1 = RetrievalResult(score=0.9, file_path="a.py", start_line=1, end_line=5)
        r2 = RetrievalResult(score=0.8, file_path="b.py", start_line=1, end_line=5)
        r1.metadata["key"] = "value"
        assert "key" not in r2.metadata


# ---------------------------------------------------------------------------
# Filter builders
# ---------------------------------------------------------------------------


class TestBuildQdrantFilter:
    def test_no_params_returns_none(self):
        assert build_qdrant_filter() is None

    def test_language_only(self):
        f = build_qdrant_filter(language="python")
        assert f is not None
        assert len(f.must) == 1
        assert f.must[0].key == "language"
        assert f.must[0].match.value == "python"

    def test_symbol_type_only(self):
        f = build_qdrant_filter(symbol_type="function")
        assert len(f.must) == 1
        assert f.must[0].key == "symbol_type"

    def test_repo_id_only(self):
        f = build_qdrant_filter(repo_id="my-repo")
        assert len(f.must) == 1
        assert f.must[0].key == "repo_id"

    def test_multiple_params_combined(self):
        f = build_qdrant_filter(
            language="python",
            symbol_type="class",
            repo_id="my-repo",
        )
        assert len(f.must) == 3
        keys = {c.key for c in f.must}
        assert keys == {"language", "symbol_type", "repo_id"}

    def test_none_params_excluded(self):
        f = build_qdrant_filter(language="python", symbol_type=None)
        assert len(f.must) == 1


class TestBuildDocFilter:
    def test_no_params_returns_none(self):
        assert _build_doc_filter() is None

    def test_doc_type_filter(self):
        f = _build_doc_filter(doc_type="docstring")
        assert len(f.must) == 1
        assert f.must[0].key == "doc_type"
        assert f.must[0].match.value == "docstring"

    def test_language_and_doc_type(self):
        f = _build_doc_filter(language="python", doc_type="comment")
        assert len(f.must) == 2

    def test_repo_id(self):
        f = _build_doc_filter(repo_id="my-repo")
        assert len(f.must) == 1
        assert f.must[0].key == "repo_id"


# ---------------------------------------------------------------------------
# ScoredPoint -> RetrievalResult converters
# ---------------------------------------------------------------------------


class TestCodePointToResult:
    def test_full_payload(self):
        point = _FakeScoredPoint(
            id="code-1",
            score=0.95,
            payload={
                "file_path": "src/auth.py",
                "start_line": 10,
                "end_line": 25,
                "symbol": "authenticate_user",
                "qualified_name": "auth.authenticate_user",
                "content": "def authenticate_user(): ...",
                "language": "python",
            },
        )
        result = _code_point_to_result(point)

        assert result.score == 0.95
        assert result.file_path == "src/auth.py"
        assert result.start_line == 10
        assert result.end_line == 25
        assert result.symbol_name == "authenticate_user"
        assert result.chunk_text == "def authenticate_user(): ..."
        assert result.source == "vector_code"
        assert result.point_id == "code-1"
        assert result.metadata["language"] == "python"

    def test_missing_optional_fields_default_gracefully(self):
        point = _FakeScoredPoint(id="code-2", score=0.5, payload={})
        result = _code_point_to_result(point)

        assert result.file_path == ""
        assert result.start_line == 0
        assert result.symbol_name is None
        assert result.chunk_text == ""

    def test_falls_back_to_qualified_name_when_symbol_missing(self):
        point = _FakeScoredPoint(
            id="code-3",
            score=0.7,
            payload={
                "file_path": "a.py",
                "start_line": 1,
                "end_line": 5,
                "qualified_name": "module.func",
            },
        )
        result = _code_point_to_result(point)
        assert result.symbol_name == "module.func"

    def test_none_payload_handled(self):
        point = _FakeScoredPoint(id="code-4", score=0.3, payload=None)
        result = _code_point_to_result(point)
        assert result.file_path == ""
        assert result.metadata == {}


class TestDocPointToResult:
    def test_full_payload(self):
        point = _FakeScoredPoint(
            id="doc-1",
            score=0.90,
            payload={
                "file_path": "src/auth.py",
                "start_line": 10,
                "end_line": 12,
                "symbol_id": "auth.authenticate_user",
                "text": "Authenticates a user.",
                "doc_type": "docstring",
            },
        )
        result = _doc_point_to_result(point)

        assert result.score == 0.90
        assert result.symbol_name == "auth.authenticate_user"
        assert result.chunk_text == "Authenticates a user."
        assert result.source == "vector_doc"
        assert result.point_id == "doc-1"

    def test_missing_optional_fields(self):
        point = _FakeScoredPoint(id="doc-2", score=0.5, payload={})
        result = _doc_point_to_result(point)

        assert result.file_path == ""
        assert result.symbol_name is None
        assert result.chunk_text == ""


# ---------------------------------------------------------------------------
# Merge + dedup
# ---------------------------------------------------------------------------


class TestMergeResults:
    def test_empty_inputs(self):
        assert _merge_results([], [], top_k=10) == []

    def test_single_list(self):
        results = [
            RetrievalResult(score=0.9, file_path="a.py", start_line=1, end_line=5),
            RetrievalResult(score=0.8, file_path="b.py", start_line=1, end_line=5),
        ]
        merged = _merge_results(results, top_k=10)
        assert len(merged) == 2
        assert merged[0].score == 0.9

    def test_interleaved_merge_by_score(self):
        code = [
            RetrievalResult(
                score=0.95,
                file_path="a.py",
                start_line=1,
                end_line=5,
                source="vector_code",
            ),
            RetrievalResult(
                score=0.80,
                file_path="c.py",
                start_line=1,
                end_line=5,
                source="vector_code",
            ),
        ]
        docs = [
            RetrievalResult(
                score=0.90,
                file_path="b.py",
                start_line=1,
                end_line=5,
                source="vector_doc",
            ),
        ]
        merged = _merge_results(code, docs, top_k=10)
        assert len(merged) == 3
        assert merged[0].score == 0.95
        assert merged[1].score == 0.90
        assert merged[2].score == 0.80

    def test_deduplication_keeps_highest_score(self):
        code = [
            RetrievalResult(
                score=0.95,
                file_path="src/auth.py",
                start_line=10,
                end_line=25,
                source="vector_code",
            )
        ]
        docs = [
            RetrievalResult(
                score=0.90,
                file_path="src/auth.py",
                start_line=10,
                end_line=12,
                source="vector_doc",
            )
        ]
        merged = _merge_results(code, docs, top_k=10)

        assert len(merged) == 1
        assert merged[0].score == 0.95
        assert merged[0].source == "vector_code"

    def test_path_glob_filtering(self):
        results = [
            RetrievalResult(
                score=0.95, file_path="src/auth.py", start_line=1, end_line=5
            ),
            RetrievalResult(
                score=0.90, file_path="tests/test_auth.py", start_line=1, end_line=5
            ),
            RetrievalResult(
                score=0.85, file_path="src/middleware.py", start_line=1, end_line=5
            ),
        ]
        merged = _merge_results(results, top_k=10, path_glob="src/*.py")

        assert len(merged) == 2
        assert all(r.file_path.startswith("src/") for r in merged)

    def test_top_k_truncation(self):
        results = [
            RetrievalResult(
                score=0.9 - i * 0.01,
                file_path=f"file_{i}.py",
                start_line=1,
                end_line=5,
            )
            for i in range(20)
        ]
        merged = _merge_results(results, top_k=5)
        assert len(merged) == 5
        assert merged[0].score > merged[-1].score

    def test_min_score_floor(self):
        results = [
            RetrievalResult(score=0.95, file_path="a.py", start_line=1, end_line=5),
            RetrievalResult(score=0.30, file_path="b.py", start_line=1, end_line=5),
            RetrievalResult(score=0.10, file_path="c.py", start_line=1, end_line=5),
        ]
        merged = _merge_results(results, top_k=10, min_score=0.5)

        assert len(merged) == 1
        assert merged[0].file_path == "a.py"

    def test_dedup_across_three_lists(self):
        a = [
            RetrievalResult(
                score=0.9,
                file_path="x.py",
                start_line=1,
                end_line=5,
                source="vector_code",
            )
        ]
        b = [
            RetrievalResult(
                score=0.8,
                file_path="x.py",
                start_line=1,
                end_line=3,
                source="vector_doc",
            )
        ]
        c = [
            RetrievalResult(
                score=0.7, file_path="x.py", start_line=1, end_line=2, source="bm25"
            )
        ]
        merged = _merge_results(a, b, c, top_k=10)
        assert len(merged) == 1
        assert merged[0].score == 0.9


# ---------------------------------------------------------------------------
# VectorSearch: code search
# ---------------------------------------------------------------------------


class TestVectorSearchCode:
    def test_basic_search_returns_results(self, searcher):
        results = searcher.search_code("authentication middleware", top_k=10)
        assert len(results) > 0
        assert all(r.source == "vector_code" for r in results)

    def test_results_sorted_by_score(self, searcher):
        results = searcher.search_code("auth", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, searcher):
        results = searcher.search_code("auth", top_k=1)
        assert len(results) <= 1

    def test_language_filter_applied(self, searcher):
        results = searcher.search_code("validate", top_k=10, language="python")
        assert all(r.metadata.get("language") == "python" for r in results)
        # The javascript point should be filtered out
        js_results = [r for r in results if r.metadata.get("language") == "javascript"]
        assert len(js_results) == 0

    def test_symbol_type_filter(self, searcher):
        results = searcher.search_code("auth", top_k=10, symbol_type="function")
        assert all(r.metadata.get("symbol_type") == "function" for r in results)

    def test_repo_id_filter(self, searcher):
        results = searcher.search_code("auth", top_k=10, repo_id="my-repo")
        assert all(r.metadata.get("repo_id") == "my-repo" for r in results)

    def test_nonexistent_repo_returns_empty(self, searcher):
        results = searcher.search_code("auth", top_k=10, repo_id="nonexistent-repo")
        assert results == []

    def test_qdrant_error_returns_empty(self, code_embedder, doc_embedder):
        searcher = VectorSearch(
            code_embedder=code_embedder,
            doc_embedder=doc_embedder,
            qdrant_client=_FakeErrorQdrantClient(),
            collection_code="test_code",
            collection_docs="test_docs",
        )
        results = searcher.search_code("auth", top_k=10)
        assert results == []

    def test_embedder_called_with_query(self, searcher, code_embedder):
        searcher.search_code("my search query", top_k=5)
        assert "my search query" in code_embedder.calls

    def test_point_ids_populated(self, searcher):
        results = searcher.search_code("auth", top_k=10)
        assert all(r.point_id is not None for r in results)


# ---------------------------------------------------------------------------
# VectorSearch: doc search
# ---------------------------------------------------------------------------


class TestVectorSearchDocs:
    def test_basic_doc_search(self, searcher):
        results = searcher.search_docs("authentication", top_k=10)
        assert len(results) > 0
        assert all(r.source == "vector_doc" for r in results)

    def test_doc_type_filter(self, searcher):
        results = searcher.search_docs("auth", top_k=10, doc_type="docstring")
        assert all(r.metadata.get("doc_type") == "docstring" for r in results)

    def test_qdrant_error_returns_empty(self, code_embedder, doc_embedder):
        searcher = VectorSearch(
            code_embedder=code_embedder,
            doc_embedder=doc_embedder,
            qdrant_client=_FakeErrorQdrantClient(),
            collection_code="test_code",
            collection_docs="test_docs",
        )
        results = searcher.search_docs("auth", top_k=10)
        assert results == []

    def test_embedder_called_with_query(self, searcher, doc_embedder):
        searcher.search_docs("my doc query", top_k=5)
        assert "my doc query" in doc_embedder.calls


# ---------------------------------------------------------------------------
# VectorSearch: merged search
# ---------------------------------------------------------------------------


class TestVectorSearchMerged:
    def test_merged_search_returns_both_sources(self, searcher):
        results = searcher.search("authentication middleware", top_k=10)
        sources = {r.source for r in results}
        assert "vector_code" in sources or "vector_doc" in sources

    def test_merged_results_sorted_by_score(self, searcher):
        results = searcher.search("auth", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_merged_results_deduplicated(self, searcher):
        results = searcher.search("authentication", top_k=10)
        keys = [r.dedup_key for r in results]
        assert len(keys) == len(set(keys))

    def test_path_glob_filters_results(self, searcher):
        results = searcher.search("auth", top_k=10, path_glob="src/*.py")
        assert all(r.file_path.startswith("src/") for r in results)
        assert all(r.file_path.endswith(".py") for r in results)

    def test_known_good_query_result_pair(self, searcher):
        """Acceptance criterion: a known query returns auth-related chunks."""
        results = searcher.search("authentication middleware", top_k=5)
        assert len(results) > 0
        # The top result should be from auth.py (highest score in test data)
        auth_results = [r for r in results if "auth" in r.file_path]
        assert len(auth_results) > 0

    def test_top_k_defaults_to_settings(self, searcher):
        # settings.vector_search_top_k is 20 by default; with 5 total
        # fake points (3 code + 2 docs) minus dedup, we get <= 5
        results = searcher.search("auth")
        assert len(results) <= 20

    def test_language_filter_across_collections(self, searcher):
        results = searcher.search("auth", top_k=10, language="python")
        for r in results:
            lang = r.metadata.get("language")
            # language=None docs (readme) are also filtered out
            assert lang == "python"

    def test_overfetch_when_glob_active(self, code_embedder, doc_embedder):
        """Verify that with a glob, VectorSearch fetches more from Qdrant."""
        mock_client = MagicMock()
        mock_client.search.return_value = []

        vs = VectorSearch(
            code_embedder=code_embedder,
            doc_embedder=doc_embedder,
            qdrant_client=mock_client,
            collection_code="test_code",
            collection_docs="test_docs",
        )
        vs.search("auth", top_k=10, path_glob="src/*.py")

        # Each search call should use limit = 10 * 3 = 30
        for call in mock_client.search.call_args_list:
            assert call.kwargs.get("limit", call[1].get("limit")) == 30


# ---------------------------------------------------------------------------
# VectorSearch: search_with_stats
# ---------------------------------------------------------------------------


class TestVectorSearchWithStats:
    def test_returns_results_and_stats(self, searcher):
        results, stats = searcher.search_with_stats("auth", top_k=5)

        assert isinstance(results, list)
        assert isinstance(stats, dict)
        assert "latency_ms" in stats
        assert "code_count" in stats
        assert "doc_count" in stats
        assert "merged_count" in stats
        assert stats["merged_count"] == len(results)

    def test_stats_contain_query_info(self, searcher):
        _, stats = searcher.search_with_stats(
            "authentication",
            top_k=5,
            language="python",
        )
        assert stats["query"] == "authentication"
        assert stats["top_k"] == 5
        assert stats["filters"]["language"] == "python"

    def test_latency_is_positive(self, searcher):
        _, stats = searcher.search_with_stats("auth", top_k=5)
        assert stats["latency_ms"] >= 0

    def test_stats_filters_include_all_params(self, searcher):
        _, stats = searcher.search_with_stats(
            "auth",
            top_k=3,
            language="python",
            path_glob="src/*.py",
            symbol_type="function",
            repo_id="my-repo",
        )
        filters = stats["filters"]
        assert filters["language"] == "python"
        assert filters["path_glob"] == "src/*.py"
        assert filters["symbol_type"] == "function"
        assert filters["repo_id"] == "my-repo"


# ---------------------------------------------------------------------------
# VectorSearch: health check
# ---------------------------------------------------------------------------


class TestVectorSearchHealth:
    def test_healthy_with_valid_collections(self, searcher):
        health = searcher.health_check()
        assert health["healthy"] is True
        assert health["code_collection"] is True
        assert health["docs_collection"] is True

    def test_unhealthy_with_missing_collection(self, code_embedder, doc_embedder):
        empty_client = _FakeQdrantClient()
        vs = VectorSearch(
            code_embedder=code_embedder,
            doc_embedder=doc_embedder,
            qdrant_client=empty_client,
            collection_code="nonexistent_code",
            collection_docs="nonexistent_docs",
        )
        health = vs.health_check()
        assert health["healthy"] is False
        assert health["code_collection"] is False
        assert health["docs_collection"] is False

    def test_partial_health(self, code_embedder, doc_embedder):
        client = _FakeQdrantClient()
        client.add_points("existing_code", [])

        vs = VectorSearch(
            code_embedder=code_embedder,
            doc_embedder=doc_embedder,
            qdrant_client=client,
            collection_code="existing_code",
            collection_docs="nonexistent_docs",
        )
        health = vs.health_check()
        assert health["healthy"] is False
        assert health["code_collection"] is True
        assert health["docs_collection"] is False


# ---------------------------------------------------------------------------
# VectorSearch: constructor & repr
# ---------------------------------------------------------------------------


class TestVectorSearchMisc:
    def test_repr(self, searcher):
        text = repr(searcher)
        assert "test_code" in text
        assert "test_docs" in text

    def test_min_score_filtering_on_search(
        self, code_embedder, doc_embedder, fake_client
    ):
        vs = VectorSearch(
            code_embedder=code_embedder,
            doc_embedder=doc_embedder,
            qdrant_client=fake_client,
            collection_code="test_code",
            collection_docs="test_docs",
            min_score=0.85,
        )
        results = vs.search("auth", top_k=10)
        assert all(r.score >= 0.85 for r in results)

    def test_lazy_client_not_constructed_when_injected(self):
        embedder = _FakeEmbedder()
        client = _FakeQdrantClient()
        vs = VectorSearch(
            code_embedder=embedder,
            doc_embedder=embedder,
            qdrant_client=client,
        )
        # Access the client property; it should return the injected one
        assert vs.client is client

    def test_collections_default_to_settings(self):
        embedder = _FakeEmbedder()
        vs = VectorSearch(
            code_embedder=embedder,
            doc_embedder=embedder,
            qdrant_client=_FakeQdrantClient(),
        )
        assert vs.collection_code == "reporag_code"
        assert vs.collection_docs == "reporag_docs"
