"""Unit tests for the VectorSearch retrieval pipeline.

All tests run without network, GPU, or a real Qdrant instance -- every
external dependency is injected as a mock/fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from reporag.retrieval.vector_search import (
    RetrievalResult,
    VectorSearch,
    _build_doc_filter,
    _merge_results,
    build_filter,
)

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeScoredPoint:
    """Minimal stand-in for ``qdrant_client.models.ScoredPoint``."""

    id: str
    score: float
    payload: dict[str, Any]


def _make_code_point(
    score: float,
    file_path: str = "src/app.py",
    start_line: int = 1,
    end_line: int = 10,
    symbol: str | None = "app.main",
    content: str = "def main(): ...",
    point_id: str = "code-pt-1",
) -> _FakeScoredPoint:
    """Build a fake ScoredPoint mimicking the code collection payload."""
    return _FakeScoredPoint(
        id=point_id,
        score=score,
        payload={
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "qualified_name": symbol,
            "symbol": symbol,
            "content": content,
            "language": "python",
            "repo_id": "my-org/my-repo",
            "symbol_type": "function",
            "chunk_kind": "function",
        },
    )


def _make_doc_point(
    score: float,
    file_path: str = "src/app.py",
    start_line: int = 1,
    end_line: int = 5,
    symbol_id: str | None = "app.main",
    text: str = "Main entry point for the application.",
    point_id: str = "doc-pt-1",
) -> _FakeScoredPoint:
    """Build a fake ScoredPoint mimicking the docs collection payload."""
    return _FakeScoredPoint(
        id=point_id,
        score=score,
        payload={
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "symbol_id": symbol_id,
            "text": text,
            "language": "python",
            "repo_id": "my-org/my-repo",
            "doc_type": "docstring",
        },
    )


def _fake_embedder(dim: int) -> MagicMock:
    """Return a mock embedder whose ``embed()`` returns a vector of *dim*."""
    embedder = MagicMock()
    embedder.embed.return_value = (
        np.random.default_rng(42).random(dim).astype(np.float32)
    )
    return embedder


def _make_search(
    code_points: list[_FakeScoredPoint] | None = None,
    doc_points: list[_FakeScoredPoint] | None = None,
) -> VectorSearch:
    """Build a fully-mocked ``VectorSearch`` instance.

    The Qdrant client's ``search()`` returns *code_points* for the code
    collection and *doc_points* for the docs collection.
    """
    code_embedder = _fake_embedder(768)
    doc_embedder = _fake_embedder(384)
    client = MagicMock()

    def _search_side_effect(
        collection_name: str, **kwargs: Any
    ) -> list[_FakeScoredPoint]:
        if collection_name == "reporag_code":
            return code_points or []
        if collection_name == "reporag_docs":
            return doc_points or []
        return []

    client.search.side_effect = _search_side_effect

    return VectorSearch(
        code_embedder=code_embedder,
        doc_embedder=doc_embedder,
        qdrant_client=client,
        collection_code="reporag_code",
        collection_docs="reporag_docs",
    )


# ---------------------------------------------------------------------------
# RetrievalResult basics
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    """Tests for the shared RetrievalResult dataclass."""

    def test_defaults(self):
        r = RetrievalResult(
            score=0.95,
            file_path="src/app.py",
            start_line=1,
            end_line=10,
        )
        assert r.source == "vector_code"
        assert r.symbol is None
        assert r.content == ""
        assert r.metadata == {}
        assert r.point_id is None

    def test_all_fields(self):
        r = RetrievalResult(
            score=0.8,
            file_path="src/db.py",
            start_line=5,
            end_line=20,
            symbol="db.connect",
            content="def connect(): ...",
            source="bm25",
            point_id="abc-123",
            metadata={"language": "python"},
        )
        assert r.score == 0.8
        assert r.source == "bm25"
        assert r.point_id == "abc-123"
        assert r.metadata["language"] == "python"


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------


class TestBuildFilter:
    """Tests for the Qdrant filter construction helpers."""

    def test_no_params_returns_none(self):
        assert build_filter() is None

    def test_language_filter(self):
        flt = build_filter(language="python")
        assert flt is not None
        assert len(flt.must) == 1
        assert flt.must[0].key == "language"

    def test_symbol_type_filter(self):
        flt = build_filter(symbol_type="function")
        assert flt is not None
        assert flt.must[0].key == "symbol_type"

    def test_repo_id_filter(self):
        flt = build_filter(repo_id="my-org/my-repo")
        assert flt is not None
        assert flt.must[0].key == "repo_id"

    def test_combined_filters(self):
        """Multiple parameters compose into a single ``must`` clause."""
        flt = build_filter(
            language="python",
            symbol_type="class",
            repo_id="org/repo",
        )
        assert flt is not None
        assert len(flt.must) == 3
        keys = {c.key for c in flt.must}
        assert keys == {"language", "symbol_type", "repo_id"}


class TestBuildDocFilter:
    """Tests for the docs-collection-specific filter builder."""

    def test_no_params_returns_none(self):
        assert _build_doc_filter() is None

    def test_doc_type_filter(self):
        flt = _build_doc_filter(doc_type="docstring")
        assert flt is not None
        assert len(flt.must) == 1
        assert flt.must[0].key == "doc_type"

    def test_combined_doc_filters(self):
        flt = _build_doc_filter(
            language="python",
            doc_type="comment",
            repo_id="org/repo",
        )
        assert flt is not None
        assert len(flt.must) == 3
        keys = {c.key for c in flt.must}
        assert keys == {"language", "doc_type", "repo_id"}


# ---------------------------------------------------------------------------
# Merge & dedup
# ---------------------------------------------------------------------------


class TestMergeResults:
    """Tests for the result merging and deduplication logic."""

    def test_sorted_by_score_descending(self):
        results = [
            RetrievalResult(score=0.3, file_path="a.py", start_line=1, end_line=5),
            RetrievalResult(score=0.9, file_path="b.py", start_line=1, end_line=5),
            RetrievalResult(score=0.6, file_path="c.py", start_line=1, end_line=5),
        ]
        merged = _merge_results(results, top_k=10)
        scores = [r.score for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_dedup_keeps_higher_score(self):
        """Same (file_path, start_line) from two lists -- higher wins."""
        code = [
            RetrievalResult(
                score=0.7,
                file_path="src/app.py",
                start_line=10,
                end_line=20,
                source="vector_code",
            ),
        ]
        doc = [
            RetrievalResult(
                score=0.9,
                file_path="src/app.py",
                start_line=10,
                end_line=20,
                source="vector_doc",
            ),
        ]
        merged = _merge_results(code, doc, top_k=10)
        assert len(merged) == 1
        assert merged[0].source == "vector_doc"
        assert merged[0].score == 0.9

    def test_top_k_limits_output(self):
        results = [
            RetrievalResult(
                score=0.5 + i * 0.01,
                file_path=f"file_{i}.py",
                start_line=1,
                end_line=5,
            )
            for i in range(20)
        ]
        merged = _merge_results(results, top_k=5)
        assert len(merged) == 5

    def test_path_glob_filters(self):
        results = [
            RetrievalResult(
                score=0.9, file_path="src/auth/login.py", start_line=1, end_line=5
            ),
            RetrievalResult(
                score=0.8, file_path="tests/test_auth.py", start_line=1, end_line=5
            ),
            RetrievalResult(
                score=0.7, file_path="src/db/models.py", start_line=1, end_line=5
            ),
        ]
        merged = _merge_results(results, top_k=10, path_glob="src/*/*.py")
        assert len(merged) == 2
        paths = {r.file_path for r in merged}
        assert "tests/test_auth.py" not in paths

    def test_empty_input(self):
        merged = _merge_results([], [], top_k=10)
        assert merged == []

    def test_three_lists_merge(self):
        """Verify more than two lists can be merged (for future use)."""
        a = [RetrievalResult(score=0.9, file_path="a.py", start_line=1, end_line=5)]
        b = [RetrievalResult(score=0.8, file_path="b.py", start_line=1, end_line=5)]
        c = [RetrievalResult(score=0.7, file_path="c.py", start_line=1, end_line=5)]
        merged = _merge_results(a, b, c, top_k=10)
        assert len(merged) == 3
        assert merged[0].file_path == "a.py"


# ---------------------------------------------------------------------------
# VectorSearch: search_code
# ---------------------------------------------------------------------------


class TestSearchCode:
    """Tests for the code-collection-only search path."""

    def test_returns_retrieval_results(self):
        vs = _make_search(
            code_points=[_make_code_point(0.85)],
        )
        results = vs.search_code("auth handler", top_k=5)
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert results[0].source == "vector_code"
        assert results[0].score == 0.85

    def test_uses_code_embedder(self):
        vs = _make_search()
        vs.search_code("test", top_k=5)
        vs.code_embedder.embed.assert_called_once_with("test")
        vs.doc_embedder.embed.assert_not_called()

    def test_passes_filter_to_qdrant(self):
        vs = _make_search()
        vs.search_code("test", top_k=5, language="python", repo_id="org/repo")
        call_kwargs = vs.client.search.call_args
        assert call_kwargs.kwargs["query_filter"] is not None

    def test_point_id_is_captured(self):
        vs = _make_search(
            code_points=[_make_code_point(0.9, point_id="uuid-abc-123")],
        )
        results = vs.search_code("test", top_k=5)
        assert results[0].point_id == "uuid-abc-123"

    def test_graceful_on_qdrant_failure(self):
        """If Qdrant throws, search_code returns [] instead of crashing."""
        vs = _make_search()
        vs.client.search.side_effect = ConnectionError("Qdrant unreachable")
        results = vs.search_code("test", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# VectorSearch: search_docs
# ---------------------------------------------------------------------------


class TestSearchDocs:
    """Tests for the doc-collection-only search path."""

    def test_returns_doc_results(self):
        vs = _make_search(
            doc_points=[_make_doc_point(0.92)],
        )
        results = vs.search_docs("authentication flow", top_k=5)
        assert len(results) == 1
        assert results[0].source == "vector_doc"
        assert results[0].score == 0.92

    def test_uses_doc_embedder(self):
        vs = _make_search()
        vs.search_docs("test", top_k=5)
        vs.doc_embedder.embed.assert_called_once_with("test")
        vs.code_embedder.embed.assert_not_called()

    def test_point_id_is_captured(self):
        vs = _make_search(
            doc_points=[_make_doc_point(0.88, point_id="uuid-doc-456")],
        )
        results = vs.search_docs("test", top_k=5)
        assert results[0].point_id == "uuid-doc-456"

    def test_graceful_on_qdrant_failure(self):
        """If Qdrant throws, search_docs returns [] instead of crashing."""
        vs = _make_search()
        vs.client.search.side_effect = ConnectionError("Qdrant unreachable")
        results = vs.search_docs("test", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# VectorSearch: combined search
# ---------------------------------------------------------------------------


class TestSearch:
    """Tests for the merged search across both collections."""

    def test_merges_code_and_doc_results(self):
        vs = _make_search(
            code_points=[
                _make_code_point(0.80, file_path="src/auth.py", start_line=1),
            ],
            doc_points=[
                _make_doc_point(0.90, file_path="src/auth.py", start_line=50),
            ],
        )
        results = vs.search("auth", top_k=10)
        assert len(results) == 2
        # Doc result scored higher, should be first
        assert results[0].source == "vector_doc"
        assert results[1].source == "vector_code"

    def test_deduplicates_across_collections(self):
        """Same file+line from code and doc -- only higher-scored survives."""
        vs = _make_search(
            code_points=[
                _make_code_point(0.70, file_path="src/app.py", start_line=10),
            ],
            doc_points=[
                _make_doc_point(0.95, file_path="src/app.py", start_line=10),
            ],
        )
        results = vs.search("test", top_k=10)
        assert len(results) == 1
        assert results[0].score == 0.95

    def test_empty_results(self):
        vs = _make_search(code_points=[], doc_points=[])
        results = vs.search("nonexistent query")
        assert results == []

    def test_top_k_respected(self):
        code_pts = [
            _make_code_point(
                0.9 - i * 0.05,
                file_path=f"f{i}.py",
                start_line=1,
                point_id=f"code-{i}",
            )
            for i in range(10)
        ]
        doc_pts = [
            _make_doc_point(
                0.85 - i * 0.05,
                file_path=f"d{i}.py",
                start_line=1,
                point_id=f"doc-{i}",
            )
            for i in range(10)
        ]
        vs = _make_search(code_points=code_pts, doc_points=doc_pts)
        results = vs.search("test", top_k=5)
        assert len(results) == 5

    def test_path_glob_filtering(self):
        vs = _make_search(
            code_points=[
                _make_code_point(
                    0.9, file_path="src/auth/login.py", start_line=1, point_id="c1"
                ),
                _make_code_point(
                    0.8, file_path="tests/test_auth.py", start_line=1, point_id="c2"
                ),
            ],
            doc_points=[],
        )
        results = vs.search("auth", top_k=10, path_glob="src/*/*.py")
        assert len(results) == 1
        assert results[0].file_path == "src/auth/login.py"

    def test_results_sorted_by_score(self):
        vs = _make_search(
            code_points=[
                _make_code_point(0.5, file_path="low.py", start_line=1, point_id="c1"),
                _make_code_point(0.9, file_path="high.py", start_line=1, point_id="c2"),
                _make_code_point(0.7, file_path="mid.py", start_line=1, point_id="c3"),
            ],
            doc_points=[],
        )
        results = vs.search("test", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_both_embedders_called(self):
        """Combined search must use BOTH embedders if no symbol_type filter."""
        vs = _make_search()
        vs.search("test query", top_k=5)
        vs.code_embedder.embed.assert_called_once_with("test query")
        vs.doc_embedder.embed.assert_called_once_with("test query")

    def test_symbol_type_skips_doc_search(self):
        """If symbol_type filter is present, docs collection is skipped entirely."""
        vs = _make_search()
        vs.search("test query", top_k=5, symbol_type="function")
        vs.code_embedder.embed.assert_called_once_with("test query")
        # Should NOT embed or search for docs if symbol_type is provided
        vs.doc_embedder.embed.assert_not_called()

    def test_partial_qdrant_failure_returns_surviving_results(self):
        """If one collection fails, results from the other still return."""
        code_embedder = _fake_embedder(768)
        doc_embedder = _fake_embedder(384)
        client = MagicMock()

        call_count = 0

        def _failing_code_search(collection_name: str, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            if collection_name == "reporag_code":
                raise ConnectionError("code collection unreachable")
            return [_make_doc_point(0.85)]

        client.search.side_effect = _failing_code_search

        vs = VectorSearch(
            code_embedder=code_embedder,
            doc_embedder=doc_embedder,
            qdrant_client=client,
            collection_code="reporag_code",
            collection_docs="reporag_docs",
        )
        results = vs.search("test", top_k=5)
        # Code search failed but doc search succeeded
        assert len(results) == 1
        assert results[0].source == "vector_doc"
