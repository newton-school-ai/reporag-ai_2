"""Unit tests for vector semantic search (Issue #16).

All tests use injected fakes for both embedders and the QdrantClient so no
real network calls or model loads occur.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qdrant_client.http import models

from reporag.retrieval.vector_search import VectorSearch

# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------


class FakeVector:
    """Minimal EmbeddingVector stub -- satisfies the protocol via tolist()."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


def make_embedder(dim: int) -> MagicMock:
    """Return a mock embedder whose embed() returns a FakeVector of given dim."""
    embedder = MagicMock()
    embedder.embed.return_value = FakeVector([0.0] * dim)
    return embedder


def scored_point(
    point_id: str,
    score: float,
    payload: dict,
) -> models.ScoredPoint:
    return models.ScoredPoint(
        id=point_id,
        version=1,
        score=score,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Common code/doc payloads matching Issue #15 schema exactly
# ---------------------------------------------------------------------------

CODE_PAYLOAD_A = {
    "file_path": "src/auth.py",
    "start_line": 10,
    "end_line": 25,
    "symbol": "authenticate",
    "content": "def authenticate(user): ...",
    "repo_id": "org/repo",
    "language": "python",
    "symbol_type": "function",
}

CODE_PAYLOAD_B = {
    "file_path": "src/utils.py",
    "start_line": 1,
    "end_line": 5,
    "symbol": "helper",
    "content": "def helper(): ...",
    "repo_id": "org/repo",
    "language": "python",
    "symbol_type": "function",
}

DOC_PAYLOAD_A = {
    "file_path": "src/auth.py",
    "start_line": 10,
    "end_line": 25,
    "symbol_id": "authenticate",
    "text": "Authenticates a user given credentials.",
    "repo_id": "org/repo",
    "language": "python",
}

DOC_PAYLOAD_B = {
    "file_path": "README.md",
    "start_line": 0,
    "end_line": 0,
    "symbol_id": None,
    "text": "Overview of the project.",
    "repo_id": "org/repo",
    "language": None,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_qdrant() -> MagicMock:
    return MagicMock()


@pytest.fixture
def code_embedder() -> MagicMock:
    return make_embedder(768)


@pytest.fixture
def doc_embedder() -> MagicMock:
    return make_embedder(384)


@pytest.fixture
def searcher(mock_qdrant, code_embedder, doc_embedder) -> VectorSearch:
    return VectorSearch(
        code_embedder=code_embedder,
        doc_embedder=doc_embedder,
        client=mock_qdrant,
    )


# ---------------------------------------------------------------------------
# Dual collection search & merge
# ---------------------------------------------------------------------------


def test_searches_both_collections(searcher, mock_qdrant):
    mock_qdrant.search.return_value = []
    searcher.search("auth query", top_k=5)
    assert mock_qdrant.search.call_count == 2
    collections_searched = [
        c.kwargs["collection_name"] for c in mock_qdrant.search.call_args_list
    ]
    assert searcher.code_collection in collections_searched
    assert searcher.docs_collection in collections_searched


def test_dual_collection_correct_vector_sizes(
    searcher, mock_qdrant, code_embedder, doc_embedder
):
    mock_qdrant.search.return_value = []
    searcher.search("query", top_k=5)

    code_call, doc_call = mock_qdrant.search.call_args_list
    assert len(code_call.kwargs["query_vector"]) == 768
    assert len(doc_call.kwargs["query_vector"]) == 384


def test_merge_and_ranking(searcher, mock_qdrant):
    mock_qdrant.search.side_effect = [
        [scored_point("code-1", 0.8, CODE_PAYLOAD_A)],
        [
            scored_point("doc-1", 0.95, DOC_PAYLOAD_A),
            scored_point("doc-2", 0.7, DOC_PAYLOAD_B),
        ],
    ]
    results = searcher.search("auth", top_k=10)

    assert len(results) == 3
    assert results[0].score == 0.95
    assert results[1].score == 0.8
    assert results[2].score == 0.7


def test_retrieval_result_mapping_code(searcher, mock_qdrant):
    mock_qdrant.search.side_effect = [
        [scored_point("code-1", 0.9, CODE_PAYLOAD_A)],
        [],
    ]
    results = searcher.search("query", top_k=5)
    r = results[0]

    assert r.score == 0.9
    assert r.file_path == "src/auth.py"
    assert r.lines == (10, 25)
    assert r.symbol == "authenticate"
    assert r.chunk_text == "def authenticate(user): ..."
    assert r.point_id == "code-1"
    assert r.source == "vector_code"


def test_retrieval_result_mapping_doc(searcher, mock_qdrant):
    mock_qdrant.search.side_effect = [
        [],
        [scored_point("doc-1", 0.88, DOC_PAYLOAD_A)],
    ]
    results = searcher.search("query", top_k=5)
    r = results[0]

    assert r.score == 0.88
    assert r.file_path == "src/auth.py"
    assert r.lines == (10, 25)
    assert r.symbol == "authenticate"
    assert r.chunk_text == "Authenticates a user given credentials."
    assert r.point_id == "doc-1"
    assert r.source == "vector_doc"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_language_filter(searcher, mock_qdrant):
    mock_qdrant.search.return_value = []
    searcher.search("q", top_k=5, language="python")

    for c in mock_qdrant.search.call_args_list:
        flt = c.kwargs["query_filter"]
        assert flt is not None
        keys = [cond.key for cond in flt.must]
        assert "language" in keys


def test_file_path_filter(searcher, mock_qdrant):
    mock_qdrant.search.return_value = []
    searcher.search("q", top_k=5, file_path="src/auth.py")

    for c in mock_qdrant.search.call_args_list:
        flt = c.kwargs["query_filter"]
        keys = [cond.key for cond in flt.must]
        assert "file_path" in keys


def test_repo_id_filter(searcher, mock_qdrant):
    mock_qdrant.search.return_value = []
    searcher.search("q", top_k=5, repo_id="org/repo")

    for c in mock_qdrant.search.call_args_list:
        flt = c.kwargs["query_filter"]
        keys = [cond.key for cond in flt.must]
        assert "repo_id" in keys


def test_symbol_type_filter_on_code_only(searcher, mock_qdrant):
    mock_qdrant.search.return_value = []
    searcher.search("q", top_k=5, symbol_type="function")

    # Only one collection should be searched when symbol_type is provided
    assert mock_qdrant.search.call_count == 1
    code_call = mock_qdrant.search.call_args_list[0]
    assert code_call.kwargs["collection_name"] == searcher.code_collection
    flt = code_call.kwargs["query_filter"]
    keys = [cond.key for cond in flt.must]
    assert "symbol_type" in keys


def test_no_filter_when_no_criteria(searcher, mock_qdrant):
    mock_qdrant.search.return_value = []
    searcher.search("q", top_k=5)

    for c in mock_qdrant.search.call_args_list:
        assert c.kwargs["query_filter"] is None


def test_combined_filters(searcher, mock_qdrant):
    mock_qdrant.search.return_value = []
    searcher.search("q", top_k=5, language="python", file_path="a.py", repo_id="r/r")

    for c in mock_qdrant.search.call_args_list:
        flt = c.kwargs["query_filter"]
        assert len(flt.must) == 3


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_deduplication_on_source_and_point_id(searcher, mock_qdrant):
    """Same (source, point_id) pair must appear only once in results."""
    mock_qdrant.search.side_effect = [
        [
            scored_point("code-1", 0.9, CODE_PAYLOAD_A),
            scored_point("code-1", 0.85, CODE_PAYLOAD_A),  # duplicate
        ],
        [
            scored_point("doc-1", 0.88, DOC_PAYLOAD_A),
        ],
    ]
    results = searcher.search("q", top_k=10)

    point_ids = [(r.source, r.point_id) for r in results]
    assert len(point_ids) == len(set(point_ids))
    assert len(results) == 2  # code-1 and doc-1


def test_code_and_doc_same_point_id_not_collapsed(searcher, mock_qdrant):
    """A code hit and a doc hit sharing the same Qdrant point_id string
    must NOT be merged, because they come from different embedding spaces
    and represent complementary views (raw code vs. docstring).
    """
    mock_qdrant.search.side_effect = [
        [scored_point("shared-uuid", 0.9, CODE_PAYLOAD_A)],
        [scored_point("shared-uuid", 0.85, DOC_PAYLOAD_A)],
    ]
    results = searcher.search("q", top_k=10)
    sources = {r.source for r in results}
    assert sources == {"vector_code", "vector_doc"}


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def test_code_collection_failure_still_returns_doc_results(searcher, mock_qdrant):
    mock_qdrant.search.side_effect = [
        Exception("Qdrant timeout"),
        [scored_point("doc-1", 0.8, DOC_PAYLOAD_A)],
    ]
    results = searcher.search("q", top_k=5)
    assert len(results) == 1
    assert results[0].source == "vector_doc"


def test_doc_collection_failure_still_returns_code_results(searcher, mock_qdrant):
    mock_qdrant.search.side_effect = [
        [scored_point("code-1", 0.9, CODE_PAYLOAD_A)],
        Exception("Qdrant timeout"),
    ]
    results = searcher.search("q", top_k=5)
    assert len(results) == 1
    assert results[0].source == "vector_code"


def test_both_collections_fail_returns_empty(searcher, mock_qdrant):
    mock_qdrant.search.side_effect = Exception("Total failure")
    results = searcher.search("q", top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# top_k behaviour
# ---------------------------------------------------------------------------


def test_top_k_limits_results(searcher, mock_qdrant):
    mock_qdrant.search.side_effect = [
        [scored_point(f"c{i}", 1.0 - i * 0.05, CODE_PAYLOAD_A) for i in range(10)],
        [],
    ]
    results = searcher.search("q", top_k=3)
    assert len(results) == 3


def test_top_k_zero_raises(searcher):
    with pytest.raises(ValueError, match="top_k"):
        searcher.search("q", top_k=0)


def test_top_k_negative_raises(searcher):
    with pytest.raises(ValueError, match="top_k"):
        searcher.search("q", top_k=-1)


# ---------------------------------------------------------------------------
# Lazy client initialization
# ---------------------------------------------------------------------------


def test_lazy_client_is_none_before_first_access():
    searcher = VectorSearch(
        code_embedder=make_embedder(768),
        doc_embedder=make_embedder(384),
    )
    assert searcher._client is None


def test_lazy_client_created_on_first_access(monkeypatch):
    fake_client = MagicMock()
    monkeypatch.setattr(
        "reporag.retrieval.vector_search.QdrantClient",
        lambda url: fake_client,
    )
    searcher = VectorSearch(
        code_embedder=make_embedder(768),
        doc_embedder=make_embedder(384),
    )
    assert searcher.client is fake_client
    assert searcher._client is fake_client


def test_lazy_client_not_recreated_on_second_access(monkeypatch):
    call_count = {"n": 0}

    def counting_client(url):
        call_count["n"] += 1
        return MagicMock()

    monkeypatch.setattr("reporag.retrieval.vector_search.QdrantClient", counting_client)
    searcher = VectorSearch(
        code_embedder=make_embedder(768),
        doc_embedder=make_embedder(384),
    )
    _ = searcher.client
    _ = searcher.client
    assert call_count["n"] == 1
