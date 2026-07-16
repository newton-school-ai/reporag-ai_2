"""Unit tests for the vector semantic search retrieval engine."""

from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from reporag.retrieval.vector_search import VectorSearch

# Define query vector sizes to match database schema
CODE_DIM = 768
DOC_DIM = 384


class MockCodeEmbedder:
    """Mock code embedder returning pre-configured vectors for unit testing."""

    def embed(self, query: str) -> np.ndarray:
        # Determine vector based on search query
        v = np.zeros(CODE_DIM, dtype=np.float32)
        if "auth" in query:
            v[0] = 1.0
        elif "format" in query:
            v[1] = 1.0
        else:
            v[2] = 1.0
        return v


class MockDocEmbedder:
    """Mock doc embedder returning pre-configured vectors for unit testing."""

    def embed(self, query: str) -> np.ndarray:
        v = np.zeros(DOC_DIM, dtype=np.float32)
        if "auth" in query:
            v[0] = 1.0
        elif "format" in query:
            v[1] = 1.0
        else:
            v[2] = 1.0
        return v


@pytest.fixture
def memory_qdrant_client() -> QdrantClient:
    """Provide an in-memory Qdrant client populated with mock code and doc points."""
    client = QdrantClient(location=":memory:")

    # Create collection schemas
    client.create_collection(
        collection_name="reporag_code",
        vectors_config=VectorParams(size=CODE_DIM, distance=Distance.COSINE),
    )
    client.create_collection(
        collection_name="reporag_docs",
        vectors_config=VectorParams(size=DOC_DIM, distance=Distance.COSINE),
    )

    # 1. Insert code chunks
    v1 = np.zeros(CODE_DIM, dtype=np.float32)
    v1[0] = 1.0  # matches "auth"
    p1 = PointStruct(
        id=1,
        vector=v1.tolist(),
        payload={
            "file_path": "src/auth.py",
            "language": "python",
            "start_line": 10,
            "end_line": 20,
            "qualified_name": "auth.login",
            "content": "def login(user, token): pass",
            "symbol_type": "function",
        },
    )

    v2 = np.zeros(CODE_DIM, dtype=np.float32)
    v2[1] = 1.0  # matches "format"
    p2 = PointStruct(
        id=2,
        vector=v2.tolist(),
        payload={
            "file_path": "src/utils.py",
            "language": "python",
            "start_line": 1,
            "end_line": 5,
            "qualified_name": "utils.format_date",
            "content": "def format_date(d): pass",
            "symbol_type": "function",
        },
    )

    v3 = np.zeros(CODE_DIM, dtype=np.float32)
    v3[2] = 1.0
    p3 = PointStruct(
        id=3,
        vector=v3.tolist(),
        payload={
            "file_path": "static/app.js",
            "language": "javascript",
            "start_line": 5,
            "end_line": 15,
            "qualified_name": "render_ui",
            "content": "function render_ui() {}",
            "symbol_type": "function",
        },
    )

    client.upsert(collection_name="reporag_code", points=[p1, p2, p3])

    # 2. Insert doc embeddings
    d1 = np.zeros(DOC_DIM, dtype=np.float32)
    d1[0] = 0.9  # matches "auth" with slightly lower similarity
    dp1 = PointStruct(
        id=4,
        vector=d1.tolist(),
        payload={
            "file_path": "src/auth.py",
            "language": "python",
            "start_line": 10,
            "end_line": 20,
            "symbol_id": "auth.login",
            "text": "Docstring for login function explaining authentication flow.",
            "doc_type": "docstring",
            "metadata": {"symbol_type": "function"},
        },
    )

    d2 = np.zeros(DOC_DIM, dtype=np.float32)
    d2[2] = 1.0
    dp2 = PointStruct(
        id=5,
        vector=d2.tolist(),
        payload={
            "file_path": "static/app.js",
            "language": "javascript",
            "start_line": 5,
            "end_line": 15,
            "symbol_id": "render_ui",
            "text": "Javascript client rendering docs.",
            "doc_type": "docstring",
            "metadata": {"symbol_type": "function"},
        },
    )

    client.upsert(collection_name="reporag_docs", points=[dp1, dp2])

    return client


@pytest.fixture
def vector_searcher(memory_qdrant_client: QdrantClient) -> VectorSearch:
    """Construct VectorSearch pointed at the in-memory Qdrant client."""
    return VectorSearch(
        client=memory_qdrant_client,
        code_embedder=MockCodeEmbedder(),
        doc_embedder=MockDocEmbedder(),
        collection_code="reporag_code",
        collection_docs="reporag_docs",
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_basic_semantic_search(vector_searcher: VectorSearch) -> None:
    """Verify basic semantic search merges code and doc results and sorts them."""
    # Query matching auth points
    results = vector_searcher.search("user auth credentials", top_k=5)

    # Both code point (1) and doc point (4) match the "auth" query vector.
    # Because they share coordinates (src/auth.py, lines 10-20), they must deduplicate,
    # leaving the one with the highest score (code point score 1.0 > doc point score 0.9).
    assert len(results) == 1
    r = results[0]
    assert r.file_path == "src/auth.py"
    assert r.start_line == 10
    assert r.end_line == 20
    assert r.symbol_name == "auth.login"
    assert r.chunk_text == "def login(user, token): pass"
    assert r.score == pytest.approx(1.0)


def test_language_filtering(vector_searcher: VectorSearch) -> None:
    """Verify language filtering is applied correctly."""
    # Query with no matches to test filter correctness
    results_py = vector_searcher.search("render", language="python")
    assert len(results_py) == 0

    results_js = vector_searcher.search("render", language="javascript")
    assert len(results_js) == 1
    assert results_js[0].file_path == "static/app.js"
    assert results_js[0].symbol_name == "render_ui"


def test_glob_file_path_filtering(vector_searcher: VectorSearch) -> None:
    """Verify glob pattern file path matching works correctly."""
    # Glob matching all Python files in src/
    results_src = vector_searcher.search("format", file_path="src/*.py")
    assert len(results_src) == 1
    assert results_src[0].file_path == "src/utils.py"

    # Glob that matches nothing (no Python files in static/)
    results_none = vector_searcher.search("format", file_path="static/*.py")
    assert len(results_none) == 0

    # Glob matching JavaScript files
    results_js = vector_searcher.search("render", file_path="static/*.js")
    assert len(results_js) == 1
    assert results_js[0].file_path == "static/app.js"


def test_symbol_type_filtering(vector_searcher: VectorSearch) -> None:
    """Verify filtering by symbol type works correctly."""
    results_func = vector_searcher.search("auth", symbol_type="function")
    assert len(results_func) == 1
    assert results_func[0].symbol_name == "auth.login"

    results_class = vector_searcher.search("auth", symbol_type="class")
    assert len(results_class) == 0


def test_top_k_limit(vector_searcher: VectorSearch) -> None:
    """Verify top_k constraints are obeyed."""
    # Insert multiple matching mock documents
    client = vector_searcher.client
    for i in range(10):
        v = np.zeros(CODE_DIM, dtype=np.float32)
        v[0] = 1.0  # matches "auth"
        client.upsert(
            collection_name="reporag_code",
            points=[
                PointStruct(
                    id=100 + i,
                    vector=v.tolist(),
                    payload={
                        "file_path": f"src/extra_{i}.py",
                        "language": "python",
                        "start_line": 1,
                        "end_line": 5,
                        "qualified_name": f"extra_{i}",
                        "content": f"extra content {i}",
                        "symbol_type": "function",
                    },
                )
            ],
        )

    results = vector_searcher.search("auth", top_k=3)
    assert len(results) == 3


def test_top_k_zero_raises(vector_searcher: VectorSearch) -> None:
    """Verify that top_k < 1 raises a ValueError immediately."""
    with pytest.raises(ValueError, match="top_k"):
        vector_searcher.search("auth", top_k=0)


def test_both_collections_fail_raises_runtime_error() -> None:
    """Verify RuntimeError is raised when both Qdrant searches fail."""

    class _AlwaysFailClient:
        def search(self, **_kwargs: object) -> None:
            raise ConnectionError("Qdrant unreachable")

    searcher = VectorSearch(
        client=_AlwaysFailClient(),
        code_embedder=MockCodeEmbedder(),
        doc_embedder=MockDocEmbedder(),
        collection_code="reporag_code",
        collection_docs="reporag_docs",
    )
    with pytest.raises(RuntimeError, match="Both vector searches failed"):
        searcher.search("auth", top_k=5)
