"""Unit tests for the vector semantic search retrieval engine.

Payload schema follows the Issue 15 / HybridIndexBuilder canonical layout:
  Code collection: file_path, language, start_line, end_line,
                   symbol (primary name key), symbol_type, qualified_name,
                   parent_symbol, chunk_kind, content
  Doc  collection: file_path, language, start_line, end_line,
                   symbol_id, doc_type, text,
                   metadata: {symbol_type: ...}
"""

from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from reporag.retrieval.vector_search import VectorSearch

# Define query vector sizes to match database schema
CODE_DIM = 768
DOC_DIM = 384


# ---------------------------------------------------------------------------
# Mock embedders
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers: canonical payload builders
# ---------------------------------------------------------------------------


def _code_point(
    point_id: int,
    vector: np.ndarray,
    *,
    file_path: str,
    language: str,
    start_line: int,
    end_line: int,
    symbol: str,
    symbol_type: str,
    content: str,
    qualified_name: str | None = None,
) -> PointStruct:
    """Build a code-collection PointStruct matching the Issue 15 payload schema."""
    return PointStruct(
        id=point_id,
        vector=vector.tolist(),
        payload={
            "file_path": file_path,
            "language": language,
            "start_line": start_line,
            "end_line": end_line,
            # `symbol` is the primary indexed key (listed in _KEYWORD_INDEX_FIELDS).
            # `qualified_name` is also stored as a secondary reference field.
            "symbol": symbol,
            "qualified_name": qualified_name or symbol,
            "parent_symbol": None,
            "chunk_kind": "definition",
            "symbol_type": symbol_type,
            "content": content,
        },
    )


def _doc_point(
    point_id: int,
    vector: np.ndarray,
    *,
    file_path: str,
    language: str,
    start_line: int,
    end_line: int,
    symbol_id: str,
    text: str,
    symbol_type: str,
) -> PointStruct:
    """Build a doc-collection PointStruct matching the Issue 15 payload schema."""
    return PointStruct(
        id=point_id,
        vector=vector.tolist(),
        payload={
            "file_path": file_path,
            "language": language,
            "start_line": start_line,
            "end_line": end_line,
            "symbol_id": symbol_id,
            "doc_type": "docstring",
            "text": text,
            # DocEmbedder stores symbol_type inside a nested metadata sub-dict
            "metadata": {"symbol_type": symbol_type},
        },
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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

    # ---- Code points --------------------------------------------------------
    v_auth = np.zeros(CODE_DIM, dtype=np.float32)
    v_auth[0] = 1.0  # matches "auth"

    v_format = np.zeros(CODE_DIM, dtype=np.float32)
    v_format[1] = 1.0  # matches "format"

    v_render = np.zeros(CODE_DIM, dtype=np.float32)
    v_render[2] = 1.0  # matches generic / "render"

    client.upsert(
        collection_name="reporag_code",
        points=[
            _code_point(
                1,
                v_auth,
                file_path="src/auth.py",
                language="python",
                start_line=10,
                end_line=20,
                symbol="auth.login",
                symbol_type="function",
                content="def login(user, token): pass",
            ),
            _code_point(
                2,
                v_format,
                file_path="src/utils.py",
                language="python",
                start_line=1,
                end_line=5,
                symbol="utils.format_date",
                symbol_type="function",
                content="def format_date(d): pass",
            ),
            _code_point(
                3,
                v_render,
                file_path="static/app.js",
                language="javascript",
                start_line=5,
                end_line=15,
                symbol="render_ui",
                symbol_type="function",
                content="function render_ui() {}",
            ),
        ],
    )

    # ---- Doc points ---------------------------------------------------------
    d_auth = np.zeros(DOC_DIM, dtype=np.float32)
    d_auth[0] = 0.9  # matches "auth" but slightly lower similarity than code

    d_render = np.zeros(DOC_DIM, dtype=np.float32)
    d_render[2] = 1.0

    client.upsert(
        collection_name="reporag_docs",
        points=[
            _doc_point(
                4,
                d_auth,
                file_path="src/auth.py",
                language="python",
                start_line=10,
                end_line=20,
                symbol_id="auth.login",
                text="Docstring for login function explaining authentication flow.",
                symbol_type="function",
            ),
            _doc_point(
                5,
                d_render,
                file_path="static/app.js",
                language="javascript",
                start_line=5,
                end_line=15,
                symbol_id="render_ui",
                text="Javascript client rendering docs.",
                symbol_type="function",
            ),
        ],
    )

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


class TestPayloadFieldAlignment:
    """Verify that retrieval results respect the Issue 15 canonical payload schema."""

    def test_code_result_uses_symbol_key(self, vector_searcher: VectorSearch) -> None:
        """RetrievalResult.symbol_name is resolved from the canonical `symbol` key."""
        results = vector_searcher.search("user auth credentials", top_k=5)
        # After deduplication (code+doc share the same span), 1 result expected.
        assert len(results) == 1
        r = results[0]
        assert r.file_path == "src/auth.py"
        assert r.start_line == 10
        assert r.end_line == 20
        # symbol_name must resolve from `symbol` (primary key) or fall back to
        # `qualified_name`; both are stored as "auth.login" in this fixture.
        assert r.symbol_name == "auth.login"
        assert r.chunk_text == "def login(user, token): pass"
        assert r.score == pytest.approx(1.0)

    def test_doc_result_symbol_id_field(self, vector_searcher: VectorSearch) -> None:
        """Doc results expose symbol_name from the `symbol_id` payload key."""
        # Query that matches only the render doc (code result also matches, same span)
        results = vector_searcher.search("render", top_k=5)
        assert any(r.symbol_name == "render_ui" for r in results)

    def test_code_result_content_field(self, vector_searcher: VectorSearch) -> None:
        """chunk_text is populated from the canonical `content` payload field."""
        results = vector_searcher.search("format", top_k=5)
        assert len(results) >= 1
        r = results[0]
        assert r.chunk_text == "def format_date(d): pass"

    def test_result_metadata_contains_full_payload(
        self, vector_searcher: VectorSearch
    ) -> None:
        """metadata on a code result contains all Issue 15 payload fields."""
        results = vector_searcher.search("format", top_k=5)
        assert results
        meta = results[0].metadata
        for key in (
            "file_path",
            "start_line",
            "end_line",
            "symbol",
            "symbol_type",
            "content",
        ):
            assert key in meta, f"Missing payload key: {key!r}"


class TestSymbolTypeFilter:
    """Verify that symbol_type filtering applies to both code AND doc results."""

    def test_function_type_returns_result(self, vector_searcher: VectorSearch) -> None:
        results = vector_searcher.search("auth", symbol_type="function")
        assert len(results) == 1
        assert results[0].symbol_name == "auth.login"

    def test_class_type_returns_empty_for_code_and_docs(
        self, vector_searcher: VectorSearch
    ) -> None:
        """When symbol_type='class', neither code nor doc results should leak through.

        This is the core regression: previously doc results bypassed the filter
        because the in-memory Qdrant client ignores nested-key conditions.
        The Python-side guard in vector_search.py must catch them.
        """
        results = vector_searcher.search("auth", symbol_type="class")
        assert (
            results == []
        ), "Doc results with symbol_type='function' must not bypass a 'class' filter"

    def test_symbol_type_filter_does_not_drop_matching_docs(
        self, vector_searcher: VectorSearch
    ) -> None:
        """Docs that DO match the symbol_type should survive the filter."""
        # render_ui code+doc both carry symbol_type="function"
        results = vector_searcher.search("render", symbol_type="function")
        assert len(results) >= 1
        assert any(r.file_path == "static/app.js" for r in results)


class TestDeduplication:
    """Verify merge/deduplication logic keeps the highest-scoring result."""

    def test_code_beats_doc_on_same_span(self, vector_searcher: VectorSearch) -> None:
        """When code (score 1.0) and doc (score 0.9) share a span, code wins."""
        results = vector_searcher.search("user auth credentials", top_k=10)
        # Code point 1 and doc point 4 share (src/auth.py, 10, 20).
        # Only one result should survive, and it should carry score 1.0.
        matching = [r for r in results if r.file_path == "src/auth.py"]
        assert len(matching) == 1
        assert matching[0].score == pytest.approx(1.0)
        # The winning entry is the code chunk, so chunk_text comes from `content`.
        assert matching[0].chunk_text == "def login(user, token): pass"

    def test_different_spans_not_deduplicated(
        self, vector_searcher: VectorSearch
    ) -> None:
        """Results with different (file, start, end) tuples are kept separately."""
        # Query matches both the auth code point and the render code+doc point
        # via dim-0 and dim-2 respectively; only dim-2 is active here for "render"
        results = vector_searcher.search("render", top_k=10)
        file_paths = {r.file_path for r in results}
        # render point is at static/app.js; auth point is at src/auth.py (no match)
        assert "static/app.js" in file_paths

    def test_higher_score_wins_on_tie_span(
        self, memory_qdrant_client: QdrantClient
    ) -> None:
        """If two points share a span, only the one with the higher score is kept."""
        # Add a second code point at the same (file, start, end) as point 1 but
        # with a weaker vector so it scores lower.
        v_weak = np.zeros(CODE_DIM, dtype=np.float32)
        v_weak[0] = 0.5  # same direction as auth but half magnitude
        v_weak /= np.linalg.norm(v_weak)
        memory_qdrant_client.upsert(
            collection_name="reporag_code",
            points=[
                _code_point(
                    200,
                    v_weak,
                    file_path="src/auth.py",
                    language="python",
                    start_line=10,
                    end_line=20,
                    symbol="auth.login_alt",
                    symbol_type="function",
                    content="# alternate login",
                )
            ],
        )
        searcher = VectorSearch(
            client=memory_qdrant_client,
            code_embedder=MockCodeEmbedder(),
            doc_embedder=MockDocEmbedder(),
            collection_code="reporag_code",
            collection_docs="reporag_docs",
        )
        results = searcher.search("user auth credentials", top_k=10)
        auth_results = [r for r in results if r.file_path == "src/auth.py"]
        # Only 1 result for that span; score must be the higher of the two.
        assert len(auth_results) == 1
        assert auth_results[0].score > 0.9


class TestLanguageFilter:
    """Verify language filtering is applied correctly."""

    def test_python_only_excludes_js(self, vector_searcher: VectorSearch) -> None:
        results = vector_searcher.search("render", language="python")
        assert all(r.file_path != "static/app.js" for r in results)

    def test_javascript_only(self, vector_searcher: VectorSearch) -> None:
        results = vector_searcher.search("render", language="javascript")
        assert len(results) == 1
        assert results[0].file_path == "static/app.js"


class TestGlobFilePathFilter:
    """Verify glob pattern file path matching works for both collections."""

    def test_glob_matches_python_src(self, vector_searcher: VectorSearch) -> None:
        results = vector_searcher.search("format", file_path="src/*.py")
        assert len(results) == 1
        assert results[0].file_path == "src/utils.py"

    def test_glob_matches_nothing(self, vector_searcher: VectorSearch) -> None:
        results = vector_searcher.search("format", file_path="static/*.py")
        assert len(results) == 0

    def test_glob_filters_doc_results_too(self, vector_searcher: VectorSearch) -> None:
        """Glob filter must also exclude doc results that don't match the pattern."""
        # render query matches both code point 3 and doc point 5 at static/app.js
        # A src/*.py glob should exclude both, leaving 0 results.
        results = vector_searcher.search("render", file_path="src/*.py")
        assert len(results) == 0

    def test_glob_js(self, vector_searcher: VectorSearch) -> None:
        results = vector_searcher.search("render", file_path="static/*.js")
        assert len(results) == 1
        assert results[0].file_path == "static/app.js"


class TestTopKLimit:
    """Verify top_k constraints are obeyed."""

    def test_top_k_caps_results(self, memory_qdrant_client: QdrantClient) -> None:
        for i in range(10):
            v = np.zeros(CODE_DIM, dtype=np.float32)
            v[0] = 1.0  # matches "auth"
            memory_qdrant_client.upsert(
                collection_name="reporag_code",
                points=[
                    _code_point(
                        100 + i,
                        v,
                        file_path=f"src/extra_{i}.py",
                        language="python",
                        start_line=1,
                        end_line=5,
                        symbol=f"extra_{i}",
                        symbol_type="function",
                        content=f"extra content {i}",
                    )
                ],
            )
        searcher = VectorSearch(
            client=memory_qdrant_client,
            code_embedder=MockCodeEmbedder(),
            doc_embedder=MockDocEmbedder(),
            collection_code="reporag_code",
            collection_docs="reporag_docs",
        )
        results = searcher.search("auth", top_k=3)
        assert len(results) == 3

    def test_top_k_zero_raises(self, vector_searcher: VectorSearch) -> None:
        with pytest.raises(ValueError, match="top_k"):
            vector_searcher.search("auth", top_k=0)


class TestFailureCases:
    """Verify graceful degradation and error propagation."""

    def test_both_collections_fail_raises_runtime_error(self) -> None:
        """RuntimeError is raised when both Qdrant searches fail."""

        class _AlwaysFailClient:
            # Match the keyword-only call signature used by _qdrant_search so
            # this fake is an honest duck-type of the real QdrantClient.
            def search(
                self,
                *,
                collection_name: str,
                query_vector: list,
                query_filter: object = None,
                limit: int = 10,
                score_threshold: float | None = None,
            ) -> None:
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

    def test_code_collection_fails_returns_doc_results(
        self, memory_qdrant_client: QdrantClient
    ) -> None:
        """When code search fails, doc results are returned without RuntimeError."""

        class _CodeFailClient:
            """Fails code searches; delegates doc searches to real client."""

            def __init__(self, real: QdrantClient) -> None:
                self._real = real

            def search(self, *, collection_name: str, **kwargs: object) -> list:
                if collection_name == "reporag_code":
                    raise ConnectionError("code collection down")
                return self._real.search(collection_name=collection_name, **kwargs)

        searcher = VectorSearch(
            client=_CodeFailClient(memory_qdrant_client),
            code_embedder=MockCodeEmbedder(),
            doc_embedder=MockDocEmbedder(),
            collection_code="reporag_code",
            collection_docs="reporag_docs",
        )
        results = searcher.search("auth", top_k=5)
        # Doc point 4 matches "auth" and should be returned
        assert len(results) >= 1
        assert results[0].file_path == "src/auth.py"

    def test_doc_collection_fails_returns_code_results(
        self, memory_qdrant_client: QdrantClient
    ) -> None:
        """When doc search fails, code results are returned without RuntimeError."""

        class _DocFailClient:
            """Fails doc searches; delegates code searches to real client."""

            def __init__(self, real: QdrantClient) -> None:
                self._real = real

            def search(self, *, collection_name: str, **kwargs: object) -> list:
                if collection_name == "reporag_docs":
                    raise ConnectionError("doc collection down")
                return self._real.search(collection_name=collection_name, **kwargs)

        searcher = VectorSearch(
            client=_DocFailClient(memory_qdrant_client),
            code_embedder=MockCodeEmbedder(),
            doc_embedder=MockDocEmbedder(),
            collection_code="reporag_code",
            collection_docs="reporag_docs",
        )
        results = searcher.search("auth", top_k=5)
        assert len(results) >= 1
        assert any(r.chunk_text == "def login(user, token): pass" for r in results)


class TestSafetyEdgeCases:
    """Regression tests for edge-case safety guards."""

    def test_score_threshold_excludes_zero_score_results(
        self, memory_qdrant_client: QdrantClient
    ) -> None:
        """Results with score exactly 0.0 are excluded by the default threshold.

        The default score_threshold=0.0 uses strict ``>`` (not ``>=``) to
        exclude orthogonal vectors. This test pins that behaviour so a future
        change from ``>`` to ``>=`` is immediately caught.
        """
        searcher = VectorSearch(
            client=memory_qdrant_client,
            code_embedder=MockCodeEmbedder(),
            doc_embedder=MockDocEmbedder(),
            collection_code="reporag_code",
            collection_docs="reporag_docs",
        )
        # "user auth credentials" -> auth vector -> dim-0=1.0.
        # Code point 2 (format_date, dim-1=1.0) is orthogonal -> score 0.0.
        # With the default score_threshold=0.0, it must be excluded.
        results = searcher.search(
            "user auth credentials", top_k=10, score_threshold=0.0
        )
        file_paths = {r.file_path for r in results}
        assert (
            "src/utils.py" not in file_paths
        ), "Orthogonal results (score=0.0) must be excluded by score_threshold=0.0"
        # The matching auth result (score=1.0) must survive.
        assert "src/auth.py" in file_paths

    def test_score_threshold_is_inclusive_for_custom_thresholds(
        self, memory_qdrant_client: QdrantClient
    ) -> None:
        """Custom non-zero score thresholds are inclusive (>=).

        A result with score exactly equal to a custom non-zero threshold must
        survive, whereas scores below it are dropped.
        """
        import numpy as np

        # Insert a specific code point with a known custom vector.
        # Query will have v[3] = 1.0.
        # Point has v[3] = 0.5, v[4] = 0.8660254 (norm = 1.0).
        # Cosine similarity is exactly 0.5.
        v = np.zeros(CODE_DIM, dtype=np.float32)
        v[3] = 0.5
        v[4] = 0.8660254
        memory_qdrant_client.upsert(
            collection_name="reporag_code",
            points=[
                _code_point(
                    888,
                    v,
                    file_path="src/custom.py",
                    language="python",
                    start_line=1,
                    end_line=2,
                    symbol="custom_symbol",
                    symbol_type="function",
                    content="custom content",
                )
            ],
        )

        class CustomEmbedder:
            def embed(self, query: str) -> np.ndarray:
                qv = np.zeros(CODE_DIM, dtype=np.float32)
                qv[3] = 1.0
                return qv

        searcher = VectorSearch(
            client=memory_qdrant_client,
            code_embedder=CustomEmbedder(),
            doc_embedder=MockDocEmbedder(),
            collection_code="reporag_code",
            collection_docs="reporag_docs",
        )

        # It must survive when threshold is exactly 0.5 (inclusive).
        results = searcher.search("custom_query", top_k=10, score_threshold=0.5)
        scores = {r.score for r in results}
        assert any(
            abs(s - 0.5) < 1e-4 for s in scores
        ), f"Result with score exactly 0.5 must survive threshold of 0.5. Got: {scores}"

        # It must be dropped when threshold is 0.51.
        results_high = searcher.search("custom_query", top_k=10, score_threshold=0.51)
        scores_high = {r.score for r in results_high}
        assert not any(
            abs(s - 0.5) < 1e-4 for s in scores_high
        ), "Result with score 0.5 must be excluded when threshold is 0.51"

    def test_malformed_doc_metadata_does_not_raise(
        self, memory_qdrant_client: QdrantClient
    ) -> None:
        """symbol_type filter must not crash when doc metadata sub-value is not a dict.

        Regression guard for the ``.get()`` on a non-dict AttributeError:
        if a doc point has ``metadata: null`` or ``metadata: "string"``,
        the Python-side filter must silently exclude it rather than raise.
        """
        import numpy as np
        from qdrant_client.models import PointStruct

        # Insert a doc point whose ``metadata`` field is None (malformed)
        v = np.zeros(DOC_DIM, dtype=np.float32)
        v[0] = 1.0
        memory_qdrant_client.upsert(
            collection_name="reporag_docs",
            points=[
                PointStruct(
                    id=999,
                    vector=v.tolist(),
                    payload={
                        "file_path": "src/broken.py",
                        "language": "python",
                        "start_line": 1,
                        "end_line": 2,
                        "symbol_id": "broken.func",
                        "doc_type": "docstring",
                        "text": "Malformed metadata doc.",
                        "metadata": None,  # <-- not a dict
                    },
                )
            ],
        )
        searcher = VectorSearch(
            client=memory_qdrant_client,
            code_embedder=MockCodeEmbedder(),
            doc_embedder=MockDocEmbedder(),
            collection_code="reporag_code",
            collection_docs="reporag_docs",
        )
        # Must not raise AttributeError; malformed doc should be silently dropped
        results = searcher.search(
            "user auth credentials", top_k=10, symbol_type="function"
        )
        broken = [r for r in results if r.file_path == "src/broken.py"]
        assert (
            broken == []
        ), "Malformed doc metadata must be silently excluded, not crash"
