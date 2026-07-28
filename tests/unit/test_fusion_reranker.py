"""Unit tests for RRF fusion and Cross-Encoder reranking (Issue 19)."""

from __future__ import annotations

import pytest

from reporag.retrieval.fusion import reciprocal_rank_fusion
from reporag.retrieval.reranker import CrossEncoderReranker
from reporag.retrieval.vector_search import RetrievalResult


class MockCrossEncoderModel:
    """Mock CrossEncoder model that returns predictable scores without network/GPU dependencies."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for query, doc in pairs:
            # Simulates relevancy by returning a higher score if query term is in doc
            if query.lower() in doc.lower():
                scores.append(5.0)
            elif "relevant" in doc.lower():
                scores.append(3.0)
            else:
                scores.append(1.0)
        return scores


@pytest.fixture
def sample_vector_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            score=0.9,
            file_path="src/auth.py",
            start_line=1,
            end_line=10,
            symbol_name="authenticate",
            chunk_text="def authenticate(username, password): pass",
        ),
        RetrievalResult(
            score=0.8,
            file_path="src/api.py",
            start_line=15,
            end_line=25,
            symbol_name="login",
            chunk_text="def login(req): pass",
        ),
        RetrievalResult(
            score=0.7,
            file_path="src/db.py",
            start_line=50,
            end_line=60,
            symbol_name="query",
            chunk_text="def execute_query(sql): pass",
        ),
    ]


@pytest.fixture
def sample_bm25_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            score=12.5,
            file_path="src/db.py",
            start_line=50,
            end_line=60,
            symbol_name="query",
            chunk_text="def execute_query(sql): pass",
        ),
        RetrievalResult(
            score=10.2,
            file_path="src/auth.py",
            start_line=1,
            end_line=10,
            symbol_name="authenticate",
            chunk_text="def authenticate(username, password): pass",
        ),
    ]


@pytest.fixture
def sample_graph_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            score=0.5,
            file_path="src/api.py",
            start_line=15,
            end_line=25,
            symbol_name="login",
            chunk_text="def login(req): pass",
        ),
        RetrievalResult(
            score=0.33,
            file_path="src/auth.py",
            start_line=1,
            end_line=10,
            symbol_name="authenticate",
            chunk_text="def authenticate(username, password): pass",
        ),
    ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF) Tests
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    """Verify that reciprocal rank fusion merges lists correctly based on RRF scores."""

    def test_rrf_correctly_fuses_results(
        self,
        sample_vector_results: list[RetrievalResult],
        sample_bm25_results: list[RetrievalResult],
        sample_graph_results: list[RetrievalResult],
    ) -> None:
        """With default k=60, verify the fusion math:

        - 'authenticate' (auth.py:1-10):
          - Rank in Vector: 1
          - Rank in BM25: 2
          - Rank in Graph: 2
          - Score = 1/(60+1) + 1/(60+2) + 1/(60+2) = 1/61 + 2/62 = 0.01639 + 0.032258 = 0.04865
        - 'login' (api.py:15-25):
          - Rank in Vector: 2
          - Rank in BM25: Not present (0)
          - Rank in Graph: 1
          - Score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.01639 = 0.03252
        - 'query' (db.py:50-60):
          - Rank in Vector: 3
          - Rank in BM25: 1
          - Rank in Graph: Not present (0)
          - Score = 1/(60+3) + 1/(60+1) = 1/63 + 1/61 = 0.015873 + 0.016393 = 0.032266
        """
        fused = reciprocal_rank_fusion(
            [sample_vector_results, sample_bm25_results, sample_graph_results], k=60
        )
        assert len(fused) == 3

        # 'authenticate' has highest RRF score and should be top-1
        assert fused[0].symbol_name == "authenticate"
        assert abs(fused[0].score - 0.04865) < 1e-4

        # 'login' should be top-2
        assert fused[1].symbol_name == "login"
        assert abs(fused[1].score - 0.03252) < 1e-4

        # 'query' should be top-3
        assert fused[2].symbol_name == "query"
        assert abs(fused[2].score - 0.03227) < 1e-4

    def test_rrf_with_custom_k(
        self,
        sample_vector_results: list[RetrievalResult],
        sample_bm25_results: list[RetrievalResult],
    ) -> None:
        """Verify that RRF supports custom values for parameter k."""
        fused = reciprocal_rank_fusion(
            [sample_vector_results, sample_bm25_results], k=10
        )
        # 'authenticate' (Vector rank 1, BM25 rank 2): 1/(10+1) + 1/(10+2) = 1/11 + 1/12 = 0.0909 + 0.0833 = 0.1742
        # 'query' (Vector rank 3, BM25 rank 1): 1/(10+3) + 1/(10+1) = 1/13 + 1/11 = 0.0769 + 0.0909 = 0.1678
        assert fused[0].symbol_name == "authenticate"
        assert abs(fused[0].score - 0.1742) < 1e-4

    def test_rrf_invalid_k_raises_error(
        self, sample_vector_results: list[RetrievalResult]
    ) -> None:
        """ValueError is raised when k is less than 1."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            reciprocal_rank_fusion([sample_vector_results], k=0)

    def test_rrf_empty_input_returns_empty(self) -> None:
        """Passing an empty list of ranked lists returns an empty list immediately."""
        assert reciprocal_rank_fusion([]) == []


# ---------------------------------------------------------------------------
# Cross-Encoder Reranker Tests
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    """Verify that CrossEncoderReranker reranks candidates correctly using cross-encoders."""

    def test_reranker_reorders_by_relevancy(self) -> None:
        """Check that CrossEncoderReranker scores and reorders candidates based on the model output."""
        candidates = [
            RetrievalResult(
                score=0.9,
                file_path="src/unrelated.py",
                start_line=1,
                end_line=2,
                symbol_name="unrelated_func",
                chunk_text="def placeholder(): pass",
            ),
            RetrievalResult(
                score=0.5,
                file_path="src/auth.py",
                start_line=1,
                end_line=10,
                symbol_name="authenticate",
                chunk_text="def authenticate(username, password): pass",
            ),
        ]

        mock_model = MockCrossEncoderModel()
        reranker = CrossEncoderReranker(model_instance=mock_model)

        # Query matches "authenticate" text, so it should score 5.0 and be top-1,
        # despite initially starting with a lower score (0.5).
        reranked = reranker.rerank(query="authenticate", candidates=candidates)

        assert len(reranked) == 2
        assert reranked[0].symbol_name == "authenticate"
        assert reranked[0].rerank_score == 5.0
        assert reranked[0].score == 5.0  # primary score field is also updated

        assert reranked[1].symbol_name == "unrelated_func"
        assert reranked[1].rerank_score == 1.0

    def test_reranker_empty_candidates_returns_empty(self) -> None:
        """Reranker returns empty list if candidate list is empty."""
        mock_model = MockCrossEncoderModel()
        reranker = CrossEncoderReranker(model_instance=mock_model)
        assert reranker.rerank("query", []) == []


# ---------------------------------------------------------------------------
# End-to-End Integration
# ---------------------------------------------------------------------------


class TestEndToEndRetrievalPipeline:
    """Verify the combined RRF fusion and reranking workflow."""

    def test_end_to_end_fusion_and_rerank(
        self,
        sample_vector_results: list[RetrievalResult],
        sample_bm25_results: list[RetrievalResult],
        sample_graph_results: list[RetrievalResult],
    ) -> None:
        # 1. Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            [sample_vector_results, sample_bm25_results, sample_graph_results], k=60
        )
        assert len(fused) == 3
        # Initially: ['authenticate', 'login', 'query']

        # 2. Rerank top candidate subset (e.g. top 2) relative to query "login"
        mock_model = MockCrossEncoderModel()
        reranker = CrossEncoderReranker(model_instance=mock_model)

        reranked = reranker.rerank(query="login", candidates=fused[:2])
        # 'login' contains query word, so it should jump to top-1 with score 5.0
        assert len(reranked) == 2
        assert reranked[0].symbol_name == "login"
        assert reranked[0].score == 5.0
        assert reranked[0].rerank_score == 5.0

        assert reranked[1].symbol_name == "authenticate"
        assert reranked[1].score == 1.0


# ======================================================================
# Semantic acceptance test (real model; skips when unavailable)
# ======================================================================


@pytest.fixture(scope="module")
def real_reranker() -> CrossEncoderReranker:
    """Load the real Cross-Encoder model, skipping the test if unavailable."""
    reranker = CrossEncoderReranker(device="cpu")
    try:
        # Warmup the model with a tiny predict
        reranker.rerank(
            "warmup",
            [
                RetrievalResult(
                    score=0.5,
                    file_path="src/warmup.py",
                    start_line=1,
                    end_line=2,
                    symbol_name="warmup",
                    chunk_text="def warmup(): pass",
                )
            ],
        )
    except Exception as exc:  # noqa: BLE001 -- offline / download failure -> skip
        pytest.skip(f"CrossEncoder model unavailable (offline?): {exc}")
    return reranker


def test_authentication_query_ranks_jwt_first(
    real_reranker: CrossEncoderReranker,
) -> None:
    """AC: Reranked results measurably outperform RRF-only.

    Asserts relative ordering and latency under 500ms for candidates.
    """
    import time

    candidates = [
        RetrievalResult(
            score=0.9,
            file_path="src/csv_parser.py",
            start_line=1,
            end_line=20,
            symbol_name="parse_csv",
            chunk_text="def parse_csv(filepath): pass\n# Parse a CSV file into a list of rows.",
        ),
        RetrievalResult(
            score=0.5,
            file_path="src/auth.py",
            start_line=5,
            end_line=15,
            symbol_name="verify_jwt",
            chunk_text="def verify_jwt(token): pass\n# Verify the JWT token and return the authenticated user.",
        ),
    ]

    t0 = time.time()
    reranked = real_reranker.rerank(
        query="how does authentication work", candidates=candidates
    )
    latency = time.time() - t0

    # The verify_jwt candidate should rank first after rerank
    assert len(reranked) == 2
    assert reranked[0].symbol_name == "verify_jwt"
    assert reranked[0].rerank_score is not None
    assert reranked[1].rerank_score is not None
    assert reranked[0].rerank_score > reranked[1].rerank_score

    # Reranking latency under 500ms
    assert latency < 0.5
