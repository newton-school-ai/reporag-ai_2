from unittest.mock import MagicMock, patch

import pytest

from reporag.agent.executor import RetrievalEngines, SubQueryExecutor
from reporag.agent.planner import DecompositionStep
from reporag.agent.router import RoutingDecision, StrategyRouter
from reporag.retrieval.vector_search import RetrievalResult


@pytest.fixture
def mock_engines():
    engines = RetrievalEngines(vector=MagicMock(), bm25=MagicMock(), graph=MagicMock())
    # Setup some basic returns
    engines.vector.search.return_value = [
        RetrievalResult(1.0, "a.py", 1, 2, "a", "vector text")
    ]
    engines.bm25.search.return_value = [
        RetrievalResult(1.0, "b.py", 1, 2, "b", "bm25 text")
    ]
    engines.graph.get_neighbors.return_value = [
        RetrievalResult(1.0, "c.py", 1, 2, "c", "graph text")
    ]
    return engines


@pytest.fixture
def mock_router():
    router = MagicMock(spec=StrategyRouter)
    router.route.return_value = RoutingDecision("vector")
    return router


def test_topological_sort_execution(mock_engines, mock_router):
    executor = SubQueryExecutor(engines=mock_engines, router=mock_router)
    steps = (
        DecompositionStep(
            id="3", query="q3", expected_answer_type="code", depends_on=("2",)
        ),
        DecompositionStep(
            id="1", query="q1", expected_answer_type="code", depends_on=()
        ),
        DecompositionStep(
            id="2", query="q2", expected_answer_type="code", depends_on=("1",)
        ),
    )
    results = executor.execute(steps)

    # We can check that the calls happened in the correct order: 1, 2, 3
    assert len(results) == 3
    assert "1" in results
    assert "2" in results
    assert "3" in results


def test_context_propagation(mock_engines, mock_router):
    mock_engines.vector.search.return_value = [
        RetrievalResult(1.0, "f.py", 1, 2, "f", "chunk content")
    ]

    executor = SubQueryExecutor(engines=mock_engines, router=mock_router)
    steps = (
        DecompositionStep(
            id="1", query="first", expected_answer_type="code", depends_on=()
        ),
        DecompositionStep(
            id="2", query="second", expected_answer_type="code", depends_on=("1",)
        ),
    )
    executor.execute(steps)

    # Check that the second search call included the context from the first
    search_calls = mock_engines.vector.search.call_args_list
    assert len(search_calls) == 2

    second_call_query = search_calls[1][0][0]
    assert "second" in second_call_query
    assert "chunk content" in second_call_query


def test_backend_exception_retries_once(mock_engines, mock_router):
    # Setup vector to fail once, then succeed
    mock_engines.vector.search.side_effect = [
        RuntimeError("Transient failure"),
        [RetrievalResult(1.0, "f.py", 1, 2, "f", "success text")],
    ]

    executor = SubQueryExecutor(engines=mock_engines, router=mock_router)
    steps = (DecompositionStep(id="1", query="q", expected_answer_type="code"),)
    results = executor.execute(steps)

    assert len(results["1"]) == 1
    assert mock_engines.vector.search.call_count == 2


def test_backend_exception_fails_gracefully(mock_engines, mock_router):
    # Setup vector to fail twice
    mock_engines.vector.search.side_effect = RuntimeError("Persistent failure")

    executor = SubQueryExecutor(engines=mock_engines, router=mock_router)
    steps = (
        DecompositionStep(id="1", query="first", expected_answer_type="code"),
        DecompositionStep(
            id="2", query="second", expected_answer_type="code", depends_on=("1",)
        ),
    )
    results = executor.execute(steps)

    assert results["1"] == []
    # Step 2 should still run, receiving no context
    assert "2" in results
    assert mock_engines.vector.search.call_count == 4  # 2 for step 1, 2 for step 2


def test_empty_result_does_not_retry(mock_engines, mock_router):
    mock_engines.vector.search.return_value = []

    executor = SubQueryExecutor(engines=mock_engines, router=mock_router)
    steps = (DecompositionStep(id="1", query="q", expected_answer_type="code"),)
    results = executor.execute(steps)

    assert results["1"] == []
    assert mock_engines.vector.search.call_count == 1  # No retry


@patch("reporag.retrieval.fusion.reciprocal_rank_fusion")
def test_hybrid_fuses_available_backends(mock_rrf, mock_engines):
    # Return hybrid with symbol
    router = MagicMock(spec=StrategyRouter)
    router.route.return_value = RoutingDecision(strategy="hybrid", symbol="sym")

    executor = SubQueryExecutor(engines=mock_engines, router=router)
    steps = (DecompositionStep(id="1", query="q", expected_answer_type="code"),)
    executor.execute(steps)

    assert mock_engines.vector.search.called
    assert mock_engines.bm25.search.called
    assert mock_engines.graph.get_neighbors.called
    assert mock_rrf.called

    # Verify the lists passed to RRF
    call_args = mock_rrf.call_args[0][0]
    assert len(call_args) == 3


def test_cycle_raises_value_error(mock_engines, mock_router):
    executor = SubQueryExecutor(engines=mock_engines, router=mock_router)
    steps = (
        DecompositionStep(
            id="1", query="q1", expected_answer_type="code", depends_on=("2",)
        ),
        DecompositionStep(
            id="2", query="q2", expected_answer_type="code", depends_on=("1",)
        ),
    )
    with pytest.raises(ValueError, match="Cycle"):
        executor.execute(steps)


@patch("reporag.retrieval.fusion.reciprocal_rank_fusion")
def test_hybrid_failure_resilience(mock_rrf, mock_engines):
    # Setup hybrid with symbol
    router = MagicMock(spec=StrategyRouter)
    router.route.return_value = RoutingDecision(strategy="hybrid", symbol="sym")

    # Vector fails persistently
    mock_engines.vector.search.side_effect = RuntimeError("Persistent vector failure")
    # BM25 succeeds
    mock_engines.bm25.search.return_value = [
        RetrievalResult(1.0, "bm25.py", 1, 2, "b", "bm25 text")
    ]
    # Graph succeeds
    mock_engines.graph.get_neighbors.return_value = [
        RetrievalResult(1.0, "graph.py", 1, 2, "g", "graph text")
    ]

    mock_rrf.return_value = [RetrievalResult(1.0, "bm25.py", 1, 2, "b", "bm25 text")]

    executor = SubQueryExecutor(engines=mock_engines, router=router)
    steps = (DecompositionStep(id="1", query="q", expected_answer_type="code"),)
    results = executor.execute(steps)

    # Vector should be attempted twice
    assert mock_engines.vector.search.call_count == 2
    # BM25 should be attempted once
    assert mock_engines.bm25.search.call_count == 1
    # Graph should be attempted once
    assert mock_engines.graph.get_neighbors.call_count == 1

    # RRF should be called with three lists, but the vector list should be empty []
    call_args = mock_rrf.call_args[0][0]
    assert len(call_args) == 3
    assert call_args[0] == []  # Vector result is empty due to failure
    assert len(call_args[1]) == 1  # BM25 result
    assert len(call_args[2]) == 1  # Graph result

    # Final result is not empty
    assert len(results["1"]) == 1


@patch("reporag.retrieval.fusion.reciprocal_rank_fusion")
def test_graph_fallback_without_symbol(mock_rrf, mock_engines):
    router = MagicMock(spec=StrategyRouter)
    # Router returns graph but no symbol
    router.route.return_value = RoutingDecision(strategy="graph", symbol=None)

    mock_engines.vector.search.return_value = [
        RetrievalResult(1.0, "v.py", 1, 2, "v", "v text")
    ]
    mock_engines.bm25.search.return_value = [
        RetrievalResult(1.0, "b.py", 1, 2, "b", "b text")
    ]
    mock_rrf.return_value = [RetrievalResult(1.0, "v.py", 1, 2, "v", "v text")]

    executor = SubQueryExecutor(engines=mock_engines, router=router)
    steps = (DecompositionStep(id="1", query="q", expected_answer_type="code"),)
    executor.execute(steps)

    # Should fallback to vector and bm25
    assert mock_engines.vector.search.called
    assert mock_engines.bm25.search.called
    assert not mock_engines.graph.get_neighbors.called

    call_args = mock_rrf.call_args[0][0]
    assert len(call_args) == 2
