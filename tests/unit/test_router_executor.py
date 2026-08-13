"""Unit tests for StrategyRouter and SubQueryExecutor (Issue 22)."""

from __future__ import annotations

from src.reporag.agent.executor import (
    SubQueryExecutor,
    extract_symbol_from_query,
    topological_sort,
)
from src.reporag.agent.planner import DecompositionStep
from src.reporag.agent.router import StrategyRouter
from src.reporag.retrieval.vector_search import RetrievalResult

# ===========================================================================
# StrategyRouter Tests
# ===========================================================================


def test_router_rule_based_routing_queries() -> None:
    """Verify rule-based routing for 10+ representative sub-queries."""
    router = StrategyRouter(use_llm=False)

    # 1. Structural / Graph routing
    assert router.route("What functions call authenticate_user?") == "graph"
    assert router.route("Find all callers of verify_token") == "graph"
    assert router.route("Who calls the DB initialization?") == "graph"
    assert router.route("What is the dependencies of this module?") == "graph"
    assert router.route("Where is helper_func used?") == "graph"

    # 2. Identifier / BM25 routing
    assert router.route("Find function authenticate_user") == "bm25"
    assert router.route("definition of get_user_session") == "bm25"
    assert router.route("where is verify_token defined?") == "bm25"
    assert router.route("lookup auth_service") == "bm25"
    assert router.route("declaration of AuthController") == "bm25"

    # 3. Semantic / Vector routing
    assert router.route("Explain how the oauth flow works") == "vector"
    assert router.route("What does the auth middleware do?") == "vector"
    assert router.route("Understand the purpose of session cache") == "vector"
    assert router.route("Describe the user registration flow") == "vector"

    # 4. Ambiguous / Hybrid routing fallback
    assert router.route("general search query") == "hybrid"
    assert router.route("") == "hybrid"


def test_router_llm_routing_success() -> None:
    """Verify LLM-assisted strategy routing returns correct strategies."""

    def mock_llm(prompt: str) -> str:
        lines = prompt.strip().split("\n")
        query_line = [line for line in lines if line.startswith("Query: ")][-1]
        if "what calls" in query_line.lower():
            return "graph"
        elif "explain" in query_line.lower():
            return "vector"
        return "bm25"

    router = StrategyRouter(llm=mock_llm, use_llm=True)
    assert router.route("what calls authenticate") == "graph"
    assert router.route("explain auth") == "vector"
    assert router.route("find authenticate") == "bm25"


def test_router_llm_routing_fallback_on_failure() -> None:
    """Verify routing falls back to rule-based classification if LLM fails."""

    def failing_llm(prompt: str) -> str:
        raise RuntimeError("API Timeout")

    router = StrategyRouter(llm=failing_llm, use_llm=True)
    # Rules should categorize "what calls" as graph
    assert router.route("what calls authenticate") == "graph"


# ===========================================================================
# SubQueryExecutor Tests
# ===========================================================================


class MockRetriever:
    """Mock retriever matching Search API patterns."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count = 0
        self.last_query = ""
        self.fail_count = 0

    def search(self, query: str) -> list[RetrievalResult]:
        self.call_count += 1
        self.last_query = query
        if self.fail_count > 0:
            self.fail_count -= 1
            raise RuntimeError(f"Mock failure in {self.name}")
        return [
            RetrievalResult(
                score=0.9,
                file_path="src/auth.py",
                start_line=10,
                end_line=20,
                symbol_name="authenticate",
                chunk_text="def authenticate(): pass",
            )
        ]

    def get_callers(self, symbol: str) -> list[RetrievalResult]:
        self.call_count += 1
        self.last_query = symbol
        if self.fail_count > 0:
            self.fail_count -= 1
            raise RuntimeError(f"Mock failure in {self.name} callers")
        return [
            RetrievalResult(
                score=0.9,
                file_path="src/handler.py",
                start_line=5,
                end_line=15,
                symbol_name="login_handler",
                chunk_text="def login_handler(): authenticate()",
            )
        ]

    def get_neighbors(
        self, symbol: str, direction: str = "both"
    ) -> list[RetrievalResult]:
        self.call_count += 1
        self.last_query = symbol
        return []


def test_topological_sort_no_dependencies() -> None:
    """Verify topological sorting of independent steps is deterministic."""
    step1 = DecompositionStep(id="step-1", query="query 1", expected_answer_type="code")
    step2 = DecompositionStep(id="step-2", query="query 2", expected_answer_type="code")
    sorted_steps = topological_sort((step2, step1))
    # Kahn's sort uses alphabetical step order or original order since degree is 0
    assert [s.id for s in sorted_steps] == ["step-1", "step-2"]


def test_topological_sort_with_dependencies() -> None:
    """Verify topological sorting of dependent steps maintains order."""
    step1 = DecompositionStep(id="step-1", query="query 1", expected_answer_type="code")
    step2 = DecompositionStep(
        id="step-2",
        query="query 2",
        expected_answer_type="code",
        depends_on=("step-3",),
    )
    step3 = DecompositionStep(
        id="step-3",
        query="query 3",
        expected_answer_type="code",
        depends_on=("step-1",),
    )

    sorted_steps = topological_sort((step2, step3, step1))
    assert [s.id for s in sorted_steps] == ["step-1", "step-3", "step-2"]


def test_topological_sort_cycle_fallback() -> None:
    """Verify cycle detection falls back to the default step ordering."""
    step1 = DecompositionStep(
        id="step-1",
        query="query 1",
        expected_answer_type="code",
        depends_on=("step-2",),
    )
    step2 = DecompositionStep(
        id="step-2",
        query="query 2",
        expected_answer_type="code",
        depends_on=("step-1",),
    )

    sorted_steps = topological_sort((step1, step2))
    assert [s.id for s in sorted_steps] == ["step-1", "step-2"]


def test_symbol_extraction() -> None:
    """Verify symbol extraction parses quotes and words successfully."""
    assert (
        extract_symbol_from_query("find the function `authenticate`") == "authenticate"
    )
    assert extract_symbol_from_query("who calls 'verify_token'") == "verify_token"
    assert extract_symbol_from_query("what calls AuthController") == "AuthController"
    assert (
        extract_symbol_from_query("where is auth_service.py used") == "auth_service.py"
    )


def test_executor_basic_flow() -> None:
    """Verify routing and execution for direct retrievers."""
    vector_ret = MockRetriever("vector")
    bm25_ret = MockRetriever("bm25")
    graph_ret = MockRetriever("graph")

    engine = {
        "vector": vector_ret,
        "bm25": bm25_ret,
        "graph": graph_ret,
    }

    executor = SubQueryExecutor(
        retrieval_engine=engine, router=StrategyRouter(use_llm=False)
    )

    step1 = DecompositionStep(
        id="step-1", query="Find function authenticate", expected_answer_type="code"
    )
    step2 = DecompositionStep(
        id="step-2",
        query="What calls step-1?",
        expected_answer_type="code",
        depends_on=("step-1",),
    )

    results = executor.execute([step2, step1])

    # Verify execution order: step-1 then step-2
    # step1 is routed to bm25
    assert bm25_ret.call_count == 1
    assert bm25_ret.last_query == "Find function authenticate"

    # step2 is routed to graph, with context from step-1 injected
    assert graph_ret.call_count == 1
    assert graph_ret.last_query == "authenticate"  # Extracted from step-1 results

    assert "step-1" in results
    assert "step-2" in results
    assert len(results["step-1"]) == 1
    assert len(results["step-2"]) == 1


def test_executor_retry_on_failure() -> None:
    """Verify that a single failure is retried once, and double failure skips."""
    vector_ret = MockRetriever("vector")
    engine = {"vector": vector_ret}

    executor = SubQueryExecutor(
        retrieval_engine=engine, router=StrategyRouter(use_llm=False)
    )
    step = DecompositionStep(
        id="step-1",
        query="Explain authentication mechanism",
        expected_answer_type="explanation",
    )

    # 1. Single failure -> retried once and succeeds
    vector_ret.fail_count = 1
    results = executor.execute([step])
    assert len(results["step-1"]) == 1
    assert vector_ret.call_count == 2

    # 2. Double failure -> skips and returns empty
    vector_ret.call_count = 0
    vector_ret.fail_count = 2
    results = executor.execute([step])
    assert len(results["step-1"]) == 0
    assert vector_ret.call_count == 2  # Max 2 attempts total (first + 1 retry)


def test_executor_hybrid_fusion() -> None:
    """Verify hybrid search executes vector and bm25 and fuses them."""
    vector_ret = MockRetriever("vector")
    bm25_ret = MockRetriever("bm25")
    engine = {"vector": vector_ret, "bm25": bm25_ret}

    executor = SubQueryExecutor(
        retrieval_engine=engine, router=StrategyRouter(use_llm=False)
    )
    step = DecompositionStep(
        id="step-1", query="general search query", expected_answer_type="code"
    )

    results = executor.execute([step])
    assert vector_ret.call_count == 1
    assert bm25_ret.call_count == 1
    assert "step-1" in results
    assert len(results["step-1"]) > 0
