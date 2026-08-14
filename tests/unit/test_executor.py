"""Unit tests for the sub-query executor (Issue 22).

Covers every acceptance criterion of Issue 22 for the executor:

* Steps execute in dependency (topological) order.
* Context from earlier steps is forwarded into dependent steps' queries.
* A failing step is skipped; its dependents still run with empty context.
* Retry: engine raises once then succeeds -> step NOT skipped.
* Cyclic dependencies raise CyclicDependencyError.
* execute([]) returns {}.
* Each StepResult.strategy matches what the router assigned.
* Hybrid route runs all strategies (dispatches to search_hybrid).

A ``_FakeRetrievalEngine`` and ``_FakeStep`` stand in for the real
retrieval infrastructure, keeping every test network-free (no Qdrant,
no Neo4j, no LLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from reporag.agent.executor import (
    CyclicDependencyError,
    RetrievalEngine,
    SubQueryExecutor,
    _augment_query,
    _build_context_summary,
    _topological_sort,
)
from reporag.agent.router import StrategyRouter
from reporag.retrieval.vector_search import RetrievalResult

# ============================================================================
# Test doubles
# ============================================================================


def _make_result(score: float = 1.0, text: str = "chunk") -> RetrievalResult:
    """Build a minimal RetrievalResult for use in tests."""
    return RetrievalResult(
        score=score,
        file_path="src/auth.py",
        start_line=1,
        end_line=10,
        symbol_name="authenticate_user",
        chunk_text=text,
    )


class _FakeRetrievalEngine:
    """Minimal stand-in for a real retrieval engine.

    Returns caller-supplied results for each strategy. Optionally raises
    on the Nth call to a particular method to simulate engine failures.
    Records every call so tests can assert on call order and arguments.
    """

    def __init__(
        self,
        bm25_results: list[RetrievalResult] | None = None,
        vector_results: list[RetrievalResult] | None = None,
        graph_results: list[RetrievalResult] | None = None,
        hybrid_results: list[RetrievalResult] | None = None,
        *,
        raise_bm25_on: int | None = None,
        raise_vector_on: int | None = None,
        raise_graph_on: int | None = None,
        raise_hybrid_on: int | None = None,
    ) -> None:
        self._results = {
            "bm25": bm25_results or [],
            "vector": vector_results or [],
            "graph": graph_results or [],
            "hybrid": hybrid_results or [],
        }
        self._raise_on = {
            "bm25": raise_bm25_on,
            "vector": raise_vector_on,
            "graph": raise_graph_on,
            "hybrid": raise_hybrid_on,
        }
        self._call_counts: dict[str, int] = {
            "bm25": 0,
            "vector": 0,
            "graph": 0,
            "hybrid": 0,
        }
        self.calls: list[tuple[str, str, int]] = []  # (method, query, top_k)

    def _dispatch(self, method: str, query: str, top_k: int) -> list[RetrievalResult]:
        self._call_counts[method] += 1
        self.calls.append((method, query, top_k))
        raise_on = self._raise_on[method]
        if raise_on is not None and self._call_counts[method] >= raise_on:
            raise RuntimeError(f"simulated {method} engine failure")
        return list(self._results[method])

    def search_bm25(self, query: str, top_k: int) -> list[RetrievalResult]:
        return self._dispatch("bm25", query, top_k)

    def search_vector(self, query: str, top_k: int) -> list[RetrievalResult]:
        return self._dispatch("vector", query, top_k)

    def search_graph(self, query: str, top_k: int) -> list[RetrievalResult]:
        return self._dispatch("graph", query, top_k)

    def search_hybrid(self, query: str, top_k: int) -> list[RetrievalResult]:
        return self._dispatch("hybrid", query, top_k)


@dataclass
class _FakeStep:
    """Minimal stand-in for a DecompositionStep."""

    id: str
    query: str
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    expected_answer_type: str = "code"


def _fixed_router(strategy: str) -> StrategyRouter:
    """Return a StrategyRouter backed by a fake LLM that always returns *strategy*."""
    from reporag.agent.router import StrategyRouter

    class _FixedLLM:
        def __call__(self, prompt: str) -> str:
            import json

            return json.dumps({"strategy": strategy, "confidence": 0.95})

    return StrategyRouter(llm=_FixedLLM(), confidence_threshold=0.0)


def _rules_executor(engine: _FakeRetrievalEngine, **kwargs) -> SubQueryExecutor:
    """Executor with a rule-based (offline) router -- no LLM needed."""
    router = StrategyRouter(use_llm=False, confidence_threshold=0.0)
    return SubQueryExecutor(engine, router=router, **kwargs)


# ============================================================================
# _topological_sort -- pure function tests
# ============================================================================


class TestTopologicalSort:
    def test_no_deps_original_order(self) -> None:
        ids = ["a", "b", "c"]
        deps: dict[str, tuple[str, ...]] = {"a": (), "b": (), "c": ()}
        order = _topological_sort(ids, deps)
        # All have in-degree 0; Kahn's algorithm processes in queue order.
        assert set(order) == {"a", "b", "c"}
        assert len(order) == 3

    def test_linear_chain(self) -> None:
        ids = ["step-1", "step-2", "step-3"]
        deps = {"step-1": (), "step-2": ("step-1",), "step-3": ("step-2",)}
        order = _topological_sort(ids, deps)
        assert order == ["step-1", "step-2", "step-3"]

    def test_diamond_dependency(self) -> None:
        # step-1 -> step-2 -> step-4
        #        -> step-3 -> step-4
        ids = ["step-1", "step-2", "step-3", "step-4"]
        deps = {
            "step-1": (),
            "step-2": ("step-1",),
            "step-3": ("step-1",),
            "step-4": ("step-2", "step-3"),
        }
        order = _topological_sort(ids, deps)
        # step-1 must be first, step-4 must be last.
        assert order[0] == "step-1"
        assert order[-1] == "step-4"
        assert set(order) == set(ids)

    def test_cycle_raises(self) -> None:
        ids = ["a", "b"]
        deps = {"a": ("b",), "b": ("a",)}
        with pytest.raises(CyclicDependencyError):
            _topological_sort(ids, deps)

    def test_self_dep_not_in_ids_ignored(self) -> None:
        # A dep pointing to a non-existent id is silently ignored
        # (validate_steps in the planner catches this; executor is defensive).
        ids = ["step-1", "step-2"]
        deps = {"step-1": (), "step-2": ("step-1", "step-99")}
        order = _topological_sort(ids, deps)
        assert order == ["step-1", "step-2"]

    def test_empty_ids(self) -> None:
        assert _topological_sort([], {}) == []


# ============================================================================
# _build_context_summary -- pure function tests
# ============================================================================


class TestBuildContextSummary:
    def test_empty_results(self) -> None:
        assert _build_context_summary([]) == ""

    def test_single_result(self) -> None:
        results = [_make_result(text="def authenticate(): pass")]
        summary = _build_context_summary(results)
        assert "def authenticate(): pass" in summary

    def test_top_3_only(self) -> None:
        results = [_make_result(text=f"chunk-{i}") for i in range(5)]
        summary = _build_context_summary(results)
        assert "chunk-0" in summary
        assert "chunk-2" in summary
        # The 4th and 5th results should not be included.
        assert "chunk-3" not in summary
        assert "chunk-4" not in summary

    def test_truncated_to_500_chars(self) -> None:
        results = [_make_result(text="x" * 600)]
        summary = _build_context_summary(results)
        assert len(summary) <= 500

    def test_blank_chunk_text_skipped(self) -> None:
        results = [_make_result(text="   "), _make_result(text="real content")]
        summary = _build_context_summary(results)
        assert "real content" in summary
        assert summary.strip()


# ============================================================================
# _augment_query -- pure function tests
# ============================================================================


class TestAugmentQuery:
    def test_no_context_returns_original(self) -> None:
        assert _augment_query("Find auth", []) == "Find auth"

    def test_empty_context_strings_ignored(self) -> None:
        assert _augment_query("Find auth", ["", "  ", ""]) == "Find auth"

    def test_context_appended(self) -> None:
        result = _augment_query("Explain the flow", ["def authenticate(): pass"])
        assert "Explain the flow" in result
        assert "def authenticate(): pass" in result
        assert "[Context from prior steps:" in result

    def test_multiple_contexts_joined(self) -> None:
        result = _augment_query("Step 3", ["ctx-A", "ctx-B"])
        assert "ctx-A" in result
        assert "ctx-B" in result


# ============================================================================
# SubQueryExecutor -- construction
# ============================================================================


class TestSubQueryExecutorConstruction:
    def test_valid_construction(self) -> None:
        engine = _FakeRetrievalEngine()
        router = StrategyRouter(use_llm=False)
        executor = SubQueryExecutor(engine, router=router)
        assert executor is not None

    def test_invalid_engine_raises(self) -> None:
        with pytest.raises(TypeError, match="RetrievalEngine protocol"):
            SubQueryExecutor(object())

    def test_default_router_is_offline(self) -> None:
        """When no router is passed, a rule-based (offline) router is used."""
        engine = _FakeRetrievalEngine()
        executor = SubQueryExecutor(engine)
        assert executor is not None

    def test_repr_contains_key_info(self) -> None:
        engine = _FakeRetrievalEngine()
        executor = SubQueryExecutor(engine, top_k=5)
        assert "top_k=5" in repr(executor)
        assert "_FakeRetrievalEngine" in repr(executor)

    def test_protocol_isinstance_check(self) -> None:
        """_FakeRetrievalEngine satisfies the RetrievalEngine protocol."""
        assert isinstance(_FakeRetrievalEngine(), RetrievalEngine)


# ============================================================================
# SubQueryExecutor.execute -- core behaviour
# ============================================================================


class TestSubQueryExecutorExecute:
    def test_empty_steps_returns_empty_dict(self) -> None:
        engine = _FakeRetrievalEngine()
        executor = _rules_executor(engine)
        assert executor.execute([]) == {}

    def test_single_step_returns_result(self) -> None:
        results = [_make_result(text="auth code")]
        engine = _FakeRetrievalEngine(bm25_results=results)
        executor = _rules_executor(engine)
        step = _FakeStep(id="step-1", query="Find the authenticate function")
        outcome = executor.execute([step])
        assert "step-1" in outcome
        assert outcome["step-1"].results == results
        assert outcome["step-1"].skipped is False

    def test_step_result_has_correct_step_id(self) -> None:
        engine = _FakeRetrievalEngine(bm25_results=[_make_result()])
        executor = _rules_executor(engine)
        step = _FakeStep(id="my-step", query="Find authenticate")
        outcome = executor.execute([step])
        assert outcome["my-step"].step_id == "my-step"

    def test_top_k_passed_to_engine(self) -> None:
        engine = _FakeRetrievalEngine(bm25_results=[])
        executor = _rules_executor(engine, top_k=7)
        executor.execute([_FakeStep(id="s1", query="Find authenticate")])
        assert engine.calls[0][2] == 7

    # ------------------------------------------------------------------
    # Dependency order (acceptance criterion)
    # ------------------------------------------------------------------

    def test_dependency_order_respected(self) -> None:
        """Steps are executed in topological order, not declaration order."""
        call_order: list[str] = []

        class _OrderTrackingEngine:
            def search_bm25(self, query: str, top_k: int) -> list[RetrievalResult]:
                call_order.append(query)
                return []

            def search_vector(self, query: str, top_k: int) -> list[RetrievalResult]:
                call_order.append(query)
                return []

            def search_graph(self, query: str, top_k: int) -> list[RetrievalResult]:
                call_order.append(query)
                return []

            def search_hybrid(self, query: str, top_k: int) -> list[RetrievalResult]:
                call_order.append(query)
                return []

        steps = [
            _FakeStep(id="step-3", query="Q3", depends_on=("step-2",)),
            _FakeStep(id="step-1", query="Q1"),
            _FakeStep(id="step-2", query="Q2", depends_on=("step-1",)),
        ]
        engine = _OrderTrackingEngine()
        executor = _rules_executor(engine)
        executor.execute(steps)

        # Q1 must appear before Q2, and Q2 before Q3 in the actual calls.
        assert call_order.index("Q1") < call_order.index("Q2")
        assert call_order.index("Q2") < call_order.index("Q3")

    def test_parallel_steps_all_execute(self) -> None:
        """Independent steps (no deps) all execute."""
        engine = _FakeRetrievalEngine(bm25_results=[_make_result()])
        executor = _rules_executor(engine)
        steps = [
            _FakeStep(id="step-1", query="Find class A"),
            _FakeStep(id="step-2", query="Find class B"),
        ]
        outcome = executor.execute(steps)
        assert "step-1" in outcome
        assert "step-2" in outcome

    def test_cyclic_dependency_raises(self) -> None:
        engine = _FakeRetrievalEngine()
        executor = _rules_executor(engine)
        steps = [
            _FakeStep(id="step-1", query="Q1", depends_on=("step-2",)),
            _FakeStep(id="step-2", query="Q2", depends_on=("step-1",)),
        ]
        with pytest.raises(CyclicDependencyError):
            executor.execute(steps)

    # ------------------------------------------------------------------
    # Context forwarding (acceptance criterion)
    # ------------------------------------------------------------------

    def test_context_forwarded_to_dependent_step(self) -> None:
        """The augmented_query of a dependent step includes prior context."""
        prior_result = _make_result(text="def authenticate(): pass")
        engine = _FakeRetrievalEngine(
            bm25_results=[prior_result],
            vector_results=[],
        )
        steps = [
            _FakeStep(id="step-1", query="Find authenticate"),
            _FakeStep(
                id="step-2",
                query="How does authentication work?",
                depends_on=("step-1",),
            ),
        ]
        executor = _rules_executor(engine)
        outcome = executor.execute(steps)

        # step-2's augmented_query must reference step-1's context.
        augmented = outcome["step-2"].augmented_query
        assert "def authenticate(): pass" in augmented
        assert "[Context from prior steps:" in augmented

    def test_context_not_forwarded_from_skipped_step(self) -> None:
        """A skipped step produces empty context -- dependent still runs."""
        engine = _FakeRetrievalEngine(
            # Make bm25 always fail so step-1 gets skipped.
            raise_bm25_on=1,
            vector_results=[_make_result(text="vector result")],
        )
        steps = [
            _FakeStep(id="step-1", query="Find authenticate"),
            _FakeStep(
                id="step-2",
                query="How does authentication work?",
                depends_on=("step-1",),
            ),
        ]
        executor = _rules_executor(engine)
        outcome = executor.execute(steps)

        # step-2 should still run.
        assert "step-2" in outcome
        # Its augmented query should NOT contain context from a skipped step.
        augmented = outcome["step-2"].augmented_query
        assert "[Context from prior steps:" not in augmented

    # ------------------------------------------------------------------
    # Failure handling (acceptance criterion)
    # ------------------------------------------------------------------

    def test_engine_failure_marks_step_skipped(self) -> None:
        """After retry exhausted, step is skipped with skipped=True."""
        engine = _FakeRetrievalEngine(raise_bm25_on=1)
        executor = _rules_executor(engine)
        step = _FakeStep(id="step-1", query="Find authenticate")
        outcome = executor.execute([step])
        assert outcome["step-1"].skipped is True
        assert outcome["step-1"].error is not None
        assert outcome["step-1"].results == []

    def test_engine_retries_once_before_skipping(self) -> None:
        """Engine is called twice (original + one retry) before skipping."""
        engine = _FakeRetrievalEngine(raise_bm25_on=1)
        executor = _rules_executor(engine)
        executor.execute([_FakeStep(id="s1", query="Find authenticate")])
        # bm25 called twice: initial call + retry.
        assert engine._call_counts["bm25"] == 2

    def test_engine_succeeds_on_retry(self) -> None:
        """If engine fails on call 1 but succeeds on call 2, step is NOT skipped."""
        good_results = [_make_result(text="auth code")]
        call_n = 0

        class _FailOnceThenSucceed:
            def search_bm25(self, query: str, top_k: int) -> list[RetrievalResult]:
                nonlocal call_n
                call_n += 1
                if call_n == 1:
                    raise RuntimeError("first call fails")
                return good_results

            def search_vector(self, q: str, k: int) -> list[RetrievalResult]:
                return []

            def search_graph(self, q: str, k: int) -> list[RetrievalResult]:
                return []

            def search_hybrid(self, q: str, k: int) -> list[RetrievalResult]:
                return []

        executor = _rules_executor(_FailOnceThenSucceed())
        outcome = executor.execute([_FakeStep(id="s1", query="Find authenticate")])
        assert outcome["s1"].skipped is False
        assert outcome["s1"].results == good_results

    def test_skipped_step_does_not_abort_plan(self) -> None:
        """A skipped step does not stop subsequent independent steps from running."""
        engine = _FakeRetrievalEngine(
            raise_bm25_on=1,
            vector_results=[_make_result(text="semantic result")],
        )
        steps = [
            _FakeStep(id="step-1", query="Find authenticate"),
            _FakeStep(id="step-2", query="Explain the auth flow"),
        ]
        executor = _rules_executor(engine)
        outcome = executor.execute(steps)
        assert outcome["step-1"].skipped is True
        assert outcome["step-2"].skipped is False

    # ------------------------------------------------------------------
    # Strategy dispatch (acceptance criterion)
    # ------------------------------------------------------------------

    def test_bm25_strategy_calls_search_bm25(self) -> None:
        engine = _FakeRetrievalEngine(bm25_results=[_make_result()])
        executor = SubQueryExecutor(engine, router=_fixed_router("bm25"), top_k=5)
        executor.execute([_FakeStep(id="s1", query="Find authenticate")])
        assert engine._call_counts["bm25"] >= 1
        assert engine._call_counts["vector"] == 0
        assert engine._call_counts["graph"] == 0

    def test_vector_strategy_calls_search_vector(self) -> None:
        engine = _FakeRetrievalEngine(vector_results=[_make_result()])
        executor = SubQueryExecutor(engine, router=_fixed_router("vector"), top_k=5)
        executor.execute([_FakeStep(id="s1", query="Explain auth")])
        assert engine._call_counts["vector"] >= 1
        assert engine._call_counts["bm25"] == 0

    def test_graph_strategy_calls_search_graph(self) -> None:
        engine = _FakeRetrievalEngine(graph_results=[_make_result()])
        executor = SubQueryExecutor(engine, router=_fixed_router("graph"), top_k=5)
        executor.execute([_FakeStep(id="s1", query="What calls authenticate?")])
        assert engine._call_counts["graph"] >= 1
        assert engine._call_counts["bm25"] == 0

    def test_hybrid_strategy_calls_search_hybrid(self) -> None:
        """Acceptance criterion: hybrid route runs combined search."""
        engine = _FakeRetrievalEngine(hybrid_results=[_make_result()])
        executor = SubQueryExecutor(engine, router=_fixed_router("hybrid"), top_k=5)
        executor.execute([_FakeStep(id="s1", query="some ambiguous query")])
        assert engine._call_counts["hybrid"] >= 1

    def test_step_result_strategy_matches_router(self) -> None:
        """StepResult.strategy records what the router decided."""
        engine = _FakeRetrievalEngine(graph_results=[_make_result()])
        executor = SubQueryExecutor(engine, router=_fixed_router("graph"), top_k=5)
        outcome = executor.execute(
            [_FakeStep(id="s1", query="What calls authenticate?")]
        )
        assert outcome["s1"].strategy == "graph"

    # ------------------------------------------------------------------
    # Multi-step plan integration test
    # ------------------------------------------------------------------

    def test_three_step_plan_end_to_end(self) -> None:
        """Integration: 3-step plan with dependencies, context, and routing."""
        step1_results = [_make_result(text="def authenticate(): pass")]
        step2_results = [_make_result(text="caller -> authenticate")]
        step3_results = [_make_result(text="auth flow explanation")]

        engine = _FakeRetrievalEngine(
            bm25_results=step1_results,
            graph_results=step2_results,
            vector_results=step3_results,
        )
        steps = [
            # step-1: identifier lookup -> BM25
            _FakeStep(id="step-1", query="Find the authenticate function"),
            # step-2: structural -> graph; depends on step-1
            _FakeStep(
                id="step-2",
                query="What functions call authenticate?",
                depends_on=("step-1",),
            ),
            # step-3: semantic -> vector; depends on step-1 + step-2
            _FakeStep(
                id="step-3",
                query="How does the authentication flow work?",
                depends_on=("step-1", "step-2"),
            ),
        ]
        executor = _rules_executor(engine)
        outcome = executor.execute(steps)

        assert len(outcome) == 3
        assert all(not r.skipped for r in outcome.values())

        # step-1 used BM25.
        assert outcome["step-1"].strategy == "bm25"
        assert outcome["step-1"].results == step1_results

        # step-2 used graph.
        assert outcome["step-2"].strategy == "graph"
        assert outcome["step-2"].results == step2_results

        # step-3's augmented query contains context from step-1 and step-2.
        aug = outcome["step-3"].augmented_query
        assert "def authenticate(): pass" in aug
        assert "caller -> authenticate" in aug
        assert "[Context from prior steps:" in aug
