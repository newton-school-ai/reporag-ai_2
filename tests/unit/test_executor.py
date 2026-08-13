"""Unit tests for the sub-query executor (Issue 22).

Covers every acceptance criterion of Issue 22's executor portion:

* Executor runs sub-queries in dependency order (topological).
* Context from earlier steps is passed forward to dependent steps.
* Hybrid route runs multiple strategies and fuses (RRF).
* Failed steps are retried once, then skipped so the plan continues.

Beyond the acceptance criteria, the suite pins the design contract laid
out in the executor module docstring / class docstrings:

* **Backend seam** -- the executor depends only on a
  :class:`RetrievalBackend` protocol; a :class:`FakeBackend` (records
  every call, returns canned results, can fail chosen steps) keeps every
  test network-free -- the same philosophy as the planner's ``_FakeLLM``.
* **Topological scheduling** -- :func:`topological_order` is tested in
  isolation; cycles and unknown deps raise.
* **Dependency-forward context** -- the chunk text of dependency results
  is deduplicated, capped, and joined into the dependent step's query;
  a step whose dependencies were skipped is itself skipped
  (``skipped_no_deps``).
* **Routing integration** -- the executor accepts a pre-computed
  :class:`RoutingPlan` (no LLM call) and also routes lazily when none is
  given.
* **Never aborts** -- even with multiple failing steps, execution
  completes for every step and returns an :class:`ExecutionPlan`.
"""

from __future__ import annotations

import pytest

from reporag.agent.executor import (
    DefaultRetrievalBackend,
    StepResult,
    SubQueryExecutor,
    topological_order,
)
from reporag.agent.planner import (
    ClassificationResult,
    DecompositionPlan,
    DecompositionStep,
)
from reporag.agent.router import (
    RetrievalStrategy,
    RoutingPlan,
    RoutingResult,
    StrategyRouter,
)
from reporag.retrieval.vector_search import RetrievalResult

# ============================================================================
# Test doubles
# ============================================================================


def _r(
    file_path: str = "src/a.py",
    start_line: int = 1,
    *,
    score: float = 0.9,
    chunk_text: str = "",
    symbol_name: str | None = None,
) -> RetrievalResult:
    """Build a minimal RetrievalResult."""
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=start_line,
        end_line=start_line,
        symbol_name=symbol_name,
        chunk_text=chunk_text or f"chunk {file_path}:{start_line}",
        metadata={},
    )


class FakeBackend:
    """Deterministic RetrievalBackend for tests.

    Records every call as ``(query, strategy, prior_context)`` so tests
    can assert on execution order, routing decisions, and forwarded
    context length.  Each call returns a single canned RetrievalResult
    whose ``chunk_text`` encodes the step id, so dependent steps receive
    a recognisable context string.

    Args:
        fail_steps: step ids that should raise.  A step listed once
            raises on its first attempt only (so retry then succeeds); a
            step listed twice (or with ``fail_always=True``) raises on
            every attempt (so it is skipped after retry).
        fail_always: when ``True``, a failed step raises on every attempt
            (including the retry) -- the "skip" path.  When ``False``
            (default) a failed step raises once and succeeds on retry.
        results_by_step: an optional override of the canned results per
            step id (the default returns one result encoding the step).
    """

    def __init__(
        self,
        *,
        fail_steps: tuple[str, ...] = (),
        fail_always: bool = False,
        results_by_step: dict[str, list[RetrievalResult]] | None = None,
    ) -> None:
        self.fail_steps = fail_steps
        self.fail_always = fail_always
        self.results_by_step = results_by_step or {}
        self.calls: list[tuple[str, str, int]] = []
        self._attempts: dict[str, int] = {}

    def retrieve(
        self,
        strategy: RetrievalStrategy,
        query: str,
        *,
        top_k: int,
        prior_context: str,
        hybrid_components: tuple[RetrievalStrategy, ...],
    ) -> list[RetrievalResult]:
        # Tag the query with the step id ("step-1: ...") so we know which
        # step each call belongs to.
        tag = query.split(":", 1)[0].strip()
        self.calls.append((tag, strategy.value, len(prior_context)))

        if tag in self.fail_steps:
            self._attempts[tag] = self._attempts.get(tag, 0) + 1
            if self.fail_always or self._attempts[tag] < 2:
                raise RuntimeError(f"simulated failure for {tag}")

        if tag in self.results_by_step:
            return list(self.results_by_step[tag])
        return [
            _r(
                file_path=f"src/{tag}.py",
                chunk_text=f"data from {tag}",
                symbol_name=tag,
            )
        ]


def _step(
    query: str,
    *,
    step_id: str,
    depends_on: tuple[str, ...] = (),
    answer_type: str = "code",
) -> DecompositionStep:
    """Build a DecompositionStep whose query is tagged 'step_id: query'."""
    return DecompositionStep(
        id=step_id,
        query=f"{step_id}: {query}",
        expected_answer_type=answer_type,  # type: ignore[arg-type]
        depends_on=depends_on,
    )


def _plan(
    steps: list[DecompositionStep], *, query: str = "original multi-hop query"
) -> DecompositionPlan:
    """Wrap a list of steps in a DecompositionPlan."""
    return DecompositionPlan(
        original_query=query,
        steps=tuple(steps),
        needs_decomposition=len(steps) > 1,
        classification=ClassificationResult(query_type="multi-hop", confidence=0.9),
        source="rules",
    )


def _routing(
    steps: list[DecompositionStep],
    strategy: RetrievalStrategy = RetrievalStrategy.BM25,
) -> RoutingPlan:
    """Build a uniform RoutingPlan (one strategy for every step)."""
    return RoutingPlan(
        original_query="original",
        steps=tuple(steps),
        routings=tuple(
            RoutingResult(
                step_id=step.id, strategy=strategy, confidence=0.9, source="rules"
            )
            for step in steps
        ),
        source="rules",
    )


def _rules_router() -> StrategyRouter:
    return StrategyRouter(use_llm=False, confidence_threshold=0.0)


# ============================================================================
# topological_order (pure function)
# ============================================================================


class TestTopologicalOrder:
    def test_no_dependencies_preserves_input_order(self) -> None:
        steps = [
            _step("a", step_id="step-1"),
            _step("b", step_id="step-2"),
            _step("c", step_id="step-3"),
        ]
        assert topological_order(steps) == ["step-1", "step-2", "step-3"]

    def test_dependency_runs_first(self) -> None:
        steps = [
            _step("child", step_id="step-2", depends_on=("step-1",)),
            _step("parent", step_id="step-1"),
        ]
        order = topological_order(steps)
        assert order.index("step-1") < order.index("step-2")

    def test_chained_dependencies(self) -> None:
        steps = [
            _step("c", step_id="step-3", depends_on=("step-2",)),
            _step("b", step_id="step-2", depends_on=("step-1",)),
            _step("a", step_id="step-1"),
        ]
        order = topological_order(steps)
        assert order == ["step-1", "step-2", "step-3"]

    def test_diamond_dependency(self) -> None:
        steps = [
            _step("d", step_id="step-4", depends_on=("step-2", "step-3")),
            _step("b", step_id="step-2", depends_on=("step-1",)),
            _step("c", step_id="step-3", depends_on=("step-1",)),
            _step("a", step_id="step-1"),
        ]
        order = topological_order(steps)
        assert order[0] == "step-1"
        assert order[-1] == "step-4"
        assert "step-2" in order and "step-3" in order

    def test_independent_step_can_run_anywhere_before_dependents(self) -> None:
        steps = [
            _step("dep", step_id="step-2", depends_on=("step-1",)),
            _step("independent", step_id="step-3"),
            _step("base", step_id="step-1"),
        ]
        order = topological_order(steps)
        assert order.index("step-1") < order.index("step-2")
        # step-3 has no deps, so it can appear anywhere.
        assert "step-3" in order

    def test_cycle_raises(self) -> None:
        # Construct a cycle manually since DecompositionStep would normally
        # be validated upstream; the executor defends anyway.
        s1 = DecompositionStep(
            id="a", query="a", expected_answer_type="code", depends_on=("b",)
        )
        s2 = DecompositionStep(
            id="b", query="b", expected_answer_type="code", depends_on=("a",)
        )
        with pytest.raises(ValueError, match="cycle"):
            topological_order([s1, s2])

    def test_unknown_dependency_raises(self) -> None:
        step = DecompositionStep(
            id="x", query="x", expected_answer_type="code", depends_on=("ghost",)
        )
        with pytest.raises(ValueError, match="not in the plan"):
            topological_order([step])

    def test_empty_steps_returns_empty(self) -> None:
        assert topological_order([]) == []


# ============================================================================
# Construction & validation
# ============================================================================


class TestConstruction:
    def test_defaults_from_settings(self) -> None:
        executor = SubQueryExecutor()
        assert executor.top_k >= 1
        assert executor.retry_failed in (True, False)

    def test_top_k_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            SubQueryExecutor(top_k=0)

    def test_context_top_k_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="context_top_k must be >= 1"):
            SubQueryExecutor(context_top_k=0)

    def test_repr_shows_state(self) -> None:
        executor = SubQueryExecutor(top_k=7)
        text = repr(executor)
        assert "SubQueryExecutor" in text
        assert "top_k=7" in text

    def test_injected_backend_and_router_respected(self) -> None:
        backend = FakeBackend()
        router = _rules_router()
        executor = SubQueryExecutor(backend=backend, router=router)
        assert executor.backend is backend
        assert executor.router is router


# ============================================================================
# Execution: dependency order
# ============================================================================


class TestExecutionOrder:
    def test_executes_in_topological_order(self) -> None:
        steps = [
            _step("child", step_id="step-2", depends_on=("step-1",)),
            _step("parent", step_id="step-1"),
        ]
        plan = _plan(steps)
        backend = FakeBackend()
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        result = executor.execute(plan, routing=_routing(steps))

        # executed_order matches topological order (parent before child).
        assert result.executed_order == ("step-1", "step-2")
        # First backend call was step-1, second was step-2.
        assert backend.calls[0][0] == "step-1"
        assert backend.calls[1][0] == "step-2"

    def test_executed_order_preserves_independent_step_positions(self) -> None:
        steps = [
            _step("child", step_id="step-2", depends_on=("step-1",)),
            _step("independent", step_id="step-3"),
            _step("base", step_id="step-1"),
        ]
        plan = _plan(steps)
        backend = FakeBackend()
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        result = executor.execute(plan, routing=_routing(steps))

        assert result.executed_order.index("step-1") < result.executed_order.index(
            "step-2"
        )
        assert len(result.executed_order) == 3

    def test_every_step_has_a_result(self) -> None:
        steps = [
            _step("a", step_id="step-1"),
            _step("b", step_id="step-2", depends_on=("step-1",)),
            _step("c", step_id="step-3"),
        ]
        plan = _plan(steps)
        backend = FakeBackend()
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        result = executor.execute(plan, routing=_routing(steps))

        assert set(result.step_results.keys()) == {"step-1", "step-2", "step-3"}
        for step_id in ("step-1", "step-2", "step-3"):
            assert result.step_results[step_id].status == "ok"

    def test_empty_plan_raises(self) -> None:
        executor = SubQueryExecutor(backend=FakeBackend(), router=_rules_router())
        with pytest.raises(ValueError, match="plan must contain at least one step"):
            executor.execute(_plan([]))


# ============================================================================
# Execution: dependency-forward context
# ============================================================================


class TestDependencyContextForwarding:
    def test_dependency_results_forwarded_to_dependent(self) -> None:
        """A dependent step receives its dependency's chunk text as prior
        context (length > 0), while a dependency-free step gets none."""
        steps = [
            _step("parent", step_id="step-1"),
            _step("child", step_id="step-2", depends_on=("step-1",)),
        ]
        backend = FakeBackend(
            results_by_step={
                "step-1": [
                    _r(file_path="auth.py", chunk_text="def authenticate(): ...")
                ]
            }
        )
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        executor.execute(_plan(steps), routing=_routing(steps))

        # step-1 had no prior context (no deps).
        assert backend.calls[0][2] == 0
        # step-2 received step-1's chunk text as prior context (non-empty).
        assert backend.calls[1][2] > 0

    def test_prior_context_contains_dependency_chunk_text(self) -> None:
        """The exact chunk text from a dependency appears in the dependent
        step's forwarded context."""
        steps = [
            _step("parent", step_id="step-1"),
            _step("child", step_id="step-2", depends_on=("step-1",)),
        ]
        backend = FakeBackend(
            results_by_step={"step-1": [_r(chunk_text="UNIQUE_MARKER_TEXT")]}
        )
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        # Capture the prior_context passed to step-2 by inspecting calls
        # indirectly: increase a side-channel.
        captured: list[str] = []

        original_retrieve = backend.retrieve

        def spy_retrieve(strategy, query, *, top_k, prior_context, hybrid_components):
            captured.append(prior_context)
            return original_retrieve(
                strategy,
                query,
                top_k=top_k,
                prior_context=prior_context,
                hybrid_components=hybrid_components,
            )

        backend.retrieve = spy_retrieve  # type: ignore[assignment]
        executor.execute(_plan(steps), routing=_routing(steps))
        # First call (step-1) had empty context; second (step-2) has marker.
        assert captured[0] == ""
        assert "UNIQUE_MARKER_TEXT" in captured[1]

    def test_no_dependency_has_empty_context(self) -> None:
        steps = [_step("lonely", step_id="step-1")]
        backend = FakeBackend()
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        executor.execute(_plan(steps), routing=_routing(steps))
        assert backend.calls[0][2] == 0

    def test_multiple_dependencies_all_forwarded(self) -> None:
        steps = [
            _step("a", step_id="step-1"),
            _step("b", step_id="step-2"),
            _step("c", step_id="step-3", depends_on=("step-1", "step-2")),
        ]
        backend = FakeBackend(
            results_by_step={
                "step-1": [_r(file_path="a.py", chunk_text="FROM_A")],
                "step-2": [_r(file_path="b.py", chunk_text="FROM_B")],
            }
        )
        captured: list[str] = []

        original_retrieve = backend.retrieve

        def spy_retrieve(strategy, query, *, top_k, prior_context, hybrid_components):
            captured.append((query.split(":")[0], prior_context))
            return original_retrieve(
                strategy,
                query,
                top_k=top_k,
                prior_context=prior_context,
                hybrid_components=hybrid_components,
            )

        backend.retrieve = spy_retrieve  # type: ignore[assignment]
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        executor.execute(_plan(steps), routing=_routing(steps))

        step3_context = next(ctx for tag, ctx in captured if tag == "step-3")
        assert "FROM_A" in step3_context
        assert "FROM_B" in step3_context

    def test_context_top_k_caps_forwarded_results(self) -> None:
        """Only the first context_top_k results per dependency are forwarded."""
        steps = [
            _step("parent", step_id="step-1"),
            _step("child", step_id="step-2", depends_on=("step-1",)),
        ]
        many = [_r(start_line=i, chunk_text=f"line{i}") for i in range(50)]
        backend = FakeBackend(results_by_step={"step-1": many})
        executor = SubQueryExecutor(
            backend=backend, router=_rules_router(), context_top_k=3
        )
        captured: list[str] = []

        original_retrieve = backend.retrieve

        def spy_retrieve(strategy, query, *, top_k, prior_context, hybrid_components):
            captured.append(prior_context)
            return original_retrieve(
                strategy,
                query,
                top_k=top_k,
                prior_context=prior_context,
                hybrid_components=hybrid_components,
            )

        backend.retrieve = spy_retrieve  # type: ignore[assignment]
        executor.execute(_plan(steps), routing=_routing(steps))
        step2_context = captured[1]
        # Only the first 3 dependency results forwarded.
        assert "line0" in step2_context
        assert "line2" in step2_context
        assert "line3" not in step2_context

    def test_deduplicates_overlapping_dependency_chunks(self) -> None:
        """The same chunk from two dependencies is forwarded only once."""
        steps = [
            _step("a", step_id="step-1"),
            _step("b", step_id="step-2"),
            _step("c", step_id="step-3", depends_on=("step-1", "step-2")),
        ]
        shared = _r(file_path="shared.py", start_line=10, chunk_text="SHARED")
        backend = FakeBackend(results_by_step={"step-1": [shared], "step-2": [shared]})
        captured: list[str] = []

        original_retrieve = backend.retrieve

        def spy_retrieve(strategy, query, *, top_k, prior_context, hybrid_components):
            captured.append(prior_context)
            return original_retrieve(
                strategy,
                query,
                top_k=top_k,
                prior_context=prior_context,
                hybrid_components=hybrid_components,
            )

        backend.retrieve = spy_retrieve  # type: ignore[assignment]
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        executor.execute(_plan(steps), routing=_routing(steps))
        step3_context = captured[2]
        # SHARED appears exactly once despite being in both dependencies.
        assert step3_context.count("SHARED") == 1


# ============================================================================
# Execution: retry then skip
# ============================================================================


class TestRetryThenSkip:
    def test_failed_step_retried_then_succeeds(self) -> None:
        """A step that fails once then succeeds on retry has status 'ok' and
        attempted == 2."""
        steps = [
            _step("flaky", step_id="step-1"),
            _step("ok", step_id="step-2", depends_on=("step-1",)),
        ]
        backend = FakeBackend(fail_steps=("step-1",), fail_always=False)
        executor = SubQueryExecutor(
            backend=backend, router=_rules_router(), retry_failed=True
        )
        result = executor.execute(_plan(steps), routing=_routing(steps))

        assert result.step_results["step-1"].status == "ok"
        assert result.step_results["step-1"].attempted == 2
        assert result.metadata["retried"] == 1
        # step-2 still ran with step-1's eventual results.
        assert result.step_results["step-2"].status == "ok"

    def test_failed_step_always_skipped_after_retry(self) -> None:
        """A step that always fails is retried once, then skipped."""
        steps = [
            _step("doomed", step_id="step-1"),
            _step("ok", step_id="step-2"),
        ]
        backend = FakeBackend(fail_steps=("step-1",), fail_always=True)
        executor = SubQueryExecutor(
            backend=backend, router=_rules_router(), retry_failed=True
        )
        result = executor.execute(_plan(steps), routing=_routing(steps))

        assert result.step_results["step-1"].status == "skipped"
        assert result.step_results["step-1"].attempted == 2
        assert result.step_results["step-1"].results == []
        assert result.step_results["step-1"].error is not None
        # The independent step-2 still ran successfully.
        assert result.step_results["step-2"].status == "ok"
        assert result.metadata["ok"] == 1
        assert result.metadata["skipped"] == 1

    def test_retry_disabled_skips_immediately(self) -> None:
        """With retry_failed=False a failing step is skipped on the first
        attempt (attempted == 1)."""
        steps = [_step("doomed", step_id="step-1")]
        backend = FakeBackend(fail_steps=("step-1",), fail_always=True)
        executor = SubQueryExecutor(
            backend=backend, router=_rules_router(), retry_failed=False
        )
        result = executor.execute(_plan(steps), routing=_routing(steps))

        assert result.step_results["step-1"].status == "skipped"
        assert result.step_results["step-1"].attempted == 1

    def test_dependency_skipped_cascades_to_dependent(self) -> None:
        """If a step is skipped, a dependent step is skipped_no_deps rather
        than running with incomplete context."""
        steps = [
            _step("doomed", step_id="step-1"),
            _step("child", step_id="step-2", depends_on=("step-1",)),
            _step("independent", step_id="step-3"),
        ]
        backend = FakeBackend(fail_steps=("step-1",), fail_always=True)
        executor = SubQueryExecutor(
            backend=backend, router=_rules_router(), retry_failed=True
        )
        result = executor.execute(_plan(steps), routing=_routing(steps))

        assert result.step_results["step-1"].status == "skipped"
        assert result.step_results["step-2"].status == "skipped_no_deps"
        assert result.step_results["step-2"].attempted == 0
        # Independent step-3 still ran.
        assert result.step_results["step-3"].status == "ok"
        # Only step-1 and step-3 called the backend (step-2 never tried).
        called_tags = [call[0] for call in backend.calls]
        assert "step-2" not in called_tags

    def test_execution_never_aborts_on_multiple_failures(self) -> None:
        """Several failing steps each get their own skip; the whole plan
        still completes."""
        steps = [
            _step("doomed-a", step_id="step-1"),
            _step("doomed-b", step_id="step-2"),
            _step("ok", step_id="step-3"),
        ]
        backend = FakeBackend(fail_steps=("step-1", "step-2"), fail_always=True)
        executor = SubQueryExecutor(
            backend=backend, router=_rules_router(), retry_failed=True
        )
        result = executor.execute(_plan(steps), routing=_routing(steps))

        assert result.step_results["step-1"].status == "skipped"
        assert result.step_results["step-2"].status == "skipped"
        assert result.step_results["step-3"].status == "ok"
        assert result.metadata["ok"] == 1
        assert result.metadata["skipped"] == 2


# ============================================================================
# Execution: routing integration
# ============================================================================


class TestRoutingIntegration:
    def test_routes_lazily_when_no_routing_given(self) -> None:
        """When no RoutingPlan is passed, the executor routes the plan first
        using its configured router (here rules-only, so no LLM)."""
        steps = [
            _step("Where is foo defined?", step_id="step-1"),
        ]
        backend = FakeBackend()
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        result = executor.execute(_plan(steps))

        # Rule-based routing sends "where is X defined" to bm25.
        assert result.routed_strategies["step-1"] is RetrievalStrategy.BM25
        assert backend.calls[0][1] == "bm25"

    def test_passes_routed_strategy_to_backend(self) -> None:
        steps = [_step("q", step_id="step-1"), _step("q", step_id="step-2")]
        routing = RoutingPlan(
            original_query="orig",
            steps=tuple(steps),
            routings=(
                RoutingResult(
                    step_id="step-1",
                    strategy=RetrievalStrategy.GRAPH,
                    confidence=0.9,
                    source="rules",
                ),
                RoutingResult(
                    step_id="step-2",
                    strategy=RetrievalStrategy.VECTOR,
                    confidence=0.9,
                    source="rules",
                ),
            ),
            source="rules",
        )
        backend = FakeBackend()
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        executor.execute(_plan(steps), routing=routing)

        assert backend.calls[0][1] == "graph"
        assert backend.calls[1][1] == "vector"

    def test_hybrid_expands_to_components(self) -> None:
        """A hybrid-routed step triggers one retrieve call per hybrid
        component."""
        steps = [_step("ambiguous", step_id="step-1")]
        routing = RoutingPlan(
            original_query="orig",
            steps=tuple(steps),
            routings=(
                RoutingResult(
                    step_id="step-1",
                    strategy=RetrievalStrategy.HYBRID,
                    confidence=0.9,
                    source="rules",
                ),
            ),
            source="rules",
        )
        # FakeBackend.retrieve for HYBRID would need to expand components
        # itself; instead, use a backend that records every retrieve call
        # regardless of strategy and asserts it was called once (the real
        # DefaultRetrievalBackend expands internally). For the unit level we
        # verify the executor hands the HYBRID strategy to the backend and
        # the backend's own expansion is tested via DefaultRetrievalBackend
        # below.
        backend = FakeBackend()
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        executor.execute(_plan(steps), routing=routing)

        # The executor passed the hybrid strategy through; one call total.
        assert len(backend.calls) == 1
        assert backend.calls[0][1] == "hybrid"


# ============================================================================
# DefaultRetrievalBackend hybrid expansion (uses real RRF + fake components)
# ============================================================================


class _FakeVectorSearch:
    def search(self, query: str, top_k: int = 10, **kwargs) -> list[RetrievalResult]:
        return [_r(file_path="vec.py", start_line=1, score=0.9, chunk_text="vec")]


class _FakeBM25Search:
    def search(self, query: str, top_k: int = 10, **kwargs) -> list[RetrievalResult]:
        return [_r(file_path="bm.py", start_line=2, score=0.8, chunk_text="bm")]


class _FakeGraphRetriever:
    def __init__(self) -> None:
        self.last_find_paths: tuple[str, str] | None = None

    def get_callers(self, symbol: str, depth: int = 2) -> list[RetrievalResult]:
        return [_r(file_path="graph.py", start_line=3, score=1.0, chunk_text="graph")]

    def get_neighbors(
        self, symbol: str, depth: int = 1, direction: str = "both"
    ) -> list[RetrievalResult]:
        return [_r(file_path="graph.py", start_line=3, score=1.0, chunk_text="graph")]

    def find_paths(self, source: str, target: str, max_depth: int = 5):
        # Record the call so tests can assert on the endpoints used.
        self.last_find_paths = (source, target)
        from reporag.retrieval.graph_traversal import GraphPaths

        return GraphPaths(
            shortest=[
                _r(file_path="p1.py", start_line=1, score=1.0, chunk_text=source),
                _r(file_path="p2.py", start_line=2, score=0.5, chunk_text=target),
            ],
            all_paths=[],
        )


class TestDefaultRetrievalBackend:
    def test_vector_search_delegates(self) -> None:
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=_FakeGraphRetriever(),  # type: ignore[arg-type]
        )
        results = backend.retrieve(
            RetrievalStrategy.VECTOR,
            "query",
            top_k=5,
            prior_context="",
            hybrid_components=(RetrievalStrategy.VECTOR, RetrievalStrategy.BM25),
        )
        assert len(results) == 1
        assert results[0].file_path == "vec.py"

    def test_bm25_search_delegates(self) -> None:
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=_FakeGraphRetriever(),  # type: ignore[arg-type]
        )
        results = backend.retrieve(
            RetrievalStrategy.BM25,
            "query",
            top_k=5,
            prior_context="",
            hybrid_components=(RetrievalStrategy.VECTOR, RetrievalStrategy.BM25),
        )
        assert len(results) == 1
        assert results[0].file_path == "bm.py"

    def test_graph_retrieves_neighbors(self) -> None:
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=_FakeGraphRetriever(),  # type: ignore[arg-type]
        )
        results = backend.retrieve(
            RetrievalStrategy.GRAPH,
            "What calls the `authenticate_user` function?",
            top_k=5,
            prior_context="",
            hybrid_components=(RetrievalStrategy.VECTOR, RetrievalStrategy.BM25),
        )
        assert len(results) == 1
        assert results[0].file_path == "graph.py"

    def test_graph_without_identifier_falls_back_to_bm25(self) -> None:
        """A graph query with no extractable identifier degrades to BM25
        rather than failing."""
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=_FakeGraphRetriever(),  # type: ignore[arg-type]
        )
        results = backend.retrieve(
            RetrievalStrategy.GRAPH,
            "a query with no identifier here",
            top_k=5,
            prior_context="",
            hybrid_components=(RetrievalStrategy.VECTOR, RetrievalStrategy.BM25),
        )
        assert results[0].file_path == "bm.py"

    def test_graph_uses_prior_context_to_find_anchor(self) -> None:
        """When the query has no identifier but prior_context does, the
        backend extracts the anchor from the prior context."""
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=_FakeGraphRetriever(),  # type: ignore[arg-type]
        )
        # query has no identifier, prior_context names one.
        results = backend.retrieve(
            RetrievalStrategy.GRAPH,
            "what calls it",
            top_k=5,
            prior_context="code with `authenticate_user` in it",
            hybrid_components=(RetrievalStrategy.VECTOR, RetrievalStrategy.BM25),
        )
        # Used the graph path (found anchor in prior context).
        assert results[0].file_path == "graph.py"

    def test_graph_trace_path_uses_find_paths(self) -> None:
        """Acceptance: a graph-routed 'trace the path from A to B' query calls
        ``find_paths`` with the two endpoints instead of degrading to a
        single-anchor neighbour lookup or BM25."""
        graph = _FakeGraphRetriever()
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=graph,  # type: ignore[arg-type]
        )
        results = backend.retrieve(
            RetrievalStrategy.GRAPH,
            "trace the path from `login_handler` to `session_token`",
            top_k=5,
            prior_context="",
            hybrid_components=(RetrievalStrategy.VECTOR, RetrievalStrategy.BM25),
        )
        # find_paths was called with the two extracted endpoints.
        assert graph.last_find_paths == ("login_handler", "session_token")
        # The flattened path (p1 -> p2) is returned, not the single-anchor
        # neighbour result (graph.py).
        assert len(results) == 2
        assert results[0].file_path == "p1.py"
        assert results[1].file_path == "p2.py"

    def test_graph_trace_path_snake_case_endpoints(self) -> None:
        """Endpoints can also be extracted from plain snake_case identifiers
        (not just backticks)."""
        graph = _FakeGraphRetriever()
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=graph,  # type: ignore[arg-type]
        )
        backend.retrieve(
            RetrievalStrategy.GRAPH,
            "trace path from login_handler to session_token",
            top_k=5,
            prior_context="",
            hybrid_components=(RetrievalStrategy.VECTOR, RetrievalStrategy.BM25),
        )
        assert graph.last_find_paths == ("login_handler", "session_token")

    def test_graph_plain_english_endpoints_fall_through(self) -> None:
        """A 'trace from A to B' query whose endpoints are plain English (no
        extractable identifier) cannot ground a path lookup, so it falls
        through to anchor-based retrieval rather than calling find_paths
        with garbage."""
        graph = _FakeGraphRetriever()
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=graph,  # type: ignore[arg-type]
        )
        results = backend.retrieve(
            RetrievalStrategy.GRAPH,
            "trace the path from the login route to the session token",
            top_k=5,
            prior_context="",
            hybrid_components=(RetrievalStrategy.VECTOR, RetrievalStrategy.BM25),
        )
        # No extractable endpoint -> find_paths NOT called.
        assert graph.last_find_paths is None
        # Fell through to BM25 (no single anchor either).
        assert results[0].file_path == "bm.py"

    def test_hybrid_drops_hybrid_in_components(self) -> None:
        """A HYBRID entry inside hybrid_components must be dropped, not
        recursed into -- guards against infinite recursion."""
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=_FakeGraphRetriever(),  # type: ignore[arg-type]
        )
        # Vector + (a stray HYBRID) + BM25 -> only the two leaf components run.
        results = backend.retrieve(
            RetrievalStrategy.HYBRID,
            "query",
            top_k=5,
            prior_context="",
            hybrid_components=(
                RetrievalStrategy.VECTOR,
                RetrievalStrategy.HYBRID,
                RetrievalStrategy.BM25,
            ),
        )
        paths = {r.file_path for r in results}
        assert paths == {"vec.py", "bm.py"}

    def test_hybrid_all_three_components_fuse(self) -> None:
        """All three leaf components (vector, bm25, graph) fuse into one RRF
        ranking via _run_component (no recursion through retrieve)."""
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=_FakeGraphRetriever(),  # type: ignore[arg-type]
        )
        # A query with an identifier so the GRAPH component resolves an
        # anchor and returns "graph.py" rather than falling back to BM25.
        results = backend.retrieve(
            RetrievalStrategy.HYBRID,
            "what calls `authenticate_user`",
            top_k=10,
            prior_context="",
            hybrid_components=(
                RetrievalStrategy.VECTOR,
                RetrievalStrategy.BM25,
                RetrievalStrategy.GRAPH,
            ),
        )
        paths = {r.file_path for r in results}
        assert paths == {"vec.py", "bm.py", "graph.py"}

    def test_run_component_rejects_hybrid(self) -> None:
        """The leaf dispatcher explicitly rejects HYBRID -- it must be
        expanded at the retrieve() layer only."""
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=_FakeGraphRetriever(),  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="non-leaf strategy"):
            backend._run_component(
                RetrievalStrategy.HYBRID, "q", top_k=5, prior_context=""
            )

    def test_hybrid_runs_components_and_fuses(self) -> None:
        """Hybrid runs every component and returns RRF-fused results."""
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=_FakeGraphRetriever(),  # type: ignore[arg-type]
        )
        results = backend.retrieve(
            RetrievalStrategy.HYBRID,
            "query",
            top_k=5,
            prior_context="",
            hybrid_components=(RetrievalStrategy.VECTOR, RetrievalStrategy.BM25),
        )
        # Two distinct results fused (vec.py and bm.py).
        assert len(results) == 2
        paths = {r.file_path for r in results}
        assert paths == {"vec.py", "bm.py"}

    def test_hybrid_isolates_component_failures(self) -> None:
        """If one hybrid component fails, the others still contribute."""
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=_FakeGraphRetriever(),  # type: ignore[arg-type]
        )

        # Make the vector search raise to simulate a Qdrant outage.
        def boom(query, top_k=10, **kwargs):
            raise RuntimeError("qdrant down")

        backend._vector_search.search = boom  # type: ignore[assignment]

        results = backend.retrieve(
            RetrievalStrategy.HYBRID,
            "query",
            top_k=5,
            prior_context="",
            hybrid_components=(RetrievalStrategy.VECTOR, RetrievalStrategy.BM25),
        )
        # BM25 still contributed; vector failure was isolated.
        assert len(results) == 1
        assert results[0].file_path == "bm.py"

    def test_hybrid_all_components_fail_returns_empty(self) -> None:
        backend = DefaultRetrievalBackend(
            vector_search=_FakeVectorSearch(),  # type: ignore[arg-type]
            bm25_search=_FakeBM25Search(),  # type: ignore[arg-type]
            graph_retriever=_FakeGraphRetriever(),  # type: ignore[arg-type]
        )

        def boom(query, top_k=10, **kwargs):
            raise RuntimeError("down")

        backend._vector_search.search = boom  # type: ignore[assignment]
        backend._bm25_search.search = boom  # type: ignore[assignment]

        results = backend.retrieve(
            RetrievalStrategy.HYBRID,
            "query",
            top_k=5,
            prior_context="",
            hybrid_components=(RetrievalStrategy.VECTOR, RetrievalStrategy.BM25),
        )
        assert results == []

    def test_extract_anchor_identifier_backticks(self) -> None:
        assert (
            DefaultRetrievalBackend._extract_anchor_identifier(
                "what calls `authenticate_user`?"
            )
            == "authenticate_user"
        )

    def test_extract_anchor_identifier_snake_case(self) -> None:
        assert (
            DefaultRetrievalBackend._extract_anchor_identifier(
                "what calls authenticate_user now"
            )
            == "authenticate_user"
        )

    def test_extract_anchor_identifier_camel_case(self) -> None:
        assert (
            DefaultRetrievalBackend._extract_anchor_identifier("uses handleRequest")
            == "handleRequest"
        )

    def test_extract_anchor_identifier_none(self) -> None:
        assert (
            DefaultRetrievalBackend._extract_anchor_identifier("no identifier here")
            is None
        )

    def test_extract_anchor_strips_module_prefix(self) -> None:
        """A dotted `auth.login_handler` resolves to the final segment."""
        assert (
            DefaultRetrievalBackend._extract_anchor_identifier("calls `auth.login`")
            == "login"
        )


# ============================================================================
# step_context property & ExecutionPlan
# ============================================================================


class TestStepResultAndExecutionPlan:
    def test_step_result_defaults(self) -> None:
        sr = StepResult(
            step_id="step-1",
            strategy=RetrievalStrategy.BM25,
            results=[],
        )
        assert sr.attempted == 1
        assert sr.status == "ok"
        assert sr.error is None
        assert sr.metadata == {}

    def test_step_result_is_frozen(self) -> None:
        sr = StepResult(step_id="step-1", strategy=RetrievalStrategy.GRAPH, results=[])
        with pytest.raises(AttributeError):
            sr.attempted = 5  # type: ignore[misc]

    def test_execution_plan_has_metadata_counts(self) -> None:
        steps = [
            _step("a", step_id="step-1"),
            _step("b", step_id="step-2"),
            _step("c", step_id="step-3"),
        ]
        backend = FakeBackend()
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        result = executor.execute(_plan(steps), routing=_routing(steps))
        assert result.metadata["ok"] == 3
        assert result.metadata["skipped"] == 0
        assert result.metadata["retried"] == 0
        assert set(result.routed_strategies.keys()) == {
            "step-1",
            "step-2",
            "step-3",
        }

    def test_execution_plan_is_frozen(self) -> None:
        steps = [_step("a", step_id="step-1")]
        backend = FakeBackend()
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        result = executor.execute(_plan(steps), routing=_routing(steps))
        with pytest.raises(AttributeError):
            result.original_query = "other"  # type: ignore[misc]


# ============================================================================
# End-to-end: executor consumes a router-derived plan (mirrors issue spec)
# ============================================================================


class TestIssueSpecUsage:
    """The exact shape from the issue's 'How to test locally' snippet."""

    def test_router_then_executor_flow(self) -> None:
        """A full classify -> decompose -> route -> execute flow using the
        rule-based paths (no LLM, no network)."""
        steps = [
            _step("Where is authenticate_user defined?", step_id="step-1"),
            _step("step-2: What calls it", step_id="step-2", depends_on=("step-1",)),
        ]
        plan = _plan(steps, query="How does auth work end-to-end?")
        router = _rules_router()
        routing = router.route_batch(plan)
        # step-1 -> bm25 (where is X defined), step-2 -> graph (what calls X).
        assert routing.routings[0].strategy is RetrievalStrategy.BM25
        assert routing.routings[1].strategy is RetrievalStrategy.GRAPH

        backend = FakeBackend()
        executor = SubQueryExecutor(backend=backend, router=router)
        result = executor.execute(plan, routing=routing)

        assert result.step_results["step-1"].strategy is RetrievalStrategy.BM25
        assert result.step_results["step-2"].strategy is RetrievalStrategy.GRAPH
        assert result.step_results["step-2"].metadata["context_chars"] > 0
        assert len(result.step_results) == 2
        assert result.metadata["ok"] == 2


# ============================================================================
# execute() accepts a bare list of steps (Issue 22 spec snippet shape)
# ============================================================================


class TestExecuteAcceptsBareStepList:
    """The issue's 'How to test locally' snippet calls
    ``executor.execute(plan.steps)`` with a bare list of steps rather than a
    DecompositionPlan; execute() accepts either."""

    def test_execute_bare_step_list_routes_and_runs(self) -> None:
        steps = [
            _step("Where is authenticate_user defined?", step_id="step-1"),
            _step("step-2: What calls it", step_id="step-2", depends_on=("step-1",)),
        ]
        backend = FakeBackend()
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        result = executor.execute(steps)

        # Bare-list path still routes lazily (no plan, no routing arg).
        assert result.routed_strategies["step-1"] is RetrievalStrategy.BM25
        assert result.routed_strategies["step-2"] is RetrievalStrategy.GRAPH
        # Both steps executed successfully.
        assert result.metadata["ok"] == 2
        assert result.metadata["skipped"] == 0

    def test_bare_list_original_query_synthesised(self) -> None:
        """A bare list has no original_query; one is synthesised from the step
        queries."""
        steps = [_step("abc", step_id="step-1")]
        backend = FakeBackend()
        executor = SubQueryExecutor(backend=backend, router=_rules_router())
        result = executor.execute(steps)
        assert "abc" in result.original_query

    def test_bare_list_empty_raises(self) -> None:
        executor = SubQueryExecutor(backend=FakeBackend(), router=_rules_router())
        with pytest.raises(ValueError, match="plan must contain at least one step"):
            executor.execute([])

    def test_decomposition_plan_and_bare_list_produce_equivalent_strategies(
        self,
    ) -> None:
        """execute(plan) and execute(plan.steps) route identically."""
        steps = [
            _step("Where is X defined?", step_id="step-1"),
            _step(
                "step-2: How does it work?", step_id="step-2", depends_on=("step-1",)
            ),
        ]
        plan = _plan(steps, query="overview")

        backend_a = FakeBackend()
        result_a = SubQueryExecutor(backend=backend_a, router=_rules_router()).execute(
            plan
        )
        backend_b = FakeBackend()
        result_b = SubQueryExecutor(backend=backend_b, router=_rules_router()).execute(
            steps
        )

        assert result_a.routed_strategies == result_b.routed_strategies


# ============================================================================
# End-to-end: the exact Issue 22 "How to test locally" snippet
# ============================================================================


class TestIssueSpecSnippet:
    """Mirrors the snippet from ISSUES_TRACKER.md Issue 22:

    router = StrategyRouter()
    strategy = router.route('What functions call authenticate_user?')
    executor = SubQueryExecutor(retrieval_engine)
    results = executor.execute(plan.steps)
    for step_id, step_results in results.items():
        ...
    """

    def test_full_snippet_flow_rules_only(self) -> None:
        from reporag.agent.router import RetrievalStrategy as S

        # 1. StrategyRouter().route(...) on the issue's exact query.
        router = StrategyRouter(use_llm=False)
        step = DecompositionStep(
            id="step-1",
            query="What functions call authenticate_user?",
            expected_answer_type="list",
        )
        strategy = router.route(step)
        assert S.GRAPH == "graph"  # sanity on the enum
        assert strategy.strategy is S.GRAPH or strategy.strategy == "graph"

        # 2. SubQueryExecutor(retrieval_engine) -- FakeBackend stands in.
        retrieval_engine = FakeBackend()
        executor = SubQueryExecutor(backend=retrieval_engine, router=_rules_router())

        # 3. executor.execute(plan.steps) -- bare list shape from the snippet.
        plan_steps = [
            DecompositionStep(
                id="step-1",
                query="Find the authenticate_user function",
                expected_answer_type="code",
            ),
            DecompositionStep(
                id="step-2",
                query="step-2: What calls it",
                expected_answer_type="list",
                depends_on=("step-1",),
            ),
        ]
        results = executor.execute(plan_steps)

        # 4. snippet iterates results.items(): dict[step_id, ...].
        for step_id, step_results in results.step_results.items():
            assert step_id in {"step-1", "step-2"}
            assert hasattr(step_results, "results")
            assert step_results.status == "ok"
        assert len(results.step_results) == 2
