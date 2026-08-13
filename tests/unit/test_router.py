"""Unit tests for the strategy router (Issue 22).

Covers every acceptance criterion of Issue 22's router portion:

* Routes identifier lookups ("where is ``X`` defined?") to ``bm25``.
* Routes structural queries ("what calls X") to ``graph``.
* Routes semantic queries ("how does X work?") to ``vector``.
* Routes ambiguous queries to ``hybrid`` (the safe default).
* LLM-assisted routing with rule-based fallback.
* Unit tests: 10+ sub-queries with expected routing decisions.

Beyond the acceptance criteria, the suite pins the design contract laid
out in the router module docstring / class docstrings:

* **Dual strategy** -- LLM primary, deterministic rule-based fallback
  when the LLM is disabled, unavailable (no API key), fails, or
  produces an unparseable / incomplete response.
* **Lazy LLM loading** -- construction does NOT load the LLM; a
  pre-injected callable is respected (the test seam).
* **Pure helpers** -- :func:`rule_based_route` and
  :func:`parse_routing_response` are tested in isolation with no LLM.
* **Confidence-gated fallback** -- below the threshold a sub-query is
  overridden to ``hybrid`` with ``fell_back=True`` and the original
  confidence preserved (per sub-query, not whole-batch).
* **Batch routing** -- a whole :class:`DecompositionPlan` is routed in
  one LLM call; per-step order is preserved regardless of the order the
  LLM returns routes.
* **Determinism** -- the rule-based router is reproducible.
* **Input validation** -- empty steps and bad thresholds raise.

A ``_FakeLLM`` (same pattern as ``test_planner.py``) stands in for the
real langchain LLM, keeping every test network-free.
"""

from __future__ import annotations

import json

import pytest

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
    _build_routing_prompt,
    _coerce_strategy,
    parse_hybrid_components,
    parse_routing_response,
    rule_based_route,
)

# ============================================================================
# Test doubles
# ============================================================================


class _FakeLLM:
    """Minimal stand-in for a langchain LLM callable.

    Returns a caller-supplied response, and records every prompt so tests
    can assert on prompt contents (e.g. that every step appears).
    """

    def __init__(
        self,
        response: str | None = None,
        *,
        raise_on_call: int | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._response = response
        self._raise_on_call = raise_on_call
        self._exc = exc or RuntimeError("simulated LLM failure")
        self.call_count = 0
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        if self._raise_on_call is not None and self.call_count >= self._raise_on_call:
            raise self._exc
        assert self._response is not None
        return self._response


def _llm_routes(
    routes: list[tuple[str, str, float]],
    *,
    wrap: str = "",
) -> str:
    """Build a JSON LLM batch-routing response string.

    Args:
        routes: A list of ``(step_id, strategy, confidence)`` tuples.
        wrap: Optional wrapping -- ``"markdown"`` for a fenced block or
            ``"prose"`` for surrounding text, to exercise tolerant parsing.
    """
    payload = json.dumps(
        {
            "routes": [
                {
                    "step_id": step_id,
                    "strategy": strategy,
                    "confidence": confidence,
                }
                for step_id, strategy, confidence in routes
            ]
        }
    )
    if wrap == "markdown":
        return f"```json\n{payload}\n```"
    if wrap == "prose":
        return f"Here are the routes:\n{payload}\nDone."
    return payload


def _step(
    query: str,
    *,
    step_id: str = "step-1",
    depends_on: tuple[str, ...] = (),
    answer_type: str = "code",
) -> DecompositionStep:
    """Build a single DecompositionStep for routing tests."""
    return DecompositionStep(
        id=step_id,
        query=query,
        expected_answer_type=answer_type,  # type: ignore[arg-type]
        depends_on=depends_on,
    )


def _plan(
    steps: list[DecompositionStep], *, query: str = "original"
) -> DecompositionPlan:
    """Wrap a list of steps in a minimal DecompositionPlan."""
    return DecompositionPlan(
        original_query=query,
        steps=tuple(steps),
        needs_decomposition=len(steps) > 1,
        classification=ClassificationResult(query_type="multi-hop", confidence=0.9),
        source="rules",
    )


# ============================================================================
# Acceptance criteria: 10+ sub-queries with expected routing decisions
# ============================================================================


class TestAcceptanceCriteria:
    """The exact routing acceptance criteria from Issue 22, tested via rules.

    The rule-based path is the deterministic source of truth for routing
    decisions, so acceptance is asserted against it.  The LLM path is
    exercised separately in :class:`TestLLMPath`.
    """

    @pytest.mark.parametrize(
        "query, expected_strategy",
        [
            # --- bm25: identifier/location lookups (4) ---
            ("Where is the authenticate_user function defined?", "bm25"),
            ("Locate the file that contains the DatabaseConfig class.", "bm25"),
            ("Show me the handle_request function definition.", "bm25"),
            ("What line is the login handler declared on?", "bm25"),
            # --- graph: structural queries (4) ---
            ("What calls the authenticate_user function?", "graph"),
            ("Who are the callers of the session manager?", "graph"),
            ("Trace the path from the login route to the session token.", "graph"),
            ("Which functions depend on the config module?", "graph"),
            # --- vector: semantic queries (3) ---
            ("How does the auth middleware validate credentials?", "vector"),
            ("Explain how the ingestion pipeline works.", "vector"),
            ("What is the purpose of the rate limiter?", "vector"),
            # --- hybrid: ambiguous / cross-cutting (2) ---
            (
                "Describe the relationship between the auth and session modules.",
                "hybrid",
            ),
            ("Give me an overview of how everything connects.", "hybrid"),
        ],
    )
    def test_rule_based_routes_each_example_correctly(
        self, query: str, expected_strategy: str
    ) -> None:
        """The rule-based path routes all 13 example queries to the expected
        strategy."""
        step = _step(query)
        result = rule_based_route(step)
        assert result.strategy == RetrievalStrategy(expected_strategy)
        assert result.source == "rules"
        assert 0.0 <= result.confidence <= 1.0

    def test_identifier_lookup_routed_to_bm25(self) -> None:
        """Acceptance: 'where is X defined?' -> bm25."""
        result = rule_based_route(_step("Where is authenticate_user defined?"))
        assert result.strategy is RetrievalStrategy.BM25

    def test_structural_query_routed_to_graph(self) -> None:
        """Acceptance: 'what calls X' -> graph."""
        result = rule_based_route(_step("What calls the login_handler function?"))
        assert result.strategy is RetrievalStrategy.GRAPH

    def test_semantic_query_routed_to_vector(self) -> None:
        """Acceptance: 'how does X work' -> vector."""
        result = rule_based_route(_step("How does the auth middleware work?"))
        assert result.strategy is RetrievalStrategy.VECTOR

    def test_ambiguous_query_routed_to_hybrid(self) -> None:
        """Acceptance: a query with no clear single signal -> hybrid."""
        result = rule_based_route(_step("Describe the relationship between modules."))
        assert result.strategy is RetrievalStrategy.HYBRID


# ============================================================================
# Rule-based router (pure function, no LLM)
# ============================================================================


class TestRuleBasedRouter:
    """The deterministic fallback router, tested in isolation."""

    @pytest.mark.parametrize(
        "query, expected_strategy",
        [
            ("Where is the authenticate function defined?", "bm25"),
            ("Locate the DatabaseConfig class.", "bm25"),
            ("Show me the handle_request function.", "bm25"),
        ],
    )
    def test_bm25_queries(self, query: str, expected_strategy: str) -> None:
        result = rule_based_route(_step(query))
        assert result.strategy == RetrievalStrategy(expected_strategy)
        assert result.source == "rules"
        assert "scores" in result.metadata
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.parametrize(
        "query",
        [
            "What calls the authenticate_user function?",
            "Who are the callers of the session manager?",
            "Trace the path from the login route to the token creator.",
            "Which modules depend on the config layer?",
        ],
    )
    def test_graph_queries(self, query: str) -> None:
        result = rule_based_route(_step(query))
        assert result.strategy is RetrievalStrategy.GRAPH
        assert result.source == "rules"

    @pytest.mark.parametrize(
        "query",
        [
            "How does the auth middleware validate credentials?",
            "Explain how the ingestion pipeline works.",
            "What is the purpose of the rate limiter module?",
        ],
    )
    def test_vector_queries(self, query: str) -> None:
        result = rule_based_route(_step(query))
        assert result.strategy is RetrievalStrategy.VECTOR
        assert result.source == "rules"

    @pytest.mark.parametrize(
        "query",
        [
            "Describe the relationship between the auth and session modules.",
            "xkcd random words with no signal at all 42",
        ],
    )
    def test_hybrid_queries(self, query: str) -> None:
        result = rule_based_route(_step(query))
        assert result.strategy is RetrievalStrategy.HYBRID
        assert result.source == "rules"

    def test_no_signal_defaults_to_hybrid_with_zero_confidence(self) -> None:
        """A query matching no patterns -> hybrid, confidence 0.0."""
        result = rule_based_route(_step("xyz random gibberish 123"))
        assert result.strategy is RetrievalStrategy.HYBRID
        assert result.confidence == 0.0
        assert result.metadata["scores"] == {
            "graph": 0,
            "vector": 0,
            "bm25": 0,
        }

    def test_confidence_is_winner_share_of_total(self) -> None:
        """Confidence = winner_votes / total_votes."""
        # "where is X defined" -> bm25=1, graph=0, vector=0 -> conf 1.0.
        result = rule_based_route(_step("Where is the foo function defined?"))
        assert result.strategy is RetrievalStrategy.BM25
        assert result.confidence == pytest.approx(1.0)

    def test_ties_resolve_to_hybrid(self) -> None:
        """When two strategies tie for the top score, the query is ambiguous
        and routes to hybrid rather than arbitrarily picking one."""
        # "explain the callers" -> vector(explain)=1 + graph(callers)=1 -> tie -> hybrid
        result = rule_based_route(_step("explain the callers of foo"))
        assert result.strategy is RetrievalStrategy.HYBRID

    def test_is_deterministic(self) -> None:
        """Two calls with the same step produce identical results."""
        step = _step("What calls the authenticate_user function?")
        r1 = rule_based_route(step)
        r2 = rule_based_route(step)
        assert r1 == r2

    def test_metadata_contains_scores(self) -> None:
        result = rule_based_route(_step("Where is foo defined?"))
        scores = result.metadata["scores"]
        assert set(scores.keys()) == {"graph", "vector", "bm25"}
        assert all(isinstance(v, int) for v in scores.values())

    def test_preserves_step_id(self) -> None:
        """The routing result echoes the step id it was asked about."""
        step = _step("Where is foo defined?", step_id="my-custom-id")
        result = rule_based_route(step)
        assert result.step_id == "my-custom-id"


# ============================================================================
# LLM response parsing (pure function, no LLM)
# ============================================================================


class TestParseRoutingResponse:
    """The LLM response parser, tested in isolation."""

    def test_valid_json_response(self) -> None:
        raw = _llm_routes([("step-1", "graph", 0.9), ("step-2", "vector", 0.8)])
        routings, error = parse_routing_response(raw, ["step-1", "step-2"])
        assert error is None
        assert routings["step-1"].strategy is RetrievalStrategy.GRAPH
        assert routings["step-1"].confidence == pytest.approx(0.9)
        assert routings["step-1"].source == "llm"
        assert routings["step-2"].strategy is RetrievalStrategy.VECTOR

    def test_json_wrapped_in_markdown_fence(self) -> None:
        raw = _llm_routes([("step-1", "bm25", 0.85)], wrap="markdown")
        routings, error = parse_routing_response(raw, ["step-1"])
        assert error is None
        assert routings["step-1"].strategy is RetrievalStrategy.BM25

    def test_json_embedded_in_prose(self) -> None:
        raw = _llm_routes([("step-1", "hybrid", 0.7)], wrap="prose")
        routings, error = parse_routing_response(raw, ["step-1"])
        assert error is None
        assert routings["step-1"].strategy is RetrievalStrategy.HYBRID

    def test_confidence_clamped_to_one(self) -> None:
        raw = _llm_routes([("step-1", "vector", 1.5)])
        routings, error = parse_routing_response(raw, ["step-1"])
        assert error is None
        assert routings["step-1"].confidence == 1.0

    def test_confidence_clamped_to_zero(self) -> None:
        raw = _llm_routes([("step-1", "vector", -0.4)])
        routings, error = parse_routing_response(raw, ["step-1"])
        assert error is None
        assert routings["step-1"].confidence == 0.0

    def test_missing_confidence_defaults_to_zero(self) -> None:
        raw = json.dumps({"routes": [{"step_id": "step-1", "strategy": "graph"}]})
        routings, error = parse_routing_response(raw, ["step-1"])
        assert error is None
        assert routings["step-1"].confidence == 0.0

    def test_strategy_synonyms_accepted(self) -> None:
        """Tolerant synonyms ('keyword' -> bm25, 'semantic' -> vector)."""
        raw = json.dumps(
            {
                "routes": [
                    {"step_id": "step-1", "strategy": "keyword", "confidence": 0.9},
                    {"step_id": "step-2", "strategy": "semantic", "confidence": 0.9},
                    {"step_id": "step-3", "strategy": "call_graph", "confidence": 0.9},
                    {"step_id": "step-4", "strategy": "fusion", "confidence": 0.9},
                ]
            }
        )
        routings, error = parse_routing_response(
            raw, ["step-1", "step-2", "step-3", "step-4"]
        )
        assert error is None
        assert routings["step-1"].strategy is RetrievalStrategy.BM25
        assert routings["step-2"].strategy is RetrievalStrategy.VECTOR
        assert routings["step-3"].strategy is RetrievalStrategy.GRAPH
        assert routings["step-4"].strategy is RetrievalStrategy.HYBRID

    def test_invalid_strategy_returns_error(self) -> None:
        raw = _llm_routes([("step-1", "banana", 0.9)])
        routings, error = parse_routing_response(raw, ["step-1"])
        assert routings == {}
        assert "invalid strategy" in error

    def test_empty_response_returns_error(self) -> None:
        routings, error = parse_routing_response("", ["step-1"])
        assert routings == {}
        assert error == "empty response"

    def test_whitespace_only_returns_error(self) -> None:
        routings, error = parse_routing_response("   \n  ", ["step-1"])
        assert routings == {}
        assert error == "empty response"

    def test_no_json_object_returns_error(self) -> None:
        routings, error = parse_routing_response(
            "I think these should go to graph.", ["step-1"]
        )
        assert routings == {}
        assert error == "no JSON object found"

    def test_missing_routes_list_returns_error(self) -> None:
        routings, error = parse_routing_response(
            json.dumps({"results": []}), ["step-1"]
        )
        assert routings == {}
        assert "missing or empty" in error

    def test_missing_expected_step_returns_error(self) -> None:
        """A response that omits an expected step id is incomplete -> error."""
        raw = _llm_routes([("step-1", "graph", 0.9)])
        routings, error = parse_routing_response(raw, ["step-1", "step-2"])
        assert routings == {}
        assert "step-2" in error

    def test_extra_unexpected_step_ignored(self) -> None:
        """A response with extra (hallucinated) step ids is accepted, extras
        are simply dropped."""
        raw = _llm_routes(
            [
                ("step-1", "graph", 0.9),
                ("step-hallucinated", "bm25", 0.5),
            ]
        )
        routings, error = parse_routing_response(raw, ["step-1"])
        assert error is None
        assert set(routings.keys()) == {"step-1"}

    def test_duplicate_step_id_returns_error(self) -> None:
        raw = json.dumps(
            {
                "routes": [
                    {"step_id": "step-1", "strategy": "graph", "confidence": 0.9},
                    {"step_id": "step-1", "strategy": "bm25", "confidence": 0.9},
                ]
            }
        )
        routings, error = parse_routing_response(raw, ["step-1"])
        assert routings == {}
        assert "duplicate" in error

    def test_strategy_case_insensitive(self) -> None:
        raw = _llm_routes([("step-1", "GRAPH", 0.9)])
        routings, error = parse_routing_response(raw, ["step-1"])
        assert error is None
        assert routings["step-1"].strategy is RetrievalStrategy.GRAPH

    def test_non_numeric_confidence_defaults_to_zero(self) -> None:
        raw = json.dumps(
            {
                "routes": [
                    {"step_id": "step-1", "strategy": "vector", "confidence": "high"}
                ]
            }
        )
        routings, error = parse_routing_response(raw, ["step-1"])
        assert error is None
        assert routings["step-1"].confidence == 0.0

    def test_raw_response_preserved(self) -> None:
        raw = _llm_routes([("step-1", "graph", 0.9)])
        routings, _ = parse_routing_response(raw, ["step-1"])
        assert routings["step-1"].raw_response == raw


# ============================================================================
# Coerce strategy helper
# ============================================================================


class TestCoerceStrategy:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("graph", RetrievalStrategy.GRAPH),
            ("GRAPH", RetrievalStrategy.GRAPH),
            ("call_graph", RetrievalStrategy.GRAPH),
            ("vector", RetrievalStrategy.VECTOR),
            ("semantic", RetrievalStrategy.VECTOR),
            ("bm25", RetrievalStrategy.BM25),
            (" keyword ", RetrievalStrategy.BM25),
            ("hybrid", RetrievalStrategy.HYBRID),
            ("fusion", RetrievalStrategy.HYBRID),
        ],
    )
    def test_known_synonyms(self, raw: str, expected: RetrievalStrategy) -> None:
        assert _coerce_strategy(raw) is expected

    def test_unknown_returns_none(self) -> None:
        assert _coerce_strategy("banana") is None

    def test_none_returns_none(self) -> None:
        assert _coerce_strategy(None) is None


# ============================================================================
# Hybrid components parsing
# ============================================================================


class TestParseHybridComponents:
    def test_default_vector_bm25(self) -> None:
        components = parse_hybrid_components("vector,bm25")
        assert components == (RetrievalStrategy.VECTOR, RetrievalStrategy.BM25)

    def test_empty_falls_back_to_conventional_mix(self) -> None:
        components = parse_hybrid_components("")
        assert components == (RetrievalStrategy.VECTOR, RetrievalStrategy.BM25)

    def test_deduplicates_tokens(self) -> None:
        components = parse_hybrid_components("vector, vector, bm25")
        assert components == (RetrievalStrategy.VECTOR, RetrievalStrategy.BM25)

    def test_drops_unknown_tokens(self) -> None:
        components = parse_hybrid_components("vector, banana, bm25")
        assert components == (RetrievalStrategy.VECTOR, RetrievalStrategy.BM25)

    def test_drops_hybrid_token(self) -> None:
        """'hybrid' is itself not a valid component -- it would be recursive."""
        components = parse_hybrid_components("hybrid, vector")
        assert components == (RetrievalStrategy.VECTOR,)

    def test_strips_whitespace(self) -> None:
        components = parse_hybrid_components(" vector , bm25 ")
        assert components == (RetrievalStrategy.VECTOR, RetrievalStrategy.BM25)

    def test_all_unknown_falls_back(self) -> None:
        components = parse_hybrid_components("banana, apple")
        assert components == (RetrievalStrategy.VECTOR, RetrievalStrategy.BM25)

    def test_preserves_order(self) -> None:
        components = parse_hybrid_components("bm25, vector")
        assert components == (RetrievalStrategy.BM25, RetrievalStrategy.VECTOR)

    def test_graph_alone(self) -> None:
        components = parse_hybrid_components("graph")
        assert components == (RetrievalStrategy.GRAPH,)


# ============================================================================
# StrategyRouter: construction, lazy loading, validation
# ============================================================================


class TestConstruction:
    def test_construction_does_not_load_llm(self) -> None:
        """Building the router must be side-effect-free (no LLM load)."""
        router = StrategyRouter()
        assert router.is_loaded is False

    def test_pre_injected_llm_is_respected(self) -> None:
        """Passing a callable is the test seam; it is used without loading."""
        fake = _FakeLLM(response=_llm_routes([("step-1", "graph", 0.9)]))
        router = StrategyRouter(fake, confidence_threshold=0.5)
        assert router.is_loaded is False
        router.route_batch([_step("What calls foo?")])
        assert router.is_loaded is True

    def test_confidence_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            StrategyRouter(confidence_threshold=1.5)
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            StrategyRouter(confidence_threshold=-0.1)

    def test_confidence_threshold_zero_is_allowed(self) -> None:
        fake = _FakeLLM(response=_llm_routes([("step-1", "vector", 0.1)]))
        router = StrategyRouter(fake, confidence_threshold=0.0)
        plan = router.route_batch([_step("How does foo work?")])
        # 0.1 >= 0.0 -> no fallback.
        assert plan.routings[0].strategy is RetrievalStrategy.VECTOR
        assert plan.routings[0].fell_back is False

    def test_empty_hybrid_components_raises(self) -> None:
        with pytest.raises(ValueError, match="hybrid_components must not be empty"):
            StrategyRouter(hybrid_components=())

    def test_repr_shows_state(self) -> None:
        router = StrategyRouter(confidence_threshold=0.7)
        text = repr(router)
        assert "StrategyRouter" in text
        assert "loaded=False" in text


# ============================================================================
# StrategyRouter: batch routing (LLM path)
# ============================================================================


class TestLLMPath:
    def test_routes_each_step_via_llm(self) -> None:
        steps = [
            _step("Where is authenticate_user defined?", step_id="step-1"),
            _step("What calls the login_handler?", step_id="step-2"),
            _step("How does the auth middleware work?", step_id="step-3"),
        ]
        fake = _FakeLLM(
            response=_llm_routes(
                [
                    ("step-1", "bm25", 0.95),
                    ("step-2", "graph", 0.92),
                    ("step-3", "vector", 0.9),
                ]
            )
        )
        router = StrategyRouter(fake, confidence_threshold=0.5)
        plan = router.route_batch(steps)

        assert plan.source == "llm"
        assert plan.fell_back is False
        assert plan.routings[0].strategy is RetrievalStrategy.BM25
        assert plan.routings[1].strategy is RetrievalStrategy.GRAPH
        assert plan.routings[2].strategy is RetrievalStrategy.VECTOR
        # Per-step order is preserved regardless of LLM return order.
        assert plan.routings[0].step_id == "step-1"
        assert plan.routings[2].step_id == "step-3"

    def test_preserves_step_order_when_llm_returns_reversed(self) -> None:
        steps = [
            _step("step one", step_id="step-1"),
            _step("step two", step_id="step-2"),
        ]
        fake = _FakeLLM(
            response=_llm_routes(
                [
                    ("step-2", "graph", 0.9),
                    ("step-1", "vector", 0.9),
                ]
            )
        )
        router = StrategyRouter(fake, confidence_threshold=0.5)
        plan = router.route_batch(steps)

        assert plan.routings[0].step_id == "step-1"
        assert plan.routings[1].step_id == "step-2"
        assert plan.routings[0].strategy is RetrievalStrategy.VECTOR

    def test_llm_called_once_per_batch(self) -> None:
        """A whole batch is routed in a single LLM call."""
        steps = [
            _step("a", step_id="step-1"),
            _step("b", step_id="step-2"),
            _step("c", step_id="step-3"),
        ]
        fake = _FakeLLM(
            response=_llm_routes(
                [
                    ("step-1", "graph", 0.9),
                    ("step-2", "vector", 0.9),
                    ("step-3", "bm25", 0.9),
                ]
            )
        )
        router = StrategyRouter(fake, confidence_threshold=0.5)
        router.route_batch(steps)
        assert fake.call_count == 1

    def test_llm_receives_few_shot_prompt(self) -> None:
        """The prompt sent to the LLM contains the few-shot examples and the
        sub-queries."""
        steps = [_step("Where is foo defined?", step_id="step-1")]
        fake = _FakeLLM(response=_llm_routes([("step-1", "bm25", 0.9)]))
        router = StrategyRouter(fake, confidence_threshold=0.5)
        router.route_batch(steps)

        assert len(fake.prompts) == 1
        prompt = fake.prompts[0]
        assert "Where is foo defined?" in prompt
        assert "graph" in prompt
        assert "vector" in prompt
        assert "bm25" in prompt
        assert "hybrid" in prompt
        assert "JSON" in prompt
        # Few-shot example should be present.
        assert "authenticate_user" in prompt

    def test_original_query_in_prompt(self) -> None:
        """The original query text is included in the prompt for context."""
        steps = [_step("sub-query A", step_id="step-1")]
        fake = _FakeLLM(response=_llm_routes([("step-1", "vector", 0.9)]))
        router = StrategyRouter(fake, confidence_threshold=0.5)
        router.route_batch(steps, original_query="HOW DOES THE WHOLE THING WORK")

        assert "HOW DOES THE WHOLE THING WORK" in fake.prompts[0]


# ============================================================================
# StrategyRouter: confidence threshold fallback (per sub-query)
# ============================================================================


class TestConfidenceFallback:
    def test_low_confidence_overrides_single_step_to_hybrid(self) -> None:
        """A single low-confidence step falls back to hybrid; other steps
        keep their LLM-assigned strategy."""
        steps = [
            _step("Where is foo defined?", step_id="step-1"),
            _step("describe ambiguous stuff", step_id="step-2"),
        ]
        fake = _FakeLLM(
            response=_llm_routes(
                [
                    ("step-1", "bm25", 0.9),
                    ("step-2", "vector", 0.3),
                ]
            )
        )
        router = StrategyRouter(fake, confidence_threshold=0.7)
        plan = router.route_batch(steps)

        assert plan.routings[0].strategy is RetrievalStrategy.BM25
        assert plan.routings[0].fell_back is False
        # step-2 below threshold -> hybrid.
        assert plan.routings[1].strategy is RetrievalStrategy.HYBRID
        assert plan.routings[1].fell_back is True
        # Original confidence preserved.
        assert plan.routings[1].confidence == pytest.approx(0.3)
        # Whole plan reports a fallback.
        assert plan.fell_back is True

    def test_high_confidence_does_not_fall_back(self) -> None:
        steps = [_step("Where is foo defined?")]
        fake = _FakeLLM(response=_llm_routes([("step-1", "bm25", 0.85)]))
        router = StrategyRouter(fake, confidence_threshold=0.7)
        plan = router.route_batch(steps)

        assert plan.routings[0].strategy is RetrievalStrategy.BM25
        assert plan.routings[0].fell_back is False
        assert plan.fell_back is False

    def test_exact_threshold_does_not_fall_back(self) -> None:
        """confidence == threshold is NOT below -> no fallback (boundary)."""
        steps = [_step("Where is foo defined?")]
        fake = _FakeLLM(response=_llm_routes([("step-1", "bm25", 0.7)]))
        router = StrategyRouter(fake, confidence_threshold=0.7)
        plan = router.route_batch(steps)

        assert plan.routings[0].strategy is RetrievalStrategy.BM25
        assert plan.routings[0].fell_back is False

    def test_zero_confidence_rule_based_is_hybrid_anyway(self) -> None:
        """The rule-based path already routes a no-signal query to hybrid with
        confidence 0.0; fell_back stays False because hybrid IS the rule-based
        answer (not a threshold override)."""
        router = StrategyRouter(use_llm=False, confidence_threshold=0.5)
        plan = router.route_batch([_step("xyz random gibberish 123")])

        assert plan.routings[0].strategy is RetrievalStrategy.HYBRID
        assert plan.routings[0].fell_back is False
        assert plan.routings[0].source == "rules"


# ============================================================================
# StrategyRouter: fallback paths
# ============================================================================


class TestFallbacks:
    def test_use_llm_false_uses_rules(self) -> None:
        router = StrategyRouter(use_llm=False, confidence_threshold=0.0)
        plan = router.route_batch(
            [_step("Where is the authenticate function defined?")]
        )
        assert plan.source == "rules"
        assert plan.routings[0].strategy is RetrievalStrategy.BM25

    def test_use_llm_false_never_loads_llm(self) -> None:
        router = StrategyRouter(use_llm=False, confidence_threshold=0.0)
        router.route_batch([_step("Where is foo defined?")])
        assert router.is_loaded is False

    def test_llm_call_failure_falls_back_to_rules(self) -> None:
        """When the LLM raises, the rule-based router is used."""
        fake = _FakeLLM(raise_on_call=1)
        router = StrategyRouter(fake, confidence_threshold=0.0)
        plan = router.route_batch(
            [_step("Where is the authenticate function defined?")]
        )
        assert plan.source == "rules"
        assert plan.routings[0].strategy is RetrievalStrategy.BM25

    def test_unparseable_llm_response_falls_back_to_rules(self) -> None:
        """When the LLM response cannot be parsed, rules are used."""
        fake = _FakeLLM(response="I think this should be graph.")
        router = StrategyRouter(fake, confidence_threshold=0.0)
        plan = router.route_batch(
            [_step("Where is the authenticate function defined?")]
        )
        assert plan.source == "rules"
        assert plan.routings[0].strategy is RetrievalStrategy.BM25

    def test_empty_llm_response_falls_back_to_rules(self) -> None:
        fake = _FakeLLM(response="")
        router = StrategyRouter(fake, confidence_threshold=0.0)
        plan = router.route_batch(
            [_step("Where is the authenticate function defined?")]
        )
        assert plan.source == "rules"
        assert plan.routings[0].strategy is RetrievalStrategy.BM25

    def test_incomplete_llm_response_falls_back_to_rules(self) -> None:
        """A response missing an expected step -> whole-batch rule fallback."""
        steps = [
            _step("a", step_id="step-1"),
            _step("b", step_id="step-2"),
        ]
        # Only step-1 is present; step-2 missing -> error -> rules.
        fake = _FakeLLM(response=_llm_routes([("step-1", "graph", 0.9)]))
        router = StrategyRouter(fake, confidence_threshold=0.0)
        plan = router.route_batch(steps)
        assert plan.source == "rules"

    def test_invalid_strategy_in_response_falls_back_to_rules(self) -> None:
        fake = _FakeLLM(response=_llm_routes([("step-1", "banana", 0.9)]))
        router = StrategyRouter(fake, confidence_threshold=0.0)
        plan = router.route_batch([_step("Where is foo defined?")])
        assert plan.source == "rules"
        assert plan.routings[0].strategy is RetrievalStrategy.BM25


# ============================================================================
# StrategyRouter: input validation
# ============================================================================


class TestInputValidation:
    def test_empty_steps_raises(self) -> None:
        router = StrategyRouter(use_llm=False)
        with pytest.raises(ValueError, match="steps must contain at least one"):
            router.route_batch([])

    def test_empty_decomposition_plan_raises(self) -> None:
        router = StrategyRouter(use_llm=False)
        empty_plan = _plan([])
        with pytest.raises(ValueError, match="steps must contain at least one"):
            router.route_batch(empty_plan)


# ============================================================================
# Prompt construction
# ============================================================================


class TestPromptConstruction:
    def test_prompt_contains_each_sub_query(self) -> None:
        steps = [
            _step("Find the entry point", step_id="step-1"),
            _step("Trace the flow", step_id="step-2"),
        ]
        prompt = _build_routing_prompt(steps, "How does it work")
        assert "Find the entry point" in prompt
        assert "Trace the flow" in prompt
        assert "How does it work" in prompt

    def test_prompt_contains_all_strategies(self) -> None:
        prompt = _build_routing_prompt([_step("q")], "orig")
        for strategy in ("graph", "vector", "bm25", "hybrid"):
            assert strategy in prompt

    def test_prompt_requests_json(self) -> None:
        prompt = _build_routing_prompt([_step("q")], "orig")
        assert "JSON" in prompt
        assert "routes" in prompt
        assert "step_id" in prompt
        assert "strategy" in prompt


# ============================================================================
# RoutingResult / RoutingPlan dataclasses
# ============================================================================


class TestRoutingResult:
    def test_default_values(self) -> None:
        result = RoutingResult(
            step_id="step-1",
            strategy=RetrievalStrategy.GRAPH,
            confidence=0.9,
        )
        assert result.fell_back is False
        assert result.source == "rules"
        assert result.raw_response == ""
        assert result.metadata == {}

    def test_is_frozen(self) -> None:
        result = RoutingResult(
            step_id="step-1", strategy=RetrievalStrategy.GRAPH, confidence=0.9
        )
        with pytest.raises(AttributeError):
            result.strategy = RetrievalStrategy.VECTOR  # type: ignore[misc]


class TestRoutingPlan:
    def test_strategy_for_existing_step(self) -> None:
        plan = RoutingPlan(
            original_query="q",
            steps=(_step("a"),),
            routings=(
                RoutingResult(
                    step_id="step-1",
                    strategy=RetrievalStrategy.GRAPH,
                    confidence=0.9,
                ),
            ),
            source="rules",
        )
        assert plan.strategy_for("step-1") is RetrievalStrategy.GRAPH

    def test_strategy_for_missing_step_raises(self) -> None:
        plan = RoutingPlan(
            original_query="q",
            steps=(_step("a"),),
            routings=(
                RoutingResult(
                    step_id="step-1",
                    strategy=RetrievalStrategy.GRAPH,
                    confidence=0.9,
                ),
            ),
            source="rules",
        )
        with pytest.raises(KeyError, match="no routing for step id"):
            plan.strategy_for("nonexistent")

    def test_per_source_metadata(self) -> None:
        steps = [_step("a", step_id="step-1"), _step("b", step_id="step-2")]
        fake = _FakeLLM(
            response=_llm_routes([("step-1", "graph", 0.9), ("step-2", "vector", 0.3)])
        )
        router = StrategyRouter(fake, confidence_threshold=0.7)
        plan = router.route_batch(steps)
        # step-2 falls back to hybrid -> still source "llm".
        assert plan.metadata["per_source"]["llm"] == 2
        assert plan.metadata["per_source"]["rules"] == 0
        assert "hybrid_components" in plan.metadata


# ============================================================================
# RetrievalStrategy enum
# ============================================================================


class TestRetrievalStrategy:
    def test_enum_values(self) -> None:
        assert RetrievalStrategy.GRAPH.value == "graph"
        assert RetrievalStrategy.VECTOR.value == "vector"
        assert RetrievalStrategy.BM25.value == "bm25"
        assert RetrievalStrategy.HYBRID.value == "hybrid"

    def test_compares_equal_to_string(self) -> None:
        """Because it subclasses str, the enum compares equal to its value."""
        assert RetrievalStrategy.GRAPH == "graph"
        assert RetrievalStrategy.VECTOR == "vector"
        assert RetrievalStrategy.BM25 == "bm25"
        assert RetrievalStrategy.HYBRID == "hybrid"


# ============================================================================
# End-to-end: router consumes a DecompositionPlan
# ============================================================================


class TestRoutingFromDecompositionPlan:
    """The router accepts a DecompositionPlan directly (the ergonomic path)."""

    def test_routes_decomposition_plan(self) -> None:
        steps = [
            _step("Where is authenticate_user defined?", step_id="step-1"),
            _step("step-2: what calls it", step_id="step-2", depends_on=("step-1",)),
        ]
        plan = _plan(steps, query="How does auth flow work end-to-end?")
        router = StrategyRouter(use_llm=False)
        routing_plan = router.route_batch(plan)

        assert routing_plan.original_query == "How does auth flow work end-to-end?"
        assert len(routing_plan.routings) == 2
        assert routing_plan.steps == tuple(steps)
        assert routing_plan.routings[0].step_id == "step-1"
        assert routing_plan.routings[1].step_id == "step-2"

    def test_single_step_route_helper(self) -> None:
        """The single-step route() convenience wrapper works."""
        router = StrategyRouter(use_llm=False)
        result = router.route(_step("Where is foo defined?"))
        assert result.strategy is RetrievalStrategy.BM25
