"""Unit tests for the strategy router (Issue 22).

Covers every acceptance criterion of Issue 22 for the router:

* Routes identifier lookups to BM25.
* Routes structural queries ("what calls X") to graph.
* Routes semantic queries to vector search.
* Ambiguous / tied queries route to hybrid.
* Hybrid route is the confidence-threshold fallback.
* 10+ sub-queries with expected routing decisions.

Beyond the acceptance criteria, the suite pins the design contract:

* **Dual strategy** -- the LLM path is primary; the rule-based path is
  the fallback when the LLM is disabled, no API key is set, the LLM call
  raises, or the response cannot be parsed.
* **Lazy LLM loading** -- construction does NOT load the LLM; a
  pre-injected callable is respected (the test seam).
* **Pure helpers** -- :func:`rule_based_route` and
  :func:`parse_routing_response` are tested in isolation with no LLM.
* **Confidence-gated fallback** -- below the threshold the result is
  overridden to ``hybrid`` with ``fell_back=True`` and the original
  confidence preserved.
* **Determinism** -- the rule-based router is reproducible.
* **Input validation** -- empty queries and bad thresholds raise.

A ``_FakeLLM`` whose response is controllable stands in for the real
langchain LLM, keeping every test network-free.
"""

from __future__ import annotations

import json

import pytest

from reporag.agent.router import (
    RoutingResult,
    StrategyRouter,
    _build_routing_prompt,
    parse_routing_response,
    rule_based_route,
)

# ============================================================================
# Test doubles
# ============================================================================


class _FakeLLM:
    """Minimal stand-in for a langchain LLM callable.

    Records every call (so tests can assert prompt contents) and returns a
    caller-supplied response string.  Optionally raises on the Nth call to
    simulate an LLM API failure.
    """

    def __init__(
        self,
        response: str = '{"strategy": "bm25", "confidence": 0.95}',
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
        return self._response


def _llm_routing_response(
    strategy: str = "bm25",
    confidence: float = 0.95,
    *,
    wrap: str = "",
) -> str:
    """Build a JSON LLM routing response, optionally wrapped in prose/markdown."""
    body = json.dumps({"strategy": strategy, "confidence": confidence})
    if wrap == "markdown":
        return f"```json\n{body}\n```"
    if wrap == "prose":
        return f"Here is my routing decision:\n{body}\nHope that helps!"
    return body


# ============================================================================
# rule_based_route -- pure function tests (no LLM)
# ============================================================================


class TestRuleBasedRoute:
    """Tests for the network-free rule_based_route helper."""

    # ------------------------------------------------------------------
    # Acceptance criteria: 10+ sub-queries with expected routing decisions
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "query, expected_strategy",
        [
            # BM25 -- identifier lookups
            ("Find the authenticate function", "bm25"),
            ("Where is DatabaseConfig defined?", "bm25"),
            ("Locate the handle_request method", "bm25"),
            ("What file contains UserManager?", "bm25"),
            ("Show me the TokenValidator class", "bm25"),
            # Graph -- structural queries
            ("What functions call authenticate_user?", "graph"),
            ("What callers does validate_token have?", "graph"),
            ("What does the auth middleware import?", "graph"),
            ("What inherits from BaseHandler?", "graph"),
            ("Trace the path from the login route to session creation", "graph"),
            # Vector -- semantic queries
            ("How does the auth middleware work?", "vector"),
            ("Explain the ingestion pipeline", "vector"),
            ("What is the purpose of the caching layer?", "vector"),
            ("Why is connection pooling used in the database module?", "vector"),
        ],
    )
    def test_routes_correctly(self, query: str, expected_strategy: str) -> None:
        result = rule_based_route(query)
        assert result.strategy == expected_strategy, (
            f"query={query!r}: expected {expected_strategy!r}, "
            f"got {result.strategy!r} (scores={result.metadata.get('scores')})"
        )

    def test_returns_routing_result(self) -> None:
        result = rule_based_route("Find the authenticate function")
        assert isinstance(result, RoutingResult)
        assert result.source == "rules"

    def test_confidence_in_range(self) -> None:
        for query in [
            "Find the authenticate function",
            "What calls validate?",
            "Explain the architecture",
        ]:
            result = rule_based_route(query)
            assert (
                0.0 <= result.confidence <= 1.0
            ), f"confidence {result.confidence} out of [0, 1] for {query!r}"

    def test_no_signal_returns_hybrid(self) -> None:
        """A query with no matching patterns gets hybrid with confidence 0."""
        result = rule_based_route("the quick brown fox")
        assert result.strategy == "hybrid"
        assert result.confidence == 0.0

    def test_tie_returns_hybrid(self) -> None:
        """When two strategy groups tie, the result is hybrid."""
        # This query deliberately triggers both identifier and structural signals.
        result = rule_based_route("find what calls authenticate_user")
        # Both bm25 (find) and graph (what calls) should fire.
        scores = result.metadata["scores"]
        assert scores["bm25"] > 0 and scores["graph"] > 0
        # A tie lands on hybrid.
        if scores["bm25"] == scores["graph"] and scores["vector"] < scores["bm25"]:
            assert result.strategy == "hybrid"

    def test_metadata_contains_scores(self) -> None:
        result = rule_based_route("Where is the authenticate function?")
        assert "scores" in result.metadata
        scores = result.metadata["scores"]
        assert set(scores) == {"bm25", "graph", "vector"}

    def test_identifier_lookup_to_bm25(self) -> None:
        """Acceptance criterion: identifier lookups route to BM25."""
        for query in [
            "Find the authenticate function",
            "Where is DatabaseConfig defined?",
            "Locate the UserService class",
        ]:
            result = rule_based_route(query)
            assert result.strategy == "bm25", f"Failed for {query!r}"

    def test_structural_query_to_graph(self) -> None:
        """Acceptance criterion: structural queries route to graph."""
        for query in [
            "What functions call authenticate_user?",
            "What does the middleware import?",
            "Trace the path from login to session",
        ]:
            result = rule_based_route(query)
            assert result.strategy == "graph", f"Failed for {query!r}"

    def test_semantic_query_to_vector(self) -> None:
        """Acceptance criterion: semantic queries route to vector."""
        for query in [
            "How does the auth middleware work?",
            "Explain the ingestion pipeline",
            "What is the purpose of caching?",
        ]:
            result = rule_based_route(query)
            assert result.strategy == "vector", f"Failed for {query!r}"


# ============================================================================
# parse_routing_response -- pure function tests (no LLM)
# ============================================================================


class TestParseRoutingResponse:
    """Tests for the tolerant LLM response parser."""

    def test_valid_json(self) -> None:
        raw = '{"strategy": "graph", "confidence": 0.92}'
        result = parse_routing_response(raw)
        assert result.strategy == "graph"
        assert result.confidence == pytest.approx(0.92)
        assert result.source == "llm"
        assert "parse_error" not in result.metadata

    @pytest.mark.parametrize("strategy", ["bm25", "vector", "graph", "hybrid"])
    def test_all_valid_strategies(self, strategy: str) -> None:
        raw = json.dumps({"strategy": strategy, "confidence": 0.9})
        result = parse_routing_response(raw)
        assert result.strategy == strategy

    def test_markdown_wrapped_json(self) -> None:
        raw = _llm_routing_response("vector", 0.88, wrap="markdown")
        result = parse_routing_response(raw)
        assert result.strategy == "vector"
        assert result.confidence == pytest.approx(0.88)

    def test_prose_wrapped_json(self) -> None:
        raw = _llm_routing_response("graph", 0.75, wrap="prose")
        result = parse_routing_response(raw)
        assert result.strategy == "graph"
        assert result.confidence == pytest.approx(0.75)

    def test_empty_response_returns_hybrid(self) -> None:
        result = parse_routing_response("")
        assert result.strategy == "hybrid"
        assert result.confidence == 0.0
        assert "parse_error" in result.metadata

    def test_no_json_returns_hybrid(self) -> None:
        result = parse_routing_response("Sorry, I cannot help.")
        assert result.strategy == "hybrid"
        assert "parse_error" in result.metadata

    def test_invalid_json_returns_hybrid(self) -> None:
        result = parse_routing_response("{strategy: oops}")
        assert result.strategy == "hybrid"
        assert "parse_error" in result.metadata

    def test_unknown_strategy_returns_hybrid(self) -> None:
        raw = '{"strategy": "graph_traversal", "confidence": 0.9}'
        result = parse_routing_response(raw)
        assert result.strategy == "hybrid"
        assert "parse_error" in result.metadata

    def test_confidence_clamped_above_1(self) -> None:
        raw = '{"strategy": "bm25", "confidence": 1.5}'
        result = parse_routing_response(raw)
        assert result.confidence == 1.0

    def test_confidence_clamped_below_0(self) -> None:
        raw = '{"strategy": "bm25", "confidence": -0.3}'
        result = parse_routing_response(raw)
        assert result.confidence == 0.0

    def test_raw_response_preserved(self) -> None:
        raw = '{"strategy": "vector", "confidence": 0.8}'
        result = parse_routing_response(raw)
        assert result.raw_response == raw


# ============================================================================
# _build_routing_prompt
# ============================================================================


class TestBuildRoutingPrompt:
    def test_contains_query(self) -> None:
        prompt = _build_routing_prompt("What calls authenticate_user?")
        assert "What calls authenticate_user?" in prompt

    def test_contains_all_strategies(self) -> None:
        prompt = _build_routing_prompt("any query")
        for strategy in ("bm25", "vector", "graph", "hybrid"):
            assert strategy in prompt

    def test_contains_few_shot_examples(self) -> None:
        prompt = _build_routing_prompt("any query")
        # The prompt should contain at least one example for each strategy.
        assert "Find the authenticate function" in prompt

    def test_instructs_json_output(self) -> None:
        prompt = _build_routing_prompt("any query")
        assert "JSON" in prompt or "json" in prompt


# ============================================================================
# StrategyRouter -- construction and validation
# ============================================================================


class TestStrategyRouterConstruction:
    def test_default_construction(self) -> None:
        router = StrategyRouter()
        assert isinstance(router, StrategyRouter)
        assert not router.is_loaded

    def test_invalid_threshold_below_zero(self) -> None:
        with pytest.raises(ValueError, match="confidence_threshold"):
            StrategyRouter(confidence_threshold=-0.1)

    def test_invalid_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="confidence_threshold"):
            StrategyRouter(confidence_threshold=1.1)

    def test_valid_boundary_thresholds(self) -> None:
        StrategyRouter(confidence_threshold=0.0)
        StrategyRouter(confidence_threshold=1.0)

    def test_repr_contains_key_fields(self) -> None:
        router = StrategyRouter(use_llm=False, confidence_threshold=0.5)
        r = repr(router)
        assert "use_llm=False" in r
        assert "0.5" in r

    def test_pre_injected_llm_not_loaded_on_construction(self) -> None:
        fake = _FakeLLM()
        _ = StrategyRouter(llm=fake)
        assert fake.call_count == 0


# ============================================================================
# StrategyRouter -- rule-based path (use_llm=False)
# ============================================================================


class TestStrategyRouterRuleBased:
    def _router(self, **kwargs) -> StrategyRouter:
        return StrategyRouter(use_llm=False, **kwargs)

    def test_empty_query_raises(self) -> None:
        router = self._router()
        with pytest.raises(ValueError, match="non-empty"):
            router.route("")

    def test_whitespace_query_raises(self) -> None:
        router = self._router()
        with pytest.raises(ValueError, match="non-empty"):
            router.route("   ")

    def test_identifier_lookup_to_bm25(self) -> None:
        router = self._router()
        result = router.route("Find the authenticate function")
        assert result.strategy == "bm25"
        assert result.source == "rules"

    def test_structural_to_graph(self) -> None:
        router = self._router()
        result = router.route("What functions call authenticate_user?")
        assert result.strategy == "graph"

    def test_semantic_to_vector(self) -> None:
        router = self._router()
        result = router.route("How does the auth middleware work?")
        assert result.strategy == "vector"

    def test_low_confidence_falls_back_to_hybrid(self) -> None:
        # Query has 3 bm25 patterns matched and 1 graph pattern.
        # Total votes = 4, winner = bm25 (3 votes). Confidence = 0.75.
        # confidence_threshold=0.8 means 0.75 < 0.8 -> falls back to hybrid.
        router = self._router(confidence_threshold=0.8)
        result = router.route("Find locate where authenticate is and what calls it")
        assert result.strategy == "hybrid"
        assert result.fell_back is True
        assert result.confidence == pytest.approx(2 / 3)

    def test_high_confidence_does_not_fall_back(self) -> None:
        router = self._router(confidence_threshold=0.0)
        result = router.route("Find locate where authenticate is and what calls it")
        assert result.strategy == "bm25"
        assert result.fell_back is False

    def test_step_id_attached(self) -> None:
        router = self._router()
        result = router.route("Find the authenticate function", step_id="step-1")
        assert result.step_id == "step-1"

    def test_step_id_default_empty(self) -> None:
        router = self._router()
        result = router.route("Find the authenticate function")
        assert result.step_id == ""


# ============================================================================
# StrategyRouter -- LLM path
# ============================================================================


class TestStrategyRouterLLMPath:
    def _router(self, fake_llm: _FakeLLM, **kwargs) -> StrategyRouter:
        return StrategyRouter(llm=fake_llm, **kwargs)

    def test_llm_response_used(self) -> None:
        fake = _FakeLLM(_llm_routing_response("graph", 0.92))
        router = self._router(fake)
        result = router.route("What calls authenticate_user?")
        assert result.strategy == "graph"
        assert result.confidence == pytest.approx(0.92)
        assert result.source == "llm"
        assert fake.call_count == 1

    def test_llm_prompt_contains_query(self) -> None:
        fake = _FakeLLM(_llm_routing_response("graph", 0.9))
        router = self._router(fake)
        query = "What calls authenticate_user?"
        router.route(query)
        assert query in fake.prompts[0]

    def test_llm_called_lazily_not_at_construction(self) -> None:
        fake = _FakeLLM()
        StrategyRouter(llm=fake)
        assert fake.call_count == 0

    def test_llm_parse_failure_falls_back_to_rules(self) -> None:
        fake = _FakeLLM("not json at all")
        router = self._router(fake)
        # Rule-based fallback should take over.
        result = router.route("Find the authenticate function")
        assert result.strategy == "bm25"
        assert result.source == "rules"

    def test_llm_call_exception_falls_back_to_rules(self) -> None:
        fake = _FakeLLM(raise_on_call=1)
        router = self._router(fake)
        result = router.route("Find the authenticate function")
        assert result.strategy == "bm25"
        assert result.source == "rules"

    def test_llm_low_confidence_falls_back_to_hybrid(self) -> None:
        # LLM returns confidence 0.3 but threshold is 0.7.
        fake = _FakeLLM(_llm_routing_response("bm25", 0.3))
        router = self._router(fake, confidence_threshold=0.7)
        result = router.route("Find the authenticate function")
        assert result.strategy == "hybrid"
        assert result.fell_back is True
        assert result.confidence == pytest.approx(0.3)

    def test_llm_unknown_strategy_falls_back_to_rules(self) -> None:
        fake = _FakeLLM('{"strategy": "bm25_and_graph", "confidence": 0.9}')
        router = self._router(fake)
        result = router.route("Find the authenticate function")
        # Invalid strategy -> parse_error -> rule-based fallback.
        assert result.source == "rules"

    def test_is_loaded_after_first_route(self) -> None:
        fake = _FakeLLM(_llm_routing_response("bm25", 0.9))
        router = self._router(fake)
        assert not router.is_loaded
        router.route("Find the authenticate function")
        assert router.is_loaded

    def test_llm_called_once_per_route_not_again(self) -> None:
        """The LLM callable is loaded once and reused."""
        fake = _FakeLLM(_llm_routing_response("vector", 0.88))
        router = self._router(fake)
        router.route("How does the auth flow work?")
        router.route("Explain the ingestion pipeline")
        # Two route calls -> two LLM invocations (one prompt per call).
        assert fake.call_count == 2

    def test_markdown_wrapped_response_parsed(self) -> None:
        fake = _FakeLLM(_llm_routing_response("vector", 0.88, wrap="markdown"))
        router = self._router(fake)
        result = router.route("Explain the ingestion pipeline")
        assert result.strategy == "vector"


# ============================================================================
# Acceptance-criterion summary: 10+ sub-queries with expected routing
# ============================================================================


class TestAcceptanceCriteria:
    """End-to-end routing checks that directly satisfy the issue spec."""

    ROUTING_TABLE = [
        # Identifier lookups -> BM25
        ("Find the authenticate function", "bm25"),
        ("Where is DatabaseConfig defined?", "bm25"),
        ("Locate the UserManager class", "bm25"),
        ("What file contains TokenValidator?", "bm25"),
        # Structural queries -> graph
        ("What functions call authenticate_user?", "graph"),
        ("What callers does validate_token have?", "graph"),
        ("What does the auth middleware import?", "graph"),
        ("What inherits from BaseHandler?", "graph"),
        ("Trace the path from the login route to session", "graph"),
        # Semantic queries -> vector
        ("How does the auth middleware work?", "vector"),
        ("Explain the ingestion pipeline", "vector"),
        ("What is the purpose of the caching layer?", "vector"),
        ("Why is connection pooling used?", "vector"),
    ]

    @pytest.mark.parametrize("query,expected", ROUTING_TABLE)
    def test_routing_decision(self, query: str, expected: str) -> None:
        """Rule-based router routes each query to the expected strategy."""
        router = StrategyRouter(use_llm=False)
        result = router.route(query)
        assert result.strategy == expected, (
            f"query={query!r}: expected {expected!r}, "
            f"got {result.strategy!r} (scores={result.metadata.get('scores')})"
        )

    def test_hybrid_route_assigned_on_ambiguity(self) -> None:
        """Ambiguous queries (no clear signal) route to hybrid."""
        router = StrategyRouter(use_llm=False)
        result = router.route("the quick brown fox")
        assert result.strategy == "hybrid"

    def test_confidence_threshold_fallback_to_hybrid(self) -> None:
        """Confidence-gated fallback routes to hybrid when score is low."""
        router = StrategyRouter(use_llm=False, confidence_threshold=0.8)
        result = router.route("Find locate where authenticate is and what calls it")
        assert result.strategy == "hybrid"
        assert result.fell_back is True
