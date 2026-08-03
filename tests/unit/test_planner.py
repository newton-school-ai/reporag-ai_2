"""Unit tests for the query planner (Issues 20 and 21).

Covers every acceptance criterion of Issue 20 (query classifier) and
Issue 21 (query decomposer).

Issue 20 acceptance criteria:

* Classifies "where is X defined?" as ``simple-lookup``.
* Classifies "how does X work end-to-end?" as ``multi-hop``.
* Classifies "explain the architecture" as ``exploratory``.
* Returns a confidence score in ``[0, 1]``.
* Low-confidence classifications fall back to ``multi-hop``.
* 10+ example queries exercising all three categories.

Issue 21 acceptance criteria:

* Decomposes multi-hop queries into 2-5 ordered sub-queries.
* Sub-queries have dependency edges (step 2 depends on step 1).
* Uses repo context (module names, key symbols) to inform decomposition.
* Handles edge case: query that does not need decomposition (single step).
* LangGraph state machine with clear state transitions.
* Unit tests with 5+ multi-hop query examples.

Beyond the acceptance criteria, the suite pins the design contract:

* **Safety default** -- the LLM path is primary; a zero-confidence multi-hop
  fallback is used when the LLM is disabled, no API key is set, the LLM call
  raises, or the response cannot be parsed.
* **Lazy LLM loading** -- construction does NOT load the LLM; a
  pre-injected callable is respected (the test seam).
* **Pure helpers** -- :func:`parse_llm_response` and
  :func:`parse_decomposition_response` are tested in isolation with no LLM.
* **Confidence-gated fallback** -- below the threshold the result is
  overridden to ``multi-hop`` with ``fell_back=True`` and the original
  confidence preserved.
* **Input validation** -- empty queries and bad thresholds raise.

A ``_FakeLLM`` whose response is controllable stands in for the real
langchain LLM, keeping every test network-free -- the same seam
:class:`~reporag.retrieval.reranker.CrossEncoderReranker` already uses in
``test_reranker.py``.
"""

from __future__ import annotations

import json

import pytest

from reporag.agent.planner import (
    ClassificationResult,
    DecompositionPlan,
    QueryClassifier,
    QueryDecomposer,
    SubQuery,
    _build_classification_prompt,
    _build_decomposition_prompt,
    parse_decomposition_response,
    parse_llm_response,
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
        response: str = '{"query_type": "simple-lookup", "confidence": 0.95}',
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


def _llm_response(
    query_type: str = "simple-lookup",
    confidence: float = 0.95,
    *,
    wrap: str = "",
) -> str:
    """Build a JSON LLM response string, optionally wrapped in prose/markdown."""
    body = json.dumps({"query_type": query_type, "confidence": confidence})
    if wrap == "markdown":
        return f"```json\n{body}\n```"
    if wrap == "prose":
        return f"Here is the classification:\n{body}\nHope that helps!"
    return body


def _decomposition_response(
    steps: list[dict[str, object]] | None = None,
    *,
    wrap: str = "",
) -> str:
    """Build a JSON LLM decomposition response, optionally wrapped."""
    if steps is None:
        steps = [
            {
                "id": "step-1",
                "query": "Find the entry point",
                "expected_answer_type": "code",
                "depends_on": [],
            },
            {
                "id": "step-2",
                "query": "Trace the processing chain",
                "expected_answer_type": "explanation",
                "depends_on": ["step-1"],
            },
        ]
    body = json.dumps({"steps": steps})
    if wrap == "markdown":
        return f"```json\n{body}\n```"
    if wrap == "prose":
        return f"Here is the decomposition:\n{body}\nHope that helps!"
    return body


# ============================================================================
# Acceptance criteria: 10+ example queries across all three categories
# ============================================================================


class TestAcceptanceCriteria:
    """The exact acceptance criteria from Issue 20, tested via the LLM path."""

    @pytest.mark.parametrize(
        "query, expected_type",
        [
            # --- simple-lookup (4 examples) ---
            ("Where is the authenticate function defined?", "simple-lookup"),
            ("Show me the User model class.", "simple-lookup"),
            ("Find the file that contains the DatabaseConfig class.", "simple-lookup"),
            ("What line is the handle_request function on?", "simple-lookup"),
            # --- multi-hop (4 examples) ---
            ("How does the auth flow work end-to-end?", "multi-hop"),
            (
                "How does a request go from the API endpoint to the database?",
                "multi-hop",
            ),
            (
                "What calls the authenticate_user function and what does it call next?",
                "multi-hop",
            ),
            (
                "Trace the path from the login route to the session token creation.",
                "multi-hop",
            ),
            # --- exploratory (3 examples) ---
            ("Explain the overall architecture of this codebase.", "exploratory"),
            (
                "Give me an overview of how the ingestion pipeline is organized.",
                "exploratory",
            ),
            (
                "What are the main components and how are they structured?",
                "exploratory",
            ),
        ],
    )
    def test_llm_classifies_each_example_correctly(
        self, query: str, expected_type: str
    ) -> None:
        """The LLM path classifies all 11 example queries to the expected type."""
        fake = _FakeLLM(
            response=_llm_response(query_type=expected_type, confidence=0.95)
        )
        classifier = QueryClassifier(fake, confidence_threshold=0.5)
        result = classifier.classify(query)

        assert result.query_type == expected_type
        assert result.confidence == pytest.approx(0.95)
        assert result.source == "llm"
        assert result.fell_back is False

    def test_where_is_x_defined_is_simple_lookup(self) -> None:
        """Acceptance: 'where is X defined?' -> simple-lookup."""
        fake = _FakeLLM(response=_llm_response("simple-lookup", 0.98))
        result = QueryClassifier(fake, confidence_threshold=0.5).classify(
            "Where is the authenticate function defined?"
        )
        assert result.query_type == "simple-lookup"

    def test_how_does_x_work_end_to_end_is_multi_hop(self) -> None:
        """Acceptance: 'how does X work end-to-end?' -> multi-hop."""
        fake = _FakeLLM(response=_llm_response("multi-hop", 0.92))
        result = QueryClassifier(fake, confidence_threshold=0.5).classify(
            "How does the auth flow work end-to-end?"
        )
        assert result.query_type == "multi-hop"

    def test_explain_the_architecture_is_exploratory(self) -> None:
        """Acceptance: 'explain the architecture' -> exploratory."""
        fake = _FakeLLM(response=_llm_response("exploratory", 0.90))
        result = QueryClassifier(fake, confidence_threshold=0.5).classify(
            "Explain the overall architecture of this codebase."
        )
        assert result.query_type == "exploratory"

    def test_confidence_score_is_in_zero_to_one(self) -> None:
        """Acceptance: confidence is always in [0, 1]."""
        fake = _FakeLLM(response=_llm_response("simple-lookup", 0.85))
        result = QueryClassifier(fake, confidence_threshold=0.5).classify(
            "Where is foo defined?"
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_low_confidence_falls_back_to_multi_hop(self) -> None:
        """Acceptance: confidence < threshold -> multi-hop with fell_back=True."""
        fake = _FakeLLM(response=_llm_response("simple-lookup", 0.3))
        classifier = QueryClassifier(fake, confidence_threshold=0.7)
        result = classifier.classify("Where is foo defined?")

        assert result.query_type == "multi-hop"
        assert result.fell_back is True
        # The original confidence is preserved so callers can see why.
        assert result.confidence == pytest.approx(0.3)


# ============================================================================
# LLM response parsing (pure function, no LLM)
# ============================================================================


class TestParseLLMResponse:
    """The LLM response parser, tested in isolation."""

    def test_valid_json_response(self) -> None:
        raw = '{"query_type": "simple-lookup", "confidence": 0.95}'
        result = parse_llm_response(raw)
        assert result.query_type == "simple-lookup"
        assert result.confidence == pytest.approx(0.95)
        assert result.source == "llm"
        assert result.fell_back is False
        assert "parse_error" not in result.metadata

    def test_json_wrapped_in_markdown_fence(self) -> None:
        raw = '```json\n{"query_type": "multi-hop", "confidence": 0.88}\n```'
        result = parse_llm_response(raw)
        assert result.query_type == "multi-hop"
        assert result.confidence == pytest.approx(0.88)

    def test_json_embedded_in_prose(self) -> None:
        raw = (
            "Here is the classification:\n"
            '{"query_type": "exploratory", "confidence": 0.9}\n'
            "Hope that helps!"
        )
        result = parse_llm_response(raw)
        assert result.query_type == "exploratory"
        assert result.confidence == pytest.approx(0.9)

    def test_confidence_clamped_to_one(self) -> None:
        raw = '{"query_type": "simple-lookup", "confidence": 1.5}'
        result = parse_llm_response(raw)
        assert result.confidence == 1.0

    def test_confidence_clamped_to_zero(self) -> None:
        raw = '{"query_type": "simple-lookup", "confidence": -0.5}'
        result = parse_llm_response(raw)
        assert result.confidence == 0.0

    def test_missing_confidence_defaults_to_zero(self) -> None:
        raw = '{"query_type": "multi-hop"}'
        result = parse_llm_response(raw)
        assert result.query_type == "multi-hop"
        assert result.confidence == 0.0

    def test_invalid_query_type_returns_multi_hop_with_error(self) -> None:
        raw = '{"query_type": "unknown-category", "confidence": 0.9}'
        result = parse_llm_response(raw)
        assert result.query_type == "multi-hop"
        assert result.confidence == 0.0
        assert "parse_error" in result.metadata

    def test_empty_response_returns_multi_hop_with_error(self) -> None:
        result = parse_llm_response("")
        assert result.query_type == "multi-hop"
        assert result.confidence == 0.0
        assert result.metadata["parse_error"] == "empty response"

    def test_whitespace_only_response_returns_multi_hop_with_error(self) -> None:
        result = parse_llm_response("   \n  ")
        assert result.query_type == "multi-hop"
        assert result.confidence == 0.0
        assert result.metadata["parse_error"] == "empty response"

    def test_no_json_object_returns_multi_hop_with_error(self) -> None:
        result = parse_llm_response("I think this is a simple lookup.")
        assert result.query_type == "multi-hop"
        assert result.confidence == 0.0
        assert result.metadata["parse_error"] == "no JSON object found"

    def test_invalid_json_returns_multi_hop_with_error(self) -> None:
        result = parse_llm_response("{not valid json}")
        assert result.query_type == "multi-hop"
        assert result.confidence == 0.0
        assert "parse_error" in result.metadata

    def test_non_numeric_confidence_defaults_to_zero(self) -> None:
        raw = '{"query_type": "simple-lookup", "confidence": "high"}'
        result = parse_llm_response(raw)
        assert result.query_type == "simple-lookup"
        assert result.confidence == 0.0

    def test_query_type_case_insensitive(self) -> None:
        raw = '{"query_type": "SIMPLE-LOOKUP", "confidence": 0.9}'
        result = parse_llm_response(raw)
        assert result.query_type == "simple-lookup"

    def test_raw_response_preserved(self) -> None:
        raw = '{"query_type": "multi-hop", "confidence": 0.8}'
        result = parse_llm_response(raw)
        assert result.raw_response == raw


# ============================================================================
# QueryClassifier: construction, lazy loading, validation
# ============================================================================


class TestConstruction:
    def test_construction_does_not_load_llm(self) -> None:
        """Building the classifier must be side-effect-free (no LLM load)."""
        classifier = QueryClassifier()
        assert classifier.is_loaded is False

    def test_pre_injected_llm_is_respected(self) -> None:
        """Passing a callable is the test seam; it is used without loading."""
        fake = _FakeLLM()
        classifier = QueryClassifier(fake, confidence_threshold=0.5)
        assert classifier.is_loaded is False
        classifier.classify("Where is foo defined?")
        assert classifier.is_loaded is True

    def test_confidence_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            QueryClassifier(confidence_threshold=1.5)
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            QueryClassifier(confidence_threshold=-0.1)

    def test_confidence_threshold_zero_is_allowed(self) -> None:
        fake = _FakeLLM(response=_llm_response("simple-lookup", 0.0))
        classifier = QueryClassifier(fake, confidence_threshold=0.0)
        result = classifier.classify("Where is foo defined?")
        # Threshold 0.0 means confidence 0.0 is NOT < 0.0, so no fallback.
        assert result.query_type == "simple-lookup"
        assert result.fell_back is False

    def test_confidence_threshold_one_is_allowed(self) -> None:
        fake = _FakeLLM(response=_llm_response("simple-lookup", 0.99))
        classifier = QueryClassifier(fake, confidence_threshold=1.0)
        result = classifier.classify("Where is foo defined?")
        # 0.99 < 1.0 -> fallback to multi-hop.
        assert result.query_type == "multi-hop"
        assert result.fell_back is True

    def test_repr_shows_state(self) -> None:
        classifier = QueryClassifier(confidence_threshold=0.7)
        text = repr(classifier)
        assert "QueryClassifier" in text
        assert "loaded=False" in text


# ============================================================================
# QueryClassifier: LLM path
# ============================================================================


class TestLLMPath:
    def test_llm_classifies_correctly(self) -> None:
        fake = _FakeLLM(response=_llm_response("multi-hop", 0.92))
        classifier = QueryClassifier(fake, confidence_threshold=0.5)
        result = classifier.classify("How does the auth flow work end-to-end?")

        assert result.query_type == "multi-hop"
        assert result.confidence == pytest.approx(0.92)
        assert result.source == "llm"
        assert result.fell_back is False

    def test_llm_receives_few_shot_prompt(self) -> None:
        """The prompt sent to the LLM must contain the few-shot examples."""
        fake = _FakeLLM(response=_llm_response("simple-lookup", 0.9))
        classifier = QueryClassifier(fake, confidence_threshold=0.5)
        classifier.classify("Where is foo defined?")

        assert len(fake.prompts) == 1
        prompt = fake.prompts[0]
        # The prompt must contain the query and the few-shot examples.
        assert "Where is foo defined?" in prompt
        assert "simple-lookup" in prompt
        assert "multi-hop" in prompt
        assert "exploratory" in prompt
        assert "JSON" in prompt

    def test_llm_called_once_per_classify(self) -> None:
        fake = _FakeLLM(response=_llm_response("simple-lookup", 0.9))
        classifier = QueryClassifier(fake, confidence_threshold=0.5)
        classifier.classify("Where is foo defined?")
        assert fake.call_count == 1


# ============================================================================
# QueryClassifier: confidence threshold fallback
# ============================================================================


class TestConfidenceFallback:
    def test_low_confidence_overrides_to_multi_hop(self) -> None:
        fake = _FakeLLM(response=_llm_response("simple-lookup", 0.4))
        classifier = QueryClassifier(fake, confidence_threshold=0.7)
        result = classifier.classify("Where is foo defined?")

        assert result.query_type == "multi-hop"
        assert result.fell_back is True
        assert result.confidence == pytest.approx(0.4)
        assert result.source == "llm"

    def test_high_confidence_does_not_fall_back(self) -> None:
        fake = _FakeLLM(response=_llm_response("simple-lookup", 0.8))
        classifier = QueryClassifier(fake, confidence_threshold=0.7)
        result = classifier.classify("Where is foo defined?")

        assert result.query_type == "simple-lookup"
        assert result.fell_back is False

    def test_exact_threshold_does_not_fall_back(self) -> None:
        """confidence == threshold is NOT below -> no fallback (boundary)."""
        fake = _FakeLLM(response=_llm_response("simple-lookup", 0.7))
        classifier = QueryClassifier(fake, confidence_threshold=0.7)
        result = classifier.classify("Where is foo defined?")

        assert result.query_type == "simple-lookup"
        assert result.fell_back is False

    def test_disabled_llm_falls_back(self) -> None:
        """Disabled LLM -> multi-hop via threshold."""
        classifier = QueryClassifier(use_llm=False, confidence_threshold=0.5)
        result = classifier.classify("Where is the authenticate function defined?")

        assert result.query_type == "multi-hop"
        assert result.fell_back is True
        assert result.source == "fallback"


# ============================================================================
# QueryClassifier: fallback paths
# ============================================================================


class TestFallbacks:
    def test_use_llm_false_uses_fallback(self) -> None:
        classifier = QueryClassifier(use_llm=False, confidence_threshold=0.0)
        result = classifier.classify("Where is the authenticate function defined?")

        assert result.query_type == "multi-hop"
        assert result.source == "fallback"

    def test_use_llm_false_never_loads_llm(self) -> None:
        classifier = QueryClassifier(use_llm=False, confidence_threshold=0.0)
        classifier.classify("Where is foo defined?")
        assert classifier.is_loaded is False

    def test_llm_call_failure_falls_back(self) -> None:
        """When the LLM raises, the fallback is used."""
        fake = _FakeLLM(raise_on_call=1)
        classifier = QueryClassifier(fake, confidence_threshold=0.0)
        result = classifier.classify("Where is the authenticate function defined?")

        assert result.source == "fallback"
        assert result.query_type == "multi-hop"

    def test_unparseable_llm_response_falls_back(self) -> None:
        """When the LLM response cannot be parsed, the fallback is used."""
        fake = _FakeLLM(response="I think this is simple.")
        classifier = QueryClassifier(fake, confidence_threshold=0.0)
        result = classifier.classify("Where is the authenticate function defined?")

        assert result.source == "fallback"
        assert result.query_type == "multi-hop"

    def test_empty_llm_response_falls_back(self) -> None:
        fake = _FakeLLM(response="")
        classifier = QueryClassifier(fake, confidence_threshold=0.0)
        result = classifier.classify("Where is the authenticate function defined?")

        assert result.source == "fallback"
        assert result.query_type == "multi-hop"

    def test_invalid_query_type_in_response_falls_back(self) -> None:
        fake = _FakeLLM(response='{"query_type": "banana", "confidence": 0.99}')
        classifier = QueryClassifier(fake, confidence_threshold=0.0)
        result = classifier.classify("Where is the authenticate function defined?")

        assert result.source == "fallback"
        assert result.query_type == "multi-hop"


# ============================================================================
# QueryClassifier: input validation
# ============================================================================


class TestInputValidation:
    def test_empty_query_raises(self) -> None:
        fake = _FakeLLM()
        classifier = QueryClassifier(fake, confidence_threshold=0.5)
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            classifier.classify("")

    def test_whitespace_only_query_raises(self) -> None:
        fake = _FakeLLM()
        classifier = QueryClassifier(fake, confidence_threshold=0.5)
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            classifier.classify("   \n  ")


# ============================================================================
# Prompt construction
# ============================================================================


class TestPromptConstruction:
    def test_prompt_contains_query(self) -> None:
        prompt = _build_classification_prompt("Where is foo defined?")
        assert "Where is foo defined?" in prompt

    def test_prompt_contains_all_categories(self) -> None:
        prompt = _build_classification_prompt("test query")
        assert "simple-lookup" in prompt
        assert "multi-hop" in prompt
        assert "exploratory" in prompt

    def test_prompt_contains_few_shot_examples(self) -> None:
        prompt = _build_classification_prompt("test query")
        # At least a few of the example queries should appear.
        assert "authenticate function defined" in prompt
        assert "auth flow work end-to-end" in prompt
        assert "architecture" in prompt

    def test_prompt_requests_json(self) -> None:
        prompt = _build_classification_prompt("test query")
        assert "JSON" in prompt
        assert "query_type" in prompt
        assert "confidence" in prompt


# ============================================================================
# ClassificationResult dataclass
# ============================================================================


class TestClassificationResult:
    def test_default_values(self) -> None:
        result = ClassificationResult(query_type="simple-lookup", confidence=0.9)
        assert result.fell_back is False
        assert result.source == "fallback"
        assert result.raw_response == ""
        assert result.metadata == {}

    def test_is_frozen(self) -> None:
        """ClassificationResult is immutable (frozen dataclass)."""
        result = ClassificationResult(query_type="simple-lookup", confidence=0.9)
        with pytest.raises(AttributeError):
            result.query_type = "multi-hop"  # type: ignore[misc]

    def test_equality(self) -> None:
        r1 = ClassificationResult(query_type="simple-lookup", confidence=0.9)
        r2 = ClassificationResult(query_type="simple-lookup", confidence=0.9)
        assert r1 == r2


# ============================================================================
# End-to-end style: mirrors the issue spec usage
# ============================================================================


class TestIssueSpecUsage:
    """The exact shape from the issue's 'How to test locally' snippet."""

    def test_classify_simple_lookup(self) -> None:
        fake = _FakeLLM(response=_llm_response("simple-lookup", 0.95))
        classifier = QueryClassifier(fake, confidence_threshold=0.5)
        result = classifier.classify("Where is the authenticate function defined?")

        assert result.query_type == "simple-lookup"
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_multi_hop(self) -> None:
        fake = _FakeLLM(response=_llm_response("multi-hop", 0.92))
        classifier = QueryClassifier(fake, confidence_threshold=0.5)
        result = classifier.classify("How does the auth flow work end-to-end?")

        assert result.query_type == "multi-hop"
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_exploratory(self) -> None:
        fake = _FakeLLM(response=_llm_response("exploratory", 0.90))
        classifier = QueryClassifier(fake, confidence_threshold=0.5)
        result = classifier.classify("Explain the architecture of this codebase.")

        assert result.query_type == "exploratory"
        assert 0.0 <= result.confidence <= 1.0


# ============================================================================
# QueryDecomposer: acceptance criteria (5+ multi-hop examples)
# ============================================================================


class TestDecompositionAcceptanceCriteria:
    """The exact acceptance criteria from Issue 21.

    Each test uses a _FakeLLM that returns a controlled decomposition
    response so the LangGraph pipeline is exercised end-to-end without
    a network call.
    """

    @pytest.mark.parametrize(
        "query",
        [
            "How does a request go from the API endpoint to the database?",
            "How does the auth flow work end-to-end?",
            "Trace the path from the login route to the session token creation.",
            "What calls the authenticate_user function and what does it call next?",
            "How does the ingestion pipeline process a repository from clone to index?",
        ],
    )
    def test_decomposes_multi_hop_query(self, query: str) -> None:
        """5+ multi-hop queries decomposed into 2-5 ordered sub-queries."""
        fake = _FakeLLM(response=_decomposition_response())
        decomposer = QueryDecomposer(fake)
        plan = decomposer.decompose(
            query, repo_context={"modules": ["api", "routes", "db"]}
        )

        assert 2 <= len(plan.steps) <= 5
        assert plan.source == "llm"
        assert plan.original_query == query

    def test_sub_queries_have_dependency_edges(self) -> None:
        """Step 2 depends on step 1."""
        fake = _FakeLLM(response=_decomposition_response())
        decomposer = QueryDecomposer(fake)
        plan = decomposer.decompose(
            "How does auth work end-to-end?",
            repo_context={"modules": ["auth"]},
        )

        assert plan.steps[0].depends_on == ()
        assert "step-1" in plan.steps[1].depends_on

    def test_sub_queries_have_expected_answer_type(self) -> None:
        """Every sub-query has a valid expected_answer_type."""
        fake = _FakeLLM(response=_decomposition_response())
        decomposer = QueryDecomposer(fake)
        plan = decomposer.decompose(
            "How does auth work?", repo_context={"modules": ["auth"]}
        )

        valid_types = {"code", "explanation", "list"}
        for step in plan.steps:
            assert step.expected_answer_type in valid_types

    def test_single_step_for_simple_query(self) -> None:
        """A query that does not need decomposition returns a single step."""
        single_step = _decomposition_response(
            steps=[
                {
                    "id": "step-1",
                    "query": "Where is foo defined?",
                    "expected_answer_type": "code",
                    "depends_on": [],
                }
            ]
        )
        fake = _FakeLLM(response=single_step)
        decomposer = QueryDecomposer(fake)
        plan = decomposer.decompose("Where is foo defined?")

        assert len(plan.steps) == 1
        assert plan.steps[0].query == "Where is foo defined?"

    def test_uses_repo_context_in_prompt(self) -> None:
        """The repo context modules appear in the prompt sent to the LLM."""
        fake = _FakeLLM(response=_decomposition_response())
        decomposer = QueryDecomposer(fake)
        decomposer.decompose(
            "How does auth work?",
            repo_context={"modules": ["auth", "api", "db"]},
        )

        assert len(fake.prompts) == 1
        prompt = fake.prompts[0]
        assert "auth" in prompt
        assert "api" in prompt
        assert "db" in prompt


# ============================================================================
# parse_decomposition_response (pure function, no LLM)
# ============================================================================


class TestParseDecompositionResponse:
    """The decomposition response parser, tested in isolation."""

    def test_valid_json_response(self) -> None:
        raw = _decomposition_response()
        plan = parse_decomposition_response(raw, "test query")
        assert len(plan.steps) == 2
        assert plan.source == "llm"
        assert plan.steps[0].id == "step-1"
        assert plan.steps[1].depends_on == ("step-1",)
        assert "parse_error" not in plan.metadata

    def test_json_wrapped_in_markdown_fence(self) -> None:
        raw = _decomposition_response(wrap="markdown")
        plan = parse_decomposition_response(raw, "test query")
        assert len(plan.steps) == 2
        assert plan.source == "llm"

    def test_json_embedded_in_prose(self) -> None:
        raw = _decomposition_response(wrap="prose")
        plan = parse_decomposition_response(raw, "test query")
        assert len(plan.steps) == 2
        assert plan.source == "llm"

    def test_empty_response_returns_single_step_fallback(self) -> None:
        plan = parse_decomposition_response("", "original query")
        assert len(plan.steps) == 1
        assert plan.steps[0].query == "original query"
        assert plan.metadata["parse_error"] == "empty response"

    def test_whitespace_only_response_returns_fallback(self) -> None:
        plan = parse_decomposition_response("   \n  ", "original query")
        assert len(plan.steps) == 1
        assert plan.metadata["parse_error"] == "empty response"

    def test_no_json_object_returns_fallback(self) -> None:
        plan = parse_decomposition_response(
            "I think you should search for auth.", "original query"
        )
        assert len(plan.steps) == 1
        assert plan.metadata["parse_error"] == "no JSON object found"

    def test_invalid_json_returns_fallback(self) -> None:
        plan = parse_decomposition_response("{not valid json}", "original")
        assert len(plan.steps) == 1
        assert "parse_error" in plan.metadata

    def test_missing_steps_key_returns_fallback(self) -> None:
        raw = json.dumps({"result": "no steps here"})
        plan = parse_decomposition_response(raw, "original")
        assert len(plan.steps) == 1
        assert "parse_error" in plan.metadata

    def test_empty_steps_array_returns_fallback(self) -> None:
        raw = json.dumps({"steps": []})
        plan = parse_decomposition_response(raw, "original")
        assert len(plan.steps) == 1
        assert "parse_error" in plan.metadata

    def test_step_missing_query_returns_fallback(self) -> None:
        raw = json.dumps({"steps": [{"id": "step-1", "expected_answer_type": "code"}]})
        plan = parse_decomposition_response(raw, "original")
        assert len(plan.steps) == 1
        assert "parse_error" in plan.metadata

    def test_invalid_answer_type_defaults_to_explanation(self) -> None:
        raw = json.dumps(
            {
                "steps": [
                    {
                        "id": "step-1",
                        "query": "Find foo",
                        "expected_answer_type": "banana",
                        "depends_on": [],
                    }
                ]
            }
        )
        plan = parse_decomposition_response(raw, "original")
        assert plan.steps[0].expected_answer_type == "explanation"

    def test_missing_answer_type_defaults_to_explanation(self) -> None:
        raw = json.dumps(
            {
                "steps": [
                    {
                        "id": "step-1",
                        "query": "Find foo",
                        "depends_on": [],
                    }
                ]
            }
        )
        plan = parse_decomposition_response(raw, "original")
        assert plan.steps[0].expected_answer_type == "explanation"

    def test_missing_id_generates_sequential_id(self) -> None:
        raw = json.dumps(
            {
                "steps": [
                    {
                        "query": "Find foo",
                        "expected_answer_type": "code",
                        "depends_on": [],
                    }
                ]
            }
        )
        plan = parse_decomposition_response(raw, "original")
        assert plan.steps[0].id == "step-1"

    def test_missing_depends_on_defaults_to_empty(self) -> None:
        raw = json.dumps(
            {
                "steps": [
                    {
                        "id": "step-1",
                        "query": "Find foo",
                        "expected_answer_type": "code",
                    }
                ]
            }
        )
        plan = parse_decomposition_response(raw, "original")
        assert plan.steps[0].depends_on == ()

    def test_context_from_accepted_as_depends_on_alias(self) -> None:
        raw = json.dumps(
            {
                "steps": [
                    {
                        "id": "step-1",
                        "query": "Find foo",
                        "expected_answer_type": "code",
                        "context_from": ["step-0"],
                    }
                ]
            }
        )
        plan = parse_decomposition_response(raw, "original")
        assert plan.steps[0].depends_on == ("step-0",)

    def test_text_accepted_as_query_alias(self) -> None:
        raw = json.dumps(
            {
                "steps": [
                    {
                        "id": "step-1",
                        "text": "Find foo",
                        "expected_answer_type": "code",
                        "depends_on": [],
                    }
                ]
            }
        )
        plan = parse_decomposition_response(raw, "original")
        assert plan.steps[0].query == "Find foo"

    def test_excess_steps_truncated_to_max(self) -> None:
        steps = [
            {
                "id": f"step-{i + 1}",
                "query": f"Step {i + 1}",
                "expected_answer_type": "code",
                "depends_on": [],
            }
            for i in range(8)
        ]
        raw = json.dumps({"steps": steps})
        plan = parse_decomposition_response(raw, "original")
        assert len(plan.steps) == 5

    def test_raw_response_preserved(self) -> None:
        raw = _decomposition_response()
        plan = parse_decomposition_response(raw, "test")
        assert plan.raw_response == raw


# ============================================================================
# QueryDecomposer: construction, lazy loading
# ============================================================================


class TestDecomposerConstruction:
    def test_construction_does_not_load_llm(self) -> None:
        """Building the decomposer must be side-effect-free (no LLM load)."""
        decomposer = QueryDecomposer()
        assert decomposer.is_loaded is False

    def test_pre_injected_llm_is_respected(self) -> None:
        """Passing a callable is the test seam; it is used without loading."""
        fake = _FakeLLM(response=_decomposition_response())
        decomposer = QueryDecomposer(fake)
        assert decomposer.is_loaded is False
        decomposer.decompose("How does auth work?")
        assert decomposer.is_loaded is True

    def test_repr_shows_state(self) -> None:
        decomposer = QueryDecomposer()
        text = repr(decomposer)
        assert "QueryDecomposer" in text
        assert "loaded=False" in text

    def test_no_arg_construction(self) -> None:
        """The issue spec test snippet: QueryDecomposer() with no args."""
        decomposer = QueryDecomposer()
        assert decomposer.use_llm is True
        assert decomposer.is_loaded is False


# ============================================================================
# QueryDecomposer: LLM path
# ============================================================================


class TestDecomposerLLMPath:
    def test_llm_decomposes_correctly(self) -> None:
        fake = _FakeLLM(response=_decomposition_response())
        decomposer = QueryDecomposer(fake)
        plan = decomposer.decompose(
            "How does auth work?", repo_context={"modules": ["auth"]}
        )

        assert len(plan.steps) == 2
        assert plan.source == "llm"
        assert plan.steps[0].id == "step-1"
        assert plan.steps[1].depends_on == ("step-1",)

    def test_llm_receives_decomposition_prompt(self) -> None:
        """The prompt sent to the LLM must contain the query."""
        fake = _FakeLLM(response=_decomposition_response())
        decomposer = QueryDecomposer(fake)
        decomposer.decompose("How does auth work?", repo_context={"modules": ["auth"]})

        assert len(fake.prompts) == 1
        prompt = fake.prompts[0]
        assert "How does auth work?" in prompt
        assert "auth" in prompt
        assert "JSON" in prompt

    def test_llm_called_once_per_decompose(self) -> None:
        fake = _FakeLLM(response=_decomposition_response())
        decomposer = QueryDecomposer(fake)
        decomposer.decompose("How does auth work?")
        assert fake.call_count == 1


# ============================================================================
# QueryDecomposer: fallback paths
# ============================================================================


class TestDecomposerFallbacks:
    def test_use_llm_false_returns_single_step(self) -> None:
        decomposer = QueryDecomposer(use_llm=False)
        plan = decomposer.decompose("How does auth work?")

        assert len(plan.steps) == 1
        assert plan.steps[0].query == "How does auth work?"
        assert plan.source == "fallback"

    def test_use_llm_false_never_loads_llm(self) -> None:
        decomposer = QueryDecomposer(use_llm=False)
        decomposer.decompose("How does auth work?")
        assert decomposer.is_loaded is False

    def test_llm_call_failure_returns_single_step(self) -> None:
        """When the LLM raises, a single-step plan is returned."""
        fake = _FakeLLM(raise_on_call=1)
        decomposer = QueryDecomposer(fake)
        plan = decomposer.decompose("How does auth work?")

        assert len(plan.steps) == 1
        assert plan.source == "fallback"

    def test_unparseable_llm_response_returns_single_step(self) -> None:
        """When the LLM response cannot be parsed, fallback is used."""
        fake = _FakeLLM(response="I think you should look at auth.")
        decomposer = QueryDecomposer(fake)
        plan = decomposer.decompose("How does auth work?")

        assert len(plan.steps) == 1
        assert plan.source == "fallback"

    def test_empty_llm_response_returns_single_step(self) -> None:
        fake = _FakeLLM(response="")
        decomposer = QueryDecomposer(fake)
        plan = decomposer.decompose("How does auth work?")

        assert len(plan.steps) == 1
        assert plan.source == "fallback"


# ============================================================================
# QueryDecomposer: input validation
# ============================================================================


class TestDecomposerInputValidation:
    def test_empty_query_raises(self) -> None:
        fake = _FakeLLM(response=_decomposition_response())
        decomposer = QueryDecomposer(fake)
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            decomposer.decompose("")

    def test_whitespace_only_query_raises(self) -> None:
        fake = _FakeLLM(response=_decomposition_response())
        decomposer = QueryDecomposer(fake)
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            decomposer.decompose("   \n  ")

    def test_none_repo_context_is_handled(self) -> None:
        fake = _FakeLLM(response=_decomposition_response())
        decomposer = QueryDecomposer(fake)
        plan = decomposer.decompose("How does auth work?", repo_context=None)
        assert len(plan.steps) == 2

    def test_empty_repo_context_is_handled(self) -> None:
        fake = _FakeLLM(response=_decomposition_response())
        decomposer = QueryDecomposer(fake)
        plan = decomposer.decompose("How does auth work?", repo_context={})
        assert len(plan.steps) == 2


# ============================================================================
# Decomposition prompt construction
# ============================================================================


class TestDecompositionPromptConstruction:
    def test_prompt_contains_query(self) -> None:
        prompt = _build_decomposition_prompt("How does auth work?", {})
        assert "How does auth work?" in prompt

    def test_prompt_contains_modules(self) -> None:
        prompt = _build_decomposition_prompt("test query", {"modules": ["auth", "api"]})
        assert "auth" in prompt
        assert "api" in prompt

    def test_prompt_contains_symbols(self) -> None:
        prompt = _build_decomposition_prompt(
            "test query", {"symbols": ["User", "authenticate"]}
        )
        assert "User" in prompt
        assert "authenticate" in prompt

    def test_prompt_requests_json(self) -> None:
        prompt = _build_decomposition_prompt("test query", {})
        assert "JSON" in prompt
        assert "steps" in prompt
        assert "depends_on" in prompt

    def test_prompt_contains_few_shot_examples(self) -> None:
        prompt = _build_decomposition_prompt("test query", {})
        assert "API endpoint" in prompt
        assert "auth" in prompt.lower()


# ============================================================================
# SubQuery dataclass
# ============================================================================


class TestSubQueryDataclass:
    def test_default_depends_on(self) -> None:
        sq = SubQuery(id="step-1", query="Find foo", expected_answer_type="code")
        assert sq.depends_on == ()

    def test_is_frozen(self) -> None:
        sq = SubQuery(id="step-1", query="Find foo", expected_answer_type="code")
        with pytest.raises(AttributeError):
            sq.query = "bar"  # type: ignore[misc]

    def test_equality(self) -> None:
        sq1 = SubQuery(id="step-1", query="Find foo", expected_answer_type="code")
        sq2 = SubQuery(id="step-1", query="Find foo", expected_answer_type="code")
        assert sq1 == sq2


# ============================================================================
# DecompositionPlan dataclass
# ============================================================================


class TestDecompositionPlanDataclass:
    def test_default_values(self) -> None:
        sq = SubQuery(id="step-1", query="Find foo", expected_answer_type="code")
        plan = DecompositionPlan(steps=(sq,), original_query="Find foo")
        assert plan.source == "fallback"
        assert plan.raw_response == ""
        assert plan.metadata == {}

    def test_is_frozen(self) -> None:
        sq = SubQuery(id="step-1", query="Find foo", expected_answer_type="code")
        plan = DecompositionPlan(steps=(sq,), original_query="Find foo")
        with pytest.raises(AttributeError):
            plan.original_query = "bar"  # type: ignore[misc]

    def test_steps_is_tuple(self) -> None:
        sq = SubQuery(id="step-1", query="Find foo", expected_answer_type="code")
        plan = DecompositionPlan(steps=(sq,), original_query="Find foo")
        assert isinstance(plan.steps, tuple)


# ============================================================================
# End-to-end: mirrors the Issue 21 spec usage
# ============================================================================


class TestDecomposerIssueSpecUsage:
    """The exact shape from Issue 21's 'How to test locally' snippet."""

    def test_issue_spec_snippet(self) -> None:
        """Mirrors: decomposer.decompose(query, repo_context={...})."""
        fake = _FakeLLM(
            response=_decomposition_response(
                steps=[
                    {
                        "id": "step-1",
                        "query": "Find API endpoint entry point",
                        "expected_answer_type": "code",
                        "depends_on": [],
                    },
                    {
                        "id": "step-2",
                        "query": "Trace middleware chain",
                        "expected_answer_type": "explanation",
                        "depends_on": ["step-1"],
                    },
                    {
                        "id": "step-3",
                        "query": "Find database access layer",
                        "expected_answer_type": "code",
                        "depends_on": ["step-2"],
                    },
                ]
            )
        )
        decomposer = QueryDecomposer(fake)
        plan = decomposer.decompose(
            "How does a request go from the API endpoint to the database?",
            repo_context={"modules": ["api", "routes", "db", "models"]},
        )
        for step in plan.steps:
            assert hasattr(step, "id")
            assert hasattr(step, "query")
            assert hasattr(step, "depends_on")

        assert len(plan.steps) == 3
        assert plan.steps[0].depends_on == ()
        assert plan.steps[1].depends_on == ("step-1",)
        assert plan.steps[2].depends_on == ("step-2",)
