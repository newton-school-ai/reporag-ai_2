"""Unit tests for the query classifier (Issue 20).

Covers every acceptance criterion of Issue 20:

* Classifies "where is X defined?" as ``simple-lookup``.
* Classifies "how does X work end-to-end?" as ``multi-hop``.
* Classifies "explain the architecture" as ``exploratory``.
* Returns a confidence score in ``[0, 1]``.
* Low-confidence classifications fall back to ``multi-hop``.
* 10+ example queries exercising all three categories.

Beyond the acceptance criteria, the suite pins the design contract:

* **Dual strategy** -- the LLM path is primary; the rule-based path is the
  fallback when the LLM is disabled, no API key is set, the LLM call
  raises, or the response cannot be parsed.
* **Lazy LLM loading** -- construction does NOT load the LLM; a
  pre-injected callable is respected (the test seam).
* **Pure helpers** -- :func:`rule_based_classify` and
  :func:`parse_llm_response` are tested in isolation with no LLM.
* **Confidence-gated fallback** -- below the threshold the result is
  overridden to ``multi-hop`` with ``fell_back=True`` and the original
  confidence preserved.
* **Determinism** -- the rule-based classifier is reproducible and ties
  break deterministically toward the safer (``multi-hop``) default.
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
    QueryClassifier,
    _build_classification_prompt,
    parse_llm_response,
    rule_based_classify,
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
# Rule-based classifier (pure function, no LLM)
# ============================================================================


class TestRuleBasedClassifier:
    """The deterministic fallback classifier, tested in isolation."""

    @pytest.mark.parametrize(
        "query, expected_type",
        [
            ("Where is the authenticate function defined?", "simple-lookup"),
            ("Show me the User model class.", "simple-lookup"),
            ("Find the file that contains the DatabaseConfig class.", "simple-lookup"),
            ("What line is the handle_request function on?", "simple-lookup"),
            ("Locate the declaration of the config variable.", "simple-lookup"),
        ],
    )
    def test_simple_lookup_queries(self, query: str, expected_type: str) -> None:
        result = rule_based_classify(query)
        assert result.query_type == expected_type
        assert result.source == "rules"
        assert 0.0 <= result.confidence <= 1.0
        assert "scores" in result.metadata

    @pytest.mark.parametrize(
        "query, expected_type",
        [
            ("How does the auth flow work end-to-end?", "multi-hop"),
            ("Trace the path from the login route to the session token.", "multi-hop"),
            ("What calls the authenticate_user function?", "multi-hop"),
            (
                "How does a request go from the API endpoint to the database?",
                "multi-hop",
            ),
        ],
    )
    def test_multi_hop_queries(self, query: str, expected_type: str) -> None:
        result = rule_based_classify(query)
        assert result.query_type == expected_type
        assert result.source == "rules"

    @pytest.mark.parametrize(
        "query, expected_type",
        [
            ("Explain the overall architecture of this codebase.", "exploratory"),
            (
                "Give me an overview of how the ingestion pipeline is organized.",
                "exploratory",
            ),
            (
                "What are the main components and how are they structured?",
                "exploratory",
            ),
            ("Summarize the high-level design of the project.", "exploratory"),
        ],
    )
    def test_exploratory_queries(self, query: str, expected_type: str) -> None:
        result = rule_based_classify(query)
        assert result.query_type == expected_type
        assert result.source == "rules"

    def test_no_signal_defaults_to_multi_hop_with_zero_confidence(self) -> None:
        """A query matching no patterns -> multi-hop, confidence 0.0."""
        result = rule_based_classify("xyz random gibberish 123")
        assert result.query_type == "multi-hop"
        assert result.confidence == 0.0
        assert result.metadata["scores"] == {
            "simple-lookup": 0,
            "multi-hop": 0,
            "exploratory": 0,
        }

    def test_confidence_is_winner_share_of_total(self) -> None:
        """Confidence = winner_votes / total_votes."""
        # "where is" (simple) + "flow" (multi-hop) + "overview" (exploratory)
        # -> scores: simple=1, multi=1, exploratory=1 -> tie -> multi-hop wins
        # by the deterministic tiebreak, confidence = 1/3.
        result = rule_based_classify("where is the flow overview")
        assert result.query_type == "multi-hop"
        assert result.confidence == pytest.approx(1 / 3)

    def test_ties_break_to_multi_hop_then_exploratory_then_simple(self) -> None:
        """Ties resolve deterministically toward the safer default."""
        # "defined" (simple=1) + "flow" (multi=1) -> tie -> multi-hop.
        result = rule_based_classify("defined flow")
        assert result.query_type == "multi-hop"

    def test_is_deterministic(self) -> None:
        """Two calls with the same query produce identical results."""
        query = "How does the auth flow work end-to-end?"
        r1 = rule_based_classify(query)
        r2 = rule_based_classify(query)
        assert r1 == r2

    def test_metadata_contains_scores(self) -> None:
        result = rule_based_classify("Where is foo defined?")
        scores = result.metadata["scores"]
        assert set(scores.keys()) == {"simple-lookup", "multi-hop", "exploratory"}
        assert all(isinstance(v, int) for v in scores.values())


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

    def test_zero_confidence_rule_based_falls_back(self) -> None:
        """Rule-based with no signal (confidence 0.0) -> multi-hop via threshold."""
        classifier = QueryClassifier(use_llm=False, confidence_threshold=0.5)
        result = classifier.classify("xyz random gibberish 123")

        assert result.query_type == "multi-hop"
        assert result.fell_back is True
        assert result.source == "rules"


# ============================================================================
# QueryClassifier: fallback paths
# ============================================================================


class TestFallbacks:
    def test_use_llm_false_uses_rule_based(self) -> None:
        classifier = QueryClassifier(use_llm=False, confidence_threshold=0.0)
        result = classifier.classify("Where is the authenticate function defined?")

        assert result.query_type == "simple-lookup"
        assert result.source == "rules"

    def test_use_llm_false_never_loads_llm(self) -> None:
        classifier = QueryClassifier(use_llm=False, confidence_threshold=0.0)
        classifier.classify("Where is foo defined?")
        assert classifier.is_loaded is False

    def test_llm_call_failure_falls_back_to_rules(self) -> None:
        """When the LLM raises, the rule-based classifier is used."""
        fake = _FakeLLM(raise_on_call=1)
        classifier = QueryClassifier(fake, confidence_threshold=0.0)
        result = classifier.classify("Where is the authenticate function defined?")

        assert result.source == "rules"
        assert result.query_type == "simple-lookup"

    def test_unparseable_llm_response_falls_back_to_rules(self) -> None:
        """When the LLM response cannot be parsed, rules are used."""
        fake = _FakeLLM(response="I think this is simple.")
        classifier = QueryClassifier(fake, confidence_threshold=0.0)
        result = classifier.classify("Where is the authenticate function defined?")

        assert result.source == "rules"
        assert result.query_type == "simple-lookup"

    def test_empty_llm_response_falls_back_to_rules(self) -> None:
        fake = _FakeLLM(response="")
        classifier = QueryClassifier(fake, confidence_threshold=0.0)
        result = classifier.classify("Where is the authenticate function defined?")

        assert result.source == "rules"
        assert result.query_type == "simple-lookup"

    def test_invalid_query_type_in_response_falls_back_to_rules(self) -> None:
        fake = _FakeLLM(response='{"query_type": "banana", "confidence": 0.99}')
        classifier = QueryClassifier(fake, confidence_threshold=0.0)
        result = classifier.classify("Where is the authenticate function defined?")

        assert result.source == "rules"
        assert result.query_type == "simple-lookup"


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
        assert result.source == "rules"
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

    def test_rule_based_classify_simple_lookup(self) -> None:
        """The rule-based path also satisfies the issue spec usage."""
        result = rule_based_classify("Where is the authenticate function defined?")
        assert result.query_type == "simple-lookup"
        assert 0.0 <= result.confidence <= 1.0

    def test_rule_based_classify_multi_hop(self) -> None:
        result = rule_based_classify("How does the auth flow work end-to-end?")
        assert result.query_type == "multi-hop"
        assert 0.0 <= result.confidence <= 1.0

    def test_rule_based_classify_exploratory(self) -> None:
        result = rule_based_classify("Explain the architecture of this codebase.")
        assert result.query_type == "exploratory"
        assert 0.0 <= result.confidence <= 1.0
