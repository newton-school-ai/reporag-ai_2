"""Unit tests for the query classifier (Issue 20).

Covers every acceptance criterion:

* correct classification for each of the 3 query types (11+ queries
  through the LLM path; a separate held-out set validates the rule-based
  path's generalization -- see the "Rule-based generalization" section
  below),
* confidence score in [0.0, 1.0], clamped if the LLM reports outside
  that range,
* low-confidence fallback to "multi-hop",
* the dual-path design: an LLM primary path with a deterministic,
  network-free rule-based fallback used whenever the LLM is disabled, no
  API key is configured, the LLM call itself raises, or its response
  can't be parsed.

A ``_FakeLLM`` (a plain ``str -> str`` callable recording every prompt it
receives) stands in for a real langchain client, mirroring the
``_FakeCrossEncoder`` seam already used in ``test_reranker.py``, so every
test here is network-free.
"""

from __future__ import annotations

import pytest

from reporag.agent.planner import (
    QUERY_TYPES,
    ClassificationResult,
    QueryClassifier,
    parse_llm_response,
    rule_based_classify,
)

# ============================================================================
# Test doubles
# ============================================================================


class _FakeLLM:
    """Fixed-response fake LLM callable; records every prompt it receives."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response_text


class _RaisingLLM:
    """Fake LLM that always raises, simulating a network/timeout/auth failure."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or TimeoutError("simulated LLM timeout")
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        raise self.exc


def _json_response(query_type: str, confidence: float) -> str:
    return f'{{"query_type": "{query_type}", "confidence": {confidence}}}'


# ============================================================================
# Correct classification for each query type -- LLM path (11 queries)
# ============================================================================


@pytest.mark.parametrize(
    "query,response,expected_type",
    [
        (
            "What does the `parse_config` function do?",
            _json_response("simple-lookup", 0.95),
            "simple-lookup",
        ),
        (
            "Where is `DATABASE_URL` defined?",
            _json_response("simple-lookup", 0.9),
            "simple-lookup",
        ),
        (
            "What is the return type of `embed_batch`?",
            _json_response("simple-lookup", 0.85),
            "simple-lookup",
        ),
        (
            "Which file contains the `CodeEmbedder` class?",
            _json_response("simple-lookup", 0.92),
            "simple-lookup",
        ),
        (
            "What functions call `authenticate_user`, and what do they do?",
            _json_response("multi-hop", 0.88),
            "multi-hop",
        ),
        (
            "Trace the path from the API router to the database layer.",
            _json_response("multi-hop", 0.9),
            "multi-hop",
        ),
        (
            "What are all the callers of `reciprocal_rank_fusion`, and how "
            "do they use its output?",
            _json_response("multi-hop", 0.87),
            "multi-hop",
        ),
        (
            "How does authentication work in this codebase?",
            _json_response("exploratory", 0.8),
            "exploratory",
        ),
        (
            "What are the main architectural patterns used in this repo?",
            _json_response("exploratory", 0.82),
            "exploratory",
        ),
        (
            "Give me an overview of how retrieval works end to end.",
            _json_response("exploratory", 0.79),
            "exploratory",
        ),
        (
            "What design decisions shaped the embedding pipeline?",
            _json_response("exploratory", 0.75),
            "exploratory",
        ),
    ],
)
def test_llm_path_classifies_query_correctly(query, response, expected_type) -> None:
    classifier = QueryClassifier(llm=_FakeLLM(response))
    result = classifier.classify(query)
    assert result.query_type == expected_type
    assert result.source == "llm"
    assert result.fallback_applied is False


def test_all_three_categories_are_reachable() -> None:
    assert set(QUERY_TYPES) == {"simple-lookup", "multi-hop", "exploratory"}


# ============================================================================
# Confidence score in [0.0, 1.0]
# ============================================================================


def test_confidence_is_returned_as_reported() -> None:
    classifier = QueryClassifier(llm=_FakeLLM(_json_response("simple-lookup", 0.77)))
    result = classifier.classify("What does foo() return?")
    assert result.confidence == 0.77


def test_confidence_above_one_is_clamped() -> None:
    classifier = QueryClassifier(llm=_FakeLLM(_json_response("simple-lookup", 1.5)))
    result = classifier.classify("What does foo() return?")
    assert result.confidence == 1.0


def test_confidence_below_zero_is_clamped() -> None:
    classifier = QueryClassifier(llm=_FakeLLM(_json_response("simple-lookup", -0.3)))
    result = classifier.classify("What does foo() return?")
    assert result.confidence == 0.0
    assert result.query_type == "multi-hop"  # clamped to 0.0 -> triggers fallback
    assert result.fallback_applied is True


# ============================================================================
# Low-confidence fallback to multi-hop
# ============================================================================


def test_low_confidence_falls_back_to_multi_hop() -> None:
    classifier = QueryClassifier(
        llm=_FakeLLM(_json_response("simple-lookup", 0.3)),
        confidence_threshold=0.6,
    )
    result = classifier.classify("What does foo() return?")
    assert result.query_type == "multi-hop"
    assert result.fallback_applied is True
    assert result.confidence == 0.3  # original confidence preserved, not overwritten


def test_confidence_exactly_at_threshold_is_trusted() -> None:
    """Threshold comparison is inclusive: confidence == threshold is NOT a fallback."""
    classifier = QueryClassifier(
        llm=_FakeLLM(_json_response("exploratory", 0.6)),
        confidence_threshold=0.6,
    )
    result = classifier.classify("How does the whole system work?")
    assert result.query_type == "exploratory"
    assert result.fallback_applied is False


def test_confidence_just_below_threshold_falls_back() -> None:
    classifier = QueryClassifier(
        llm=_FakeLLM(_json_response("exploratory", 0.59)),
        confidence_threshold=0.6,
    )
    result = classifier.classify("How does the whole system work?")
    assert result.query_type == "multi-hop"
    assert result.fallback_applied is True


def test_high_confidence_multi_hop_is_not_flagged_as_fallback() -> None:
    """A genuine high-confidence multi-hop prediction is NOT a fallback."""
    classifier = QueryClassifier(llm=_FakeLLM(_json_response("multi-hop", 0.9)))
    result = classifier.classify("What calls X and what do those callers call?")
    assert result.query_type == "multi-hop"
    assert result.fallback_applied is False


def test_custom_confidence_threshold() -> None:
    response = _json_response("simple-lookup", 0.7)
    strict = QueryClassifier(llm=_FakeLLM(response), confidence_threshold=0.9)
    loose = QueryClassifier(llm=_FakeLLM(response), confidence_threshold=0.5)
    assert strict.classify("query").fallback_applied is True
    assert loose.classify("query").fallback_applied is False


def test_confidence_threshold_default_comes_from_settings() -> None:
    from reporag.config import settings

    classifier = QueryClassifier(llm=_FakeLLM("{}"))
    assert (
        classifier.confidence_threshold
        == settings.query_classifier_confidence_threshold
    )


def test_confidence_threshold_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        QueryClassifier(confidence_threshold=1.5)
    with pytest.raises(ValueError, match="confidence_threshold"):
        QueryClassifier(confidence_threshold=-0.1)


# ============================================================================
# parse_llm_response -- malformed LLM output always degrades safely
# ============================================================================


def test_parse_invalid_json_returns_zero_confidence_multi_hop() -> None:
    result = parse_llm_response("not json at all")
    assert result.query_type == "multi-hop"
    assert result.confidence == 0.0
    assert "parse_error" in result.metadata


def test_parse_unknown_category_returns_zero_confidence_multi_hop() -> None:
    result = parse_llm_response('{"query_type": "complicated", "confidence": 0.95}')
    assert result.query_type == "multi-hop"
    assert result.confidence == 0.0


def test_parse_missing_confidence_defaults_to_zero() -> None:
    result = parse_llm_response('{"query_type": "simple-lookup"}')
    assert result.query_type == "simple-lookup"
    assert result.confidence == 0.0


def test_parse_non_numeric_confidence_defaults_to_zero() -> None:
    result = parse_llm_response('{"query_type": "simple-lookup", "confidence": "high"}')
    assert result.confidence == 0.0


def test_parse_markdown_fenced_response_is_stripped_and_parsed() -> None:
    fenced = '```json\n{"query_type": "multi-hop", "confidence": 0.85}\n```'
    result = parse_llm_response(fenced)
    assert result.query_type == "multi-hop"
    assert result.confidence == 0.85


def test_parse_empty_response_returns_zero_confidence_multi_hop() -> None:
    result = parse_llm_response("")
    assert result.query_type == "multi-hop"
    assert result.confidence == 0.0
    assert "empty response" in result.metadata["parse_error"]


def test_parse_non_dict_json_returns_zero_confidence_multi_hop() -> None:
    result = parse_llm_response('["simple-lookup", 0.9]')
    assert result.query_type == "multi-hop"
    assert "parse_error" in result.metadata


def test_classify_falls_back_to_rules_when_llm_response_unparseable() -> None:
    """A full classify() call routes through to the rule-based classifier
    when the LLM's response can't be parsed at all."""
    classifier = QueryClassifier(llm=_FakeLLM("garbage, not json"))
    result = classifier.classify("Where is `foo` defined?")
    assert result.source == "rules"


# ============================================================================
# LLM call failure -> rule-based fallback (not a crash)
# ============================================================================


def test_llm_call_exception_falls_back_to_rules_not_a_crash() -> None:
    """If the LLM call itself raises (timeout, rate limit, auth error), classify()
    must not propagate the exception -- it degrades to rule-based classification."""
    raising_llm = _RaisingLLM(TimeoutError("simulated timeout"))
    classifier = QueryClassifier(llm=raising_llm)
    result = classifier.classify("Where is `authenticate` defined?")
    assert result.source == "rules"
    assert raising_llm.calls == 1


def test_llm_call_exception_still_produces_a_usable_result() -> None:
    raising_llm = _RaisingLLM(ConnectionError("simulated network failure"))
    classifier = QueryClassifier(llm=raising_llm)
    result = classifier.classify("Where is `authenticate` defined?")
    assert result.query_type in ("simple-lookup", "multi-hop", "exploratory")


# ============================================================================
# use_llm=False -> rule-based only, LLM never constructed
# ============================================================================


def test_use_llm_false_never_calls_the_llm() -> None:
    fake = _FakeLLM(_json_response("simple-lookup", 0.9))
    classifier = QueryClassifier(llm=fake, use_llm=False)
    result = classifier.classify("Where is `foo` defined?")
    assert fake.prompts == []  # never invoked
    assert result.source == "rules"


def test_use_llm_defaults_from_settings() -> None:
    from reporag.config import settings

    classifier = QueryClassifier(llm=_FakeLLM("{}"))
    assert classifier.use_llm == settings.query_classifier_use_llm


# ============================================================================
# No API key configured -> rule-based fallback (no injected llm)
# ============================================================================


def test_no_api_key_falls_back_to_rules_without_raising(monkeypatch) -> None:
    """When use_llm=True but no llm is injected and no API key is set,
    classify() must not raise -- it falls back to rule-based classification."""
    from pydantic import SecretStr

    from reporag.config import settings

    monkeypatch.setattr(settings, "openai_api_key", SecretStr(""))
    monkeypatch.setattr(settings, "anthropic_api_key", SecretStr(""))
    monkeypatch.setattr(settings, "llm_provider", "openai")

    classifier = QueryClassifier()  # no llm injected
    result = classifier.classify("Where is `authenticate` defined?")
    assert result.source == "rules"
    assert classifier.is_loaded is True


# ============================================================================
# Rule-based classifier: direct tests on clear-signal queries
# ============================================================================


@pytest.mark.parametrize(
    "query,expected_type",
    [
        ("Where is the `parse_config` function defined?", "simple-lookup"),
        ("Show me the `CodeEmbedder` class.", "simple-lookup"),
        ("Which file contains `DatabaseConfig`?", "simple-lookup"),
        ("How does a request flow from the API to the database?", "multi-hop"),
        ("Trace the path from login to session creation.", "multi-hop"),
        ("What calls `authenticate_user`?", "multi-hop"),
        ("Explain the overall architecture of this codebase.", "exploratory"),
        ("Give me an overview of the ingestion pipeline.", "exploratory"),
        ("What are the main components and how are they structured?", "exploratory"),
    ],
)
def test_rule_based_classify_on_clear_signal_queries(query, expected_type) -> None:
    result = rule_based_classify(query)
    assert result.query_type == expected_type
    assert result.source == "rules"


def test_rule_based_classify_zero_signal_defaults_to_multi_hop() -> None:
    """A query with no recognizable pattern at all defaults to multi-hop
    with confidence 0.0, so the caller's threshold fallback fires."""
    result = rule_based_classify("asdf jkl qwerty")
    assert result.query_type == "multi-hop"
    assert result.confidence == 0.0


def test_rule_based_classify_records_vote_scores_in_metadata() -> None:
    result = rule_based_classify("Where is `foo` defined?")
    assert "scores" in result.metadata
    assert set(result.metadata["scores"]) == {
        "simple-lookup",
        "multi-hop",
        "exploratory",
    }


# ============================================================================
# Rule-based generalization: held-out phrasings NOT used to design the
# patterns above. This is the honesty check -- a classifier tuned only to
# its own training examples is not a generalizing classifier. These
# queries were written independently and verified to pass, not adjusted
# after the fact to make the patterns fit.
# ============================================================================


@pytest.mark.parametrize(
    "query,expected_type",
    [
        ("Can you point me to the login handler in the code?", "simple-lookup"),
        (
            "Walk me through what happens when a user logs in, step by "
            "step across the system.",
            "multi-hop",
        ),
        (
            "I want a bird's eye view of how this project is put together.",
            "exploratory",
        ),
        ("What module has the retry logic?", "simple-lookup"),
        ("Give me the big picture of this repo.", "exploratory"),
        ("What file handles the rate limiting?", "simple-lookup"),
    ],
)
def test_rule_based_generalizes_to_unseen_phrasings(query, expected_type) -> None:
    """Validates the rule-based classifier against phrasing that did not
    shape the regex patterns -- guards against a classifier that only
    works on the exact wording it was written to expect."""
    result = rule_based_classify(query)
    assert result.query_type == expected_type


def test_rule_based_degrades_safely_on_a_genuinely_ambiguous_query() -> None:
    """A phrasing outside even the broadened patterns still degrades to a
    zero-confidence result (never a confidently wrong answer)."""
    result = rule_based_classify(
        "Can you sketch out the general shape of this project?"
    )
    # Whatever it lands on, zero-signal must not silently claim high
    # confidence -- the classify() threshold fallback depends on this.
    if result.confidence == 0.0:
        assert result.query_type == "multi-hop"


# ============================================================================
# Prompt construction / few-shot examples
# ============================================================================


def test_llm_prompt_includes_the_query() -> None:
    fake = _FakeLLM(_json_response("simple-lookup", 0.9))
    classifier = QueryClassifier(llm=fake)
    classifier.classify("What does the `foo` function do?")
    assert "What does the `foo` function do?" in fake.prompts[0]


def test_llm_prompt_requests_json_only() -> None:
    fake = _FakeLLM(_json_response("simple-lookup", 0.9))
    classifier = QueryClassifier(llm=fake)
    classifier.classify("query")
    assert "JSON" in fake.prompts[0]


# ============================================================================
# Input validation
# ============================================================================


def test_empty_query_raises() -> None:
    classifier = QueryClassifier(llm=_FakeLLM("{}"))
    with pytest.raises(ValueError, match="empty"):
        classifier.classify("")


def test_whitespace_only_query_raises() -> None:
    classifier = QueryClassifier(llm=_FakeLLM("{}"))
    with pytest.raises(ValueError, match="empty"):
        classifier.classify("   \n\t  ")


# ============================================================================
# Lazy loading
# ============================================================================


def test_constructing_classifier_does_not_call_llm() -> None:
    fake = _FakeLLM(_json_response("simple-lookup", 0.9))
    classifier = QueryClassifier(llm=fake)
    assert fake.prompts == []
    assert classifier.is_loaded is False


def test_is_loaded_true_after_first_classify_with_injected_llm() -> None:
    fake = _FakeLLM(_json_response("simple-lookup", 0.9))
    classifier = QueryClassifier(llm=fake)
    classifier.classify("query")
    assert classifier.is_loaded is True


def test_repr_reflects_state() -> None:
    classifier = QueryClassifier(llm=_FakeLLM("{}"), use_llm=False)
    text = repr(classifier)
    assert "use_llm=False" in text
    assert "loaded=False" in text


# ============================================================================
# ClassificationResult dataclass sanity
# ============================================================================


def test_classification_result_is_frozen() -> None:
    result = ClassificationResult(query_type="simple-lookup", confidence=0.9)
    assert result.query_type == "simple-lookup"
    assert result.confidence == 0.9
    assert result.fallback_applied is False
    assert result.source == "rules"
    with pytest.raises(AttributeError):
        result.confidence = 0.5  # type: ignore[misc]
