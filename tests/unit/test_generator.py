"""Unit tests for the LLM answer generator (Issue 25).

Covers every acceptance criterion of the generator half of Issue 25:

* calls a configurable LLM (provider selection is exercised via the
  injected-``llm`` seam, since a real network call is never made here),
* handles LLM errors gracefully -- timeout, rate limit, auth, invalid
  (empty) response, and generic API errors are all classified and
  reported via a structured result rather than raised,
* one retry before giving up, matching this codebase's established
  retry-once-then-report shape,
* the ``{answer, citations}`` structured result the issue asks for,

plus edge cases: constructor validation, `BuiltPrompt` vs. plain-string
input, and citation validation using an overridden context.

All tests are fully offline: every generator here is constructed with an
injected `llm` callable, so no test makes a real network or LLM API call.
"""

from __future__ import annotations

import time

import pytest

from reporag.generation.citation import CitationReport
from reporag.generation.generator import (
    AnsweredQuery,
    AnswerGenerator,
)

CONTEXT = (
    "## src/app/api/routes/auth.py (lines 20-31)\n"
    "```python\n"
    "def login_route(payload):\n"
    "    return authenticate_user(payload)\n"
    "```"
)


class _FakeBuiltPrompt:
    """Duck-typed stand-in for `prompt_builder.BuiltPrompt`."""

    def __init__(self, system: str, user: str, context: str = CONTEXT) -> None:
        self.system = system
        self.user = user
        self.sections = {"context": context}


# ---------------------------------------------------------------------------
# Acceptance: calls a configurable LLM
# ---------------------------------------------------------------------------


class TestCallsConfigurableLLM:
    def test_injected_llm_is_called_with_system_and_user(self) -> None:
        received = {}

        def fake_llm(system: str, user: str) -> str:
            received["system"] = system
            received["user"] = user
            return "the answer"

        gen = AnswerGenerator(llm=fake_llm)
        result = gen.generate("system text", "user text")
        assert result.success is True
        assert result.text == "the answer"
        assert received == {"system": "system text", "user": "user text"}

    def test_built_prompt_input_uses_its_system_and_user(self) -> None:
        received = {}

        def fake_llm(system: str, user: str) -> str:
            received["system"] = system
            received["user"] = user
            return "the answer"

        prompt = _FakeBuiltPrompt(system="SYS", user="USR")
        gen = AnswerGenerator(llm=fake_llm)
        result = gen.generate(prompt)
        assert received == {"system": "SYS", "user": "USR"}
        assert result.success is True

    def test_model_and_provider_are_recorded_on_success(self) -> None:
        gen = AnswerGenerator(llm=lambda s, u: "ok", model="gpt-4o")
        result = gen.generate("s", "u")
        assert result.model == "gpt-4o"
        assert result.provider in ("openai", "anthropic")

    def test_default_model_comes_from_settings_when_not_overridden(self) -> None:
        gen = AnswerGenerator(llm=lambda s, u: "ok")
        assert gen.model  # non-empty, resolved from settings

    def test_repr_is_informative(self) -> None:
        gen = AnswerGenerator(llm=lambda s, u: "ok", model="gpt-4o")
        assert "AnswerGenerator(provider=" in repr(gen)
        assert "gpt-4o" in repr(gen)


# ---------------------------------------------------------------------------
# Acceptance: handles LLM errors gracefully
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_generic_exception_is_reported_not_raised(self) -> None:
        def broken_llm(system: str, user: str) -> str:
            raise RuntimeError("boom")

        gen = AnswerGenerator(llm=broken_llm, max_retries=0)
        result = gen.generate("s", "u")  # must not raise
        assert result.success is False
        assert result.error == "boom"
        assert result.error_kind == "api_error"
        assert result.text == ""

    @pytest.mark.parametrize(
        ("message", "expected_kind"),
        [
            ("RateLimitError: quota exceeded", "rate_limit"),
            ("Error code: 429", "rate_limit"),
            ("AuthenticationError: invalid api_key", "auth"),
            ("PermissionDeniedError", "auth"),
            ("Error code: 401 Unauthorized", "auth"),
            ("Request timed out after 30s", "timeout"),
            ("TimeoutError", "timeout"),
            ("Something totally unrecognised happened", "api_error"),
        ],
    )
    def test_error_classification(self, message: str, expected_kind: str) -> None:
        def failing_llm(system: str, user: str) -> str:
            raise Exception(message)

        gen = AnswerGenerator(llm=failing_llm, max_retries=0)
        result = gen.generate("s", "u")
        assert result.error_kind == expected_kind

    def test_empty_response_is_classified_as_invalid_response(self) -> None:
        gen = AnswerGenerator(llm=lambda s, u: "   ", max_retries=0)
        result = gen.generate("s", "u")
        assert result.success is False
        assert result.error_kind == "invalid_response"

    def test_none_response_is_classified_as_invalid_response(self) -> None:
        gen = AnswerGenerator(llm=lambda s, u: None, max_retries=0)
        result = gen.generate("s", "u")
        assert result.success is False
        assert result.error_kind == "invalid_response"

    def test_timeout_is_enforced_and_reported(self) -> None:
        def slow_llm(system: str, user: str) -> str:
            time.sleep(2)
            return "too slow"

        gen = AnswerGenerator(llm=slow_llm, timeout_seconds=0.2, max_retries=0)
        start = time.monotonic()
        result = gen.generate("s", "u")
        elapsed = time.monotonic() - start
        assert result.success is False
        assert result.error_kind == "timeout"
        assert elapsed < 1.0  # did not wait for the full 2s sleep

    def test_result_never_raises_regardless_of_failure_kind(self) -> None:
        for exc in (RuntimeError, ValueError, ConnectionError, TimeoutError):

            def raiser(system: str, user: str, _exc: type[Exception] = exc) -> str:
                raise _exc("failure")

            gen = AnswerGenerator(llm=raiser, max_retries=0)
            result = gen.generate("s", "u")  # must not raise
            assert result.success is False


# ---------------------------------------------------------------------------
# Acceptance: retries once before giving up
# ---------------------------------------------------------------------------


class TestRetries:
    def test_succeeds_after_one_retry(self) -> None:
        calls = {"n": 0}

        def flaky_llm(system: str, user: str) -> str:
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("temporary blip")
            return "recovered"

        gen = AnswerGenerator(llm=flaky_llm, max_retries=1)
        result = gen.generate("s", "u")
        assert result.success is True
        assert result.text == "recovered"
        assert result.retries == 1
        assert calls["n"] == 2

    def test_default_max_retries_is_one(self) -> None:
        calls = {"n": 0}

        def flaky_llm(system: str, user: str) -> str:
            calls["n"] += 1
            raise RuntimeError("always fails")

        gen = AnswerGenerator(llm=flaky_llm)  # default max_retries
        gen.generate("s", "u")
        assert calls["n"] == 2  # one initial attempt + one retry

    def test_max_retries_zero_disables_retrying(self) -> None:
        calls = {"n": 0}

        def flaky_llm(system: str, user: str) -> str:
            calls["n"] += 1
            raise RuntimeError("fails")

        gen = AnswerGenerator(llm=flaky_llm, max_retries=0)
        gen.generate("s", "u")
        assert calls["n"] == 1

    def test_retries_reported_as_zero_on_first_attempt_success(self) -> None:
        gen = AnswerGenerator(llm=lambda s, u: "ok", max_retries=3)
        result = gen.generate("s", "u")
        assert result.retries == 0

    def test_all_retries_exhausted_reports_max_retries(self) -> None:
        gen = AnswerGenerator(
            llm=lambda s, u: (_ for _ in ()).throw(RuntimeError("x")),
            max_retries=2,
        )
        result = gen.generate("s", "u")
        assert result.retries == 2


# ---------------------------------------------------------------------------
# Acceptance: structured {answer, citations} result
# ---------------------------------------------------------------------------


class TestGenerateWithCitations:
    def test_successful_generation_returns_answer_and_citations(self) -> None:
        answer_text = (
            "login_route calls authenticate_user " "[src/app/api/routes/auth.py:20-31]."
        )
        prompt = _FakeBuiltPrompt(system="SYS", user="USR", context=CONTEXT)
        gen = AnswerGenerator(llm=lambda s, u: answer_text)
        answered = gen.generate_with_citations(prompt)
        assert isinstance(answered, AnsweredQuery)
        assert answered.success is True
        assert answered.answer == answer_text
        assert isinstance(answered.citations, CitationReport)
        assert answered.citations.valid_count == 1

    def test_failed_generation_returns_empty_answer_and_empty_citations(self) -> None:
        prompt = _FakeBuiltPrompt(system="SYS", user="USR", context=CONTEXT)

        def broken_llm(system: str, user: str) -> str:
            raise RuntimeError("down")

        gen = AnswerGenerator(llm=broken_llm, max_retries=0)
        answered = gen.generate_with_citations(prompt)
        assert answered.success is False
        assert answered.answer == ""
        assert answered.citations.citations == []
        assert answered.generation.error == "down"

    def test_explicit_context_override_is_used_for_validation(self) -> None:
        other_context = "## other.py (lines 1-3)\n```python\nx = 1\n```"
        answer_text = "It does X [other.py:1-3]."
        prompt = _FakeBuiltPrompt(system="SYS", user="USR", context=CONTEXT)
        gen = AnswerGenerator(llm=lambda s, u: answer_text)
        # Without the override, "other.py" isn't in `prompt`'s own context.
        without_override = gen.generate_with_citations(prompt)
        assert without_override.citations.valid_count == 0
        # With the override, it validates against `other_context` instead.
        with_override = gen.generate_with_citations(prompt, context=other_context)
        assert with_override.citations.valid_count == 1

    def test_answered_query_success_property_matches_generation(self) -> None:
        prompt = _FakeBuiltPrompt(system="SYS", user="USR")
        gen = AnswerGenerator(llm=lambda s, u: "answer text")
        answered = gen.generate_with_citations(prompt)
        assert answered.success == answered.generation.success


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_rejects_non_positive_timeout(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
            AnswerGenerator(timeout_seconds=0)
        with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
            AnswerGenerator(timeout_seconds=-1)

    def test_rejects_negative_max_retries(self) -> None:
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            AnswerGenerator(max_retries=-1)

    def test_accepts_zero_max_retries(self) -> None:
        AnswerGenerator(max_retries=0)  # must not raise
