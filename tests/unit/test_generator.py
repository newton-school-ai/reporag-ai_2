"""Unit tests for the Issue 25 answer generator.

Every test is offline.  The generator's ``llm=`` seam replaces the provider
client with a plain callable, so nothing here needs an API key, a network,
or a provider SDK.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from reporag.config import settings
from reporag.generation.citation import ContextIndex
from reporag.generation.context_assembler import ContextAssembler
from reporag.generation.generator import (
    AnswerGenerator,
    GeneratedAnswer,
    GenerationError,
    classify_error,
    is_retryable,
)
from reporag.generation.prompt_builder import PromptBuilder
from reporag.retrieval.vector_search import RetrievalResult

ROUTE_CODE = (
    "def login_route(payload):\n    token = issue_token(payload)\n    return token"
)
SERVICE_CODE = (
    "def authenticate_user(email):\n    user = repo.get(email)\n    return user"
)

CITED_ANSWER = (
    "The route issues a token [src/app/api/routes/auth.py:20-22]. "
    "It delegates the credential check to the service "
    "[src/app/auth/service.py:88-90]."
)


def make_result(
    file_path: str,
    start_line: int,
    end_line: int,
    text: str,
    score: float = 0.9,
) -> RetrievalResult:
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=None,
        chunk_text=text,
        metadata={},
    )


@pytest.fixture
def results() -> list[RetrievalResult]:
    return [
        make_result("src/app/api/routes/auth.py", 20, 22, ROUTE_CODE, score=0.9),
        make_result("src/app/auth/service.py", 88, 90, SERVICE_CODE, score=0.8),
    ]


@pytest.fixture
def no_sleep() -> Any:
    """A sleep seam that records delays instead of waiting for them."""
    return []


def constant_llm(text: str = CITED_ANSWER):
    """An LLM stand-in that always answers with *text*."""

    def _call(payload: Any) -> str:
        _call.payloads.append(payload)
        return text

    _call.payloads = []  # type: ignore[attr-defined]
    return _call


def failing_llm(
    exc: Exception, succeed_on: int | None = None, text: str = CITED_ANSWER
):
    """An LLM stand-in that raises *exc* until attempt *succeed_on*."""

    def _call(payload: Any) -> str:
        _call.calls += 1  # type: ignore[attr-defined]
        if succeed_on is not None and _call.calls >= succeed_on:
            return text
        raise exc

    _call.calls = 0  # type: ignore[attr-defined]
    return _call


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_provider_and_model_default_to_settings():
    generator = AnswerGenerator(llm=constant_llm())
    assert generator.provider == settings.llm_provider
    assert generator.model in (settings.openai_model, settings.anthropic_model)


def test_provider_and_model_can_be_chosen_explicitly():
    # The constructor call from the issue's usage example.
    generator = AnswerGenerator(provider="openai", model="gpt-4o", llm=constant_llm())
    assert (generator.provider, generator.model) == ("openai", "gpt-4o")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider": "cohere"},
        {"max_retries": -1},
        {"backoff_seconds": -0.1},
        {"backoff_multiplier": 0.5},
        {"max_output_tokens": 0},
    ],
)
def test_invalid_construction_arguments_raise(kwargs: dict[str, Any]):
    with pytest.raises(ValueError):
        AnswerGenerator(llm=constant_llm(), **kwargs)


def test_repr_is_informative():
    text = repr(AnswerGenerator(provider="openai", model="gpt-4o", llm=constant_llm()))
    assert "gpt-4o" in text and "openai" in text


def test_an_injected_llm_counts_as_loaded():
    # Nothing may reach for an API key when a callable was supplied.
    assert AnswerGenerator(llm=constant_llm()).is_loaded


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_generate_returns_the_answer_with_validated_citations(results):
    result = AnswerGenerator(llm=constant_llm()).generate("Some prompt text", results)
    assert result.ok
    assert result.answer_text == CITED_ANSWER
    assert [c.file_path for c in result.citations] == [
        "src/app/api/routes/auth.py",
        "src/app/auth/service.py",
    ]
    assert all(c.verified for c in result.citations)
    assert result.citation_coverage == 1.0
    assert result.meets_coverage_target


def test_invalid_citations_are_reported_not_silently_kept(results, caplog):
    answer = "Hashing happens in the hasher module [src/app/auth/hasher.py:5-9]."
    with caplog.at_level("WARNING"):
        result = AnswerGenerator(llm=constant_llm(answer)).generate("p", results)
    assert result.citations == ()
    assert result.invalid_citations[0].status == "unknown-file"
    assert result.citation_coverage == 0.0
    assert "could not be backed" in caplog.text


def test_coverage_below_the_target_is_flagged(results, caplog):
    # The signal Issue 35's faithfulness eval will look for: the answer is
    # drifting off the retrieved code even though every citation is real.
    answer = (
        "The route issues a token [src/app/api/routes/auth.py:20-22]. "
        "The password is hashed with bcrypt and stored in Postgres."
    )
    with caplog.at_level("WARNING"):
        result = AnswerGenerator(llm=constant_llm(answer)).generate("p", results)
    assert not result.meets_coverage_target
    assert result.invalid_citations == ()
    assert "citation coverage 50% is below the 90% target" in caplog.text


def test_result_records_provider_model_and_attempts(results):
    result = AnswerGenerator(
        provider="openai", model="gpt-4o", llm=constant_llm()
    ).generate("p", results)
    assert (result.provider, result.model) == ("openai", "gpt-4o")
    assert result.attempts == 1
    assert result.latency_ms >= 0
    assert result.metadata["indexed_files"] == 2


def test_result_serializes_for_the_api(results):
    payload = AnswerGenerator(llm=constant_llm()).generate("p", results).to_dict()
    assert payload["ok"] is True
    assert payload["error"] is None
    assert payload["citation_coverage"] == 1.0
    assert len(payload["citations"]) == 2
    assert payload["citations"][0]["snippet"] == ROUTE_CODE


def test_result_str_is_the_answer(results):
    result = AnswerGenerator(llm=constant_llm()).generate("p", results)
    assert str(result) == CITED_ANSWER


def test_answer_is_stripped_but_the_raw_response_is_kept(results):
    result = AnswerGenerator(llm=constant_llm("  answer text here  ")).generate(
        "p", results
    )
    assert result.answer_text == "answer text here"
    assert result.raw_response == "  answer text here  "


# ---------------------------------------------------------------------------
# Prompt handling
# ---------------------------------------------------------------------------


def test_a_built_prompt_is_sent_as_chat_messages(results):
    llm = constant_llm()
    built = PromptBuilder(max_tokens=4000).build_prompt(
        "How does auth work?",
        "multi-hop",
        ContextAssembler(max_tokens=2000).assemble(results),
    )
    AnswerGenerator(llm=llm).generate(built)
    payload = llm.payloads[0]
    assert [message["role"] for message in payload] == ["system", "user"]
    assert payload[1]["content"] == built.user


def test_chat_messages_can_be_turned_off(results):
    llm = constant_llm()
    built = PromptBuilder(max_tokens=4000).build_prompt("q?", "multi-hop", "")
    AnswerGenerator(llm=llm, use_chat_messages=False).generate(built)
    assert llm.payloads[0] == built.text


def test_a_raw_message_list_is_passed_through():
    llm = constant_llm()
    messages = [{"role": "user", "content": "hello"}]
    AnswerGenerator(llm=llm).generate(messages)
    assert llm.payloads[0] == messages


@pytest.mark.parametrize("prompt", ["", "   ", [], 42, None])
def test_an_unusable_prompt_is_a_programmer_error(prompt: Any):
    # Provider failures are returned; a bad call signature still raises.
    with pytest.raises(ValueError):
        AnswerGenerator(llm=constant_llm()).generate(prompt)


# ---------------------------------------------------------------------------
# Choosing the context to validate against
# ---------------------------------------------------------------------------


def test_a_built_prompt_validates_against_its_own_context(results):
    # No context_chunks argument: the prompt's context section is the only
    # code the model saw, so it is what the citations are checked against.
    built = PromptBuilder(max_tokens=4000).build_prompt(
        "How does auth work?",
        "multi-hop",
        ContextAssembler(max_tokens=2000).assemble(results),
    )
    result = AnswerGenerator(llm=constant_llm()).generate(built)
    assert result.citation_coverage == 1.0
    assert all(c.verified for c in result.citations)


def test_explicit_context_chunks_win_over_the_prompt(results):
    built = PromptBuilder(max_tokens=4000).build_prompt("q?", "multi-hop", "")
    result = AnswerGenerator(llm=constant_llm()).generate(built, results)
    assert all(c.verified for c in result.citations)


def test_a_string_prompt_without_context_leaves_citations_unverified():
    result = AnswerGenerator(llm=constant_llm()).generate("some prompt")
    assert result.report.context_available is False
    assert all(c.status == "unverified" for c in result.all_citations)


def test_a_prepared_index_is_accepted(results):
    result = AnswerGenerator(llm=constant_llm()).generate(
        "p", ContextIndex.from_results(results)
    )
    assert all(c.verified for c in result.citations)


# ---------------------------------------------------------------------------
# Retries and error handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (TimeoutError("took too long"), "timeout"),
        (ConnectionError("connection reset"), "connection"),
        (RuntimeError("Error code: 429 - rate limit reached"), "rate_limit"),
        (RuntimeError("Request timed out."), "timeout"),
        (RuntimeError("Incorrect API key provided"), "auth"),
        (RuntimeError("Error code: 503 - overloaded"), "server_error"),
        (RuntimeError("maximum context length is 8192 tokens"), "invalid_request"),
        (RuntimeError("something entirely new"), "unknown"),
    ],
)
def test_provider_errors_are_classified_without_importing_a_sdk(exc, expected):
    assert classify_error(exc) == expected


@pytest.mark.parametrize(
    ("kind", "retryable"),
    [
        ("timeout", True),
        ("rate_limit", True),
        ("connection", True),
        ("server_error", True),
        ("empty_response", True),
        ("auth", False),
        ("invalid_request", False),
        ("unknown", False),
    ],
)
def test_only_transient_failures_are_retryable(kind, retryable):
    assert is_retryable(kind) is retryable


def test_a_transient_failure_is_retried_with_exponential_backoff(results, no_sleep):
    llm = failing_llm(RuntimeError("Error code: 429"), succeed_on=3)
    generator = AnswerGenerator(
        llm=llm, backoff_seconds=0.5, backoff_multiplier=2.0, sleep=no_sleep.append
    )
    result = generator.generate("p", results)
    assert result.ok
    assert result.attempts == 3
    assert no_sleep == [0.5, 1.0]


def test_retries_are_bounded_and_the_failure_is_returned(results, no_sleep):
    llm = failing_llm(RuntimeError("Error code: 429"))
    result = AnswerGenerator(llm=llm, max_retries=2, sleep=no_sleep.append).generate(
        "p", results
    )
    assert not result.ok
    assert result.error.kind == "rate_limit"
    assert result.error.attempts == 3
    assert result.error.retryable
    assert llm.calls == 3
    assert result.answer_text == ""


def test_a_permanent_failure_is_not_retried(results, no_sleep):
    llm = failing_llm(RuntimeError("Incorrect API key provided"))
    result = AnswerGenerator(llm=llm, sleep=no_sleep.append).generate("p", results)
    assert result.error.kind == "auth"
    assert not result.error.retryable
    assert llm.calls == 1
    assert no_sleep == []


def test_retries_can_be_switched_off(results, no_sleep):
    llm = failing_llm(TimeoutError("slow"))
    result = AnswerGenerator(llm=llm, max_retries=0, sleep=no_sleep.append).generate(
        "p", results
    )
    assert llm.calls == 1
    assert result.error.kind == "timeout"


def test_an_empty_response_is_a_failure_not_an_empty_answer(results, no_sleep):
    result = AnswerGenerator(
        llm=constant_llm("   "), max_retries=1, sleep=no_sleep.append
    ).generate("p", results)
    assert not result.ok
    assert result.error.kind == "empty_response"
    assert result.error.attempts == 2


def test_a_non_string_response_is_a_failure(results, no_sleep):
    result = AnswerGenerator(
        llm=lambda payload: None, max_retries=0, sleep=no_sleep.append
    ).generate("p", results)
    assert result.error.kind == "empty_response"


def test_a_provider_failure_is_logged(results, no_sleep, caplog):
    with caplog.at_level("ERROR"):
        AnswerGenerator(
            llm=failing_llm(RuntimeError("Incorrect API key")), sleep=no_sleep.append
        ).generate("p", results)
    assert "generation failed" in caplog.text


def test_a_missing_api_key_degrades_instead_of_raising(monkeypatch, caplog, results):
    monkeypatch.setattr(settings, "openai_api_key", SecretStr(""))
    generator = AnswerGenerator(provider="openai")
    with caplog.at_level("WARNING"):
        result = generator.generate("p", results)
    assert not result.ok
    assert result.error.kind == "auth"
    assert result.attempts == 0
    assert "OPENAI_API_KEY" in result.error.message
    assert "no API key configured" in caplog.text


def test_generation_error_serializes():
    payload = GenerationError("rate_limit", "slow down", attempts=3).to_dict()
    assert payload == {
        "kind": "rate_limit",
        "message": "slow down",
        "attempts": 3,
        "retryable": True,
        "exception_type": None,
    }
    assert str(GenerationError("timeout", "gone")) == "timeout: gone"


def test_a_failed_result_still_answers_every_accessor():
    result = GeneratedAnswer(error=GenerationError("timeout", "gone"))
    assert not result.ok
    assert result.citations == ()
    assert result.invalid_citations == ()
    assert result.citation_coverage == 1.0
    assert result.cited_files == ()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class _StreamingClient:
    """A fake provider client that streams deltas, like langchain's does."""

    def __init__(self, deltas: list[str], fail_after: int | None = None) -> None:
        self.deltas = deltas
        self.fail_after = fail_after

    def __call__(self, payload: Any) -> str:
        return "".join(self.deltas)

    def stream(self, payload: Any):
        for position, delta in enumerate(self.deltas):
            if self.fail_after is not None and position >= self.fail_after:
                raise RuntimeError("Error code: 503 - overloaded")
            yield delta


def test_streaming_forwards_deltas_and_still_validates(results):
    client = _StreamingClient(
        ["The route issues a token ", "[src/app/api/routes/auth.py:20-22]."]
    )
    tokens: list[str] = []
    result = AnswerGenerator(llm=client).generate("p", results, on_token=tokens.append)
    assert tokens == client.deltas
    assert result.ok
    assert result.metadata["streamed"] is True
    assert result.citations[0].verified


def test_a_plain_callable_still_delivers_the_answer_to_on_token(results):
    tokens: list[str] = []
    AnswerGenerator(llm=constant_llm()).generate("p", results, on_token=tokens.append)
    assert tokens == [CITED_ANSWER]


def test_a_stream_that_fails_mid_answer_is_not_replayed(results, no_sleep):
    # Retrying would emit the opening tokens a second time and corrupt what
    # the caller already rendered.
    client = _StreamingClient(["one ", "two ", "three"], fail_after=2)
    tokens: list[str] = []
    result = AnswerGenerator(llm=client, sleep=no_sleep.append).generate(
        "p", results, on_token=tokens.append
    )
    assert tokens == ["one ", "two "]
    assert not result.ok
    assert result.error.kind == "server_error"
    assert no_sleep == []


def test_a_stream_that_fails_before_any_token_is_retried(results, no_sleep):
    client = _StreamingClient(["one ", "two"], fail_after=0)
    result = AnswerGenerator(llm=client, max_retries=1, sleep=no_sleep.append).generate(
        "p", results, on_token=lambda _: None
    )
    assert not result.ok
    assert result.error.attempts == 2
    assert no_sleep == [0.5]


# ---------------------------------------------------------------------------
# Retrieval results straight to a cited answer
# ---------------------------------------------------------------------------


def test_generate_from_results_builds_the_prompt_and_cites(results):
    llm = constant_llm()
    result = AnswerGenerator(llm=llm, model="gpt-4o").generate_from_results(
        "How does auth work?", results, "multi-hop"
    )
    system = llm.payloads[0][0]["content"]
    user = llm.payloads[0][1]["content"]
    assert "[file_path:start_line-end_line]" in system
    assert "src/app/auth/service.py" in user
    assert result.citation_coverage == 1.0


def test_generate_from_results_passes_prior_findings(results):
    llm = constant_llm()
    AnswerGenerator(llm=llm, model="gpt-4o").generate_from_results(
        "How does auth work?",
        results,
        "multi-hop",
        sub_query_answers={"step-1": "The login route is login_route."},
    )
    assert "The login route is login_route." in llm.payloads[0][1]["content"]


def test_generate_from_results_validates_against_what_survived_the_budget(results):
    # A chunk trimmed to fit the context window was never seen by the model,
    # so citing it must be flagged even though retrieval did return it.
    filler = make_result(
        "src/zzz/filler.py", 1, 400, "\n".join("filler line" for _ in range(400)), 0.99
    )
    llm = constant_llm("The answer is in [src/app/auth/service.py:88-90].")
    generator = AnswerGenerator(
        llm=llm,
        model="gpt-4o",
        prompt_builder=PromptBuilder(model="gpt-4o", max_tokens=700),
    )

    result = generator.generate_from_results("How?", [*results, filler], "multi-hop")
    sent = llm.payloads[0][1]["content"]
    assert "src/zzz/filler.py" not in sent, "the filler should not have fitted"
    assert result.metadata["indexed_files"] < 3

    dropped = AnswerGenerator(
        llm=constant_llm("See [src/zzz/filler.py:1-4].")
    ).generate("p", sent)
    assert dropped.all_citations[0].status == "unknown-file"
