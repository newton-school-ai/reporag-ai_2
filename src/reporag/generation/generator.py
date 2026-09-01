"""LLM answer generation with validated citations (Issue 25).

Calls the configured LLM with a built prompt, then hands the response to
:mod:`reporag.generation.citation` so the answer arrives with its citations
already parsed, resolved against the retrieved code, and scored for
coverage.

Why
---
This is the last step of the pipeline and the one the user actually reads,
so two things decide whether it is trustworthy:

* **The citations must be checked, not just parsed.**  An answer citing
  ``[src/auth/service.py:88-107]`` for a file that was never retrieved is a
  confident hallucination wearing the costume of a grounded answer.  The
  generator therefore validates against the context that was *sent to the
  model*, not against everything retrieval found -- the prompt builder may
  have dropped chunks to fit the window, and the model cannot cite what it
  never saw.
* **A provider outage must not become a stack trace.**  Rate limits,
  timeouts and transient 5xx are normal operating conditions for an LLM
  call.  Transient failures are retried with exponential backoff; whatever
  survives is returned as a structured error on the result rather than
  raised, so the Issue 26 endpoint can answer with a clean 503 instead of
  crashing mid-request.

Design
------
The layout follows :class:`reporag.agent.planner.QueryClassifier`, the
other LLM-backed component in this codebase:

* **Injectable LLM seam** -- passing ``llm=`` a ``(payload) -> str``
  callable replaces the provider client entirely.  That is the supported
  test seam, and it is what keeps this module's unit tests offline and
  free of API keys.
* **Lazy client construction** -- the langchain client is built on the
  first :meth:`AnswerGenerator.generate` call, never at import or
  construction time.  With no API key configured the generator degrades to
  a structured ``auth`` error instead of raising, exactly as the classifier
  degrades to its rule-based path.
* **Structured result, never a raised provider error** -- every call
  returns a :class:`GeneratedAnswer`.  ``ok`` says whether the model
  answered; ``error`` says why not, classified into a small taxonomy
  (:data:`GenerationErrorKind`) that is derived from the exception rather
  than from a provider SDK import, so the classification works whichever
  client is installed.  ``ValueError`` is still raised for programmer
  errors -- an empty prompt, an unknown provider -- because those are bugs,
  not outages.
* **Streaming without a second code path** -- passing ``on_token`` streams
  deltas to the caller while still returning the same fully validated
  :class:`GeneratedAnswer` at the end, so the frontend gets live tokens and
  the API gets checked citations from one call.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from reporag.config import settings
from reporag.generation.citation import (
    MIN_CITATION_COVERAGE,
    Citation,
    CitationReport,
    ContextIndex,
    build_context_index,
    extract_citations,
)
from reporag.generation.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

__all__ = [
    "AnswerGenerator",
    "GeneratedAnswer",
    "GenerationError",
    "GenerationErrorKind",
    "LLMCallable",
    "classify_error",
    "is_retryable",
]


LLMCallable = Callable[[Any], str]
"""A provider-agnostic ``(payload) -> str`` call.

The payload is either the prompt as a single string or a list of chat
messages (``[{"role": ..., "content": ...}]``), depending on
``AnswerGenerator(use_chat_messages=...)``.  Passing one of these to the
constructor is the supported way to run the generator without a network or
an API key.
"""

Provider = Literal["openai", "anthropic"]

GenerationErrorKind = Literal[
    "auth",
    "timeout",
    "rate_limit",
    "connection",
    "server_error",
    "invalid_request",
    "empty_response",
    "unknown",
]
"""Why a generation attempt failed, independent of which SDK raised."""

# Failures worth another attempt: the request was fine, the provider was
# momentarily not.  ``invalid_request`` and ``auth`` are deliberately absent
# -- retrying a malformed request or a missing key only burns latency.
_RETRYABLE_KINDS: frozenset[str] = frozenset(
    {"timeout", "rate_limit", "connection", "server_error", "empty_response"}
)

# Substring probes against the exception type name and message, in priority
# order.  Matching on text rather than on SDK exception classes keeps this
# working whether the caller installed ``openai``, ``anthropic``, both, or
# neither -- and keeps the unit tests free of both.
_ERROR_SIGNATURES: tuple[tuple[GenerationErrorKind, tuple[str, ...]], ...] = (
    ("rate_limit", ("ratelimit", "rate limit", "429", "quota", "too many requests")),
    ("timeout", ("timeout", "timed out", "deadline")),
    (
        "auth",
        (
            "authentication",
            "unauthorized",
            "permissiondenied",
            "permission denied",
            "invalid api key",
            "api key",
            "401",
            "403",
        ),
    ),
    (
        "server_error",
        ("internalserver", "internal server", "overloaded", "502", "503", "504", "500"),
    ),
    ("connection", ("connection", "network", "unreachable", "dns", "socket")),
    (
        "invalid_request",
        (
            "badrequest",
            "bad request",
            "invalidrequest",
            "invalid request",
            "context length",
            "maximum context",
            "not found",
            "400",
            "404",
        ),
    ),
)

_DEFAULT_MAX_RETRIES = 2
_DEFAULT_BACKOFF_SECONDS = 0.5
_DEFAULT_BACKOFF_MULTIPLIER = 2.0
_DEFAULT_TIMEOUT_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def classify_error(exc: BaseException) -> GenerationErrorKind:
    """Classify a provider exception into a :data:`GenerationErrorKind`.

    The exception's class name and message are probed for known signatures,
    so an ``openai.RateLimitError``, an ``anthropic.RateLimitError`` and a
    bare ``RuntimeError("429 too many requests")`` all classify the same
    way without importing either SDK.

    Args:
        exc: The exception raised by the provider client.

    Returns:
        The matching kind, or ``"unknown"`` when nothing matches.
    """
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection"

    haystack = f"{type(exc).__name__} {exc}".lower()
    for kind, needles in _ERROR_SIGNATURES:
        if any(needle in haystack for needle in needles):
            return kind
    return "unknown"


def is_retryable(kind: GenerationErrorKind) -> bool:
    """``True`` when another attempt could plausibly succeed."""
    return kind in _RETRYABLE_KINDS


@dataclass(frozen=True)
class GenerationError:
    """Why generation failed, after every retry was spent.

    Attributes:
        kind: The failure category.
        message: The provider's message, or an explanation of a local
            precondition that was not met (no API key, say).
        attempts: How many calls were made before giving up.
        exception_type: Class name of the last exception, when there was
            one.
    """

    kind: GenerationErrorKind
    message: str
    attempts: int = 1
    exception_type: str | None = None

    @property
    def retryable(self) -> bool:
        """``True`` when this kind of failure is worth retrying later."""
        return is_retryable(self.kind)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready view, for the Issue 26 query endpoint."""
        return {
            "kind": self.kind,
            "message": self.message,
            "attempts": self.attempts,
            "retryable": self.retryable,
            "exception_type": self.exception_type,
        }

    def __str__(self) -> str:
        return f"{self.kind}: {self.message}"


class _AttemptError(Exception):
    """Internal: one attempt failed, carrying its classification.

    ``emitted`` records whether any streamed delta already reached the
    caller before the failure.  Once it has, the attempt is no longer
    retryable in practice however transient the cause was -- a second
    attempt would replay the answer from the top on top of what the caller
    has already rendered.
    """

    def __init__(
        self,
        kind: GenerationErrorKind,
        message: str,
        exc: BaseException | None = None,
        attempts: int = 1,
        emitted: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.attempts = attempts
        self.emitted = emitted
        self.exception_type = type(exc).__name__ if exc is not None else None

    def as_error(self) -> GenerationError:
        """The public :class:`GenerationError` for this failure."""
        return GenerationError(
            kind=self.kind,
            message=self.message,
            attempts=self.attempts,
            exception_type=self.exception_type,
        )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedAnswer:
    """A generated answer with its validated citations.

    Attributes:
        answer_text: The model's answer, verbatim.  Empty when the call
            failed.
        report: The full citation analysis -- every marker, the claims, and
            the coverage ratio.
        provider: The provider that was called.
        model: The model that was called.
        error: ``None`` on success; otherwise why the call failed.
        raw_response: The unmodified response text, kept for debugging and
            for the Issue 35 evaluator.
        attempts: How many provider calls were made.
        latency_ms: Wall-clock time spent in the provider call(s).
        metadata: Free-form extras (streaming, prompt token count, ...).
    """

    answer_text: str = ""
    report: CitationReport = field(default_factory=CitationReport)
    provider: str = ""
    model: str = ""
    error: GenerationError | None = None
    raw_response: str = ""
    attempts: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """``True`` when the model answered."""
        return self.error is None

    @property
    def citations(self) -> tuple[Citation, ...]:
        """The valid citations, deduplicated -- what a UI should render."""
        return self.report.unique_citations

    @property
    def all_citations(self) -> tuple[Citation, ...]:
        """Every marker found in the answer, in order, valid or not."""
        return self.report.citations

    @property
    def invalid_citations(self) -> tuple[Citation, ...]:
        """Citations that could not be backed by the retrieved code."""
        return self.report.invalid_citations

    @property
    def citation_coverage(self) -> float:
        """Share of claims backed by at least one valid citation."""
        return self.report.coverage

    @property
    def meets_coverage_target(self) -> bool:
        """``True`` when coverage reaches Issue 25's 90% target."""
        return self.report.meets_coverage_target

    @property
    def cited_files(self) -> tuple[str, ...]:
        """Distinct files the answer cites, in first-cited order."""
        return self.report.cited_files

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready view, for the Issue 26 query endpoint."""
        return {
            "answer_text": self.answer_text,
            "ok": self.ok,
            "provider": self.provider,
            "model": self.model,
            "error": self.error.to_dict() if self.error else None,
            "attempts": self.attempts,
            "latency_ms": round(self.latency_ms, 3),
            **self.report.to_dict(),
        }

    def __str__(self) -> str:
        return self.answer_text


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class AnswerGenerator:
    """Calls the configured LLM and returns an answer with checked citations.

    Args:
        provider: ``"openai"`` or ``"anthropic"``.  Defaults to
            ``settings.llm_provider``.
        model: The model id.  Defaults to the configured model for the
            provider.
        llm: A pre-built ``(payload) -> str`` callable that replaces the
            provider client.  This is the test seam -- with it the
            generator needs no API key and no network.
        temperature: Sampling temperature.  Defaults to ``0.0``: an answer
            about code should be reproducible, and creative paraphrase is
            how citations drift off the retrieved lines.
        max_output_tokens: Cap on the completion length, passed to the
            provider when set.
        max_retries: Additional attempts after a retryable failure.
        backoff_seconds: Delay before the first retry.
        backoff_multiplier: Factor the delay grows by per retry.
        timeout_seconds: Per-request timeout handed to the provider client.
        use_chat_messages: When ``True`` (default) a
            :class:`~reporag.generation.prompt_builder.BuiltPrompt` is sent
            as system/user chat messages rather than one flattened string.
        prompt_builder: The Issue 24 builder used by
            :meth:`generate_from_results`.  When ``None`` one is created
            lazily for this generator's model.
        sleep: The delay function used between retries.  Injectable so
            tests do not actually wait.

    Raises:
        ValueError: If *provider* is not a supported provider, or if the
            retry/backoff settings are negative.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        llm: LLMCallable | None = None,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_seconds: float = _DEFAULT_BACKOFF_SECONDS,
        backoff_multiplier: float = _DEFAULT_BACKOFF_MULTIPLIER,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        use_chat_messages: bool = True,
        prompt_builder: PromptBuilder | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        resolved_provider = (provider or settings.llm_provider).strip().lower()
        if resolved_provider not in ("openai", "anthropic"):
            raise ValueError(
                f"provider must be 'openai' or 'anthropic', got {provider!r}."
            )
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries!r}.")
        if backoff_seconds < 0:
            raise ValueError(f"backoff_seconds must be >= 0, got {backoff_seconds!r}.")
        if backoff_multiplier < 1:
            raise ValueError(
                f"backoff_multiplier must be >= 1, got {backoff_multiplier!r}."
            )
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError(
                f"max_output_tokens must be positive, got {max_output_tokens!r}."
            )

        self.provider: Provider = cast(Provider, resolved_provider)
        self.model = model or self._default_model(self.provider)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.backoff_multiplier = backoff_multiplier
        self.timeout_seconds = timeout_seconds
        self.use_chat_messages = use_chat_messages

        self._sleep = sleep
        self._llm: LLMCallable | None = llm
        self._client: Any | None = None
        self._loaded = llm is not None
        self._prompt_builder = prompt_builder

    @staticmethod
    def _default_model(provider: str) -> str:
        """Return the configured model id for *provider*."""
        if provider == "anthropic":
            return settings.anthropic_model
        return settings.openai_model

    # ------------------------------------------------------------------
    # Lazy client construction
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """``True`` once the LLM has been resolved (or found unavailable)."""
        return self._loaded

    def _ensure_loaded(self) -> LLMCallable | None:
        """Resolve the LLM callable, building a real client if needed.

        A pre-injected callable is respected, so tests never touch the
        network.  Returns ``None`` when no usable API key is configured;
        the caller turns that into an ``auth`` error rather than raising,
        which keeps a misconfigured deployment answering with a clean 503.
        """
        if self._loaded:
            return self._llm

        api_key = (
            settings.anthropic_api_key
            if self.provider == "anthropic"
            else settings.openai_api_key
        )
        if _is_unset_secret(api_key):
            logger.warning(
                "AnswerGenerator: no API key configured for provider '%s'; "
                "generation is unavailable.",
                self.provider,
            )
            self._loaded = True
            return None

        self._client = self._build_client()
        self._llm = self._invoke_client
        self._loaded = True
        return self._llm

    def _build_client(self) -> Any:
        """Construct the langchain chat client for the active provider.

        The client's own retry logic is disabled (``max_retries=0``) so that
        this class's backoff policy is the only one in play -- two
        independent retry loops would multiply, not add.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "timeout": self.timeout_seconds,
            "max_retries": 0,
        }
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens

        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value(), **kwargs
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(api_key=settings.openai_api_key.get_secret_value(), **kwargs)

    def _invoke_client(self, payload: Any) -> str:
        """Call the langchain client and return the response text."""
        response = self._client.invoke(payload)
        return _response_text(response)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: Any,
        context_chunks: Any = None,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> GeneratedAnswer:
        """Generate an answer for *prompt* and validate its citations.

        Args:
            prompt: The built prompt -- a
                :class:`~reporag.generation.prompt_builder.BuiltPrompt`, a
                plain string, or a chat-messages list.
            context_chunks: The retrieved code to validate citations
                against: retrieval results, an assembled context string, or
                a prepared :class:`~reporag.generation.citation.ContextIndex`.
                When omitted and *prompt* is a ``BuiltPrompt``, the prompt's
                own context section is used -- which is the right default,
                because the model can only honestly cite what the prompt
                actually carried.
            on_token: Called with each delta as it arrives.  Passing it
                turns on streaming when the client supports it; the fully
                validated answer is still returned at the end.

        Returns:
            A :class:`GeneratedAnswer`.  Provider failures are reported on
            :attr:`GeneratedAnswer.error`, not raised.

        Raises:
            ValueError: If *prompt* is empty or of an unsupported type.
        """
        payload = self._payload_for(prompt)
        index = self._index_for(prompt, context_chunks)

        llm = self._ensure_loaded()
        if llm is None:
            return self._failure(
                GenerationError(
                    kind="auth",
                    message=(
                        f"No API key configured for provider '{self.provider}'. "
                        f"Set {self.provider.upper()}_API_KEY to enable generation."
                    ),
                    attempts=0,
                ),
                attempts=0,
                latency_ms=0.0,
            )

        started = time.perf_counter()
        try:
            text, attempts = self._call_with_retries(llm, payload, on_token)
        except _AttemptError as failure:
            latency_ms = (time.perf_counter() - started) * 1000
            logger.error(
                "AnswerGenerator: generation failed after %d attempt(s): %s",
                failure.attempts,
                failure.message,
            )
            return self._failure(
                failure.as_error(),
                attempts=failure.attempts,
                latency_ms=latency_ms,
            )

        latency_ms = (time.perf_counter() - started) * 1000
        report = extract_citations(text, index)
        if report.invalid_citations:
            logger.warning(
                "AnswerGenerator: %d of %d citation(s) could not be backed by "
                "the retrieved context: %s",
                len(report.invalid_citations),
                len(report.citations),
                ", ".join(
                    f"{citation.marker} ({citation.status})"
                    for citation in report.invalid_citations
                ),
            )
        if report.claims and not report.meets_coverage_target:
            # Issue 25 targets 90% of claims carrying a citation; below that
            # the answer is drifting off the retrieved code, which is what
            # the Issue 35 faithfulness eval will score it down for.
            logger.warning(
                "AnswerGenerator: citation coverage %.0f%% is below the %.0f%% "
                "target (%d of %d claims uncited).",
                report.coverage * 100,
                MIN_CITATION_COVERAGE * 100,
                len(report.uncited_claims),
                len(report.claims),
            )

        return GeneratedAnswer(
            answer_text=text.strip(),
            report=report,
            provider=self.provider,
            model=self.model,
            raw_response=text,
            attempts=attempts,
            latency_ms=latency_ms,
            metadata={
                "streamed": on_token is not None,
                "indexed_files": len(index.files),
                "prompt_tokens": getattr(prompt, "token_count", None),
            },
        )

    def generate_from_results(
        self,
        query: str,
        results: Sequence[Any],
        query_type: Any = "multi-hop",
        *,
        sub_query_answers: Any = None,
        on_token: Callable[[str], None] | None = None,
    ) -> GeneratedAnswer:
        """Build a prompt from retrieval results, then generate and validate.

        The convenience path from retrieval straight to a cited answer: the
        Issue 24 builder assembles and fits the context, and the citations
        are validated against the context that survived that fit -- so a
        chunk trimmed out of the prompt cannot be cited as if the model had
        seen it.

        Args:
            query: The user's question.
            results: The retrieval results to ground the answer in.
            query_type: ``simple-lookup``, ``multi-hop`` or ``exploratory``.
            sub_query_answers: Prior sub-query findings, for multi-hop.
            on_token: Streaming callback, as for :meth:`generate`.

        Returns:
            A :class:`GeneratedAnswer`.
        """
        if self._prompt_builder is None:
            self._prompt_builder = PromptBuilder(model=self.model)
        built = self._prompt_builder.build_from_results(
            query,
            query_type=query_type,
            results=list(results),
            sub_query_answers=sub_query_answers,
        )
        return self.generate(built, on_token=on_token)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_with_retries(
        self,
        llm: LLMCallable,
        payload: Any,
        on_token: Callable[[str], None] | None,
    ) -> tuple[str, int]:
        """Call the LLM, retrying retryable failures with exponential backoff.

        Raises:
            _AttemptError: When every allowed attempt has failed.  The
                exception carries the classification and the attempt count.
        """
        delay = self.backoff_seconds

        for attempt in range(1, self.max_retries + 2):
            try:
                if on_token is not None:
                    text = self._stream(llm, payload, on_token)
                else:
                    text = llm(payload)
                if not isinstance(text, str) or not text.strip():
                    raise _AttemptError(
                        "empty_response", "The model returned an empty response."
                    )
                return text, attempt
            except _AttemptError as failure:
                current = failure
            except Exception as exc:  # noqa: BLE001 - classified, not swallowed
                current = _AttemptError(classify_error(exc), str(exc) or repr(exc), exc)

            current.attempts = attempt
            if not is_retryable(current.kind) or attempt > self.max_retries:
                raise current
            if current.emitted:
                # Deltas already reached the caller; a retry would replay the
                # answer from the top and corrupt what they already have.
                logger.warning(
                    "AnswerGenerator: stream failed after emitting tokens (%s); "
                    "not retrying.",
                    current.message,
                )
                raise current

            logger.warning(
                "AnswerGenerator: attempt %d failed (%s: %s); retrying in %.2fs.",
                attempt,
                current.kind,
                current.message,
                delay,
            )
            if delay:
                self._sleep(delay)
            delay *= self.backoff_multiplier

        raise _AttemptError(  # pragma: no cover - loop always returns or raises
            "unknown", "Generation exhausted every attempt.", None, self.max_retries + 1
        )

    def _stream(
        self,
        llm: LLMCallable,
        payload: Any,
        on_token: Callable[[str], None],
    ) -> str:
        """Stream the response, forwarding deltas to *on_token*.

        The streaming source is the provider client, or the injected ``llm``
        when it exposes a ``stream`` method -- so a fake client streams in
        tests exactly as the real one does.  A plain callable with no
        ``stream`` falls back to a single call, and ``on_token`` still sees
        the answer, so a caller never has to ask which mode it got.

        Raises:
            _AttemptError: If the stream breaks, tagged with whether any
                delta had already reached the caller.
        """
        source = self._client if self._client is not None else llm
        stream = getattr(source, "stream", None)
        if not callable(stream):
            text = llm(payload)
            if isinstance(text, str) and text:
                on_token(text)
            return text

        parts: list[str] = []
        try:
            for chunk in stream(payload):
                delta = _response_text(chunk)
                if not delta:
                    continue
                parts.append(delta)
                on_token(delta)
        except Exception as exc:  # noqa: BLE001 - classified, not swallowed
            raise _AttemptError(
                classify_error(exc),
                str(exc) or repr(exc),
                exc,
                emitted=bool(parts),
            ) from exc
        return "".join(parts)

    def _payload_for(self, prompt: Any) -> Any:
        """Normalize *prompt* into what the client should be handed."""
        messages = getattr(prompt, "messages", None)
        if messages is not None and getattr(prompt, "text", None) is not None:
            return messages if self.use_chat_messages else prompt.text

        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValueError("prompt must be a non-empty string.")
            return prompt

        if isinstance(prompt, Sequence) and not isinstance(prompt, str | bytes):
            payload = list(prompt)
            if not payload:
                raise ValueError("prompt must contain at least one message.")
            if all(isinstance(item, Mapping) for item in payload):
                return payload

        raise ValueError(
            "prompt must be a BuiltPrompt, a non-empty string, or a list of "
            f"chat messages, got {type(prompt).__name__}."
        )

    @staticmethod
    def _index_for(prompt: Any, context_chunks: Any) -> ContextIndex:
        """Pick the context to validate citations against.

        An explicit *context_chunks* always wins.  Otherwise a
        ``BuiltPrompt`` validates against its own context section, and a
        bare string prompt leaves citations unverified rather than
        pretending they were checked.
        """
        if context_chunks is not None:
            return build_context_index(context_chunks)
        if isinstance(getattr(prompt, "sections", None), Mapping):
            return build_context_index(prompt)
        return ContextIndex()

    def _failure(
        self, error: GenerationError, *, attempts: int, latency_ms: float
    ) -> GeneratedAnswer:
        """Wrap a :class:`GenerationError` in an empty result."""
        return GeneratedAnswer(
            provider=self.provider,
            model=self.model,
            error=error,
            attempts=attempts,
            latency_ms=latency_ms,
        )

    def __repr__(self) -> str:
        return (
            f"AnswerGenerator(provider={self.provider!r}, model={self.model!r}, "
            f"max_retries={self.max_retries}, loaded={self._loaded})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _response_text(response: Any) -> str:
    """Extract the text from a langchain response or chunk.

    Handles the plain-string case, the ``.content`` attribute of a message,
    and the content-block list that Anthropic models return for structured
    output.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def _is_unset_secret(secret: Any) -> bool:
    """``True`` when *secret* is empty or still a documented placeholder.

    Mirrors :func:`reporag.agent.planner._is_unset_secret` rather than
    importing it, keeping the generation package independent of the agent
    package's internals.
    """
    from pydantic import SecretStr

    value = (
        secret.get_secret_value() if isinstance(secret, SecretStr) else str(secret)
    ).strip()
    return value in {
        "",
        "change-me",
        "change-me-to-a-random-string",
        "sk-your-key-here",
        "sk-ant-your-key-here",
    }
