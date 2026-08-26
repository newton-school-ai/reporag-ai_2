"""LLM answer generator (Issue 25).

Calls the configured LLM (OpenAI or Anthropic, via langchain, mirroring
:func:`reporag.agent.planner._build_langchain_llm`'s provider-selection
pattern) with a built prompt and returns the raw response text, or a
structured failure when the call could not be completed.

Why
---
This is the one place in the pipeline that makes a real network call to a
third-party API, so it is also the one place several classes of failure
that don't exist anywhere else in the pipeline can happen: the provider is
slow or unreachable, the request is rate-limited, the API key is missing or
wrong, or the provider returns something that isn't usable text. None of
these are programming errors -- they're normal, expected conditions for
calling an external service -- so they are handled here and turned into a
:class:`GenerationResult` the caller can inspect, not exceptions that
propagate up and crash a request.

Design
------
* **One retry, then a structured failure, never a raised exception** --
  the same shape :class:`~reporag.agent.executor.SubQueryExecutor` uses for
  its own engine calls (one retry on failure, then mark the unit of work
  failed and let the caller continue). A citation-extraction pipeline that
  raises on a transient network blip takes down the whole request for a
  problem that would have gone away a second later.
* **A timeout this module enforces itself**, via a background thread and a
  bounded wait, rather than trusting a provider-specific timeout
  constructor argument. langchain's OpenAI/Anthropic wrappers have each
  changed their timeout parameter's name and default across versions; a
  self-enforced timeout works the same way regardless, and is exactly what
  the issue means by "handle timeout" -- this module guaranteeing it will
  give up by a deadline, not merely hoping the SDK does.
* **Failures are classified, not just captured** -- ``error_kind`` sorts a
  failure into ``timeout``, ``rate_limit``, ``auth``, ``invalid_response``,
  or ``api_error`` by inspecting the exception's type name and message
  rather than importing every provider SDK's specific exception classes.
  This keeps the module working with either provider (or a future third
  one) without a hard dependency on either SDK's exception hierarchy, the
  same "duck type the collaborator" approach
  :class:`~reporag.agent.router.StrategyRouter` and
  :class:`~reporag.agent.executor.SubQueryExecutor` use for their retrieval
  engines.
* **The LLM client is injectable** -- the constructor accepts an ``llm``
  callable (``(system: str, user: str) -> str``) in place of building a
  real langchain client, so every test in this module runs fully offline,
  matching this codebase's established pattern for testing LLM-adjacent
  code (see e.g. :mod:`tests.unit.test_planner`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from reporag.config import settings
from reporag.generation.citation import CitationReport, analyze_citations

if TYPE_CHECKING:
    from reporag.generation.prompt_builder import BuiltPrompt

logger = logging.getLogger(__name__)

__all__ = [
    "AnswerGenerator",
    "AnsweredQuery",
    "GenerationResult",
    "LLMChatCallable",
]

# A provider-agnostic "(system, user) -> response text" callable -- what a
# real langchain client is wrapped into, and what a test can inject
# directly in its place.
LLMChatCallable = Callable[[str, str], str]

ErrorKind = Literal["timeout", "rate_limit", "auth", "invalid_response", "api_error"]

_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_RETRIES = 1

# Shared across every `AnswerGenerator` instance rather than one executor
# (and one background thread) per instance -- a generator is cheap to
# construct (e.g. one per request in a web handler), and a per-instance
# executor would leak a thread for each one that's never explicitly closed.
_TIMEOUT_EXECUTOR = ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="reporag-llm-call"
)

# Substrings checked (case-insensitively) against an exception's type name
# and message to classify it, in priority order -- checked top to bottom,
# first match wins, so e.g. an exception whose message happens to mention
# both "rate limit" and "timeout" is classified by whichever this table
# checks first (rate limit, since a provider that's about to time out is a
# less actionable diagnosis than one that's telling you exactly why it
# declined the request).
_ERROR_KIND_MARKERS: tuple[tuple[str, ErrorKind], ...] = (
    ("ratelimit", "rate_limit"),
    ("rate_limit", "rate_limit"),
    ("rate limit", "rate_limit"),
    ("429", "rate_limit"),
    ("authenticat", "auth"),
    ("permission", "auth"),
    ("unauthorized", "auth"),
    ("401", "auth"),
    ("403", "auth"),
    ("api_key", "auth"),
    ("apikey", "auth"),
    ("timeout", "timeout"),
    ("timed out", "timeout"),
    ("empty response", "invalid_response"),
    ("invalid response", "invalid_response"),
    ("invalid_response", "invalid_response"),
)


@dataclass
class GenerationResult:
    """The outcome of one :meth:`AnswerGenerator.generate` call.

    Attributes:
        text: The raw response text. Empty when ``success`` is ``False``.
        success: ``True`` when a usable response was obtained.
        error: A human-readable description of the failure, or ``None`` on
            success.
        error_kind: A rough classification of the failure (see
            :data:`ErrorKind`), or ``None`` on success.
        model: The model name used for this call.
        provider: ``"openai"`` or ``"anthropic"``.
        latency_seconds: Wall-clock time the call took, including any
            retry. ``0.0`` if the call never started.
        retries: How many retries were attempted (``0`` means the first
            attempt succeeded).
    """

    text: str = ""
    success: bool = False
    error: str | None = None
    error_kind: ErrorKind | None = None
    model: str = ""
    provider: str = ""
    latency_seconds: float = 0.0
    retries: int = 0


@dataclass
class AnsweredQuery:
    """The structured ``{answer, citations}`` result the issue asks for.

    Bundles a :meth:`AnswerGenerator.generate` call with citation
    extraction and validation (:func:`~reporag.generation.citation.analyze_citations`)
    against the same context the prompt showed the model, so a caller gets
    one object with everything needed to render an answer and its
    citations, or to detect that generation failed at all.

    Attributes:
        answer: The raw answer text (empty on generation failure).
        citations: The full citation report -- empty citations list and
            ``coverage=1.0`` when generation failed, since there is no
            answer text to have made claims in.
        generation: The underlying :class:`GenerationResult`, for anything
            not already surfaced above (timing, retry count, the specific
            failure).
    """

    answer: str
    citations: CitationReport
    generation: GenerationResult

    @property
    def success(self) -> bool:
        """``True`` when generation succeeded (independent of citation validity)."""
        return self.generation.success


def _classify_error(exc: Exception) -> ErrorKind:
    """Classify *exc* into an :data:`ErrorKind` by inspecting its type and message.

    Avoids importing any provider SDK's specific exception classes (see the
    module docstring) -- checked against :data:`_ERROR_KIND_MARKERS`, in
    order, falling back to ``"api_error"`` for anything unrecognised, which
    still tells a caller "the provider call failed" without pretending to
    know a specific cause it can't verify without those imports.
    """
    haystack = f"{type(exc).__name__} {exc}".lower()
    for marker, kind in _ERROR_KIND_MARKERS:
        if marker in haystack:
            return kind
    return "api_error"


def _build_llm_client(model: str | None) -> LLMChatCallable:
    """Build a provider-agnostic ``(system, user) -> text`` langchain client.

    Mirrors :func:`reporag.agent.planner._build_langchain_llm`'s
    provider-selection logic (``settings.llm_provider``), but keeps the
    system/user role split langchain's chat message API gives instead of
    collapsing to one string, since a generation call benefits from the
    model actually seeing the citation rules and grounding instructions as
    a system message rather than folded into the user turn.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        client = ChatAnthropic(
            model=model or settings.anthropic_model,
            api_key=settings.anthropic_api_key.get_secret_value(),
            temperature=0.0,
        )
    else:
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(
            model=model or settings.openai_model,
            api_key=settings.openai_api_key.get_secret_value(),
            temperature=0.0,
        )

    def _invoke(system: str, user: str) -> str:
        response = client.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return str(getattr(response, "content", response))

    return _invoke


class AnswerGenerator:
    """Calls the configured LLM and (optionally) extracts citations.

    Args:
        model: Model name override. Defaults to the configured provider's
            default model (``settings.openai_model`` or
            ``settings.anthropic_model``).
        timeout_seconds: How long to wait for one attempt before treating
            it as a timeout failure. Enforced by this module directly (see
            the module docstring), not delegated to the provider SDK.
        max_retries: How many additional attempts to make after the first
            one fails. ``0`` disables retrying.
        llm: A pre-built ``(system: str, user: str) -> str`` callable to
            use instead of constructing a real langchain client -- for
            dependency injection in tests, or to share one client across
            several generators.

    Raises:
        ValueError: If *timeout_seconds* is not positive, or *max_retries*
            is negative.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        llm: LLMChatCallable | None = None,
    ) -> None:
        """Initialise the generator with retry/timeout policy and provider."""
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}.")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries!r}.")
        self.provider = settings.llm_provider
        self.model = model or (
            settings.anthropic_model
            if self.provider == "anthropic"
            else settings.openai_model
        )
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: BuiltPrompt | str, user: str = "") -> GenerationResult:
        """Call the LLM with *prompt* and return the raw response.

        Args:
            prompt: A :class:`~reporag.generation.prompt_builder.BuiltPrompt`
                (its ``.system``/``.user`` are used directly), or a plain
                system-message string paired with *user*.
            user: The user-message text, when *prompt* is a plain string
                rather than a :class:`BuiltPrompt`. Ignored when *prompt*
                is a :class:`BuiltPrompt`.

        Returns:
            A :class:`GenerationResult`. Never raises for a call-time
            failure (network error, timeout, rate limit, bad response) --
            those are reported via ``success=False`` and ``error_kind``.
            Only a genuine misuse (an unusable *prompt* argument) raises.
        """
        system, user_text = self._resolve_messages(prompt, user)
        llm = self._llm or _build_llm_client(self.model)

        start = time.monotonic()
        last_error: Exception | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                text = self._invoke_with_timeout(llm, system, user_text)
                if not text or not text.strip():
                    raise ValueError("LLM returned an empty response.")
                return GenerationResult(
                    text=text,
                    success=True,
                    model=self.model,
                    provider=self.provider,
                    latency_seconds=time.monotonic() - start,
                    retries=attempt,
                )
            except Exception as exc:  # noqa: BLE001 -- classified below, not swallowed
                last_error = exc
                if attempt < attempts - 1:
                    logger.warning(
                        "AnswerGenerator: attempt %d/%d failed (%s); retrying.",
                        attempt + 1,
                        attempts,
                        exc,
                    )

        assert last_error is not None  # the loop only exits via return or here
        logger.warning(
            "AnswerGenerator: all %d attempt(s) failed; giving up (%s).",
            attempts,
            last_error,
        )
        return GenerationResult(
            text="",
            success=False,
            error=str(last_error),
            error_kind=_classify_error(last_error),
            model=self.model,
            provider=self.provider,
            latency_seconds=time.monotonic() - start,
            retries=attempts - 1,
        )

    def generate_with_citations(
        self, prompt: BuiltPrompt, context: str | None = None
    ) -> AnsweredQuery:
        """Generate an answer for *prompt* and validate its citations in one call.

        The structured ``{answer, citations}`` result the issue asks for.

        Args:
            prompt: The :class:`~reporag.generation.prompt_builder.BuiltPrompt`
                to answer. Its own context (``prompt.sections["context"]``)
                is used for citation validation unless *context* overrides
                it.
            context: An explicit context string to validate citations
                against, overriding *prompt*'s own context. Rarely needed
                -- mainly for a caller that assembled context separately
                from what it put in the prompt.

        Returns:
            An :class:`AnsweredQuery`. When generation fails, ``answer`` is
            ``""`` and ``citations`` is an empty, vacuously-full-coverage
            report -- check ``.success`` (or ``.generation.success``)
            before treating an empty citation list as "no citations found
            in a real answer".
        """
        result = self.generate(prompt)
        if not result.success:
            return AnsweredQuery(
                answer="",
                citations=CitationReport(),
                generation=result,
            )
        report = analyze_citations(result.text, context or prompt)
        return AnsweredQuery(answer=result.text, citations=report, generation=result)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_messages(prompt: BuiltPrompt | str, user: str) -> tuple[str, str]:
        """Resolve *prompt* (a `BuiltPrompt` or a plain string) to (system, user)."""
        system = getattr(prompt, "system", None)
        if system is not None:
            return system, getattr(prompt, "user", "")
        return str(prompt), user

    def _invoke_with_timeout(self, llm: LLMChatCallable, system: str, user: str) -> str:
        """Call *llm* with a self-enforced deadline (see the module docstring).

        Uses the shared :data:`_TIMEOUT_EXECUTOR` rather than a
        per-instance one -- see that constant's docstring for why.
        """
        future = _TIMEOUT_EXECUTOR.submit(llm, system, user)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"LLM call did not complete within {self.timeout_seconds}s."
            ) from exc

    def __repr__(self) -> str:
        return (
            f"AnswerGenerator(provider={self.provider!r}, model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds}, max_retries={self.max_retries})"
        )
