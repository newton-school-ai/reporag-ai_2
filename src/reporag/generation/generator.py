"""LLM answer generator with citation extraction.

Calls the configured LLM (OpenAI or Anthropic) with the built prompt,
handles rate limits, timeouts, and API errors, and returns a structured
GenerationResult containing answer text and validated citations.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from reporag.config import settings
from reporag.generation.citation import Citation, CitationExtractor
from reporag.generation.prompt_builder import BuiltPrompt
from reporag.retrieval.vector_search import RetrievalResult

logger = logging.getLogger(__name__)

# Max retries for transient LLM API errors
_DEFAULT_MAX_RETRIES = 2
_INITIAL_BACKOFF_SEC = 1.0


@dataclass(frozen=True)
class GenerationResult:
    """Structured output returned by AnswerGenerator.

    Attributes:
        answer_text: Generated response text from the LLM.
        citations: List of validated line-level citations extracted from the response.
        coverage: Ratio of valid citations to total citations (0.0 to 1.0).
        raw_response: Full unparsed response string from the LLM.
        provider: The LLM provider used ('openai' or 'anthropic').
        model: The LLM model name used.
    """

    answer_text: str
    citations: list[Citation] = field(default_factory=list)
    coverage: float = 1.0
    raw_response: str = ""
    provider: str = "openai"
    model: str = "gpt-4o"


class AnswerGenerator:
    """Calls configured LLM provider and extracts line-level citations against context.

    Args:
        provider: 'openai' or 'anthropic' (default: settings.llm_provider).
        model: LLM model name (default: configured model for provider).
        api_key: Optional API key override.
        timeout: Maximum seconds to wait for LLM completion (default: 30.0).
        max_retries: Number of exponential backoff retries for API errors (default: 2).
        llm_client: Optional callable override (prompt: str) -> str for testing.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        llm_client: Callable[[str], str] | None = None,
    ) -> None:
        self.provider = (provider or settings.llm_provider).lower()
        if self.provider not in ("openai", "anthropic"):
            raise ValueError(
                f"Unsupported provider {self.provider!r}. Must be 'openai' or 'anthropic'"
            )

        if self.provider == "openai":
            self.model = model or settings.openai_model
            self._api_key = api_key or settings.openai_api_key.get_secret_value()
        else:
            self.model = model or settings.anthropic_model
            self._api_key = api_key or settings.anthropic_api_key.get_secret_value()

        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self._custom_llm_client = llm_client
        self._extractor = CitationExtractor()

    def _call_llm_with_retry(self, prompt_text: str) -> str:
        """Execute LLM call with exponential backoff retry for transient errors."""
        if self._custom_llm_client is not None:
            return self._custom_llm_client(prompt_text)

        if (
            not self._api_key
            or self._api_key.startswith("sk-your-key")
            or self._api_key.startswith("sk-ant-your-key")
        ):
            logger.warning(
                "No valid API key configured for provider %s. Returning fallback response.",
                self.provider,
            )
            return (
                f"[FALLBACK RESPONSE] Configured provider '{self.provider}' with model '{self.model}' "
                f"requires an API key. Prompt received for processing."
            )

        last_err: Exception | None = None
        backoff = _INITIAL_BACKOFF_SEC

        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == "anthropic":
                    from langchain_anthropic import ChatAnthropic

                    client = ChatAnthropic(
                        model=self.model,
                        api_key=self._api_key,
                        timeout=self.timeout,
                        temperature=0.0,
                    )
                    res = client.invoke(prompt_text)
                    return str(getattr(res, "content", res))
                else:
                    from langchain_openai import ChatOpenAI

                    client = ChatOpenAI(
                        model=self.model,
                        api_key=self._api_key,
                        request_timeout=self.timeout,
                        temperature=0.0,
                    )
                    res = client.invoke(prompt_text)
                    return str(getattr(res, "content", res))

            except Exception as err:
                last_err = err
                if attempt < self.max_retries:
                    logger.warning(
                        "%s API call attempt %d/%d failed: %s. Retrying in %.1fs...",
                        self.provider.capitalize(),
                        attempt + 1,
                        self.max_retries + 1,
                        err,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    logger.error(
                        "%s API call failed after %d attempts: %s",
                        self.provider.capitalize(),
                        self.max_retries + 1,
                        err,
                        exc_info=True,
                    )

        raise RuntimeError(
            f"{self.provider.capitalize()} API error: {last_err}"
        ) from last_err

    def generate(
        self,
        prompt: BuiltPrompt | str,
        context_chunks: (
            Sequence[RetrievalResult] | str | Sequence[dict[str, Any]] | None
        ) = None,
    ) -> GenerationResult:
        """Generate answer from prompt and extract validated line-level citations.

        Args:
            prompt: BuiltPrompt object or formatted prompt string.
            context_chunks: Retrieved context chunks for validating citations.

        Returns:
            A :class:`GenerationResult` containing answer, validated citations, and metadata.
        """
        if isinstance(prompt, BuiltPrompt):
            prompt_text = prompt.full_prompt
        else:
            prompt_text = str(prompt)

        try:
            raw_text = self._call_llm_with_retry(prompt_text)
        except Exception as err:
            logger.error("LLM generation encountered an error: %s", err)
            fallback_text = f"Error generating answer: {err}"
            return GenerationResult(
                answer_text=fallback_text,
                citations=[],
                coverage=0.0,
                raw_response=fallback_text,
                provider=self.provider,
                model=self.model,
            )

        citation_result = self._extractor.extract_and_validate(raw_text, context_chunks)

        return GenerationResult(
            answer_text=raw_text,
            citations=citation_result.citations,
            coverage=citation_result.citation_coverage,
            raw_response=raw_text,
            provider=self.provider,
            model=self.model,
        )
