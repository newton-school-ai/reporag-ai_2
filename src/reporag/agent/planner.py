"""Agentic query planner.

Contains the query classifier and query decomposer. Classifies queries
into simple-lookup / multi-hop / exploratory, then decomposes complex
queries into ordered sub-queries using a LangGraph state machine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from reporag.config import settings

logger = logging.getLogger(__name__)


class QueryType(StrEnum):
    """Categories of queries supported by the agentic pipeline."""

    SIMPLE_LOOKUP = "simple-lookup"
    MULTI_HOP = "multi-hop"
    EXPLORATORY = "exploratory"


@dataclass(frozen=True)
class Classification:
    """The result of query classification."""

    query_type: QueryType
    confidence: float


CLASSIFY_PROMPT = """You are an expert query classifier for a code repository RAG system.
Your job is to classify the user's software engineering query into exactly one of three categories:
- "simple-lookup": The user wants to locate a specific definition, class, file, or exact symbol (e.g., "Where is X defined?", "Show me the auth middleware.").
- "multi-hop": The user asks a structural, tracing, or "how does it work" question that requires combining multiple pieces of code (e.g., "How does a request go from API to DB?", "What calls function Y?").
- "exploratory": The user asks a broad, high-level, or architectural question (e.g., "Explain the architecture of the ingestion pipeline.", "What is the overall testing strategy?").

You must return ONLY valid JSON matching this schema:
{{"query_type": "string", "confidence": float}}

Do not include markdown code blocks. Do not include any explanations.

Examples:
User: "Where is authenticate defined?"
{{"query_type": "simple-lookup", "confidence": 0.95}}

User: "How does auth work end-to-end?"
{{"query_type": "multi-hop", "confidence": 0.92}}

User: "Explain the architecture"
{{"query_type": "exploratory", "confidence": 0.90}}

User: "Tell me about the code"
{{"query_type": "multi-hop", "confidence": 0.45}}

User: "{query}"
"""


class QueryClassifier:
    """LLM-based query classifier.

    Classifies queries into simple-lookup, multi-hop, or exploratory.
    If the LLM is unconfident or fails, it falls back safely to multi-hop.
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        fallback_threshold: float = 0.6,
    ) -> None:
        self.fallback_threshold = fallback_threshold
        self._llm_client = llm_client

    def _ensure_loaded(self) -> None:
        """Lazily instantiate the LLM client using settings if not injected."""
        if self._llm_client is not None:
            return

        if settings.llm_provider == "anthropic":
            import anthropic

            logger.info("Initializing Anthropic client for QueryClassifier")
            self._llm_client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
        else:
            import openai

            logger.info("Initializing OpenAI client for QueryClassifier")
            self._llm_client = openai.Client(
                api_key=settings.openai_api_key.get_secret_value()
            )

    def classify(self, query: str) -> Classification:
        """Classify a query into a QueryType with a confidence score."""
        query = query.strip()
        if not query:
            return Classification(QueryType.MULTI_HOP, 0.0)

        self._ensure_loaded()
        prompt = CLASSIFY_PROMPT.format(query=query)

        try:
            raw_response = self._invoke_llm(prompt)
            return self._parse_response(raw_response)
        except Exception as e:
            # Allow programming errors to surface, catch only expected runtime failures
            if isinstance(
                e, TypeError | AttributeError | NameError | ImportError | RuntimeError
            ):
                raise
            logger.warning(
                "Query classification failed (%s). Falling back to multi-hop.",
                type(e).__name__,
            )
            return Classification(QueryType.MULTI_HOP, 0.0)

    def _invoke_llm(self, prompt: str) -> str:
        """Call the underlying LLM client."""
        # If it's a test mock (Callable), just call it
        if callable(self._llm_client):
            return self._llm_client(prompt)

        try:
            import openai

            if isinstance(self._llm_client, openai.Client):
                response = self._llm_client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=100,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content or "{}"
        except ImportError:
            pass

        try:
            import anthropic

            if isinstance(self._llm_client, anthropic.Anthropic):
                resp = self._llm_client.messages.create(
                    model=settings.anthropic_model,
                    max_tokens=100,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                if hasattr(resp, "content") and resp.content:
                    return getattr(resp.content[0], "text", "{}")
                return "{}"
        except ImportError:
            pass

        raise RuntimeError("Unsupported llm_client type")

    def _parse_response(self, text: str) -> Classification:
        """Parse the JSON output and handle threshold fallbacks."""
        data = json.loads(text.strip())

        q_type_str = str(data["query_type"]).lower().strip()
        query_type = QueryType(q_type_str)
        confidence = float(data["confidence"])

        if not (confidence >= self.fallback_threshold):
            logger.debug(
                "Low confidence classification (%f < %f). Falling back to multi-hop.",
                confidence,
                self.fallback_threshold,
            )
            return Classification(QueryType.MULTI_HOP, 0.0)

        return Classification(query_type, confidence)


# TODO: Implement in Issue 21
#
# QueryDecomposer (Issue 21):
# - LangGraph state machine for decomposition
# - Input: complex query + repo context (modules, key symbols)
# - Output: ordered list of SubQuery objects with dependency edges
# - Each SubQuery: text, expected_answer_type, context_from (prior IDs)
# - Handles queries that do not need decomposition (single step)
