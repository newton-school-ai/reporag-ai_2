import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from langchain_core.prompts import PromptTemplate

from reporag.agent.planner import _build_langchain_llm

logger = logging.getLogger(__name__)

Strategy = Literal["bm25", "vector", "graph", "hybrid"]


@dataclass
class RoutingDecision:
    strategy: Strategy
    symbol: str | None = None


_ROUTER_PROMPT = PromptTemplate.from_template(
    """You are a query routing assistant. Your job is to determine the best retrieval strategy for a sub-query.

Strategies:
- "bm25": Use when the query asks for a specific identifier (e.g., "Where is function X defined?").
- "graph": Use when the query asks about structural relationships (e.g., "What calls function X?", "Trace the path...").
- "vector": Use when the query is conceptual or semantic (e.g., "How does authentication work?").
- "hybrid": Use when the query is ambiguous or combines multiple needs.

If the strategy is "bm25" or "graph", you MUST also extract the target identifier (symbol) from the query.

Query: {query}

Return ONLY a valid JSON object with the following schema:
{{
  "strategy": "bm25" | "vector" | "graph" | "hybrid",
  "symbol": "extracted_symbol_name" or null
}}
"""
)


def _extract_json_object(raw: str) -> str | None:
    """Extract the first balanced ``{...}`` block from *raw*."""
    start = raw.find("{")
    if start == -1:
        return None
    end = raw.rfind("}")
    if end == -1 or end < start:
        return None
    return raw[start : end + 1]


# Matches a camelCase transition (lowercase->uppercase), underscore (snake_case),
# or dot (qualified name).  Used to distinguish code identifiers from plain English.
_IDENTIFIER_RE = re.compile(r"[_.]|[a-z][A-Z]")


def _looks_like_identifier(sym: str) -> bool:
    """Return True if *sym* resembles a code identifier rather than a plain English word.

    Accepts:
    - snake_case:  ``authenticate_user``, ``foo_bar``
    - qualified:   ``UserService.login``, ``module.auth.X``
    - camelCase:   ``UserService`` (lowercase->uppercase transition inside the word)

    Rejects plain English words such as ``callers``, ``references``, ``handling``.
    Single-word PascalCase names (e.g. ``Database``) intentionally return *False*
    here; they are still captured by the *strict* patterns which require a structural
    keyword (``class``, ``function``, etc.) to precede the symbol.

    Using a structural heuristic avoids the fragility of a growing stopword list and
    aligns with the actual GraphRetriever contract: only pass concrete code symbols,
    not ordinary English words.
    """
    return bool(_IDENTIFIER_RE.search(sym))


def rule_based_route(query: str) -> RoutingDecision:
    """Fallback rule-based routing heuristics using regex."""
    bm25_strict = re.compile(
        r"\b(?:where\s+(?:is|are)|find|locate)\s+(?:the\s+)?(?:function|class|method|module)\s+([a-zA-Z0-9_.]+)",
        re.IGNORECASE,
    )
    bm25_loose = re.compile(
        r"\b(?:where\s+(?:is|are)|find|locate)\s+([a-zA-Z0-9_.]+)",
        re.IGNORECASE,
    )

    graph_strict = re.compile(
        r"\b(?:what\s+calls|trace|path\s+from|depend(?:s|encies)\s+on)\s+(?:the\s+)?(?:function|class|method|module)\s+([a-zA-Z0-9_.]+)",
        re.IGNORECASE,
    )
    graph_loose = re.compile(
        r"\b(?:what\s+calls|trace|path\s+from|depend(?:s|encies)\s+on)\s+([a-zA-Z0-9_.]+)",
        re.IGNORECASE,
    )

    vector_pattern = re.compile(
        r"\b(?:how\s+does|explain|what\s+does)\b", re.IGNORECASE
    )

    if match := graph_strict.search(query):
        return RoutingDecision(strategy="graph", symbol=match.group(1).rstrip("."))
    if match := graph_loose.search(query):
        sym = match.group(1).rstrip(".")
        if _looks_like_identifier(sym):
            return RoutingDecision(strategy="graph", symbol=sym)

    if match := bm25_strict.search(query):
        return RoutingDecision(strategy="bm25", symbol=match.group(1).rstrip("."))
    if match := bm25_loose.search(query):
        sym = match.group(1).rstrip(".")
        if _looks_like_identifier(sym):
            return RoutingDecision(strategy="bm25", symbol=sym)

    if vector_pattern.search(query):
        return RoutingDecision(strategy="vector")

    return RoutingDecision(strategy="hybrid")


class StrategyRouter:
    """Routes a sub-query to the optimal retrieval strategy."""

    def __init__(self, llm_callable: Callable[[str], str] | None = None) -> None:
        """Initialize the router with an optional LLM callable for tests."""
        self._llm = llm_callable
        self._loaded = False

    def _ensure_llm(self) -> None:
        """Lazily initialize the LLM chain if no callable was injected.

        The chain handles prompt formatting internally via LangChain's
        RunnableSequence (``_ROUTER_PROMPT | backend``).  ``route()`` therefore
        passes the *raw* query string to ``self._llm``; the chain formats the
        template exactly once before calling the underlying model.

        Note: ``_build_langchain_llm()`` returns a plain ``(str) -> str`` callable
        (see :func:`reporag.agent.planner._build_langchain_llm`).  When a plain
        function is the last step in a LangChain ``RunnableSequence``,
        ``chain.invoke(...)`` returns the function's return value directly as a
        ``str``.  Calling ``.content`` on a ``str`` would raise ``AttributeError``
        on every invocation, silently degrading to the rule-based fallback.
        """
        if self._llm is None and not self._loaded:
            chain = _ROUTER_PROMPT | _build_langchain_llm()
            # chain.invoke returns a plain str because _build_langchain_llm()
            # is a str-returning callable, not a ChatModel.  Do NOT call .content.
            self._llm = lambda q: chain.invoke({"query": q})  # type: ignore
            self._loaded = True

    def route(self, query: str) -> RoutingDecision:
        """Route the query using LLM-assisted classification with rule-based fallback."""
        try:
            self._ensure_llm()
            # Pass the raw query -- the LangChain chain formats the prompt template
            # internally.  Do NOT pre-format _ROUTER_PROMPT here; doing so and
            # passing the result as the chain's {query} variable would cause the
            # instructions to be embedded twice in the prompt the LLM receives.
            raw_response = self._llm(query)  # type: ignore
            json_str = _extract_json_object(raw_response)
            if not json_str:
                raise ValueError("No JSON object found in response")

            data = json.loads(json_str)
            strategy = data.get("strategy")
            symbol = data.get("symbol")

            if strategy not in ("bm25", "vector", "graph", "hybrid"):
                raise ValueError(f"Invalid strategy returned by LLM: {strategy}")

            return RoutingDecision(strategy=strategy, symbol=symbol)  # type: ignore
        except Exception as e:
            logger.warning("LLM routing failed, falling back to rule-based: %s", e)
            return rule_based_route(query)
