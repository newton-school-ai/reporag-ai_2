"""Strategy router.

Routes each sub-query to the optimal retrieval strategy: graph, vector,
bm25, or hybrid. Routing based on sub-query characteristics with
LLM-assisted classification and rule-based fallback.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Literal

from src.reporag.agent.planner import _build_langchain_llm, _is_unset_secret
from src.reporag.config import settings

logger = logging.getLogger(__name__)

LLMCallable = Callable[[str], str]
StrategyType = Literal["graph", "vector", "bm25", "hybrid"]


class StrategyRouter:
    """Routes a sub-query to the optimal retrieval strategy.

    Uses an LLM for classification if enabled and available; falls back
    to deterministic, keyword/pattern-based routing when LLM is unavailable.
    """

    def __init__(
        self,
        llm: LLMCallable | None = None,
        *,
        use_llm: bool | None = None,
    ) -> None:
        """Initialize the StrategyRouter.

        Args:
            llm: Pre-built LLM callable for testing / dependency injection.
            use_llm: Set to True to use LLM routing. Defaults to config settings.
        """
        self._resolved_llm = llm
        self._loaded = False
        self.use_llm = (
            use_llm if use_llm is not None else settings.query_classifier_use_llm
        )

    def _ensure_loaded(self) -> LLMCallable | None:
        """Resolve the LLM callable, lazy-constructing a real langchain client if needed."""
        if self._loaded:
            return self._resolved_llm

        if self._resolved_llm is None:
            api_key = settings.active_llm_api_key
            if _is_unset_secret(api_key):
                logger.warning(
                    "StrategyRouter: LLM is enabled but no API key is "
                    "configured for provider '%s'; falling back to rule-based "
                    "routing.",
                    settings.llm_provider,
                )
                self._loaded = True
                return None

            try:
                self._resolved_llm = _build_langchain_llm()
            except Exception as e:
                logger.warning(f"Failed to build LLM for StrategyRouter: {e}")
                self._resolved_llm = None

        self._loaded = True
        return self._resolved_llm

    def route(self, sub_query: str) -> StrategyType:
        """Route the sub-query to graph, vector, bm25, or hybrid.

        Args:
            sub_query: The sub-query text to route.

        Returns:
            The selected strategy string ("graph", "vector", "bm25", or "hybrid").
        """
        if not sub_query or not sub_query.strip():
            return "hybrid"

        # Try LLM if enabled
        if self.use_llm:
            llm = self._ensure_loaded()
            if llm is not None:
                try:
                    strategy = self._route_llm(llm, sub_query)
                    if strategy in {"graph", "vector", "bm25", "hybrid"}:
                        return strategy
                except Exception as e:
                    logger.warning(f"LLM routing failed: {e}; falling back to rules.")

        # Fallback to rules
        return self._route_rules(sub_query)

    def _route_rules(self, sub_query: str) -> StrategyType:
        """Deterministic rule-based fallback strategy classifier."""
        q = sub_query.lower().strip()

        # 1. Structural/Graph checks (tracing paths, calling relations, imports)
        structural_keywords = [
            r"\bcall\b",
            r"\bcalls\b",
            r"\bcalled\b",
            r"\bcaller(s)?\b",
            r"\bdependencies\b",
            r"\bdepends\b",
            r"\bimported\b",
            r"\bimports\b",
            r"\bparents\b",
            r"\bchildren\b",
            r"\binherits\b",
            r"\bderived\b",
            r"\bwhere is.*used\b",
            r"\breferences to\b",
        ]
        if any(re.search(pat, q) for pat in structural_keywords):
            return "graph"

        # 2. Semantic/Vector checks
        semantic_keywords = [
            r"\bhow does\b",
            r"\bwhat does\b",
            r"\bexplain\b",
            r"\bpurpose of\b",
            r"\bdescribe\b",
            r"\boverview\b",
            r"\bhow to\b",
            r"\bwhy does\b",
            r"\bunderstand\b",
            r"\bconcept\b",
        ]
        is_semantic = any(re.search(pat, q) for pat in semantic_keywords)

        # 3. Identifier/BM25 checks
        identifier_keywords = [
            r"\bdefinition of\b",
            r"\bwhere is.*defined\b",
            r"\bfind function\b",
            r"\bfind class\b",
            r"\bget function\b",
            r"\bget class\b",
            r"\blookup\b",
            r"\bdeclaration of\b",
            r"\bdeclare\b",
        ]
        is_identifier_phrase = any(re.search(pat, q) for pat in identifier_keywords)

        # Look for code identifiers in the query
        words = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_\.:]*)\b", sub_query)
        stop_words = {
            "what",
            "who",
            "where",
            "how",
            "why",
            "which",
            "function",
            "class",
            "method",
            "variable",
            "calls",
            "called",
            "caller",
            "call",
            "dependencies",
            "dependency",
            "depends",
            "import",
            "imports",
            "imported",
            "define",
            "defined",
            "definition",
            "references",
            "reference",
            "find",
            "show",
            "get",
            "lookup",
            "trace",
            "auth",
            "user",
        }

        has_code_identifier = False
        for word in words:
            if word.lower() in stop_words:
                continue
            # Underscores (e.g. auth_user)
            if "_" in word:
                has_code_identifier = True
                break
            # CamelCase transition (lowercase followed by uppercase)
            if re.search(r"[a-z][A-Z]", word):
                has_code_identifier = True
                break
            # Starts with uppercase and has lowercase (PascalCase e.g., AuthenticateUser)
            if (
                word[0].isupper()
                and len(word) > 1
                and word[1:].islower() is False
                and word.isupper() is False
            ):
                has_code_identifier = True
                break
            # Dotted or colon-separated paths (e.g. auth.authenticate)
            if "." in word or ":" in word:
                has_code_identifier = True
                break

        if is_identifier_phrase or (has_code_identifier and not is_semantic):
            return "bm25"

        if is_semantic:
            return "vector"

        # 4. Ambiguous fallback
        return "hybrid"

    def _route_llm(self, llm: LLMCallable, sub_query: str) -> StrategyType:
        """Route the query using LLM-assisted classification."""
        prompt = (
            "You are a routing agent for a codebase QA system. Given a user's sub-query, you must route it to the most specific and optimal retrieval strategy:\n"
            '- "bm25": Use this for exact identifier/keyword lookups, function/class name lookups, or finding where a specific term/symbol is defined.\n'
            '- "graph": Use this for structural or dependency queries, such as tracing call graphs, "what calls X", "what does X call", "show dependencies of X", or inheritance chains.\n'
            '- "vector": Use this for semantic, natural language conceptual queries, such as "how does X work", "what does X do", explaining behavior, or understanding logic.\n'
            '- "hybrid": Use this for complex, ambiguous, or multi-faceted queries that require a combination of semantic and keyword search.\n\n'
            "Here are some examples:\n"
            'Query: "find the authenticate function" -> bm25\n'
            'Query: "what calls authenticate?" -> graph\n'
            'Query: "how does the auth middleware work?" -> vector\n'
            'Query: "where is the db connection initialized?" -> bm25\n'
            'Query: "what functions call verify_token?" -> graph\n'
            'Query: "explain the user registration flow" -> vector\n'
            'Query: "find details about verify_token and how it is used in handlers" -> hybrid\n\n'
            f'Query: "{sub_query}"\n\n'
            "Return ONLY the strategy name (bm25, graph, vector, or hybrid) in lowercase with no other text."
        )

        response = llm(prompt).strip().lower()
        cleaned = re.sub(r"[^a-z0-9]", "", response)
        for val in ["graph", "vector", "bm25", "hybrid"]:
            if val in cleaned:
                return val
        return "hybrid"
