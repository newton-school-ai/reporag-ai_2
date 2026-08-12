"""Strategy router.

Routes each sub-query to the optimal retrieval strategy: graph, vector,
bm25, or hybrid. Routing based on sub-query characteristics with
LLM-assisted classification and rule-based fallback.

Why
---
Different sub-queries benefit from different retrieval strategies:

* ``bm25``   -- the query names a specific symbol and asks where it is
  (e.g. "Find the authenticate function", "Where is DatabaseConfig
  defined?"). BM25's lexical overlap beats vector search here because
  embedding space blurs ``get_user`` / ``fetch_user`` together, while
  BM25 rewards exact token hits.
* ``graph``  -- the query asks about structural relationships
  ("What calls authenticate_user?", "What does auth middleware import?",
  "Trace the path from the login route to session creation"). Graph
  traversal follows typed edges (CALLS, IMPORTS, INHERITS) -- something
  neither BM25 nor vector can do.
* ``vector`` -- the query is conceptual / semantic ("How does the auth
  middleware work?", "Explain the ingestion pipeline", "Why is caching
  used here?"). Dense embeddings find semantically similar code chunks
  even when they use different vocabulary.
* ``hybrid`` -- the query is ambiguous, or scores tie across categories;
  running all strategies and fusing via RRF is always safe (if slower).

Design
------
Mirrors :class:`~reporag.agent.planner.QueryClassifier` exactly:

* **Dual strategy** -- LLM-based classifier is the primary path;
  deterministic, network-free rule-based routing is the fallback.
  The rule-based path is also the zero-config default (when no API key
  is set or ``use_llm=False``), so the router always works offline.
* **Lazy LLM loading** -- the langchain LLM client is constructed on the
  first :meth:`StrategyRouter.route` call, not at construction time.
  A pre-injected ``llm`` callable is respected, making tests
  network-free.
* **Pure module-level helpers** -- ``rule_based_route`` and
  ``parse_routing_response`` are free functions with no side effects,
  so they are unit-testable without any LLM or model.
* **Confidence-gated fallback** -- if the LLM's confidence is below
  ``confidence_threshold`` the result is overridden to ``hybrid`` (the
  safest default: running multiple strategies is always correct, if
  slower, while picking the wrong single strategy misses results).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from reporag.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

Strategy = Literal["bm25", "vector", "graph", "hybrid"]
"""The four retrieval strategies a sub-query can be routed to."""

_VALID_STRATEGIES: frozenset[str] = frozenset({"bm25", "vector", "graph", "hybrid"})


@dataclass(frozen=True)
class RoutingResult:
    """The outcome of routing a single sub-query.

    Attributes:
        step_id: The id of the :class:`~reporag.agent.planner.DecompositionStep`
            this result was produced for (``""`` when routing a bare query
            string without a step context).
        strategy: The retrieval strategy to use -- one of ``bm25``,
            ``vector``, ``graph``, or ``hybrid``.
        confidence: Router confidence in the decision, in ``[0.0, 1.0]``.
            When the confidence-threshold fallback fires this is the
            *original* confidence (so callers can see *why* the fallback
            triggered), and :attr:`fell_back` is set to ``True``.
        fell_back: ``True`` when the low-confidence fallback overrode the
            original strategy to ``hybrid``.
        source: ``"llm"`` when the LLM produced the routing decision,
            ``"rules"`` when the rule-based fallback was used.
        raw_response: The raw LLM response text (``""`` for the rule-based
            path). Kept for debugging -- not for programmatic use.
        metadata: Free-form extras (e.g. rule scores for the rule-based
            path, or the parsed JSON payload for the LLM path).
    """

    step_id: str = ""
    strategy: Strategy = "hybrid"
    confidence: float = 0.0
    fell_back: bool = False
    source: Literal["llm", "rules"] = "rules"
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Few-shot prompt
# ---------------------------------------------------------------------------

# Ten examples covering all four strategies and the boundary cases (an
# identifier query that also has a structural smell, a broad conceptual
# query that is clearly vector, etc.).
_FEW_SHOT_EXAMPLES: tuple[tuple[str, Strategy], ...] = (
    ("Find the authenticate function", "bm25"),
    ("Where is the DatabaseConfig class defined?", "bm25"),
    ("Locate the handle_request method", "bm25"),
    ("What file contains the TokenManager class?", "bm25"),
    ("What functions call authenticate_user?", "graph"),
    ("What does the auth middleware import?", "graph"),
    ("Trace the path from the login route to session creation", "graph"),
    ("What inherits from BaseHandler?", "graph"),
    ("How does the auth middleware work?", "vector"),
    ("Explain what the ingestion pipeline does", "vector"),
    ("What is the purpose of the caching layer?", "vector"),
    ("Why is connection pooling used in the database module?", "vector"),
)


def _build_routing_prompt(query: str) -> str:
    """Build the few-shot routing prompt for *query*.

    The prompt asks the LLM to return **strict JSON** with ``strategy``
    and ``confidence`` keys so the response is machine-parseable.
    """
    examples_block = "\n".join(
        f"Query: {example}\nStrategy: {label}" for example, label in _FEW_SHOT_EXAMPLES
    )
    return (
        "You are a retrieval strategy router for a code intelligence system.\n"
        "Choose the best retrieval strategy for the user's sub-query:\n"
        "\n"
        "- bm25: The query names a specific symbol (function, class, method)\n"
        "  and asks where it is defined or located. BM25 lexical search excels\n"
        "  at exact identifier matches.\n"
        "- graph: The query asks about structural relationships -- callers,\n"
        "  callees, imports, inheritance, or a path/trace between symbols.\n"
        "  Graph traversal follows typed edges (CALLS, IMPORTS, INHERITS).\n"
        "- vector: The query is conceptual or semantic -- it asks how something\n"
        "  works, why it is designed that way, or what its purpose is. Dense\n"
        "  embeddings find semantically similar code even with different vocab.\n"
        "- hybrid: The query is ambiguous or spans multiple strategies.\n"
        "  Run all strategies and fuse results.\n"
        "\n"
        "Examples:\n"
        f"{examples_block}\n"
        "\n"
        "Now route this query:\n"
        f"Query: {query}\n"
        "\n"
        "Respond with ONLY a JSON object on a single line in this exact "
        "format (no markdown, no explanation):\n"
        '{"strategy": "<one of bm25|vector|graph|hybrid>", '
        '"confidence": <float between 0.0 and 1.0>}'
    )


# ---------------------------------------------------------------------------
# Rule-based router (pure, network-free, unit-testable)
# ---------------------------------------------------------------------------

# Compiled regex patterns for identifier-lookup signals.
# These fire when the query names a *specific* symbol and asks for its
# location / definition.  Patterns are intentionally broad (verb-anchored)
# rather than compound so natural-language variations like
# "Find the authenticate function" match correctly.
_IDENTIFIER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bfind\b", re.IGNORECASE),
    re.compile(r"\bwhere\s+(is|are)\b", re.IGNORECASE),
    re.compile(r"\blocate\b", re.IGNORECASE),
    re.compile(r"\bdefined\b", re.IGNORECASE),
    re.compile(r"\bdefinition\s+of\b", re.IGNORECASE),
    re.compile(r"\bshow\s+me\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(file|line)\b", re.IGNORECASE),
    re.compile(r"\bdeclare[ds]?\b", re.IGNORECASE),
    re.compile(r"\bcontains?\b", re.IGNORECASE),
)

# Structural / graph signals: the query asks about relationships between
# code entities -- who calls what, what is inherited, import chains, paths.
_STRUCTURAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "what calls X" and "what callers does X have"
    re.compile(r"\bwhat\s+(calls|callers?)\b", re.IGNORECASE),
    re.compile(r"\bcallers?\b", re.IGNORECASE),
    # "what functions call X" -- functions is the subject, call is the verb
    re.compile(r"\bfunctions?\s+call\b", re.IGNORECASE),
    re.compile(r"\binherits?\b", re.IGNORECASE),
    re.compile(r"\bsubclass(es)?\b", re.IGNORECASE),
    re.compile(r"\bimports?\b", re.IGNORECASE),
    re.compile(r"\bdepends?\s+on\b", re.IGNORECASE),
    re.compile(r"\bpath\s+from\b", re.IGNORECASE),
    re.compile(r"\btrace\b", re.IGNORECASE),
    re.compile(r"\bchain\b", re.IGNORECASE),
    re.compile(r"\bsubgraph\b", re.IGNORECASE),
    re.compile(r"\bflow\s+from\b", re.IGNORECASE),
)

# Semantic / vector signals: the query is conceptual, asking for an
# explanation, the purpose, or the high-level behaviour of something.
# Note: ``\bwhat\s+does\b`` is intentionally excluded -- "What does X import?"
# fires the graph ``imports?`` signal and should stay in that category.
_SEMANTIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhow\s+does\b", re.IGNORECASE),
    re.compile(r"\bexplain\b", re.IGNORECASE),
    re.compile(r"\bwhy\b", re.IGNORECASE),
    re.compile(r"\bpurpose\b", re.IGNORECASE),
    re.compile(r"\boverview\b", re.IGNORECASE),
    re.compile(r"\barchitecture\b", re.IGNORECASE),
    re.compile(r"\bhigh-?level\b", re.IGNORECASE),
    re.compile(r"\bsummar(y|ize)\b", re.IGNORECASE),
    re.compile(r"\bdesign\b", re.IGNORECASE),
)


def _score_patterns(query: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    """Return the number of *patterns* that match *query*.

    Each pattern contributes at most one vote so a query that repeats the
    same signal word does not drown out other signals.
    """
    return sum(1 for p in patterns if p.search(query))


def rule_based_route(query: str) -> RoutingResult:
    """Route *query* to a retrieval strategy using weighted regex patterns.

    This is the deterministic, network-free fallback.  It scores each
    strategy by counting matching signal patterns; the strategy with the
    highest score wins.  Confidence is ``winner_score / total_score``.
    When all categories score zero (no signal at all) the query is routed
    to ``hybrid`` with confidence ``0.0`` so the confidence-threshold
    fallback in :meth:`StrategyRouter.route` fires naturally.

    Ties are resolved to ``hybrid`` -- running multiple strategies is
    always safe and never wrong, just slower.

    Args:
        query: The natural-language sub-query to route.

    Returns:
        A :class:`RoutingResult` with ``source="rules"`` and the
        per-strategy vote counts in ``metadata["scores"]``.
    """
    scores: dict[str, int] = {
        "bm25": _score_patterns(query, _IDENTIFIER_PATTERNS),
        "graph": _score_patterns(query, _STRUCTURAL_PATTERNS),
        "vector": _score_patterns(query, _SEMANTIC_PATTERNS),
    }
    total = sum(scores.values())

    if total == 0:
        return RoutingResult(
            strategy="hybrid",
            confidence=0.0,
            source="rules",
            metadata={"scores": scores},
        )

    top_score = max(scores.values())
    winners = [k for k, v in scores.items() if v == top_score]

    if len(winners) > 1:
        # Tie -> hybrid (safer to run multiple strategies than guess wrong).
        return RoutingResult(
            strategy="hybrid",
            confidence=top_score / total,
            source="rules",
            metadata={"scores": scores},
        )

    winner = winners[0]
    return RoutingResult(
        strategy=winner,  # type: ignore[arg-type]
        confidence=top_score / total,
        source="rules",
        metadata={"scores": scores},
    )


# ---------------------------------------------------------------------------
# LLM response parsing (pure, unit-testable)
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_routing_response(raw: str) -> RoutingResult:
    """Parse the raw LLM response text into a :class:`RoutingResult`.

    The LLM is prompted to return strict JSON::

        {"strategy": "graph", "confidence": 0.92}

    This parser is tolerant: it extracts the first ``{...}`` block (so
    accidental markdown fences or leading prose do not break parsing),
    coerces the ``strategy`` to a valid category (rejecting unknown values),
    and clamps ``confidence`` to ``[0.0, 1.0]``.

    Args:
        raw: The raw text returned by the LLM.

    Returns:
        A :class:`RoutingResult` with ``source="llm"``.  If the response
        cannot be parsed, a zero-confidence ``hybrid`` result is returned
        (so the caller's threshold fallback fires) with the parse error
        recorded in ``metadata["parse_error"]``.
    """
    if not raw or not raw.strip():
        return RoutingResult(
            strategy="hybrid",
            confidence=0.0,
            source="llm",
            raw_response=raw,
            metadata={"parse_error": "empty response"},
        )

    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        return RoutingResult(
            strategy="hybrid",
            confidence=0.0,
            source="llm",
            raw_response=raw,
            metadata={"parse_error": "no JSON object found"},
        )

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return RoutingResult(
            strategy="hybrid",
            confidence=0.0,
            source="llm",
            raw_response=raw,
            metadata={"parse_error": f"invalid JSON: {exc}"},
        )

    raw_strategy = str(payload.get("strategy", "")).strip().lower()
    if raw_strategy not in _VALID_STRATEGIES:
        return RoutingResult(
            strategy="hybrid",
            confidence=0.0,
            source="llm",
            raw_response=raw,
            metadata={
                "parse_error": f"invalid strategy: {raw_strategy!r}",
                "raw_payload": payload,
            },
        )

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return RoutingResult(
        strategy=raw_strategy,  # type: ignore[arg-type]
        confidence=confidence,
        source="llm",
        raw_response=raw,
        metadata={"raw_payload": payload},
    )


# ---------------------------------------------------------------------------
# StrategyRouter
# ---------------------------------------------------------------------------

# A callable that takes a prompt string and returns the LLM's response text.
# This is the test seam: tests inject a deterministic fake instead of a real
# langchain LLM, keeping every test network-free.
LLMCallable = Callable[[str], str]


def _is_unset_secret(secret: Any) -> bool:
    """Return True if *secret* is an unset or placeholder SecretStr.

    Mirrors :func:`reporag.agent.planner._is_unset_secret` but is kept
    local to avoid coupling to planner internals.
    """
    from pydantic import SecretStr

    if isinstance(secret, SecretStr):
        value = secret.get_secret_value().strip()
    else:
        value = str(secret).strip()
    return value in {
        "",
        "change-me",
        "change-me-to-a-random-string",
        "sk-your-key-here",
        "sk-ant-your-key-here",
    }


def _build_langchain_llm() -> LLMCallable:
    """Construct a provider-agnostic ``(prompt: str) -> str`` langchain client.

    Reuses the same provider-selection logic as the planner module so the
    router always stays in lockstep with the classifier/decomposer.
    """
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        client = ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key.get_secret_value(),
            temperature=0.0,
        )
    else:
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key.get_secret_value(),
            temperature=0.0,
        )

    def _invoke(prompt: str) -> str:
        response = client.invoke(prompt)
        return str(getattr(response, "content", response))

    return _invoke


class StrategyRouter:
    """Routes a sub-query to the optimal retrieval strategy.

    The primary path is an LLM-based router with few-shot examples; a
    deterministic rule-based router is the fallback when the LLM is
    disabled, no API key is configured, or the LLM response cannot be
    parsed.  If the LLM's confidence is below *confidence_threshold* the
    result is overridden to ``hybrid`` (the safest default -- running all
    strategies is always correct, just slower).

    Args:
        llm: A pre-built callable ``(prompt: str) -> str`` that stands in
            for the langchain LLM client.  Passing a callable is the
            supported test seam -- it makes the router network-free and
            avoids the API key requirement, exactly like the planner's
            ``_FakeLLM`` seam.  When ``None`` (default) a real langchain
            LLM is constructed lazily on the first :meth:`route` call
            using ``settings.llm_provider`` and the configured API key.
        confidence_threshold: Below this confidence the routing falls
            back to ``hybrid``.  Defaults to
            ``settings.query_classifier_confidence_threshold`` (the same
            threshold reused for routing, keeping configuration simple).
            Must be in ``[0.0, 1.0]``.
        use_llm: When ``True`` (default) the LLM path is used; when
            ``False`` only the rule-based router runs.  Defaults to
            ``settings.query_classifier_use_llm``.

    Raises:
        ValueError: If *confidence_threshold* is outside ``[0.0, 1.0]``.
    """

    def __init__(
        self,
        llm: LLMCallable | None = None,
        *,
        confidence_threshold: float | None = None,
        use_llm: bool | None = None,
    ) -> None:
        if confidence_threshold is None:
            confidence_threshold = settings.query_classifier_confidence_threshold
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], "
                f"got {confidence_threshold!r}."
            )

        self._resolved_llm: LLMCallable | None = llm
        self._loaded = False
        self.confidence_threshold = confidence_threshold
        self.use_llm = (
            use_llm if use_llm is not None else settings.query_classifier_use_llm
        )

    # ------------------------------------------------------------------
    # Lazy LLM loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> LLMCallable | None:
        """Resolve the LLM callable, constructing a real client if needed.

        Returns ``None`` (and logs a warning) when the LLM is enabled but
        no API key is configured -- the caller then falls back to the
        rule-based router.
        """
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

            self._resolved_llm = _build_langchain_llm()

        self._loaded = True
        return self._resolved_llm

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True once the LLM has been resolved (or its absence detected)."""
        return self._loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, query: str, *, step_id: str = "") -> RoutingResult:
        """Route *query* to the optimal retrieval strategy.

        The routing path is:

        1. If ``use_llm`` is ``False``, jump straight to the rule-based
           router (step 4).
        2. Resolve the LLM (lazy load).  If no API key is configured, fall
           back to the rule-based router.
        3. Call the LLM with the few-shot prompt and parse the response.
           If the response cannot be parsed, fall back to the rule-based
           router.
        4. Apply the confidence threshold: if ``confidence <
           confidence_threshold``, override ``strategy`` to ``hybrid``
           and set ``fell_back=True`` (the original confidence is preserved
           so callers can see why the fallback fired).

        Args:
            query: The natural-language sub-query to route.
            step_id: Optional step id to attach to the result (from
                :class:`~reporag.agent.planner.DecompositionStep`).

        Returns:
            A :class:`RoutingResult` with ``strategy``, ``confidence``,
            ``source``, and ``fell_back`` populated.

        Raises:
            ValueError: If *query* is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        result = self._route_without_threshold(query)

        # Attach the caller's step_id (replace the frozen dataclass by
        # reconstructing -- frozen fields cannot be mutated in place).
        result = RoutingResult(
            step_id=step_id,
            strategy=result.strategy,
            confidence=result.confidence,
            fell_back=result.fell_back,
            source=result.source,
            raw_response=result.raw_response,
            metadata=result.metadata,
        )

        # Confidence-gated fallback: low confidence -> hybrid (safest).
        if result.confidence < self.confidence_threshold:
            logger.info(
                "StrategyRouter: confidence %.3f < threshold %.3f for "
                "query %r; falling back to hybrid.",
                result.confidence,
                self.confidence_threshold,
                query,
            )
            return RoutingResult(
                step_id=step_id,
                strategy="hybrid",
                confidence=result.confidence,
                fell_back=True,
                source=result.source,
                raw_response=result.raw_response,
                metadata=result.metadata,
            )

        return result

    def _route_without_threshold(self, query: str) -> RoutingResult:
        """Run the LLM or rule-based router, before the threshold check."""
        if not self.use_llm:
            return rule_based_route(query)

        llm = self._ensure_loaded()
        if llm is None:
            return rule_based_route(query)

        prompt = _build_routing_prompt(query)
        try:
            raw_response = llm(prompt)
        except Exception as exc:
            logger.warning(
                "StrategyRouter: LLM call failed (%s); falling back to "
                "rule-based routing.",
                exc,
            )
            return rule_based_route(query)

        result = parse_routing_response(raw_response)
        if result.metadata.get("parse_error"):
            logger.warning(
                "StrategyRouter: could not parse LLM response (%s); "
                "falling back to rule-based routing.",
                result.metadata["parse_error"],
            )
            return rule_based_route(query)

        return result

    def __repr__(self) -> str:
        return (
            f"StrategyRouter(use_llm={self.use_llm}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"loaded={self._loaded})"
        )
