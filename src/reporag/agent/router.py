"""Strategy router (Issue 22).

Routes each sub-query (from the Issue 21 decomposer) to the retrieval
strategy that will answer it best:

* ``graph``    -- structural queries ("what calls X", "trace the path
  from A to B"). Graph traversal follows typed ``CALLS`` / ``IMPORTS`` /
  ``INHERITS`` edges that vector search and BM25 cannot see.
* ``vector``   -- semantic queries ("how does the auth middleware work?").
  Dense-vector similarity blurs related-but-different symbols together,
  which is exactly what you want for open-ended "how/why" questions.
* ``bm25``     -- identifier lookups ("where is ``authenticate_user``
  defined?"). Pure lexical overlap returns the chunk that actually
  contains those tokens, regardless of semantic drift.
* ``hybrid``   -- runs several component strategies and fuses their
  rankings with Reciprocal Rank Fusion (Issue 19) so no single path can
  starve the answer. The safest default for ambiguous queries.

Design
------
The router follows exactly the conventions already established by
:class:`~reporag.agent.planner.QueryClassifier` and
:class:`~reporag.agent.planner.QueryDecomposer` so the three LLM-backed
components of the agentic planner stay consistent and easy to test:

* **Dual strategy** -- an LLM-based router is the primary path; a
  deterministic, network-free rule-based router is the fallback.  The
  rule-based path is also the zero-config default (no API key set or
  ``STRATEGY_ROUTER_USE_LLM=false``), so the router always works
  offline.
* **Lazy LLM loading** -- the langchain LLM client is constructed on the
  first :meth:`StrategyRouter.route_batch` call, not at construction time
  (cheap, test-friendly).  A pre-injected ``llm`` callable is respected,
  making tests network-free exactly like the planner's ``_FakeLLM`` seam.
* **Pure module-level helpers** -- the rule-based scorer and the LLM
  response parser are free functions with no side effects, so they are
  unit-testable without any LLM or model.
* **Confidence-gated fallback** -- if the LLM's confidence is below
  ``settings.strategy_router_confidence_threshold`` the result is
  overridden to ``hybrid`` (the safest default: it covers every
  component strategy, so a low-confidence single-strategy decision never
  misses a retrieval path).
* **Batch routing** -- :meth:`StrategyRouter.route_batch` routes every
  step of a :class:`~reporag.agent.planner.DecompositionPlan` in a single
  call.  The LLM path sends every sub-query in one prompt to amortise the
  network round-trip; the rule path is just a tight loop over
  :func:`rule_based_route`.

The ``RetrievalStrategy`` enum and the :class:`RoutingResult` /
:class:`RoutingPlan` dataclasses mirror
:class:`~reporag.agent.planner.ClassificationResult` /
:class:`~reporag.agent.planner.DecompositionPlan` so the three planner
components present a uniform, frozen-dataclass, provenance-bearing API.
"""

from __future__ import annotations

import enum
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from reporag.agent.planner import (
    DecompositionPlan,
    DecompositionStep,
    LLMCallable,
    _build_langchain_llm,
    _extract_json_object,
)
from reporag.config import _is_unset, settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class RetrievalStrategy(enum.StrEnum):
    """The four retrieval strategies a sub-query can be routed to.

    ``str`` + ``enum.Enum`` so the value serialises cleanly to JSON and
    compares equal to its plain-string spelling (``RetrievalStrategy.GRAPH
    == "graph"``), which keeps downstream code readable.  The four values
    are exactly the four named strategies in the Issue 22 acceptance
    criteria: ``graph`` / ``vector`` / ``bm25`` / ``hybrid``.
    """

    GRAPH = "graph"
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


# A plain-string alias accepted everywhere a strategy is consumed: the
# router returns ``RetrievalStrategy`` but the executor and tests compare
# against bare strings, and ``RetrievalStrategy`` subclasses ``str`` so
# those comparisons work without an explicit cast.
StrategyName = Literal["graph", "vector", "bm25", "hybrid"]

_VALID_STRATEGIES: frozenset[str] = frozenset(
    member.value for member in RetrievalStrategy
)


@dataclass(frozen=True)
class RoutingResult:
    """The outcome of routing a single sub-query.

    Attributes:
        step_id: The id of the :class:`~reporag.agent.planner.DecompositionStep`
            this routing decision applies to.  Kept so a whole-plan
            :class:`RoutingPlan` can be reconstructed from a flat list of
            routing results without assuming step ordering.
        strategy: The chosen :class:`RetrievalStrategy`.
        confidence: Model / rule confidence in the routing decision, in
            ``[0.0, 1.0]``.  When the confidence-threshold fallback fires
            this is the *original* confidence (so callers can see *why* the
            fallback triggered), and :attr:`fell_back` is set to ``True``.
        fell_back: ``True`` when the low-confidence fallback overrode the
            original strategy to ``hybrid``, or when the rule-based path
            was used because the LLM was disabled / unavailable / failed.
        source: ``"llm"`` when the LLM produced the routing,
            ``"rules"`` when the rule-based fallback was used.
        raw_response: The raw LLM response text (``""`` for the rule-based
            path).  Kept for debugging -- not for programmatic use.
        metadata: Free-form extras (rule scores for the rule-based path,
            or the parsed JSON payload for the LLM path).
    """

    step_id: str
    strategy: RetrievalStrategy
    confidence: float
    fell_back: bool = False
    source: Literal["llm", "rules"] = "rules"
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingPlan:
    """Per-sub-query routing decisions for an entire decomposition plan.

    Attributes:
        original_query: The query that was routed.
        steps: The ordered sub-queries the routing decisions apply to
            (copied from the input :class:`DecompositionPlan` so the
            routing plan is self-contained).  Always at least one step.
        routings: A list of :class:`RoutingResult`, one per step, in the
            same order as :attr:`steps`.
        source: ``"llm"`` when the LLM produced every routing,
            ``"rules"`` when the rule-based fallback was used (for every
            step, or for the steps the LLM could not route).
        raw_response: The raw LLM response text for the batch (``""`` for
            the rule-based path).
        fell_back: ``True`` when at least one routing fell back (either
            to the rule-based path or via the confidence threshold).
        metadata: Free-form extras (e.g. per-route source breakdown).
    """

    original_query: str
    steps: tuple[DecompositionStep, ...]
    routings: tuple[RoutingResult, ...]
    source: Literal["llm", "rules"]
    raw_response: str = ""
    fell_back: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def strategy_for(self, step_id: str) -> RetrievalStrategy:
        """Return the routed strategy for a step id.

        Args:
            step_id: The id of a step in :attr:`steps`.

        Returns:
            The :class:`RetrievalStrategy` that step was routed to.

        Raises:
            KeyError: If *step_id* is not present in this plan.
        """
        for routing in self.routings:
            if routing.step_id == step_id:
                return routing.strategy
        raise KeyError(f"no routing for step id {step_id!r}")


# ---------------------------------------------------------------------------
# Few-shot prompt
# ---------------------------------------------------------------------------

# Six carefully-chosen examples covering all four strategies and the
# boundary cases.  These pin the LLM's notion of "structural -> graph",
# "semantic -> vector", "identifier -> bm25", "ambiguous -> hybrid" so
# the prompt and the unit tests stay in lockstep.
_LLM_FEW_SHOT_EXAMPLES: tuple[tuple[str, str, str], ...] = (
    ("Where is the authenticate_user function defined?", "step-1", "bm25"),
    ("Locate the login route in the routes module.", "step-1", "bm25"),
    ("How does the auth middleware validate credentials?", "step-1", "vector"),
    ("Explain how the ingestion pipeline is organized.", "step-1", "vector"),
    ("What functions call the authenticate_user function?", "step-1", "graph"),
    (
        "Describe the relationship between the auth and session modules.",
        "step-1",
        "hybrid",
    ),
)


def _build_routing_prompt(steps: list[DecompositionStep], original_query: str) -> str:
    """Build the few-shot batch routing prompt for *steps*.

    The prompt sends every sub-query in *one* LLM call (amortising the
    network round-trip) and asks for strict JSON mapping each step id to a
    strategy plus a confidence.  Few-shot examples anchor the four
    categories and the expected confidence range.
    """
    examples_block = "\n".join(
        f"Sub-query: {example_query}\nStep id: {example_id}\nStrategy: {example_strategy}"
        for example_query, example_id, example_strategy in _LLM_FEW_SHOT_EXAMPLES
    )
    sub_queries_block = "\n".join(
        f"- step_id: {step.id}, query: {step.query}" for step in steps
    )
    return (
        "You are a retrieval-strategy router for a code intelligence "
        "system. Route each sub-query to exactly one retrieval "
        "strategy:\n"
        "\n"
        "- graph: structural queries about call/dependency/inheritance "
        "relationships ('what calls X', 'trace the path from A to B', "
        "'who inherits from Y').\n"
        "- vector: semantic queries about behaviour or meaning ('how "
        "does X work', 'what does X do', 'explain the design of Y').\n"
        "- bm25: exact identifier/location lookups ('where is X "
        "defined', 'locate the function named Z', 'show me the class "
        "Foo').\n"
        "- hybrid: ambiguous queries that could benefit from multiple "
        "strategies; broad or cross-cutting questions where no single "
        "strategy clearly dominates.\n"
        "\n"
        "Examples:\n"
        f"{examples_block}\n"
        "\n"
        f"Original query (for context): {original_query}\n"
        "Sub-queries to route:\n"
        f"{sub_queries_block}\n"
        "\n"
        "Respond with ONLY a JSON object on a single line in this exact "
        "format (no markdown, no explanation):\n"
        '{"routes": [{"step_id": "<id>", "strategy": '
        '"graph|vector|bm25|hybrid", "confidence": <float between 0.0 '
        "and 1.0>}, ...]}"
    )


# ---------------------------------------------------------------------------
# Rule-based router (pure, network-free, unit-testable)
# ---------------------------------------------------------------------------

# Weighted regex signal patterns.  Each pattern votes for a strategy; the
# strategy with the highest total weight wins.  Patterns are ordered from
# most specific to least specific so the first match in a group dominates.
#
# ``graph`` signals: the query asks about relationships, callers, callees,
# inheritance, imports, or traces a path between nodes.
_GRAPH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhat\s+(calls|calls|is\s+called\s+by)\b", re.IGNORECASE),
    # "What functions call X" / "what modules call Y" -- a subject word sits
    # between "what" and "call(s)"; tolerate up to a few words so the plural
    # subjects ("functions", "modules", "classes") still vote graph.
    re.compile(r"\bwhat\s+\w+(?:\s+\w+)?\s+call(?:s|ers?)?\b", re.IGNORECASE),
    re.compile(r"\bwho\s+calls\b", re.IGNORECASE),
    re.compile(r"\binherits\s+from\b", re.IGNORECASE),
    re.compile(r"\bimport[es]?\s+(from|by)\b", re.IGNORECASE),
    re.compile(r"\bcalled\s+by\b", re.IGNORECASE),
    re.compile(r"\bcallers?\b", re.IGNORECASE),
    re.compile(r"\bcallees?\b", re.IGNORECASE),
    re.compile(r"\btrace\s+(the\s+)?(path|chain|flow)\b", re.IGNORECASE),
    re.compile(r"\bdependencies?\b", re.IGNORECASE),
    re.compile(r"\bdepend(s|ed)?\s+on\b", re.IGNORECASE),
    re.compile(r"\brelationship\s+between\b", re.IGNORECASE),
)

# ``bm25`` signals: the query names a specific identifier and asks for its
# location / definition / line.  The ``(find|locate|show me) ... <noun>``
# pattern tolerates an identifier between the verb and the category noun
# (e.g. "Locate the DatabaseConfig class", "Show me the handle_request
# function") so the pattern still fires when the named symbol sits in the
# middle -- the common phrasing for identifier lookups.  Kept intentionally
# tight (a location verb must be present) so a pure "how does X work" with a
# stray "function" noun never gets misrouted to BM25.
_BM25_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhere\s+(is|are)\b", re.IGNORECASE),
    re.compile(
        r"\b(find|locate|show\s+me)\b"
        r".{0,40}\b(file|class|function|method|def|symbol|module|variable)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdefinition\s+of\b", re.IGNORECASE),
    re.compile(r"\bdefined\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(line|file)\b", re.IGNORECASE),
    re.compile(r"\bdeclare[ds]?\b", re.IGNORECASE),
    re.compile(r"\bfile\s+that\s+contains\b", re.IGNORECASE),
)

# ``vector`` signals: the query asks about behaviour, meaning, or design
# -- the kind of question dense embeddings answer best.
_VECTOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhow\s+does\b.{0,40}\bwork\b", re.IGNORECASE),
    re.compile(r"\bhow\s+do(es)?\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+does\b.{0,40}\bdo\b", re.IGNORECASE),
    re.compile(r"\bexplain\b", re.IGNORECASE),
    re.compile(r"\bdescribe\b", re.IGNORECASE),
    re.compile(r"\bpurpose\s+of\b", re.IGNORECASE),
    re.compile(r"\bbehavi[ou]r\b", re.IGNORECASE),
)

# Unless a query matches at least this many vector/graph/bm25 signals we
# treat it as ambiguous and route to ``hybrid``.  This keeps the rule-based
# path from confidently picking a single strategy on a single weak signal,
# which is the "safest default" philosophy shared with the classifier's
# tiebreak toward ``multi-hop``.
_HYBRID_FALLBACK_MIN_VOTES = 1


def _score_patterns(query: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    """Return the number of *patterns* that match *query*.

    Each pattern contributes at most one vote (not one per match) so a
    query that repeats the same signal word does not drown out other
    signals.
    """
    return sum(1 for pattern in patterns if pattern.search(query))


def rule_based_route(step: DecompositionStep) -> RoutingResult:
    """Route *step* using weighted regex signal patterns.

    This is the deterministic, network-free fallback.  It scores each
    strategy by counting how many of its signal patterns match the
    sub-query text; the strategy with the highest score wins *only if it
    has strictly more votes than every other* and at least
    :data:`_HYBRID_FALLBACK_MIN_VOTES` -- otherwise the query is treated
    as ambiguous and routed to ``hybrid``.  Confidence is the winner's
    share of the total votes (``winner / total``), clamped to ``[0, 1]``.
    When all strategies tie at zero the query is routed to ``hybrid``
    (the safest default) with confidence ``0.0`` so the caller's threshold
    fallback fires naturally.

    Args:
        step: The :class:`~reporag.agent.planner.DecompositionStep` to
            route.  Only ``step.id`` and ``step.query`` are read.

    Returns:
        A :class:`RoutingResult` with ``source="rules"`` and the
        per-strategy vote counts in ``metadata["scores"]``.
    """
    scores = {
        "graph": _score_patterns(step.query, _GRAPH_PATTERNS),
        "vector": _score_patterns(step.query, _VECTOR_PATTERNS),
        "bm25": _score_patterns(step.query, _BM25_PATTERNS),
    }
    total = sum(scores.values())

    if total == 0:
        # No signal at all -- safest default is hybrid with zero
        # confidence so the caller's threshold fallback can promote it
        # (the fallback is identical to the primary decision here, but
        # fell_back stays False since hybrid *is* the right answer).
        return RoutingResult(
            step_id=step.id,
            strategy=RetrievalStrategy.HYBRID,
            confidence=0.0,
            source="rules",
            metadata={"scores": scores},
        )

    winner = max(scores, key=lambda k: scores[k])
    top_score = scores[winner]

    # Strict-winner rule: if the top score is tied with another strategy,
    # the query is ambiguous -> hybrid.  Ties at score 1 across strategies
    # are the common case (e.g. "what calls X" matches both graph and
    # vector signals), and hybrid is the safe default there.
    tied = [k for k, v in scores.items() if v == top_score]
    if len(tied) > 1 or top_score < _HYBRID_FALLBACK_MIN_VOTES:
        winner = "hybrid"

    confidence = (scores[winner] if winner != "hybrid" else top_score) / max(total, 1)
    strategy = RetrievalStrategy(winner)
    return RoutingResult(
        step_id=step.id,
        strategy=strategy,
        confidence=confidence,
        source="rules",
        metadata={"scores": scores},
    )


# ---------------------------------------------------------------------------
# LLM response parsing (pure, unit-testable)
# ---------------------------------------------------------------------------


# The batch routing response contains a nested ``routes`` list, so
# (like the decomposer's parser) we walk brace depth to find the matching
# closing brace rather than a flat ``{...}`` regex.  The helper is shared
# with the decomposer to avoid drift -- see
# :func:`reporag.agent.planner._extract_json_object`.


def _coerce_strategy(raw: Any) -> RetrievalStrategy | None:
    """Map a raw LLM-provided label onto a :class:`RetrievalStrategy`.

    Tolerant of capitalisation / spacing and of the common synonym
    ``"keyword"`` for ``bm25``.  Returns ``None`` for anything
    unrecognised so the caller can flag a parse error.
    """
    key = str(raw).strip().lower()
    synonyms = {
        "graph": RetrievalStrategy.GRAPH,
        "call_graph": RetrievalStrategy.GRAPH,
        "vector": RetrievalStrategy.VECTOR,
        "semantic": RetrievalStrategy.VECTOR,
        "embedding": RetrievalStrategy.VECTOR,
        "bm25": RetrievalStrategy.BM25,
        "keyword": RetrievalStrategy.BM25,
        "lexical": RetrievalStrategy.BM25,
        "hybrid": RetrievalStrategy.HYBRID,
        "fusion": RetrievalStrategy.HYBRID,
        "mixed": RetrievalStrategy.HYBRID,
    }
    return synonyms.get(key)


def parse_routing_response(
    raw: str, expected_step_ids: list[str]
) -> tuple[dict[str, RoutingResult], str | None]:
    """Parse the raw LLM batch routing response into per-step routings.

    The LLM is prompted to return strict JSON::

        {"routes": [{"step_id": "step-1", "strategy": "graph",
                     "confidence": 0.9}, ...]}

    This parser is tolerant of markdown fences / leading prose (via
    :func:`_extract_json_object`) and of fuzzy strategy labels (via
    :func:`_coerce_strategy`), but is strict about the contract that the
    response covers every expected step id.  A route whose ``confidence``
    is missing/invalid defaults to ``0.0`` (so the caller's threshold
    fallback fires).  Step ids in the response that are *not* among the
    expected ids are ignored (the LLM may hallucinate extra steps); an
    expected step that is *missing* from the response is flagged as an
    error so the caller falls back to rules for the whole batch.

    Args:
        raw: The raw text returned by the LLM.
        expected_step_ids: The ids that *must* appear in the response for
            it to be considered a complete plan.

    Returns:
        A ``(routings_by_id, error)`` tuple.  On success ``error`` is
        ``None`` and ``routings_by_id`` maps every expected step id to a
        :class:`RoutingResult`.  On failure ``routings_by_id`` is ``{}``
        and ``error`` describes what went wrong, for the retry/fallback
        decision.
    """
    if not raw or not raw.strip():
        return {}, "empty response"

    json_text = _extract_json_object(raw)
    if json_text is None:
        return {}, "no JSON object found"

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON: {exc}"

    raw_routes = payload.get("routes") if isinstance(payload, dict) else None
    if not isinstance(raw_routes, list) or not raw_routes:
        return {}, "missing or empty 'routes' list"

    routings: dict[str, RoutingResult] = {}
    for index, raw_route in enumerate(raw_routes):
        if not isinstance(raw_route, dict):
            return {}, f"route {index} is not an object"

        step_id = str(raw_route.get("step_id", "")).strip()
        if not step_id:
            return {}, f"route {index} is missing 'step_id'"
        # Ignore routes for step ids we did not ask about -- they cannot
        # affect the plan and would only pollute the output mapping.
        if step_id not in expected_step_ids:
            continue

        strategy = _coerce_strategy(raw_route.get("strategy"))
        if strategy is None:
            return {}, (
                f"route {index} ({step_id!r}) has an invalid strategy: "
                f"{raw_route.get('strategy')!r}"
            )

        try:
            confidence = float(raw_route.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        if step_id in routings:
            # A duplicate step_id in the response is a malformed plan.
            return {}, f"duplicate step_id in routes: {step_id!r}"

        routings[step_id] = RoutingResult(
            step_id=step_id,
            strategy=strategy,
            confidence=confidence,
            source="llm",
            raw_response=raw,
            metadata={"raw_payload": raw_route},
        )

    missing = sorted(set(expected_step_ids) - set(routings.keys()))
    if missing:
        return {}, f"missing routes for step ids: {', '.join(missing)}"

    return routings, None


# ---------------------------------------------------------------------------
# Hybrid composition
# ---------------------------------------------------------------------------


def parse_hybrid_components(spec: str | None) -> tuple[RetrievalStrategy, ...]:
    """Parse the comma-separated ``hybrid_components`` setting.

    Returns the enabled component strategies (always excluding ``hyybrid``
    itself) in the order given, deduplicated and lower-cased.  Unknown /
    blank entries are dropped (logged at debug) so a typo in the setting
    cannot break hybrid routing -- an empty spec falls back to
    ``(vector, bm25)`` which is the conventional hybrid mix.

    Args:
        spec: The raw ``STRATEGY_ROUTER_HYBRID_COMPONENTS`` value.  May be
            ``None`` (use the settings default).

    Returns:
        A non-empty tuple of component :class:`RetrievalStrategy` values
        (never including :attr:`RetrievalStrategy.HYBRID`).
    """
    source = spec if spec is not None else settings.strategy_router_hybrid_components
    seen: set[str] = set()
    components: list[RetrievalStrategy] = []
    for token in str(source).split(","):
        token = token.strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        strategy = _coerce_strategy(token)
        if strategy is None or strategy is RetrievalStrategy.HYBRID:
            logger.debug(
                "Dropping invalid / hybrid token %r from hybrid_components %r.",
                token,
                source,
            )
            continue
        components.append(strategy)

    if not components:
        # Degraded to the conventional hybrid mix -- never return an empty
        # tuple, as that would make hybrid routing a silent no-op.
        components = [RetrievalStrategy.VECTOR, RetrievalStrategy.BM25]
    return tuple(components)


# ---------------------------------------------------------------------------
# StrategyRouter
# ---------------------------------------------------------------------------


class StrategyRouter:
    """Routes each sub-query to the optimal retrieval strategy.

    The primary path is an LLM-based router with few-shot examples that
    routes an entire batch of sub-queries in one prompt; a deterministic
    rule-based router is the fallback when the LLM is disabled, no API
    key is configured, or the LLM response cannot be parsed.  If a
    sub-query's LLM confidence is below
    ``settings.strategy_router_confidence_threshold`` that single
    sub-query is overridden to ``hybrid`` (the safest default).

    Args:
        llm: A pre-built callable ``(prompt: str) -> str`` that stands in
            for the langchain LLM client.  Passing a callable is the
            supported test seam -- it makes the router network-free and
            avoids the API key requirement, exactly like the planner's
            ``_FakeLLM``.  When ``None`` (default) a real langchain LLM is
            constructed lazily on the first :meth:`route_batch` call.
        confidence_threshold: Below this confidence a routing falls back
            to ``hybrid``.  Defaults to
            ``settings.strategy_router_confidence_threshold``.  Must be in
            ``[0.0, 1.0]``.
        use_llm: When ``True`` (default) the LLM path is used; when
            ``False`` only the rule-based router runs (no LLM is ever
            loaded).  Defaults to ``settings.strategy_router_use_llm``.
        hybrid_components: The component strategies a "hybrid" route
            expands to.  Defaults to
            :func:`parse_hybrid_components(settings.strategy_router_hybrid_components)`.

    Raises:
        ValueError: If *confidence_threshold* is outside ``[0.0, 1.0]``.
    """

    def __init__(
        self,
        llm: LLMCallable | None = None,
        *,
        confidence_threshold: float | None = None,
        use_llm: bool | None = None,
        hybrid_components: tuple[RetrievalStrategy, ...] | None = None,
    ) -> None:
        if confidence_threshold is None:
            confidence_threshold = settings.strategy_router_confidence_threshold
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], "
                f"got {confidence_threshold!r}."
            )

        self._resolved_llm: LLMCallable | None = llm
        self._loaded = False
        self.confidence_threshold = confidence_threshold
        self.use_llm = (
            use_llm if use_llm is not None else settings.strategy_router_use_llm
        )
        self.hybrid_components = (
            hybrid_components
            if hybrid_components is not None
            else parse_hybrid_components(settings.strategy_router_hybrid_components)
        )
        if not self.hybrid_components:
            # Defensive: parse_hybrid_components never returns an empty
            # tuple, but a caller can pass one explicitly.
            raise ValueError("hybrid_components must not be empty.")

    # ------------------------------------------------------------------
    # Lazy LLM loading (mirrors QueryClassifier._ensure_loaded)
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> LLMCallable | None:
        """Resolve the LLM callable, constructing a real client if needed.

        Called automatically before the first LLM-based routing.  A
        pre-injected ``_resolved_llm`` is respected, making tests
        network-free.  Returns ``None`` (and logs a warning) when the LLM
        is enabled but no API key is configured -- the caller then falls
        back to the rule-based router.
        """
        if self._loaded:
            return self._resolved_llm

        if self._resolved_llm is None:
            api_key = settings.active_llm_api_key
            if _is_unset(api_key):
                logger.warning(
                    "StrategyRouter: LLM is enabled but no API key is "
                    "configured for provider '%s'; falling back to "
                    "rule-based routing.",
                    settings.llm_provider,
                )
                self._loaded = True
                return None

            self._resolved_llm = _build_langchain_llm()

        self._loaded = True
        return self._resolved_llm

    @property
    def is_loaded(self) -> bool:
        """True once the LLM has been resolved (or the lack of a key was
        detected)."""
        return self._loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, step: DecompositionStep) -> RoutingResult:
        """Route a single *step* to a :class:`RetrievalStrategy`.

        A thin convenience wrapper around :meth:`route_batch` for callers
        that only have one sub-query; routing a whole plan in one
        :meth:`route_batch` call is preferred (single LLM round-trip).

        Args:
            step: The :class:`~reporag.agent.planner.DecompositionStep` to
                route.

        Returns:
            A :class:`RoutingResult` for *step*.
        """
        plan = self.route_batch([step], original_query=step.query)
        return plan.routings[0]

    def route_batch(
        self,
        steps: list[DecompositionStep] | DecompositionPlan,
        *,
        original_query: str | None = None,
    ) -> RoutingPlan:
        """Route every step in a batch to a :class:`RetrievalStrategy`.

        The routing path is:

        1. Normalise the input into a flat list of steps (accepts a
           :class:`DecompositionPlan` directly for ergonomics).
        2. If ``use_llm`` is ``False``, jump straight to the rule-based
           router (step 4).
        3. Resolve the LLM (lazy load).  If no API key is configured, fall
           back to the rule-based router.  Call the LLM with the few-shot
           batch prompt and parse the response.  If the response cannot
           be parsed or is missing any expected step, fall back to rules
           for the *whole* batch (cheaper and more consistent than mixing
           LLM and rule routes within one plan).
        4. Apply the confidence threshold *per sub-query*: any sub-query
           whose LLM confidence is below ``confidence_threshold`` is
           overridden to ``hybrid`` with ``fell_back=True`` (the original
           confidence is preserved so callers can see why the fallback
           fired).

        Args:
            steps: Either a :class:`DecompositionPlan` or a flat list of
                :class:`DecompositionStep` objects.  Empty input raises
                ``ValueError`` -- there is nothing to route.
            original_query: The original whole-query text, included in
                the LLM prompt for context.  Defaults to the concatenation
                of the step queries (good enough when the caller does not
                have the original query handy).

        Returns:
            A :class:`RoutingPlan` with one :class:`RoutingResult` per
            step, in the same order as *steps*.

        Raises:
            ValueError: If *steps* is empty.
        """
        if isinstance(steps, DecompositionPlan):
            original_query = original_query or steps.original_query
            step_list = list(steps.steps)
        else:
            step_list = list(steps)

        if not step_list:
            raise ValueError("steps must contain at least one DecompositionStep.")

        if original_query is None:
            original_query = " ".join(step.query for step in step_list)

        routings = self._route_without_threshold(step_list, original_query)

        # Confidence-gated fallback per sub-query: low confidence -> hybrid.
        any_fell_back = False
        final_routings: list[RoutingResult] = []
        for routing in routings:
            if (
                routing.source == "llm"
                and routing.confidence < self.confidence_threshold
            ):
                logger.info(
                    "StrategyRouter: confidence %.3f < threshold %.3f for "
                    "step %r; falling back to hybrid.",
                    routing.confidence,
                    self.confidence_threshold,
                    routing.step_id,
                )
                any_fell_back = True
                final_routings.append(
                    RoutingResult(
                        step_id=routing.step_id,
                        strategy=RetrievalStrategy.HYBRID,
                        confidence=routing.confidence,
                        fell_back=True,
                        source=routing.source,
                        raw_response=routing.raw_response,
                        metadata=routing.metadata,
                    )
                )
            else:
                if routing.fell_back:
                    any_fell_back = True
                final_routings.append(routing)

        source = "llm" if any(r.source == "llm" for r in final_routings) else "rules"
        # If at least one routing came from rules (even on an otherwise
        # LLM plan), report "rules" so callers can detect a partial
        # fallback.  The per-routing ``source`` field carries the truth.
        if any(r.source == "rules" for r in final_routings):
            source = "rules"
        raw_response = next(
            (r.raw_response for r in final_routings if r.raw_response), ""
        )

        per_source = {
            "llm": sum(1 for r in final_routings if r.source == "llm"),
            "rules": sum(1 for r in final_routings if r.source == "rules"),
        }
        return RoutingPlan(
            original_query=original_query,
            steps=tuple(step_list),
            routings=tuple(final_routings),
            source=source,
            raw_response=raw_response,
            fell_back=any_fell_back,
            metadata={
                "hybrid_components": tuple(
                    member.value for member in self.hybrid_components
                ),
                "per_source": per_source,
            },
        )

    def _route_without_threshold(
        self, steps: list[DecompositionStep], original_query: str
    ) -> list[RoutingResult]:
        """Run the LLM or rule-based router, before the threshold check."""
        if not self.use_llm:
            return [rule_based_route(step) for step in steps]

        llm = self._ensure_loaded()
        if llm is None:
            # No API key -- rule-based fallback.
            return [rule_based_route(step) for step in steps]

        prompt = _build_routing_prompt(steps, original_query)
        try:
            raw_response = llm(prompt)
        except Exception as exc:  # noqa: BLE001 - any LLM client failure
            logger.warning(
                "StrategyRouter: LLM call failed (%s); falling back to "
                "rule-based routing.",
                exc,
            )
            return [rule_based_route(step) for step in steps]

        routings_by_id, parse_error = parse_routing_response(
            raw_response, [step.id for step in steps]
        )
        if parse_error:
            logger.warning(
                "StrategyRouter: could not parse LLM response (%s); "
                "falling back to rule-based routing.",
                parse_error,
            )
            return [rule_based_route(step) for step in steps]

        # Preserve step order regardless of the order the LLM returned.
        return [routings_by_id[step.id] for step in steps]

    def __repr__(self) -> str:
        return (
            f"StrategyRouter(use_llm={self.use_llm}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"loaded={self._loaded})"
        )
