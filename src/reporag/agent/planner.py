"""Agentic query planner -- query classifier (Issue 20).

Classifies a natural-language question about a codebase into one of three
strategies so the downstream pipeline can pick the cheapest retrieval path
that will still answer it:

* ``simple-lookup``  -- a single direct retrieval answers it.  Examples:
  "where is ``authenticate`` defined?", "show me the ``User`` model".  These
  go straight to BM25 / graph lookup -- no decomposition, no multi-hop
  orchestration, minimum latency and cost.
* ``multi-hop``      -- the answer requires chaining two or more retrieval
  steps.  Examples: "how does a request go from the API endpoint to the
  database?", "trace the auth flow end-to-end".  These need the decomposer
  (Issue 21) to break them into ordered sub-queries.
* ``exploratory``    -- broad, open-ended questions that need wide retrieval
  across many files.  Examples: "explain the architecture", "give me an
  overview of the codebase".  These skip decomposition and instead fan out
  a broad hybrid retrieval.

Classifying first saves latency and cost: a simple lookup never pays the
decomposition tax, and an exploratory query never wastes hops on narrow
sub-queries.

Design
------
The classifier follows the same conventions as
:class:`~reporag.retrieval.reranker.CrossEncoderReranker` and
:class:`~reporag.embedding.doc_embedder.DocEmbedder` so the three
LLM/model-backed components stay consistent and easy to test:

* **Dual strategy** -- an LLM-based classifier is the primary path; a
  deterministic, network-free rule-based classifier is the fallback.  The
  rule-based path is also the zero-config default (when no API key is set
  or ``QUERY_CLASSIFIER_USE_LLM=false``), so the classifier always works
  offline.
* **Lazy LLM loading** -- the langchain LLM client is constructed on the
  first :meth:`classify` call, not at construction time (cheap,
  test-friendly).  A pre-injected ``llm`` callable is respected, making
  tests network-free exactly like the reranker's ``_FakeCrossEncoder`` seam.
* **Pure module-level helpers** -- the rule-based scorer and the LLM
  response parser are free functions with no side effects, so they are
  unit-testable without any LLM or model.
* **Confidence-gated fallback** -- if the LLM's confidence is below
  ``settings.query_classifier_confidence_threshold`` the result is
  overridden to ``multi-hop`` (the safest default: decomposition is
  correct for both genuinely multi-hop queries and ambiguous ones, while
  a wrong ``simple-lookup`` would skip needed hops).

The :class:`QueryDecomposer` (Issue 21) will be added to this module in a
later issue; the classifier is implemented first because the decomposer
consumes its output.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from reporag.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

QueryType = Literal["simple-lookup", "multi-hop", "exploratory"]
"""The three classification categories from Issue 20."""

_VALID_QUERY_TYPES: frozenset[str] = frozenset(
    {"simple-lookup", "multi-hop", "exploratory"}
)


@dataclass(frozen=True)
class ClassificationResult:
    """The outcome of classifying a single query.

    Attributes:
        query_type: One of ``simple-lookup``, ``multi-hop``, or
            ``exploratory``.
        confidence: Model / rule confidence in the classification, in
            ``[0.0, 1.0]``.  When the confidence-threshold fallback fires
            this is the *original* confidence (so callers can see *why* the
            fallback triggered), and :attr:`fell_back` is set to ``True``.
        fell_back: ``True`` when the low-confidence fallback overrode the
            original classification to ``multi-hop``.
        source: ``"llm"`` when the LLM produced the classification,
            ``"rules"`` when the rule-based fallback was used (either because
            the LLM was disabled, no API key was configured, or the LLM
            response could not be parsed).
        raw_response: The raw LLM response text (``""`` for the rule-based
            path).  Kept for debugging and observability -- not for
            programmatic use.
        metadata: Free-form extras (e.g. rule scores for the rule-based
            path, or the parsed JSON payload for the LLM path).
    """

    query_type: QueryType
    confidence: float
    fell_back: bool = False
    source: Literal["llm", "rules"] = "rules"
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Few-shot prompt
# ---------------------------------------------------------------------------

# Ten carefully-chosen examples covering all three categories and the
# boundary cases (a "how does X work" can be multi-hop *or* exploratory
# depending on breadth).  These are the same examples the unit tests
# assert against, so the prompt and the tests stay in lockstep.
_FEW_SHOT_EXAMPLES: tuple[tuple[str, QueryType], ...] = (
    ("Where is the authenticate function defined?", "simple-lookup"),
    ("Show me the User model class.", "simple-lookup"),
    ("Find the file that contains the DatabaseConfig class.", "simple-lookup"),
    ("What line is the handle_request function on?", "simple-lookup"),
    ("How does the auth flow work end-to-end?", "multi-hop"),
    ("How does a request go from the API endpoint to the database?", "multi-hop"),
    (
        "What calls the authenticate_user function and what does it call next?",
        "multi-hop",
    ),
    ("Trace the path from the login route to the session token creation.", "multi-hop"),
    ("Explain the overall architecture of this codebase.", "exploratory"),
    ("Give me an overview of how the ingestion pipeline is organized.", "exploratory"),
    ("What are the main components and how are they structured?", "exploratory"),
)


def _build_classification_prompt(query: str) -> str:
    """Build the few-shot classification prompt for *query*.

    The prompt asks the LLM to return **strict JSON** with ``query_type`` and
    ``confidence`` keys so the response is machine-parseable.  Few-shot
    examples anchor the three categories and the expected confidence range.
    """
    examples_block = "\n".join(
        f"Query: {example}\nType: {label}" for example, label in _FEW_SHOT_EXAMPLES
    )
    return (
        "You are a query classifier for a code intelligence system.\n"
        "Classify the user's query into exactly one of three categories:\n"
        "\n"
        "- simple-lookup: A single direct retrieval (BM25 or graph lookup) "
        "answers it. The query asks where a specific symbol is defined, "
        "asks to find/show a specific file/class/function, or asks for a "
        "specific line number.\n"
        "- multi-hop: The answer requires chaining two or more retrieval "
        "steps. The query asks how something works end-to-end, asks to "
        "trace a path/flow, or asks what calls X and what X calls.\n"
        "- exploratory: Broad, open-ended questions needing wide retrieval "
        "across many files. The query asks for an overview, architecture "
        "explanation, or high-level summary of the codebase.\n"
        "\n"
        "Examples:\n"
        f"{examples_block}\n"
        "\n"
        "Now classify this query:\n"
        f"Query: {query}\n"
        "\n"
        "Respond with ONLY a JSON object on a single line in this exact "
        "format (no markdown, no explanation):\n"
        '{"query_type": "<one of simple-lookup|multi-hop|exploratory>", '
        '"confidence": <float between 0.0 and 1.0>}'
    )


# ---------------------------------------------------------------------------
# Rule-based classifier (pure, network-free, unit-testable)
# ---------------------------------------------------------------------------

# Weighted regex signal patterns.  Each pattern votes for a category; the
# category with the highest total weight wins.  Patterns are ordered from
# most specific to least specific so the first match in a group dominates.
#
# ``simple-lookup`` signals: the query names a specific symbol and asks for
# its location / definition / line.
_SIMPLE_LOOKUP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhere\s+(is|are)\b", re.IGNORECASE),
    re.compile(r"\bfind\s+(the\s+)?(file|class|function|method|def)\b", re.IGNORECASE),
    re.compile(r"\bshow\s+me\b", re.IGNORECASE),
    re.compile(r"\bdefinition\s+of\b", re.IGNORECASE),
    re.compile(r"\bdefined\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(line|file)\b", re.IGNORECASE),
    re.compile(r"\blocate\b", re.IGNORECASE),
    re.compile(r"\bdeclare[ds]?\b", re.IGNORECASE),
)

# ``multi-hop`` signals: the query asks about a flow, a path, a chain, or
# uses "how does X work" with a specific subject (narrow enough to trace).
_MULTI_HOP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhow\s+does\b.{0,40}\bwork\b", re.IGNORECASE),
    re.compile(r"\bend-to-end\b", re.IGNORECASE),
    re.compile(r"\btrace\b", re.IGNORECASE),
    re.compile(r"\bflow\b", re.IGNORECASE),
    re.compile(r"\bpath\s+from\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+calls\b", re.IGNORECASE),
    re.compile(r"\bchain\b", re.IGNORECASE),
    re.compile(r"\bstep\s+by\s+step\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+.+\s+to\s+the\b", re.IGNORECASE),
)

# ``exploratory`` signals: the query asks for an overview, architecture, or
# high-level structure -- broad and open-ended with no specific symbol.
_EXPLORATORY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bexplain\s+(the\s+)?(overall\s+)?architecture\b", re.IGNORECASE),
    re.compile(r"\boverview\b", re.IGNORECASE),
    re.compile(r"\barchitecture\b", re.IGNORECASE),
    re.compile(r"\bhigh-?level\b", re.IGNORECASE),
    re.compile(r"\bmain\s+components\b", re.IGNORECASE),
    re.compile(r"\bstructured\b", re.IGNORECASE),
    re.compile(r"\borganized\b", re.IGNORECASE),
    re.compile(r"\bsummarize\b", re.IGNORECASE),
    re.compile(r"\bcodebase\b", re.IGNORECASE),
)


def _score_patterns(query: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    """Return the number of *patterns* that match *query*.

    Each pattern contributes at most one vote (not one per match) so a query
    that repeats the same signal word does not drown out other signals.
    """
    return sum(1 for pattern in patterns if pattern.search(query))


def rule_based_classify(query: str) -> ClassificationResult:
    """Classify *query* using weighted regex signal patterns.

    This is the deterministic, network-free fallback.  It scores each
    category by counting how many of its signal patterns match the query;
    the category with the highest score wins.  Confidence is the winner's
    share of the total votes (``winner / total``), clamped to ``[0, 1]``.
    When all categories tie at zero the query is treated as ``multi-hop``
    (the safest default) with confidence ``0.0`` so the confidence-threshold
    fallback in :meth:`QueryClassifier.classify` fires naturally.

    Args:
        query: The natural-language query to classify.

    Returns:
        A :class:`ClassificationResult` with ``source="rules"`` and the
        per-category vote counts in ``metadata["scores"]``.
    """
    scores = {
        "simple-lookup": _score_patterns(query, _SIMPLE_LOOKUP_PATTERNS),
        "multi-hop": _score_patterns(query, _MULTI_HOP_PATTERNS),
        "exploratory": _score_patterns(query, _EXPLORATORY_PATTERNS),
    }
    total = sum(scores.values())

    if total == 0:
        # No signal at all -- safest default is multi-hop with zero
        # confidence so the caller's threshold fallback can promote it.
        return ClassificationResult(
            query_type="multi-hop",
            confidence=0.0,
            source="rules",
            metadata={"scores": scores},
        )

    winner = max(scores, key=lambda k: scores[k])
    # Resolve ties deterministically: multi-hop > exploratory > simple-lookup.
    # This ordering prefers the safer (more expensive) default when signals
    # are ambiguous, matching the issue's "fallback to multi-hop" guidance.
    top_score = scores[winner]
    tied = [k for k, v in scores.items() if v == top_score]
    if len(tied) > 1:
        for preference in ("multi-hop", "exploratory", "simple-lookup"):
            if preference in tied:
                winner = preference
                break

    confidence = top_score / total
    return ClassificationResult(
        query_type=winner,  # type: ignore[arg-type]
        confidence=confidence,
        source="rules",
        metadata={"scores": scores},
    )


# ---------------------------------------------------------------------------
# LLM response parsing (pure, unit-testable)
# ---------------------------------------------------------------------------

# Matches a JSON object anywhere in the response.  The LLM is instructed to
# return a single-line JSON object, but we tolerate surrounding prose /
# markdown fences by extracting the first ``{...}`` block.
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_llm_response(raw: str) -> ClassificationResult:
    """Parse the raw LLM response text into a :class:`ClassificationResult`.

    The LLM is prompted to return strict JSON::

        {"query_type": "simple-lookup", "confidence": 0.95}

    This parser is tolerant: it extracts the first ``{...}`` block (so
    accidental markdown fences or leading prose do not break parsing),
    coerces the ``query_type`` to a valid category (rejecting unknown
    values), and clamps ``confidence`` to ``[0.0, 1.0]``.

    Args:
        raw: The raw text returned by the LLM.

    Returns:
        A :class:`ClassificationResult` with ``source="llm"``.  If the
        response cannot be parsed at all, a zero-confidence ``multi-hop``
        result is returned (so the caller's threshold fallback fires) with
        the parse error recorded in ``metadata["parse_error"]``.
    """
    if not raw or not raw.strip():
        return ClassificationResult(
            query_type="multi-hop",
            confidence=0.0,
            source="llm",
            raw_response=raw,
            metadata={"parse_error": "empty response"},
        )

    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        return ClassificationResult(
            query_type="multi-hop",
            confidence=0.0,
            source="llm",
            raw_response=raw,
            metadata={"parse_error": "no JSON object found"},
        )

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return ClassificationResult(
            query_type="multi-hop",
            confidence=0.0,
            source="llm",
            raw_response=raw,
            metadata={"parse_error": f"invalid JSON: {exc}"},
        )

    raw_type = str(payload.get("query_type", "")).strip().lower()
    if raw_type not in _VALID_QUERY_TYPES:
        return ClassificationResult(
            query_type="multi-hop",
            confidence=0.0,
            source="llm",
            raw_response=raw,
            metadata={
                "parse_error": f"invalid query_type: {raw_type!r}",
                "raw_payload": payload,
            },
        )

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return ClassificationResult(
        query_type=raw_type,  # type: ignore[arg-type]
        confidence=confidence,
        source="llm",
        raw_response=raw,
        metadata={"raw_payload": payload},
    )


# ---------------------------------------------------------------------------
# QueryClassifier
# ---------------------------------------------------------------------------

# A callable that takes a prompt string and returns the LLM's response text.
# This is the test seam: tests inject a deterministic fake instead of a real
# langchain LLM, keeping every test network-free.
LLMCallable = Callable[[str], str]


class QueryClassifier:
    """Classifies a query into simple-lookup / multi-hop / exploratory.

    The primary path is an LLM-based classifier with few-shot examples; a
    deterministic rule-based classifier is the fallback when the LLM is
    disabled, no API key is configured, or the LLM response cannot be
    parsed.  If the LLM's confidence is below
    ``settings.query_classifier_confidence_threshold`` the result is
    overridden to ``multi-hop`` (the safest default).

    Args:
        llm: A pre-built callable ``(prompt: str) -> str`` that stands in for
            the langchain LLM client.  Passing a callable is the supported
            test seam -- it makes the classifier network-free and avoids the
            API key requirement, exactly like the reranker's
            ``_FakeCrossEncoder``.  When ``None`` (default) a real langchain
            LLM is constructed lazily on the first :meth:`classify` call
            using ``settings.llm_provider`` and the configured API key.
        confidence_threshold: Below this confidence the classification
            falls back to ``multi-hop``.  Defaults to
            ``settings.query_classifier_confidence_threshold``.  Must be in
            ``[0.0, 1.0]``.
        use_llm: When ``True`` (default) the LLM path is used; when ``False``
            only the rule-based classifier runs (no LLM is ever loaded).
            Defaults to ``settings.query_classifier_use_llm``.

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

        Called automatically before the first LLM-based classification.  A
        pre-injected ``_resolved_llm`` is respected, making tests
        network-free.  Returns ``None`` (and logs a warning) when the LLM
        is enabled but no API key is configured -- the caller then falls
        back to the rule-based classifier.
        """
        if self._loaded:
            return self._resolved_llm

        if self._resolved_llm is None:
            api_key = settings.active_llm_api_key
            if _is_unset_secret(api_key):
                logger.warning(
                    "QueryClassifier: LLM is enabled but no API key is "
                    "configured for provider '%s'; falling back to rule-based "
                    "classification.",
                    settings.llm_provider,
                )
                self._loaded = True
                return None

            self._resolved_llm = self._build_langchain_llm()

        self._loaded = True
        return self._resolved_llm

    @staticmethod
    def _build_langchain_llm() -> LLMCallable:
        """Construct the langchain LLM client from settings.

        The langchain ``invoke`` API returns a message object whose
        ``content`` attribute holds the text; we wrap it in a plain
        ``(prompt) -> str`` callable so the rest of the classifier is
        provider-agnostic and testable with a simple fake.
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
            # langchain returns a message object with a ``content`` attr.
            return str(getattr(response, "content", response))

        return _invoke

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True once the LLM has been resolved (or the lack of a key was detected)."""
        return self._loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> ClassificationResult:
        """Classify *query* into simple-lookup, multi-hop, or exploratory.

        The classification path is:

        1. If ``use_llm`` is ``False``, jump straight to the rule-based
           classifier (step 4).
        2. Resolve the LLM (lazy load).  If no API key is configured, fall
           back to the rule-based classifier.
        3. Call the LLM with the few-shot prompt and parse the response.  If
           the response cannot be parsed, fall back to the rule-based
           classifier.
        4. Apply the confidence threshold: if ``confidence <
           confidence_threshold``, override ``query_type`` to ``multi-hop``
           and set ``fell_back=True`` (the original confidence is preserved
           so callers can see why the fallback fired).

        Args:
            query: The natural-language query to classify.

        Returns:
            A :class:`ClassificationResult` with ``query_type``,
            ``confidence``, ``source``, and ``fell_back`` populated.

        Raises:
            ValueError: If *query* is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        result = self._classify_without_threshold(query)

        # Confidence-gated fallback: low confidence -> multi-hop (safest).
        if result.confidence < self.confidence_threshold:
            logger.info(
                "QueryClassifier: confidence %.3f < threshold %.3f for "
                "query %r; falling back to multi-hop.",
                result.confidence,
                self.confidence_threshold,
                query,
            )
            return ClassificationResult(
                query_type="multi-hop",
                confidence=result.confidence,
                fell_back=True,
                source=result.source,
                raw_response=result.raw_response,
                metadata=result.metadata,
            )

        return result

    def _classify_without_threshold(self, query: str) -> ClassificationResult:
        """Run the LLM or rule-based classifier, before the threshold check."""
        if not self.use_llm:
            return rule_based_classify(query)

        llm = self._ensure_loaded()
        if llm is None:
            # No API key -- rule-based fallback.
            return rule_based_classify(query)

        prompt = _build_classification_prompt(query)
        try:
            raw_response = llm(prompt)
        except Exception as exc:
            logger.warning(
                "QueryClassifier: LLM call failed (%s); falling back to "
                "rule-based classification.",
                exc,
            )
            return rule_based_classify(query)

        result = parse_llm_response(raw_response)
        if result.metadata.get("parse_error"):
            logger.warning(
                "QueryClassifier: could not parse LLM response (%s); "
                "falling back to rule-based classification.",
                result.metadata["parse_error"],
            )
            return rule_based_classify(query)

        return result

    def __repr__(self) -> str:
        return (
            f"QueryClassifier(use_llm={self.use_llm}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"loaded={self._loaded})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_unset_secret(secret: Any) -> bool:
    """Return True if *secret* is an unset or placeholder SecretStr.

    Mirrors :func:`reporag.config._is_unset` but is duplicated here to avoid
    importing the private helper from config (which would couple this module
    to config internals).  Accepts a :class:`pydantic.SecretStr` or a plain
    string.
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


# ============================================================================
# QueryDecomposer Section (Issue 21)
# ============================================================================


AnswerType = Literal["code", "explanation", "list"]


@dataclass(frozen=True)
class SubQuery:
    """An atomic query step to be run against the codebase."""

    id: str
    query: str = ""
    expected_answer_type: AnswerType = "explanation"
    depends_on: tuple[str, ...] = ()
    text: str = ""
    context_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.query and self.text:
            object.__setattr__(self, "query", self.text)
        elif self.query and not self.text:
            object.__setattr__(self, "text", self.query)

        if not self.depends_on and self.context_from:
            object.__setattr__(self, "depends_on", self.context_from)
        elif self.depends_on and not self.context_from:
            object.__setattr__(self, "context_from", self.depends_on)


@dataclass(frozen=True)
class DecompositionPlan:
    """An ordered list of sub-queries to retrieve the answer to a complex query."""

    original_query: str
    steps: tuple[SubQuery, ...]
    source: Literal["llm", "rules"]
    metadata: dict[str, Any] = field(default_factory=dict)


class DecompositionState(TypedDict):
    """LangGraph execution state for QueryDecomposer."""

    query: str
    repo_context: dict[str, Any]
    prompt: str
    raw_response: str
    steps: list[dict[str, Any]]
    source: Literal["llm", "rules"]
    error: str | None


# ---------------------------------------------------------------------------
# Prompt Construction & Parsing
# ---------------------------------------------------------------------------


def _build_decomposition_prompt(
    query: str, repo_context: dict[str, Any] | None = None
) -> str:
    """Build a few-shot prompt instructing the LLM to return structured steps."""
    context_str = ""
    if repo_context:
        if "modules" in repo_context:
            context_str += f"Available modules: {repo_context['modules']}\n"
        if "symbols" in repo_context:
            context_str += f"Key symbols: {repo_context['symbols']}\n"

    return (
        "You are an expert query decomposer for a code repository.\n"
        "Your task is to break down a complex, multi-hop user query into a sequence of 2-5 ordered sub-queries.\n"
        "If a query is simple and does not need decomposition, return a single sub-query.\n"
        "\n"
        "Use the repository context (available modules, key symbols) to construct query steps that target specific parts of the codebase.\n"
        "Identify dependencies between steps: a step should depend on prior steps if it requires their retrieved context to proceed.\n"
        "\n"
        "Repository Context:\n"
        f"{context_str or 'None'}\n"
        "\n"
        "Format the output as a strict JSON list of objects, with no markdown formatting or explanation. Each object must have:\n"
        '- "id": unique identifier (e.g. "step_1")\n'
        '- "query": the sub-query string targeting a specific search or lookup\n'
        '- "expected_answer_type": "code" (for files/symbols/definitions), "explanation" (for flow/architecture), or "list"\n'
        '- "depends_on": list of step IDs this step depends on (e.g. ["step_1"])\n'
        "\n"
        "Example 1 (Multi-hop):\n"
        "Query: How does a request go from the API endpoint to the database?\n"
        "Repository Context: modules=['api', 'routes', 'db', 'models']\n"
        "Response:\n"
        "[\n"
        '  {"id": "step_1", "query": "Find where API endpoints are defined in the api or routes modules.", "expected_answer_type": "code", "depends_on": []},\n'
        '  {"id": "step_2", "query": "Find how database queries or connections are made in the db or models modules.", "expected_answer_type": "code", "depends_on": []},\n'
        '  {"id": "step_3", "query": "Trace the flow from the API endpoint handling to the database connection/query execution.", "expected_answer_type": "explanation", "depends_on": ["step_1", "step_2"]}\n'
        "]\n"
        "\n"
        "Example 2 (Simple lookup):\n"
        "Query: Where is the authenticate function defined?\n"
        "Repository Context: modules=['auth', 'utils']\n"
        "Response:\n"
        "[\n"
        '  {"id": "step_1", "query": "Find the definition of the authenticate function in the auth or utils modules.", "expected_answer_type": "code", "depends_on": []}\n'
        "]\n"
        "\n"
        "Now decompose this query:\n"
        f"Query: {query}\n"
    )


def parse_decomposition_response(raw: str) -> list[dict[str, Any]]:
    """Parse the raw JSON response from the LLM containing a list of sub-queries."""
    if not raw or not raw.strip():
        raise ValueError("Empty response")

    # Tolerant parsing: find the JSON list
    match = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
    text = match.group(0) if match else raw.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, list):
        if isinstance(data, dict):
            data = [data]
        else:
            raise ValueError("Expected a list of sub-queries")

    parsed_steps = []
    for step in data:
        if not isinstance(step, dict):
            continue

        step_id = str(step.get("id", "")).strip()
        query_text = str(step.get("query", "")).strip()
        ans_type = str(step.get("expected_answer_type", "explanation")).strip().lower()
        if ans_type not in ("code", "explanation", "list"):
            ans_type = "explanation"

        raw_deps = step.get("depends_on") or step.get("context_from") or []
        if isinstance(raw_deps, str):
            deps = [d.strip() for d in raw_deps.split(",") if d.strip()]
        elif isinstance(raw_deps, list):
            deps = [str(d).strip() for d in raw_deps if str(d).strip()]
        else:
            deps = []

        if not step_id or not query_text:
            continue

        parsed_steps.append(
            {
                "id": step_id,
                "query": query_text,
                "expected_answer_type": ans_type,
                "depends_on": deps,
            }
        )

    if not parsed_steps:
        raise ValueError("No valid sub-queries parsed")

    return parsed_steps


# ---------------------------------------------------------------------------
# Rule-based fallback decomposition (pure, deterministic, unit-testable)
# ---------------------------------------------------------------------------


def rule_based_decompose(
    query: str, repo_context: dict[str, Any] | None = None
) -> list[SubQuery]:
    """Fallback query decomposition using rule-based keyword heuristics."""
    classification = rule_based_classify(query)

    if classification.query_type == "simple-lookup":
        return [
            SubQuery(
                id="step_1",
                query=f"Find the definition and references for: {query}",
                expected_answer_type="code",
                depends_on=(),
            )
        ]

    modules = repo_context.get("modules", []) if repo_context else []
    symbols = repo_context.get("symbols", []) if repo_context else []

    matched_modules = []
    for m in modules:
        if m.lower() in query.lower():
            matched_modules.append(m)

    matched_symbols = []
    for s in symbols:
        if s.lower() in query.lower():
            matched_symbols.append(s)

    # Heuristic flow when 2+ modules are detected in query
    if len(matched_modules) >= 2:
        mod_1 = matched_modules[0]
        mod_2 = matched_modules[1]
        return [
            SubQuery(
                id="step_1",
                query=f"Analyze components, endpoints, or functions related to {mod_1}.",
                expected_answer_type="code",
                depends_on=(),
            ),
            SubQuery(
                id="step_2",
                query=f"Analyze components, schemas, or query execution related to {mod_2}.",
                expected_answer_type="code",
                depends_on=(),
            ),
            SubQuery(
                id="step_3",
                query=f"Trace the logic path and execution flow from {mod_1} to {mod_2} in the query: '{query}'.",
                expected_answer_type="explanation",
                depends_on=("step_1", "step_2"),
            ),
        ]

    # Heuristic flow when 2+ symbols are detected in query
    if len(matched_symbols) >= 2:
        sym_1 = matched_symbols[0]
        sym_2 = matched_symbols[1]
        return [
            SubQuery(
                id="step_1",
                query=f"Find definitions and trace callers/callees of symbol: '{sym_1}'.",
                expected_answer_type="code",
                depends_on=(),
            ),
            SubQuery(
                id="step_2",
                query=f"Find definitions and trace callers/callees of symbol: '{sym_2}'.",
                expected_answer_type="code",
                depends_on=(),
            ),
            SubQuery(
                id="step_3",
                query=f"Determine how token flows or call relationships link '{sym_1}' and '{sym_2}'.",
                expected_answer_type="explanation",
                depends_on=("step_1", "step_2"),
            ),
        ]

    # General fallback flow (3 steps)
    step_1_query = "Identify initial components and entry points related to the query."
    if matched_modules:
        step_1_query = f"Identify entry points and definitions in the '{', '.join(matched_modules)}' modules."
    elif matched_symbols:
        step_1_query = f"Locate definitions and references for symbol(s): {', '.join(matched_symbols)}."

    return [
        SubQuery(
            id="step_1",
            query=step_1_query,
            expected_answer_type="code",
            depends_on=(),
        ),
        SubQuery(
            id="step_2",
            query="Analyze downstream components, helper functions, or databases connected to the entry points.",
            expected_answer_type="code",
            depends_on=("step_1",),
        ),
        SubQuery(
            id="step_3",
            query=f"Trace the end-to-end execution flow and explain the relationship for the query: '{query}'.",
            expected_answer_type="explanation",
            depends_on=("step_1", "step_2"),
        ),
    ]


# ---------------------------------------------------------------------------
# QueryDecomposer Class
# ---------------------------------------------------------------------------


class QueryDecomposer:
    """Decomposes complex query into ordered sub-queries using a LangGraph state machine.

    Includes lazy loading, deterministic fallback, and maximum step limits.
    """

    def __init__(
        self,
        llm: LLMCallable | None = None,
        *,
        max_steps: int | None = None,
        use_llm: bool | None = None,
    ) -> None:
        if max_steps is None:
            max_steps = settings.query_decomposer_max_steps
        self.max_steps = max_steps
        self.use_llm = (
            use_llm if use_llm is not None else settings.query_classifier_use_llm
        )
        self._resolved_llm = llm
        self._loaded = False

        from langgraph.graph import END, START, StateGraph

        workflow = StateGraph(DecompositionState)
        workflow.add_node("build_prompt", self._node_build_prompt)
        workflow.add_node("call_llm", self._node_call_llm)
        workflow.add_node("parse_response", self._node_parse_response)
        workflow.add_node("rule_based", self._node_rule_based)

        workflow.add_edge(START, "build_prompt")
        workflow.add_edge("build_prompt", "call_llm")
        workflow.add_edge("call_llm", "parse_response")

        workflow.add_conditional_edges(
            "parse_response",
            self._router,
            {"rule_based": "rule_based", END: END},
        )
        workflow.add_edge("rule_based", END)

        self._graph = workflow.compile()

    def _ensure_loaded(self) -> LLMCallable | None:
        if self._loaded:
            return self._resolved_llm

        if self._resolved_llm is None:
            api_key = settings.active_llm_api_key
            if _is_unset_secret(api_key):
                logger.warning(
                    "QueryDecomposer: LLM is enabled but no API key is "
                    "configured for provider '%s'; falling back to rule-based "
                    "classification.",
                    settings.llm_provider,
                )
                self._loaded = True
                return None

            self._resolved_llm = QueryClassifier._build_langchain_llm()

        self._loaded = True
        return self._resolved_llm

    def _node_build_prompt(self, state: DecompositionState) -> dict[str, Any]:
        prompt = _build_decomposition_prompt(state["query"], state.get("repo_context"))
        return {"prompt": prompt}

    def _node_call_llm(self, state: DecompositionState) -> dict[str, Any]:
        if not self.use_llm:
            return {"error": "LLM disabled"}
        llm = self._ensure_loaded()
        if llm is None:
            return {"error": "LLM not configured"}
        try:
            raw_response = llm(state["prompt"])
            return {"raw_response": raw_response}
        except Exception as exc:
            logger.warning("QueryDecomposer: LLM call failed (%s)", exc)
            return {"error": f"LLM call failed: {exc}"}

    def _node_parse_response(self, state: DecompositionState) -> dict[str, Any]:
        raw = state.get("raw_response", "")
        try:
            steps = parse_decomposition_response(raw)
            return {"steps": steps, "source": "llm", "error": None}
        except Exception as exc:
            logger.warning("QueryDecomposer: failed to parse response (%s)", exc)
            return {"error": f"Parse error: {exc}"}

    def _node_rule_based(self, state: DecompositionState) -> dict[str, Any]:
        steps = rule_based_decompose(state["query"], state.get("repo_context"))
        step_dicts = [
            {
                "id": s.id,
                "query": s.query,
                "expected_answer_type": s.expected_answer_type,
                "depends_on": list(s.depends_on),
            }
            for s in steps
        ]
        return {"steps": step_dicts, "source": "rules"}

    def _router(self, state: DecompositionState) -> str:
        from langgraph.graph import END

        if state.get("error") is not None:
            return "rule_based"
        return END

    def decompose(
        self, query: str, repo_context: dict[str, Any] | None = None
    ) -> DecompositionPlan:
        """Decompose a query into a sequence of sub-queries.

        Args:
            query: The user query string.
            repo_context: Dictionary containing available modules and key symbols.

        Returns:
            A DecompositionPlan containing the steps of sub-queries.

        Raises:
            ValueError: If the query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        initial_state: DecompositionState = {
            "query": query,
            "repo_context": repo_context or {},
            "prompt": "",
            "raw_response": "",
            "steps": [],
            "source": "rules",
            "error": None,
        }

        final_state = self._graph.invoke(initial_state)

        # Extract steps
        step_dicts = final_state.get("steps") or []

        # Clamp/trim to max_steps
        if len(step_dicts) > self.max_steps:
            step_dicts = step_dicts[: self.max_steps]

        # Fix dependency chains for remaining steps
        valid_ids = {s["id"] for s in step_dicts}
        steps = []
        for s in step_dicts:
            deps = tuple(d for d in s.get("depends_on", []) if d in valid_ids)
            steps.append(
                SubQuery(
                    id=s["id"],
                    query=s["query"],
                    expected_answer_type=s["expected_answer_type"],
                    depends_on=deps,
                )
            )

        return DecompositionPlan(
            original_query=query,
            steps=tuple(steps),
            source=final_state.get("source", "rules"),
            metadata={
                "raw_response": final_state.get("raw_response", ""),
                "error": final_state.get("error"),
            },
        )

    def __repr__(self) -> str:
        return (
            f"QueryDecomposer(use_llm={self.use_llm}, "
            f"max_steps={self.max_steps}, "
            f"loaded={self._loaded})"
        )
