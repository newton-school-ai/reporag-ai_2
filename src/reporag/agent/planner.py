"""Agentic query planner -- classifier (Issue 20) and decomposer (Issue 21).

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

* **Safety default** -- an LLM-based classifier is the primary path; if
  the LLM fails, is disabled, or returns an unparseable response, it
  falls back to a zero-confidence ``multi-hop`` result.
* **Lazy LLM loading** -- the langchain LLM client is constructed on the
  first :meth:`classify` call, not at construction time (cheap,
  test-friendly).  A pre-injected ``llm`` callable is respected, making
  tests network-free exactly like the reranker's ``_FakeCrossEncoder`` seam.
* **Pure module-level helpers** -- the LLM
  response parser is a free function with no side effects, so it is
  unit-testable without any LLM or model.
* **Confidence-gated fallback** -- if the LLM's confidence is below
  ``settings.query_classifier_confidence_threshold`` the result is
  overridden to ``multi-hop`` (the safest default: decomposition is
  correct for both genuinely multi-hop queries and ambiguous ones, while
  a wrong ``simple-lookup`` would skip needed hops).

The :class:`QueryDecomposer` (Issue 21) decomposes multi-hop queries into
ordered sub-queries with dependency edges, using a LangGraph state machine.
The classifier is run first; the decomposer consumes its output.
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
            ``"fallback"`` when the fallback was used (either because
            the LLM was disabled, no API key was configured, or the LLM
            response could not be parsed).
        raw_response: The raw LLM response text (``""`` for the fallback
            path).  Kept for debugging and observability -- not for
            programmatic use.
        metadata: Free-form extras (e.g. the parsed JSON payload for the
            LLM path, or error details for the fallback path).
    """

    query_type: QueryType
    confidence: float
    fell_back: bool = False
    source: Literal["llm", "fallback"] = "fallback"
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


AnswerType = Literal["code", "explanation", "list"]
"""Expected answer type for a decomposition sub-query step (Issue 21)."""

_VALID_ANSWER_TYPES: frozenset[str] = frozenset({"code", "explanation", "list"})


@dataclass(frozen=True)
class SubQuery:
    """A single step in a :class:`DecompositionPlan`.

    Attributes:
        id: Unique step identifier within the plan (e.g. ``"step-1"``).
        query: The sub-query text to be answered by a retrieval step.
        expected_answer_type: What kind of answer this sub-query expects:
            ``"code"`` for source code, ``"explanation"`` for prose, or
            ``"list"`` for an enumeration.
        depends_on: IDs of prior steps whose results should be available
            as context when answering this step.  Empty for the first step.
    """

    id: str
    query: str
    expected_answer_type: AnswerType
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecompositionPlan:
    """The outcome of decomposing a query into ordered sub-queries.

    Attributes:
        steps: Ordered sub-queries.  A single-element tuple means the
            query did not require decomposition.
        original_query: The input query that was decomposed.
        source: ``"llm"`` when the LLM produced the decomposition,
            ``"fallback"`` when the single-step fallback was used.
        raw_response: The raw LLM response text (``""`` for the
            fallback path).  Kept for debugging and observability.
        metadata: Free-form extras (e.g. the parsed JSON payload for the
            LLM path, or a ``"parse_error"`` key when parsing failed).
    """

    steps: tuple[SubQuery, ...]
    original_query: str
    source: Literal["llm", "fallback"] = "fallback"
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
    zero-confidence multi-hop result is the fallback when the LLM is
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
            the fallback runs (no LLM is ever loaded).
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
        back to the zero-confidence fallback.
        """
        if self._loaded:
            return self._resolved_llm

        if self._resolved_llm is None:
            api_key = settings.active_llm_api_key
            if _is_unset_secret(api_key):
                logger.warning(
                    "QueryClassifier: LLM is enabled but no API key is "
                    "configured for provider '%s'; falling back to zero-confidence "
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

        1. If ``use_llm`` is ``False``, jump straight to the zero-confidence
           fallback (step 4).
        2. Resolve the LLM (lazy load).  If no API key is configured, fall
           back to the zero-confidence fallback.
        3. Call the LLM with the few-shot prompt and parse the response.  If
           the response cannot be parsed, fall back to the zero-confidence
           fallback.
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
        """Run the LLM classifier, before the threshold check."""
        if not self.use_llm:
            return ClassificationResult(
                query_type="multi-hop",
                confidence=0.0,
                source="fallback",
                metadata={"error": "LLM disabled"},
            )

        llm = self._ensure_loaded()
        if llm is None:
            # No API key -- fallback.
            return ClassificationResult(
                query_type="multi-hop",
                confidence=0.0,
                source="fallback",
                metadata={"error": "No API key configured"},
            )

        prompt = _build_classification_prompt(query)
        try:
            raw_response = llm(prompt)
        except Exception as exc:
            logger.warning(
                "QueryClassifier: LLM call failed (%s); falling back to "
                "zero-confidence classification.",
                exc,
            )
            return ClassificationResult(
                query_type="multi-hop",
                confidence=0.0,
                source="fallback",
                metadata={"error": f"LLM call failed: {exc}"},
            )

        result = parse_llm_response(raw_response)
        if result.metadata.get("parse_error"):
            logger.warning(
                "QueryClassifier: could not parse LLM response (%s); "
                "falling back to zero-confidence classification.",
                result.metadata["parse_error"],
            )
            return ClassificationResult(
                query_type="multi-hop",
                confidence=0.0,
                source="fallback",
                metadata={"error": f"Parse error: {result.metadata['parse_error']}"},
            )

        return result

    def __repr__(self) -> str:
        return (
            f"QueryClassifier(use_llm={self.use_llm}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"loaded={self._loaded})"
        )


# ---------------------------------------------------------------------------
# Decomposition prompt
# ---------------------------------------------------------------------------

# Few-shot examples for the decomposition prompt.  Each entry is a
# (query, expected_steps) pair where expected_steps is a list of dicts
# matching the JSON format the LLM is asked to return.
_DECOMPOSITION_EXAMPLES: tuple[tuple[str, list[dict[str, Any]]], ...] = (
    (
        "How does a request go from the API endpoint to the database?",
        [
            {
                "id": "step-1",
                "query": "Find the API endpoint entry point",
                "expected_answer_type": "code",
                "depends_on": [],
            },
            {
                "id": "step-2",
                "query": "Trace the request processing chain",
                "expected_answer_type": "explanation",
                "depends_on": ["step-1"],
            },
            {
                "id": "step-3",
                "query": "Find the database access layer",
                "expected_answer_type": "code",
                "depends_on": ["step-2"],
            },
        ],
    ),
    (
        "How does the auth flow work end-to-end?",
        [
            {
                "id": "step-1",
                "query": "Find the authentication entry point",
                "expected_answer_type": "code",
                "depends_on": [],
            },
            {
                "id": "step-2",
                "query": "Trace the auth middleware chain",
                "expected_answer_type": "explanation",
                "depends_on": ["step-1"],
            },
            {
                "id": "step-3",
                "query": "Find the token validation logic",
                "expected_answer_type": "code",
                "depends_on": ["step-2"],
            },
        ],
    ),
)


def _build_decomposition_prompt(query: str, repo_context: dict[str, Any]) -> str:
    """Build the decomposition prompt for *query* with *repo_context*.

    The prompt asks the LLM to return **strict JSON** with a ``steps`` array
    where each step has ``id``, ``query``, ``expected_answer_type``, and
    ``depends_on`` keys.
    """
    examples_block = "\n".join(
        f"Query: {eq}\nDecomposition: {json.dumps({'steps': es})}"
        for eq, es in _DECOMPOSITION_EXAMPLES
    )

    context_block = ""
    if repo_context:
        modules = repo_context.get("modules")
        if modules:
            context_block += (
                "Available modules: " + ", ".join(str(m) for m in modules) + "\n"
            )
        symbols = repo_context.get("symbols")
        if symbols:
            context_block += "Key symbols: " + ", ".join(str(s) for s in symbols) + "\n"
        if context_block:
            context_block = "\nRepository context:\n" + context_block

    return (
        "You are a query decomposer for a code intelligence system.\n"
        "Break the user's complex query into 2-5 ordered sub-queries that "
        "each retrieve one piece of the answer.\n"
        "\n"
        "Rules:\n"
        "- Each sub-query should be answerable by a single retrieval step.\n"
        "- Later steps can depend on earlier steps via the depends_on field.\n"
        "- expected_answer_type must be one of: code, explanation, list.\n"
        "- If the query does NOT need decomposition, return a single step "
        "with the original query.\n"
        "\n"
        "Examples:\n"
        f"{examples_block}\n"
        f"{context_block}"
        "\n"
        "Now decompose this query:\n"
        f"Query: {query}\n"
        "\n"
        "Respond with ONLY a JSON object in this exact format "
        "(no markdown, no explanation):\n"
        '{"steps": [{"id": "step-1", "query": "<sub-query text>", '
        '"expected_answer_type": "<code|explanation|list>", '
        '"depends_on": ["<prior step IDs>"]}, ...]}'
    )


# ---------------------------------------------------------------------------
# Decomposition response parsing (pure, unit-testable)
# ---------------------------------------------------------------------------

_MAX_SUB_QUERIES = 5
"""Upper bound from the Issue 21 spec: 'decomposes into 2-5 sub-queries'."""


def _single_step_plan(
    query: str,
    *,
    source: Literal["llm", "fallback"] = "fallback",
    raw_response: str = "",
    metadata: dict[str, Any] | None = None,
) -> DecompositionPlan:
    """Build a single-step plan that wraps *query* unchanged.

    This is the safe fallback: decomposition is skipped and the original
    query is passed through as a single retrieval step.
    """
    return DecompositionPlan(
        steps=(
            SubQuery(
                id="step-1",
                query=query,
                expected_answer_type="explanation",
                depends_on=(),
            ),
        ),
        original_query=query,
        source=source,
        raw_response=raw_response,
        metadata=metadata or {},
    )


def parse_decomposition_response(raw: str, original_query: str) -> DecompositionPlan:
    """Parse the raw LLM response text into a :class:`DecompositionPlan`.

    The LLM is prompted to return strict JSON::

        {"steps": [{"id": "step-1", "query": "...", ...}, ...]}

    This parser is tolerant: it extracts the first valid JSON object from
    the response (so accidental markdown fences or leading prose do not
    break parsing), validates the ``steps`` array, and clamps the step
    count to :data:`_MAX_SUB_QUERIES`.  Unlike :func:`parse_llm_response`
    the JSON extraction uses :meth:`json.JSONDecoder.raw_decode` instead
    of a regex because the response contains nested objects.

    Args:
        raw: The raw text returned by the LLM.
        original_query: The original query, used to build the single-step
            fallback when parsing fails.

    Returns:
        A :class:`DecompositionPlan` with ``source="llm"``.  If the
        response cannot be parsed at all, a single-step fallback plan is
        returned with the parse error recorded in
        ``metadata["parse_error"]``.
    """
    if not raw or not raw.strip():
        return _single_step_plan(
            original_query,
            raw_response=raw or "",
            metadata={"parse_error": "empty response"},
        )

    # Extract the first valid JSON object.  We scan forward from each '{'
    # and try to parse, rather than using _JSON_OBJECT_RE, because the
    # decomposition response contains nested objects that a flat regex
    # cannot match.
    payload: dict[str, Any] | None = None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(raw, i)
                if isinstance(obj, dict):
                    payload = obj
                    break
            except json.JSONDecodeError:
                continue

    if payload is None:
        return _single_step_plan(
            original_query,
            raw_response=raw,
            metadata={"parse_error": "no JSON object found"},
        )

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return _single_step_plan(
            original_query,
            raw_response=raw,
            metadata={
                "parse_error": "missing or empty 'steps' array",
                "raw_payload": payload,
            },
        )

    steps: list[SubQuery] = []
    for i, step_data in enumerate(raw_steps[:_MAX_SUB_QUERIES]):
        if not isinstance(step_data, dict):
            return _single_step_plan(
                original_query,
                raw_response=raw,
                metadata={"parse_error": f"step {i} is not an object"},
            )

        step_id = str(step_data.get("id", f"step-{i + 1}"))

        query_text = step_data.get("query") or step_data.get("text")
        if not query_text or not str(query_text).strip():
            return _single_step_plan(
                original_query,
                raw_response=raw,
                metadata={
                    "parse_error": f"step {step_id!r} has no query text",
                },
            )

        answer_type = (
            str(step_data.get("expected_answer_type", "explanation")).strip().lower()
        )
        if answer_type not in _VALID_ANSWER_TYPES:
            answer_type = "explanation"

        raw_deps = step_data.get("depends_on") or step_data.get("context_from") or []
        if not isinstance(raw_deps, list):
            raw_deps = []
        depends_on = tuple(str(d) for d in raw_deps)

        steps.append(
            SubQuery(
                id=step_id,
                query=str(query_text).strip(),
                expected_answer_type=answer_type,  # type: ignore[arg-type]
                depends_on=depends_on,
            )
        )

    return DecompositionPlan(
        steps=tuple(steps),
        original_query=original_query,
        source="llm",
        raw_response=raw,
        metadata={"raw_payload": payload},
    )


# ---------------------------------------------------------------------------
# QueryDecomposer
# ---------------------------------------------------------------------------


class _DecomposerState(TypedDict):
    """Internal state for the decomposer LangGraph state machine."""

    query: str
    repo_context: dict[str, Any]
    prompt: str
    raw_response: str


class QueryDecomposer:
    """Decomposes a multi-hop query into ordered sub-queries.

    Uses a LangGraph state machine to call the LLM with a decomposition
    prompt and parse the response into a :class:`DecompositionPlan`.  When
    the LLM is disabled, no API key is configured, the LLM call fails, or
    the response cannot be parsed, a single-step plan wrapping the original
    query is returned.

    Args:
        llm: A pre-built callable ``(prompt: str) -> str`` that stands in
            for the langchain LLM client.  Passing a callable is the
            supported test seam -- it makes the decomposer network-free.
            When ``None`` (default) a real langchain LLM is constructed
            lazily on the first :meth:`decompose` call using
            ``settings.llm_provider`` and the configured API key.
        use_llm: When ``True`` (default) the LLM path is used; when
            ``False`` a single-step plan is returned immediately (no LLM
            is ever loaded).
    """

    def __init__(
        self,
        llm: LLMCallable | None = None,
        *,
        use_llm: bool = True,
    ) -> None:
        self._resolved_llm: LLMCallable | None = llm
        self._loaded = False
        self.use_llm = use_llm

    # ------------------------------------------------------------------
    # Lazy LLM loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> LLMCallable | None:
        """Resolve the LLM callable, constructing a real client if needed.

        Called automatically before the first LLM-based decomposition.  A
        pre-injected ``_resolved_llm`` is respected, making tests
        network-free.  Returns ``None`` (and logs a warning) when the LLM
        is enabled but no API key is configured.
        """
        if self._loaded:
            return self._resolved_llm

        if self._resolved_llm is None:
            api_key = settings.active_llm_api_key
            if _is_unset_secret(api_key):
                logger.warning(
                    "QueryDecomposer: LLM is enabled but no API key is "
                    "configured for provider '%s'; returning single-step "
                    "plan.",
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
        ``(prompt) -> str`` callable so the rest of the decomposer is
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

    def decompose(
        self, query: str, repo_context: dict[str, Any] | None = None
    ) -> DecompositionPlan:
        """Decompose *query* into ordered sub-queries.

        Args:
            query: The natural-language query to decompose.
            repo_context: Optional dict with repository metadata (e.g.
                ``{"modules": ["api", "db"], "symbols": ["User", "auth"]}``)
                used to inform decomposition decisions.

        Returns:
            A :class:`DecompositionPlan` with one or more :class:`SubQuery`
            steps.  A single-step plan means decomposition was not needed
            or the LLM was unavailable.

        Raises:
            ValueError: If *query* is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if repo_context is None:
            repo_context = {}

        return self._decompose_impl(query, repo_context)

    def _decompose_impl(
        self, query: str, repo_context: dict[str, Any]
    ) -> DecompositionPlan:
        """Run the LangGraph decomposition pipeline."""
        if not self.use_llm:
            return _single_step_plan(query)

        llm = self._ensure_loaded()
        if llm is None:
            return _single_step_plan(query)

        try:
            raw_response = self._run_graph(llm, query, repo_context)
        except Exception as exc:
            logger.warning(
                "QueryDecomposer: LLM call failed (%s); returning " "single-step plan.",
                exc,
            )
            return _single_step_plan(query)

        plan = parse_decomposition_response(raw_response, query)
        if plan.metadata.get("parse_error"):
            logger.warning(
                "QueryDecomposer: could not parse LLM response (%s); "
                "returning single-step plan.",
                plan.metadata["parse_error"],
            )
            return _single_step_plan(
                query,
                raw_response=raw_response,
                metadata=plan.metadata,
            )

        return plan

    def _run_graph(
        self,
        llm: LLMCallable,
        query: str,
        repo_context: dict[str, Any],
    ) -> str:
        """Build and execute the LangGraph state machine.

        The graph has two nodes:

        * **build_prompt** -- constructs the decomposition prompt from the
          query and repository context.
        * **call_llm** -- sends the prompt to the LLM and returns the raw
          response text.

        Returns the raw LLM response text for detailed parsing by
        :func:`parse_decomposition_response`.
        """
        from langgraph.graph import END, START, StateGraph

        def build_prompt_node(state: _DecomposerState) -> dict[str, str]:
            prompt = _build_decomposition_prompt(state["query"], state["repo_context"])
            return {"prompt": prompt}

        def call_llm_node(state: _DecomposerState) -> dict[str, str]:
            return {"raw_response": llm(state["prompt"])}

        graph = StateGraph(_DecomposerState)
        graph.add_node("build_prompt", build_prompt_node)
        graph.add_node("call_llm", call_llm_node)
        graph.add_edge(START, "build_prompt")
        graph.add_edge("build_prompt", "call_llm")
        graph.add_edge("call_llm", END)
        compiled = graph.compile()

        result = compiled.invoke(
            {
                "query": query,
                "repo_context": repo_context,
                "prompt": "",
                "raw_response": "",
            }
        )
        return result["raw_response"]

    def __repr__(self) -> str:
        return f"QueryDecomposer(use_llm={self.use_llm}, " f"loaded={self._loaded})"


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
