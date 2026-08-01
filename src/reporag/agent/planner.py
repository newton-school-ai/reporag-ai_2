"""Agentic query planner -- query classifier (Issue 20).

Classifies a natural-language question about a codebase into one of three
strategies so the downstream pipeline can pick the cheapest retrieval path
that will still answer it:

* ``simple-lookup``  -- a single direct retrieval answers it. Examples:
  "where is `authenticate` defined?", "show me the `User` model". These go
  straight to BM25 / graph lookup -- no decomposition, no multi-hop
  orchestration, minimum latency and cost.
* ``multi-hop``      -- the answer requires chaining two or more retrieval
  steps. Examples: "how does a request go from the API endpoint to the
  database?", "what calls `authenticate_user` and what does it call next?".
  These need the decomposer (Issue 21) to break them into ordered
  sub-queries.
* ``exploratory``    -- broad, open-ended questions that need wide
  retrieval across many files. Examples: "explain the architecture", "give
  me an overview of the codebase". These skip decomposition and instead
  fan out a broad hybrid retrieval.

Classifying first saves latency and cost: a simple lookup never pays the
decomposition tax, and an exploratory query never wastes hops on narrow
sub-queries.

Design
------
* **Dual strategy** -- an LLM-based classifier (few-shot prompted) is the
  primary path; a deterministic, network-free rule-based classifier is
  the fallback. The rule-based path also runs whenever ``use_llm=False``
  or no API key is configured, and it's what a call falls back to if the
  LLM request itself raises (timeout, rate limit, auth failure) or
  returns something unparseable -- so a flaky LLM call degrades to a
  worse-but-working classification rather than propagating an exception
  up through the whole pipeline.
* **Generalizes past its own examples** -- the rule-based classifier's
  signal patterns are validated in ``tests/unit/test_planner.py`` against
  queries that were *not* used to design the patterns (see
  ``test_rule_based_generalizes_to_unseen_phrasings``), specifically to
  guard against a classifier that only works on the phrasing it was
  written to expect.
* **Lazy LLM loading** -- the ``langchain`` LLM client (already a project
  dependency -- see ``langchain-openai``/``langchain-anthropic`` in
  ``requirements.txt``) is constructed on the first :meth:`classify`
  call, not at construction time. A pre-injected ``llm`` callable is
  respected, making tests network-free the same way
  :class:`~reporag.retrieval.reranker.CrossEncoderReranker` injects a
  fake cross-encoder.
* **Confidence-gated fallback** -- if the final confidence is below
  ``settings.query_classifier_confidence_threshold`` the result is
  overridden to ``multi-hop`` (the safest default: decomposition is
  correct for both genuinely multi-hop queries and ambiguous ones, while
  a wrongly-chosen ``simple-lookup`` would skip hops the query actually
  needed).

This module has no functional dependency on Issue 19's fusion/reranker
code -- it only classifies the raw query string. The
:class:`~reporag.retrieval.vector_search.RetrievalResult`-consuming
pipeline those modules power is a downstream *consumer* of this
classification, not something this module calls into.

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
from typing import Any, Literal

from reporag.config import _is_unset, settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

QueryType = Literal["simple-lookup", "multi-hop", "exploratory"]
"""The three classification categories from Issue 20."""

QUERY_TYPES: tuple[QueryType, ...] = ("simple-lookup", "multi-hop", "exploratory")

_VALID_QUERY_TYPES: frozenset[str] = frozenset(QUERY_TYPES)


@dataclass(frozen=True)
class ClassificationResult:
    """The outcome of classifying a single query.

    Attributes:
        query_type: One of ``simple-lookup``, ``multi-hop``, or
            ``exploratory``.
        confidence: Classifier confidence in ``[0.0, 1.0]``. When the
            confidence-threshold fallback fires this is the *original*
            confidence (so callers can see *why* the fallback triggered),
            not a value overwritten by the fallback.
        fallback_applied: ``True`` when the low-confidence fallback
            overrode the original classification to ``multi-hop``.
        source: ``"llm"`` when the LLM produced the classification,
            ``"rules"`` when the rule-based fallback was used -- because
            the LLM was disabled, no API key was configured, the LLM call
            raised, or the LLM response could not be parsed.
        raw_response: The raw LLM response text (``""`` for the
            rule-based path). Kept for debugging, not programmatic use.
        metadata: Free-form extras -- rule vote scores for the rule-based
            path, or the parsed JSON payload for the LLM path.
    """

    query_type: QueryType
    confidence: float
    fallback_applied: bool = False
    source: Literal["llm", "rules"] = "rules"
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Few-shot prompt
# ---------------------------------------------------------------------------

# Few-shot examples for the LLM prompt. These intentionally overlap only
# partially with the rule-based classifier's own validation queries (see
# test_planner.py) -- the two paths are meant to be assessed independently,
# not tuned to the same fixed set.
_FEW_SHOT_EXAMPLES: tuple[tuple[str, QueryType], ...] = (
    ("Where is the `authenticate` function defined?", "simple-lookup"),
    ("Show me the `User` model class.", "simple-lookup"),
    ("Which file contains `DatabaseConfig`?", "simple-lookup"),
    ("What is the return type of `embed_batch`?", "simple-lookup"),
    ("How does a request travel from the API endpoint to the database?", "multi-hop"),
    (
        "What functions call `authenticate_user`, and what do those " "callers do?",
        "multi-hop",
    ),
    ("Trace the path from the login route to session token creation.", "multi-hop"),
    ("Explain the overall architecture of this codebase.", "exploratory"),
    ("What are the main components and how are they structured?", "exploratory"),
    ("Give me an overview of how retrieval works end to end.", "exploratory"),
)


def _build_classification_prompt(query: str) -> str:
    """Build the few-shot classification prompt for *query*.

    Asks the LLM to return **strict JSON** with ``query_type`` and
    ``confidence`` keys so :func:`parse_llm_response` can parse it without
    a second round trip.
    """
    examples_block = "\n".join(
        f'Query: "{example}"\n'
        f'Response: {{"query_type": "{label}", "confidence": 0.9}}'
        for example, label in _FEW_SHOT_EXAMPLES
    )
    return (
        "Classify the following code-repository search query into "
        "exactly one of these categories:\n\n"
        "- simple-lookup: a single, direct fact -- one function, one "
        "variable, one file. Answerable from a single retrieved chunk.\n"
        "- multi-hop: requires following relationships between multiple "
        "symbols (callers, callees, imports, inheritance) or combining "
        "several pieces of code to answer.\n"
        "- exploratory: broad, open-ended questions about how a system "
        "works, architecture, or design, with no single obvious answer "
        "location.\n\n"
        "Respond with ONLY a JSON object of the form "
        '{"query_type": "<one of the three categories>", '
        '"confidence": <float between 0.0 and 1.0>}. '
        "No other text, no markdown formatting.\n\n"
        "Examples:\n"
        f"{examples_block}\n\n"
        f'Query: "{query}"\n'
        "Response:"
    )


# ---------------------------------------------------------------------------
# Rule-based classifier (pure, network-free, unit-testable)
# ---------------------------------------------------------------------------

# Weighted regex signal patterns. Each pattern votes for a category; the
# category with the highest total weight wins. Patterns are written broadly
# (many phrasing variants per concept) specifically so the classifier isn't
# brittle against rewordings of the same underlying question -- see
# test_rule_based_generalizes_to_unseen_phrasings in test_planner.py, which
# validates this against queries not used while writing these patterns.

_SIMPLE_LOOKUP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhere\s+(is|are|can\s+i\s+find)\b", re.IGNORECASE),
    re.compile(
        r"\bfind\s+(the\s+)?(file|class|function|method|def|module)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(show|display|open|point\s+me\s+to)\b", re.IGNORECASE),
    re.compile(r"\bdefinition\s+of\b", re.IGNORECASE),
    re.compile(r"\b(is|are)\s+.{0,30}\bdefined\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(line|file|module)\b", re.IGNORECASE),
    re.compile(r"\blocate[ds]?\b", re.IGNORECASE),
    re.compile(r"\bdeclare[ds]?\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+(file|module)\b", re.IGNORECASE),
    re.compile(
        r"\b(what|which)\s+(is|are)\s+the\s+(return\s+type|signature|parameters?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bcontains?\s+(the\s+)?(class|function|def)\b", re.IGNORECASE),
    re.compile(r"\b(has|handles?)\s+the\s+.{0,30}\blogic\b", re.IGNORECASE),
)

_MULTI_HOP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhow\s+does\b.{0,60}\b(work|flow|travel|propagate)\b", re.IGNORECASE),
    re.compile(r"\bend[\s-]to[\s-]end\b", re.IGNORECASE),
    re.compile(r"\btrace\b", re.IGNORECASE),
    re.compile(r"\bflow\b", re.IGNORECASE),
    re.compile(r"\bpath\s+from\b", re.IGNORECASE),
    re.compile(r"\b(what|who)\s+calls\b", re.IGNORECASE),
    re.compile(r"\bcallers?\s+of\b", re.IGNORECASE),
    re.compile(r"\bchain\s+of\b", re.IGNORECASE),
    re.compile(r"\bstep[\s-]by[\s-]step\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+.+\s+to\s+the\b", re.IGNORECASE),
    re.compile(r"\btravels?\b", re.IGNORECASE),
    re.compile(r"\bpropagates?\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+.{0,20}\s+depend[s]?\s+on\b", re.IGNORECASE),
)

_EXPLORATORY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bexplain\s+(the\s+)?(overall\s+)?architecture\b", re.IGNORECASE),
    re.compile(r"\boverview\b", re.IGNORECASE),
    re.compile(r"\barchitecture\b", re.IGNORECASE),
    re.compile(r"\bhigh[\s-]?level\b", re.IGNORECASE),
    re.compile(r"\bmain\s+components?\b", re.IGNORECASE),
    re.compile(r"\b(structured|organized)\b", re.IGNORECASE),
    re.compile(r"\bsummarize\b", re.IGNORECASE),
    re.compile(r"\bdesign\s+(patterns?|decisions?)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+are\s+the\s+main\b", re.IGNORECASE),
    re.compile(r"\bwalk\s+me\s+through\s+the\s+(whole|entire)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+is\s+.{0,30}\borganized\b", re.IGNORECASE),
    re.compile(r"\b(big\s+picture|bird'?s[\s-]eye)\b", re.IGNORECASE),
    re.compile(r"\bput\s+together\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(this|the)\s+.{0,20}\bworks?\s+in\s+general\b", re.IGNORECASE),
)


def _score_patterns(query: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    """Return the number of *patterns* that match *query*.

    Each pattern contributes at most one vote (not one per match) so a
    query that repeats the same signal word doesn't drown out other
    signals.
    """
    return sum(1 for pattern in patterns if pattern.search(query))


def rule_based_classify(query: str) -> ClassificationResult:
    """Classify *query* using weighted regex signal patterns.

    The deterministic, network-free fallback. Scores each category by
    counting how many of its signal patterns match the query; the
    category with the highest score wins. Confidence is the winner's
    share of total votes (``winner / total``).

    When every category ties at zero votes, the query is classified as
    ``multi-hop`` (the safest default) with confidence ``0.0``, so the
    confidence-threshold fallback in
    :meth:`QueryClassifier.classify` fires naturally on a totally
    unrecognized query rather than guessing.

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
        return ClassificationResult(
            query_type="multi-hop",
            confidence=0.0,
            source="rules",
            metadata={"scores": scores},
        )

    winner = max(scores, key=lambda k: scores[k])
    top_score = scores[winner]
    tied = [k for k, v in scores.items() if v == top_score]
    if len(tied) > 1:
        # Ties resolve toward the safer (more capable) category:
        # multi-hop > exploratory > simple-lookup, matching the issue's
        # own "fall back to multi-hop" guidance for ambiguous cases.
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

# Matches a ```json ... ``` or ``` ... ``` fenced block, or a bare
# {...} object, so both a raw JSON reply and a markdown-fenced one parse.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_response(raw: str) -> ClassificationResult:
    """Parse the raw LLM response text into a :class:`ClassificationResult`.

    The LLM is prompted to return strict JSON::

        {"query_type": "simple-lookup", "confidence": 0.95}

    Tolerant of a markdown code fence around the JSON, or (falling back)
    the first bare ``{...}`` block anywhere in the text. Rejects an
    unknown ``query_type`` and clamps ``confidence`` to ``[0.0, 1.0]``.

    Args:
        raw: The raw text returned by the LLM.

    Returns:
        A :class:`ClassificationResult` with ``source="llm"``. If the
        response can't be parsed at all, returns a zero-confidence
        ``multi-hop`` result (so the caller's threshold fallback fires
        naturally) with the failure reason in
        ``metadata["parse_error"]``.
    """
    if not raw or not raw.strip():
        return ClassificationResult(
            query_type="multi-hop",
            confidence=0.0,
            source="llm",
            raw_response=raw,
            metadata={"parse_error": "empty response"},
        )

    fenced = _JSON_FENCE_RE.search(raw)
    candidate = fenced.group(1) if fenced else raw

    match = _JSON_OBJECT_RE.search(candidate)
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

    if not isinstance(payload, dict):
        return ClassificationResult(
            query_type="multi-hop",
            confidence=0.0,
            source="llm",
            raw_response=raw,
            metadata={"parse_error": "JSON payload was not an object"},
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

# A callable that takes a prompt string and returns the LLM's response
# text. This is the test seam: tests inject a deterministic fake instead
# of a real langchain LLM, keeping every test network-free.
LLMCallable = Callable[[str], str]


class QueryClassifier:
    """Classifies a query into simple-lookup / multi-hop / exploratory.

    The primary path is an LLM-based classifier with few-shot examples; a
    deterministic rule-based classifier (:func:`rule_based_classify`) is
    the fallback whenever the LLM is disabled, no API key is configured,
    the LLM call itself raises, or the LLM's response can't be parsed.

    Args:
        llm: A pre-built callable ``(prompt: str) -> str`` standing in for
            the langchain LLM client. Passing a callable is the supported
            test seam -- it keeps the classifier network-free without
            needing an API key, the same way
            :class:`~reporag.retrieval.reranker.CrossEncoderReranker`
            accepts an injected fake cross-encoder. When ``None``
            (default), a real langchain LLM is constructed lazily on the
            first :meth:`classify` call, using ``settings.llm_provider``
            and the configured API key.
        confidence_threshold: Below this confidence, classification falls
            back to ``multi-hop``. Defaults to
            ``settings.query_classifier_confidence_threshold``. Must be
            in ``[0.0, 1.0]``.
        use_llm: When ``True`` (default), the LLM path is attempted
            first; when ``False``, only the rule-based classifier runs
            and the LLM is never constructed. Defaults to
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
                f"confidence_threshold must be in [0.0, 1.0], got "
                f"{confidence_threshold!r}."
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

        Called before the first LLM-based classification. A pre-injected
        callable is respected and never overwritten. Returns ``None``
        (after logging a warning) when the LLM is enabled but no API key
        is configured -- the caller then falls back to
        :func:`rule_based_classify`.
        """
        if self._loaded:
            return self._resolved_llm
        if self._resolved_llm is None:
            if _is_unset(settings.active_llm_api_key):
                logger.warning(
                    "QueryClassifier: LLM is enabled but no API key is "
                    "configured for provider '%s'; falling back to "
                    "rule-based classification.",
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

        ``langchain``'s ``invoke`` returns a message object whose
        ``content`` attribute holds the text; this wraps it in a plain
        ``(prompt) -> str`` callable so the rest of the classifier stays
        provider-agnostic and trivially fakeable in tests.
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
        """``True`` once the LLM has been resolved (or found unavailable)."""
        return self._loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> ClassificationResult:
        """Classify *query* into simple-lookup, multi-hop, or exploratory.

        Path:

        1. If ``use_llm`` is ``False``, go straight to
           :func:`rule_based_classify`.
        2. Resolve the LLM (lazy load). No API key -> rule-based.
        3. Call the LLM. If the call itself raises (timeout, rate limit,
           auth failure, or any other exception) -> rule-based.
        4. Parse the response. If it can't be parsed -> rule-based.
        5. Apply the confidence threshold: if the winning confidence is
           below :attr:`confidence_threshold`, override ``query_type`` to
           ``multi-hop`` and set ``fallback_applied=True`` (the original
           confidence is preserved so callers can see why the fallback
           fired).

        Args:
            query: The natural-language query to classify.

        Returns:
            A :class:`ClassificationResult`.

        Raises:
            ValueError: If *query* is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty or whitespace-only.")

        result = self._classify_without_threshold(query)

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
                fallback_applied=True,
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
            return rule_based_classify(query)

        prompt = _build_classification_prompt(query)
        try:
            raw_response = llm(prompt)
        except Exception as exc:  # noqa: BLE001 - any LLM-call failure degrades safely
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
# Issue 21 -- QueryDecomposer
# ---------------------------------------------------------------------------

# TODO: Implement in Issue 21
#
# QueryDecomposer (Issue 21):
# - LangGraph state machine for decomposition
# - Input: complex query + repo context (modules, key symbols)
# - Output: ordered list of SubQuery objects with dependency edges
# - Each SubQuery: text, expected_answer_type, context_from (prior IDs)
# - Handles queries that do not need decomposition (single step)
