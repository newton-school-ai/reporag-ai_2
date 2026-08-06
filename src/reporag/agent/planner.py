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

Decomposer (Issue 21)
----------------------
:class:`QueryDecomposer` consumes the classifier's output above.  A
``multi-hop`` query is broken into 2-5 ordered sub-queries with dependency
edges (``depends_on``) using an LLM, orchestrated by a small LangGraph state
machine::

    classify --(simple-lookup / exploratory)--> passthrough --> END
             --(multi-hop)--------------------> decompose --> validate --(ok)--> END
                                                     ^             |
                                                     '---(retry)---+--(exhausted)--> rule_fallback --> END

* ``classify``      -- reuses :class:`QueryClassifier` to decide whether the
  query needs decomposition at all (mirrors the classifier's own dual
  LLM/rules strategy, so this also works fully offline).
* ``passthrough``   -- ``simple-lookup``/``exploratory`` queries are wrapped
  in a single-step plan (``needs_decomposition=False``) so every downstream
  consumer (the Issue 22 router/executor) can always iterate ``plan.steps``
  uniformly, whether or not decomposition happened.
* ``decompose``     -- prompts the LLM with few-shot examples plus the
  caller-supplied ``repo_context`` (module names / key symbols) so the
  sub-queries are grounded in the actual repo rather than generic.
* ``validate``      -- structurally checks the parsed plan (2-5 steps,
  unique ids, only-backward dependency edges, a valid
  ``expected_answer_type``) and retries the LLM once on failure.
* ``rule_fallback``  -- a deterministic, network-free decomposer (same
  contract as :func:`rule_based_classify`) used when the LLM is disabled, no
  API key is configured, or the LLM plan still fails validation after
  retries -- so :meth:`QueryDecomposer.decompose` never raises for a
  well-formed query.
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

        Delegates to the module-level :func:`_build_langchain_llm`, which
        is shared with :class:`QueryDecomposer` (see the "Decomposer"
        section below) so there is exactly one place that knows how to
        build a provider-agnostic ``(prompt) -> str`` callable.
        """
        return _build_langchain_llm()

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


# ---------------------------------------------------------------------------
# Shared LLM construction (used by both QueryClassifier and QueryDecomposer)
# ---------------------------------------------------------------------------


def _build_langchain_llm() -> LLMCallable:
    """Construct a provider-agnostic ``(prompt: str) -> str`` langchain client.

    Shared by :class:`QueryClassifier` and :class:`QueryDecomposer` so the
    two LLM-backed components in this module stay in lockstep -- one place
    to add a provider, one place to fix a bug.
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


# ===========================================================================
# Query decomposer (Issue 21)
# ===========================================================================
#
# See the module docstring for the full LangGraph state-machine diagram.
# Layout mirrors the classifier above: public types, few-shot prompt, pure
# rule-based fallback, pure LLM-response parser, then the stateful
# orchestrator class -- each layer independently unit-testable.

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

ExpectedAnswerType = Literal["code", "explanation", "list"]
"""What kind of result a sub-query is expected to surface."""

_VALID_ANSWER_TYPES: frozenset[str] = frozenset({"code", "explanation", "list"})

# Tolerant synonym map so a slightly-off LLM label (e.g. "snippet",
# "summary") still lands on a valid category instead of failing validation
# outright -- the same "be tolerant of LLM phrasing" philosophy as
# :func:`parse_llm_response`. Anything unrecognized defaults to
# "explanation" (the safest label: it never implies a structural guarantee
# like "code" or "list" that downstream rendering might rely on).
_ANSWER_TYPE_SYNONYMS: dict[str, ExpectedAnswerType] = {
    "code": "code",
    "snippet": "code",
    "code_snippet": "code",
    "location": "code",
    "definition": "code",
    "explanation": "explanation",
    "description": "explanation",
    "summary": "explanation",
    "narrative": "explanation",
    "list": "list",
    "items": "list",
    "enumeration": "list",
}


def _coerce_answer_type(raw_type: Any) -> ExpectedAnswerType:
    """Map a raw LLM-provided label onto one of the three valid categories."""
    key = str(raw_type).strip().lower()
    return _ANSWER_TYPE_SYNONYMS.get(key, "explanation")


@dataclass(frozen=True)
class DecompositionStep:
    """A single ordered sub-query within a :class:`DecompositionPlan`.

    Attributes:
        id: A short, unique identifier within the plan (e.g. ``"step-1"``).
        query: The sub-query text to execute.
        expected_answer_type: What kind of result this sub-query should
            surface -- one of ``code``, ``explanation``, or ``list``.
        depends_on: Ids of *earlier* steps in the same plan whose results
            this sub-query needs as context. Empty for steps that can run
            immediately. These are the dependency edges from the issue's
            acceptance criteria.

    ``text`` and ``context_from`` are read-only aliases for ``query`` and
    ``depends_on`` respectively, matching the field names used in the issue
    description (``text``, ``expected_answer_type``, ``context_from``) so
    callers can use either vocabulary.
    """

    id: str
    query: str
    expected_answer_type: ExpectedAnswerType
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Dataclasses do not enforce Literal types at runtime, and this
        # class can be constructed directly (not just via the parser
        # above, which already coerces), so validate defensively here too.
        if not self.id or not self.id.strip():
            raise ValueError("DecompositionStep.id must be a non-empty string.")
        if not self.query or not self.query.strip():
            raise ValueError("DecompositionStep.query must be a non-empty string.")
        if self.expected_answer_type not in _VALID_ANSWER_TYPES:
            raise ValueError(
                f"expected_answer_type must be one of {sorted(_VALID_ANSWER_TYPES)}, "
                f"got {self.expected_answer_type!r}."
            )

    @property
    def text(self) -> str:
        return self.query

    @property
    def context_from(self) -> tuple[str, ...]:
        return self.depends_on


@dataclass(frozen=True)
class DecompositionPlan:
    """The outcome of decomposing a single query.

    Attributes:
        original_query: The query that was decomposed.
        steps: The ordered sub-queries. Always at least one step -- a query
            that does not need decomposition still gets a single-step,
            passthrough plan so callers can always iterate ``plan.steps``
            uniformly.
        needs_decomposition: ``False`` for the single-step passthrough case
            (``simple-lookup`` / ``exploratory`` queries), ``True`` when the
            query actually went through multi-step decomposition.
        classification: The :class:`ClassificationResult` from the Issue 20
            classifier that decided whether decomposition was needed.
        source: ``"llm"`` when the LLM produced the plan, ``"rules"`` when
            the deterministic fallback was used, ``"passthrough"`` when no
            decomposition was needed at all.
        raw_response: The raw LLM response text for the attempt that
            ultimately produced the plan (``""`` for ``rules`` /
            ``passthrough``).
        fell_back: ``True`` when the rule-based fallback fired (LLM
            disabled, unavailable, or its output failed validation after
            retries).
        metadata: Free-form extras -- the normalized repo context and the
            number of LLM attempts made.
    """

    original_query: str
    steps: tuple[DecompositionStep, ...]
    needs_decomposition: bool
    classification: ClassificationResult
    source: Literal["llm", "rules", "passthrough"]
    raw_response: str = ""
    fell_back: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Repo context normalization
# ---------------------------------------------------------------------------


def _normalize_repo_context(
    repo_context: dict[str, Any] | None,
) -> dict[str, list[str]]:
    """Normalize the caller-supplied repo context to ``{"modules": [...], "symbols": [...]}``.

    Accepts ``modules`` and either ``symbols`` or ``key_symbols`` (both
    appear across the issue text and README) so callers do not need to
    remember the exact key name. Missing or malformed input degrades to
    empty lists rather than raising -- repo context is an optional
    grounding aid, not a required input.
    """
    if not repo_context:
        return {"modules": [], "symbols": []}
    modules = repo_context.get("modules") or []
    symbols = repo_context.get("symbols") or repo_context.get("key_symbols") or []
    return {
        "modules": [str(m) for m in modules],
        "symbols": [str(s) for s in symbols],
    }


def _matching_context_items(text: str, repo_context: dict[str, list[str]]) -> list[str]:
    """Return module/symbol names from *repo_context* that appear (case-insensitively) in *text*.

    Checks both ``modules`` and ``symbols`` -- a query mentioning a known
    symbol (e.g. ``authenticate_user``) is just as groundable as one
    mentioning a known module (e.g. ``auth``), so both are considered.
    Order is preserved and duplicates are dropped.
    """
    lowered = text.lower()
    candidates = list(repo_context.get("modules", [])) + list(
        repo_context.get("symbols", [])
    )
    matches: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate.lower() in lowered and candidate not in seen:
            matches.append(candidate)
            seen.add(candidate)
    return matches


def _grounding_note(text: str, repo_context: dict[str, list[str]]) -> str:
    """Build a short ``" (relevant context: ...)"`` suffix when *text* mentions a known module or symbol."""
    matches = _matching_context_items(text, repo_context)
    if not matches:
        return ""
    return f" (relevant context: {', '.join(matches)})"


# ---------------------------------------------------------------------------
# Few-shot prompt
# ---------------------------------------------------------------------------

# Four worked examples spanning the shapes of multi-hop query this
# decomposer needs to handle well: an explicit "from A to B" flow, a named
# subsystem flow grounded in repo context, a caller/callee trace, and a
# "trace the path" phrasing. Each shows repo context actually changing the
# sub-query wording (Issue 21's "uses repo context for informed
# decomposition" criterion), not just being echoed back.
_DECOMPOSITION_FEW_SHOT: tuple[tuple[str, str, str], ...] = (
    (
        "How does a request go from the API endpoint to the database?",
        '{"modules": ["api", "routes", "db", "models"]}',
        json.dumps(
            {
                "steps": [
                    {
                        "id": "step-1",
                        "query": "Locate the API route handler that receives the incoming request",
                        "expected_answer_type": "code",
                        "depends_on": [],
                    },
                    {
                        "id": "step-2",
                        "query": "Locate the database access layer in the models/db module",
                        "expected_answer_type": "code",
                        "depends_on": [],
                    },
                    {
                        "id": "step-3",
                        "query": "Trace how the route handler passes data through to the database layer",
                        "expected_answer_type": "explanation",
                        "depends_on": ["step-1", "step-2"],
                    },
                ]
            }
        ),
    ),
    (
        "How does the auth flow work end-to-end?",
        '{"modules": ["auth", "middleware", "session"]}',
        json.dumps(
            {
                "steps": [
                    {
                        "id": "step-1",
                        "query": "Locate the auth entry point (login route or auth middleware)",
                        "expected_answer_type": "code",
                        "depends_on": [],
                    },
                    {
                        "id": "step-2",
                        "query": "Locate how the session module validates credentials and issues a session",
                        "expected_answer_type": "code",
                        "depends_on": ["step-1"],
                    },
                    {
                        "id": "step-3",
                        "query": "Explain how the auth entry point connects to session creation end-to-end",
                        "expected_answer_type": "explanation",
                        "depends_on": ["step-1", "step-2"],
                    },
                ]
            }
        ),
    ),
    (
        "What calls the authenticate_user function and what does it call next?",
        "{}",
        json.dumps(
            {
                "steps": [
                    {
                        "id": "step-1",
                        "query": "Find all callers of authenticate_user",
                        "expected_answer_type": "list",
                        "depends_on": [],
                    },
                    {
                        "id": "step-2",
                        "query": "Find what authenticate_user calls internally",
                        "expected_answer_type": "list",
                        "depends_on": [],
                    },
                    {
                        "id": "step-3",
                        "query": "Explain the call chain from the callers, through authenticate_user, to its callees",
                        "expected_answer_type": "explanation",
                        "depends_on": ["step-1", "step-2"],
                    },
                ]
            }
        ),
    ),
    (
        "Trace the path from the login route to the session token creation.",
        '{"modules": ["routes", "auth", "tokens"]}',
        json.dumps(
            {
                "steps": [
                    {
                        "id": "step-1",
                        "query": "Locate the login route in the routes module",
                        "expected_answer_type": "code",
                        "depends_on": [],
                    },
                    {
                        "id": "step-2",
                        "query": "Locate the token creation logic in the tokens module",
                        "expected_answer_type": "code",
                        "depends_on": [],
                    },
                    {
                        "id": "step-3",
                        "query": "Trace the path from the login route, through auth, to token creation",
                        "expected_answer_type": "explanation",
                        "depends_on": ["step-1", "step-2"],
                    },
                ]
            }
        ),
    ),
)


def _build_decomposition_prompt(query: str, repo_context: dict[str, list[str]]) -> str:
    """Build the few-shot decomposition prompt for *query* and *repo_context*."""
    examples_block = "\n\n".join(
        f"Query: {example_query}\nRepo context: {example_ctx}\nOutput: {example_out}"
        for example_query, example_ctx, example_out in _DECOMPOSITION_FEW_SHOT
    )
    context_str = json.dumps(repo_context)
    return (
        "You are a query decomposer for a code intelligence system. Break "
        "the user's multi-hop query into 2 to 5 ordered sub-queries that, "
        "executed in order, retrieve everything needed to answer it.\n"
        "\n"
        "Rules:\n"
        "- Return between 2 and 5 sub-queries.\n"
        '- Each sub-query needs: id (e.g. "step-1"), query (the '
        'sub-query text), expected_answer_type (one of "code", '
        '"explanation", "list"), and depends_on (a list of ids of '
        "EARLIER sub-queries whose results this one needs as context; "
        "[] if none).\n"
        "- depends_on may only reference ids that appear earlier in the "
        "list -- no forward references, no cycles.\n"
        "- Use the repo context (module names / key symbols) below to make "
        "sub-queries concrete and grounded in this repo, not generic.\n"
        "- If the query does not actually need multiple retrieval steps, "
        "still return at least 2 steps by splitting it into a lookup step "
        "and an explanation step.\n"
        "\n"
        "Examples:\n"
        f"{examples_block}\n"
        "\n"
        "Now decompose this query:\n"
        f"Query: {query}\n"
        f"Repo context: {context_str}\n"
        "\n"
        "Respond with ONLY a JSON object on a single line in this exact "
        "format (no markdown, no explanation):\n"
        '{"steps": [{"id": "step-1", "query": "...", '
        '"expected_answer_type": "code|explanation|list", '
        '"depends_on": []}, ...]}'
    )


# ---------------------------------------------------------------------------
# Rule-based decomposer (pure, network-free, unit-testable)
# ---------------------------------------------------------------------------

# Matches an explicit "from A to B [to C ...]" flow description and
# captures everything after "from " up to the end of the query. The chain
# is split into individual endpoints by ``_TO_SPLIT_RE`` below, so a query
# naming 2, 3, or 4 endpoints (e.g. "from source to destination to sink")
# gets one locate step per endpoint rather than only the first and last.
_FROM_RE = re.compile(r"\bfrom\s+(?:the\s+)?(.+)$", re.IGNORECASE)

# Splits the captured "from ..." remainder on each "to" -- this is what
# turns a 3+ hop chain into separate endpoints instead of one blob.
_TO_SPLIT_RE = re.compile(r"\s+to\s+(?:the\s+)?", re.IGNORECASE)

# Boilerplate stripped from a query to leave just the "subject" -- used to
# phrase generic fallback sub-queries naturally (e.g. "How does the auth
# flow work end-to-end?" -> "the auth flow").
_BOILERPLATE_RE = re.compile(
    r"^(how\s+does|how\s+do|explain|trace|describe|walk\s+me\s+through)\s+"
    r"|\s+(works?(\s+end-to-end)?|end-to-end|step\s+by\s+step)\s*[\?\.]*$",
    re.IGNORECASE,
)


def _extract_subject(query: str) -> str:
    """Strip boilerplate phrasing from *query*, leaving the core subject."""
    subject = query.strip().rstrip("?.")
    subject = _BOILERPLATE_RE.sub("", subject).strip()
    return subject or query.strip()


def _split_from_to_chain(query: str) -> list[str] | None:
    """Split a ``"from A to B [to C ...]"`` query into its ordered endpoints.

    Handles chains of any length, not just a single "A to B" pair: "from
    source to destination to sink" splits into three endpoints
    (``["source", "destination", "sink"]``), each of which gets its own
    locate step in :func:`rule_based_decompose` rather than the extra hop
    being folded into one endpoint's label.

    Args:
        query: The natural-language query to inspect.

    Returns:
        The ordered list of endpoint strings if the query contains a
        ``"from ... to ..."`` chain with at least two endpoints, otherwise
        ``None`` (no "from", or "from" with no "to" at all).
    """
    match = _FROM_RE.search(query)
    if match is None:
        return None
    remainder = match.group(1).rstrip("?.")
    endpoints = [
        segment.strip().rstrip("?.") for segment in _TO_SPLIT_RE.split(remainder)
    ]
    endpoints = [endpoint for endpoint in endpoints if endpoint]
    if len(endpoints) < 2:
        return None
    return endpoints


def rule_based_decompose(
    query: str, repo_context: dict[str, list[str]] | None = None
) -> list[DecompositionStep]:
    """Deterministically decompose *query* without an LLM.

    This is the network-free fallback, used when the LLM is disabled, no
    API key is configured, or an LLM-produced plan fails validation after
    retries. It always returns a valid plan (never fails validation
    itself), so :meth:`QueryDecomposer.decompose` never raises for a
    well-formed, non-empty query.

    Two shapes are handled:

    * An explicit ``"... from A to B [to C ...] ..."`` flow -- one "locate"
      step per endpoint (2 to 4 of them), plus one "trace" step depending
      on all of them. A 2-endpoint chain mirrors the issue's own example
      query almost exactly; a longer chain (e.g. "from source to
      destination to sink") gets a locate step *per hop* rather than
      folding everything past the first "to" into one endpoint label. A
      chain of more than 4 endpoints would need more than 5 steps total,
      so it falls through to the generic shape below instead.
    * Everything else -- a generic "identify entry point", "retrieve
      implementation", "explain end-to-end" template built around the
      query's core subject.

    When *repo_context* module or symbol names appear in the sub-query
    text, a ``" (relevant context: ...)"`` note is appended so the
    rule-based path also honors "uses repo context for informed
    decomposition", not just the LLM path.

    Args:
        query: The natural-language query to decompose.
        repo_context: Normalized repo context (see
            :func:`_normalize_repo_context`); ``None`` is treated as empty.

    Returns:
        A list of :class:`DecompositionStep` objects (3 for the generic
        shape, or ``len(endpoints) + 1`` for a from/to chain) with valid,
        backward-only dependency edges.
    """
    repo_context = repo_context or {"modules": [], "symbols": []}

    endpoints = _split_from_to_chain(query)
    if endpoints is not None and 2 <= len(endpoints) <= 4:
        locate_steps = [
            DecompositionStep(
                id=f"step-{index + 1}",
                query=f"Locate {endpoint}{_grounding_note(endpoint, repo_context)}",
                expected_answer_type="code",
                depends_on=(),
            )
            for index, endpoint in enumerate(endpoints)
        ]
        if len(endpoints) == 2:
            trace_query = f"Trace how {endpoints[0]} connects to {endpoints[1]}"
        else:
            middle = ", ".join(endpoints[1:-1])
            trace_query = (
                f"Trace the path from {endpoints[0]}, through {middle}, "
                f"to {endpoints[-1]}"
            )
        trace_step = DecompositionStep(
            id=f"step-{len(endpoints) + 1}",
            query=trace_query,
            expected_answer_type="explanation",
            depends_on=tuple(step.id for step in locate_steps),
        )
        return [*locate_steps, trace_step]

    # No from/to chain (or one too long to fit in 5 steps) -- fall back to
    # the generic three-step template built around the query's subject.
    subject = _extract_subject(query)
    return [
        DecompositionStep(
            id="step-1",
            query=f"Identify the entry point and key symbols for {subject}",
            expected_answer_type="code",
            depends_on=(),
        ),
        DecompositionStep(
            id="step-2",
            query=(
                f"Retrieve the implementation details for {subject}"
                f"{_grounding_note(subject, repo_context)}"
            ),
            expected_answer_type="code",
            depends_on=("step-1",),
        ),
        DecompositionStep(
            id="step-3",
            query=f"Explain how {subject} flows end-to-end",
            expected_answer_type="explanation",
            depends_on=("step-1", "step-2"),
        ),
    ]


# ---------------------------------------------------------------------------
# LLM response parsing (pure, unit-testable)
# ---------------------------------------------------------------------------


def _extract_json_object(raw: str) -> str | None:
    """Extract the first balanced ``{...}`` block from *raw*.

    Unlike the classifier's single-level regex (``{[^{}]*}``), a
    decomposition response contains a nested ``steps`` array, so this walks
    brace depth to find the *matching* closing brace -- tolerant of
    surrounding markdown fences or prose, same as the classifier's parser.
    """
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(raw)):
        char = raw[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    return None


def parse_decomposition_response(
    raw: str,
) -> tuple[list[DecompositionStep], str | None]:
    """Parse the raw LLM response text into a list of :class:`DecompositionStep`.

    The LLM is prompted to return strict JSON: ``{"steps": [...]}``. This
    parser is tolerant of markdown fences / leading prose (via
    :func:`_extract_json_object`) and of a slightly-off
    ``expected_answer_type`` label (via :func:`_coerce_answer_type`), but
    is strict about structure: a missing ``query``, a non-list
    ``depends_on``, or a non-object step is a parse error, since those
    cannot be safely guessed.

    Args:
        raw: The raw text returned by the LLM.

    Returns:
        A ``(steps, error)`` tuple. On success ``error`` is ``None`` and
        ``steps`` is non-empty (though not yet checked for count /
        dependency validity -- see :func:`validate_steps` for that). On
        failure ``steps`` is ``[]`` and ``error`` describes what went
        wrong, for logging and for the retry/fallback decision.
    """
    if not raw or not raw.strip():
        return [], "empty response"

    json_text = _extract_json_object(raw)
    if json_text is None:
        return [], "no JSON object found"

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return [], f"invalid JSON: {exc}"

    raw_steps = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(raw_steps, list) or not raw_steps:
        return [], "missing or empty 'steps' list"

    steps: list[DecompositionStep] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            return [], f"step {index} is not an object"

        step_id = str(raw_step.get("id", "")).strip() or f"step-{index + 1}"

        step_query = str(raw_step.get("query", raw_step.get("text", ""))).strip()
        if not step_query:
            return [], f"step {index} ({step_id!r}) is missing 'query' text"

        answer_type = _coerce_answer_type(
            raw_step.get("expected_answer_type", "explanation")
        )

        deps_raw = raw_step.get("depends_on", raw_step.get("context_from", []))
        if not isinstance(deps_raw, list):
            return [], f"step {index} ({step_id!r}) has a non-list 'depends_on'"
        depends_on = tuple(str(dep).strip() for dep in deps_raw)

        steps.append(
            DecompositionStep(
                id=step_id,
                query=step_query,
                expected_answer_type=answer_type,
                depends_on=depends_on,
            )
        )

    return steps, None


def validate_steps(steps: list[DecompositionStep]) -> str | None:
    """Structurally validate a parsed decomposition plan.

    Checks (in order, returning the first failure):

    1. Non-empty.
    2. Between 2 and 5 steps (Issue 21's acceptance criterion).
    3. Every step id is non-empty and unique.
    4. Every ``depends_on`` entry references a step id that appears
       *strictly earlier* in the list. Walking ids into a ``seen`` set in
       list order and checking membership before insertion enforces both
       "no forward references" and "no cycles" in one pass -- a step can
       never depend on itself or on anything after it.

    ``expected_answer_type`` is not re-checked here: ``DecompositionStep``
    validates it (against ``code`` / ``explanation`` / ``list``) in its own
    ``__post_init__``, so by the time a step exists in *steps* it is
    already guaranteed valid.

    Args:
        steps: The parsed steps to validate.

    Returns:
        ``None`` if *steps* is a structurally valid plan, otherwise a
        human-readable error string describing the first problem found.
    """
    if not steps:
        return "no steps"
    if not 2 <= len(steps) <= 5:
        return f"expected 2-5 steps, got {len(steps)}"

    seen_ids: set[str] = set()
    for step in steps:
        if not step.id:
            return "a step has an empty id"
        if step.id in seen_ids:
            return f"duplicate step id: {step.id!r}"
        for dep in step.depends_on:
            if dep not in seen_ids:
                return (
                    f"step {step.id!r} has depends_on {dep!r}, which is not "
                    "an earlier step id (forward reference or cycle)"
                )
        seen_ids.add(step.id)

    return None


# ---------------------------------------------------------------------------
# QueryDecomposer
# ---------------------------------------------------------------------------


class _DecomposerState(TypedDict, total=False):
    """LangGraph state threaded through the decomposition state machine."""

    query: str
    repo_context: dict[str, list[str]]
    classification: ClassificationResult
    attempt: int
    steps: list[DecompositionStep]
    source: Literal["llm", "rules", "passthrough"]
    raw_response: str
    fell_back: bool
    error: str | None


class QueryDecomposer:
    """Breaks a multi-hop query into ordered sub-queries with dependency edges.

    Orchestrated by a LangGraph state machine (see the module docstring for
    the diagram):

    1. ``classify`` -- reuse the Issue 20 :class:`QueryClassifier` to decide
       whether the query needs decomposition at all.
    2. ``simple-lookup`` / ``exploratory`` -> ``passthrough``: wrap the
       original query in a single-step plan. ``multi-hop`` -> ``decompose``.
    3. ``decompose`` -- prompt the LLM (few-shot, repo-context-grounded) and
       parse its response into candidate steps.
    4. ``validate`` -- structurally check the candidate plan
       (:func:`validate_steps`). Valid -> done. Invalid due to a parse
       error or failed LLM call -> retry ``decompose`` up to
       ``max_retries`` times. Invalid because the LLM is disabled/
       unavailable, or retries are exhausted -> ``rule_fallback``.
    5. ``rule_fallback`` -- deterministic, network-free decomposition
       (:func:`rule_based_decompose`), which always produces a valid plan.

    This means :meth:`decompose` never raises for a well-formed, non-empty
    query, regardless of whether an LLM is configured, reachable, or
    well-behaved.

    Args:
        llm: A pre-built callable ``(prompt: str) -> str`` -- the same test
            seam as :class:`QueryClassifier`, making tests network-free.
            When ``None`` (default) a real langchain LLM is constructed
            lazily on the first LLM-path call.
        classifier: A pre-built :class:`QueryClassifier` to reuse (e.g. to
            share one instance's LLM connection across callers, or to
            inject a fake in tests). When ``None`` a new one is constructed
            with ``use_llm=use_llm``.
        use_llm: When ``True`` (default) the LLM decomposition path is
            used; when ``False`` only the rule-based decomposer ever runs
            (no LLM is loaded, and the classifier is also constructed with
            ``use_llm=False`` unless a *classifier* was explicitly passed
            in). Defaults to ``settings.query_classifier_use_llm``.
        max_retries: Additional LLM attempts after a parse/validation
            failure before falling back to rules. A configuration or
            availability error (LLM disabled, no API key) skips straight to
            the fallback instead of spending retries on a call that cannot
            succeed. Defaults to ``1`` (two attempts total). Must be
            ``>= 0``.

    Raises:
        ValueError: If *max_retries* is negative.
    """

    def __init__(
        self,
        llm: LLMCallable | None = None,
        *,
        classifier: QueryClassifier | None = None,
        use_llm: bool | None = None,
        max_retries: int = 1,
    ) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries!r}.")

        self._resolved_llm: LLMCallable | None = llm
        self._loaded = False
        self.use_llm = (
            use_llm if use_llm is not None else settings.query_classifier_use_llm
        )
        self.classifier = classifier or QueryClassifier(use_llm=self.use_llm)
        self.max_retries = max_retries
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Lazy LLM loading (mirrors QueryClassifier._ensure_loaded)
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> LLMCallable | None:
        """Resolve the LLM callable, constructing a real client if needed.

        Returns ``None`` (and logs a warning) when no API key is
        configured for the active provider, so the caller can route
        straight to the rule-based fallback.
        """
        if self._loaded:
            return self._resolved_llm

        if self._resolved_llm is None:
            api_key = settings.active_llm_api_key
            if _is_unset_secret(api_key):
                logger.warning(
                    "QueryDecomposer: LLM is enabled but no API key is "
                    "configured for provider '%s'; falling back to "
                    "rule-based decomposition.",
                    settings.llm_provider,
                )
                self._loaded = True
                return None
            self._resolved_llm = _build_langchain_llm()

        self._loaded = True
        return self._resolved_llm

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------

    def _node_classify(self, state: _DecomposerState) -> dict[str, Any]:
        return {"classification": self.classifier.classify(state["query"])}

    def _node_passthrough(self, state: _DecomposerState) -> dict[str, Any]:
        classification = state["classification"]
        # Exploratory queries want a broad summary; simple-lookup queries
        # want a direct symbol/location result.
        answer_type: ExpectedAnswerType = (
            "explanation" if classification.query_type == "exploratory" else "code"
        )
        step = DecompositionStep(
            id="step-1",
            query=state["query"],
            expected_answer_type=answer_type,
            depends_on=(),
        )
        return {
            "steps": [step],
            "source": "passthrough",
            "raw_response": "",
            "fell_back": False,
            "error": None,
        }

    def _node_decompose_llm(self, state: _DecomposerState) -> dict[str, Any]:
        attempt = state.get("attempt", 0) + 1

        if not self.use_llm:
            return {
                "steps": [],
                "source": "llm",
                "error": "llm_disabled",
                "attempt": attempt,
            }

        llm = self._ensure_loaded()
        if llm is None:
            return {
                "steps": [],
                "source": "llm",
                "error": "llm_unavailable",
                "attempt": attempt,
            }

        prompt = _build_decomposition_prompt(state["query"], state["repo_context"])
        try:
            raw_response = llm(prompt)
        except Exception as exc:  # noqa: BLE001 - any LLM client failure
            logger.warning(
                "QueryDecomposer: LLM call failed (%s); will retry or fall "
                "back to rule-based decomposition.",
                exc,
            )
            return {
                "steps": [],
                "source": "llm",
                "raw_response": "",
                "error": f"llm_call_failed: {exc}",
                "attempt": attempt,
            }

        steps, parse_error = parse_decomposition_response(raw_response)
        return {
            "steps": steps,
            "source": "llm",
            "raw_response": raw_response,
            "error": parse_error,
            "attempt": attempt,
        }

    def _node_validate(self, state: _DecomposerState) -> dict[str, Any]:
        if state.get("error"):
            # Already failed upstream (LLM disabled/unavailable/call
            # failed) -- nothing to structurally validate.
            return {}
        error = validate_steps(state.get("steps", []))
        if error:
            logger.info("QueryDecomposer: LLM plan failed validation (%s).", error)
        return {"error": error}

    def _node_rule_fallback(self, state: _DecomposerState) -> dict[str, Any]:
        steps = rule_based_decompose(state["query"], state["repo_context"])
        return {"steps": steps, "source": "rules", "error": None, "fell_back": True}

    # ------------------------------------------------------------------
    # LangGraph routing
    # ------------------------------------------------------------------

    def _route_after_classify(self, state: _DecomposerState) -> str:
        if state["classification"].query_type == "multi-hop":
            return "decompose"
        return "passthrough"

    def _route_after_validate(self, state: _DecomposerState) -> str:
        error = state.get("error")
        if error is None:
            return "done"
        # Config/availability errors are deterministic -- retrying against
        # the same unconfigured or unreachable-by-design LLM cannot
        # succeed, so go straight to the fallback instead of burning the
        # retry budget.
        if error in ("llm_disabled", "llm_unavailable"):
            return "fallback"
        # A parse/validation error, or a call failure (which may well be
        # transient -- a rate limit, a network blip), is worth one retry
        # before giving up on the LLM path.
        if state.get("attempt", 0) <= self.max_retries:
            return "retry"
        return "fallback"

    def _build_graph(self):  # noqa: ANN201 - langgraph's compiled-graph type
        """Build and compile the decomposition state machine.

        Imported lazily so ``import reporag.agent.planner`` stays cheap and
        does not require ``langgraph`` to be installed for callers that
        only need :class:`QueryClassifier`.
        """
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(_DecomposerState)
        graph.add_node("classify", self._node_classify)
        graph.add_node("passthrough", self._node_passthrough)
        graph.add_node("decompose", self._node_decompose_llm)
        graph.add_node("validate", self._node_validate)
        graph.add_node("rule_fallback", self._node_rule_fallback)

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {"passthrough": "passthrough", "decompose": "decompose"},
        )
        graph.add_edge("passthrough", END)
        graph.add_edge("decompose", "validate")
        graph.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {"done": END, "retry": "decompose", "fallback": "rule_fallback"},
        )
        graph.add_edge("rule_fallback", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(
        self, query: str, repo_context: dict[str, Any] | None = None
    ) -> DecompositionPlan:
        """Decompose *query* into an ordered :class:`DecompositionPlan`.

        Args:
            query: The natural-language query to decompose.
            repo_context: Optional grounding context, e.g.
                ``{"modules": ["api", "routes", "db"], "symbols": [...]}``.
                ``key_symbols`` is also accepted as an alias for
                ``symbols``. Used to make LLM (and, where possible,
                rule-based) sub-queries concrete rather than generic.

        Returns:
            A :class:`DecompositionPlan`. ``plan.steps`` always has at
            least one entry -- 1 for a passthrough plan, 2-5 for an actual
            decomposition.

        Raises:
            ValueError: If *query* is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        normalized_context = _normalize_repo_context(repo_context)
        initial_state: _DecomposerState = {
            "query": query,
            "repo_context": normalized_context,
            "attempt": 0,
            "error": None,
        }
        final_state = self._graph.invoke(initial_state)

        source = final_state.get("source", "rules")
        steps = tuple(final_state.get("steps") or [])
        return DecompositionPlan(
            original_query=query,
            steps=steps,
            needs_decomposition=source != "passthrough",
            classification=final_state["classification"],
            source=source,
            raw_response=final_state.get("raw_response", ""),
            fell_back=bool(final_state.get("fell_back", False)),
            metadata={
                "repo_context": normalized_context,
                "attempts": final_state.get("attempt", 0),
            },
        )

    def __repr__(self) -> str:
        return (
            f"QueryDecomposer(use_llm={self.use_llm}, "
            f"max_retries={self.max_retries}, loaded={self._loaded})"
        )
