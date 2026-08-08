"""Tests for StrategyRouter and rule_based_route (Issue #22)."""

from unittest.mock import patch

from reporag.agent.router import (
    StrategyRouter,
    _looks_like_identifier,
    rule_based_route,
)

# ---------------------------------------------------------------------------
# _looks_like_identifier unit tests
# ---------------------------------------------------------------------------


def test_identifier_accepts_snake_case():
    assert _looks_like_identifier("authenticate_user") is True
    assert _looks_like_identifier("foo_bar") is True
    assert _looks_like_identifier("get_user_by_id") is True


def test_identifier_accepts_qualified_names():
    assert _looks_like_identifier("UserService.login") is True
    assert _looks_like_identifier("module.auth.authenticate_user") is True


def test_identifier_accepts_camelcase():
    # Requires a lowercase->uppercase transition inside the word
    assert _looks_like_identifier("UserService") is True  # r->S
    assert _looks_like_identifier("UserID") is True  # r->I


def test_identifier_rejects_plain_english():
    assert _looks_like_identifier("callers") is False
    assert _looks_like_identifier("references") is False
    assert _looks_like_identifier("handling") is False
    assert _looks_like_identifier("the") is False
    assert _looks_like_identifier("some") is False
    assert _looks_like_identifier("authenticate") is False  # plain lowercase, no _/.


def test_identifier_rejects_single_word_pascal_case():
    # Single-word PascalCase (no internal uppercase transition) is intentionally
    # excluded from the loose patterns; the strict patterns handle it via structural
    # keywords such as "class Database".
    assert _looks_like_identifier("Database") is False
    assert _looks_like_identifier("Authenticator") is False


# ---------------------------------------------------------------------------
# rule_based_route -- BM25 routing
# ---------------------------------------------------------------------------


def test_rule_bm25_strict_extracts_symbol():
    d = rule_based_route("Where is the function foo_bar?")
    assert d.strategy == "bm25"
    assert d.symbol == "foo_bar"


def test_rule_bm25_loose_snake_case():
    d = rule_based_route("Find authenticate_user")
    assert d.strategy == "bm25"
    assert d.symbol == "authenticate_user"


def test_rule_bm25_loose_qualified():
    d = rule_based_route("Find module.auth.authenticate_user")
    assert d.strategy == "bm25"
    assert d.symbol == "module.auth.authenticate_user"


def test_rule_bm25_loose_rejects_plain_english_noun():
    # "references" must not be extracted as a BM25 symbol
    d = rule_based_route("Find references to authenticate_user")
    assert d.strategy == "hybrid"
    assert d.symbol is None


def test_rule_bm25_loose_rejects_stopword_the():
    d = rule_based_route("Find the file")
    assert d.strategy == "hybrid"


def test_rule_no_false_positive_file_handling():
    # Regression: "handling" must not become a BM25 symbol
    d = rule_based_route("Where is the file handling auth?")
    assert d.strategy == "hybrid"
    assert d.symbol is None


# ---------------------------------------------------------------------------
# rule_based_route -- Graph routing
# ---------------------------------------------------------------------------


def test_rule_graph_strict_extracts_pascal_class():
    d = rule_based_route("What calls the class Database?")
    assert d.strategy == "graph"
    assert d.symbol == "Database"


def test_rule_graph_loose_snake_case():
    d = rule_based_route("Trace foo_bar")
    assert d.strategy == "graph"
    assert d.symbol == "foo_bar"


def test_rule_graph_strict_qualified():
    d = rule_based_route("What calls method UserService.login?")
    assert d.strategy == "graph"
    assert d.symbol == "UserService.login"


def test_rule_graph_loose_rejects_english_noun_callers():
    # "callers" must not be extracted as a graph symbol
    d = rule_based_route("Trace callers of UserService.login")
    assert d.strategy == "hybrid"
    assert d.symbol is None


def test_rule_graph_fallback_when_no_identifier_after_trigger():
    # "function" with no following identifier -> no strict match, no loose identifier
    d = rule_based_route("What calls the function?")
    assert d.strategy == "hybrid"


# ---------------------------------------------------------------------------
# rule_based_route -- Vector / hybrid routing
# ---------------------------------------------------------------------------


def test_rule_vector_semantic_query():
    assert rule_based_route("How does this work?").strategy == "vector"


def test_rule_vector_explain():
    assert rule_based_route("Explain the authentication flow").strategy == "vector"


def test_rule_hybrid_ambiguous():
    assert rule_based_route("Should we refactor this?").strategy == "hybrid"


# ---------------------------------------------------------------------------
# StrategyRouter -- LLM path (injected callable, network-free)
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Fake LLM callable for test injection."""

    def __init__(
        self,
        response: str = '{"strategy": "bm25", "symbol": "foo"}',
        *,
        raise_on_call: bool = False,
    ) -> None:
        self._response = response
        self._raise_on_call = raise_on_call

    def __call__(self, prompt: str) -> str:
        if self._raise_on_call:
            raise RuntimeError("simulated LLM failure")
        return self._response


def test_llm_parses_json_routing():
    router = StrategyRouter(
        llm_callable=lambda q: '{"strategy": "graph", "symbol": "foo"}'
    )
    decision = router.route("query")
    assert decision.strategy == "graph"
    assert decision.symbol == "foo"


def test_llm_invalid_strategy_falls_back_to_rules():
    # An invalid strategy triggers ValueError -> caught -> rule-based fallback fires
    router = StrategyRouter(
        llm_callable=lambda q: '{"strategy": "magic", "symbol": "foo"}'
    )
    # The raw query "query" doesn't match any rule-based pattern -> hybrid
    decision = router.route("query")
    assert decision.strategy == "hybrid"


def test_llm_failure_triggers_fallback():
    fake = _FakeLLM(raise_on_call=True)
    router = StrategyRouter(llm_callable=fake)
    decision = router.route("Where is db_connect?")
    assert decision.strategy == "bm25"
    assert decision.symbol == "db_connect"


def test_llm_malformed_json_triggers_fallback():
    fake = _FakeLLM(response="this is not json")
    router = StrategyRouter(llm_callable=fake)
    decision = router.route("What calls foo_bar?")
    assert decision.strategy == "graph"
    assert decision.symbol == "foo_bar"


def test_llm_null_symbol_is_allowed():
    router = StrategyRouter(
        llm_callable=lambda q: '{"strategy": "vector", "symbol": null}'
    )
    decision = router.route("How does auth work?")
    assert decision.strategy == "vector"
    assert decision.symbol is None


def test_llm_receives_raw_query_not_prefilled_prompt():
    """Injected callable receives the raw query string, not a pre-formatted prompt."""
    received: list[str] = []

    def recording_llm(q: str) -> str:
        received.append(q)
        return '{"strategy": "bm25", "symbol": "foo"}'

    router = StrategyRouter(llm_callable=recording_llm)
    router.route("Where is foo?")

    assert received == ["Where is foo?"]


# ---------------------------------------------------------------------------
# StrategyRouter -- Production LangChain chain path (regression for double-format)
# ---------------------------------------------------------------------------


def test_production_llm_chain_formats_query_exactly_once():
    """Production path must not double-format the prompt.

    When no llm_callable is injected, _ensure_llm() builds a LangChain
    RunnableSequence (``_ROUTER_PROMPT | backend``).  route() must pass the
    *raw* query to the chain; the chain formats the template.  If route()
    were to pre-format the template and then pass the result as the chain's
    {query} slot the instructions would appear twice in the final prompt.

    This test mocks _build_langchain_llm at the module level so no real API
    key is required while still exercising the actual chain construction path.
    """
    received_prompts: list[str] = []

    def fake_backend(prompt_value) -> str:
        received_prompts.append(str(prompt_value))
        return '{"strategy": "bm25", "symbol": "foo"}'

    with patch("reporag.agent.router._build_langchain_llm", return_value=fake_backend):
        router = StrategyRouter()  # No injected callable -> _ensure_llm builds chain
        decision = router.route("Where is foo?")

    assert len(received_prompts) == 1, "LLM backend must be called exactly once"

    received = received_prompts[0]
    # The query text must appear exactly once in the formatted prompt.
    # Double-formatting would embed it inside a pre-formatted prompt that is then
    # passed as the {query} value again, producing e.g. "Query: ...Query: Where is foo?"
    assert (
        received.count("Where is foo?") == 1
    ), "Query text appears more than once -- prompt is being double-formatted"
    # The routing instruction preamble must also appear only once
    assert (
        received.count("query routing assistant") == 1
    ), "Routing instructions appear more than once -- prompt is being double-formatted"
    assert decision.strategy == "bm25"
    assert decision.symbol == "foo"
