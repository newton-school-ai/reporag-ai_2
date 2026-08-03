"""Unit tests for the QueryDecomposer pipeline (Issue 21)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from reporag.agent.planner import (
    DecompositionPlan,
    QueryDecomposer,
    SubQuery,
    parse_decomposition_response,
    rule_based_decompose,
)


class _FakeLLM:
    """Fake LLM callable for network-free testing of QueryDecomposer."""

    def __init__(self, response: str = "", raise_on_call: bool = False) -> None:
        self.response = response
        self.raise_on_call = raise_on_call
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.raise_on_call:
            raise RuntimeError("Fake LLM error")
        return self.response


# ---------------------------------------------------------------------------
# Test Dataclasses Immutability & Structure
# ---------------------------------------------------------------------------


def test_subquery_is_frozen() -> None:
    sub = SubQuery(id="step_1", query="Query", expected_answer_type="code")
    with pytest.raises(FrozenInstanceError):
        sub.id = "step_2"  # type: ignore[misc]


def test_decomposition_plan_is_frozen() -> None:
    plan = DecompositionPlan(
        original_query="What?",
        steps=(SubQuery(id="step_1", query="Query", expected_answer_type="code"),),
        source="rules",
    )
    with pytest.raises(FrozenInstanceError):
        plan.original_query = "Who?"  # type: ignore[misc]


def test_subquery_context_from_alias() -> None:
    sub = SubQuery(
        id="step_2",
        query="Query 2",
        expected_answer_type="code",
        depends_on=("step_1",),
    )
    assert sub.context_from == ("step_1",)
    assert sub.text == "Query 2"

    sub_alt = SubQuery(
        id="step_3",
        text="Query 3",
        expected_answer_type="explanation",
        context_from=("step_2",),
    )
    assert sub_alt.query == "Query 3"
    assert sub_alt.text == "Query 3"
    assert sub_alt.depends_on == ("step_2",)
    assert sub_alt.context_from == ("step_2",)


# ---------------------------------------------------------------------------
# Test Input Validation
# ---------------------------------------------------------------------------


def test_decomposer_empty_query_raises() -> None:
    decomposer = QueryDecomposer(use_llm=False)
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        decomposer.decompose("")


def test_decomposer_whitespace_query_raises() -> None:
    decomposer = QueryDecomposer(use_llm=False)
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        decomposer.decompose("   \n   ")


# ---------------------------------------------------------------------------
# Test Pure Parser logic
# ---------------------------------------------------------------------------


def test_parse_valid_response() -> None:
    raw = """
    Some pre-text
    [
        {"id": "step_1", "query": "Find endpoint", "expected_answer_type": "code", "depends_on": []},
        {"id": "step_2", "query": "Trace path", "expected_answer_type": "explanation", "depends_on": ["step_1"]}
    ]
    Post text.
    """
    steps = parse_decomposition_response(raw)
    assert len(steps) == 2
    assert steps[0]["id"] == "step_1"
    assert steps[0]["expected_answer_type"] == "code"
    assert steps[1]["depends_on"] == ["step_1"]


def test_parse_malformed_json_raises() -> None:
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_decomposition_response("invalid json here")


def test_parse_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="No valid sub-queries parsed"):
        parse_decomposition_response("[]")


def test_parse_coerces_expected_answer_type() -> None:
    raw = '[{"id": "step_1", "query": "Look up foo", "expected_answer_type": "invalid_type", "depends_on": []}]'
    steps = parse_decomposition_response(raw)
    assert steps[0]["expected_answer_type"] == "explanation"


# ---------------------------------------------------------------------------
# Test Rule-based Fallback & Heuristics
# ---------------------------------------------------------------------------


def test_rule_based_decomposes_simple_query_to_single_step() -> None:
    query = "Where is the login function defined?"
    steps = rule_based_decompose(query)
    assert len(steps) == 1
    assert steps[0].id == "step_1"
    assert steps[0].expected_answer_type == "code"
    assert "login" in steps[0].query


def test_rule_based_decomposes_multi_hop_query_with_modules() -> None:
    query = "How does the request flow from routing to db?"
    context = {"modules": ["routing", "db", "auth"]}
    steps = rule_based_decompose(query, repo_context=context)

    # Should detect routing and db, producing a 3-step dependency flow
    assert len(steps) == 3
    assert steps[0].id == "step_1"
    assert "routing" in steps[0].query
    assert steps[1].id == "step_2"
    assert "db" in steps[1].query
    assert steps[2].id == "step_3"
    assert steps[2].depends_on == ("step_1", "step_2")


def test_rule_based_decomposes_multi_hop_query_with_symbols() -> None:
    query = "How is authenticate connected to save_session?"
    context = {"symbols": ["authenticate", "save_session"]}
    steps = rule_based_decompose(query, repo_context=context)

    assert len(steps) == 3
    assert "authenticate" in steps[0].query
    assert "save_session" in steps[1].query
    assert steps[2].depends_on == ("step_1", "step_2")


# ---------------------------------------------------------------------------
# QueryDecomposer Class Tests (Lazy loading, Graph Nodes, Fallbacks, Max Steps)
# ---------------------------------------------------------------------------


def test_decomposer_lazy_loading() -> None:
    decomposer = QueryDecomposer(use_llm=True)
    assert decomposer._loaded is False

    # Trigger ensure loaded (fails gracefully without key because it falls back to rules)
    decomposer._ensure_loaded()
    assert decomposer._loaded is True


def test_decomposer_max_steps_clamping() -> None:
    # LLM returns a plan with 6 steps
    steps_data = [
        {
            "id": f"step_{i}",
            "query": f"Query {i}",
            "expected_answer_type": "code",
            "depends_on": [],
        }
        for i in range(1, 7)
    ]
    # Set step 6 to depend on step 1 and step 5
    steps_data[5]["depends_on"] = ["step_1", "step_5"]

    fake_llm = _FakeLLM(json.dumps(steps_data))
    # Instantiate decomposer with max_steps=4
    decomposer = QueryDecomposer(llm=fake_llm, max_steps=4, use_llm=True)

    plan = decomposer.decompose("Let's retrieve something complex")

    assert len(plan.steps) == 4
    # The output step IDs should be exactly step_1 to step_4
    step_ids = [s.id for s in plan.steps]
    assert step_ids == ["step_1", "step_2", "step_3", "step_4"]


def test_decomposer_max_steps_prunes_out_of_bounds_dependencies() -> None:
    # LLM returns steps where step 3 depends on step 5 (which will get pruned)
    steps_data = [
        {
            "id": "step_1",
            "query": "Q1",
            "expected_answer_type": "code",
            "depends_on": [],
        },
        {
            "id": "step_2",
            "query": "Q2",
            "expected_answer_type": "code",
            "depends_on": [],
        },
        {
            "id": "step_3",
            "query": "Q3",
            "expected_answer_type": "code",
            "depends_on": ["step_1", "step_5"],
        },
        {
            "id": "step_4",
            "query": "Q4",
            "expected_answer_type": "code",
            "depends_on": ["step_2"],
        },
        {
            "id": "step_5",
            "query": "Q5",
            "expected_answer_type": "code",
            "depends_on": [],
        },
    ]

    fake_llm = _FakeLLM(json.dumps(steps_data))
    decomposer = QueryDecomposer(llm=fake_llm, max_steps=3, use_llm=True)

    plan = decomposer.decompose("Find details")

    assert len(plan.steps) == 3
    # step_3's depends_on was ["step_1", "step_5"]. Because step_5 was sliced out, it should be pruned to just ("step_1",)
    assert plan.steps[2].id == "step_3"
    assert plan.steps[2].depends_on == ("step_1",)


def test_decomposer_llm_failure_routes_to_rule_based() -> None:
    fake_llm = _FakeLLM(raise_on_call=True)
    decomposer = QueryDecomposer(llm=fake_llm, use_llm=True)

    # Decomposing a simple query on LLM fail should fallback to rule-based decomposition
    plan = decomposer.decompose("Where is verify_jwt defined?")
    assert plan.source == "rules"
    assert len(plan.steps) == 1
    assert "verify_jwt" in plan.steps[0].query


def test_decomposer_parse_failure_routes_to_rule_based() -> None:
    fake_llm = _FakeLLM("completely unparseable response")
    decomposer = QueryDecomposer(llm=fake_llm, use_llm=True)

    plan = decomposer.decompose("How does signup connect to email service?")
    assert plan.source == "rules"
    # Should run rule based multi-hop trace
    assert len(plan.steps) == 3
    assert plan.steps[2].depends_on == ("step_1", "step_2")


# ---------------------------------------------------------------------------
# Acceptance Criteria: 5+ Multi-hop query examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_step_count"),
    [
        # Example 1: Endpoint to database trace
        ("How does a request go from the API endpoint to the database?", 3),
        # Example 2: Trace login flow end-to-end
        ("Trace the auth flow from the login route to token generation.", 3),
        # Example 3: Multiple modules chain
        ("How do the router, controller, and repository connect during fetch?", 3),
        # Example 4: Symbol callers/callees flow
        ("What calls build_graph and what does build_graph call next?", 3),
        # Example 5: Ingestion pipeline pipeline
        ("How are files cloned and chunked before embedding?", 3),
    ],
)
def test_multihop_decompositions(query: str, expected_step_count: int) -> None:
    """Verify that 5 distinct multi-hop queries decompose correctly using rules fallback."""
    decomposer = QueryDecomposer(use_llm=False)
    context = {
        "modules": [
            "api",
            "db",
            "router",
            "auth",
            "repository",
            "cloner",
            "chunker",
            "embedder",
        ]
    }
    plan = decomposer.decompose(query, repo_context=context)

    assert plan.source == "rules"
    assert len(plan.steps) == expected_step_count
    # Verify step dependencies (ordering constraint)
    assert plan.steps[0].depends_on == ()
    assert len(plan.steps[-1].depends_on) >= 1


# ---------------------------------------------------------------------------
# Safety & Hardening Tests
# ---------------------------------------------------------------------------


def test_decomposer_invalid_max_steps_raises() -> None:
    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        QueryDecomposer(max_steps=0)


def test_decomposer_filters_self_dependencies() -> None:
    steps_data = [
        {
            "id": "step_1",
            "query": "Q1",
            "expected_answer_type": "code",
            "depends_on": ["step_1"],
        },
    ]
    fake_llm = _FakeLLM(json.dumps(steps_data))
    decomposer = QueryDecomposer(llm=fake_llm, use_llm=True)
    plan = decomposer.decompose("Find details")
    assert plan.steps[0].depends_on == ()


def test_decomposer_detects_dependency_cycle_and_falls_back() -> None:
    # LLM returns a plan with cyclic dependency (step_1 depends on step_2, step_2 depends on step_1)
    steps_data = [
        {
            "id": "step_1",
            "query": "Q1",
            "expected_answer_type": "code",
            "depends_on": ["step_2"],
        },
        {
            "id": "step_2",
            "query": "Q2",
            "expected_answer_type": "code",
            "depends_on": ["step_1"],
        },
    ]
    fake_llm = _FakeLLM(json.dumps(steps_data))
    decomposer = QueryDecomposer(llm=fake_llm, use_llm=True)
    # On cycle detection, it should fail parsing/validation and fallback to rule_based
    plan = decomposer.decompose("How does signup connect to email service?")
    assert plan.source == "rules"
    assert len(plan.steps) == 3
