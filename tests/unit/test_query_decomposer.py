"""Unit tests for the agentic query decomposer (Issue 21).

Covers every acceptance criterion of Issue 21:

* Decomposes multi-hop queries into 2-5 ordered sub-queries.
* Sub-queries have dependency edges (step 2 depends on step 1).
* Uses repo context (module names, key symbols) to inform decomposition.
* Handles the edge case of a query that does not need decomposition
  (returns a single step).
* LangGraph state machine with clear state transitions.
* 5+ multi-hop query examples.

Beyond the acceptance criteria, the suite pins the design contract laid out
in the module docstring / class docstrings:

* **Dual strategy** -- LLM primary, deterministic rule-based fallback when
  the LLM is disabled, unavailable (no API key), fails, or produces an
  invalid plan after retries.
* **Reuses the Issue 20 classifier** -- ``simple-lookup``/``exploratory``
  short-circuit to a single-step passthrough plan; only ``multi-hop`` goes
  through decomposition.
* **Retry policy** -- parse/validation failures and call failures get one
  retry (default ``max_retries=1``) before falling back; config/
  availability errors (disabled, no key) skip straight to the fallback.
* **Structural validation** -- 2-5 steps, unique ids, only-backward
  dependency edges (no forward references, no cycles).
* **Tolerant parsing** -- markdown/prose-wrapped JSON, and a slightly-off
  ``expected_answer_type`` label, are both handled gracefully.
* **Never raises for a well-formed query** -- every failure mode above
  still produces a valid plan via the rule-based fallback.

A ``_FakeLLM`` (same pattern as ``test_planner.py``) stands in for the real
langchain LLM, keeping every test network-free.
"""

from __future__ import annotations

import json

import pytest

from reporag.agent.planner import (
    DecompositionPlan,
    DecompositionStep,
    QueryClassifier,
    QueryDecomposer,
    _build_decomposition_prompt,
    _extract_json_object,
    _normalize_repo_context,
    parse_decomposition_response,
    rule_based_decompose,
    validate_steps,
)

# ============================================================================
# Test doubles
# ============================================================================


class _FakeLLM:
    """Minimal stand-in for a langchain LLM callable.

    Returns a caller-supplied response, or steps through a list of
    responses (one per call) when *responses* is given -- used to simulate
    "fails once, then succeeds" retry scenarios. Records every prompt so
    tests can assert on prompt contents (e.g. repo context grounding).
    """

    def __init__(
        self,
        response: str | None = None,
        *,
        responses: list[str] | None = None,
        raise_on_call: int | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._response = response
        self._responses = responses
        self._raise_on_call = raise_on_call
        self._exc = exc or RuntimeError("simulated LLM failure")
        self.call_count = 0
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        if self._raise_on_call is not None and self.call_count >= self._raise_on_call:
            raise self._exc
        if self._responses is not None:
            index = min(self.call_count - 1, len(self._responses) - 1)
            return self._responses[index]
        assert self._response is not None
        return self._response


def _valid_plan_json(n_steps: int = 3) -> str:
    """Build a valid ``{"steps": [...]}`` response with *n_steps* steps."""
    steps = []
    for i in range(n_steps):
        deps = [f"step-{j + 1}" for j in range(i)] if i > 0 else []
        steps.append(
            {
                "id": f"step-{i + 1}",
                "query": f"Sub-query number {i + 1}",
                "expected_answer_type": ["code", "explanation", "list"][i % 3],
                "depends_on": deps[:1],  # depend only on the immediately prior step
            }
        )
    return json.dumps({"steps": steps})


def _rules_only_decomposer(**kwargs) -> QueryDecomposer:
    """A fully offline QueryDecomposer: rule-based classifier + rule-based decomposer."""
    classifier = QueryClassifier(use_llm=False)
    return QueryDecomposer(classifier=classifier, use_llm=False, **kwargs)


# ============================================================================
# DecompositionStep / DecompositionPlan
# ============================================================================


class TestDecompositionStep:
    def test_text_is_an_alias_for_query(self) -> None:
        step = DecompositionStep(
            id="step-1", query="find X", expected_answer_type="code"
        )
        assert step.text == step.query == "find X"

    def test_context_from_is_an_alias_for_depends_on(self) -> None:
        step = DecompositionStep(
            id="step-2",
            query="trace X",
            expected_answer_type="explanation",
            depends_on=("step-1",),
        )
        assert step.context_from == step.depends_on == ("step-1",)

    def test_default_depends_on_is_empty(self) -> None:
        step = DecompositionStep(
            id="step-1", query="find X", expected_answer_type="code"
        )
        assert step.depends_on == ()
        assert step.context_from == ()


class TestDecompositionStepValidation:
    """DecompositionStep.__post_init__ validates eagerly, at construction
    time, independent of validate_steps -- this matters because it also
    protects any code that builds a DecompositionStep directly rather than
    going through the parser or rule-based decomposer."""

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError):
            DecompositionStep(id="", query="a", expected_answer_type="code")

    def test_whitespace_only_id_raises(self) -> None:
        with pytest.raises(ValueError):
            DecompositionStep(id="   ", query="a", expected_answer_type="code")

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError):
            DecompositionStep(id="step-1", query="", expected_answer_type="code")

    def test_invalid_expected_answer_type_raises(self) -> None:
        with pytest.raises(ValueError):
            DecompositionStep(id="step-1", query="a", expected_answer_type="bogus")  # type: ignore[arg-type]

    def test_valid_construction_does_not_raise(self) -> None:
        DecompositionStep(id="step-1", query="a", expected_answer_type="list")


# ============================================================================
# Repo context normalization
# ============================================================================


class TestNormalizeRepoContext:
    def test_none_becomes_empty_lists(self) -> None:
        assert _normalize_repo_context(None) == {"modules": [], "symbols": []}

    def test_empty_dict_becomes_empty_lists(self) -> None:
        assert _normalize_repo_context({}) == {"modules": [], "symbols": []}

    def test_modules_are_preserved(self) -> None:
        ctx = _normalize_repo_context({"modules": ["api", "db"]})
        assert ctx["modules"] == ["api", "db"]

    def test_symbols_key_is_accepted(self) -> None:
        ctx = _normalize_repo_context({"symbols": ["authenticate_user"]})
        assert ctx["symbols"] == ["authenticate_user"]

    def test_key_symbols_is_an_accepted_alias(self) -> None:
        ctx = _normalize_repo_context({"key_symbols": ["authenticate_user"]})
        assert ctx["symbols"] == ["authenticate_user"]

    def test_non_string_entries_are_coerced_to_str(self) -> None:
        ctx = _normalize_repo_context({"modules": [123, "db"]})
        assert ctx["modules"] == ["123", "db"]


# ============================================================================
# validate_steps
# ============================================================================


class TestValidateSteps:
    def _steps(self, n: int) -> list[DecompositionStep]:
        return [
            DecompositionStep(
                id=f"step-{i + 1}",
                query=f"q{i + 1}",
                expected_answer_type="code",
                depends_on=(f"step-{i}",) if i > 0 else (),
            )
            for i in range(n)
        ]

    def test_valid_2_step_plan_passes(self) -> None:
        assert validate_steps(self._steps(2)) is None

    def test_valid_5_step_plan_passes(self) -> None:
        assert validate_steps(self._steps(5)) is None

    def test_empty_list_fails(self) -> None:
        assert validate_steps([]) is not None

    def test_single_step_fails_count_check(self) -> None:
        error = validate_steps(self._steps(1))
        assert error is not None
        assert "2-5" in error

    def test_six_steps_fails_count_check(self) -> None:
        error = validate_steps(self._steps(6))
        assert error is not None
        assert "2-5" in error

    def test_duplicate_ids_fail(self) -> None:
        steps = [
            DecompositionStep(id="step-1", query="a", expected_answer_type="code"),
            DecompositionStep(id="step-1", query="b", expected_answer_type="code"),
        ]
        error = validate_steps(steps)
        assert error is not None
        assert "duplicate" in error.lower()

    def test_forward_reference_fails(self) -> None:
        steps = [
            DecompositionStep(
                id="step-1",
                query="a",
                expected_answer_type="code",
                depends_on=("step-2",),  # step-2 doesn't exist yet
            ),
            DecompositionStep(id="step-2", query="b", expected_answer_type="code"),
        ]
        error = validate_steps(steps)
        assert error is not None
        assert "step-2" in error

    def test_self_reference_fails(self) -> None:
        steps = [
            DecompositionStep(
                id="step-1",
                query="a",
                expected_answer_type="code",
                depends_on=("step-1",),
            ),
            DecompositionStep(id="step-2", query="b", expected_answer_type="code"),
        ]
        error = validate_steps(steps)
        assert error is not None

    def test_dependency_on_unknown_id_fails(self) -> None:
        steps = [
            DecompositionStep(id="step-1", query="a", expected_answer_type="code"),
            DecompositionStep(
                id="step-2",
                query="b",
                expected_answer_type="code",
                depends_on=("step-99",),
            ),
        ]
        error = validate_steps(steps)
        assert error is not None

    def test_empty_id_is_rejected_at_construction(self) -> None:
        """DecompositionStep.__post_init__ now rejects this before it can
        ever reach validate_steps -- see TestDecompositionStepValidation."""
        with pytest.raises(ValueError):
            DecompositionStep(id="", query="a", expected_answer_type="code")


# ============================================================================
# _extract_json_object
# ============================================================================


class TestExtractJsonObject:
    def test_extracts_flat_object(self) -> None:
        assert _extract_json_object('{"a": 1}') == '{"a": 1}'

    def test_extracts_nested_object_ignoring_trailing_text(self) -> None:
        raw = 'prose {"steps": [{"id": "step-1", "depends_on": []}]} more prose'
        result = _extract_json_object(raw)
        assert result == '{"steps": [{"id": "step-1", "depends_on": []}]}'
        json.loads(result)  # must be valid JSON on its own

    def test_extracts_from_markdown_fence(self) -> None:
        raw = '```json\n{"steps": [{"id": "step-1"}]}\n```'
        result = _extract_json_object(raw)
        assert json.loads(result) == {"steps": [{"id": "step-1"}]}

    def test_returns_none_when_no_brace(self) -> None:
        assert _extract_json_object("no json here") is None

    def test_returns_none_on_unbalanced_braces(self) -> None:
        assert _extract_json_object('{"a": [{"b": 1}') is None


# ============================================================================
# parse_decomposition_response
# ============================================================================


class TestParseDecompositionResponse:
    def test_valid_response_parses_all_fields(self) -> None:
        raw = json.dumps(
            {
                "steps": [
                    {
                        "id": "step-1",
                        "query": "Locate X",
                        "expected_answer_type": "code",
                        "depends_on": [],
                    },
                    {
                        "id": "step-2",
                        "query": "Explain X",
                        "expected_answer_type": "explanation",
                        "depends_on": ["step-1"],
                    },
                ]
            }
        )
        steps, error = parse_decomposition_response(raw)
        assert error is None
        assert len(steps) == 2
        assert steps[0] == DecompositionStep(
            id="step-1", query="Locate X", expected_answer_type="code", depends_on=()
        )
        assert steps[1].depends_on == ("step-1",)

    def test_markdown_wrapped_response_parses(self) -> None:
        body = _valid_plan_json(2)
        steps, error = parse_decomposition_response(f"```json\n{body}\n```")
        assert error is None
        assert len(steps) == 2

    def test_prose_wrapped_response_parses(self) -> None:
        body = _valid_plan_json(2)
        steps, error = parse_decomposition_response(f"Here is the plan:\n{body}\nDone.")
        assert error is None
        assert len(steps) == 2

    def test_empty_response_is_an_error(self) -> None:
        steps, error = parse_decomposition_response("")
        assert steps == []
        assert error == "empty response"

    def test_no_json_object_is_an_error(self) -> None:
        steps, error = parse_decomposition_response("I cannot decompose this.")
        assert steps == []
        assert error is not None

    def test_invalid_json_is_an_error(self) -> None:
        steps, error = parse_decomposition_response('{"steps": [invalid}')
        assert steps == []
        assert error is not None

    def test_missing_steps_key_is_an_error(self) -> None:
        steps, error = parse_decomposition_response('{"plan": []}')
        assert steps == []
        assert "steps" in error

    def test_empty_steps_list_is_an_error(self) -> None:
        steps, error = parse_decomposition_response('{"steps": []}')
        assert steps == []
        assert error is not None

    def test_step_missing_query_text_is_an_error(self) -> None:
        raw = json.dumps({"steps": [{"id": "step-1", "expected_answer_type": "code"}]})
        steps, error = parse_decomposition_response(raw)
        assert steps == []
        assert "query" in error

    def test_non_list_depends_on_is_an_error(self) -> None:
        raw = json.dumps(
            {
                "steps": [
                    {"id": "step-1", "query": "a", "depends_on": "step-0"},
                ]
            }
        )
        steps, error = parse_decomposition_response(raw)
        assert steps == []
        assert "depends_on" in error

    def test_missing_id_is_auto_generated(self) -> None:
        raw = json.dumps({"steps": [{"query": "a"}, {"query": "b"}]})
        steps, error = parse_decomposition_response(raw)
        assert error is None
        assert steps[0].id == "step-1"
        assert steps[1].id == "step-2"

    @pytest.mark.parametrize(
        "raw_type, expected",
        [
            ("code", "code"),
            ("snippet", "code"),
            ("CODE_SNIPPET", "code"),
            ("explanation", "explanation"),
            ("description", "explanation"),
            ("list", "list"),
            ("items", "list"),
            ("something-unrecognized", "explanation"),
        ],
    )
    def test_expected_answer_type_is_coerced(
        self, raw_type: str, expected: str
    ) -> None:
        raw = json.dumps(
            {
                "steps": [
                    {"id": "step-1", "query": "a", "expected_answer_type": raw_type}
                ]
            }
        )
        steps, error = parse_decomposition_response(raw)
        assert error is None
        assert steps[0].expected_answer_type == expected


# ============================================================================
# rule_based_decompose
# ============================================================================


class TestRuleBasedDecompose:
    def test_from_to_pattern_produces_three_grounded_steps(self) -> None:
        steps = rule_based_decompose(
            "How does a request go from the API endpoint to the database?",
            {"modules": ["api", "db"], "symbols": []},
        )
        assert len(steps) == 3
        assert validate_steps(steps) is None
        assert "API endpoint" in steps[0].query
        assert "database" in steps[1].query
        assert steps[2].depends_on == ("step-1", "step-2")

    def test_from_to_pattern_appends_module_grounding_note(self) -> None:
        steps = rule_based_decompose(
            "How does a request go from the api endpoint to the db layer?",
            {"modules": ["api", "db"], "symbols": []},
        )
        assert "relevant context: api" in steps[0].query
        assert "relevant context: db" in steps[1].query

    def test_three_hop_chain_gets_a_locate_step_per_endpoint(self) -> None:
        """A chain with more than one "to" must not fold the extra hop
        into a single endpoint label -- each endpoint gets its own step."""
        steps = rule_based_decompose(
            "How do I trace data from source to destination to sink?",
            {"modules": ["source", "sink"], "symbols": []},
        )
        assert len(steps) == 4
        assert validate_steps(steps) is None
        assert steps[0].query.startswith("Locate source")
        assert steps[1].query.startswith("Locate destination")
        assert steps[2].query.startswith("Locate sink")
        assert steps[3].depends_on == ("step-1", "step-2", "step-3")
        assert "source" in steps[3].query
        assert "destination" in steps[3].query
        assert "sink" in steps[3].query

    def test_four_hop_chain_still_fits_within_five_steps(self) -> None:
        steps = rule_based_decompose(
            "Trace the flow from A to B to C to D.", {"modules": [], "symbols": []}
        )
        assert len(steps) == 5  # 4 locate steps + 1 trace step
        assert validate_steps(steps) is None
        assert steps[4].depends_on == ("step-1", "step-2", "step-3", "step-4")

    def test_five_hop_chain_falls_back_to_generic_template(self) -> None:
        """A chain too long to fit in 5 steps degrades to the generic
        3-step template rather than producing an invalid (>5 step) plan."""
        steps = rule_based_decompose(
            "from A to B to C to D to E", {"modules": [], "symbols": []}
        )
        assert len(steps) == 3
        assert validate_steps(steps) is None

    def test_from_with_no_to_is_not_treated_as_a_chain(self) -> None:
        steps = rule_based_decompose(
            "Copy files from the backup directory.", {"modules": [], "symbols": []}
        )
        assert len(steps) == 3  # generic template, not a from/to chain
        assert validate_steps(steps) is None

    def test_generic_pattern_used_when_no_from_to(self) -> None:
        steps = rule_based_decompose(
            "How does the auth flow work end-to-end?",
            {"modules": ["auth"], "symbols": []},
        )
        assert len(steps) == 3
        assert validate_steps(steps) is None
        assert steps[1].depends_on == ("step-1",)
        assert steps[2].depends_on == ("step-1", "step-2")

    def test_generic_pattern_grounds_with_matching_module(self) -> None:
        steps = rule_based_decompose(
            "How does the auth flow work end-to-end?",
            {"modules": ["auth", "billing"], "symbols": []},
        )
        assert "relevant context: auth" in steps[1].query

    def test_no_grounding_note_when_no_module_matches(self) -> None:
        steps = rule_based_decompose(
            "How does the auth flow work end-to-end?",
            {"modules": ["billing"], "symbols": []},
        )
        assert "relevant context" not in steps[1].query

    def test_grounding_also_matches_symbols_not_just_modules(self) -> None:
        steps = rule_based_decompose(
            "How does authenticate_user work end-to-end?",
            {"modules": [], "symbols": ["authenticate_user"]},
        )
        assert "relevant context: authenticate_user" in steps[1].query

    def test_none_repo_context_does_not_raise(self) -> None:
        steps = rule_based_decompose(
            "Trace the path from login to token creation.", None
        )
        assert validate_steps(steps) is None

    @pytest.mark.parametrize(
        "query",
        [
            "How does the auth flow work end-to-end?",
            "How does a request go from the API endpoint to the database?",
            "What calls the authenticate_user function and what does it call next?",
            "Trace the path from the login route to the session token creation.",
            "Explain how the ingestion pipeline connects to the embedding step.",
            "What happens between receiving a webhook and updating the database?",
        ],
    )
    def test_always_produces_a_structurally_valid_plan(self, query: str) -> None:
        """The rule-based fallback must never itself fail validation."""
        steps = rule_based_decompose(query, {"modules": ["api", "db"], "symbols": []})
        assert validate_steps(steps) is None
        assert 2 <= len(steps) <= 5


# ============================================================================
# _build_decomposition_prompt
# ============================================================================


class TestBuildDecompositionPrompt:
    def test_includes_the_query(self) -> None:
        prompt = _build_decomposition_prompt(
            "How does auth work end-to-end?", {"modules": [], "symbols": []}
        )
        assert "How does auth work end-to-end?" in prompt

    def test_includes_repo_context(self) -> None:
        prompt = _build_decomposition_prompt(
            "How does auth work?", {"modules": ["auth", "session"], "symbols": []}
        )
        assert "auth" in prompt
        assert "session" in prompt

    def test_includes_json_schema_instructions(self) -> None:
        prompt = _build_decomposition_prompt("q", {"modules": [], "symbols": []})
        assert "depends_on" in prompt
        assert "expected_answer_type" in prompt

    def test_includes_few_shot_examples(self) -> None:
        prompt = _build_decomposition_prompt("q", {"modules": [], "symbols": []})
        assert "authenticate_user" in prompt  # from one of the few-shot examples


# ============================================================================
# QueryDecomposer construction
# ============================================================================


class TestQueryDecomposerConstruction:
    def test_negative_max_retries_raises(self) -> None:
        with pytest.raises(ValueError):
            QueryDecomposer(max_retries=-1)

    def test_default_classifier_is_constructed_when_none_given(self) -> None:
        decomposer = QueryDecomposer(use_llm=False)
        assert isinstance(decomposer.classifier, QueryClassifier)

    def test_injected_classifier_is_reused(self) -> None:
        classifier = QueryClassifier(use_llm=False)
        decomposer = QueryDecomposer(classifier=classifier, use_llm=False)
        assert decomposer.classifier is classifier

    def test_repr_reports_key_state(self) -> None:
        decomposer = _rules_only_decomposer()
        text = repr(decomposer)
        assert "use_llm=False" in text
        assert "max_retries=" in text


# ============================================================================
# LangGraph state machine structure (clear transitions)
# ============================================================================


class TestStateMachineStructure:
    """Pins the graph topology described in the module docstring."""

    def test_all_expected_nodes_are_present(self) -> None:
        decomposer = _rules_only_decomposer()
        nodes = set(decomposer._graph.get_graph().nodes.keys())
        assert {
            "classify",
            "passthrough",
            "decompose",
            "validate",
            "rule_fallback",
        } <= nodes

    def test_classify_branches_to_passthrough_and_decompose(self) -> None:
        decomposer = _rules_only_decomposer()
        edges = decomposer._graph.get_graph().edges
        targets = {e.target for e in edges if e.source == "classify"}
        assert targets == {"passthrough", "decompose"}

    def test_validate_branches_to_end_retry_and_fallback(self) -> None:
        decomposer = _rules_only_decomposer()
        edges = decomposer._graph.get_graph().edges
        validate_edges = {(e.data, e.target) for e in edges if e.source == "validate"}
        assert ("retry", "decompose") in validate_edges
        assert ("fallback", "rule_fallback") in validate_edges
        assert any(data == "done" for data, _target in validate_edges)

    def test_terminal_nodes_reach_end(self) -> None:
        decomposer = _rules_only_decomposer()
        edges = decomposer._graph.get_graph().edges
        for terminal in ("passthrough", "rule_fallback"):
            targets = {e.target for e in edges if e.source == terminal}
            assert "__end__" in targets


# ============================================================================
# QueryDecomposer.decompose -- input validation
# ============================================================================


class TestDecomposeInputValidation:
    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError):
            _rules_only_decomposer().decompose("")

    def test_whitespace_only_query_raises(self) -> None:
        with pytest.raises(ValueError):
            _rules_only_decomposer().decompose("   ")


# ============================================================================
# Acceptance: query that does not need decomposition -> single step
# ============================================================================


class TestPassthroughForNonMultiHop:
    def test_simple_lookup_returns_single_step_plan(self) -> None:
        plan = _rules_only_decomposer().decompose(
            "Where is the authenticate function defined?"
        )
        assert len(plan.steps) == 1
        assert plan.needs_decomposition is False
        assert plan.source == "passthrough"
        assert plan.classification.query_type == "simple-lookup"
        assert plan.steps[0].query == "Where is the authenticate function defined?"

    def test_exploratory_returns_single_step_plan(self) -> None:
        plan = _rules_only_decomposer().decompose(
            "Explain the overall architecture of this codebase."
        )
        assert len(plan.steps) == 1
        assert plan.needs_decomposition is False
        assert plan.source == "passthrough"
        assert plan.classification.query_type == "exploratory"

    def test_simple_lookup_expects_a_code_answer(self) -> None:
        plan = _rules_only_decomposer().decompose("Show me the User model class.")
        assert plan.steps[0].expected_answer_type == "code"

    def test_exploratory_expects_an_explanation_answer(self) -> None:
        plan = _rules_only_decomposer().decompose(
            "Give me an overview of how the ingestion pipeline is organized."
        )
        assert plan.steps[0].expected_answer_type == "explanation"

    def test_passthrough_plan_is_a_valid_decomposition_plan_instance(self) -> None:
        plan = _rules_only_decomposer().decompose("Locate the DatabaseConfig class.")
        assert isinstance(plan, DecompositionPlan)
        assert plan.fell_back is False


# ============================================================================
# Acceptance: 5+ multi-hop query examples, always 2-5 steps with valid edges
# ============================================================================


class TestMultiHopAcceptanceExamples:
    """Issue 21's own acceptance criteria, run offline (deterministic)."""

    @pytest.mark.parametrize(
        "query",
        [
            "How does the auth flow work end-to-end?",
            "How does a request go from the API endpoint to the database?",
            "What calls the authenticate_user function and what does it call next?",
            "Trace the path from the login route to the session token creation.",
            "Explain how the ingestion pipeline connects to the embedding step end-to-end.",
            "How does data flow from the parser to the graph store?",
        ],
    )
    def test_multi_hop_query_decomposes_into_2_to_5_ordered_steps(
        self, query: str
    ) -> None:
        plan = _rules_only_decomposer().decompose(
            query, repo_context={"modules": ["api", "auth", "db", "ingestion"]}
        )
        assert plan.classification.query_type == "multi-hop"
        assert plan.needs_decomposition is True
        assert 2 <= len(plan.steps) <= 5
        assert validate_steps(list(plan.steps)) is None

    def test_later_step_depends_on_earlier_step(self) -> None:
        plan = _rules_only_decomposer().decompose(
            "How does the auth flow work end-to-end?",
            repo_context={"modules": ["auth"]},
        )
        last_step = plan.steps[-1]
        assert len(last_step.depends_on) > 0
        earlier_ids = {s.id for s in plan.steps[:-1]}
        assert set(last_step.depends_on) <= earlier_ids

    def test_repo_context_informs_the_rule_based_decomposition(self) -> None:
        plan = _rules_only_decomposer().decompose(
            "How does a request go from the API endpoint to the database?",
            repo_context={"modules": ["api", "db"]},
        )
        joined = " ".join(s.query for s in plan.steps)
        # The endpoint text is grounded with the matching "api" module note;
        # "database" (the literal captured phrase) is preserved verbatim.
        assert "relevant context: api" in joined
        assert "database" in joined.lower()


# ============================================================================
# LLM path
# ============================================================================


class TestQueryDecomposerLLMPath:
    def _decomposer(self, llm, **kwargs) -> QueryDecomposer:
        # Rule-based classifier keeps routing to "multi-hop" deterministic,
        # so these tests exercise only the decomposition LLM path.
        classifier = QueryClassifier(use_llm=False)
        return QueryDecomposer(llm=llm, classifier=classifier, use_llm=True, **kwargs)

    def test_valid_llm_response_is_used_directly(self) -> None:
        fake = _FakeLLM(response=_valid_plan_json(3))
        plan = self._decomposer(fake).decompose(
            "How does the auth flow work end-to-end?"
        )
        assert plan.source == "llm"
        assert plan.fell_back is False
        assert len(plan.steps) == 3
        assert fake.call_count == 1

    def test_repo_context_is_included_in_the_prompt(self) -> None:
        fake = _FakeLLM(response=_valid_plan_json(2))
        self._decomposer(fake).decompose(
            "How does the auth flow work end-to-end?",
            repo_context={"modules": ["auth", "session"]},
        )
        assert "session" in fake.prompts[0]

    def test_malformed_response_then_valid_response_retries_once(self) -> None:
        fake = _FakeLLM(responses=["not json", _valid_plan_json(3)])
        plan = self._decomposer(fake, max_retries=1).decompose(
            "How does the auth flow work end-to-end?"
        )
        assert plan.source == "llm"
        assert plan.fell_back is False
        assert fake.call_count == 2
        assert plan.metadata["attempts"] == 2

    def test_always_malformed_falls_back_after_exhausting_retries(self) -> None:
        fake = _FakeLLM(response="not json at all")
        plan = self._decomposer(fake, max_retries=1).decompose(
            "How does the auth flow work end-to-end?"
        )
        assert plan.source == "rules"
        assert plan.fell_back is True
        assert fake.call_count == 2  # 1 initial + 1 retry, then fallback
        assert validate_steps(list(plan.steps)) is None

    def test_llm_call_raising_falls_back_after_retry(self) -> None:
        fake = _FakeLLM(response="unused", raise_on_call=1)
        plan = self._decomposer(fake, max_retries=1).decompose(
            "How does the auth flow work end-to-end?"
        )
        assert plan.source == "rules"
        assert plan.fell_back is True
        assert fake.call_count == 2

    def test_zero_max_retries_falls_back_after_a_single_attempt(self) -> None:
        fake = _FakeLLM(response="not json")
        plan = self._decomposer(fake, max_retries=0).decompose(
            "How does the auth flow work end-to-end?"
        )
        assert plan.source == "rules"
        assert fake.call_count == 1

    def test_plan_with_forward_reference_fails_validation_and_falls_back(self) -> None:
        bad_plan = json.dumps(
            {
                "steps": [
                    {
                        "id": "step-1",
                        "query": "a",
                        "expected_answer_type": "code",
                        "depends_on": ["step-2"],  # forward reference
                    },
                    {
                        "id": "step-2",
                        "query": "b",
                        "expected_answer_type": "code",
                        "depends_on": [],
                    },
                ]
            }
        )
        fake = _FakeLLM(response=bad_plan)
        plan = self._decomposer(fake, max_retries=0).decompose(
            "How does the auth flow work end-to-end?"
        )
        assert plan.source == "rules"
        assert plan.fell_back is True

    def test_plan_with_too_few_steps_fails_validation_and_falls_back(self) -> None:
        fake = _FakeLLM(response=_valid_plan_json(1))
        plan = self._decomposer(fake, max_retries=0).decompose(
            "How does the auth flow work end-to-end?"
        )
        assert plan.source == "rules"
        assert plan.fell_back is True

    def test_use_llm_false_skips_llm_and_uses_rules_immediately(self) -> None:
        fake = _FakeLLM(response=_valid_plan_json(3))
        classifier = QueryClassifier(use_llm=False)
        decomposer = QueryDecomposer(llm=fake, classifier=classifier, use_llm=False)
        plan = decomposer.decompose("How does the auth flow work end-to-end?")
        assert plan.source == "rules"
        assert plan.fell_back is True
        assert fake.call_count == 0  # LLM never invoked at all

    def test_no_api_key_configured_falls_back_without_raising(self) -> None:
        # No `llm=` injected and no real API key in the test environment ->
        # _ensure_loaded must detect the placeholder secret and fall back.
        classifier = QueryClassifier(use_llm=False)
        decomposer = QueryDecomposer(classifier=classifier, use_llm=True)
        plan = decomposer.decompose("How does the auth flow work end-to-end?")
        assert plan.source == "rules"
        assert plan.fell_back is True

    def test_llm_response_preserved_on_success(self) -> None:
        raw = _valid_plan_json(2)
        fake = _FakeLLM(response=raw)
        plan = self._decomposer(fake).decompose(
            "How does the auth flow work end-to-end?"
        )
        assert plan.raw_response == raw

    def test_llm_path_still_respects_classifier_passthrough(self) -> None:
        """Even with use_llm=True, a simple-lookup query never reaches the LLM."""
        fake = _FakeLLM(response=_valid_plan_json(3))
        classifier = QueryClassifier(use_llm=False)  # rules -> simple-lookup
        decomposer = QueryDecomposer(llm=fake, classifier=classifier, use_llm=True)
        plan = decomposer.decompose("Where is the authenticate function defined?")
        assert plan.source == "passthrough"
        assert fake.call_count == 0


# ============================================================================
# End-to-end: the exact "how to test locally" script from the issue
# ============================================================================


class TestIssueExampleScript:
    def test_issue_example_query_and_context(self) -> None:
        decomposer = _rules_only_decomposer()
        plan = decomposer.decompose(
            "How does a request go from the API endpoint to the database?",
            repo_context={"modules": ["api", "routes", "db", "models"]},
        )
        assert 2 <= len(plan.steps) <= 5
        for step in plan.steps:
            assert isinstance(step.id, str) and step.id
            assert isinstance(step.query, str) and step.query
            assert isinstance(step.depends_on, tuple)
        assert validate_steps(list(plan.steps)) is None
