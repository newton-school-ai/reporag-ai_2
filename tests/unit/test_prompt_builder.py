"""Unit tests for the Issue 24 prompt builder.

Every test is offline: the prompt builder never calls an LLM, so there is
nothing to fake here beyond the odd duck-typed step object.
"""

from __future__ import annotations

import pytest

from reporag.generation.context_assembler import ContextAssembler
from reporag.generation.prompt_builder import (
    CITATION_FORMAT,
    BuiltPrompt,
    PromptBuilder,
    SubQueryAnswer,
    extract_file_index,
    normalize_sub_query_answers,
    resolve_context_window,
    truncate_context_blocks,
)
from reporag.retrieval.vector_search import RetrievalResult

QUERY_TYPES = ("simple-lookup", "multi-hop", "exploratory")

CONTEXT = (
    "## src/app/api/routes/auth.py (lines 20-31)\n"
    "```python\n"
    "def login_route(payload: LoginRequest) -> TokenResponse:\n"
    "    return issue_token(authenticate_user(payload.email, payload.password))\n"
    "```\n\n"
    "## src/app/auth/service.py (lines 88-107)\n"
    "```python\n"
    "def authenticate_user(email: str, password: str) -> User | None:\n"
    "    user = UserRepository().get_by_email(email)\n"
    "    return user if verify(password, user.hash) else None\n"
    "```"
)


def make_result(
    score: float,
    file_path: str,
    start_line: int,
    end_line: int,
    text: str,
) -> RetrievalResult:
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=None,
        chunk_text=text,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Templates: one per query type, all sections present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query_type", QUERY_TYPES)
def test_every_query_type_has_a_template(query_type: str) -> None:
    prompt = PromptBuilder().build(
        "How does auth work?", query_type=query_type, context=CONTEXT
    )
    assert prompt


@pytest.mark.parametrize("query_type", QUERY_TYPES)
def test_prompt_contains_all_expected_sections(query_type: str) -> None:
    built = PromptBuilder().build_prompt(
        "How does auth work?", query_type=query_type, context=CONTEXT
    )

    assert "You are RepoRAG" in built.text
    assert "Grounding rules:" in built.text
    assert "Citation rules:" in built.text
    assert "Answer shape:" in built.text
    assert "=== FILES IN CONTEXT ===" in built.text
    assert "=== CODE CONTEXT ===" in built.text
    assert "=== END CODE CONTEXT ===" in built.text
    assert "=== QUESTION ===" in built.text
    assert "How does auth work?" in built.text
    assert CONTEXT in built.text


@pytest.mark.parametrize("query_type", QUERY_TYPES)
def test_citation_format_is_instructed(query_type: str) -> None:
    built = PromptBuilder().build_prompt("Where is login?", query_type=query_type)

    assert CITATION_FORMAT in built.system
    assert "[src/reporag/api/routes/auth.py:42-57]" in built.system
    assert "Never invent a path" in built.system


def test_templates_differ_by_query_type() -> None:
    builder = PromptBuilder()
    prompts = {
        query_type: builder.build("How does auth work?", query_type=query_type)
        for query_type in QUERY_TYPES
    }
    assert len(set(prompts.values())) == 3

    # Each template asks for the shape its category needs.
    assert "No headings" in prompts["simple-lookup"]
    assert "numbered list" in prompts["multi-hop"]
    assert "one short section per component" in prompts["exploratory"]


@pytest.mark.parametrize("query_type", ("multi-hop", "exploratory"))
def test_few_shot_examples_present_for_multi_hop_and_exploratory(
    query_type: str,
) -> None:
    built = PromptBuilder().build_prompt("How does auth work?", query_type=query_type)

    assert "=== EXAMPLES ===" in built.system
    assert built.system.count("Question: ") >= 2
    assert built.system.count("Answer: ") >= 2
    # The examples must model the citation format, not just describe it.
    assert "[src/app/auth/service.py:88-107]" in built.system
    # And they must be marked as not citable, so the model does not cite them.
    assert "NOT part of this" in built.system


def test_multi_hop_example_models_admitting_a_missing_hop() -> None:
    built = PromptBuilder().build_prompt("How does auth work?", query_type="multi-hop")
    assert "cannot be confirmed from this context" in built.system


# ---------------------------------------------------------------------------
# Query type coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    (
        ("multi_hop", "multi-hop"),
        ("multihop", "multi-hop"),
        ("MULTI-HOP", "multi-hop"),
        ("simple_lookup", "simple-lookup"),
        ("lookup", "simple-lookup"),
        ("  exploratory  ", "exploratory"),
    ),
)
def test_query_type_spellings_are_coerced(given: str, expected: str) -> None:
    built = PromptBuilder().build_prompt("Where is login?", query_type=given)
    assert built.query_type == expected


def test_classification_result_can_be_passed_directly() -> None:
    from reporag.agent.planner import ClassificationResult

    classification = ClassificationResult(query_type="exploratory", confidence=0.9)
    built = PromptBuilder().build_prompt(
        "Explain the architecture.", query_type=classification
    )
    assert built.query_type == "exploratory"


def test_unknown_query_type_raises() -> None:
    with pytest.raises(ValueError, match="query_type must be one of"):
        PromptBuilder().build("Where is login?", query_type="nonsense")


@pytest.mark.parametrize("query", ("", "   ", "\n\t"))
def test_empty_query_raises(query: str) -> None:
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        PromptBuilder().build(query, query_type="multi-hop")


# ---------------------------------------------------------------------------
# Sub-query answers
# ---------------------------------------------------------------------------


def test_sub_query_answers_injected_for_multi_hop() -> None:
    built = PromptBuilder().build_prompt(
        "How does a login request reach the database?",
        query_type="multi-hop",
        context=CONTEXT,
        sub_query_answers={
            "step-1": "The login route is login_route in routes/auth.py.",
            "step-2": "authenticate_user verifies the password hash.",
        },
    )

    assert "=== PRIOR FINDINGS ===" in built.user
    assert "[step-1]" in built.user
    assert "[step-2]" in built.user
    assert "The login route is login_route in routes/auth.py." in built.user
    assert built.metadata["sub_query_answer_count"] == 2
    # Prior findings must sit above the context so the question stays last.
    assert built.user.index("=== PRIOR FINDINGS ===") < built.user.index(
        "=== CODE CONTEXT ==="
    )


def test_prior_findings_section_absent_without_answers() -> None:
    built = PromptBuilder().build_prompt(
        "How does auth work?", query_type="multi-hop", context=CONTEXT
    )
    assert "=== PRIOR FINDINGS ===" not in built.text
    assert built.metadata["sub_query_answer_count"] == 0


def test_normalize_sub_query_answers_accepts_every_shape() -> None:
    assert normalize_sub_query_answers(None) == ()
    assert normalize_sub_query_answers([]) == ()
    assert normalize_sub_query_answers({}) == ()

    single = normalize_sub_query_answers("just one finding")
    assert [a.answer for a in single] == ["just one finding"]

    listed = normalize_sub_query_answers(["first", "second"])
    assert [a.step_id for a in listed] == ["step-1", "step-2"]

    mapped = normalize_sub_query_answers({"a": "first", "b": "second"})
    assert [a.step_id for a in mapped] == ["a", "b"]

    dicts = normalize_sub_query_answers(
        [{"id": "s1", "query": "Where is login?", "answer": "In auth.py"}]
    )
    assert dicts[0].step_id == "s1"
    assert dicts[0].query == "Where is login?"
    assert dicts[0].answer == "In auth.py"

    passthrough = normalize_sub_query_answers(
        SubQueryAnswer(step_id="s1", answer="text")
    )
    assert passthrough[0].step_id == "s1"


def test_normalize_sub_query_answers_drops_empty_findings() -> None:
    answers = normalize_sub_query_answers(
        {"step-1": "found it", "step-2": "   ", "step-3": None}
    )
    assert [a.step_id for a in answers] == ["step-1"]


def test_normalize_sub_query_answers_reads_executor_step_results() -> None:
    """StepResult objects from Issue 22's executor drop straight in."""
    from reporag.agent.executor import StepResult

    results = {
        "step-1": StepResult(step_id="step-1", context_summary="def login_route():"),
        "step-2": StepResult(step_id="step-2", skipped=True),
    }
    answers = normalize_sub_query_answers(results)

    assert [a.step_id for a in answers] == ["step-1"]
    assert answers[0].answer == "def login_route():"


def test_normalize_sub_query_answers_ignores_unsupported_types() -> None:
    assert normalize_sub_query_answers(42) == ()


def test_sub_query_answer_query_is_rendered_as_a_heading() -> None:
    built = PromptBuilder().build_prompt(
        "How does auth work?",
        query_type="multi-hop",
        context=CONTEXT,
        sub_query_answers=[
            {"id": "step-1", "query": "Locate the login route", "answer": "auth.py:20"}
        ],
    )
    assert "[step-1] Locate the login route" in built.user


# ---------------------------------------------------------------------------
# File index ("file structure context")
# ---------------------------------------------------------------------------


def test_file_index_lists_every_citable_range() -> None:
    built = PromptBuilder().build_prompt(
        "How does auth work?", query_type="multi-hop", context=CONTEXT
    )

    assert "- src/app/api/routes/auth.py (lines 20-31)" in built.user
    assert "- src/app/auth/service.py (lines 88-107)" in built.user
    assert "only files and line ranges you may cite" in built.user


def test_extract_file_index_groups_ranges_per_file() -> None:
    context = (
        "## a.py (lines 1-5)\n```python\nx\n```\n\n"
        "## a.py (lines 20-22)\n```python\ny\n```\n\n"
        "## b.py (lines 3-4)\n```python\nz\n```"
    )
    assert extract_file_index(context) == [
        ("a.py", ["1-5", "20-22"]),
        ("b.py", ["3-4"]),
    ]


def test_extract_file_index_handles_unknown_line_bounds() -> None:
    assert extract_file_index("## README.md (lines ?-?)\n```\ntext\n```") == [
        ("README.md", ["?-?"])
    ]


def test_extract_file_index_on_free_text_context_is_empty() -> None:
    assert extract_file_index("some hand written context") == []


def test_file_index_omitted_when_disabled() -> None:
    built = PromptBuilder(include_file_index=False).build_prompt(
        "How does auth work?", query_type="multi-hop", context=CONTEXT
    )
    assert "FILES IN CONTEXT" not in built.text


# ---------------------------------------------------------------------------
# Empty context
# ---------------------------------------------------------------------------


def test_empty_context_tells_the_model_to_say_it_does_not_know() -> None:
    built = PromptBuilder().build_prompt("Where is login?", query_type="simple-lookup")

    assert "=== CODE CONTEXT ===" in built.text
    assert "No code was retrieved for this query" in built.text
    assert "FILES IN CONTEXT" not in built.text


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


def test_context_window_is_resolved_by_longest_prefix() -> None:
    assert resolve_context_window("gpt-4o") == 128_000
    assert resolve_context_window("gpt-4o-2024-08-06") == 128_000
    assert resolve_context_window("gpt-4") == 8_192
    assert resolve_context_window("claude-sonnet-4-20250514") == 200_000
    assert resolve_context_window("GPT-4O") == 128_000
    # Unknown models fall back to the conservative default.
    assert resolve_context_window("some-future-model") == 8_192
    assert resolve_context_window("") == 8_192


def test_budget_defaults_to_window_minus_completion_reserve() -> None:
    builder = PromptBuilder(model="gpt-4o", completion_reserve_tokens=1000)
    assert builder.context_window == 128_000
    assert builder.token_budget == 127_000


def test_explicit_max_tokens_overrides_the_derived_budget() -> None:
    assert PromptBuilder(model="gpt-4o", max_tokens=500).token_budget == 500


def test_invalid_budget_arguments_raise() -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        PromptBuilder(max_tokens=0)
    with pytest.raises(ValueError, match="completion_reserve_tokens must be >= 0"):
        PromptBuilder(completion_reserve_tokens=-1)
    with pytest.raises(ValueError, match="leaves no room"):
        PromptBuilder(model="gpt-4", completion_reserve_tokens=8_192)


def test_prompt_fits_within_the_configured_budget() -> None:
    big_context = "\n\n".join(
        f"## src/mod_{i}.py (lines 1-3)\n```python\ndef f_{i}():\n    return {i}\n```"
        for i in range(200)
    )
    built = PromptBuilder(max_tokens=1500).build_prompt(
        "How does auth work?", query_type="multi-hop", context=big_context
    )

    assert built.token_count <= 1500
    assert built.fits_budget


def test_examples_are_dropped_before_prior_findings_and_context() -> None:
    built = PromptBuilder(max_tokens=900).build_prompt(
        "How does auth work?",
        query_type="multi-hop",
        context=CONTEXT,
        sub_query_answers={"step-1": "The login route is login_route."},
    )

    assert built.dropped_sections == ("examples",)
    assert "=== EXAMPLES ===" not in built.text
    # The evidence and the forwarded findings survive.
    assert "=== PRIOR FINDINGS ===" in built.text
    assert CONTEXT in built.text
    assert not built.truncated


def test_prior_findings_are_dropped_before_the_code_context() -> None:
    long_finding = "The login route calls authenticate_user. " * 60
    built = PromptBuilder(max_tokens=700).build_prompt(
        "How does auth work?",
        query_type="multi-hop",
        context=CONTEXT,
        sub_query_answers={"step-1": long_finding},
    )

    assert built.dropped_sections == ("examples", "prior_findings")
    assert "=== PRIOR FINDINGS ===" not in built.text
    assert long_finding.strip() not in built.text
    assert CONTEXT in built.text


def test_context_is_truncated_last_and_at_chunk_boundaries() -> None:
    big_context = "\n\n".join(
        f"## src/mod_{i}.py (lines 1-3)\n```python\ndef f_{i}():\n    return {i}\n```"
        for i in range(100)
    )
    built = PromptBuilder(max_tokens=800).build_prompt(
        "How does auth work?", query_type="multi-hop", context=big_context
    )

    assert built.truncated
    assert built.dropped_sections == ("examples",)
    assert "code context truncated" in built.text
    # Kept chunks are whole: every opened fence is closed, and the first
    # chunk (highest priority) survives.
    assert "## src/mod_0.py (lines 1-3)" in built.text
    assert built.sections["context"].count("```") % 2 == 0
    assert "## src/mod_99.py" not in built.text


def test_file_index_reflects_the_truncated_context_only() -> None:
    big_context = "\n\n".join(
        f"## src/mod_{i}.py (lines 1-3)\n```python\ndef f_{i}():\n    return {i}\n```"
        for i in range(100)
    )
    built = PromptBuilder(max_tokens=800).build_prompt(
        "How does auth work?", query_type="multi-hop", context=big_context
    )

    index_block = built.sections["file_index"]
    assert "src/mod_0.py" in index_block
    # A file that was trimmed away must not be advertised as citable.
    assert "src/mod_99.py" not in index_block


def test_question_and_rules_survive_an_impossible_budget() -> None:
    built = PromptBuilder(max_tokens=1).build_prompt(
        "Where is login_route defined?", query_type="simple-lookup", context=CONTEXT
    )

    assert "Where is login_route defined?" in built.text
    assert CITATION_FORMAT in built.text
    assert not built.fits_budget
    assert built.truncated


def test_per_call_max_tokens_overrides_the_builder_budget() -> None:
    builder = PromptBuilder(max_tokens=100_000)
    built = builder.build_prompt(
        "How does auth work?",
        query_type="multi-hop",
        context=CONTEXT,
        max_tokens=900,
    )

    assert built.token_budget == 900
    assert built.token_count <= 900
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        builder.build_prompt("q", max_tokens=-5)


def test_truncate_context_blocks_keeps_whole_blocks() -> None:
    context = "\n\n".join(
        f"## f{i}.py (lines 1-2)\n```python\nx = {i}\n```" for i in range(10)
    )
    truncated, was_truncated = truncate_context_blocks(context, 60)

    assert was_truncated
    assert "## f0.py" in truncated
    assert "code context truncated" in truncated
    assert truncated.count("```") % 2 == 0


def test_truncate_context_blocks_is_a_no_op_when_it_fits() -> None:
    context = "## f0.py (lines 1-2)\n```python\nx = 0\n```"
    assert truncate_context_blocks(context, 10_000) == (context, False)
    assert truncate_context_blocks("", 10) == ("", False)


def test_truncate_context_blocks_degrades_to_the_marker() -> None:
    context = "## f0.py (lines 1-2)\n```python\nx = 0\n```"
    truncated, was_truncated = truncate_context_blocks(context, 1)
    assert was_truncated
    assert truncated == truncate_context_blocks(context, 1)[0]
    assert "truncated" in truncated


# ---------------------------------------------------------------------------
# BuiltPrompt shape
# ---------------------------------------------------------------------------


def test_build_returns_the_same_text_as_build_prompt() -> None:
    builder = PromptBuilder()
    text = builder.build("How does auth work?", "multi-hop", CONTEXT)
    built = builder.build_prompt("How does auth work?", "multi-hop", CONTEXT)

    assert isinstance(text, str)
    assert text == built.text
    assert str(built) == built.text
    # The issue's usage example slices the return value.
    assert text[:100] == built.text[:100]


def test_built_prompt_exposes_chat_messages() -> None:
    built = PromptBuilder().build_prompt(
        "How does auth work?", query_type="multi-hop", context=CONTEXT
    )
    messages = built.messages

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == built.system
    assert messages[1]["content"] == built.user
    assert built.text == f"{built.system}\n\n{built.user}"


def test_built_prompt_metadata_and_accounting() -> None:
    built = PromptBuilder(model="gpt-4o").build_prompt(
        "How does auth work?", query_type="multi-hop", context=CONTEXT
    )

    assert isinstance(built, BuiltPrompt)
    assert built.query == "How does auth work?"
    assert built.metadata["model"] == "gpt-4o"
    assert built.metadata["context_window"] == 128_000
    assert built.token_count > 0
    assert built.dropped_sections == ()
    assert not built.truncated


def test_question_is_the_last_thing_before_the_final_instruction() -> None:
    built = PromptBuilder().build_prompt(
        "How does auth work?", query_type="multi-hop", context=CONTEXT
    )
    assert built.user.index("=== QUESTION ===") > built.user.index(
        "=== CODE CONTEXT ==="
    )
    assert built.user.rstrip().endswith("citation rules.")


def test_prompt_is_ascii_only() -> None:
    built = PromptBuilder().build_prompt(
        "How does auth work?",
        query_type="exploratory",
        context=CONTEXT,
        sub_query_answers={"step-1": "found it"},
    )
    built.text.encode("ascii")


def test_repr_is_informative() -> None:
    assert "PromptBuilder(model=" in repr(PromptBuilder(model="gpt-4o"))


# ---------------------------------------------------------------------------
# Integration with the Issue 23 context assembler
# ---------------------------------------------------------------------------


def test_build_from_results_assembles_then_builds() -> None:
    results = [
        make_result(0.9, "src/a.py", 1, 2, "def a():\n    return 1"),
        make_result(0.8, "src/b.py", 10, 11, "def b():\n    return 2"),
    ]
    built = PromptBuilder(max_tokens=4000).build_from_results(
        "How does a work?", query_type="multi-hop", results=results
    )

    assert "## src/a.py (lines 1-2)" in built.text
    assert "## src/b.py (lines 10-11)" in built.text
    assert "- src/a.py (lines 1-2)" in built.user
    assert built.token_count <= 4000


def test_build_from_results_without_results_uses_the_empty_context_note() -> None:
    built = PromptBuilder().build_from_results("Where is login?", "simple-lookup")
    assert "No code was retrieved for this query" in built.text


def test_build_from_results_accepts_an_injected_assembler() -> None:
    builder = PromptBuilder(assembler=ContextAssembler(max_tokens=50))
    built = builder.build_from_results(
        "How does a work?",
        query_type="multi-hop",
        results=[make_result(0.9, "src/a.py", 1, 2, "def a():\n    return 1")],
    )
    assert "## src/a.py (lines 1-2)" in built.text
