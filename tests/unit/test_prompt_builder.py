"""Unit tests for the prompt builder (Issue 24).

Covers every acceptance criterion of Issue 24:

* templates exist for all three query types and differ meaningfully,
* the [file_path:start_line-end_line] citation format is clearly
  instructed in every template,
* few-shot examples are present in every template and model the citation
  format inline,
* prior sub-query findings are injected for multi-hop and accepted in
  every documented shape (string, list, dict, duck-typed step-result
  objects),
* the assembled prompt fits within the resolved model context window, with
  a fixed section-dropping priority (examples, then prior findings, then
  the code context) and a verified (not estimated) fit for context
  truncation,

plus edge cases found while building against the codebase's real
`ContextAssembler` (a plain-string-returning implementation, not the
richer object model this module's docstring explains it deliberately does
not assume).

All tests are fully offline -- no LLM calls, no network.
"""

from __future__ import annotations

import pytest

from reporag.generation.prompt_builder import (
    _TEMPLATES,
    PromptBuilder,
    _extract_file_index,
    _resolve_context_window,
    normalize_prior_findings,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _context(*chunks: tuple[str, int, int, str]) -> str:
    """Build a context string in this codebase's real assembler format.

    Each chunk is ``(file_path, start_line, end_line, body)``.
    """
    return "\n\n".join(
        f"## {path} (lines {start}-{end})\n```python\n{body}\n```"
        for path, start, end, body in chunks
    )


_SIMPLE_CONTEXT = _context(
    ("src/app/api/routes/auth.py", 20, 24, "def login_route(payload):\n    ...")
)


class _FakeStepResult:
    """Duck-typed stand-in for `reporag.agent.executor.StepResult`."""

    def __init__(self, step_id, query, context_summary, skipped=False):
        self.step_id = step_id
        self.query = query
        self.context_summary = context_summary
        self.skipped = skipped


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder(max_tokens=50_000)


# ---------------------------------------------------------------------------
# Acceptance: templates for all 3 query types
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_every_query_type_has_a_template(self) -> None:
        assert set(_TEMPLATES) == {"simple-lookup", "multi-hop", "exploratory"}

    def test_templates_differ_by_query_type(self, builder: PromptBuilder) -> None:
        prompts = {
            qt: builder.build("How does auth work?", qt, _SIMPLE_CONTEXT)
            for qt in _TEMPLATES
        }
        assert prompts["simple-lookup"] != prompts["multi-hop"]
        assert prompts["multi-hop"] != prompts["exploratory"]
        assert prompts["simple-lookup"] != prompts["exploratory"]

    def test_unknown_query_type_raises(self, builder: PromptBuilder) -> None:
        with pytest.raises(ValueError, match="query_type must be one of"):
            builder.build_prompt("Q", "not-a-real-type", _SIMPLE_CONTEXT)

    @pytest.mark.parametrize("query_type", list(_TEMPLATES))
    def test_prompt_contains_all_expected_sections(
        self, builder: PromptBuilder, query_type: str
    ) -> None:
        built = builder.build_prompt("How does auth work?", query_type, _SIMPLE_CONTEXT)
        full = built.system + "\n\n" + built.user
        assert "code assistant" in full.lower()
        assert (
            "only from" in full.lower()
            or "only using" in full.lower()
            or "using only" in full.lower()
        )
        assert "[file_path:start_line-end_line]" in full
        assert "ANSWER SHAPE:" in full
        assert "FILES IN CONTEXT" in full
        assert "CODE CONTEXT" in full
        assert "How does auth work?" in full


# ---------------------------------------------------------------------------
# Acceptance: citation format clearly instructed
# ---------------------------------------------------------------------------


class TestCitationFormat:
    @pytest.mark.parametrize("query_type", list(_TEMPLATES))
    def test_citation_format_is_instructed(
        self, builder: PromptBuilder, query_type: str
    ) -> None:
        built = builder.build_prompt("Q", query_type, _SIMPLE_CONTEXT)
        assert "[file_path:start_line-end_line]" in built.system
        # A worked example ties the abstract format to a concrete instance.
        assert "[src/app/auth/service.py:88-92]" in built.system

    def test_citation_rule_forbids_inventing_paths(
        self, builder: PromptBuilder
    ) -> None:
        built = builder.build_prompt("Q", "simple-lookup", _SIMPLE_CONTEXT)
        assert "never invent" in built.system.lower()

    def test_files_in_context_whitelists_exactly_the_headers_shown(
        self, builder: PromptBuilder
    ) -> None:
        context = _context(
            ("a.py", 1, 5, "x = 1"),
            ("b.py", 10, 20, "y = 2"),
        )
        index = _extract_file_index(context)
        assert "a.py (lines 1-5)" in index
        assert "b.py (lines 10-20)" in index

    def test_files_in_context_handles_missing_line_placeholder(
        self, builder: PromptBuilder
    ) -> None:
        context = "## sym.py (lines ?-?)\n```python\nclass Foo: ...\n```"
        index = _extract_file_index(context)
        assert "sym.py (lines ?-?)" in index


# ---------------------------------------------------------------------------
# Acceptance: few-shot examples included
# ---------------------------------------------------------------------------


class TestFewShotExamples:
    @pytest.mark.parametrize("query_type", list(_TEMPLATES))
    def test_every_template_has_at_least_two_examples(self, query_type: str) -> None:
        assert len(_TEMPLATES[query_type].examples) >= 2

    def test_examples_appear_in_the_system_prompt_by_default(
        self, builder: PromptBuilder
    ) -> None:
        built = builder.build_prompt("Q", "multi-hop", _SIMPLE_CONTEXT)
        assert "EXAMPLES" in built.system
        assert "Q: How does a login request reach the database?" in built.system

    def test_examples_model_the_citation_format_inline(self) -> None:
        multi_hop_examples = _TEMPLATES["multi-hop"].examples
        assert any("[" in ex.answer and "]" in ex.answer for ex in multi_hop_examples)

    def test_multi_hop_example_models_admitting_a_missing_hop(self) -> None:
        examples = _TEMPLATES["multi-hop"].examples
        assert any(
            "can't trace" in ex.answer.lower() or "can't confirm" in ex.answer.lower()
            for ex in examples
        )

    def test_simple_lookup_example_models_declining_when_absent(self) -> None:
        examples = _TEMPLATES["simple-lookup"].examples
        assert any("can't find" in ex.answer.lower() for ex in examples)

    def test_examples_marked_as_not_part_of_the_repository(
        self, builder: PromptBuilder
    ) -> None:
        built = builder.build_prompt("Q", "exploratory", _SIMPLE_CONTEXT)
        assert (
            "not part of this codebase" in built.system.lower()
            or "not part of" in built.system.lower()
        )
        assert (
            "never be cited" in built.system.lower()
            or "never cited" in built.system.lower()
        )


# ---------------------------------------------------------------------------
# Acceptance: sub-query answers injected for multi-hop
# ---------------------------------------------------------------------------


class TestPriorFindingsNormalization:
    def test_none_and_empty_return_empty_list(self) -> None:
        assert normalize_prior_findings(None) == []
        assert normalize_prior_findings("") == []
        assert normalize_prior_findings([]) == []
        assert normalize_prior_findings({}) == []

    def test_plain_string(self) -> None:
        assert normalize_prior_findings("a finding") == [("", "", "a finding")]

    def test_whitespace_only_string_is_empty(self) -> None:
        assert normalize_prior_findings("   ") == []

    def test_list_of_strings(self) -> None:
        assert normalize_prior_findings(["a", "b"]) == [
            ("", "", "a"),
            ("", "", "b"),
        ]

    def test_list_of_strings_drops_blank_entries(self) -> None:
        assert normalize_prior_findings(["a", "  ", "b"]) == [
            ("", "", "a"),
            ("", "", "b"),
        ]

    def test_dict_mapping_preserves_insertion_order(self) -> None:
        result = normalize_prior_findings({"step-2": "second", "step-1": "first"})
        assert result == [("step-2", "", "second"), ("step-1", "", "first")]

    def test_dict_drops_blank_values(self) -> None:
        result = normalize_prior_findings({"step-1": "text", "step-2": "  "})
        assert result == [("step-1", "", "text")]

    def test_step_result_objects_read_context_summary(self) -> None:
        steps = [_FakeStepResult("step-1", "sub query", "the finding")]
        assert normalize_prior_findings(steps) == [
            ("step-1", "sub query", "the finding")
        ]

    def test_step_result_skipped_is_dropped(self) -> None:
        steps = [
            _FakeStepResult("step-1", "q1", "kept"),
            _FakeStepResult("step-2", "q2", "should not appear", skipped=True),
        ]
        result = normalize_prior_findings(steps)
        assert len(result) == 1
        assert result[0][0] == "step-1"

    def test_step_result_empty_context_summary_is_dropped(self) -> None:
        steps = [_FakeStepResult("step-1", "q1", "")]
        assert normalize_prior_findings(steps) == []

    def test_normalize_prior_findings_accepts_every_documented_shape(self) -> None:
        # A single smoke test exercising every shape the docstring promises.
        assert normalize_prior_findings("s") != []
        assert normalize_prior_findings(["s"]) != []
        assert normalize_prior_findings({"id": "s"}) != []
        assert normalize_prior_findings([_FakeStepResult("id", "q", "s")]) != []

    def test_prior_findings_injected_for_multi_hop(
        self, builder: PromptBuilder
    ) -> None:
        built = builder.build_prompt(
            "Q", "multi-hop", _SIMPLE_CONTEXT, {"step-1": "prior text"}
        )
        assert "PRIOR FINDINGS" in built.user
        assert "prior text" in built.user

    def test_prior_findings_ignored_for_non_multi_hop(
        self, builder: PromptBuilder
    ) -> None:
        built = builder.build_prompt(
            "Q", "simple-lookup", _SIMPLE_CONTEXT, {"step-1": "prior text"}
        )
        assert "PRIOR FINDINGS" not in built.user

    def test_no_prior_findings_section_when_none_given(
        self, builder: PromptBuilder
    ) -> None:
        built = builder.build_prompt("Q", "multi-hop", _SIMPLE_CONTEXT)
        assert "PRIOR FINDINGS" not in built.user


# ---------------------------------------------------------------------------
# Acceptance: prompt fits within model context window
# ---------------------------------------------------------------------------


class TestContextWindowResolution:
    def test_exact_match(self) -> None:
        assert _resolve_context_window("gpt-4") == 8_192

    def test_longest_prefix_wins_over_shorter_one(self) -> None:
        # "gpt-4o-2024-08-06" matches both "gpt-4" and "gpt-4o" as
        # prefixes; the longer, more specific one must win.
        assert _resolve_context_window("gpt-4o-2024-08-06") == 128_000

    def test_unknown_model_falls_back_to_default(self) -> None:
        assert _resolve_context_window("some-unknown-model-9000") == 8_192

    def test_claude_model_resolves(self) -> None:
        assert _resolve_context_window("claude-sonnet-4-20250514") == 200_000

    def test_default_model_used_when_none_given(self) -> None:
        builder = PromptBuilder()
        assert builder.model  # non-empty, resolved from settings

    def test_explicit_max_tokens_overrides_model_resolution(self) -> None:
        builder = PromptBuilder(model="gpt-4o", max_tokens=123)
        assert builder.token_budget == 123

    def test_completion_reserve_is_subtracted(self) -> None:
        builder = PromptBuilder(model="gpt-4", completion_reserve=1000)
        assert builder.token_budget == 8_192 - 1000

    def test_constructor_rejects_non_positive_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            PromptBuilder(max_tokens=0)

    def test_constructor_rejects_negative_completion_reserve(self) -> None:
        with pytest.raises(ValueError, match="completion_reserve"):
            PromptBuilder(completion_reserve=-1)


class TestBudgetFitting:
    def test_prompt_fits_within_the_configured_budget(self) -> None:
        builder = PromptBuilder(max_tokens=100_000)
        built = builder.build_prompt("Q", "multi-hop", _SIMPLE_CONTEXT)
        assert built.token_count <= built.token_budget
        assert built.fits_budget is True
        assert built.dropped_sections == []
        assert built.truncated is False

    def test_examples_dropped_before_prior_findings_and_context(self) -> None:
        # A budget big enough for everything minus examples, small enough
        # that examples alone tip it over.
        builder = PromptBuilder(max_tokens=100_000)
        full = builder.build_prompt("Q", "multi-hop", _SIMPLE_CONTEXT, {"s1": "prior"})
        no_examples_tokens = full.token_count - 1  # force a squeeze
        tight = PromptBuilder(max_tokens=no_examples_tokens)
        built = tight.build_prompt("Q", "multi-hop", _SIMPLE_CONTEXT, {"s1": "prior"})
        assert "few_shot_examples" in built.dropped_sections
        assert "EXAMPLES" not in built.system
        # Prior findings should have survived, since only examples needed
        # to go to fit this particular budget.
        assert "PRIOR FINDINGS" in built.user

    def test_prior_findings_dropped_before_the_code_context(self) -> None:
        # Budget so tight that both examples and findings must go, but the
        # (small) context still fits.
        builder = PromptBuilder(max_tokens=10_000)
        baseline = builder.build_prompt(
            "Q", "multi-hop", _SIMPLE_CONTEXT, {"s1": "x" * 2000}
        )
        # Squeeze to just past "no examples, with findings" -- must drop findings.
        tight = PromptBuilder(max_tokens=baseline.token_count - 600)
        built = tight.build_prompt(
            "Q", "multi-hop", _SIMPLE_CONTEXT, {"s1": "x" * 2000}
        )
        assert "few_shot_examples" in built.dropped_sections
        assert "prior_findings" in built.dropped_sections
        assert "PRIOR FINDINGS" not in built.user
        assert "CODE CONTEXT" in built.user

    def test_context_is_truncated_last_and_at_chunk_boundaries(self) -> None:
        big_context = "\n\n".join(
            f"## src/mod_{i}.py (lines 1-3)\n```python\ndef f_{i}():\n    return {i}\n```"
            for i in range(50)
        )
        builder = PromptBuilder(max_tokens=700)
        built = builder.build_prompt("Q", "multi-hop", big_context)
        assert built.truncated is True
        assert built.token_count <= built.token_budget
        # Every surviving fenced block must be a complete pair of fences.
        assert built.user.count("```") % 2 == 0

    def test_question_and_rules_survive_an_impossible_budget(self) -> None:
        builder = PromptBuilder(max_tokens=5)
        built = builder.build_prompt(
            "Very important question", "simple-lookup", _SIMPLE_CONTEXT
        )
        assert built.fits_budget is False
        assert "Very important question" in built.user
        assert "[file_path:start_line-end_line]" in built.system

    def test_fits_budget_false_only_when_core_alone_exceeds_it(self) -> None:
        builder = PromptBuilder(max_tokens=1)
        built = builder.build_prompt("Q", "simple-lookup", _SIMPLE_CONTEXT)
        assert built.fits_budget is False

    def test_messages_have_system_and_user_roles(self) -> None:
        builder = PromptBuilder(max_tokens=50_000)
        built = builder.build_prompt("Q", "simple-lookup", _SIMPLE_CONTEXT)
        assert [m["role"] for m in built.messages] == ["system", "user"]
        assert built.messages[0]["content"] == built.system
        assert built.messages[1]["content"] == built.user

    def test_build_returns_plain_concatenated_string(self) -> None:
        builder = PromptBuilder(max_tokens=50_000)
        built = builder.build_prompt("Q", "simple-lookup", _SIMPLE_CONTEXT)
        plain = builder.build("Q", "simple-lookup", _SIMPLE_CONTEXT)
        assert plain == f"{built.system}\n\n{built.user}"


# ---------------------------------------------------------------------------
# build_from_results integration with the real ContextAssembler
# ---------------------------------------------------------------------------


class TestBuildFromResults:
    def test_assembles_and_builds_in_one_call(self) -> None:
        from reporag.retrieval.vector_search import RetrievalResult

        results = [
            RetrievalResult(
                0.9,
                "src/app/api/routes/auth.py",
                20,
                24,
                "login_route",
                "def login_route(payload):\n    return authenticate_user(payload)",
                {},
            ),
        ]
        builder = PromptBuilder(max_tokens=50_000)
        built = builder.build_from_results(
            "How does auth work?", "simple-lookup", results
        )
        assert "src/app/api/routes/auth.py" in built.user
        assert built.fits_budget is True

    def test_shrinks_context_budget_when_prompt_does_not_fit(self) -> None:
        from reporag.retrieval.vector_search import RetrievalResult

        results = [
            RetrievalResult(
                0.9 - i * 0.01,
                f"src/mod_{i}.py",
                1,
                40,
                None,
                "\n".join(f"line {j}" for j in range(40)),
                {},
            )
            for i in range(30)
        ]
        builder = PromptBuilder(max_tokens=1200)
        built = builder.build_from_results(
            "What does this codebase do?", "exploratory", results
        )
        assert built.token_count <= built.token_budget

    def test_higher_score_results_preferred_when_context_shrinks(self) -> None:
        from reporag.retrieval.vector_search import RetrievalResult

        low = RetrievalResult(
            0.01, "low.py", 1, 40, None, "\n".join(f"l{i}" for i in range(40)), {}
        )
        high = RetrievalResult(
            0.99, "high.py", 1, 40, None, "\n".join(f"l{i}" for i in range(40)), {}
        )
        builder = PromptBuilder(max_tokens=600)
        built = builder.build_from_results(
            "Q", "simple-lookup", [low, high], context_max_tokens=4000
        )
        assert "high.py" in built.user

    def test_unknown_query_type_raises(self) -> None:
        builder = PromptBuilder(max_tokens=50_000)
        with pytest.raises(ValueError, match="query_type must be one of"):
            builder.build_from_results(
                "Q",
                "not-real",
                [],
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_context_string(self, builder: PromptBuilder) -> None:
        built = builder.build_prompt("Q", "simple-lookup", "")
        assert "no code retrieved" in built.user.lower()

    def test_context_with_a_single_chunk_containing_blank_lines(
        self, builder: PromptBuilder
    ) -> None:
        # A chunk's own internal blank line must not be mistaken for a
        # chunk boundary by the truncation splitter.
        body = "def f():\n\n\n    return 1"
        context = f"## a.py (lines 1-4)\n```python\n{body}\n```"
        built = builder.build_prompt("Q", "simple-lookup", context)
        assert "return 1" in built.user

    def test_query_containing_curly_braces_does_not_break_rendering(
        self, builder: PromptBuilder
    ) -> None:
        built = builder.build_prompt(
            "What does {'key': 'value'} do?", "simple-lookup", _SIMPLE_CONTEXT
        )
        assert "{'key': 'value'}" in built.user

    def test_two_word_query_type_values_are_the_only_valid_ones(
        self, builder: PromptBuilder
    ) -> None:
        for valid in ("simple-lookup", "multi-hop", "exploratory"):
            builder.build_prompt("Q", valid, _SIMPLE_CONTEXT)  # must not raise

    def test_context_with_no_headers_still_builds(self, builder: PromptBuilder) -> None:
        built = builder.build_prompt(
            "Q", "simple-lookup", "just some raw text, no headers"
        )
        assert "FILES IN CONTEXT" not in built.user

    def test_repeated_calls_are_deterministic(self, builder: PromptBuilder) -> None:
        first = builder.build_prompt("Q", "multi-hop", _SIMPLE_CONTEXT, {"s": "x"})
        second = builder.build_prompt("Q", "multi-hop", _SIMPLE_CONTEXT, {"s": "x"})
        assert first.system == second.system
        assert first.user == second.user

    def test_prior_findings_dict_with_non_string_step_id(
        self, builder: PromptBuilder
    ) -> None:
        # A caller might use int step indices; normalization should not choke.
        result = normalize_prior_findings({1: "finding one", 2: "finding two"})
        assert result == [("1", "", "finding one"), ("2", "", "finding two")]


# ---------------------------------------------------------------------------
# Query-type tolerance and classifier-object passthrough
# ---------------------------------------------------------------------------


class TestQueryTypeCoercion:
    @pytest.mark.parametrize(
        "spelling",
        [
            "multi_hop",
            "MULTI-HOP",
            "multihop",
            "  exploratory  ",
            "lookup",
            "simple_lookup",
            "simple",
        ],
    )
    def test_tolerant_spellings_are_accepted(
        self, builder: PromptBuilder, spelling: str
    ) -> None:
        builder.build_prompt("Q", spelling, "")  # must not raise

    def test_object_with_query_type_attribute_is_accepted(
        self, builder: PromptBuilder
    ) -> None:
        class _FakeClassification:
            query_type = "exploratory"

        built = builder.build_prompt("Q", _FakeClassification(), "")
        assert (
            "one short section per component" in built.system.lower()
            or "breadth over depth" in built.system.lower()
        )

    def test_query_type_defaults_to_multi_hop(self, builder: PromptBuilder) -> None:
        default_built = builder.build_prompt("Q", context="")
        explicit_built = builder.build_prompt("Q", "multi-hop", context="")
        assert default_built.system == explicit_built.system

    def test_unrecognized_spelling_still_raises(self, builder: PromptBuilder) -> None:
        with pytest.raises(ValueError, match="query_type must be one of"):
            builder.build_prompt("Q", "totally-unknown-type", "")


# ---------------------------------------------------------------------------
# Empty-query validation
# ---------------------------------------------------------------------------


class TestQueryValidation:
    @pytest.mark.parametrize("bad_query", ["", "   ", "\n\t"])
    def test_empty_or_whitespace_query_raises(
        self, builder: PromptBuilder, bad_query: str
    ) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            builder.build_prompt(bad_query, "simple-lookup", "")

    def test_build_from_results_also_validates_query(
        self, builder: PromptBuilder
    ) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            builder.build_from_results("  ", "simple-lookup", [])


# ---------------------------------------------------------------------------
# Truncation marker
# ---------------------------------------------------------------------------


class TestTruncationMarker:
    def test_marker_present_when_context_is_truncated(self) -> None:
        big_context = "\n\n".join(
            f"## src/mod_{i}.py (lines 1-3)\n```python\ndef f_{i}():\n    return {i}\n```"
            for i in range(80)
        )
        builder = PromptBuilder(max_tokens=700)
        built = builder.build_prompt("Q", "multi-hop", big_context)
        assert built.truncated is True
        assert "omitted to fit" in built.text
        assert built.token_count <= built.token_budget

    def test_marker_absent_when_context_fits_untouched(
        self, builder: PromptBuilder
    ) -> None:
        built = builder.build_prompt("Q", "simple-lookup", _SIMPLE_CONTEXT)
        assert built.truncated is False
        assert "omitted to fit" not in built.text

    def test_marker_reservation_never_pushes_result_over_budget(self) -> None:
        # A budget picked to land awkwardly close to a chunk boundary, to
        # stress the marker's reserved-cost accounting. Starts safely above
        # the non-droppable core's own cost (~404 tokens for this template
        # and query) so every budget tested is actually achievable --
        # below that, `fits_budget=False` is the correct, documented
        # outcome (see `test_impossible_budget_reports_fits_budget_false`),
        # not something this loop should assert against.
        for budget in range(450, 900, 37):
            big_context = "\n\n".join(
                f"## src/mod_{i}.py (lines 1-3)\n```python\ndef f_{i}():\n    return {i}\n```"
                for i in range(60)
            )
            built = PromptBuilder(max_tokens=budget).build_prompt(
                "Q", "multi-hop", big_context
            )
            assert built.token_count <= built.token_budget, budget

    def test_impossible_budget_reports_fits_budget_false(self) -> None:
        # A budget smaller than the non-droppable core itself (rules +
        # question, even with the context fully dropped) cannot be met --
        # `fits_budget=False` is the correct outcome here, not an
        # exception or an infinite loop.
        built = PromptBuilder(max_tokens=50).build_prompt(
            "Q", "multi-hop", "## a.py (lines 1-1)\n```python\nx=1\n```"
        )
        assert built.fits_budget is False
        assert built.token_count > built.token_budget


# ---------------------------------------------------------------------------
# Grouped file index
# ---------------------------------------------------------------------------


class TestGroupedFileIndex:
    def test_multiple_ranges_of_the_same_file_are_grouped(self) -> None:
        context = (
            "## a.py (lines 1-5)\n```python\nx\n```\n\n"
            "## a.py (lines 20-22)\n```python\ny\n```\n\n"
            "## b.py (lines 3-4)\n```python\nz\n```"
        )
        index = _extract_file_index(context)
        assert index.count("a.py") == 1
        assert "a.py (lines 1-5, 20-22)" in index
        assert "b.py (lines 3-4)" in index

    def test_duplicate_identical_ranges_are_not_repeated(self) -> None:
        context = (
            "## a.py (lines 1-5)\n```python\nx\n```\n\n"
            "## a.py (lines 1-5)\n```python\nx\n```"
        )
        index = _extract_file_index(context)
        assert index.count("1-5") == 1


# ---------------------------------------------------------------------------
# Assembler dependency injection
# ---------------------------------------------------------------------------


class TestAssemblerInjection:
    def test_injected_assembler_is_used(self) -> None:
        from reporag.generation.context_assembler import ContextAssembler
        from reporag.retrieval.vector_search import RetrievalResult

        injected = ContextAssembler(max_tokens=50)
        builder = PromptBuilder(max_tokens=50_000, assembler=injected)
        results = [
            RetrievalResult(0.9, "src/a.py", 1, 2, None, "def a():\n    return 1", {})
        ]
        built = builder.build_from_results("Q", "simple-lookup", results)
        assert "src/a.py" in built.text

    def test_no_injected_assembler_still_works(self, builder: PromptBuilder) -> None:
        from reporag.retrieval.vector_search import RetrievalResult

        results = [
            RetrievalResult(0.9, "src/a.py", 1, 2, None, "def a():\n    return 1", {})
        ]
        built = builder.build_from_results("Q", "simple-lookup", results)
        assert "src/a.py" in built.text


# ---------------------------------------------------------------------------
# BuiltPrompt.text / .sections / __str__
# ---------------------------------------------------------------------------


class TestBuiltPromptShape:
    def test_text_property_matches_system_plus_user(
        self, builder: PromptBuilder
    ) -> None:
        built = builder.build_prompt("Q", "multi-hop", _SIMPLE_CONTEXT)
        assert built.text == f"{built.system}\n\n{built.user}"

    def test_str_returns_text(self, builder: PromptBuilder) -> None:
        built = builder.build_prompt("Q", "multi-hop", _SIMPLE_CONTEXT)
        assert str(built) == built.text

    def test_sections_exposes_every_named_piece(self, builder: PromptBuilder) -> None:
        built = builder.build_prompt(
            "Q", "multi-hop", _SIMPLE_CONTEXT, {"s1": "a finding"}
        )
        assert set(built.sections) == {
            "role_and_rules",
            "examples",
            "file_index",
            "prior_findings",
            "context",
            "question",
        }
        assert "a finding" in built.sections["prior_findings"]
        assert "Q" in built.sections["question"]

    def test_sections_reflect_dropped_pieces_as_empty(self) -> None:
        big_context = "\n\n".join(
            f"## src/mod_{i}.py (lines 1-3)\n```python\ndef f_{i}():\n    return {i}\n```"
            for i in range(80)
        )
        built = PromptBuilder(max_tokens=700).build_prompt(
            "Q", "multi-hop", big_context, {"s1": "a finding"}
        )
        assert built.sections["examples"] == ""
        assert built.sections["prior_findings"] == ""
        assert built.sections["context"]  # truncated but not empty
