"""Unit tests for line-level citation extraction (Issue 25).

Covers every acceptance criterion of the citation half of Issue 25:

* citation markers are extracted from an answer,
* citations are validated against retrieved context (containment, not
  exact-range matching, since the prompt tells the model to cite the
  narrowest supporting range),
* invalid (hallucinated, out-of-range, reversed) citations are flagged,
* citation coverage is computed as a documented heuristic,

plus edge cases: the "?" unknown-line-number placeholder, a file split
across multiple retrieved chunks, a `BuiltPrompt`-shaped input, and
citation-only trailing fragments that shouldn't be scored as separate,
uncited claims.

All tests are fully offline -- this module makes no LLM or network calls.
"""

from __future__ import annotations

import pytest

from reporag.generation.citation import (
    CitationReport,
    analyze_citations,
    compute_citation_coverage,
    extract_citations,
    validate_citations,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONTEXT = (
    "## src/app/api/routes/auth.py (lines 20-31)\n"
    "```python\n"
    "def login_route(payload):\n"
    "    user = authenticate_user(payload.email, payload.password)\n"
    "    return issue_token(user)\n"
    "```\n\n"
    "## src/app/auth/service.py (lines 88-92)\n"
    "```python\n"
    "def authenticate_user(email, password):\n"
    "    user = UserRepository().get_by_email(email)\n"
    "    return user if verify(password, user.hash) else None\n"
    "```"
)


class _FakeBuiltPrompt:
    """Duck-typed stand-in for `prompt_builder.BuiltPrompt`."""

    def __init__(self, context: str) -> None:
        self.sections = {"context": context}
        self.user = f"=== CODE CONTEXT ===\n{context}\n=== END CODE CONTEXT ==="


# ---------------------------------------------------------------------------
# Acceptance: extracts citation markers
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_extracts_a_single_citation(self) -> None:
        citations = extract_citations("It does X [src/a.py:10-20].")
        assert len(citations) == 1
        assert citations[0].file_path == "src/a.py"
        assert citations[0].start_line == 10
        assert citations[0].end_line == 20
        assert citations[0].raw == "[src/a.py:10-20]"
        assert citations[0].valid is None

    def test_extracts_multiple_citations_in_order(self) -> None:
        text = "First [a.py:1-2]. Second [b.py:3-4]. Third [c.py:5-6]."
        citations = extract_citations(text)
        assert [c.file_path for c in citations] == ["a.py", "b.py", "c.py"]

    def test_extracts_back_to_back_citations(self) -> None:
        citations = extract_citations("Combined claim [a.py:1-5][b.py:9-9].")
        assert len(citations) == 2
        assert citations[0].file_path == "a.py"
        assert citations[1].file_path == "b.py"

    def test_extracts_unknown_line_placeholder(self) -> None:
        citations = extract_citations("See the docs [README.md:?-?].")
        assert citations[0].start_line is None
        assert citations[0].end_line is None

    def test_no_citations_in_plain_text(self) -> None:
        assert extract_citations("No markers here at all.") == []

    def test_empty_text_returns_empty_list(self) -> None:
        assert extract_citations("") == []
        assert extract_citations(None) == []  # type: ignore[arg-type]

    def test_duplicate_citations_kept_as_separate_entries(self) -> None:
        text = "Claim one [a.py:1-2]. Claim two, same source [a.py:1-2]."
        citations = extract_citations(text)
        assert len(citations) == 2

    def test_malformed_marker_is_not_extracted(self) -> None:
        # Missing the dash, and missing brackets -- neither is a citation.
        assert extract_citations("Not a citation [a.py:10] or a.py:1-2.") == []

    def test_single_line_citation_start_equals_end(self) -> None:
        citations = extract_citations("See this line [a.py:5-5].")
        assert citations[0].start_line == 5
        assert citations[0].end_line == 5

    def test_negative_looking_line_number_is_not_extracted(self) -> None:
        # Real line numbers are never negative; a "-5" isn't a valid start,
        # so this is correctly not recognised as a citation at all.
        assert extract_citations("Bogus range [a.py:-5-10].") == []

    def test_extra_whitespace_inside_brackets_is_not_extracted(self) -> None:
        # The format is exact -- "[ file : 1 - 2 ]" is not a citation,
        # it's just text that happens to contain brackets.
        assert extract_citations("[ a.py : 1 - 2 ]") == []


# ---------------------------------------------------------------------------
# Acceptance: validates citations against retrieved context
# ---------------------------------------------------------------------------


class TestValidation:
    def test_exact_range_match_is_valid(self) -> None:
        citations = extract_citations("[src/app/auth/service.py:88-92]")
        validated = validate_citations(citations, CONTEXT)
        assert validated[0].valid is True
        assert "authenticate_user" in validated[0].snippet

    def test_narrower_range_within_a_shown_block_is_valid(self) -> None:
        """The prompt tells the model to cite the narrowest supporting
        range, so a citation narrower than what was shown must not be
        flagged as invalid."""
        citations = extract_citations("[src/app/auth/service.py:88-89]")
        validated = validate_citations(citations, CONTEXT)
        assert validated[0].valid is True
        assert validated[0].snippet == (
            "def authenticate_user(email, password):\n"
            "    user = UserRepository().get_by_email(email)"
        )

    def test_unretrieved_file_is_invalid(self) -> None:
        citations = extract_citations("[src/app/nope.py:1-5]")
        validated = validate_citations(citations, CONTEXT)
        assert validated[0].valid is False
        assert validated[0].snippet == ""

    def test_out_of_range_line_numbers_are_invalid(self) -> None:
        citations = extract_citations("[src/app/auth/service.py:200-210]")
        validated = validate_citations(citations, CONTEXT)
        assert validated[0].valid is False

    def test_reversed_range_is_invalid(self) -> None:
        citations = extract_citations("[src/app/auth/service.py:92-88]")
        validated = validate_citations(citations, CONTEXT)
        assert validated[0].valid is False

    def test_range_partially_overlapping_but_not_contained_is_invalid(self) -> None:
        # Starts inside the shown block but extends past its end.
        citations = extract_citations("[src/app/auth/service.py:90-100]")
        validated = validate_citations(citations, CONTEXT)
        assert validated[0].valid is False

    def test_unknown_placeholder_matches_only_an_identical_placeholder_block(
        self,
    ) -> None:
        context = "## README.md (lines ?-?)\n```\nSome docs\n```"
        citations = extract_citations("[README.md:?-?]")
        validated = validate_citations(citations, context)
        assert validated[0].valid is True
        assert validated[0].snippet == "Some docs"

    def test_numeric_citation_against_a_placeholder_block_is_invalid(self) -> None:
        context = "## README.md (lines ?-?)\n```\nSome docs\n```"
        citations = extract_citations("[README.md:1-2]")
        validated = validate_citations(citations, context)
        assert validated[0].valid is False

    def test_file_split_across_multiple_blocks_checks_all_of_them(self) -> None:
        context = (
            "## a.py (lines 1-10)\n```python\none to ten\n```\n\n"
            "## a.py (lines 50-60)\n```python\nfifty to sixty\n```"
        )
        citations = extract_citations("[a.py:52-55]")
        validated = validate_citations(citations, context)
        assert validated[0].valid is True

    def test_accepts_a_built_prompt_shaped_object(self) -> None:
        citations = extract_citations("[src/app/auth/service.py:88-92]")
        validated = validate_citations(citations, _FakeBuiltPrompt(CONTEXT))
        assert validated[0].valid is True

    def test_does_not_mutate_input_citations(self) -> None:
        citations = extract_citations("[src/app/auth/service.py:88-92]")
        original_valid = citations[0].valid
        validate_citations(citations, CONTEXT)
        assert citations[0].valid == original_valid  # still None, untouched

    def test_empty_context_invalidates_every_citation(self) -> None:
        citations = extract_citations("[a.py:1-2]")
        validated = validate_citations(citations, "")
        assert validated[0].valid is False

    def test_duplicate_identical_context_blocks_still_validate(self) -> None:
        # A file block that appears twice in context (e.g. surfaced by two
        # retrieval strategies) shouldn't confuse validation.
        context = (
            "## a.py (lines 1-5)\n```python\nx = 1\n```\n\n"
            "## a.py (lines 1-5)\n```python\nx = 1\n```"
        )
        citations = extract_citations("[a.py:1-5]")
        validated = validate_citations(citations, context)
        assert validated[0].valid is True

    def test_object_missing_both_sections_and_user_falls_back_to_empty(
        self,
    ) -> None:
        class _BareObject:
            pass

        citations = extract_citations("[a.py:1-2]")
        validated = validate_citations(citations, _BareObject())
        assert validated[0].valid is False


# ---------------------------------------------------------------------------
# Acceptance: citation coverage
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_fully_cited_answer_has_full_coverage(self) -> None:
        text = (
            "This claim is cited properly right here [a.py:1-5]. "
            "This other claim is also cited over here [b.py:1-5]."
        )
        assert compute_citation_coverage(text) == 1.0

    def test_fully_uncited_answer_has_zero_coverage(self) -> None:
        text = (
            "This is a claim with no citation attached to it whatsoever. "
            "This is another claim, also with nothing backing it at all."
        )
        assert compute_citation_coverage(text) == 0.0

    def test_partial_coverage_is_a_fraction(self) -> None:
        text = (
            "This claim has a citation right here [a.py:1-5]. "
            "This other claim has no citation attached to it at all."
        )
        assert compute_citation_coverage(text) == 0.5

    def test_empty_text_is_vacuously_fully_covered(self) -> None:
        assert compute_citation_coverage("") == 1.0

    def test_only_trivial_fragments_is_vacuously_fully_covered(self) -> None:
        assert compute_citation_coverage("1. 2. 3.") == 1.0

    def test_trailing_citation_only_fragment_is_not_a_separate_uncited_claim(
        self,
    ) -> None:
        """ "Claim. [a.py:1-5]." must be scored as one cited claim, not as a
        cited claim plus a separate short "claim" that looks uncited."""
        text = "This is a real, fully formed claim about the code. [a.py:1-5]."
        assert compute_citation_coverage(text) == 1.0

    def test_multiple_trailing_citations_still_count_as_one_claim(self) -> None:
        text = (
            "This is a real, fully formed claim about the code. "
            "[a.py:1-5][b.py:9-9]."
        )
        assert compute_citation_coverage(text) == 1.0

    def test_bullet_list_with_no_terminal_punctuation_splits_per_item(self) -> None:
        """A regression test: without newline-based splitting on list-item
        boundaries, this whole 3-bullet list reads as one run-on claim, and
        the single citation on the first bullet falsely covers all three."""
        text = (
            "- Auth layer handles login [a.py:1-5]\n"
            "- Storage layer persists data with no citation at all here\n"
            "- Both layers work together with no citation either"
        )
        assert compute_citation_coverage(text) == pytest.approx(1 / 3)

    def test_numbered_list_does_not_misattribute_citations_across_items(
        self,
    ) -> None:
        """A citation on item 2 must not be counted as covering items 1 or
        3, which have none of their own."""
        text = (
            "1. Hop one happens first with absolutely no citation here.\n"
            "2. Hop two happens next and is properly cited [x.py:1-2].\n"
            "3. Hop three happens last with no citation attached either."
        )
        assert compute_citation_coverage(text) == pytest.approx(1 / 3)

    def test_asterisk_and_unicode_bullet_markers_also_split(self) -> None:
        text = (
            "* Point one, cited properly right here [a.py:1-2]\n"
            "\u2022 Point two, with no citation attached at all here"
        )
        assert compute_citation_coverage(text) == 0.5

    def test_fully_cited_numbered_list_has_full_coverage(self) -> None:
        text = (
            "1. First hop happens here and is cited properly [a.py:1-2].\n"
            "2. Second hop happens here and is cited properly [b.py:3-4]."
        )
        assert compute_citation_coverage(text) == 1.0


# ---------------------------------------------------------------------------
# Acceptance: structured end-to-end result
# ---------------------------------------------------------------------------


class TestAnalyzeCitations:
    def test_full_report_shape(self) -> None:
        answer = (
            "login_route calls authenticate_user "
            "[src/app/api/routes/auth.py:20-24]. "
            "It also invents a source [src/app/nope.py:1-2]."
        )
        report = analyze_citations(answer, CONTEXT)
        assert isinstance(report, CitationReport)
        assert len(report.citations) == 2
        assert report.valid_count == 1
        assert report.invalid_count == 1
        assert report.all_valid is False
        assert report.coverage == 1.0

    def test_report_with_no_citations_at_all(self) -> None:
        report = analyze_citations("Totally uncited answer text here.", CONTEXT)
        assert report.citations == []
        assert report.valid_count == 0
        assert report.invalid_count == 0
        assert report.all_valid is True  # vacuously -- see CitationReport.all_valid
        assert report.coverage == 0.0

    def test_all_valid_true_when_every_citation_passes(self) -> None:
        answer = "login_route [src/app/api/routes/auth.py:20-24]."
        report = analyze_citations(answer, CONTEXT)
        assert report.all_valid is True

    def test_accepts_a_built_prompt_shaped_object(self) -> None:
        answer = "login_route [src/app/api/routes/auth.py:20-24]."
        report = analyze_citations(answer, _FakeBuiltPrompt(CONTEXT))
        assert report.valid_count == 1
