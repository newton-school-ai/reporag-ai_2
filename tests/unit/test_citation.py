"""Unit tests for the Issue 25 citation extractor.

Every test is offline: nothing here builds an LLM client, so there is no
network, no API key, and no model download.
"""

from __future__ import annotations

import pytest

from reporag.generation.citation import (
    CITATION_FORMAT,
    MIN_CITATION_COVERAGE,
    Citation,
    ContextIndex,
    SourceSpan,
    build_context_index,
    extract_citations,
    find_citation_markers,
    mask_code_blocks,
    split_claims,
)
from reporag.generation.context_assembler import ContextAssembler
from reporag.generation.prompt_builder import PromptBuilder
from reporag.retrieval.vector_search import RetrievalResult

ROUTE_CODE = (
    "def login_route(payload):\n    token = issue_token(payload)\n    return token"
)
SERVICE_CODE = (
    "def authenticate_user(email):\n    user = repo.get(email)\n    return user"
)


def make_result(
    file_path: str,
    start_line: int | None,
    end_line: int | None,
    text: str,
    score: float = 0.9,
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


@pytest.fixture
def results() -> list[RetrievalResult]:
    return [
        make_result("src/app/api/routes/auth.py", 20, 22, ROUTE_CODE, score=0.9),
        make_result("src/app/auth/service.py", 88, 90, SERVICE_CODE, score=0.8),
    ]


@pytest.fixture
def index(results: list[RetrievalResult]) -> ContextIndex:
    return ContextIndex.from_results(results)


# ---------------------------------------------------------------------------
# Marker parsing
# ---------------------------------------------------------------------------


def test_marker_matches_the_format_the_prompt_asks_for():
    # The extractor must parse exactly what Issue 24 instructs; the shared
    # constant is what keeps the two from drifting apart.
    assert CITATION_FORMAT == "[file_path:start_line-end_line]"
    markers = find_citation_markers("See [src/app/auth/service.py:88-90].")
    assert len(markers) == 1
    assert markers[0].raw_path == "src/app/auth/service.py"
    assert (markers[0].start_line, markers[0].end_line) == (88, 90)


def test_single_line_marker_is_a_one_line_range():
    (marker,) = find_citation_markers("Defined at [src/app/main.py:42].")
    assert (marker.start_line, marker.end_line) == (42, 42)


@pytest.mark.parametrize(
    "text",
    [
        "[src/a.py:L10-L20]",
        "[ src/a.py : 10 - 20 ]",
        "[src/a.py:10-20]",
    ],
)
def test_marker_spellings_are_tolerated(text: str):
    (marker,) = find_citation_markers(text)
    assert marker.raw_path == "src/a.py"
    assert (marker.start_line, marker.end_line) == (10, 20)


@pytest.mark.parametrize(
    "text",
    [
        "[Note: 3] is not a citation.",
        "[Step 2: 5] is not a citation.",
        "See the [docs](https://example.com/page:1) link.",
        "An empty bracket [] and a bare [42].",
    ],
)
def test_prose_in_brackets_is_not_a_citation(text: str):
    # Requiring a '.' or '/' in the path is what separates a file reference
    # from ordinary bracketed prose, without a list of known extensions.
    assert find_citation_markers(text) == []


def test_markers_carry_their_offsets_in_the_answer():
    text = "Head. [src/a.py:1-2] tail."
    (marker,) = find_citation_markers(text)
    assert text[marker.start_offset : marker.end_offset] == "[src/a.py:1-2]"


def test_several_markers_are_returned_in_order():
    markers = find_citation_markers("[src/a.py:1-2] then [src/b.py:3-4]")
    assert [m.raw_path for m in markers] == ["src/a.py", "src/b.py"]


@pytest.mark.parametrize("fence", ["```", "~~~", "```python"])
def test_markers_inside_fenced_code_are_ignored(fence: str):
    # A model demonstrating the citation format inside a code sample is not
    # citing the repository.
    text = f"Real [src/a.py:1-2].\n\n{fence}\n# [src/fake.py:9-9]\n```\n"
    assert [m.raw_path for m in find_citation_markers(text)] == ["src/a.py"]


def test_markers_in_inline_code_are_still_citations():
    # Backticks around a marker are formatting, not a code sample.
    assert len(find_citation_markers("as in `[src/a.py:1-2]`")) == 1


def test_masking_preserves_every_offset():
    text = "before\n```\nhidden\n```\nafter"
    masked = mask_code_blocks(text)
    assert len(masked) == len(text)
    assert masked.startswith("before")
    assert masked.endswith("after")
    assert "hidden" not in masked


# ---------------------------------------------------------------------------
# Building the index
# ---------------------------------------------------------------------------


def test_index_from_retrieval_results(index: ContextIndex):
    assert index.files == ("src/app/api/routes/auth.py", "src/app/auth/service.py")
    assert not index.is_empty
    assert index.covered_ranges("src/app/auth/service.py") == [(88, 90)]


def test_index_from_an_assembled_context(results: list[RetrievalResult]):
    # The assembled string is the form the prompt actually carried, so it
    # must index identically to the results it came from.
    context = ContextAssembler(max_tokens=4000).assemble(results)
    from_context = ContextIndex.from_context(context)
    assert set(from_context.files) == set(ContextIndex.from_results(results).files)
    assert from_context.covered_ranges("src/app/api/routes/auth.py") == [(20, 22)]
    assert from_context.spans_for("src/app/api/routes/auth.py")[0].text == ROUTE_CODE


def test_index_accepts_mappings_and_spans():
    index = ContextIndex.from_results(
        [
            {"file_path": "src/a.py", "start_line": "1", "end_line": "3", "text": "x"},
            SourceSpan("src/b.py", 5, 6, "y"),
        ]
    )
    assert index.files == ("src/a.py", "src/b.py")
    assert index.covered_ranges("src/a.py") == [(1, 3)]


def test_index_skips_chunks_without_a_file_path():
    index = ContextIndex.from_results([{"start_line": 1, "end_line": 2}, object()])
    assert index.is_empty


def test_index_normalizes_paths():
    index = ContextIndex.from_results([make_result("./src\\app/a.py", 1, 2, "x")])
    assert index.files == ("src/app/a.py",)


def test_adjacent_ranges_are_merged():
    # A file retrieved as 1-10 and 11-20 does contain lines 5-15, even
    # though no single chunk does.
    index = ContextIndex.from_results(
        [
            make_result("src/a.py", 1, 10, "a"),
            make_result("src/a.py", 11, 20, "b"),
            make_result("src/a.py", 40, 44, "c"),
        ]
    )
    assert index.covered_ranges("src/a.py") == [(1, 20), (40, 44)]


def test_unknown_line_bounds_are_indexed_without_ranges():
    index = ContextIndex.from_results([make_result("README.md", None, None, "docs")])
    assert index.files == ("README.md",)
    assert index.covered_ranges("README.md") == []


@pytest.mark.parametrize(
    "chunks_factory",
    [
        lambda results, context, prompt: results,
        lambda results, context, prompt: context,
        lambda results, context, prompt: prompt,
        lambda results, context, prompt: ContextIndex.from_results(results),
    ],
    ids=["results", "assembled-string", "built-prompt", "prepared-index"],
)
def test_build_context_index_accepts_every_shape(results, chunks_factory):
    context = ContextAssembler(max_tokens=4000).assemble(results)
    prompt = PromptBuilder(max_tokens=4000).build_prompt(
        "How does auth work?", "multi-hop", context
    )
    index = build_context_index(chunks_factory(results, context, prompt))
    assert "src/app/auth/service.py" in index.files


def test_build_context_index_on_nothing_is_empty():
    assert build_context_index(None).is_empty


def test_build_context_index_warns_on_something_unindexable(caplog):
    with caplog.at_level("WARNING"):
        assert build_context_index(42).is_empty
    assert "cannot index" in caplog.text


def test_index_repr_is_informative(index: ContextIndex):
    assert "files=2" in repr(index)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_a_real_citation_is_verified(index: ContextIndex):
    report = extract_citations(
        "Issues a token [src/app/api/routes/auth.py:20-22].", index
    )
    (citation,) = report.citations
    assert citation.verified
    assert citation.valid
    assert citation.file_path == "src/app/api/routes/auth.py"


def test_an_invented_file_is_flagged(index: ContextIndex):
    report = extract_citations("Hashing lives in [src/app/auth/hasher.py:5-9].", index)
    (citation,) = report.citations
    assert not citation.valid
    assert citation.status == "unknown-file"
    assert report.invalid_citations == report.citations


def test_a_real_file_with_invented_lines_is_flagged(index: ContextIndex):
    # The file is right, so a file-only check would pass this. The lines
    # were never retrieved, so the claim is still unsupported.
    report = extract_citations("See [src/app/auth/service.py:200-260].", index)
    (citation,) = report.citations
    assert citation.status == "line-range-outside-context"
    assert citation.snippet == ""


def test_a_range_spanning_two_adjacent_chunks_is_verified():
    index = ContextIndex.from_results(
        [make_result("src/a.py", 1, 10, "a"), make_result("src/a.py", 11, 20, "b")]
    )
    report = extract_citations("See [src/a.py:5-15].", index)
    assert report.citations[0].verified


def test_a_partial_path_is_resolved_to_the_retrieved_one(index: ContextIndex):
    report = extract_citations("See [auth/service.py:88-90].", index)
    (citation,) = report.citations
    assert citation.verified
    assert citation.file_path == "src/app/auth/service.py"
    assert citation.raw_path == "auth/service.py"
    assert citation.was_resolved


def test_a_bare_file_name_is_resolved_when_it_is_unambiguous(index: ContextIndex):
    (citation,) = extract_citations("See [service.py:88-90].", index).citations
    assert citation.file_path == "src/app/auth/service.py"


def test_an_ambiguous_file_name_is_not_guessed():
    index = ContextIndex.from_results(
        [
            make_result("src/a/util.py", 1, 2, "x"),
            make_result("src/b/util.py", 1, 2, "y"),
        ]
    )
    (citation,) = extract_citations("See [util.py:1-2].", index).citations
    assert citation.status == "ambiguous-file"
    assert not citation.valid


def test_an_inverted_range_is_invalid(index: ContextIndex):
    (citation,) = extract_citations(
        "See [src/app/auth/service.py:90-88].", index
    ).citations
    assert citation.status == "inverted-range"


def test_without_context_citations_are_unverified_rather_than_wrong():
    report = extract_citations("See [src/anything.py:1-2].", None)
    (citation,) = report.citations
    assert citation.status == "unverified"
    assert citation.valid
    assert not citation.verified
    assert report.context_available is False


def test_a_file_with_unknown_line_bounds_leaves_lines_unverified():
    index = ContextIndex.from_results([make_result("README.md", None, None, "docs")])
    (citation,) = extract_citations("See [README.md:1-4].", index).citations
    assert citation.status == "unverified"
    assert citation.valid


# ---------------------------------------------------------------------------
# Snippets
# ---------------------------------------------------------------------------


def test_snippet_is_exactly_the_cited_lines(index: ContextIndex):
    (citation,) = extract_citations(
        "See [src/app/auth/service.py:89-89].", index
    ).citations
    assert citation.snippet == "    user = repo.get(email)"


def test_snippet_covers_the_whole_range(index: ContextIndex):
    (citation,) = extract_citations(
        "See [src/app/auth/service.py:88-90].", index
    ).citations
    assert citation.snippet == SERVICE_CODE


def test_snippet_falls_back_to_the_whole_chunk_when_the_bounds_do_not_line_up():
    # Two declared lines, three lines of text: slicing would show the wrong
    # code, so the whole chunk is returned instead.
    index = ContextIndex.from_results([make_result("src/a.py", 1, 2, "a\nb\nc")])
    (citation,) = extract_citations("See [src/a.py:1-2].", index).citations
    assert citation.snippet == "a\nb\nc"


def test_citation_helpers():
    citation = Citation("src/a.py", 10, 12, snippet="x")
    assert citation.as_marker() == "[src/a.py:10-12]"
    assert str(citation) == "[src/a.py:10-12]"
    assert citation.line_count == 3
    assert citation.to_dict()["valid"] is True


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def test_sentences_become_claims():
    claims = split_claims(
        "The route issues a token. The service verifies the password."
    )
    assert len(claims) == 2
    assert claims[1].text.startswith("The service verifies")


def test_a_file_name_mid_sentence_does_not_split_it():
    # "auth.py handles" must not be read as the end of a sentence.
    claims = split_claims("The module auth.py handles the login flow for the API.")
    assert len(claims) == 1


def test_abbreviations_do_not_split_a_sentence():
    claims = split_claims("The router picks a strategy, e.g. BM25 for identifiers.")
    assert len(claims) == 1


def test_headings_rules_and_code_are_not_claims():
    text = (
        "# Overview\n\n"
        "---\n\n"
        "The route issues a token for the caller.\n\n"
        "```python\n"
        "this is code and asserts nothing\n"
        "```\n"
    )
    claims = split_claims(text)
    assert len(claims) == 1
    assert claims[0].text.startswith("The route")


def test_each_bullet_is_its_own_claim():
    claims = split_claims(
        "- The route accepts the payload.\n- The service verifies the password.\n"
    )
    assert len(claims) == 2
    assert not claims[0].text.startswith("-")


def test_a_bare_marker_is_not_a_claim():
    assert split_claims("[src/a.py:1-2]") == []


def test_claim_offsets_point_into_the_original_text():
    text = "  The route issues a token for the caller.\n"
    (claim,) = split_claims(text)
    assert text[claim.start_offset : claim.end_offset] == claim.text


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_a_fully_cited_answer_scores_one(index: ContextIndex):
    report = extract_citations(
        "The route issues a token [src/app/api/routes/auth.py:20-22]. "
        "The service loads the user [src/app/auth/service.py:88-90].",
        index,
    )
    assert report.coverage == 1.0
    assert report.meets_coverage_target


def test_a_marker_after_the_full_stop_still_supports_the_claim(index: ContextIndex):
    # Models write citations both inside and after the sentence; both count.
    report = extract_citations(
        "The route issues a token. [src/app/api/routes/auth.py:20-22]", index
    )
    assert report.coverage == 1.0


def test_an_uncited_claim_lowers_coverage(index: ContextIndex):
    report = extract_citations(
        "The route issues a token [src/app/api/routes/auth.py:20-22]. "
        "The password is hashed with bcrypt.",
        index,
    )
    assert report.coverage == pytest.approx(0.5)
    assert not report.meets_coverage_target
    assert len(report.uncited_claims) == 1


def test_an_invalid_citation_does_not_support_its_claim(index: ContextIndex):
    report = extract_citations(
        "The password is hashed with bcrypt [src/app/auth/hasher.py:5-9].", index
    )
    assert report.coverage == 0.0
    assert report.uncited_claims[0].has_citation


def test_an_answer_with_no_claims_scores_one(index: ContextIndex):
    assert extract_citations("", index).coverage == 1.0


def test_the_coverage_target_matches_the_issue():
    assert MIN_CITATION_COVERAGE == 0.9


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_repeated_citations_are_deduplicated(index: ContextIndex):
    text = (
        "The route issues a token [src/app/api/routes/auth.py:20-22]. "
        "It returns it to the caller [src/app/api/routes/auth.py:20-22]."
    )
    report = extract_citations(text, index)
    assert len(report.citations) == 2
    assert len(report.unique_citations) == 1


def test_cited_files_are_listed_in_first_cited_order(index: ContextIndex):
    report = extract_citations(
        "The service loads the user [src/app/auth/service.py:88-90]. "
        "The route issues a token [src/app/api/routes/auth.py:20-22].",
        index,
    )
    assert report.cited_files == (
        "src/app/auth/service.py",
        "src/app/api/routes/auth.py",
    )


def test_report_serializes_for_the_api(index: ContextIndex):
    report = extract_citations(
        "The route issues a token [src/app/api/routes/auth.py:20-22]. "
        "Hashing is in [src/app/auth/hasher.py:1-2].",
        index,
    )
    payload = report.to_dict()
    assert payload["citation_coverage"] == pytest.approx(0.5)
    assert payload["claim_count"] == 2
    assert len(payload["citations"]) == 1
    assert payload["invalid_citations"][0]["status"] == "unknown-file"


def test_report_repr_is_informative(index: ContextIndex):
    report = extract_citations(
        "Token issued [src/app/api/routes/auth.py:20-22].", index
    )
    assert "coverage=1.00" in repr(report)


# ---------------------------------------------------------------------------
# Integration with the rest of the pipeline
# ---------------------------------------------------------------------------


def test_a_chunk_dropped_from_the_prompt_cannot_be_cited(results):
    # The prompt builder trims the context to fit the window. A chunk the
    # model never saw must not validate just because retrieval found it.
    filler = make_result(
        "src/zzz/filler.py", 1, 400, "\n".join("x" * 40 for _ in range(400)), score=0.99
    )
    built = PromptBuilder(max_tokens=700).build_prompt(
        "How does auth work?",
        "multi-hop",
        ContextAssembler(max_tokens=20_000).assemble([*results, filler]),
    )
    assert built.truncated

    dropped = [
        result.file_path
        for result in [*results, filler]
        if result.file_path not in built.sections["context"]
    ]
    assert dropped, "expected the budget to drop at least one chunk"

    report = extract_citations(f"It is defined in [{dropped[0]}:1-2].", built)
    assert report.citations[0].status == "unknown-file"
