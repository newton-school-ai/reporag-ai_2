"""Unit tests for citation extractor and coverage computation."""

import pytest

from reporag.generation.citation import (
    CitationExtractor,
    compute_citation_coverage,
)
from reporag.retrieval.vector_search import RetrievalResult


def make_chunk(
    file_path: str,
    start_line: int | None,
    end_line: int | None,
    chunk_text: str,
    score: float = 0.9,
) -> RetrievalResult:
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=None,
        chunk_text=chunk_text,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Extraction and pattern matching
# ---------------------------------------------------------------------------


def test_extract_single_valid_citation():
    extractor = CitationExtractor()
    text = "The user model has an email field [src/models/user.py:10-15]."
    lines = [f"line_{i}" for i in range(1, 30)]
    chunk = make_chunk("src/models/user.py", 1, 29, "\n".join(lines))

    citations = extractor.extract(text, [chunk])
    assert len(citations) == 1
    c = citations[0]
    assert c.file_path == "src/models/user.py"
    assert c.start_line == 10
    assert c.end_line == 15
    assert c.valid is True
    assert "line_10" in c.snippet
    assert "line_15" in c.snippet


def test_extract_multiple_citations_in_one_text():
    extractor = CitationExtractor()
    text = (
        "First claim [src/auth/service.py:10-20]. "
        "Second claim [src/api/routes.py:5-15] and also [src/auth/service.py:25-30]."
    )
    chunk1 = make_chunk(
        "src/auth/service.py",
        1,
        40,
        "\n".join(f"auth_{i}" for i in range(1, 41)),
    )
    chunk2 = make_chunk(
        "src/api/routes.py", 1, 20, "\n".join(f"route_{i}" for i in range(1, 21))
    )

    citations = extractor.extract(text, [chunk1, chunk2])
    assert len(citations) == 3
    assert all(c.valid for c in citations)
    assert [c.file_path for c in citations] == [
        "src/auth/service.py",
        "src/api/routes.py",
        "src/auth/service.py",
    ]


def test_extract_inverted_line_numbers():
    extractor = CitationExtractor()
    text = "Citation with inverted lines [src/models/user.py:20-10]."
    chunk = make_chunk(
        "src/models/user.py", 1, 30, "\n".join(f"L{i}" for i in range(1, 31))
    )

    citations = extractor.extract(text, [chunk])
    assert len(citations) == 1
    assert citations[0].start_line == 10
    assert citations[0].end_line == 20
    assert citations[0].valid is True


def test_extract_no_citations():
    extractor = CitationExtractor()
    text = "A simple answer with no citations anywhere in the text."
    citations = extractor.extract(text, [])
    assert citations == []


@pytest.mark.parametrize(
    "malformed",
    [
        "Invalid marker [src/file.py]",
        "Invalid line format [src/file.py:abc-def]",
        "Single line not range [src/file.py:10]",
        "Missing brackets src/file.py:10-20",
    ],
)
def test_extract_malformed_markers_ignored(malformed: str):
    extractor = CitationExtractor()
    citations = extractor.extract(malformed, [])
    assert citations == []


# ---------------------------------------------------------------------------
# Validation rules (overlaps, hallucinated files, boundaries)
# ---------------------------------------------------------------------------


def test_hallucinated_file_marked_invalid():
    extractor = CitationExtractor()
    text = "According to fake module [src/nonexistent/module.py:10-20]."
    chunk = make_chunk("src/real/module.py", 1, 30, "def real(): pass")

    citations = extractor.extract(text, [chunk])
    assert len(citations) == 1
    assert citations[0].valid is False
    assert citations[0].snippet == ""


def test_out_of_range_lines_marked_invalid():
    extractor = CitationExtractor()
    text = "Out of range lines [src/real/module.py:50-60]."
    chunk = make_chunk(
        "src/real/module.py", 1, 20, "\n".join(f"L{i}" for i in range(1, 21))
    )

    citations = extractor.extract(text, [chunk])
    assert len(citations) == 1
    assert citations[0].valid is False
    assert citations[0].snippet == ""


def test_partial_overlap_at_start():
    extractor = CitationExtractor()
    text = "Partially overlapping start [src/file.py:5-15]."
    chunk = make_chunk("src/file.py", 10, 25, "\n".join(f"L{i}" for i in range(10, 26)))

    citations = extractor.extract(text, [chunk])
    assert len(citations) == 1
    assert citations[0].valid is True


def test_partial_overlap_at_end():
    extractor = CitationExtractor()
    text = "Partially overlapping end [src/file.py:20-30]."
    chunk = make_chunk("src/file.py", 10, 25, "\n".join(f"L{i}" for i in range(10, 26)))

    citations = extractor.extract(text, [chunk])
    assert len(citations) == 1
    assert citations[0].valid is True


def test_multiple_chunks_same_file_matches_correct_one():
    extractor = CitationExtractor()
    text = "Claim in second chunk [src/file.py:50-60]."
    chunk1 = make_chunk("src/file.py", 1, 20, "\n".join(f"L{i}" for i in range(1, 21)))
    chunk2 = make_chunk(
        "src/file.py", 45, 70, "\n".join(f"L{i}" for i in range(45, 71))
    )

    citations = extractor.extract(text, [chunk1, chunk2])
    assert len(citations) == 1
    assert citations[0].valid is True
    assert "L50" in citations[0].snippet


def test_chunk_with_none_line_numbers_marks_valid():
    extractor = CitationExtractor()
    text = "Claim from markdown doc [docs/architecture.md:1-10]."
    chunk = make_chunk("docs/architecture.md", None, None, "# Architecture Overview")

    citations = extractor.extract(text, [chunk])
    assert len(citations) == 1
    assert citations[0].valid is True
    assert citations[0].snippet == "# Architecture Overview"


# ---------------------------------------------------------------------------
# Citation Coverage Metric
# ---------------------------------------------------------------------------


def test_compute_citation_coverage_all_cited():
    text = (
        "User auth is defined in auth.py [src/auth.py:1-5]. "
        "Token verification is in token.py [src/token.py:10-20]."
    )
    extractor = CitationExtractor()
    citations = extractor.extract(text, [])
    coverage = compute_citation_coverage(text, citations)
    assert coverage == 1.0


def test_compute_citation_coverage_partially_cited():
    text = (
        "User auth is defined here [src/auth.py:1-5]. But this claim has no citation."
    )
    extractor = CitationExtractor()
    citations = extractor.extract(text, [])
    coverage = compute_citation_coverage(text, citations)
    assert 0.4 <= coverage <= 0.6


def test_compute_citation_coverage_zero_cited():
    text = "Claim one without citation. Claim two without citation."
    extractor = CitationExtractor()
    citations = extractor.extract(text, [])
    coverage = compute_citation_coverage(text, citations)
    assert coverage == 0.0


def test_compute_citation_coverage_empty_or_whitespace():
    assert compute_citation_coverage("", []) == 0.0
    assert compute_citation_coverage("   \n\t  ", []) == 0.0


def test_compute_citation_coverage_ignores_code_blocks_and_headers():
    text = (
        "# Main Heading\n"
        "Here is the cited statement [src/main.py:1-5].\n"
        "```python\ndef foo(): pass\n```\n"
    )
    extractor = CitationExtractor()
    citations = extractor.extract(text, [])
    coverage = compute_citation_coverage(text, citations)
    assert coverage == 1.0
