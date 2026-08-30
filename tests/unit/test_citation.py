"""Unit tests for CitationExtractor (Issue 25)."""

from __future__ import annotations

from reporag.generation.citation import CitationExtractor
from reporag.retrieval.vector_search import RetrievalResult


def test_parse_citations_single_and_range() -> None:
    extractor = CitationExtractor()
    text = "Check authentication in [src/auth.py:10-25] and session in [src/session.py:42]."
    parsed = extractor.parse_citations(text)

    assert len(parsed) == 2
    assert parsed[0] == ("src/auth.py", 10, 25)
    assert parsed[1] == ("src/session.py", 42, 42)


def test_extract_and_validate_valid_citation() -> None:
    extractor = CitationExtractor()
    context = [
        RetrievalResult(
            score=0.9,
            file_path="src/auth.py",
            start_line=10,
            end_line=30,
            symbol_name=None,
            chunk_text="def authenticate(user, pwd):\n    # line 11\n    return True\n",
            metadata={},
        )
    ]
    response = "The function validates credentials [src/auth.py:10-15]."
    result = extractor.extract_and_validate(response, context)

    assert len(result.citations) == 1
    c = result.citations[0]
    assert c.file_path == "src/auth.py"
    assert c.start_line == 10
    assert c.end_line == 15
    assert c.valid is True
    assert "def authenticate" in c.snippet
    assert result.citation_coverage == 1.0


def test_extract_and_validate_invalid_file() -> None:
    extractor = CitationExtractor()
    context = [
        RetrievalResult(
            score=0.9,
            file_path="src/auth.py",
            start_line=10,
            end_line=30,
            symbol_name=None,
            chunk_text="def auth(): pass",
            metadata={},
        )
    ]
    response = "See database config [src/db/config.py:10-20]."
    result = extractor.extract_and_validate(response, context)

    assert len(result.citations) == 1
    assert result.citations[0].valid is False
    assert len(result.invalid_citations) == 1
    assert result.citation_coverage == 0.0


def test_extract_and_validate_out_of_bounds_lines() -> None:
    extractor = CitationExtractor()
    context = [
        RetrievalResult(
            score=0.9,
            file_path="src/auth.py",
            start_line=10,
            end_line=20,
            symbol_name=None,
            chunk_text="def auth(): pass",
            metadata={},
        )
    ]
    # File matches, but lines 100-120 do not overlap with 10-20
    response = "Details at [src/auth.py:100-120]."
    result = extractor.extract_and_validate(response, context)

    assert len(result.citations) == 1
    assert result.citations[0].valid is False
    assert result.citation_coverage == 0.0


def test_extract_and_validate_formatted_context_string() -> None:
    extractor = CitationExtractor()
    context_str = (
        "## src/models.py (lines 5-15)\n"
        "```python\n"
        "class User:\n"
        "    id: int\n"
        "```"
    )
    response = "User model defined here [src/models.py:5-10]."
    result = extractor.extract_and_validate(response, context_str)

    assert len(result.citations) == 1
    assert result.citations[0].valid is True
    assert result.citation_coverage == 1.0


def test_empty_citations_coverage() -> None:
    extractor = CitationExtractor()
    result = extractor.extract_and_validate("Response with no citation markers.", None)

    assert len(result.citations) == 0
    assert result.citation_coverage == 1.0
