"""Unit tests for ContextAssembler (Issue 23)."""

from __future__ import annotations

import pytest

from src.reporag.generation.context_assembler import (
    ContextAssembler,
    _approximate_tokens,
    _Chunk,
    _format_chunk,
    _infer_language,
    _merge_chunks,
    _overlap_fraction,
)
from src.reporag.retrieval.vector_search import RetrievalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(
    file_path: str,
    start_line: int,
    end_line: int,
    chunk_text: str,
    score: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=None,
        chunk_text=chunk_text,
    )


# ---------------------------------------------------------------------------
# _infer_language
# ---------------------------------------------------------------------------


def test_infer_language_python() -> None:
    assert _infer_language("src/auth.py") == "python"


def test_infer_language_typescript() -> None:
    assert _infer_language("app/index.tsx") == "typescript"


def test_infer_language_unknown_extension() -> None:
    assert _infer_language("Makefile") == ""


# ---------------------------------------------------------------------------
# _overlap_fraction
# ---------------------------------------------------------------------------


def test_overlap_fraction_no_overlap() -> None:
    a = _Chunk("f.py", 1, 10, "a", 1.0)
    b = _Chunk("f.py", 20, 30, "b", 1.0)
    assert _overlap_fraction(a, b) == 0.0


def test_overlap_fraction_full_containment() -> None:
    outer = _Chunk("f.py", 1, 20, "outer", 1.0)
    inner = _Chunk("f.py", 5, 10, "inner", 1.0)
    # inner is fully contained; overlap = its entire span
    assert _overlap_fraction(outer, inner) == 1.0


def test_overlap_fraction_partial() -> None:
    a = _Chunk("f.py", 1, 10, "a", 1.0)
    b = _Chunk("f.py", 5, 15, "b", 1.0)
    # overlap lines: 5-10 = 5 lines; smaller span = min(9, 10) = 9 -> fraction ~= 0.56
    frac = _overlap_fraction(a, b)
    assert 0.5 < frac < 0.7


# ---------------------------------------------------------------------------
# _merge_chunks
# ---------------------------------------------------------------------------


def test_merge_chunks_no_overlap_different_files() -> None:
    a = _Chunk("a.py", 1, 5, "code a", 1.0)
    b = _Chunk("b.py", 1, 5, "code b", 1.0)
    # No merge -- different files
    merged = _merge_chunks([a, b])
    assert len(merged) == 2


def test_merge_chunks_adjacent_same_file_below_threshold() -> None:
    # Non-overlapping chunks in same file: should NOT merge
    a = _Chunk("f.py", 1, 5, "first block", 1.0)
    b = _Chunk("f.py", 10, 15, "second block", 1.0)
    merged = _merge_chunks([a, b])
    assert len(merged) == 2


def test_merge_chunks_overlapping_same_file() -> None:
    # Chunks a: lines 1-10, b: lines 3-15 -> overlap=7 lines, smaller span=9 -> 78% > 50%
    a = _Chunk("f.py", 1, 10, "\n".join(f"line {i}" for i in range(1, 11)), 0.9)
    b = _Chunk("f.py", 3, 15, "\n".join(f"line {i}" for i in range(3, 16)), 0.8)
    merged = _merge_chunks([a, b])
    assert len(merged) == 1
    m = merged[0]
    assert m.start_line == 1
    assert m.end_line == 15
    assert m.score == 0.9


def test_merge_chunks_fully_contained() -> None:
    outer = _Chunk("f.py", 1, 20, "outer code", 1.0)
    inner = _Chunk("f.py", 5, 10, "inner code", 0.5)
    merged = _merge_chunks([outer, inner])
    assert len(merged) == 1
    assert merged[0].start_line == 1
    assert merged[0].end_line == 20


# ---------------------------------------------------------------------------
# _format_chunk
# ---------------------------------------------------------------------------


def test_format_chunk_includes_header() -> None:
    chunk = _Chunk("src/auth.py", 10, 20, "def authenticate(): pass", 1.0, "python")
    text = _format_chunk(chunk)
    assert "## src/auth.py (lines 10-20)" in text
    assert "```python" in text
    assert "def authenticate(): pass" in text
    assert text.endswith("```")


def test_format_chunk_no_language() -> None:
    chunk = _Chunk("Makefile", 1, 5, "all: build", 1.0, "")
    text = _format_chunk(chunk)
    assert "```\n" in text


# ---------------------------------------------------------------------------
# ContextAssembler.assemble -- core behaviour
# ---------------------------------------------------------------------------


def test_assemble_empty_results() -> None:
    assembler = ContextAssembler()
    ctx = assembler.assemble([])
    assert ctx.text == ""
    assert ctx.chunk_count == 0
    assert not ctx.truncated


def test_assemble_skips_missing_line_metadata() -> None:
    r = RetrievalResult(
        score=0.9,
        file_path="src/auth.py",
        start_line=None,
        end_line=None,
        symbol_name=None,
        chunk_text="def foo(): pass",
    )
    assembler = ContextAssembler()
    ctx = assembler.assemble([r])
    assert ctx.text == ""
    assert ctx.chunk_count == 0


def test_assemble_single_chunk_reading_order() -> None:
    r = _result("src/auth.py", 5, 10, "def authenticate(): pass")
    assembler = ContextAssembler()
    ctx = assembler.assemble([r])
    assert "## src/auth.py (lines 5-10)" in ctx.text
    assert ctx.chunk_count == 1
    assert not ctx.truncated


def test_assemble_sorted_by_file_then_line() -> None:
    r1 = _result("src/b.py", 1, 5, "code b")
    r2 = _result("src/a.py", 10, 15, "code a2")
    r3 = _result("src/a.py", 1, 5, "code a1")
    assembler = ContextAssembler()
    ctx = assembler.assemble([r1, r2, r3])
    # src/a.py should come before src/b.py, and within a.py line 1 before 10
    a1_pos = ctx.text.find("## src/a.py (lines 1-5)")
    a2_pos = ctx.text.find("## src/a.py (lines 10-15)")
    b_pos = ctx.text.find("## src/b.py (lines 1-5)")
    assert a1_pos < a2_pos < b_pos


def test_assemble_overlapping_chunks_are_merged() -> None:
    """Overlapping chunks from the same file produce a single, non-redundant block."""
    lines = [f"line {i}" for i in range(1, 21)]
    code_a = "\n".join(lines[:10])  # lines 1-10
    code_b = "\n".join(lines[2:15])  # lines 3-15  (overlap = 7/9 ~= 78%)

    r_a = _result("src/auth.py", 1, 10, code_a, score=0.9)
    r_b = _result("src/auth.py", 3, 15, code_b, score=0.8)

    assembler = ContextAssembler()
    ctx = assembler.assemble([r_a, r_b])

    # Should be exactly one chunk after merging
    assert ctx.chunk_count == 1
    # Should span lines 1-15
    assert "## src/auth.py (lines 1-15)" in ctx.text


def test_assemble_non_overlapping_chunks_not_merged() -> None:
    r1 = _result("src/auth.py", 1, 5, "def foo(): pass")
    r2 = _result("src/auth.py", 50, 60, "def bar(): pass")
    assembler = ContextAssembler()
    ctx = assembler.assemble([r1, r2])
    assert ctx.chunk_count == 2


def test_assemble_highest_ranked_prioritised_on_truncation() -> None:
    """When budget is tight, highest-score chunks should be included first."""
    # Use a very small max_tokens so only one chunk fits.
    low_priority = _result("src/b.py", 1, 5, "B " * 50, score=0.2)
    high_priority = _result("src/a.py", 1, 5, "A " * 50, score=0.9)

    assembler = ContextAssembler(max_tokens=50)
    ctx = assembler.assemble([low_priority, high_priority])

    assert ctx.truncated
    # Only the high-priority chunk should be in the output
    assert "src/a.py" in ctx.text
    assert "src/b.py" not in ctx.text


def test_assemble_truncation_respects_max_tokens() -> None:
    results = [_result(f"src/f{i}.py", 1, 5, f"code{i} " * 200) for i in range(10)]
    assembler = ContextAssembler(max_tokens=200)
    ctx = assembler.assemble(results)
    assert ctx.total_tokens <= 200
    assert ctx.truncated


def test_assemble_each_chunk_has_file_and_line_prefix() -> None:
    r1 = _result("src/auth.py", 10, 20, "def authenticate(): pass")
    r2 = _result("src/session.py", 5, 8, "def get_session(): pass")
    assembler = ContextAssembler()
    ctx = assembler.assemble([r1, r2])
    assert "## src/auth.py (lines 10-20)" in ctx.text
    assert "## src/session.py (lines 5-8)" in ctx.text


# ---------------------------------------------------------------------------
# ContextAssembler validation
# ---------------------------------------------------------------------------


def test_assembler_rejects_invalid_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        ContextAssembler(max_tokens=0)


def test_assembler_rejects_invalid_overlap_threshold() -> None:
    with pytest.raises(ValueError, match="overlap_threshold"):
        ContextAssembler(overlap_threshold=1.5)


# ---------------------------------------------------------------------------
# AssembledContext attrs
# ---------------------------------------------------------------------------


def test_assembled_context_has_correct_chunk_count() -> None:
    r1 = _result("a.py", 1, 5, "code a")
    r2 = _result("b.py", 1, 5, "code b")
    ctx = ContextAssembler().assemble([r1, r2])
    assert ctx.chunk_count == 2


def test_approximate_tokens_nonzero() -> None:
    assert _approximate_tokens("hello world") >= 1
    assert _approximate_tokens("") >= 1
