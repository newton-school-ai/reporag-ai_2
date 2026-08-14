"""Unit tests for ContextAssembler (Issue 23).

Covers every acceptance criterion R1-R10 from the requirements document.

Test classes are organised by requirement so it is easy to trace a failing
test back to the relevant requirement.  A final integration smoke test
verifies the end-to-end output shape with multiple files.
"""

from __future__ import annotations

import pytest

from reporag.generation.context_assembler import AssembledContext, ContextAssembler
from reporag.ingestion.chunker import count_tokens
from reporag.retrieval.vector_search import RetrievalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r(
    file_path: str,
    start_line: int,
    end_line: int,
    chunk_text: str,
    score: float = 1.0,
    language: str = "python",
) -> RetrievalResult:
    """Build a minimal RetrievalResult for tests.

    Mirrors the helper pattern from test_fusion.py so test code stays concise.
    """
    return RetrievalResult(
        score=score,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=None,
        chunk_text=chunk_text,
        metadata={"language": language},
    )


# ---------------------------------------------------------------------------
# R1 -- Ordering
# ---------------------------------------------------------------------------


class TestOrdering:
    """R1: output sections ordered by file_path (lexicographic), then start_line."""

    def test_different_files_appear_in_alphabetical_order(self) -> None:
        """Two results from different files must appear file-alphabetically."""
        assembler = ContextAssembler(max_tokens=4000)
        results = [
            _r("z_last.py", 1, 5, "def z(): pass", score=0.9),
            _r("a_first.py", 1, 5, "def a(): pass", score=0.8),
        ]
        ctx = assembler.assemble(results)

        z_pos = ctx.text.index("z_last.py")
        a_pos = ctx.text.index("a_first.py")
        assert a_pos < z_pos, "a_first.py should appear before z_last.py"

    def test_same_file_chunks_appear_in_start_line_order(self) -> None:
        """Two chunks from the same file must appear in start_line order."""
        assembler = ContextAssembler(max_tokens=4000)
        # Deliberately provide higher-scored chunk with larger start_line first
        results = [
            _r("auth.py", 50, 55, "def logout(): pass", score=0.9),
            _r("auth.py", 1, 5, "def login(): pass", score=0.5),
        ]
        ctx = assembler.assemble(results)

        # Both chunks end up in the same section (auth.py).
        # Because overlap is small (no overlap at all), they should be
        # separate blocks. Line 1 block must precede line 50 block.
        assert ctx.text.index("login") < ctx.text.index("logout")

    def test_multiple_files_all_in_alphabetical_order(self) -> None:
        """Three files appear in lexicographic order."""
        assembler = ContextAssembler(max_tokens=4000)
        results = [
            _r("c.py", 1, 3, "c code", score=0.9),
            _r("a.py", 1, 3, "a code", score=0.8),
            _r("b.py", 1, 3, "b code", score=0.7),
        ]
        ctx = assembler.assemble(results)

        a_pos = ctx.text.index("a.py")
        b_pos = ctx.text.index("b.py")
        c_pos = ctx.text.index("c.py")
        assert a_pos < b_pos < c_pos


# ---------------------------------------------------------------------------
# R2 -- Overlap deduplication
# ---------------------------------------------------------------------------


class TestOverlapMerge:
    """R2: chunks overlapping by > 50% of the shorter chunk are merged."""

    def test_exact_same_range_merges_to_one_block(self) -> None:
        """Two results with identical (file, start, end) merge to one section."""
        assembler = ContextAssembler(max_tokens=4000)
        text = "\n".join(f"line {i}" for i in range(1, 11))  # 10 lines
        results = [
            _r("mod.py", 1, 10, text, score=0.9),
            _r("mod.py", 1, 10, text, score=0.8),
        ]
        ctx = assembler.assemble(results)

        # Only one header for mod.py should be present
        assert ctx.text.count("## mod.py") == 1

    def test_partial_overlap_above_threshold_merges(self) -> None:
        """Lines 10-20 and 14-25: overlap=7, shorter=11, ratio~=0.64 > 0.5 -> merge."""
        assembler = ContextAssembler(max_tokens=4000)
        text_a = "\n".join(f"line {i}" for i in range(10, 21))  # lines 10-20
        text_b = "\n".join(f"line {i}" for i in range(14, 26))  # lines 14-25
        results = [
            _r("src/util.py", 10, 20, text_a, score=0.9),
            _r("src/util.py", 14, 25, text_b, score=0.8),
        ]
        ctx = assembler.assemble(results)

        # Merged block must span lines 10-25
        assert "(lines 10-25)" in ctx.text
        # Only one code block for this file
        assert ctx.text.count("## src/util.py") == 1

    def test_small_overlap_below_threshold_stays_separate(self) -> None:
        """Lines 1-10 and 9-20: overlap=2, shorter=10, ratio=0.2 <= 0.5 -> no merge."""
        assembler = ContextAssembler(max_tokens=4000)
        text_a = "\n".join(f"line {i}" for i in range(1, 11))  # lines 1-10
        text_b = "\n".join(f"line {i}" for i in range(9, 21))  # lines 9-20
        results = [
            _r("src/main.py", 1, 10, text_a, score=0.9),
            _r("src/main.py", 9, 20, text_b, score=0.8),
        ]
        ctx = assembler.assemble(results)

        # Two separate blocks, each with its own range header
        assert "(lines 1-10)" in ctx.text
        assert "(lines 9-20)" in ctx.text

    def test_no_overlap_stays_separate(self) -> None:
        """Non-overlapping chunks in the same file stay as separate blocks."""
        assembler = ContextAssembler(max_tokens=4000)
        results = [
            _r("pkg/a.py", 1, 5, "def foo(): pass", score=0.9),
            _r("pkg/a.py", 50, 55, "def bar(): pass", score=0.8),
        ]
        ctx = assembler.assemble(results)

        assert "(lines 1-5)" in ctx.text
        assert "(lines 50-55)" in ctx.text


# ---------------------------------------------------------------------------
# R3 -- Formatted output
# ---------------------------------------------------------------------------


class TestFormattedOutput:
    """R3: header format and line-numbered body."""

    def test_header_format(self) -> None:
        """Header must be '## src/auth.py (lines 10-15)'."""
        assembler = ContextAssembler(max_tokens=4000)
        results = [
            _r("src/auth.py", 10, 15, "\n".join(f"code {i}" for i in range(10, 16))),
        ]
        ctx = assembler.assemble(results)

        assert "## src/auth.py (lines 10-15)" in ctx.text

    def test_line_number_format(self) -> None:
        """Each line prefixed with its actual source line number and ' | '."""
        assembler = ContextAssembler(max_tokens=4000)
        results = [
            _r("src/auth.py", 10, 12, "def auth():\n    pass\n    return"),
        ]
        ctx = assembler.assemble(results)

        assert "10 | def auth():" in ctx.text
        assert "11 |     pass" in ctx.text
        assert "12 |     return" in ctx.text

    def test_language_in_fenced_block(self) -> None:
        """Fenced code block uses the language from metadata."""
        assembler = ContextAssembler(max_tokens=4000)
        results = [
            _r(
                "main.ts",
                1,
                3,
                "const x = 1;\nconst y = 2;\nconst z = 3;",
                language="typescript",
            ),
        ]
        ctx = assembler.assemble(results)

        assert "```typescript" in ctx.text

    def test_no_line_numbers_when_start_line_is_zero(self) -> None:
        """When start_line is 0 (or None treated as 0), omit line numbers and range header."""
        assembler = ContextAssembler(max_tokens=4000)
        r = RetrievalResult(
            score=1.0,
            file_path="unknown.py",
            start_line=None,
            end_line=None,
            symbol_name=None,
            chunk_text="some code here",
            metadata={"language": "python"},
        )
        ctx = assembler.assemble([r])

        # No "(lines ..." in the header
        assert "(lines" not in ctx.text
        # The file header should still appear
        assert "## unknown.py" in ctx.text
        # Content is still present
        assert "some code here" in ctx.text

    def test_files_separated_by_horizontal_rule(self) -> None:
        """Different files must be separated by '\\n---\\n'."""
        assembler = ContextAssembler(max_tokens=4000)
        results = [
            _r("a.py", 1, 2, "def a(): pass"),
            _r("b.py", 1, 2, "def b(): pass"),
        ]
        ctx = assembler.assemble(results)

        assert "\n---\n" in ctx.text

    def test_blocks_within_same_file_separated_by_blank_line(self) -> None:
        """Two non-merged blocks in the same file are separated by a blank line."""
        assembler = ContextAssembler(max_tokens=4000)
        results = [
            _r("mod.py", 1, 5, "def foo(): pass"),
            _r("mod.py", 50, 55, "def bar(): pass"),
        ]
        ctx = assembler.assemble(results)

        # No '---' separator (same file), but a blank line between blocks
        assert "\n---\n" not in ctx.text
        assert "\n\n" in ctx.text


# ---------------------------------------------------------------------------
# R4 -- Token budget
# ---------------------------------------------------------------------------


class TestTokenBudget:
    """R4: chunks beyond the budget are excluded; dropped count is correct."""

    def test_dropped_count_is_correct(self) -> None:
        """With 3 chunks and budget for only 2, dropped must equal 1."""
        # Each chunk_text is "x " * N; pick sizes so exactly 2 fit.
        chunk_a = "alpha " * 20  # ~20 tokens
        chunk_b = "beta " * 20  # ~20 tokens
        chunk_c = "gamma " * 20  # ~20 tokens

        tok_a = count_tokens(chunk_a)
        tok_b = count_tokens(chunk_b)
        # budget allows a + b but not c
        budget = tok_a + tok_b + tok_a // 2  # comfortably fits a and b, not c

        assembler = ContextAssembler(max_tokens=budget)
        results = [
            _r("f.py", 1, 5, chunk_a, score=0.9),
            _r("f.py", 6, 10, chunk_b, score=0.8),
            _r("f.py", 11, 15, chunk_c, score=0.7),
        ]
        ctx = assembler.assemble(results)

        assert ctx.dropped == 1
        assert ctx.included == 2

    def test_included_plus_dropped_equals_non_empty_results(self) -> None:
        """included + dropped always equals the number of non-empty results."""
        chunk = "word " * 50
        tok = count_tokens(chunk)
        budget = tok * 2 + tok // 2  # fits exactly 2

        assembler = ContextAssembler(max_tokens=budget)
        results = [_r("f.py", i, i + 1, chunk, score=1.0 / (i + 1)) for i in range(5)]
        ctx = assembler.assemble(results)

        assert ctx.included + ctx.dropped == len(results)


# ---------------------------------------------------------------------------
# R5 -- Priority: highest-ranked first
# ---------------------------------------------------------------------------


class TestPriority:
    """R5: highest-score chunk always included when budget is tight."""

    def test_highest_score_chunk_included_over_lower(self) -> None:
        """When budget allows only 1 chunk, the first (highest-score) chunk wins."""
        chunk_high = "important function code " * 5
        chunk_low = "less relevant code " * 5

        tok_high = count_tokens(chunk_high)
        tok_low = count_tokens(chunk_low)
        # Budget only fits one of them
        budget = max(tok_high, tok_low) + 1

        assembler = ContextAssembler(max_tokens=budget)
        results = [
            _r("src/core.py", 1, 5, chunk_high, score=0.95),  # rank 0 -- best
            _r("src/util.py", 1, 5, chunk_low, score=0.30),  # rank 1 -- lower
        ]
        ctx = assembler.assemble(results)

        assert ctx.included == 1
        assert ctx.dropped == 1
        assert "important function code" in ctx.text
        assert "less relevant code" not in ctx.text


# ---------------------------------------------------------------------------
# R6 -- Minimum one chunk
# ---------------------------------------------------------------------------


class TestMinimumOneChunk:
    """R6: single over-budget chunk is always included."""

    def test_over_budget_single_chunk_still_included(self) -> None:
        """A chunk larger than max_tokens is included anyway (1-chunk guarantee)."""
        big_chunk = "very long code line\n" * 200  # definitely > 5 tokens
        assembler = ContextAssembler(max_tokens=5)
        results = [_r("big.py", 1, 200, big_chunk)]
        ctx = assembler.assemble(results)

        assert ctx.included == 1
        assert ctx.dropped == 0
        assert "very long code line" in ctx.text

    def test_over_budget_first_chunk_second_dropped(self) -> None:
        """Over-budget first chunk is included; second is dropped."""
        big_chunk = "very long code line\n" * 200
        small_chunk = "small"
        tok_big = count_tokens(big_chunk)
        assembler = ContextAssembler(max_tokens=tok_big // 2)
        results = [
            _r("big.py", 1, 200, big_chunk, score=0.9),
            _r("small.py", 1, 1, small_chunk, score=0.5),
        ]
        ctx = assembler.assemble(results)

        assert ctx.included == 1
        assert ctx.dropped == 1


# ---------------------------------------------------------------------------
# R7 -- AssembledContext metadata
# ---------------------------------------------------------------------------


class TestAssembledContextMetadata:
    """R7: all fields of AssembledContext are correct types and values."""

    def test_all_fields_present_and_typed(self) -> None:
        """AssembledContext carries text, token_count, included, dropped, files_covered."""
        assembler = ContextAssembler(max_tokens=4000)
        results = [
            _r("src/auth.py", 10, 15, "def auth(): pass"),
            _r("src/db.py", 1, 5, "def connect(): pass"),
        ]
        ctx = assembler.assemble(results)

        assert isinstance(ctx.text, str)
        assert isinstance(ctx.token_count, int)
        assert isinstance(ctx.included, int)
        assert isinstance(ctx.dropped, int)
        assert isinstance(ctx.files_covered, frozenset)

        assert ctx.token_count > 0
        assert ctx.included == 2
        assert ctx.dropped == 0
        assert ctx.files_covered == frozenset({"src/auth.py", "src/db.py"})

    def test_files_covered_contains_only_included_files(self) -> None:
        """files_covered reflects only the files that made it into the context."""
        chunk_big = "long " * 200
        tok = count_tokens(chunk_big)
        assembler = ContextAssembler(max_tokens=tok + 10)

        results = [
            _r("included.py", 1, 5, chunk_big, score=0.9),
            _r("excluded.py", 1, 5, chunk_big, score=0.5),  # dropped
        ]
        ctx = assembler.assemble(results)

        assert "included.py" in ctx.files_covered
        assert "excluded.py" not in ctx.files_covered


# ---------------------------------------------------------------------------
# R8 -- Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """R8: empty results returns empty AssembledContext without raising."""

    def test_empty_results_returns_empty_context(self) -> None:
        """assembler.assemble([]) returns a fully zero AssembledContext."""
        assembler = ContextAssembler(max_tokens=4000)
        ctx = assembler.assemble([])

        assert ctx == AssembledContext(
            text="",
            token_count=0,
            included=0,
            dropped=0,
            files_covered=frozenset(),
        )

    def test_all_empty_chunk_text_returns_empty_context(self) -> None:
        """Chunks with empty chunk_text are skipped and produce empty context."""
        assembler = ContextAssembler(max_tokens=4000)
        results = [
            _r("a.py", 1, 5, ""),
            _r("b.py", 1, 5, ""),
        ]
        ctx = assembler.assemble(results)

        assert ctx.text == ""
        assert ctx.included == 0


# ---------------------------------------------------------------------------
# R9 -- Token counting consistency
# ---------------------------------------------------------------------------


class TestTokenCountConsistency:
    """R9: token_count == count_tokens(ctx.text) for every result."""

    def test_token_count_matches_count_tokens(self) -> None:
        """ctx.token_count must equal count_tokens(ctx.text)."""
        assembler = ContextAssembler(max_tokens=4000)
        results = [
            _r("src/auth.py", 10, 15, "def auth(): pass\n    return True"),
            _r("src/db.py", 1, 3, "def connect():\n    pass"),
        ]
        ctx = assembler.assemble(results)

        assert ctx.token_count == count_tokens(ctx.text)

    def test_token_count_consistent_single_file(self) -> None:
        """Even for a single file, token_count stays consistent."""
        assembler = ContextAssembler(max_tokens=4000)
        results = [_r("only.py", 1, 10, "import os\nprint(os.getcwd())")]
        ctx = assembler.assemble(results)

        assert ctx.token_count == count_tokens(ctx.text)


# ---------------------------------------------------------------------------
# R10 -- Invalid configuration
# ---------------------------------------------------------------------------


class TestInvalidConfig:
    """R10: max_tokens <= 0 raises ValueError at construction time."""

    def test_max_tokens_zero_raises(self) -> None:
        """ContextAssembler(max_tokens=0) must raise ValueError."""
        with pytest.raises(ValueError, match="max_tokens"):
            ContextAssembler(max_tokens=0)

    def test_max_tokens_negative_raises(self) -> None:
        """ContextAssembler(max_tokens=-5) must raise ValueError."""
        with pytest.raises(ValueError, match="max_tokens"):
            ContextAssembler(max_tokens=-5)

    def test_max_tokens_positive_does_not_raise(self) -> None:
        """ContextAssembler with a valid max_tokens constructs without error."""
        assembler = ContextAssembler(max_tokens=1)
        assert assembler.max_tokens == 1


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


class TestIntegrationSmoke:
    """End-to-end: 3 results from 2 files, no truncation, full output verified."""

    def test_full_output_contains_both_file_headers(self) -> None:
        """3 results from 2 files produce correct headers and content."""
        assembler = ContextAssembler(max_tokens=4000)
        results = [
            _r(
                "src/auth.py",
                10,
                12,
                "def authenticate(user, pwd):\n    return check(user, pwd)\n    # end",
                score=0.95,
            ),
            _r(
                "src/auth.py",
                40,
                42,
                "def logout(user):\n    session.clear()\n    return True",
                score=0.85,
            ),
            _r(
                "src/db.py",
                1,
                3,
                "def connect():\n    return engine.connect()\n    # done",
                score=0.75,
            ),
        ]
        ctx = assembler.assemble(results)

        # Both files covered
        assert "src/auth.py" in ctx.files_covered
        assert "src/db.py" in ctx.files_covered

        # Both file headers present
        assert "## src/auth.py" in ctx.text
        assert "## src/db.py" in ctx.text

        # Correct line ranges
        assert "(lines 10-12)" in ctx.text
        assert "(lines 40-42)" in ctx.text
        assert "(lines 1-3)" in ctx.text

        # Line-numbered content present
        assert "10 | def authenticate" in ctx.text
        assert "40 | def logout" in ctx.text
        assert "1 | def connect" in ctx.text

        # Files separated by ---
        assert "\n---\n" in ctx.text

        # Counters
        assert ctx.included == 3
        assert ctx.dropped == 0
        assert ctx.token_count == count_tokens(ctx.text)
