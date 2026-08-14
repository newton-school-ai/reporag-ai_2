from reporag.generation.context_assembler import ContextAssembler
from reporag.retrieval.vector_search import RetrievalResult


def make_result(
    score: float,
    file_path: str,
    start_line: int | None,
    end_line: int | None,
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


def test_basic_assembly():
    assembler = ContextAssembler(max_tokens=4000)

    # Unordered chunks
    c1 = make_result(0.8, "src/b.py", 10, 12, "b_line10\nb_line11\nb_line12")
    c2 = make_result(0.9, "src/a.py", 20, 22, "a_line20\na_line21\na_line22")
    c3 = make_result(0.7, "src/a.py", 5, 6, "a_line5\na_line6")

    result = assembler.assemble([c1, c2, c3])

    # Should order by file (a.py then b.py), and within a.py by start_line (5 then 20)
    expected = (
        "## src/a.py (lines 5-6)\n"
        "```python\n"
        "a_line5\n"
        "a_line6\n"
        "```\n\n"
        "## src/a.py (lines 20-22)\n"
        "```python\n"
        "a_line20\n"
        "a_line21\n"
        "a_line22\n"
        "```\n\n"
        "## src/b.py (lines 10-12)\n"
        "```python\n"
        "b_line10\n"
        "b_line11\n"
        "b_line12\n"
        "```"
    )
    assert result == expected


def test_merge_overlapping_chunks():
    assembler = ContextAssembler(max_tokens=4000)

    # Chunks overlapping by >50%
    c1 = make_result(0.9, "src/a.py", 10, 15, "L10\nL11\nL12\nL13\nL14\nL15")
    # c2 overlaps from 12 to 18 (4 lines overlap, which is > 50% of the 6 lines)
    # Simulate a chunker header on c2
    c2 = make_result(
        0.8,
        "src/a.py",
        12,
        18,
        "def func(): # ... continued\nL12\nL13\nL14\nL15\nL16\nL17\nL18",
    )

    result = assembler.assemble([c1, c2])

    expected = (
        "## src/a.py (lines 10-18)\n"
        "```python\n"
        "L10\n"
        "L11\n"
        "L12\n"
        "L13\n"
        "L14\n"
        "L15\n"
        "L16\n"
        "L17\n"
        "L18\n"
        "```"
    )
    assert result == expected


def test_merge_engulfed_chunk():
    assembler = ContextAssembler(max_tokens=4000)

    # A large class chunk
    c1 = make_result(0.9, "src/a.py", 10, 30, "CLASS_START\n...\nCLASS_END")
    # A small method chunk entirely inside the class
    c2 = make_result(0.95, "src/a.py", 15, 20, "METHOD_START\n...\nMETHOD_END")

    result = assembler.assemble([c1, c2])

    # The method chunk should be completely absorbed, leaving only the class chunk
    expected = (
        "## src/a.py (lines 10-30)\n"
        "```python\n"
        "CLASS_START\n"
        "...\n"
        "CLASS_END\n"
        "```"
    )
    assert result == expected


def test_truncation_prioritizes_scores():
    # Set max tokens very low, so only a few chunks fit
    assembler = ContextAssembler(
        max_tokens=10
    )  # ~10 tokens fits only 1 or 2 small chunks

    # Create chunks. count_tokens will be called.
    # text lengths roughly correspond to token count (about 2-3 tokens each here)
    c1 = make_result(0.9, "src/a.py", 10, 10, "high_score_chunk")
    c2 = make_result(
        0.2,
        "src/a.py",
        20,
        20,
        "low_score_chunk_that_should_be_skipped_completely_because_it_does_not_fit_in_budget",
    )
    c3 = make_result(0.8, "src/a.py", 30, 30, "med_score_chunk")

    # We will pass c1, c2, c3. We expect c1 and c3 to be included, c2 to be dropped due to token limit
    # Actually, if we just make c2 huge, it will be skipped.
    result = assembler.assemble([c1, c2, c3])

    assert "high_score_chunk" in result
    assert "med_score_chunk" in result
    assert "low_score" not in result


def test_none_line_numbers():
    assembler = ContextAssembler(max_tokens=4000)

    c1 = make_result(0.9, "docs/readme.md", None, None, "# Title\nDoc content")
    c2 = make_result(0.8, "docs/readme.md", 10, 15, "Some specific lines")

    result = assembler.assemble([c1, c2])

    # None should sort first
    expected = (
        "## docs/readme.md (lines ?-?)\n"
        "```python\n"
        "# Title\n"
        "Doc content\n"
        "```\n\n"
        "## docs/readme.md (lines 10-15)\n"
        "```python\n"
        "Some specific lines\n"
        "```"
    )
    assert result == expected
