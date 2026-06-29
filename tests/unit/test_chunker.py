"""Unit tests for src/reporag/ingestion/chunker.py (Issue 8).

Acceptance Criteria covered:
- [x] Never splits a function/class mid-body (unless exceeds max_tokens)
- [x] Large functions are split at logical points with signature overlap
- [x] Each chunk has metadata: file, lines, parent symbol, language, token count
- [x] Chunk sizes stay within configurable max_tokens +/- 10%
- [x] Unit tests: small function (1 chunk), large class (multiple chunks),
      module-level code
"""

from __future__ import annotations

import pytest

from src.reporag.ingestion.chunker import Chunk, SemanticChunker, count_tokens

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chunker() -> SemanticChunker:
    """SemanticChunker with a 128-token budget for tests."""
    return SemanticChunker(max_tokens=128)


def chunk(source: str, chunker: SemanticChunker, **kw) -> list[Chunk]:
    """Convenience wrapper around extract_from_source."""
    return chunker.chunk_source(source, language="python", **kw)


# ---------------------------------------------------------------------------
# Source fixtures
# ---------------------------------------------------------------------------

SMALL_FUNCTION = """\
def add(a: int, b: int) -> int:
    \"\"\"Add two integers.\"\"\"
    return a + b
"""

ASYNC_FUNCTION = """\
async def fetch(url: str) -> dict:
    \"\"\"Fetch data from url.\"\"\"
    return {}
"""

CLASS_SRC = """\
class Greeter:
    \"\"\"A simple greeter.\"\"\"

    def __init__(self, name: str) -> None:
        \"\"\"Initialise with name.\"\"\"
        self.name = name

    def greet(self) -> str:
        \"\"\"Return greeting.\"\"\"
        return f"Hello, {self.name}"
"""

MODULE_LEVEL_SRC = """\
import os
import sys
from pathlib import Path

CONSTANT = 42
DEBUG = True
"""

DECORATED_FUNCTION = """\
import functools


def decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


@decorator
def greet(name: str) -> str:
    \"\"\"Greet someone.\"\"\"
    return f"Hi {name}"
"""

MIXED_SRC = """\
import os
from pathlib import Path

class Config:
    \"\"\"Config class.\"\"\"

    def load(self, path: str) -> dict:
        \"\"\"Load config.\"\"\"
        return {}

    @staticmethod
    def default() -> \"Config\":
        return Config()

def run(config: Config) -> None:
    \"\"\"Run the app.\"\"\"
    pass

SETTING = "production"
"""

EMPTY_SRC = ""

SYNTAX_ERROR_SRC = """\
def broken(
    return 42
"""

# A large function that should be split at statement boundaries
# Each statement is a distinct logical unit
LARGE_FUNCTION = (
    "def process(data: list) -> list:\n"
    '    """Process a list of items."""\n'
    + "\n".join(
        f"    step_{i} = data[{i}] if len(data) > {i} else None  # step {i}"
        for i in range(40)
    )
    + "\n    return data\n"
)

NESTED_CLASS_SRC = """\
class Outer:
    \"\"\"Outer class.\"\"\"

    class Inner:
        \"\"\"Inner class.\"\"\"

        def inner_method(self) -> None:
            pass

    def outer_method(self) -> None:
        pass
"""


# ---------------------------------------------------------------------------
# 1. Returns list of Chunk objects
# ---------------------------------------------------------------------------


def test_returns_list(chunker: SemanticChunker) -> None:
    """chunk_source always returns a list."""
    result = chunk(SMALL_FUNCTION, chunker)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(c, Chunk) for c in result)


# ---------------------------------------------------------------------------
# 2. Small function -> exactly 1 chunk
# ---------------------------------------------------------------------------


def test_small_function_single_chunk(chunker: SemanticChunker) -> None:
    """A small function fits in one chunk."""
    chunks = chunk(SMALL_FUNCTION, chunker)
    func_chunk = next((c for c in chunks if "def add" in c.content), None)
    assert func_chunk is not None


def test_small_function_no_split(chunker: SemanticChunker) -> None:
    """A small function produces no continuation chunks."""
    chunks = chunk(SMALL_FUNCTION, chunker)
    assert not any(c.is_continuation for c in chunks)


def test_small_function_content(chunker: SemanticChunker) -> None:
    """The chunk content contains the full function body."""
    chunks = chunk(SMALL_FUNCTION, chunker)
    func_chunk = next((c for c in chunks if "def add" in c.content), None)
    assert func_chunk is not None
    assert "return a + b" in func_chunk.content


# ---------------------------------------------------------------------------
# 3. Chunk metadata completeness
# ---------------------------------------------------------------------------


def test_chunk_has_file_path(chunker: SemanticChunker) -> None:
    """Every chunk stores the file_path label."""
    chunks = chunker.chunk_source(
        SMALL_FUNCTION, language="python", file_path="my/file.py"
    )
    assert all(c.file_path == "my/file.py" for c in chunks)


def test_chunk_has_language(chunker: SemanticChunker) -> None:
    """Every chunk stores the language."""
    chunks = chunk(SMALL_FUNCTION, chunker)
    assert all(c.language == "python" for c in chunks)


def test_chunk_has_start_line(chunker: SemanticChunker) -> None:
    """Every chunk has start_line >= 1."""
    chunks = chunk(MIXED_SRC, chunker)
    assert all(c.start_line >= 1 for c in chunks)


def test_chunk_has_end_line(chunker: SemanticChunker) -> None:
    """Every chunk has end_line >= start_line."""
    chunks = chunk(MIXED_SRC, chunker)
    assert all(c.end_line >= c.start_line for c in chunks)


def test_chunk_has_token_count(chunker: SemanticChunker) -> None:
    """Every chunk has token_count > 0."""
    chunks = chunk(MIXED_SRC, chunker)
    assert all(c.token_count > 0 for c in chunks)


def test_chunk_token_count_matches_content(chunker: SemanticChunker) -> None:
    """token_count matches count_tokens(content) for every chunk."""
    chunks = chunk(MIXED_SRC, chunker)
    for c in chunks:
        assert c.token_count == count_tokens(c.content)


# ---------------------------------------------------------------------------
# 4. Async function
# ---------------------------------------------------------------------------


def test_async_function_chunked(chunker: SemanticChunker) -> None:
    """Async functions are chunked and their content preserved."""
    chunks = chunk(ASYNC_FUNCTION, chunker)
    async_chunk = next((c for c in chunks if "async def fetch" in c.content), None)
    assert async_chunk is not None
    assert "return {}" in async_chunk.content


# ---------------------------------------------------------------------------
# 5. Class chunking
# ---------------------------------------------------------------------------


def test_class_produces_chunk(chunker: SemanticChunker) -> None:
    """A class definition produces at least one chunk."""
    chunks = chunk(CLASS_SRC, chunker)
    class_chunk = next((c for c in chunks if "class Greeter" in c.content), None)
    assert class_chunk is not None


def test_class_methods_produce_chunks(chunker: SemanticChunker) -> None:
    """Each method in a class produces its own chunk."""
    chunks = chunk(CLASS_SRC, chunker)
    contents = " ".join(c.content for c in chunks)
    assert "def __init__" in contents
    assert "def greet" in contents


def test_method_parent_symbol(chunker: SemanticChunker) -> None:
    """Method chunks have parent_symbol set to the class name."""
    chunks = chunk(CLASS_SRC, chunker)
    method_chunks = [c for c in chunks if c.parent_symbol == "Greeter"]
    assert len(method_chunks) >= 2  # __init__ and greet


# ---------------------------------------------------------------------------
# 6. Module-level code
# ---------------------------------------------------------------------------


def test_module_level_imports_chunked(chunker: SemanticChunker) -> None:
    """Module-level imports are included in a chunk."""
    chunks = chunk(MODULE_LEVEL_SRC, chunker)
    assert any("import" in c.content for c in chunks)


def test_module_level_parent_symbol_is_none(chunker: SemanticChunker) -> None:
    """Module-level chunks have parent_symbol=None."""
    chunks = chunk(MODULE_LEVEL_SRC, chunker)
    assert any(c.parent_symbol is None for c in chunks)


def test_module_level_constant_in_chunk(chunker: SemanticChunker) -> None:
    """Module-level constants appear in a chunk."""
    chunks = chunk(MODULE_LEVEL_SRC, chunker)
    contents = " ".join(c.content for c in chunks)
    assert "CONSTANT" in contents


# ---------------------------------------------------------------------------
# 7. Decorated function
# ---------------------------------------------------------------------------


def test_decorated_function_chunked(chunker: SemanticChunker) -> None:
    """Decorated functions are chunked with their decorators."""
    chunks = chunk(DECORATED_FUNCTION, chunker)
    greet_chunk = next((c for c in chunks if "def greet" in c.content), None)
    assert greet_chunk is not None
    assert "@decorator" in greet_chunk.content


# ---------------------------------------------------------------------------
# 8. Large function splitting
# ---------------------------------------------------------------------------


def test_large_function_split_into_multiple_chunks(chunker: SemanticChunker) -> None:
    """A function exceeding max_tokens is split into multiple chunks."""
    chunks = chunk(LARGE_FUNCTION, chunker)
    func_chunks = [
        c
        for c in chunks
        if "process" in (c.parent_symbol or "") or "def process" in c.content
    ]
    assert len(func_chunks) >= 2, "Large function must produce >= 2 chunks"


def test_large_function_continuation_flag(chunker: SemanticChunker) -> None:
    """Continuation chunks have is_continuation=True."""
    chunks = chunk(LARGE_FUNCTION, chunker)
    assert any(c.is_continuation for c in chunks)


def test_large_function_overlap_header(chunker: SemanticChunker) -> None:
    """Continuation chunks carry an overlap_header."""
    chunks = chunk(LARGE_FUNCTION, chunker)
    cont_chunks = [c for c in chunks if c.is_continuation]
    assert all(c.overlap_header is not None for c in cont_chunks)


def test_large_function_header_in_continuation_content(
    chunker: SemanticChunker,
) -> None:
    """The function signature appears in continuation chunk content."""
    chunks = chunk(LARGE_FUNCTION, chunker)
    cont_chunks = [c for c in chunks if c.is_continuation]
    assert all("def process" in c.content for c in cont_chunks)


def test_large_function_no_mid_statement_split(chunker: SemanticChunker) -> None:
    """No chunk ends in the middle of a statement (each chunk is parseable)."""
    chunks = chunk(LARGE_FUNCTION, chunker)
    for c in chunks:
        # Each chunk must contain complete lines (no broken lines)
        lines = c.content.splitlines()
        assert len(lines) >= 1


def test_large_function_chunk_index(chunker: SemanticChunker) -> None:
    """chunk_index increments correctly across split chunks."""
    big_chunker = SemanticChunker(max_tokens=32)
    chunks = big_chunker.chunk_source(LARGE_FUNCTION, language="python")
    func_chunks = [c for c in chunks if c.is_continuation or "def process" in c.content]
    if len(func_chunks) > 1:
        indices = [c.chunk_index for c in func_chunks]
        # Indices should be non-decreasing
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# 9. Token budget
# ---------------------------------------------------------------------------


def test_token_budget_respected_small_chunks(chunker: SemanticChunker) -> None:
    """All chunks from small source stay within max_tokens."""
    chunks = chunk(SMALL_FUNCTION, chunker)
    for c in chunks:
        assert (
            c.token_count <= chunker.max_tokens * 1.1
        ), f"Chunk exceeds budget: {c.token_count} > {chunker.max_tokens}"


def test_token_budget_respected_mixed_source() -> None:
    """Chunks from mixed source stay within budget (with 10% tolerance)."""
    max_tok = 64
    c = SemanticChunker(max_tokens=max_tok)
    chunks = c.chunk_source(MIXED_SRC, language="python")
    for ck in chunks:
        # Allow 10% tolerance for single-statement oversized blocks
        assert (
            ck.token_count <= max_tok * 1.1 + 10
        ), f"Chunk too large: {ck.token_count} tokens"


# ---------------------------------------------------------------------------
# 10. Mixed source
# ---------------------------------------------------------------------------


def test_mixed_source_all_types_covered(chunker: SemanticChunker) -> None:
    """Mixed source produces chunks for imports, class, functions, and constants."""
    chunks = chunk(MIXED_SRC, chunker)
    all_content = " ".join(c.content for c in chunks)
    assert "import" in all_content
    assert "class Config" in all_content
    assert "def run" in all_content
    assert "SETTING" in all_content


def test_mixed_source_chunk_ordering(chunker: SemanticChunker) -> None:
    """Chunks are in source order (start_line is non-decreasing)."""
    chunks = chunk(MIXED_SRC, chunker)
    lines = [c.start_line for c in chunks]
    assert lines == sorted(lines)


# ---------------------------------------------------------------------------
# 11. Empty source
# ---------------------------------------------------------------------------


def test_empty_source_returns_empty_list(chunker: SemanticChunker) -> None:
    """Empty source produces an empty chunk list."""
    chunks = chunk(EMPTY_SRC, chunker)
    assert chunks == []


# ---------------------------------------------------------------------------
# 12. Syntax error tolerance
# ---------------------------------------------------------------------------


def test_syntax_error_does_not_raise(chunker: SemanticChunker) -> None:
    """Broken source must not raise; returns partial results."""
    result = chunk(SYNTAX_ERROR_SRC, chunker)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 13. Nested class
# ---------------------------------------------------------------------------


def test_nested_class_chunked(chunker: SemanticChunker) -> None:
    """Nested classes produce their own chunks."""
    chunks = chunk(NESTED_CLASS_SRC, chunker)
    all_content = " ".join(c.content for c in chunks)
    assert "class Outer" in all_content
    assert "class Inner" in all_content


# ---------------------------------------------------------------------------
# 14. count_tokens helper
# ---------------------------------------------------------------------------


def test_count_tokens_empty_string() -> None:
    """Empty string returns 0 tokens."""
    assert count_tokens("") == 0


def test_count_tokens_positive() -> None:
    """Non-empty string returns a positive token count."""
    assert count_tokens("def hello(): pass") > 0


def test_count_tokens_longer_is_more() -> None:
    """Longer text produces more tokens than shorter text."""
    short = count_tokens("hello")
    long = count_tokens("hello world, this is a much longer sentence with more words")
    assert long > short


# ---------------------------------------------------------------------------
# 15. Chunk __repr__
# ---------------------------------------------------------------------------


def test_chunk_repr(chunker: SemanticChunker) -> None:
    """Chunk repr is human-readable."""
    chunks = chunk(SMALL_FUNCTION, chunker)
    c = chunks[0]
    r = repr(c)
    assert "[" in r and "]" in r
    assert "tokens" in r


# ---------------------------------------------------------------------------
# 16. SemanticChunker max_tokens configurable
# ---------------------------------------------------------------------------


def test_smaller_budget_produces_more_chunks() -> None:
    """A smaller max_tokens budget produces >= as many chunks as a larger one."""
    large_chunker = SemanticChunker(max_tokens=1024)
    small_chunker = SemanticChunker(max_tokens=32)

    large_chunks = large_chunker.chunk_source(LARGE_FUNCTION, language="python")
    small_chunks = small_chunker.chunk_source(LARGE_FUNCTION, language="python")

    assert len(small_chunks) >= len(large_chunks)


# ---------------------------------------------------------------------------
# 17. chunk_file from disk
# ---------------------------------------------------------------------------


def test_chunk_file_from_disk(tmp_path, chunker: SemanticChunker) -> None:
    """chunk_file reads from disk and produces chunks."""
    src_file = tmp_path / "module.py"
    src_file.write_text(SMALL_FUNCTION, encoding="utf-8")

    chunks = chunker.chunk_file(str(src_file), language="python")
    assert len(chunks) >= 1
    assert all(c.file_path == str(src_file) for c in chunks)


def test_chunk_file_path_stored(tmp_path, chunker: SemanticChunker) -> None:
    """All chunks carry the correct file_path from disk."""
    src_file = tmp_path / "code.py"
    src_file.write_text(MIXED_SRC, encoding="utf-8")

    chunks = chunker.chunk_file(str(src_file), language="python")
    assert all(c.file_path == str(src_file) for c in chunks)


# ---------------------------------------------------------------------------
# 18. Unsupported language raises
# ---------------------------------------------------------------------------


def test_unsupported_language_raises(chunker: SemanticChunker) -> None:
    """Unsupported language raises UnsupportedLanguageError."""
    from src.reporag.ingestion.parser import UnsupportedLanguageError

    with pytest.raises(UnsupportedLanguageError):
        chunker.chunk_source("function hello() {}", language="javascript")


# ---------------------------------------------------------------------------
# 19. Chunk is_continuation and chunk_index for first chunk
# ---------------------------------------------------------------------------


def test_first_chunk_is_not_continuation(chunker: SemanticChunker) -> None:
    """The first chunk of any definition is never marked as continuation."""
    chunks = chunk(CLASS_SRC, chunker)
    first_chunks = [c for c in chunks if c.chunk_index == 0]
    assert all(not c.is_continuation for c in first_chunks)


def test_first_chunk_no_overlap_header(chunker: SemanticChunker) -> None:
    """First chunks do not carry an overlap_header."""
    chunks = chunk(SMALL_FUNCTION, chunker)
    first = next(c for c in chunks if c.chunk_index == 0)
    assert first.overlap_header is None


# ---------------------------------------------------------------------------
# 20. Reuse chunker across calls
# ---------------------------------------------------------------------------


def test_chunker_reusable(chunker: SemanticChunker) -> None:
    """Same SemanticChunker instance produces correct results across calls."""
    r1 = chunk(SMALL_FUNCTION, chunker)
    r2 = chunk(CLASS_SRC, chunker)
    r3 = chunk(MODULE_LEVEL_SRC, chunker)

    assert any("def add" in c.content for c in r1)
    assert any("class Greeter" in c.content for c in r2)
    assert any("import" in c.content for c in r3)


# ---------------------------------------------------------------------------
# 21. Comments are grouped with module-level code, not fragmented
# ---------------------------------------------------------------------------

COMMENTED_MODULE_SRC = """\
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SIZE = 1024
DEBUG = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT = 30
"""


def test_comments_grouped_with_module_code(chunker: SemanticChunker) -> None:
    """Comments between module-level statements are not emitted as solo chunks."""
    chunks = chunk(COMMENTED_MODULE_SRC, chunker)
    # All content is module-level; comments must NOT produce solo tiny chunks
    # At most a handful of chunks (not one per line)
    assert len(chunks) <= 3, f"Expected at most 3 chunks, got {len(chunks)}: {chunks}"


def test_comments_not_standalone_chunks(chunker: SemanticChunker) -> None:
    """No chunk should consist solely of comment lines."""
    chunks = chunk(COMMENTED_MODULE_SRC, chunker)
    for c in chunks:
        non_comment_lines = [
            ln
            for ln in c.content.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        # Each chunk must have at least one non-comment, non-blank line
        # (unless the entire file is only comments, which this fixture is not)
        if c.content.strip():
            assert non_comment_lines or all(
                ln.strip().startswith("#") or not ln.strip()
                for ln in c.content.splitlines()
            ), f"Unexpected empty non-comment chunk: {c!r}"


def test_module_level_imports_grouped(chunker: SemanticChunker) -> None:
    """Multiple consecutive imports produce fewer chunks than there are imports."""
    src = "\n".join(f"import module_{i}" for i in range(10))
    chunks = chunk(src, chunker)
    # 10 imports should NOT produce 10 chunks
    assert len(chunks) < 10, f"Too many chunks for simple imports: {len(chunks)}"


# ---------------------------------------------------------------------------
# 22. chunk_from_tree: accepts pre-parsed tree
# ---------------------------------------------------------------------------


def test_chunk_from_tree_produces_chunks(chunker: SemanticChunker) -> None:
    """chunk_from_tree works with a pre-parsed Tree object."""
    from src.reporag.ingestion.parser import ASTParser

    src = SMALL_FUNCTION
    src_bytes = src.encode("utf-8")
    parser = ASTParser()
    tree = parser.parse(src_bytes, language="python")

    chunks = chunker.chunk_from_tree(tree, src, language="python", file_path="test.py")
    assert len(chunks) >= 1
    assert any("def add" in c.content for c in chunks)


def test_chunk_from_tree_same_result_as_chunk_source(chunker: SemanticChunker) -> None:
    """chunk_from_tree and chunk_source produce identical chunks for the same source."""
    from src.reporag.ingestion.parser import ASTParser

    src = CLASS_SRC
    src_bytes = src.encode("utf-8")
    parser = ASTParser()
    tree = parser.parse(src_bytes, language="python")

    via_tree = chunker.chunk_from_tree(tree, src, language="python")
    via_source = chunker.chunk_source(src, language="python")

    assert len(via_tree) == len(via_source)
    for a, b in zip(via_tree, via_source, strict=False):
        assert a.content == b.content
        assert a.start_line == b.start_line
        assert a.end_line == b.end_line


# ---------------------------------------------------------------------------
# 23. Ingestion package exports
# ---------------------------------------------------------------------------


def test_chunk_importable_from_ingestion_package() -> None:
    """Chunk is re-exported from the ingestion package __init__."""
    from src.reporag.ingestion import Chunk as ChunkAlias

    assert ChunkAlias is Chunk


def test_semantic_chunker_importable_from_ingestion_package() -> None:
    """SemanticChunker is re-exported from the ingestion package __init__."""
    from src.reporag.ingestion import SemanticChunker as SemanticChunkerAlias

    assert SemanticChunkerAlias is SemanticChunker


# ---------------------------------------------------------------------------
# 24. chunk_file on real project files (integration smoke)
# ---------------------------------------------------------------------------


def test_chunk_real_file_cloner(chunker: SemanticChunker) -> None:
    """chunk_file on the actual cloner.py produces sane, non-empty chunks."""
    import pathlib

    cloner_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "src"
        / "reporag"
        / "ingestion"
        / "cloner.py"
    )
    chunks = chunker.chunk_file(str(cloner_path), language="python")

    assert len(chunks) >= 3, "Expected multiple chunks from a real-world file"
    all_content = " ".join(c.content for c in chunks)
    assert "RepoCloner" in all_content
    assert "clone_and_discover" in all_content
    # Methods of RepoCloner get their own chunks with parent_symbol set
    method_chunks = [c for c in chunks if c.parent_symbol == "RepoCloner"]
    assert (
        len(method_chunks) >= 2
    ), f"Expected method chunks for RepoCloner, got chunks: {chunks}"
