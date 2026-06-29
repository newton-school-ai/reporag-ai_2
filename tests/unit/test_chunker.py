"""Unit tests for the SemanticChunker (Issue 8)."""

from __future__ import annotations

import pytest

from src.reporag.ingestion.chunker import (
    CLASS,
    FUNCTION,
    METHOD,
    MODULE,
    Chunk,
    SemanticChunker,
    TokenCounter,
    chunk_source,
)
from src.reporag.ingestion.parser import UnsupportedLanguageError


@pytest.fixture
def chunker() -> SemanticChunker:
    """A chunker with a generous budget (nothing splits)."""
    return SemanticChunker(max_tokens=512)


def _ceiling(chunker: SemanticChunker) -> int:
    """Hard upper bound a chunk may reach before it must be split."""
    return chunker._max_allowed


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------


def test_token_counter_counts_positive() -> None:
    """The token counter returns positive counts for real text and 0 empty."""
    counter = TokenCounter()
    assert counter.count("") == 0
    assert counter.count("def hello(): return 42") > 0


def test_custom_token_counter_is_used() -> None:
    """A custom token_counter callable overrides the default."""
    chunker = SemanticChunker(max_tokens=10, token_counter=lambda text: len(text))
    chunks = chunker.chunk_source("x = 1\n", language="python")
    # The custom counter is character length, applied to the chunk's own text.
    assert chunks[0].token_count == len(chunks[0].text)


# ---------------------------------------------------------------------------
# Small function -> single chunk
# ---------------------------------------------------------------------------


def test_small_function_single_chunk(chunker: SemanticChunker) -> None:
    """A small function produces exactly one chunk that is not split."""
    code = "def add(a, b):\n" '    """Add two numbers."""\n' "    return a + b\n"
    chunks = chunker.chunk_source(code, language="python", file_path="m.py")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.parent_symbol == "add"
    assert chunk.symbol_type == FUNCTION
    assert chunk.start_line == 1
    assert chunk.end_line == 3
    assert chunk.total_parts == 1
    assert chunk.is_continuation is False
    assert chunk.file_path == "m.py"
    assert chunk.language == "python"
    assert "def add(a, b):" in chunk.text
    assert chunk.token_count > 0


def test_chunk_index_is_sequential(chunker: SemanticChunker) -> None:
    """Chunks are returned in source order with sequential chunk_index."""
    code = (
        "import os\n"
        "\n"
        "def first():\n"
        "    return 1\n"
        "\n"
        "def second():\n"
        "    return 2\n"
    )
    chunks = chunker.chunk_source(code, language="python")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    names = [c.parent_symbol for c in chunks]
    assert names == [None, "first", "second"]


# ---------------------------------------------------------------------------
# Classes and methods
# ---------------------------------------------------------------------------


def test_small_class_single_chunk(chunker: SemanticChunker) -> None:
    """A small class stays whole as one chunk."""
    code = (
        "class Point:\n"
        '    """A 2D point."""\n'
        "    def __init__(self, x, y):\n"
        "        self.x = x\n"
        "        self.y = y\n"
    )
    chunks = chunker.chunk_source(code, language="python")
    assert len(chunks) == 1
    assert chunks[0].symbol_type == CLASS
    assert chunks[0].parent_symbol == "Point"


def test_large_class_splits_into_methods() -> None:
    """An oversized class is split into header + per-method chunks."""
    # Each method body padded so a tight budget forces splitting.
    pad = "\n".join(f"        value_{i} = {i} * 2" for i in range(12))
    code = (
        "class Service:\n"
        '    """Service docstring."""\n'
        "    def alpha(self):\n"
        f"{pad}\n"
        "        return 0\n"
        "\n"
        "    def beta(self):\n"
        f"{pad}\n"
        "        return 1\n"
        "\n"
        "    def gamma(self):\n"
        f"{pad}\n"
        "        return 2\n"
    )
    chunker = SemanticChunker(max_tokens=60)
    chunks = chunker.chunk_source(code, language="python")

    assert len(chunks) > 1
    # Every method appears as its own (method) chunk owned by the class.
    method_owners = {c.parent_symbol for c in chunks if c.symbol_type == METHOD}
    assert {"Service.alpha", "Service.beta", "Service.gamma"} <= method_owners
    # The class header chunk carries the class signature.
    header = next(c for c in chunks if c.symbol_type == CLASS)
    assert "class Service" in header.text


def test_method_type_and_qualified_name() -> None:
    """Methods of a split class are tagged METHOD with qualified names."""
    body = "\n".join(f"        n{i} = {i}" for i in range(20))
    code = "class Big:\n" f"    def work(self):\n{body}\n        return n0\n"
    chunker = SemanticChunker(max_tokens=40)
    chunks = chunker.chunk_source(code, language="python")
    method_chunks = [c for c in chunks if c.parent_symbol == "Big.work"]
    assert method_chunks
    assert all(c.symbol_type == METHOD for c in method_chunks)


# ---------------------------------------------------------------------------
# Large function splitting with signature overlap
# ---------------------------------------------------------------------------


def test_large_function_splits_with_signature_overlap() -> None:
    """A function exceeding the budget splits with signature overlap."""
    statements = "\n".join(f"    step_{i} = compute({i})" for i in range(40))
    code = (
        "def pipeline(data):\n"
        '    """Run the pipeline."""\n'
        f"{statements}\n"
        "    return data\n"
    )
    chunker = SemanticChunker(max_tokens=50)
    chunks = chunker.chunk_source(code, language="python", file_path="p.py")

    assert len(chunks) > 1
    assert all(c.parent_symbol == "pipeline" for c in chunks)
    assert all(c.symbol_type == FUNCTION for c in chunks)

    first, *rest = chunks
    assert first.part == 1
    assert first.is_continuation is False
    assert "def pipeline(data):" in first.text
    # Continuation chunks repeat the signature as overlap context.
    for cont in rest:
        assert cont.is_continuation is True
        assert "def pipeline(data):" in cont.text
    # Parts are numbered 1..N of N.
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.total_parts == len(chunks) for c in chunks)


def test_overlap_can_be_disabled() -> None:
    """With overlap_signature=False continuation chunks omit the signature."""
    statements = "\n".join(f"    step_{i} = compute({i})" for i in range(40))
    code = "def pipeline(data):\n" f"{statements}\n" "    return data\n"
    chunker = SemanticChunker(max_tokens=50, overlap_signature=False)
    chunks = chunker.chunk_source(code, language="python")
    continuations = [c for c in chunks if c.is_continuation]
    assert continuations
    assert all("def pipeline(data):" not in c.text for c in continuations)


def test_chunks_respect_max_tokens_tolerance() -> None:
    """No chunk exceeds max_tokens * (1 + tolerance)."""
    statements = "\n".join(f"    step_{i} = compute({i}, {i} + 1)" for i in range(60))
    code = (
        "def big(data):\n"
        '    """Big function."""\n'
        f"{statements}\n"
        "    return data\n"
    )
    chunker = SemanticChunker(max_tokens=80, size_tolerance=0.1)
    chunks = chunker.chunk_source(code, language="python")
    assert len(chunks) > 1
    ceiling = _ceiling(chunker)
    for chunk in chunks:
        assert chunk.token_count <= ceiling


def test_line_ranges_are_ordered_and_within_source() -> None:
    """Chunk line ranges are ordered and stay within the source file."""
    statements = "\n".join(f"    step_{i} = {i}" for i in range(50))
    code = "def f():\n" f"{statements}\n" "    return 1\n"
    total_lines = code.count("\n")
    chunker = SemanticChunker(max_tokens=40)
    chunks = chunker.chunk_source(code, language="python")
    for chunk in chunks:
        assert 1 <= chunk.start_line <= chunk.end_line <= total_lines
    starts = [c.start_line for c in chunks]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Module-level code
# ---------------------------------------------------------------------------


def test_module_level_code_is_chunked(chunker: SemanticChunker) -> None:
    """Module-level imports/statements become module chunks (no parent)."""
    code = (
        "import os\n"
        "import sys\n"
        "\n"
        "CONFIG = {'a': 1}\n"
        "VALUE = os.getenv('X')\n"
    )
    chunks = chunker.chunk_source(code, language="python")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.symbol_type == MODULE
    assert chunk.parent_symbol is None
    assert "import os" in chunk.text
    assert "CONFIG" in chunk.text


def test_module_code_packs_under_budget() -> None:
    """Many module-level statements pack into multiple bounded chunks."""
    lines = "\n".join(f"CONST_{i} = {i} + {i} * 2" for i in range(40))
    chunker = SemanticChunker(max_tokens=40)
    chunks = chunker.chunk_source(lines + "\n", language="python")
    assert len(chunks) > 1
    ceiling = _ceiling(chunker)
    for chunk in chunks:
        assert chunk.symbol_type == MODULE
        assert chunk.parent_symbol is None
        assert chunk.token_count <= ceiling


def test_mixed_module_and_definitions_ordering(chunker: SemanticChunker) -> None:
    """Module code and definitions interleave in source order."""
    code = (
        "import os\n"
        "\n"
        "def f():\n"
        "    return 1\n"
        "\n"
        "X = 10\n"
        "\n"
        "class C:\n"
        "    pass\n"
    )
    chunks = chunker.chunk_source(code, language="python")
    types = [c.symbol_type for c in chunks]
    assert types == [MODULE, FUNCTION, MODULE, CLASS]


# ---------------------------------------------------------------------------
# Decorators, async, nested
# ---------------------------------------------------------------------------


def test_decorated_function_includes_decorator(chunker: SemanticChunker) -> None:
    """A decorated function chunk spans the decorator lines."""
    code = "@app.route('/health')\n" "def health():\n" "    return 'ok'\n"
    chunks = chunker.chunk_source(code, language="python")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.start_line == 1
    assert "@app.route" in chunk.text
    assert chunk.parent_symbol == "health"


def test_async_function_is_chunked(chunker: SemanticChunker) -> None:
    """An async function is captured as a single function chunk."""
    code = "async def fetch(url):\n    return await get(url)\n"
    chunks = chunker.chunk_source(code, language="python")
    assert len(chunks) == 1
    assert chunks[0].parent_symbol == "fetch"
    assert "async def fetch" in chunks[0].text


def test_nested_function_kept_inside_parent(chunker: SemanticChunker) -> None:
    """A small parent with a nested function stays as one chunk."""
    code = (
        "def outer():\n"
        "    def inner():\n"
        "        return 1\n"
        "    return inner()\n"
    )
    chunks = chunker.chunk_source(code, language="python")
    assert len(chunks) == 1
    assert chunks[0].parent_symbol == "outer"
    assert "def inner():" in chunks[0].text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_source_yields_no_chunks(chunker: SemanticChunker) -> None:
    """Empty source produces no chunks."""
    assert chunker.chunk_source("", language="python") == []
    assert chunker.chunk_source("\n\n   \n", language="python") == []


def test_syntax_error_still_chunks(chunker: SemanticChunker) -> None:
    """Broken source still yields chunks (tree-sitter is error-tolerant)."""
    code = "def broken(:\n    return 1\n\ndef ok():\n    return 2\n"
    chunks = chunker.chunk_source(code, language="python")
    assert chunks
    assert any(c.parent_symbol == "ok" for c in chunks)


def test_non_python_language_raises(chunker: SemanticChunker) -> None:
    """Unsupported languages raise UnsupportedLanguageError."""
    with pytest.raises(UnsupportedLanguageError):
        chunker.chunk_source("function f() {}", language="javascript")


def test_invalid_max_tokens_raises() -> None:
    """A non-positive max_tokens is rejected."""
    with pytest.raises(ValueError, match="max_tokens"):
        SemanticChunker(max_tokens=0)


# ---------------------------------------------------------------------------
# File-based API and sample repo
# ---------------------------------------------------------------------------


def test_chunk_file_infers_language(tmp_path) -> None:
    """chunk_file infers the language from the extension."""
    path = tmp_path / "sample.py"
    path.write_text("def f():\n    return 1\n", encoding="utf-8")
    chunker = SemanticChunker(max_tokens=512)
    chunks = chunker.chunk_file(path)
    assert len(chunks) == 1
    assert chunks[0].language == "python"
    assert chunks[0].parent_symbol == "f"


def test_chunk_file_unknown_extension_raises(tmp_path) -> None:
    """An unknown extension raises UnsupportedLanguageError."""
    path = tmp_path / "data.unknownext"
    path.write_text("noop\n", encoding="utf-8")
    with pytest.raises(UnsupportedLanguageError):
        SemanticChunker().chunk_file(path)


def test_sample_repo_app_py() -> None:
    """The Issue-8 smoke test on the sample repo yields expected chunks."""
    chunker = SemanticChunker(max_tokens=512)
    chunks = chunker.chunk_file("examples/sample_repo/app.py")
    parents = [c.parent_symbol for c in chunks]
    assert "handle_login" in parents
    assert "handle_profile" in parents
    # Module-level imports are captured with no parent symbol.
    assert any(c.parent_symbol is None and "import" in c.text for c in chunks)
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.token_count > 0
        assert chunk.start_line <= chunk.end_line


def test_convenience_wrapper() -> None:
    """The module-level chunk_source helper mirrors the class API."""
    chunks = chunk_source("def f():\n    return 1\n", max_tokens=128)
    assert len(chunks) == 1
    assert chunks[0].parent_symbol == "f"
