"""Unit tests for the SemanticChunker (Issue 8).

Acceptance criteria verified:
- Never splits a function/class mid-body (unless exceeds max_tokens)
- Large functions are split at logical points with signature overlap
- Each chunk has metadata: file, lines, parent_symbol, language, token_count
- Chunk sizes stay within configurable max_tokens +/- one statement tolerance
- Unit tests: small function (1 chunk), large class (multiple chunks),
  module-level code
"""

from __future__ import annotations

import pathlib

import pytest

from src.reporag.ingestion.chunker import (
    Chunk,
    SemanticChunker,
    _Accumulator,
    count_tokens,
)
from src.reporag.ingestion.parser import UnsupportedLanguageError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chunker() -> SemanticChunker:
    """A single SemanticChunker instance reused across the module."""
    return SemanticChunker(max_tokens=128)


# ---------------------------------------------------------------------------
# 1. count_tokens helper
# ---------------------------------------------------------------------------


def test_count_tokens_empty_string() -> None:
    """Empty string returns 0 tokens without invoking the encoder."""
    assert count_tokens("") == 0


def test_count_tokens_nonempty_is_positive() -> None:
    """Non-empty source produces a positive token count."""
    assert count_tokens("def hello(): pass") > 0


def test_count_tokens_monotone() -> None:
    """Longer text produces more tokens than shorter text."""
    assert count_tokens("hello world foo bar baz qux") > count_tokens("hi")


# ---------------------------------------------------------------------------
# 2. _Accumulator unit tests
# ---------------------------------------------------------------------------


def test_accumulator_empty_on_init() -> None:
    """A fresh _Accumulator is empty."""
    acc = _Accumulator()
    assert acc.is_empty()


def test_accumulator_not_empty_after_add() -> None:
    """Adding a statement makes the accumulator non-empty."""
    acc = _Accumulator()
    acc.add("x = 1", 3, 1, 1)
    assert not acc.is_empty()


def test_accumulator_would_not_overflow_when_empty() -> None:
    """An empty accumulator never reports overflow (there is nothing to flush)."""
    acc = _Accumulator()
    assert not acc.would_overflow(extra_tokens=9999, budget=1)


def test_accumulator_would_overflow_when_over_budget() -> None:
    """Accumulator detects when adding tokens would exceed the budget."""
    acc = _Accumulator()
    acc.add("x = 1", 3, 1, 1)
    assert acc.would_overflow(extra_tokens=200, budget=10)


def test_accumulator_flush_returns_correct_values() -> None:
    """flush() returns the joined text, line range, and token total."""
    acc = _Accumulator()
    acc.add("x = 1", 3, 5, 5)
    acc.add("y = 2", 4, 6, 6)
    text, start, end, tokens = acc.flush()
    assert text == "x = 1\ny = 2"
    assert start == 5
    assert end == 6
    assert tokens == 7


def test_accumulator_flush_resets_state() -> None:
    """After flush(), the accumulator is empty and ready for reuse."""
    acc = _Accumulator()
    acc.add("a = 1", 3, 1, 1)
    acc.flush()
    assert acc.is_empty()
    assert acc.start_line is None


# ---------------------------------------------------------------------------
# 3. Chunk dataclass
# ---------------------------------------------------------------------------


def test_chunk_post_init_computes_token_count() -> None:
    """Chunk.__post_init__ pre-computes token_count when left at 0."""
    c = Chunk(
        content="def add(a, b): return a + b",
        file_path="f.py",
        language="python",
        start_line=1,
        end_line=1,
    )
    assert c.token_count > 0
    assert c.token_count == count_tokens(c.content)


def test_chunk_kind_module_level() -> None:
    """A chunk with no parent_symbol and no qualified_name has chunk_kind='module'."""
    c = Chunk(
        content="import os",
        file_path="f.py",
        language="python",
        start_line=1,
        end_line=1,
        parent_symbol=None,
        qualified_name=None,
    )
    assert c.chunk_kind == "module"


def test_chunk_kind_definition() -> None:
    """A chunk with a qualified_name and no continuation has chunk_kind='definition'."""
    c = Chunk(
        content="def add(a, b): return a + b",
        file_path="f.py",
        language="python",
        start_line=1,
        end_line=1,
        qualified_name="add",
        is_continuation=False,
    )
    assert c.chunk_kind == "definition"


def test_chunk_kind_continuation() -> None:
    """A chunk with is_continuation=True has chunk_kind='continuation'."""
    c = Chunk(
        content="def f():  # ... continued\n    pass",
        file_path="f.py",
        language="python",
        start_line=2,
        end_line=3,
        qualified_name="f",
        is_continuation=True,
        overlap_header="def f():",
    )
    assert c.chunk_kind == "continuation"


def test_chunk_kind_is_not_manually_settable() -> None:
    """chunk_kind set at construction is overridden by __post_init__ logic."""
    # Passing chunk_kind="module" but is_continuation=True should give "continuation"
    c = Chunk(
        content="pass",
        file_path="f.py",
        language="python",
        start_line=1,
        end_line=1,
        qualified_name="f",
        chunk_kind="module",  # type: ignore[arg-type]
        is_continuation=True,
    )
    assert c.chunk_kind == "continuation"


def test_chunk_repr_shows_qualified_name() -> None:
    """Chunk repr uses qualified_name when available."""
    c = Chunk(
        content="def greet(self): pass",
        file_path="f.py",
        language="python",
        start_line=5,
        end_line=5,
        qualified_name="Greeter.greet",
    )
    assert "Greeter.greet" in repr(c)
    assert "[5-5]" in repr(c)


def test_chunk_repr_falls_back_to_module() -> None:
    """Chunk repr shows <module> when no qualified_name or parent_symbol."""
    c = Chunk(
        content="import os",
        file_path="f.py",
        language="python",
        start_line=1,
        end_line=1,
    )
    assert "<module>" in repr(c)


def test_chunk_to_dict_has_all_keys() -> None:
    """to_dict() returns a dict with all expected metadata keys."""
    c = Chunk(
        content="def f(): pass",
        file_path="f.py",
        language="python",
        start_line=1,
        end_line=1,
        qualified_name="f",
    )
    d = c.to_dict()
    expected_keys = {
        "content",
        "file_path",
        "language",
        "start_line",
        "end_line",
        "parent_symbol",
        "qualified_name",
        "chunk_kind",
        "token_count",
        "chunk_index",
        "is_continuation",
        "overlap_header",
        "has_parse_error",
    }
    assert set(d.keys()) == expected_keys


def test_chunk_to_dict_values_are_json_serialisable() -> None:
    """to_dict() values are all JSON-primitive types (str, int, bool, None)."""
    import json

    c = Chunk(
        content="x = 1",
        file_path="f.py",
        language="python",
        start_line=1,
        end_line=1,
    )
    # Must not raise
    json.dumps(c.to_dict())


def test_chunk_to_dict_qualified_name_field() -> None:
    """to_dict() contains the qualified_name field value."""
    c = Chunk(
        content="def f(): pass",
        file_path="f.py",
        language="python",
        start_line=1,
        end_line=1,
        qualified_name="MyClass.f",
    )
    assert c.to_dict()["qualified_name"] == "MyClass.f"
    assert c.to_dict()["chunk_kind"] == "definition"


# ---------------------------------------------------------------------------
# 4. Empty source
# ---------------------------------------------------------------------------


def test_chunk_empty_source_returns_empty_list(chunker: SemanticChunker) -> None:
    """Empty source produces an empty chunk list."""
    assert chunker.chunk_source("", language="python") == []


# ---------------------------------------------------------------------------
# 5. Small function -> one chunk
# ---------------------------------------------------------------------------


def test_small_function_produces_one_chunk(chunker: SemanticChunker) -> None:
    """A function that fits within max_tokens produces exactly one chunk."""
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    chunks = chunker.chunk_source(code, language="python")
    func_chunks = [c for c in chunks if "def add" in c.content]
    assert len(func_chunks) == 1


def test_small_function_content_preserved(chunker: SemanticChunker) -> None:
    """The entire function body is preserved in the chunk."""
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    chunks = chunker.chunk_source(code, language="python")
    func_chunk = next(c for c in chunks if "def add" in c.content)
    assert "return a + b" in func_chunk.content


def test_small_function_not_a_continuation(chunker: SemanticChunker) -> None:
    """A single-chunk function is never marked as a continuation."""
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    chunks = chunker.chunk_source(code, language="python")
    assert not any(c.is_continuation for c in chunks)


def test_small_function_no_overlap_header(chunker: SemanticChunker) -> None:
    """A single-chunk function has no overlap_header."""
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    chunks = chunker.chunk_source(code, language="python")
    func_chunk = next(c for c in chunks if "def add" in c.content)
    assert func_chunk.overlap_header is None


def test_small_function_qualified_name(chunker: SemanticChunker) -> None:
    """A module-level function chunk carries qualified_name equal to its name."""
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    chunks = chunker.chunk_source(code, language="python")
    func_chunk = next(c for c in chunks if "def add" in c.content)
    assert func_chunk.qualified_name == "add"


def test_small_function_chunk_kind(chunker: SemanticChunker) -> None:
    """A module-level function chunk has chunk_kind='definition'."""
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    chunks = chunker.chunk_source(code, language="python")
    func_chunk = next(c for c in chunks if "def add" in c.content)
    assert func_chunk.chunk_kind == "definition"


# ---------------------------------------------------------------------------
# 6. Chunk metadata fields
# ---------------------------------------------------------------------------


def test_chunk_carries_file_path(chunker: SemanticChunker) -> None:
    """Every chunk stores the file_path label."""
    code = "def f(): pass\n"
    chunks = chunker.chunk_source(code, language="python", file_path="my/file.py")
    assert all(c.file_path == "my/file.py" for c in chunks)


def test_chunk_carries_language(chunker: SemanticChunker) -> None:
    """Every chunk stores the language name."""
    code = "def f(): pass\n"
    chunks = chunker.chunk_source(code, language="python")
    assert all(c.language == "python" for c in chunks)


def test_chunk_start_line_one_based(chunker: SemanticChunker) -> None:
    """start_line is always >= 1 (1-based line numbers)."""
    code = "import os\n\ndef f(): pass\n"
    chunks = chunker.chunk_source(code, language="python")
    assert all(c.start_line >= 1 for c in chunks)


def test_chunk_end_line_geq_start_line(chunker: SemanticChunker) -> None:
    """end_line is always >= start_line."""
    code = "import os\n\ndef f():\n    pass\n"
    chunks = chunker.chunk_source(code, language="python")
    assert all(c.end_line >= c.start_line for c in chunks)


def test_chunk_token_count_matches_content(chunker: SemanticChunker) -> None:
    """token_count equals count_tokens(content) for every chunk."""
    code = "import os\n\ndef f():\n    pass\n\nclass C:\n    def m(self): pass\n"
    for c in chunker.chunk_source(code, language="python"):
        assert c.token_count == count_tokens(c.content)


# ---------------------------------------------------------------------------
# 7. Async function
# ---------------------------------------------------------------------------


def test_async_function_chunked(chunker: SemanticChunker) -> None:
    """Async functions are chunked and their full content is preserved."""
    code = "async def fetch(url: str) -> dict:\n    return {}\n"
    chunks = chunker.chunk_source(code, language="python")
    async_chunk = next((c for c in chunks if "async def fetch" in c.content), None)
    assert async_chunk is not None
    assert "return {}" in async_chunk.content


# ---------------------------------------------------------------------------
# 8. Class chunking
# ---------------------------------------------------------------------------


def test_class_produces_chunk(chunker: SemanticChunker) -> None:
    """A class definition produces at least one chunk containing the class."""
    code = (
        "class Greeter:\n"
        '    """Greet someone."""\n'
        "    def __init__(self, name: str) -> None:\n"
        "        self.name = name\n"
        "    def greet(self) -> str:\n"
        '        return f"Hello, {self.name}"\n'
    )
    chunks = chunker.chunk_source(code, language="python")
    assert any("class Greeter" in c.content for c in chunks)


def test_class_methods_get_own_chunks(chunker: SemanticChunker) -> None:
    """Each method in a class gets its own chunk for fine-grained retrieval."""
    code = (
        "class Greeter:\n"
        "    def __init__(self, name: str) -> None:\n"
        "        self.name = name\n"
        "    def greet(self) -> str:\n"
        '        return f"Hello, {self.name}"\n'
    )
    chunks = chunker.chunk_source(code, language="python")
    all_content = " ".join(c.content for c in chunks)
    assert "def __init__" in all_content
    assert "def greet" in all_content


def test_method_chunks_have_parent_symbol(chunker: SemanticChunker) -> None:
    """Method chunks carry parent_symbol equal to the enclosing class name."""
    code = (
        "class Greeter:\n"
        "    def __init__(self, name: str) -> None:\n"
        "        self.name = name\n"
        "    def greet(self) -> str:\n"
        '        return f"Hello, {self.name}"\n'
    )
    chunks = chunker.chunk_source(code, language="python")
    method_chunks = [c for c in chunks if c.parent_symbol == "Greeter"]
    assert len(method_chunks) >= 2


def test_method_chunks_have_qualified_name(chunker: SemanticChunker) -> None:
    """Method chunks carry qualified_name of the form 'ClassName.method_name'."""
    code = "class Greeter:\n" "    def greet(self) -> str:\n" '        return "hello"\n'
    chunks = chunker.chunk_source(code, language="python")
    greet_chunk = next((c for c in chunks if c.parent_symbol == "Greeter"), None)
    assert greet_chunk is not None
    assert greet_chunk.qualified_name == "Greeter.greet"


# ---------------------------------------------------------------------------
# 9. Decorated function and class
# ---------------------------------------------------------------------------


def test_decorated_function_includes_decorator(chunker: SemanticChunker) -> None:
    """A decorated function chunk includes the decorator line."""
    code = (
        "def dec(fn): return fn\n"
        "\n"
        "@dec\n"
        "def greet(name: str) -> str:\n"
        '    return f"Hi {name}"\n'
    )
    chunks = chunker.chunk_source(code, language="python")
    greet_chunk = next((c for c in chunks if "def greet" in c.content), None)
    assert greet_chunk is not None
    assert "@dec" in greet_chunk.content


def test_decorated_class_includes_decorator(chunker: SemanticChunker) -> None:
    """A decorated class chunk includes its decorator."""
    code = "@singleton\nclass App:\n    pass\n"
    chunks = chunker.chunk_source(code, language="python")
    assert any("@singleton" in c.content for c in chunks)


# ---------------------------------------------------------------------------
# 10. Module-level code
# ---------------------------------------------------------------------------


def test_module_imports_appear_in_chunks(chunker: SemanticChunker) -> None:
    """Module-level imports are included in at least one chunk."""
    code = "import os\nfrom pathlib import Path\n"
    chunks = chunker.chunk_source(code, language="python")
    assert any("import" in c.content for c in chunks)


def test_module_level_chunks_have_chunk_kind_module(chunker: SemanticChunker) -> None:
    """Module-level chunks have chunk_kind='module'."""
    code = "import os\nCONSTANT = 42\n"
    chunks = chunker.chunk_source(code, language="python")
    assert any(c.chunk_kind == "module" for c in chunks)


def test_module_level_chunks_have_no_qualified_name(chunker: SemanticChunker) -> None:
    """Module-level chunks have qualified_name=None."""
    code = "import os\nCONSTANT = 42\n"
    chunks = chunker.chunk_source(code, language="python")
    module_chunks = [c for c in chunks if c.chunk_kind == "module"]
    assert all(c.qualified_name is None for c in module_chunks)


def test_module_level_comments_grouped_not_fragmented(
    chunker: SemanticChunker,
) -> None:
    """Section-divider comments are grouped with surrounding code, not emitted solo."""
    code = (
        "# -----------\n"
        "# Constants\n"
        "# -----------\n"
        "\n"
        "MAX = 1024\n"
        "DEBUG = False\n"
        "\n"
        "# -----------\n"
        "# Settings\n"
        "# -----------\n"
        "\n"
        "TIMEOUT = 30\n"
    )
    chunks = chunker.chunk_source(code, language="python")
    assert len(chunks) <= 3


def test_consecutive_imports_grouped(chunker: SemanticChunker) -> None:
    """Ten consecutive imports produce fewer chunks than there are imports."""
    code = "\n".join(f"import module_{i}" for i in range(10))
    chunks = chunker.chunk_source(code, language="python")
    assert len(chunks) < 10


# ---------------------------------------------------------------------------
# 11. Large function splitting
# ---------------------------------------------------------------------------


def _large_function_source() -> str:
    """Build a function that is too large for a 128-token budget."""
    body = "\n".join(
        f"    step_{i} = data[{i}] if len(data) > {i} else None  # step {i}"
        for i in range(40)
    )
    return f'def process(data: list) -> list:\n    """Process items."""\n{body}\n    return data\n'


def test_large_function_split_into_multiple_chunks(chunker: SemanticChunker) -> None:
    """A function exceeding max_tokens is split into multiple chunks."""
    chunks = chunker.chunk_source(_large_function_source(), language="python")
    func_chunks = [c for c in chunks if "def process" in c.content]
    assert len(func_chunks) >= 2


def test_large_function_continuation_flag(chunker: SemanticChunker) -> None:
    """Continuation chunks have is_continuation=True."""
    chunks = chunker.chunk_source(_large_function_source(), language="python")
    assert any(c.is_continuation for c in chunks)


def test_large_function_continuation_chunk_kind(chunker: SemanticChunker) -> None:
    """Continuation chunks have chunk_kind='continuation'."""
    chunks = chunker.chunk_source(_large_function_source(), language="python")
    cont = [c for c in chunks if c.is_continuation]
    assert cont
    assert all(c.chunk_kind == "continuation" for c in cont)


def test_large_function_overlap_header_present(chunker: SemanticChunker) -> None:
    """Every continuation chunk carries an overlap_header."""
    chunks = chunker.chunk_source(_large_function_source(), language="python")
    cont_chunks = [c for c in chunks if c.is_continuation]
    assert cont_chunks
    assert all(c.overlap_header is not None for c in cont_chunks)


def test_large_function_signature_in_continuation(chunker: SemanticChunker) -> None:
    """The function signature appears in every continuation chunk content."""
    chunks = chunker.chunk_source(_large_function_source(), language="python")
    cont_chunks = [c for c in chunks if c.is_continuation]
    assert all("def process" in c.content for c in cont_chunks)


def test_large_function_qualified_name_on_all_chunks(
    chunker: SemanticChunker,
) -> None:
    """All split chunks of a function carry the same qualified_name."""
    chunks = chunker.chunk_source(_large_function_source(), language="python")
    func_chunks = [c for c in chunks if "def process" in c.content]
    assert all(c.qualified_name == "process" for c in func_chunks)


def test_large_function_chunk_index_ascending(chunker: SemanticChunker) -> None:
    """chunk_index is non-decreasing across the split chunks."""
    chunks = chunker.chunk_source(_large_function_source(), language="python")
    func_chunks = [c for c in chunks if "def process" in c.content]
    indices = [c.chunk_index for c in func_chunks]
    assert indices == sorted(indices)


def test_first_chunk_never_a_continuation(chunker: SemanticChunker) -> None:
    """The first chunk (chunk_index=0) of any definition is not a continuation."""
    code = "class MyClass:\n    def method(self): pass\n"
    chunks = chunker.chunk_source(code, language="python")
    first_chunks = [c for c in chunks if c.chunk_index == 0]
    assert all(not c.is_continuation for c in first_chunks)


# ---------------------------------------------------------------------------
# 12. Token budget
# ---------------------------------------------------------------------------


def test_token_budget_respected_small_source(chunker: SemanticChunker) -> None:
    """All chunks from a small source stay within max_tokens."""
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    for c in chunker.chunk_source(code, language="python"):
        assert c.token_count <= chunker.max_tokens * 1.1


def test_smaller_budget_produces_more_chunks() -> None:
    """A smaller max_tokens budget produces at least as many chunks."""
    src = _large_function_source()
    large = SemanticChunker(max_tokens=1024).chunk_source(src, language="python")
    small = SemanticChunker(max_tokens=32).chunk_source(src, language="python")
    assert len(small) >= len(large)


# ---------------------------------------------------------------------------
# 13. Nested class
# ---------------------------------------------------------------------------


def test_nested_class_both_levels_present(chunker: SemanticChunker) -> None:
    """Outer and inner classes both produce chunks."""
    code = (
        "class Outer:\n"
        '    """Outer class."""\n'
        "    class Inner:\n"
        '        """Inner class."""\n'
        "        def inner_method(self) -> None:\n"
        "            pass\n"
        "    def outer_method(self) -> None:\n"
        "        pass\n"
    )
    chunks = chunker.chunk_source(code, language="python")
    all_content = " ".join(c.content for c in chunks)
    assert "class Outer" in all_content
    assert "class Inner" in all_content


# ---------------------------------------------------------------------------
# 14. Syntax error tolerance
# ---------------------------------------------------------------------------


def test_syntax_error_does_not_raise(chunker: SemanticChunker) -> None:
    """Broken source must not raise; partial results are returned."""
    result = chunker.chunk_source("def broken(\n    return 42\n", language="python")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 15. chunk_file from disk
# ---------------------------------------------------------------------------


def test_chunk_file_reads_disk(
    tmp_path: pathlib.Path, chunker: SemanticChunker
) -> None:
    """chunk_file reads a file from disk and produces chunks."""
    src = "def hello():\n    return 42\n"
    f = tmp_path / "mod.py"
    f.write_text(src, encoding="utf-8")
    chunks = chunker.chunk_file(str(f), language="python")
    assert len(chunks) >= 1
    assert all(c.file_path == str(f) for c in chunks)


def test_chunk_file_infers_language_from_extension(
    tmp_path: pathlib.Path, chunker: SemanticChunker
) -> None:
    """chunk_file infers language from the .py extension without language=."""
    f = tmp_path / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")
    chunks = chunker.chunk_file(str(f))
    assert chunks


# ---------------------------------------------------------------------------
# 16. chunk_from_tree
# ---------------------------------------------------------------------------


def test_chunk_from_tree_same_as_chunk_source(chunker: SemanticChunker) -> None:
    """chunk_from_tree and chunk_source produce identical results."""
    from src.reporag.ingestion.parser import ASTParser

    src = "class Greeter:\n    def greet(self) -> str:\n        return 'hello'\n"
    src_bytes = src.encode("utf-8")
    parser = ASTParser()
    tree = parser.parse(src_bytes, language="python")

    via_tree = chunker.chunk_from_tree(tree, "<string>", src, language="python")
    via_source = chunker.chunk_source(src, language="python")

    assert len(via_tree) == len(via_source)
    for a, b in zip(via_tree, via_source, strict=False):
        assert a.content == b.content
        assert a.start_line == b.start_line
        assert a.end_line == b.end_line


def test_chunk_from_tree_carries_file_path(chunker: SemanticChunker) -> None:
    """chunk_from_tree stores the supplied file_path in every chunk."""
    from src.reporag.ingestion.parser import ASTParser

    src = "def f(): pass\n"
    tree = ASTParser().parse(src.encode(), language="python")
    chunks = chunker.chunk_from_tree(tree, "custom/path.py", src, language="python")
    assert all(c.file_path == "custom/path.py" for c in chunks)


# ---------------------------------------------------------------------------
# 17. Unsupported language
# ---------------------------------------------------------------------------


def test_unsupported_language_raises(chunker: SemanticChunker) -> None:
    """Unsupported language raises UnsupportedLanguageError, not a silent []."""
    with pytest.raises(UnsupportedLanguageError):
        chunker.chunk_source("function hello() {}", language="javascript")


# ---------------------------------------------------------------------------
# 18. Chunker reuse across calls
# ---------------------------------------------------------------------------


def test_chunker_reusable_across_calls(chunker: SemanticChunker) -> None:
    """The same SemanticChunker instance produces correct results for different files."""
    r1 = chunker.chunk_source("def add(a, b): return a + b\n", language="python")
    r2 = chunker.chunk_source("class C:\n    pass\n", language="python")
    r3 = chunker.chunk_source("import os\nimport sys\n", language="python")

    assert any("def add" in c.content for c in r1)
    assert any("class C" in c.content for c in r2)
    assert any("import" in c.content for c in r3)


# ---------------------------------------------------------------------------
# 19. Ingestion package exports
# ---------------------------------------------------------------------------


def test_chunk_exported_from_ingestion_package() -> None:
    """Chunk is re-exported from the ingestion package __init__."""
    from src.reporag.ingestion import Chunk as ChunkAlias

    assert ChunkAlias is Chunk


def test_semantic_chunker_exported_from_ingestion_package() -> None:
    """SemanticChunker is re-exported from the ingestion package __init__."""
    from src.reporag.ingestion import SemanticChunker as SCAlias

    assert SCAlias is SemanticChunker


# ---------------------------------------------------------------------------
# 20. Integration: spec example on examples/sample_repo/app.py
# ---------------------------------------------------------------------------


def test_spec_example_on_sample_repo() -> None:
    """Reproduces the exact 'how to test locally' command from the issue spec.

    Issue 8 spec::

        chunker = SemanticChunker(max_tokens=512)
        chunks = chunker.chunk_file('examples/sample_repo/app.py')
        for c in chunks:
            print(f'[{c.start_line}-{c.end_line}] {c.parent_symbol} ({c.token_count} tokens)')
    """
    sample = (
        pathlib.Path(__file__).parent.parent.parent
        / "examples"
        / "sample_repo"
        / "app.py"
    )
    chunker = SemanticChunker(max_tokens=512)
    chunks = chunker.chunk_file(str(sample))

    assert len(chunks) >= 2
    all_content = " ".join(c.content for c in chunks)
    assert "handle_login" in all_content
    assert "handle_profile" in all_content
    assert all(str(sample) in c.file_path for c in chunks)
    # Function chunks must have qualified names
    assert any(c.qualified_name == "handle_login" for c in chunks)
    assert any(c.qualified_name == "handle_profile" for c in chunks)


# ---------------------------------------------------------------------------
# 21. Integration: real-world file (cloner.py)
# ---------------------------------------------------------------------------


def test_chunk_real_file_cloner(chunker: SemanticChunker) -> None:
    """chunk_file on cloner.py produces correct class and method chunks."""
    cloner = (
        pathlib.Path(__file__).parent.parent.parent
        / "src"
        / "reporag"
        / "ingestion"
        / "cloner.py"
    )
    chunks = chunker.chunk_file(str(cloner), language="python")

    assert len(chunks) >= 3
    all_content = " ".join(c.content for c in chunks)
    assert "RepoCloner" in all_content
    assert "clone_and_discover" in all_content
    # Methods of RepoCloner must get their own chunks with parent_symbol set
    method_chunks = [c for c in chunks if c.parent_symbol == "RepoCloner"]
    assert len(method_chunks) >= 2
    # Method chunks must have dot-qualified names
    assert all("RepoCloner." in (c.qualified_name or "") for c in method_chunks)
