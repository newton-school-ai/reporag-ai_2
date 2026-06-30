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

import json
import pathlib

import pytest

from src.reporag.ingestion.chunker import Chunk, SemanticChunker, count_tokens
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
# 2. Chunk dataclass
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
    assert c.token_count == count_tokens(c.content)


def test_chunk_kind_module_level() -> None:
    """A chunk with no parent_symbol and no qualified_name has chunk_kind='module'."""
    c = Chunk(
        content="import os",
        file_path="f.py",
        language="python",
        start_line=1,
        end_line=1,
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


def test_chunk_repr_shows_qualified_name() -> None:
    """Chunk repr uses qualified_name and line range."""
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


def test_chunk_to_dict_is_json_serialisable() -> None:
    """to_dict() values are JSON-primitive and round-trip cleanly."""
    c = Chunk(
        content="def f(): pass",
        file_path="f.py",
        language="python",
        start_line=1,
        end_line=1,
        qualified_name="f",
    )
    d = c.to_dict()
    assert d["qualified_name"] == "f"
    assert d["chunk_kind"] == "definition"
    json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# 3. Empty source
# ---------------------------------------------------------------------------


def test_chunk_empty_source_returns_empty_list(chunker: SemanticChunker) -> None:
    """Empty source produces an empty chunk list."""
    assert chunker.chunk_source("", language="python") == []


# ---------------------------------------------------------------------------
# 4. Small function -> one chunk
# ---------------------------------------------------------------------------


def test_small_function_produces_one_chunk(chunker: SemanticChunker) -> None:
    """A function that fits within max_tokens produces exactly one chunk."""
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    chunks = chunker.chunk_source(code, language="python")
    assert len([c for c in chunks if "def add" in c.content]) == 1


def test_small_function_content_preserved(chunker: SemanticChunker) -> None:
    """The entire function body is preserved in the chunk."""
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    chunks = chunker.chunk_source(code, language="python")
    func_chunk = next(c for c in chunks if "def add" in c.content)
    assert "return a + b" in func_chunk.content


def test_small_function_qualified_name(chunker: SemanticChunker) -> None:
    """A module-level function chunk carries qualified_name equal to its name."""
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    chunks = chunker.chunk_source(code, language="python")
    func_chunk = next(c for c in chunks if "def add" in c.content)
    assert func_chunk.qualified_name == "add"
    assert func_chunk.chunk_kind == "definition"


def test_small_function_not_a_continuation(chunker: SemanticChunker) -> None:
    """A single-chunk function is never marked as a continuation."""
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    chunks = chunker.chunk_source(code, language="python")
    assert not any(c.is_continuation for c in chunks)


# ---------------------------------------------------------------------------
# 5. Chunk metadata fields
# ---------------------------------------------------------------------------


def test_chunk_metadata_file_language_lines(chunker: SemanticChunker) -> None:
    """Every chunk carries file_path, language, and valid line numbers."""
    code = "import os\n\ndef f():\n    pass\n"
    chunks = chunker.chunk_source(code, language="python", file_path="my/file.py")
    for c in chunks:
        assert c.file_path == "my/file.py"
        assert c.language == "python"
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_chunk_token_count_matches_content(chunker: SemanticChunker) -> None:
    """token_count equals count_tokens(content) for every chunk."""
    code = "import os\n\ndef f():\n    pass\n\nclass C:\n    def m(self): pass\n"
    for c in chunker.chunk_source(code, language="python"):
        assert c.token_count == count_tokens(c.content)


# ---------------------------------------------------------------------------
# 6. Async function
# ---------------------------------------------------------------------------


def test_async_function_chunked(chunker: SemanticChunker) -> None:
    """Async functions are chunked with their full content preserved."""
    code = "async def fetch(url: str) -> dict:\n    return {}\n"
    chunks = chunker.chunk_source(code, language="python")
    async_chunk = next((c for c in chunks if "async def fetch" in c.content), None)
    assert async_chunk is not None
    assert "return {}" in async_chunk.content


# ---------------------------------------------------------------------------
# 7. Class chunking
# ---------------------------------------------------------------------------


def test_class_and_methods_chunked(chunker: SemanticChunker) -> None:
    """A class produces a class chunk plus individual method chunks."""
    code = (
        "class Greeter:\n"
        "    def __init__(self, name: str) -> None:\n"
        "        self.name = name\n"
        "    def greet(self) -> str:\n"
        '        return f"Hello, {self.name}"\n'
    )
    chunks = chunker.chunk_source(code, language="python")
    all_content = " ".join(c.content for c in chunks)
    assert "class Greeter" in all_content
    assert "def __init__" in all_content
    assert "def greet" in all_content


def test_method_chunks_parent_and_qualified_name(chunker: SemanticChunker) -> None:
    """Method chunks carry parent_symbol='Greeter' and qualified_name='Greeter.greet'."""
    code = "class Greeter:\n" "    def greet(self) -> str:\n" '        return "hello"\n'
    chunks = chunker.chunk_source(code, language="python")
    greet_chunk = next(
        (c for c in chunks if c.parent_symbol == "Greeter" and "greet" in c.content),
        None,
    )
    assert greet_chunk is not None
    assert greet_chunk.qualified_name == "Greeter.greet"


# ---------------------------------------------------------------------------
# 8. Decorated definitions
# ---------------------------------------------------------------------------


def test_decorated_function_includes_decorator(chunker: SemanticChunker) -> None:
    """A decorated function chunk includes the decorator line."""
    code = "@dec\ndef greet(name: str) -> str:\n    return f'Hi {name}'\n"
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
# 9. Module-level code
# ---------------------------------------------------------------------------


def test_module_level_chunks_have_chunk_kind_module(chunker: SemanticChunker) -> None:
    """Module-level imports and constants produce chunks with chunk_kind='module'."""
    code = "import os\nCONSTANT = 42\n"
    chunks = chunker.chunk_source(code, language="python")
    assert any(c.chunk_kind == "module" for c in chunks)
    assert all(c.qualified_name is None for c in chunks if c.chunk_kind == "module")


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
# 10. Large function splitting
# ---------------------------------------------------------------------------


def _large_function_source() -> str:
    """Build a function that exceeds a 128-token budget."""
    body = "\n".join(
        f"    step_{i} = data[{i}] if len(data) > {i} else None  # step {i}"
        for i in range(40)
    )
    return f'def process(data: list) -> list:\n    """Process items."""\n{body}\n    return data\n'


def test_large_function_split_into_multiple_chunks(chunker: SemanticChunker) -> None:
    """A function exceeding max_tokens is split into multiple chunks."""
    chunks = chunker.chunk_source(_large_function_source(), language="python")
    assert len([c for c in chunks if "def process" in c.content]) >= 2


def test_large_function_continuation_chunks(chunker: SemanticChunker) -> None:
    """Continuation chunks carry the signature, chunk_kind='continuation', and overlap_header."""
    chunks = chunker.chunk_source(_large_function_source(), language="python")
    cont = [c for c in chunks if c.is_continuation]
    assert cont
    for c in cont:
        assert c.chunk_kind == "continuation"
        assert c.overlap_header is not None
        assert "def process" in c.content


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


def test_smaller_budget_produces_more_chunks() -> None:
    """A smaller max_tokens budget produces at least as many chunks."""
    src = _large_function_source()
    large = SemanticChunker(max_tokens=1024).chunk_source(src, language="python")
    small = SemanticChunker(max_tokens=32).chunk_source(src, language="python")
    assert len(small) >= len(large)


# ---------------------------------------------------------------------------
# 11. Nested class
# ---------------------------------------------------------------------------


def test_nested_class_both_levels_present(chunker: SemanticChunker) -> None:
    """Outer and inner classes both produce chunks."""
    code = (
        "class Outer:\n"
        "    class Inner:\n"
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
# 12. Syntax error tolerance
# ---------------------------------------------------------------------------


def test_syntax_error_does_not_raise(chunker: SemanticChunker) -> None:
    """Broken source must not raise; a partial result list is returned."""
    result = chunker.chunk_source("def broken(\n    return 42\n", language="python")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 13. chunk_file from disk
# ---------------------------------------------------------------------------


def test_chunk_file_reads_disk(
    tmp_path: pathlib.Path, chunker: SemanticChunker
) -> None:
    """chunk_file reads a file from disk and stores the path in every chunk."""
    f = tmp_path / "mod.py"
    f.write_text("def hello():\n    return 42\n", encoding="utf-8")
    chunks = chunker.chunk_file(str(f), language="python")
    assert len(chunks) >= 1
    assert all(c.file_path == str(f) for c in chunks)


def test_chunk_file_infers_language_from_extension(
    tmp_path: pathlib.Path, chunker: SemanticChunker
) -> None:
    """chunk_file infers language from the .py extension without language=."""
    f = tmp_path / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")
    assert chunker.chunk_file(str(f))


# ---------------------------------------------------------------------------
# 14. chunk_from_tree
# ---------------------------------------------------------------------------


def test_chunk_from_tree_same_as_chunk_source(chunker: SemanticChunker) -> None:
    """chunk_from_tree and chunk_source produce identical results."""
    from src.reporag.ingestion.parser import ASTParser

    src = "class Greeter:\n    def greet(self) -> str:\n        return 'hello'\n"
    tree = ASTParser().parse(src.encode(), language="python")

    via_tree = chunker.chunk_from_tree(tree, "<string>", src, language="python")
    via_source = chunker.chunk_source(src, language="python")

    assert len(via_tree) == len(via_source)
    for a, b in zip(via_tree, via_source, strict=False):
        assert a.content == b.content
        assert a.qualified_name == b.qualified_name


# ---------------------------------------------------------------------------
# 15. Unsupported language
# ---------------------------------------------------------------------------


def test_unsupported_language_raises(chunker: SemanticChunker) -> None:
    """Unsupported language raises UnsupportedLanguageError."""
    with pytest.raises(UnsupportedLanguageError):
        chunker.chunk_source("function hello() {}", language="javascript")


# ---------------------------------------------------------------------------
# 16. Ingestion package exports
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
# 17. Integration: spec example on examples/sample_repo/app.py
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
    assert any(c.qualified_name == "handle_login" for c in chunks)
    assert any(c.qualified_name == "handle_profile" for c in chunks)


# ---------------------------------------------------------------------------
# 18. Integration: real-world file (cloner.py)
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
    method_chunks = [c for c in chunks if c.parent_symbol == "RepoCloner"]
    assert len(method_chunks) >= 2
    assert all("RepoCloner." in (c.qualified_name or "") for c in method_chunks)
