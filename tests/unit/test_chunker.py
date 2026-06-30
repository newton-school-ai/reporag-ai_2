"""Unit tests for the semantic code chunker (Issue 8)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import tiktoken

from src.reporag.ingestion.chunker import CodeChunk, SemanticChunker
from src.reporag.ingestion.parser import ASTParser


@pytest.fixture(scope="module")
def token_encoding() -> tiktoken.Encoding:
    """Use a deterministic local tiktoken encoding for hermetic tests."""
    return tiktoken.Encoding(
        name="test_byte_encoding",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([idx]): idx for idx in range(256)},
        special_tokens={},
    )


@pytest.fixture
def make_chunker(
    token_encoding: tiktoken.Encoding,
) -> Callable[[int], SemanticChunker]:
    """Factory that creates chunkers without network-backed BPE loading."""

    def _make(max_tokens: int) -> SemanticChunker:
        return SemanticChunker(max_tokens=max_tokens, encoding=token_encoding)

    return _make


@pytest.fixture
def parser() -> ASTParser:
    """Parser used to verify chunk syntax."""
    return ASTParser()


def assert_valid_python_chunks(chunks: list[CodeChunk], parser: ASTParser) -> None:
    """Assert each emitted chunk parses cleanly as Python."""
    for chunk in chunks:
        tree = parser.parse(chunk.text, language="python")
        assert not parser.has_errors(tree), chunk.text


def test_empty_source_returns_no_chunks(
    make_chunker: Callable[[int], SemanticChunker],
) -> None:
    """Empty files produce no semantic chunks."""
    assert make_chunker(100).chunk_source("", file_path="empty.py") == []


def test_small_function_stays_in_one_chunk(
    make_chunker: Callable[[int], SemanticChunker],
) -> None:
    """A function below the token limit is not split."""
    code = (
        "def compute_value(x: int) -> int:\n"
        '    """Compute a value."""\n'
        "    return x + 1\n"
    )
    chunker = make_chunker(1_000)
    chunks = chunker.chunk_source(code, file_path="math.py")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == code.rstrip("\n")
    assert chunk.file_path == "math.py"
    assert chunk.start_line == 1
    assert chunk.end_line == 3
    assert chunk.parent_symbol is None
    assert chunk.language == "python"
    assert chunk.token_count == chunker.count_tokens(chunk.text)


def test_class_with_nested_methods_stays_intact_when_under_limit(
    make_chunker: Callable[[int], SemanticChunker],
    parser: ASTParser,
) -> None:
    """Classes and nested methods remain together while the class fits."""
    code = (
        "class Calculator:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n"
        "\n"
        "    def subtract(self, a, b):\n"
        "        return a - b\n"
    )
    chunks = make_chunker(1_000).chunk_source(code, file_path="calc.py")

    assert len(chunks) == 1
    assert "def add" in chunks[0].text
    assert "def subtract" in chunks[0].text
    assert_valid_python_chunks(chunks, parser)


def test_top_level_statements_are_chunked(
    make_chunker: Callable[[int], SemanticChunker],
) -> None:
    """Files without functions still produce top-level code chunks."""
    code = "# module constants\nimport os\nVALUE = 1\nNAME = os.name\n"
    chunker = make_chunker(30)
    chunks = chunker.chunk_source(code, file_path="settings.py")

    assert len(chunks) >= 2
    assert chunks[0].text.startswith("# module constants")
    assert all(chunk.parent_symbol is None for chunk in chunks)
    assert all(chunk.file_path == "settings.py" for chunk in chunks)


def test_oversized_function_splits_between_complete_statements(
    make_chunker: Callable[[int], SemanticChunker],
    parser: ASTParser,
) -> None:
    """Large functions split at body statement boundaries, not mid-statement."""
    statements = "\n".join(f"    value_{idx} = {idx}" for idx in range(6))
    code = f"def build_values():\n{statements}\n    return value_5\n"
    sizing_chunker = make_chunker(1_000)
    max_tokens = sizing_chunker.count_tokens(
        "def build_values():\n    value_0 = 0\n    value_1 = 1"
    )
    chunker = make_chunker(max_tokens)

    chunks = chunker.chunk_source(code, file_path="values.py")

    assert len(chunks) > 1
    assert_valid_python_chunks(chunks, parser)
    assert all(chunk.token_count <= int(max_tokens * 1.1) for chunk in chunks)
    assert all(chunk.text.startswith("def build_values():") for chunk in chunks)
    assert not any(
        "value_0 =" in chunk.text and "value_0 = 0" not in chunk.text
        for chunk in chunks
    )


def test_continuation_chunks_include_only_signature_overlap(
    make_chunker: Callable[[int], SemanticChunker],
) -> None:
    """Continuation chunks repeat the signature without decorators or old body."""
    code = (
        "@trace\n"
        "async def collect(items):\n"
        "    first = items[0]\n"
        "    second = items[1]\n"
        "    third = items[2]\n"
    )
    sizing_chunker = make_chunker(1_000)
    max_tokens = sizing_chunker.count_tokens(
        "@trace\nasync def collect(items):\n    first = items[0]"
    )
    chunks = make_chunker(max_tokens).chunk_source(code, file_path="collect.py")

    assert len(chunks) > 1
    assert chunks[0].text.startswith("@trace\nasync def collect(items):")
    for chunk in chunks[1:]:
        assert chunk.text.startswith("async def collect(items):\n")
        assert not chunk.text.startswith("@trace")
        assert "first = items[0]" not in chunk.text


def test_large_class_splits_into_valid_class_chunks(
    make_chunker: Callable[[int], SemanticChunker],
    parser: ASTParser,
) -> None:
    """Large classes split between complete class-body statements."""
    code = (
        "class Service:\n"
        "    def first(self):\n"
        "        return 1\n"
        "\n"
        "    def second(self):\n"
        "        return 2\n"
        "\n"
        "    def third(self):\n"
        "        return 3\n"
    )
    sizing_chunker = make_chunker(1_000)
    max_tokens = sizing_chunker.count_tokens(
        "class Service:\n    def first(self):\n        return 1"
    )
    chunks = make_chunker(max_tokens).chunk_source(code, file_path="service.py")

    assert len(chunks) == 3
    assert all(chunk.text.startswith("class Service:") for chunk in chunks)
    assert [chunk.end_line for chunk in chunks] == [3, 6, 9]
    assert_valid_python_chunks(chunks, parser)


def test_oversized_nested_method_uses_parent_symbol_metadata(
    make_chunker: Callable[[int], SemanticChunker],
    parser: ASTParser,
) -> None:
    """Nested methods split with class context and method parent metadata."""
    statements = "\n".join(
        f"        step_{idx} = self.steps[{idx}]" for idx in range(5)
    )
    code = f"class Worker:\n    def run(self):\n{statements}\n        return step_4\n"
    sizing_chunker = make_chunker(1_000)
    max_tokens = sizing_chunker.count_tokens(
        "class Worker:\n    def run(self):\n        step_0 = self.steps[0]\n        step_1 = self.steps[1]"
    )
    chunks = make_chunker(max_tokens).chunk_source(code, file_path="worker.py")

    assert len(chunks) > 1
    assert all(
        chunk.text.startswith("class Worker:\n    def run(self):") for chunk in chunks
    )
    assert all(chunk.parent_symbol == "Worker" for chunk in chunks)
    assert_valid_python_chunks(chunks, parser)


def test_token_count_uses_tiktoken_encoding(
    make_chunker: Callable[[int], SemanticChunker],
    token_encoding: tiktoken.Encoding,
) -> None:
    """Token counts come from the configured tiktoken encoding."""
    text = "def f(x):\n    return x + 1"
    chunker = make_chunker(100)
    assert chunker.count_tokens(text) == len(token_encoding.encode(text))
    assert chunker.count_tokens(text) != len(text.split())


def test_exact_token_boundary_is_not_split(
    make_chunker: Callable[[int], SemanticChunker],
) -> None:
    """A definition whose token count equals max_tokens remains whole."""
    code = "def exact():\n    return 1\n"
    sizing_chunker = make_chunker(1_000)
    exact_tokens = sizing_chunker.count_tokens(code.rstrip("\n"))

    chunks = make_chunker(exact_tokens).chunk_source(code)

    assert len(chunks) == 1
    assert chunks[0].token_count == exact_tokens


def test_syntax_errors_are_handled_gracefully(
    make_chunker: Callable[[int], SemanticChunker],
) -> None:
    """Malformed files do not crash chunking of recoverable source regions."""
    code = "def valid_one():\n    return 1\n\ndef broken(\n"

    chunks = make_chunker(1_000).chunk_source(code, file_path="broken.py")

    assert any("def valid_one" in chunk.text for chunk in chunks)
    assert all(chunk.file_path == "broken.py" for chunk in chunks)


def test_chunk_file_infers_language(
    make_chunker: Callable[[int], SemanticChunker],
    tmp_path: Path,
) -> None:
    """chunk_file reads a file once and infers Python from the extension."""
    source_file = tmp_path / "module.py"
    source_file.write_text("x = 1\n\ndef f():\n    return x\n")

    chunks = make_chunker(1_000).chunk_file(source_file)

    assert len(chunks) == 2
    assert chunks[0].file_path == str(source_file)
    assert chunks[1].language == "python"
