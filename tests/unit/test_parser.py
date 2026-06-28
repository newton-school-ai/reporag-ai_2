"""Unit tests for the tree-sitter AST parser (Issue 6).

Tests cover Python and JavaScript parsing, error tolerance, the walk()
iterator, parse_file(), and all public helper methods.  Every test
creates source code in-memory; no external network or disk access is
required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.reporag.ingestion.parser import (
    ASTNode,
    ASTParser,
    ParseError,
    UnsupportedLanguageError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parser() -> ASTParser:
    """A single ASTParser instance reused across the module."""
    return ASTParser()


# ---------------------------------------------------------------------------
# 1. Empty file
# ---------------------------------------------------------------------------


def test_parse_empty_file(parser: ASTParser) -> None:
    """Parsing an empty string returns a tree with a root node."""
    tree = parser.parse("", language="python")
    assert tree is not None
    assert tree.root_node is not None
    assert tree.root_node.type == "module"


# ---------------------------------------------------------------------------
# 2. Single function
# ---------------------------------------------------------------------------


def test_parse_single_function(parser: ASTParser) -> None:
    """Parsing a simple function produces a function_definition child."""
    src = "def hello():\n    return 42\n"
    tree = parser.parse(src, language="python")
    children = tree.root_node.children
    assert len(children) == 1
    assert children[0].type == "function_definition"


# ---------------------------------------------------------------------------
# 3. Class with methods
# ---------------------------------------------------------------------------


def test_parse_class_with_methods(parser: ASTParser) -> None:
    """A class body with methods is parsed without errors."""
    src = (
        "class Greeter:\n"
        "    def greet(self, name: str) -> str:\n"
        "        return f'Hello, {name}'\n"
        "\n"
        "    def farewell(self) -> None:\n"
        "        pass\n"
    )
    tree = parser.parse(src, language="python")
    assert tree.root_node.type == "module"
    assert not parser.has_errors(tree)
    types = [c.type for c in tree.root_node.children]
    assert "class_definition" in types


# ---------------------------------------------------------------------------
# 4. Syntax error returns partial AST (does not crash)
# ---------------------------------------------------------------------------


def test_parse_syntax_error_returns_partial_ast(parser: ASTParser) -> None:
    """Broken source yields a tree that contains ERROR nodes."""
    src = "def broken(\n    return\n"
    tree = parser.parse(src, language="python")
    assert tree is not None
    assert tree.root_node is not None
    assert parser.has_errors(tree)


def test_find_errors_returns_error_nodes(parser: ASTParser) -> None:
    """find_errors() lists ERROR/MISSING nodes from a broken file."""
    src = "def broken(\n    return\n"
    tree = parser.parse(src, language="python")
    errors = parser.find_errors(tree)
    assert len(errors) > 0
    assert all(e.type in ("ERROR", "MISSING") for e in errors)


# ---------------------------------------------------------------------------
# 5. Language-agnostic interface
# ---------------------------------------------------------------------------


def test_language_agnostic_python(parser: ASTParser) -> None:
    """parse() accepts language='python' (case-insensitive)."""
    tree = parser.parse("x = 1", language="Python")
    assert tree.root_node.type == "module"


def test_language_agnostic_javascript(parser: ASTParser) -> None:
    """parse() accepts language='javascript'."""
    src = "function greet(name) { return 'hi ' + name; }"
    tree = parser.parse(src, language="javascript")
    assert tree.root_node is not None
    assert not parser.has_errors(tree)


def test_unsupported_language_raises(parser: ASTParser) -> None:
    """Requesting an unknown language raises UnsupportedLanguageError."""
    with pytest.raises(UnsupportedLanguageError):
        parser.parse("x = 1", language="cobol")


# ---------------------------------------------------------------------------
# 6. Accepts bytes
# ---------------------------------------------------------------------------


def test_parse_accepts_bytes(parser: ASTParser) -> None:
    """parse() accepts raw bytes as well as str."""
    src = b"x = 1\n"
    tree = parser.parse(src, language="python")
    assert tree.root_node.type == "module"


# ---------------------------------------------------------------------------
# 7. walk() yields ASTNode with correct fields
# ---------------------------------------------------------------------------


def test_walk_yields_ast_nodes(parser: ASTParser) -> None:
    """walk() yields ASTNode dataclass instances."""
    tree = parser.parse("x = 1\n", language="python")
    nodes = list(parser.walk(tree))
    assert len(nodes) > 0
    assert all(isinstance(n, ASTNode) for n in nodes)


def test_walk_node_fields(parser: ASTParser) -> None:
    """ASTNode has type, text, start_line, end_line, start_col, end_col."""
    src = "def hello():\n    return 42\n"
    tree = parser.parse(src, language="python")
    # Find the function_definition node
    func_node = next(n for n in parser.walk(tree) if n.type == "function_definition")
    assert func_node.type == "function_definition"
    assert "hello" in func_node.text
    assert func_node.start_line == 1  # 1-based
    assert func_node.end_line == 2
    assert func_node.start_col == 0
    assert isinstance(func_node.end_col, int)
    assert isinstance(func_node.is_named, bool)
    assert func_node.is_named is True


def test_walk_named_only_skips_anonymous(parser: ASTParser) -> None:
    """walk(named_only=True) skips anonymous punctuation nodes."""
    tree = parser.parse("x = 1\n", language="python")
    all_nodes = list(parser.walk(tree, named_only=False))
    named_nodes = list(parser.walk(tree, named_only=True))
    # named_only should yield fewer or equal nodes
    assert len(named_nodes) <= len(all_nodes)
    assert all(n.is_named for n in named_nodes)


def test_walk_line_numbers_are_one_based(parser: ASTParser) -> None:
    """start_line of the first real node is 1 (not 0)."""
    tree = parser.parse("x = 1\n", language="python")
    nodes = list(parser.walk(tree, named_only=True))
    assert nodes[0].start_line >= 1


# ---------------------------------------------------------------------------
# 8. parse_file()
# ---------------------------------------------------------------------------


def test_parse_file_infers_language(parser: ASTParser, tmp_path: Path) -> None:
    """parse_file() infers the language from the .py extension."""
    f = tmp_path / "hello.py"
    f.write_text("def hi(): pass\n")
    tree = parser.parse_file(f)
    assert tree.root_node.type == "module"
    assert not parser.has_errors(tree)


def test_parse_file_unknown_extension_raises(parser: ASTParser, tmp_path: Path) -> None:
    """parse_file() raises UnsupportedLanguageError for unknown extension."""
    f = tmp_path / "script.rb"
    f.write_text("puts 'hello'\n")
    with pytest.raises(UnsupportedLanguageError):
        parser.parse_file(f)


def test_parse_file_missing_file_raises(parser: ASTParser, tmp_path: Path) -> None:
    """parse_file() raises ParseError if the file does not exist."""
    with pytest.raises(ParseError):
        parser.parse_file(tmp_path / "nonexistent.py")


# ---------------------------------------------------------------------------
# 9. Parser caching (same instance reused)
# ---------------------------------------------------------------------------


def test_parser_caches_language_instance(parser: ASTParser) -> None:
    """The internal parser object for 'python' is created only once."""
    parser.parse("x = 1", language="python")
    parser.parse("y = 2", language="python")
    assert "python" in parser._parsers
    p1 = parser._parsers["python"]
    parser.parse("z = 3", language="python")
    assert parser._parsers["python"] is p1  # same object


# ---------------------------------------------------------------------------
# 10. has_errors on valid source
# ---------------------------------------------------------------------------


def test_has_errors_false_for_valid_source(parser: ASTParser) -> None:
    """has_errors() returns False for syntactically correct Python."""
    tree = parser.parse("x = 1\ny = 2\n", language="python")
    assert not parser.has_errors(tree)
