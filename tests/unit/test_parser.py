"""Unit tests for the tree-sitter AST parser (Issue 6).

Verifies language-agnostic parsing into tree-sitter trees, structured node
walking with line/column metadata, error-tolerant partial ASTs for broken
source, and file-based parsing with extension-driven language inference.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tree_sitter import Tree

from src.reporag.ingestion.parser import (
    ASTNode,
    ASTParser,
    ParseError,
    UnsupportedLanguageError,
)


@pytest.fixture
def parser() -> ASTParser:
    """Return a fresh parser instance for each test."""
    return ASTParser()


def _named_types(parser: ASTParser, tree: Tree) -> list[str]:
    """Collect the node types of every named node in a tree."""
    return [node.type for node in parser.walk(tree, named_only=True)]


def test_parse_empty_file(parser: ASTParser) -> None:
    """An empty source produces a valid, error-free module with no children."""
    tree = parser.parse("", language="python")

    assert isinstance(tree, Tree)
    assert tree.root_node.type == "module"
    assert tree.root_node.children == []
    assert parser.has_errors(tree) is False


def test_parse_single_function(parser: ASTParser) -> None:
    """A single function parses to one top-level function_definition node."""
    tree = parser.parse("def hello():\n    return 42\n", language="python")

    # Matches the usage shown in the issue: tree.root_node.children.
    children = tree.root_node.children
    assert len(children) == 1
    assert children[0].type == "function_definition"
    assert "function_definition" in _named_types(parser, tree)
    assert parser.has_errors(tree) is False


def test_parse_class_with_methods(parser: ASTParser) -> None:
    """A class with methods exposes the class and its nested functions."""
    source = (
        "class Greeter:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "\n"
        "    def greet(self):\n"
        "        return f'hi {self.name}'\n"
    )
    tree = parser.parse(source, language="python")

    named = _named_types(parser, tree)
    assert "class_definition" in named
    # __init__ and greet are both function_definition nodes nested in the class.
    assert named.count("function_definition") == 2
    assert parser.has_errors(tree) is False


def test_parse_async_function(parser: ASTParser) -> None:
    """Async functions parse without errors as function_definition nodes."""
    tree = parser.parse("async def fetch():\n    await do_work()\n", language="python")

    assert parser.has_errors(tree) is False
    assert "function_definition" in _named_types(parser, tree)
    # The async keyword is preserved as an anonymous node in the full walk.
    all_types = [node.type for node in parser.walk(tree)]
    assert "async" in all_types


def test_parse_nested_classes(parser: ASTParser) -> None:
    """Nested classes are all discoverable via the walk traversal."""
    source = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def method(self):\n"
        "            return 1\n"
    )
    tree = parser.parse(source, language="python")

    named = _named_types(parser, tree)
    assert named.count("class_definition") == 2
    assert named.count("function_definition") == 1
    assert parser.has_errors(tree) is False


def test_parse_syntax_error_returns_partial_ast(parser: ASTParser) -> None:
    """Broken source yields a partial tree flagged with errors, not a crash."""
    # A valid function followed by a broken one: the parser recovers the valid
    # structure while still flagging the error region.
    source = "def ok():\n    return 1\n\ndef broken(:\n    x =\n"
    tree = parser.parse(source, language="python")

    assert isinstance(tree, Tree)
    assert parser.has_errors(tree) is True

    errors = parser.find_errors(tree)
    assert len(errors) >= 1
    assert all(node.is_error for node in errors)
    # Despite the error, the partial tree still surfaces the valid function.
    assert "function_definition" in [node.type for node in parser.walk(tree)]


def test_node_data_includes_position_metadata(parser: ASTParser) -> None:
    """Walked nodes carry type, text, and 1-based line / 0-based column data."""
    tree = parser.parse("x = 1\ny = 2\n", language="python")

    func = next(
        node for node in parser.walk(tree, named_only=True) if node.type == "assignment"
    )
    assert isinstance(func, ASTNode)
    assert func.type == "assignment"
    assert func.text == "x = 1"
    assert func.start_line == 1  # 1-based line numbering
    assert func.end_line == 1
    assert func.start_col == 0  # 0-based column numbering
    assert func.end_col == 5


def test_multiline_node_line_range(parser: ASTParser) -> None:
    """A function spanning multiple lines reports the full line range."""
    tree = parser.parse("def outer():\n    return 1\n", language="python")

    func = next(
        node
        for node in parser.walk(tree, named_only=True)
        if node.type == "function_definition"
    )
    assert func.start_line == 1
    assert func.end_line == 2


def test_language_agnostic_interface(parser: ASTParser) -> None:
    """The same parse(source, language) call handles a second language."""
    py_tree = parser.parse("def f():\n    pass\n", language="python")
    js_tree = parser.parse("function f() { return 1; }\n", language="javascript")

    assert py_tree.root_node.type == "module"
    assert js_tree.root_node.type == "program"
    assert "function_definition" in _named_types(parser, py_tree)
    assert "function_declaration" in _named_types(parser, js_tree)
    assert "python" in parser.supported_languages
    assert "javascript" in parser.supported_languages


def test_language_name_is_case_insensitive(parser: ASTParser) -> None:
    """Language names are normalized so casing does not matter."""
    tree = parser.parse("x = 1\n", language="PYTHON")
    assert tree.root_node.type == "module"
    assert parser.has_errors(tree) is False


def test_bytes_source_is_accepted(parser: ASTParser) -> None:
    """Source provided as bytes parses identically to a string."""
    tree = parser.parse(b"def f():\n    pass\n", language="python")
    assert "function_definition" in _named_types(parser, tree)


def test_unsupported_language_raises(parser: ASTParser) -> None:
    """Requesting a language with no grammar raises UnsupportedLanguageError."""
    with pytest.raises(UnsupportedLanguageError):
        parser.parse("SELECT 1;", language="cobol")

    # UnsupportedLanguageError is a ParseError subclass.
    assert issubclass(UnsupportedLanguageError, ParseError)


def test_parse_file_infers_language_from_extension(
    parser: ASTParser, tmp_path: Path
) -> None:
    """parse_file infers the language from the file extension."""
    file_path = tmp_path / "module.py"
    file_path.write_text("def loaded():\n    return True\n")

    tree = parser.parse_file(file_path)
    assert parser.has_errors(tree) is False
    assert "function_definition" in _named_types(parser, tree)


def test_parse_file_unknown_extension_raises(parser: ASTParser, tmp_path: Path) -> None:
    """parse_file raises when the extension maps to no known language."""
    file_path = tmp_path / "notes.txt"
    file_path.write_text("just some text")

    with pytest.raises(UnsupportedLanguageError):
        parser.parse_file(file_path)


def test_parse_file_explicit_language_override(
    parser: ASTParser, tmp_path: Path
) -> None:
    """An explicit language overrides extension inference in parse_file."""
    file_path = tmp_path / "script.unknown"
    file_path.write_text("x = 1\n")

    tree = parser.parse_file(file_path, language="python")
    assert parser.has_errors(tree) is False


def test_parser_instance_is_reusable_across_languages(parser: ASTParser) -> None:
    """One parser instance parses multiple languages and caches grammars."""
    parser.parse("x = 1\n", language="python")
    parser.parse("var x = 1;\n", language="javascript")

    # Re-parsing reuses the cached parser objects without error.
    assert parser.has_errors(parser.parse("y = 2\n", language="python")) is False
    assert (
        parser.has_errors(parser.parse("var y = 2;\n", language="javascript")) is False
    )
