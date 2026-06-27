"""Unit tests for tree-sitter AST parser."""

import pytest

from src.reporag.ingestion.parser import (
    ASTParser,
    UnsupportedLanguageError,
)


def test_parse_empty_file():
    parser = ASTParser()

    tree = parser.parse("")

    assert tree.root_node.type == "module"
    assert parser.has_errors(tree) is False
    assert len(parser.find_errors(tree)) == 0


def test_parse_single_function():
    parser = ASTParser()

    tree = parser.parse("def hello():\n" "    return 42\n")

    node_types = [node.type for node in parser.walk(tree, named_only=True)]

    assert "function_definition" in node_types
    assert "identifier" in node_types
    assert "return_statement" in node_types


def test_parse_class_with_methods():
    parser = ASTParser()

    tree = parser.parse(
        "class User:\n" "    def greet(self):\n" "        return 'hello'\n"
    )

    node_types = [node.type for node in parser.walk(tree, named_only=True)]

    assert "class_definition" in node_types
    assert "function_definition" in node_types


def test_parse_async_function():
    parser = ASTParser()

    tree = parser.parse("async def fetch():\n" "    return 1\n")

    node_types = [node.type for node in parser.walk(tree, named_only=True)]

    assert "function_definition" in node_types


def test_parse_syntax_error_returns_partial_ast():
    parser = ASTParser()

    tree = parser.parse("def hello(\n" "    return 42")

    assert parser.has_errors(tree) is True
    assert len(parser.find_errors(tree)) > 0


def test_parse_nested_classes():
    parser = ASTParser()

    tree = parser.parse("class A:\n" "    class B:\n" "        pass\n")

    class_nodes = [
        node
        for node in parser.walk(tree, named_only=True)
        if node.type == "class_definition"
    ]

    assert len(class_nodes) == 2


def test_language_agnostic_interface():
    parser = ASTParser()

    tree = parser.parse(
        "def hello():\n" "    return 42\n",
        language="python",
    )

    assert tree.root_node.type == "module"
    assert parser.has_errors(tree) is False


def test_parse_unsupported_language():
    parser = ASTParser()

    with pytest.raises(UnsupportedLanguageError):
        parser.parse("print('hello')", language="rust")


def test_parse_non_ascii_source():
    parser = ASTParser()

    tree = parser.parse('x = "hello world"\n')

    assert not parser.has_errors(tree)

    string_nodes = [
        node for node in parser.walk(tree, named_only=True) if node.type == "string"
    ]

    assert len(string_nodes) == 1
    assert "hello world" in string_nodes[0].text


def test_parse_file(tmp_path):
    source = tmp_path / "hello.py"

    source.write_text("def hello():\n" "    return 42\n")

    parser = ASTParser()

    tree = parser.parse_file(source)

    assert tree.root_node.type == "module"
    assert parser.has_errors(tree) is False


def test_parse_accepts_bytes_input():
    parser = ASTParser()
    tree = parser.parse(b"def hello():\n    return 42\n")
    assert tree.root_node.type == "module"
    assert parser.has_errors(tree) is False


def test_parser_instance_is_cached():
    parser = ASTParser()
    parser.parse("x = 1")
    cached = parser._parsers["python"]
    parser.parse("y = 2")
    assert parser._parsers["python"] is cached


def test_parse_file_missing_file_raises():
    parser = ASTParser()
    with pytest.raises(FileNotFoundError):
        parser.parse_file("/nonexistent/path/does_not_exist.py")


def test_parse_javascript():
    parser = ASTParser()
    tree = parser.parse("function hello() { return 42; }", language="javascript")
    assert tree.root_node.type == "program"
    assert parser.has_errors(tree) is False
