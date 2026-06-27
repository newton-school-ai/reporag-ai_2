"""Unit tests for tree-sitter AST parser."""

import pytest

from src.reporag.ingestion.parser import (
    ASTParser,
    UnsupportedLanguageError,
)


def test_parse_empty_file():
    parser = ASTParser()

    result = parser.parse("")

    assert result.language == "python"
    assert result.tree.root_node.type == "module"
    assert result.has_errors is False
    assert result.error_count == 0


def test_parse_single_function():
    parser = ASTParser()

    result = parser.parse("def hello():\n" "    return 42\n")

    node_types = [node.type for node in result.nodes]

    assert "function_definition" in node_types
    assert "identifier" in node_types
    assert "return_statement" in node_types


def test_parse_class_with_methods():
    parser = ASTParser()

    result = parser.parse(
        "class User:\n" "    def greet(self):\n" "        return 'hello'\n"
    )

    node_types = [node.type for node in result.nodes]

    assert "class_definition" in node_types
    assert "function_definition" in node_types


def test_parse_async_function():
    parser = ASTParser()

    result = parser.parse("async def fetch():\n" "    return 1\n")

    node_types = [node.type for node in result.nodes]

    assert "function_definition" in node_types


def test_parse_syntax_error_returns_partial_ast():
    parser = ASTParser()

    result = parser.parse("def hello(\n" "    return 42")

    assert result.has_errors is True
    assert result.error_count > 0


def test_parse_nested_classes():
    parser = ASTParser()

    result = parser.parse("class A:\n" "    class B:\n" "        pass\n")

    class_nodes = [node for node in result.nodes if node.type == "class_definition"]

    assert len(class_nodes) == 2


def test_language_agnostic_interface():
    parser = ASTParser()

    result = parser.parse(
        "def hello():\n" "    return 42\n",
        language="python",
    )

    assert result.language == "python"


def test_parse_unsupported_language():
    parser = ASTParser()

    with pytest.raises(UnsupportedLanguageError):
        parser.parse("print('hello')", language="rust")


def test_parse_file(tmp_path):
    source = tmp_path / "hello.py"

    source.write_text("def hello():\n" "    return 42\n")

    parser = ASTParser()

    result = parser.parse_file(source)

    assert result.language == "python"
    assert result.has_errors is False
