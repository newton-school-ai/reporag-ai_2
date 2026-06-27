"""Unit tests for src/reporag/ingestion/parser.py (Issue 6).

Acceptance Criteria:
- [x] Parses valid Python files into tree-sitter AST
- [x] Returns partial AST for files with syntax errors (not crash)
- [x] Parser interface is language-agnostic (parser.parse(source, language=lang))
- [x] Node data includes: type, text, start_line, end_line, start_col, end_col
- [x] Unit tests: empty file, single function, class with methods, syntax error
"""

from __future__ import annotations

import pytest

from reporag.ingestion.parser import ASTParser, NodeInfo, ParserError, ParseResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_FUNCTION = """\
def hello():
    return 42
"""

CLASS_WITH_METHODS = """\
class Greeter:
    \"\"\"A simple greeter class.\"\"\"

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}!"

    @staticmethod
    def farewell() -> str:
        return "Goodbye!"
"""

ASYNC_FUNCTION = """\
import asyncio

async def fetch_data(url: str) -> dict:
    await asyncio.sleep(0)
    return {"url": url}
"""

SYNTAX_ERROR_SOURCE = """\
def broken(
    return 42
"""

NESTED_CLASSES = """\
class Outer:
    class Inner:
        def method(self):
            pass
"""

JAVASCRIPT_FUNCTION = """\
function greet(name) {
    return "Hello, " + name;
}
"""


# ---------------------------------------------------------------------------
# 1. test_parse_empty_file
# ---------------------------------------------------------------------------


def test_parse_empty_file():
    """Parsing an empty string should return a module node with no errors."""
    parser = ASTParser()
    result = parser.parse("", language="python")

    assert isinstance(result, ParseResult)
    assert result.language == "python"
    assert result.has_error is False
    assert result.root_node.type == "module"
    assert result.source == b""


# ---------------------------------------------------------------------------
# 2. test_parse_single_function
# ---------------------------------------------------------------------------


def test_parse_single_function():
    """A simple function definition parses without errors."""
    parser = ASTParser()
    result = parser.parse(SIMPLE_FUNCTION, language="python")

    assert result.has_error is False
    assert result.root_node.type == "module"
    # The root module should have at least one child: the function_definition
    named_children = [c for c in result.root_node.children if c.is_named]
    assert len(named_children) >= 1
    assert named_children[0].type == "function_definition"


# ---------------------------------------------------------------------------
# 3. test_parse_class_with_methods
# ---------------------------------------------------------------------------


def test_parse_class_with_methods():
    """A class with multiple methods parses correctly."""
    parser = ASTParser()
    result = parser.parse(CLASS_WITH_METHODS, language="python")

    assert result.has_error is False
    named_children = [c for c in result.root_node.children if c.is_named]
    assert any(c.type == "class_definition" for c in named_children)


# ---------------------------------------------------------------------------
# 4. test_parse_async_function
# ---------------------------------------------------------------------------


def test_parse_async_function():
    """An async function definition is parsed without errors."""
    parser = ASTParser()
    result = parser.parse(ASYNC_FUNCTION, language="python")

    assert result.has_error is False
    # Walk the tree and look for an async_function_definition node
    all_types = {info.node_type for info in parser.walk(result)}
    assert (
        "decorated_definition" in all_types
        or "function_definition" in all_types
        or any("async" in t for t in all_types)
    )


# ---------------------------------------------------------------------------
# 5. test_parse_syntax_error_returns_partial_ast
# ---------------------------------------------------------------------------


def test_parse_syntax_error_returns_partial_ast():
    """Broken source returns a ParseResult with has_error=True, not an exception."""
    parser = ASTParser()
    # Must NOT raise
    result = parser.parse(SYNTAX_ERROR_SOURCE, language="python")

    assert isinstance(result, ParseResult)
    assert result.has_error is True
    # Root node must still be valid (partial AST)
    assert result.root_node is not None
    assert result.root_node.type == "module"


# ---------------------------------------------------------------------------
# 6. test_parse_nested_classes
# ---------------------------------------------------------------------------


def test_parse_nested_classes():
    """Nested class definitions are reachable via tree walk."""
    parser = ASTParser()
    result = parser.parse(NESTED_CLASSES, language="python")

    assert result.has_error is False
    all_types = [info.node_type for info in parser.walk(result)]
    assert all_types.count("class_definition") >= 2


# ---------------------------------------------------------------------------
# 7. test_language_agnostic_interface
# ---------------------------------------------------------------------------


def test_language_agnostic_interface_javascript():
    """The same ASTParser.parse interface works for JavaScript."""
    parser = ASTParser()
    result = parser.parse(JAVASCRIPT_FUNCTION, language="javascript")

    assert isinstance(result, ParseResult)
    assert result.language == "javascript"
    assert result.has_error is False
    assert result.root_node.type == "program"


# ---------------------------------------------------------------------------
# 8. test_node_info_fields
# ---------------------------------------------------------------------------


def test_node_info_fields():
    """NodeInfo must expose type, text, start_line, end_line, start_col, end_col."""
    parser = ASTParser()
    result = parser.parse(SIMPLE_FUNCTION, language="python")

    nodes = list(parser.walk(result))
    assert len(nodes) > 0

    for node_info in nodes:
        assert isinstance(node_info, NodeInfo)
        assert isinstance(node_info.node_type, str)
        assert isinstance(node_info.text, str)
        assert isinstance(node_info.start_line, int)
        assert isinstance(node_info.end_line, int)
        assert isinstance(node_info.start_col, int)
        assert isinstance(node_info.end_col, int)
        assert node_info.start_line >= 0
        assert node_info.end_line >= node_info.start_line


# ---------------------------------------------------------------------------
# 9. test_node_info_line_positions
# ---------------------------------------------------------------------------


def test_node_info_line_positions():
    """Function node should start at line 0 and end at line 1 (0-indexed)."""
    parser = ASTParser()
    result = parser.parse(SIMPLE_FUNCTION, language="python")

    # Find the function_definition node
    func_nodes = [
        n for n in parser.walk(result) if n.node_type == "function_definition"
    ]
    assert len(func_nodes) == 1
    fn = func_nodes[0]
    assert fn.start_line == 0
    assert fn.end_line == 1
    assert fn.start_col == 0


# ---------------------------------------------------------------------------
# 10. test_unsupported_language_raises
# ---------------------------------------------------------------------------


def test_unsupported_language_raises():
    """Requesting an unsupported language raises ParserError."""
    parser = ASTParser()
    with pytest.raises(ParserError, match="Unsupported language"):
        parser.parse("x = 1", language="cobol")


# ---------------------------------------------------------------------------
# 11. test_parse_bytes_input
# ---------------------------------------------------------------------------


def test_parse_bytes_input():
    """Parser accepts raw bytes as source input."""
    parser = ASTParser()
    source_bytes = SIMPLE_FUNCTION.encode("utf-8")
    result = parser.parse(source_bytes, language="python")

    assert result.has_error is False
    assert result.source == source_bytes


# ---------------------------------------------------------------------------
# 12. test_supported_languages
# ---------------------------------------------------------------------------


def test_supported_languages():
    """ASTParser.supported_languages includes at least python and javascript."""
    parser = ASTParser()
    langs = parser.supported_languages
    assert "python" in langs
    assert "javascript" in langs


# ---------------------------------------------------------------------------
# 13. test_walk_named_only
# ---------------------------------------------------------------------------


def test_walk_named_only():
    """walk(named_only=True) should yield fewer nodes than walk(named_only=False)."""
    parser = ASTParser()
    result = parser.parse(CLASS_WITH_METHODS, language="python")

    all_nodes = list(parser.walk(result, named_only=False))
    named_nodes = list(parser.walk(result, named_only=True))

    assert len(named_nodes) < len(all_nodes)
    # All nodes from named_only=True must have is_named=True
    assert all(n.is_named for n in named_nodes)


# ---------------------------------------------------------------------------
# 14. test_parse_file
# ---------------------------------------------------------------------------


def test_parse_file(tmp_path):
    """parse_file() reads a file from disk and parses it correctly."""
    src_file = tmp_path / "example.py"
    src_file.write_text(SIMPLE_FUNCTION, encoding="utf-8")

    parser = ASTParser()
    result = parser.parse_file(str(src_file), language="python")

    assert result.has_error is False
    assert result.root_node.type == "module"


# ---------------------------------------------------------------------------
# 15. test_register_language
# ---------------------------------------------------------------------------


def test_register_language():
    """register_language() makes a new grammar available for parsing."""
    import tree_sitter_python as tspython
    from tree_sitter import Language

    parser = ASTParser(languages={})  # Start with empty registry
    assert "python" not in parser.supported_languages

    parser.register_language("python", Language(tspython.language()))
    assert "python" in parser.supported_languages

    result = parser.parse(SIMPLE_FUNCTION, language="python")
    assert result.has_error is False


# ---------------------------------------------------------------------------
# 16. test_error_node_flagged_in_walk
# ---------------------------------------------------------------------------


def test_error_node_flagged_in_walk():
    """When source has syntax errors, at least one node in the walk is an ERROR node."""
    parser = ASTParser()
    result = parser.parse(SYNTAX_ERROR_SOURCE, language="python")

    error_nodes = [n for n in parser.walk(result) if n.is_error]
    assert len(error_nodes) >= 1
