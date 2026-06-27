"""Unit tests for src/reporag/ingestion/parser.py (Issue 6).

Acceptance Criteria covered:
- [x] Parses valid Python files into tree-sitter AST
- [x] Returns partial AST for files with syntax errors (not crash)
- [x] Parser interface is language-agnostic (parser.parse(source, language=lang))
- [x] Node data includes: type, text, start_line, end_line, start_col, end_col
- [x] Unit tests: empty file, single function, class with methods, syntax error
"""

from __future__ import annotations

import pytest

from reporag.ingestion.parser import (
    EXTENSION_TO_LANGUAGE,
    ASTParser,
    NodeInfo,
    ParserError,
    ParseResult,
)

# ---------------------------------------------------------------------------
# Source fixtures (module-level constants keep tests readable)
# ---------------------------------------------------------------------------

EMPTY_SOURCE = ""

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

DECORATED_FUNCTION = """\
import functools

def decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

@decorator
def add(a, b):
    return a + b
"""

MULTILINE_STRING = '''\
def doc():
    """
    This is a
    multiline docstring.
    """
    pass
'''

JAVASCRIPT_FUNCTION = """\
function greet(name) {
    return "Hello, " + name;
}
"""

JAVASCRIPT_CLASS = """\
class Animal {
    constructor(name) {
        this.name = name;
    }
    speak() {
        return this.name + " makes a sound.";
    }
}
"""

JAVASCRIPT_SYNTAX_ERROR = """\
function broken( {
    return 1;
}
"""


# ---------------------------------------------------------------------------
# 1. test_parse_empty_file
# ---------------------------------------------------------------------------


def test_parse_empty_file():
    """Empty source parses to a module root with no errors and empty source bytes."""
    parser = ASTParser()
    result = parser.parse(EMPTY_SOURCE, language="python")

    assert isinstance(result, ParseResult)
    assert result.language == "python"
    assert result.has_error is False
    assert result.root_node.type == "module"
    assert result.source == b""
    # An empty Python file has no named children
    named = [c for c in result.root_node.children if c.is_named]
    assert named == []


# ---------------------------------------------------------------------------
# 2. test_parse_single_function
# ---------------------------------------------------------------------------


def test_parse_single_function():
    """A single function definition produces a function_definition child of module."""
    parser = ASTParser()
    result = parser.parse(SIMPLE_FUNCTION, language="python")

    assert result.has_error is False
    assert result.root_node.type == "module"

    named_children = [c for c in result.root_node.children if c.is_named]
    assert len(named_children) >= 1
    assert named_children[0].type == "function_definition"


# ---------------------------------------------------------------------------
# 3. test_parse_class_with_methods
# ---------------------------------------------------------------------------


def test_parse_class_with_methods():
    """A class body with __init__, greet, and farewell methods all appear in AST."""
    parser = ASTParser()
    result = parser.parse(CLASS_WITH_METHODS, language="python")

    assert result.has_error is False

    # Top-level node must be a class_definition
    named_top = [c for c in result.root_node.children if c.is_named]
    assert any(c.type == "class_definition" for c in named_top)

    # Walk entire tree and collect all function_definition nodes
    func_nodes = [
        n for n in parser.walk(result) if n.node_type == "function_definition"
    ]
    # __init__, greet, farewell -> at least 3
    assert len(func_nodes) >= 3


# ---------------------------------------------------------------------------
# 4. test_parse_async_function
# ---------------------------------------------------------------------------


def test_parse_async_function():
    """async def produces a function_definition with async keyword in its text."""
    parser = ASTParser()
    result = parser.parse(ASYNC_FUNCTION, language="python")

    assert result.has_error is False

    # Find the async function node -- tree-sitter models it as function_definition
    # with an "async" keyword child, not a separate node type.
    func_nodes = [
        n for n in parser.walk(result) if n.node_type == "function_definition"
    ]
    assert len(func_nodes) >= 1

    # The text of the function node must contain the async keyword
    assert any("async" in fn.text for fn in func_nodes)


# ---------------------------------------------------------------------------
# 5. test_parse_syntax_error_returns_partial_ast
# ---------------------------------------------------------------------------


def test_parse_syntax_error_returns_partial_ast():
    """Broken source must NOT raise -- has_error=True with a valid module root."""
    parser = ASTParser()
    result = parser.parse(SYNTAX_ERROR_SOURCE, language="python")

    # Must not raise anything
    assert isinstance(result, ParseResult)
    assert result.has_error is True
    assert result.root_node is not None
    assert result.root_node.type == "module"
    # Tree must still have children (partial AST, not empty)
    assert result.root_node.child_count >= 1


# ---------------------------------------------------------------------------
# 6. test_parse_nested_classes
# ---------------------------------------------------------------------------


def test_parse_nested_classes():
    """Nested class_definition nodes are reachable via depth-first walk."""
    parser = ASTParser()
    result = parser.parse(NESTED_CLASSES, language="python")

    assert result.has_error is False

    all_types = [n.node_type for n in parser.walk(result)]
    # Outer and Inner -> at least 2 class_definition nodes
    assert all_types.count("class_definition") >= 2


# ---------------------------------------------------------------------------
# 7. test_language_agnostic_interface_javascript
# ---------------------------------------------------------------------------


def test_language_agnostic_interface_javascript():
    """Identical parse() call works for JavaScript; root is 'program'."""
    parser = ASTParser()
    result = parser.parse(JAVASCRIPT_FUNCTION, language="javascript")

    assert isinstance(result, ParseResult)
    assert result.language == "javascript"
    assert result.has_error is False
    assert result.root_node.type == "program"

    func_nodes = [
        n for n in parser.walk(result) if n.node_type == "function_declaration"
    ]
    assert len(func_nodes) >= 1


# ---------------------------------------------------------------------------
# 8. test_node_info_fields_types
# ---------------------------------------------------------------------------


def test_node_info_fields_types():
    """Every NodeInfo in a walk has the correct field types."""
    parser = ASTParser()
    result = parser.parse(SIMPLE_FUNCTION, language="python")

    nodes = list(parser.walk(result))
    assert len(nodes) > 0

    for node_info in nodes:
        assert isinstance(node_info, NodeInfo)
        assert isinstance(node_info.node_type, str) and node_info.node_type
        assert isinstance(node_info.text, str)
        assert isinstance(node_info.start_line, int)
        assert isinstance(node_info.end_line, int)
        assert isinstance(node_info.start_col, int)
        assert isinstance(node_info.end_col, int)
        assert isinstance(node_info.is_error, bool)
        assert isinstance(node_info.is_named, bool)
        assert isinstance(node_info.child_count, int)


# ---------------------------------------------------------------------------
# 9. test_node_info_position_constraints
# ---------------------------------------------------------------------------


def test_node_info_position_constraints():
    """All node positions must satisfy basic ordering invariants."""
    parser = ASTParser()
    result = parser.parse(CLASS_WITH_METHODS, language="python")

    for node_info in parser.walk(result):
        assert node_info.start_line >= 0, "start_line must be non-negative"
        assert node_info.end_line >= node_info.start_line, "end_line >= start_line"
        assert node_info.start_col >= 0, "start_col must be non-negative"
        assert node_info.end_col >= 0, "end_col must be non-negative"
        assert node_info.child_count >= 0, "child_count must be non-negative"


# ---------------------------------------------------------------------------
# 10. test_node_info_line_positions_exact
# ---------------------------------------------------------------------------


def test_node_info_line_positions_exact():
    """Function node in SIMPLE_FUNCTION must span lines 0-1, starting at col 0."""
    parser = ASTParser()
    result = parser.parse(SIMPLE_FUNCTION, language="python")

    func_nodes = [
        n for n in parser.walk(result) if n.node_type == "function_definition"
    ]
    assert len(func_nodes) == 1

    fn = func_nodes[0]
    assert fn.start_line == 0, f"Expected start_line=0, got {fn.start_line}"
    assert fn.end_line == 1, f"Expected end_line=1, got {fn.end_line}"
    assert fn.start_col == 0, f"Expected start_col=0, got {fn.start_col}"


# ---------------------------------------------------------------------------
# 11. test_node_text_content
# ---------------------------------------------------------------------------


def test_node_text_content():
    """NodeInfo.text must contain the actual source text spanned by the node."""
    parser = ASTParser()
    result = parser.parse(SIMPLE_FUNCTION, language="python")

    func_nodes = [
        n for n in parser.walk(result) if n.node_type == "function_definition"
    ]
    assert len(func_nodes) == 1
    fn = func_nodes[0]

    # The function node text must include the function signature and body
    assert "hello" in fn.text
    assert "return" in fn.text
    assert "42" in fn.text


# ---------------------------------------------------------------------------
# 12. test_unsupported_language_raises
# ---------------------------------------------------------------------------


def test_unsupported_language_raises():
    """Asking for an unknown language raises ParserError with helpful message."""
    parser = ASTParser()
    with pytest.raises(ParserError, match="Unsupported language 'cobol'"):
        parser.parse("x = 1", language="cobol")


def test_unsupported_language_error_lists_available():
    """ParserError message lists the languages that ARE available."""
    parser = ASTParser()
    with pytest.raises(ParserError, match="python"):
        parser.parse("", language="brainfuck")


# ---------------------------------------------------------------------------
# 13. test_parse_bytes_input
# ---------------------------------------------------------------------------


def test_parse_bytes_input():
    """parse() accepts raw bytes; source field is preserved as-is."""
    parser = ASTParser()
    source_bytes = SIMPLE_FUNCTION.encode("utf-8")
    result = parser.parse(source_bytes, language="python")

    assert result.has_error is False
    assert result.source == source_bytes


# ---------------------------------------------------------------------------
# 14. test_parse_str_input_stored_as_bytes
# ---------------------------------------------------------------------------


def test_parse_str_input_stored_as_bytes():
    """When source is a str, result.source is its UTF-8 byte encoding."""
    parser = ASTParser()
    result = parser.parse(SIMPLE_FUNCTION, language="python")
    assert result.source == SIMPLE_FUNCTION.encode("utf-8")


# ---------------------------------------------------------------------------
# 15. test_supported_languages
# ---------------------------------------------------------------------------


def test_supported_languages():
    """supported_languages includes at least python and javascript, is sorted."""
    parser = ASTParser()
    langs = parser.supported_languages
    assert "python" in langs
    assert "javascript" in langs
    assert langs == sorted(langs), "supported_languages must be sorted"


# ---------------------------------------------------------------------------
# 16. test_walk_named_only_fewer_nodes
# ---------------------------------------------------------------------------


def test_walk_named_only_fewer_nodes():
    """named_only=True yields strictly fewer nodes than named_only=False."""
    parser = ASTParser()
    result = parser.parse(CLASS_WITH_METHODS, language="python")

    all_nodes = list(parser.walk(result, named_only=False))
    named_nodes = list(parser.walk(result, named_only=True))

    assert len(named_nodes) < len(
        all_nodes
    ), "named_only=True should skip anonymous punctuation/keyword nodes"
    assert all(
        n.is_named for n in named_nodes
    ), "Every node yielded with named_only=True must have is_named=True"


# ---------------------------------------------------------------------------
# 17. test_walk_order_is_preorder
# ---------------------------------------------------------------------------


def test_walk_order_is_preorder():
    """Parents appear before their children in the walk output (pre-order DFS)."""
    parser = ASTParser()
    result = parser.parse(SIMPLE_FUNCTION, language="python")

    nodes = list(parser.walk(result, named_only=True))
    # module must come before function_definition
    types = [n.node_type for n in nodes]
    assert types.index("module") < types.index("function_definition")


# ---------------------------------------------------------------------------
# 18. test_parse_file
# ---------------------------------------------------------------------------


def test_parse_file(tmp_path):
    """parse_file() reads from disk and returns the same result as parse()."""
    src_file = tmp_path / "example.py"
    src_file.write_text(SIMPLE_FUNCTION, encoding="utf-8")

    parser = ASTParser()
    result = parser.parse_file(str(src_file), language="python")

    assert result.has_error is False
    assert result.root_node.type == "module"
    assert result.source == SIMPLE_FUNCTION.encode("utf-8")


def test_parse_file_not_found_raises(tmp_path):
    """parse_file() raises OSError when the path does not exist."""
    parser = ASTParser()
    with pytest.raises(OSError):
        parser.parse_file(str(tmp_path / "nonexistent.py"), language="python")


# ---------------------------------------------------------------------------
# 19. test_register_language
# ---------------------------------------------------------------------------


def test_register_language():
    """register_language() adds a grammar; empty-registry instance can parse it."""
    import tree_sitter_python as tspython
    from tree_sitter import Language

    parser = ASTParser(languages={})  # no grammars
    assert "python" not in parser.supported_languages

    parser.register_language("python", Language(tspython.language()))
    assert "python" in parser.supported_languages

    result = parser.parse(SIMPLE_FUNCTION, language="python")
    assert result.has_error is False


def test_register_language_replaces_cached_parser():
    """Re-registering a language invalidates the old cached Parser."""
    import tree_sitter_python as tspython
    from tree_sitter import Language

    lang = Language(tspython.language())
    parser = ASTParser(languages={"python": lang})

    # Force Parser creation
    _ = parser.parse(SIMPLE_FUNCTION, language="python")
    assert "python" in parser._parsers  # noqa: SLF001

    # Re-register the same grammar -- cached parser must be dropped
    parser.register_language("python", lang)
    assert "python" not in parser._parsers  # noqa: SLF001


# ---------------------------------------------------------------------------
# 20. test_error_node_flagged_in_walk
# ---------------------------------------------------------------------------


def test_error_node_flagged_in_walk():
    """At least one ERROR node is reachable via walk() when source is broken."""
    parser = ASTParser()
    result = parser.parse(SYNTAX_ERROR_SOURCE, language="python")
    assert result.has_error is True

    error_nodes = [n for n in parser.walk(result) if n.is_error]
    assert len(error_nodes) >= 1, "Expected at least one ERROR node in broken source"


# ---------------------------------------------------------------------------
# 21. test_javascript_syntax_error_partial_ast
# ---------------------------------------------------------------------------


def test_javascript_syntax_error_partial_ast():
    """JavaScript broken source returns has_error=True with a valid 'program' root."""
    parser = ASTParser()
    result = parser.parse(JAVASCRIPT_SYNTAX_ERROR, language="javascript")

    assert result.has_error is True
    assert result.root_node.type == "program"
    assert result.root_node is not None


# ---------------------------------------------------------------------------
# 22. test_node_count_property
# ---------------------------------------------------------------------------


def test_node_count_property():
    """ParseResult.node_count must be > 0 for non-empty source."""
    parser = ASTParser()
    result = parser.parse(CLASS_WITH_METHODS, language="python")
    assert result.node_count > 0

    empty_result = parser.parse(EMPTY_SOURCE, language="python")
    # Even an empty file has at least the root 'module' node
    assert empty_result.node_count >= 1


# ---------------------------------------------------------------------------
# 23. test_decorated_function
# ---------------------------------------------------------------------------


def test_decorated_function():
    """Decorated functions are parsed; decorator node type appears in the walk."""
    parser = ASTParser()
    result = parser.parse(DECORATED_FUNCTION, language="python")

    assert result.has_error is False
    all_types = {n.node_type for n in parser.walk(result)}
    # tree-sitter models @decorator as 'decorated_definition'
    assert "decorated_definition" in all_types


# ---------------------------------------------------------------------------
# 24. test_multiline_string_no_error
# ---------------------------------------------------------------------------


def test_multiline_string_no_error():
    """A function with a multiline docstring parses without errors."""
    parser = ASTParser()
    result = parser.parse(MULTILINE_STRING, language="python")
    assert result.has_error is False


# ---------------------------------------------------------------------------
# 25. test_javascript_class
# ---------------------------------------------------------------------------


def test_javascript_class():
    """A JavaScript class body is parsed with class_declaration node in the AST."""
    parser = ASTParser()
    result = parser.parse(JAVASCRIPT_CLASS, language="javascript")

    assert result.has_error is False
    all_types = {n.node_type for n in parser.walk(result)}
    assert "class_declaration" in all_types


# ---------------------------------------------------------------------------
# 26. test_child_count_field
# ---------------------------------------------------------------------------


def test_child_count_field():
    """NodeInfo.child_count matches the actual number of children in the raw tree."""
    parser = ASTParser()
    result = parser.parse(SIMPLE_FUNCTION, language="python")

    # Verify root node
    root_info = next(parser.walk(result))
    assert root_info.node_type == "module"
    assert root_info.child_count == result.root_node.child_count


# ---------------------------------------------------------------------------
# 27. test_parser_is_reusable_across_calls
# ---------------------------------------------------------------------------


def test_parser_is_reusable_across_calls():
    """The same ASTParser instance can parse multiple sources without corruption."""
    parser = ASTParser()

    r1 = parser.parse(SIMPLE_FUNCTION, language="python")
    r2 = parser.parse(CLASS_WITH_METHODS, language="python")
    r3 = parser.parse(JAVASCRIPT_FUNCTION, language="javascript")

    assert r1.has_error is False
    assert r2.has_error is False
    assert r3.has_error is False

    assert r1.root_node.type == "module"
    assert r2.root_node.type == "module"
    assert r3.root_node.type == "program"

    # Results must be independent of each other
    assert r1.source != r2.source


# ---------------------------------------------------------------------------
# 28. test_node_count_is_precomputed_and_stable
# ---------------------------------------------------------------------------


def test_node_count_is_precomputed_and_stable():
    """node_count is an int field set once at parse time; repeated reads are O(1)."""
    parser = ASTParser()
    result = parser.parse(CLASS_WITH_METHODS, language="python")

    # Must be a positive integer
    assert isinstance(result.node_count, int)
    assert result.node_count > 0

    # Repeated access returns the same value (no re-traversal)
    first = result.node_count
    second = result.node_count
    assert first == second

    # Empty file still has at least the root 'module' node
    empty = parser.parse("", language="python")
    assert empty.node_count >= 1


# ---------------------------------------------------------------------------
# 29. test_parse_file_auto_detects_language
# ---------------------------------------------------------------------------


def test_parse_file_auto_detects_python(tmp_path):
    """parse_file() with no language= infers 'python' from .py extension."""
    src = tmp_path / "module.py"
    src.write_text(SIMPLE_FUNCTION, encoding="utf-8")

    parser = ASTParser()
    result = parser.parse_file(str(src))  # no language= kwarg

    assert result.language == "python"
    assert result.has_error is False
    assert result.root_node.type == "module"


def test_parse_file_auto_detects_javascript(tmp_path):
    """parse_file() with no language= infers 'javascript' from .js extension."""
    src = tmp_path / "app.js"
    src.write_text(JAVASCRIPT_FUNCTION, encoding="utf-8")

    parser = ASTParser()
    result = parser.parse_file(str(src))  # no language= kwarg

    assert result.language == "javascript"
    assert result.has_error is False
    assert result.root_node.type == "program"


def test_parse_file_unknown_extension_raises(tmp_path):
    """parse_file() with unrecognised extension and no language= raises ParserError."""
    src = tmp_path / "config.toml"
    src.write_text("key = 'value'\n", encoding="utf-8")

    parser = ASTParser()
    with pytest.raises(ParserError, match=".toml"):
        parser.parse_file(str(src))  # no language= and .toml not in map


# ---------------------------------------------------------------------------
# 30. test_extension_to_language_map
# ---------------------------------------------------------------------------


def test_extension_to_language_map():
    """EXTENSION_TO_LANGUAGE covers the core extensions and maps to correct names."""
    assert EXTENSION_TO_LANGUAGE[".py"] == "python"
    assert EXTENSION_TO_LANGUAGE[".js"] == "javascript"
    # All values must be non-empty strings
    for ext, lang in EXTENSION_TO_LANGUAGE.items():
        assert ext.startswith("."), f"Extension '{ext}' must start with '.'"
        assert (
            isinstance(lang, str) and lang
        ), f"Language for '{ext}' must be a non-empty string"


# ---------------------------------------------------------------------------
# 31. test_parse_result_repr
# ---------------------------------------------------------------------------


def test_parse_result_repr():
    """ParseResult.__repr__ is human-readable and includes key fields."""
    parser = ASTParser()
    result = parser.parse(SIMPLE_FUNCTION, language="python")

    r = repr(result)
    assert "ParseResult" in r
    assert "python" in r
    assert "has_error=False" in r
    assert "node_count=" in r
    # source bytes should be truncated, not dumped in full
    assert "def hello" in r  # preview of first 40 chars
