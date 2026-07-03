"""Unit tests for the CallGraphBuilder (Issue 9).

Acceptance criteria covered:
- [x] Direct function calls resolved to target symbol
- [x] Method calls (self.method, obj.method)
- [x] Cross-file calls resolved via import aliases
- [x] Recursive calls detected
- [x] Edge metadata: caller, callee, call_site_file, call_site_line
- [x] Chained attribute calls (a.b.c())
- [x] Constructor calls (MyClass())
- [x] Module-level calls are ignored (no enclosing function)
- [x] Empty / no-call source returns empty list
- [x] Aliased imports resolved correctly
- [x] from-imports resolved correctly
"""

from __future__ import annotations

import pytest

from src.reporag.graph.call_graph import (
    CallEdge,
    CallGraphBuilder,
    _build_import_alias_map,
    _extract_call_name,
    _resolve_callee,
)
from src.reporag.ingestion.parser import ASTParser
from src.reporag.ingestion.symbol_extractor import SymbolExtractor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def builder() -> CallGraphBuilder:
    """Provide a fresh CallGraphBuilder instance."""
    return CallGraphBuilder()


@pytest.fixture
def parser() -> ASTParser:
    return ASTParser()


@pytest.fixture
def extractor(parser: ASTParser) -> SymbolExtractor:
    return SymbolExtractor(parser=parser)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build(source_map: dict[str, str]) -> list[CallEdge]:
    """One-shot build from {file_path: source} using build_from_source."""
    return CallGraphBuilder().build_from_source(source_map)


def _edges_for(source: str, file_path: str = "test.py") -> list[CallEdge]:
    """Build call edges for a single-file source snippet."""
    return _build({file_path: source})


# ===========================================================================
# 1. Direct function call
# ===========================================================================


def test_direct_call_edge_exists(builder: CallGraphBuilder) -> None:
    """A direct call foo()->bar() produces a caller->callee edge."""
    source = "def foo():\n" "    bar()\n" "\n" "def bar():\n" "    pass\n"
    edges = _edges_for(source)
    assert any(e.caller == "foo" and e.callee == "bar" for e in edges)


def test_direct_call_edge_metadata(builder: CallGraphBuilder) -> None:
    """Edge metadata (file path, line number) is accurate for a direct call."""
    source = "def alpha():\n" "    beta()\n" "\n" "def beta():\n" "    pass\n"
    edges = _edges_for(source, file_path="/repo/mod.py")
    edge = next(e for e in edges if e.caller == "alpha" and e.callee == "beta")
    assert edge.call_site_file == "/repo/mod.py"
    assert edge.call_site_line == 2


# ===========================================================================
# 2. Method calls -- self.method
# ===========================================================================


def test_self_method_call(builder: CallGraphBuilder) -> None:
    """self.method() call is captured with the full dotted callee name."""
    source = (
        "class Worker:\n"
        "    def run(self):\n"
        "        self.process()\n"
        "\n"
        "    def process(self):\n"
        "        pass\n"
    )
    edges = _edges_for(source)
    assert any(e.caller == "Worker.run" and e.callee == "self.process" for e in edges)


def test_obj_method_call(builder: CallGraphBuilder) -> None:
    """obj.method() call produces a dotted callee name."""
    source = "def orchestrate(client):\n" "    client.connect()\n" "    client.send()\n"
    edges = _edges_for(source)
    callees = {e.callee for e in edges if e.caller == "orchestrate"}
    assert "client.connect" in callees
    assert "client.send" in callees


# ===========================================================================
# 3. Chained attribute calls
# ===========================================================================


def test_chained_attribute_call(builder: CallGraphBuilder) -> None:
    """a.b.c() is captured as callee 'a.b.c'."""
    source = "def do_work():\n" "    obj.sub.action()\n"
    edges = _edges_for(source)
    assert any(e.caller == "do_work" and e.callee == "obj.sub.action" for e in edges)


# ===========================================================================
# 4. Constructor call
# ===========================================================================


def test_constructor_call(builder: CallGraphBuilder) -> None:
    """MyClass() is treated as a call with callee 'MyClass'."""
    source = "def factory():\n" "    return MyClass()\n"
    edges = _edges_for(source)
    assert any(e.caller == "factory" and e.callee == "MyClass" for e in edges)


# ===========================================================================
# 5. Recursive call
# ===========================================================================


def test_recursive_call(builder: CallGraphBuilder) -> None:
    """A function calling itself produces a self-loop edge."""
    source = "def countdown(n):\n" "    if n > 0:\n" "        countdown(n - 1)\n"
    edges = _edges_for(source)
    assert any(e.caller == "countdown" and e.callee == "countdown" for e in edges)


# ===========================================================================
# 6. Cross-file calls via import resolution
# ===========================================================================


def test_cross_file_bare_import(builder: CallGraphBuilder) -> None:
    """'import math' -> math.sin() resolves to callee 'math.sin'."""
    source = "import math\n" "\n" "def compute():\n" "    return math.sin(1.0)\n"
    edges = _edges_for(source)
    assert any(e.caller == "compute" and e.callee == "math.sin" for e in edges)


def test_cross_file_aliased_module_import(builder: CallGraphBuilder) -> None:
    """'import numpy as np' -> np.zeros() resolves to callee 'numpy.zeros'."""
    source = (
        "import numpy as np\n" "\n" "def make_array():\n" "    return np.zeros(10)\n"
    )
    edges = _edges_for(source)
    assert any(e.caller == "make_array" and e.callee == "numpy.zeros" for e in edges)


def test_cross_file_from_import(builder: CallGraphBuilder) -> None:
    """'from math import sin' -> sin() resolves to callee 'math.sin'."""
    source = "from math import sin\n" "\n" "def compute():\n" "    return sin(1.0)\n"
    edges = _edges_for(source)
    assert any(e.caller == "compute" and e.callee == "math.sin" for e in edges)


def test_cross_file_from_import_aliased(builder: CallGraphBuilder) -> None:
    """'from typing import List as L' -> L() resolves to callee 'typing.List'."""
    source = (
        "from typing import List as L\n" "\n" "def get_list():\n" "    return L()\n"
    )
    edges = _edges_for(source)
    # The extractor stores import_source as "typing.List" for aliased from-imports
    edge = next(
        (e for e in edges if e.caller == "get_list" and "List" in e.callee), None
    )
    assert edge is not None


def test_cross_file_two_files(builder: CallGraphBuilder) -> None:
    """Function in file A calling a function imported from file B is captured."""
    source_a = "from utils import helper\n" "\n" "def main():\n" "    helper()\n"
    source_b = "def helper():\n" "    pass\n"
    edges = _build({"a.py": source_a, "b.py": source_b})
    assert any(e.caller == "main" and e.callee == "utils.helper" for e in edges)


# ===========================================================================
# 7. Module-level calls are ignored
# ===========================================================================


def test_module_level_call_ignored(builder: CallGraphBuilder) -> None:
    """Calls at module scope (not inside a function) are not added to the graph."""
    source = "setup()\n" "\n" "def init():\n" "    prepare()\n"
    edges = _edges_for(source)
    # 'setup' is at module level -- no caller, must be absent
    assert not any(e.callee == "setup" for e in edges)
    # 'prepare' is inside init -- must be present
    assert any(e.caller == "init" and e.callee == "prepare" for e in edges)


# ===========================================================================
# 8. Empty / no-call sources
# ===========================================================================


def test_empty_source_returns_no_edges(builder: CallGraphBuilder) -> None:
    """Empty source yields an empty edge list."""
    assert _edges_for("") == []


def test_source_with_no_calls(builder: CallGraphBuilder) -> None:
    """Source with functions but no calls yields no edges."""
    source = (
        "def foo():\n" "    x = 1\n" "    return x\n" "\n" "def bar():\n" "    pass\n"
    )
    assert _edges_for(source) == []


def test_empty_symbols_returns_no_edges(builder: CallGraphBuilder) -> None:
    """build_from_symbols with empty inputs returns empty list."""
    from src.reporag.ingestion.parser import ASTParser

    parser = ASTParser()
    tree = parser.parse("def foo(): pass")
    assert builder.build_from_symbols([], {"f.py": tree}) == []
    assert builder.build_from_symbols([], {}) == []


# ===========================================================================
# 9. Multiple calls inside same function
# ===========================================================================


def test_multiple_calls_same_function(builder: CallGraphBuilder) -> None:
    """Multiple calls inside one function produce multiple edges."""
    source = "def pipeline():\n" "    load()\n" "    transform()\n" "    save()\n"
    edges = _edges_for(source)
    callees = {e.callee for e in edges if e.caller == "pipeline"}
    assert {"load", "transform", "save"} == callees


# ===========================================================================
# 10. Nested functions
# ===========================================================================


def test_nested_function_caller(builder: CallGraphBuilder) -> None:
    """Calls inside a nested (inner) function are attributed to the inner function."""
    source = "def outer():\n" "    def inner():\n" "        helper()\n" "    inner()\n"
    edges = _edges_for(source)
    # helper() is called by inner
    assert any(
        e.caller == "outer.<locals>.inner" and e.callee == "helper" for e in edges
    )
    # inner() is called by outer
    assert any(e.caller == "outer" and e.callee == "inner" for e in edges)


# ===========================================================================
# 11. Async function
# ===========================================================================


def test_async_function_call(builder: CallGraphBuilder) -> None:
    """Calls inside async functions are captured correctly."""
    source = "async def fetch():\n" "    data = await get_data()\n" "    return data\n"
    edges = _edges_for(source)
    assert any(e.caller == "fetch" and e.callee == "get_data" for e in edges)


# ===========================================================================
# 12. Method inside class -- qualified caller name
# ===========================================================================


def test_method_caller_qualified_name(builder: CallGraphBuilder) -> None:
    """Method calls use the fully qualified class.method caller name."""
    source = (
        "class Engine:\n"
        "    def start(self):\n"
        "        self._init()\n"
        "\n"
        "    def _init(self):\n"
        "        pass\n"
    )
    edges = _edges_for(source)
    assert any(e.caller == "Engine.start" for e in edges)


# ===========================================================================
# 13. CallEdge dataclass field checks
# ===========================================================================


def test_call_edge_fields(builder: CallGraphBuilder) -> None:
    """CallEdge contains all required fields with correct types."""
    source = "def a():\n    b()\n"
    edges = _edges_for(source, "/some/file.py")
    assert edges, "Expected at least one edge"
    e = edges[0]
    assert isinstance(e.caller, str)
    assert isinstance(e.callee, str)
    assert isinstance(e.call_site_file, str)
    assert isinstance(e.call_site_line, int)
    assert isinstance(e.raw_call_text, str)
    assert e.call_site_line >= 1


# ===========================================================================
# 14. Unit tests for internal helpers
# ===========================================================================


class TestExtractCallName:
    """Tests for _extract_call_name helper."""

    def _call_node_for(self, source: str):
        parser = ASTParser()
        tree = parser.parse(source)
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "call":
                return node
            stack.extend(reversed(node.children))
        return None

    def test_identifier_call(self) -> None:
        node = self._call_node_for("def f():\n    foo()\n")
        assert _extract_call_name(node) == "foo"

    def test_attribute_call(self) -> None:
        node = self._call_node_for("def f():\n    self.run()\n")
        assert _extract_call_name(node) == "self.run"

    def test_chained_attribute_call(self) -> None:
        node = self._call_node_for("def f():\n    a.b.c()\n")
        assert _extract_call_name(node) == "a.b.c"


class TestResolveCallee:
    """Tests for _resolve_callee helper."""

    def test_bare_from_import(self) -> None:
        alias_map = {"sin": "math.sin"}
        assert _resolve_callee("sin", alias_map) == "math.sin"

    def test_aliased_module_root(self) -> None:
        alias_map = {"np": "numpy"}
        assert _resolve_callee("np.zeros", alias_map) == "numpy.zeros"

    def test_no_alias_unchanged(self) -> None:
        assert _resolve_callee("self.run", {}) == "self.run"

    def test_bare_import_module(self) -> None:
        alias_map = {"os": "os"}
        assert _resolve_callee("os.path.join", alias_map) == "os.path.join"

    def test_deep_chain_with_alias(self) -> None:
        alias_map = {"pd": "pandas"}
        assert (
            _resolve_callee("pd.DataFrame.from_dict", alias_map)
            == "pandas.DataFrame.from_dict"
        )


class TestBuildImportAliasMap:
    """Tests for _build_import_alias_map helper."""

    def _import_syms(self, source: str) -> list:
        extractor = SymbolExtractor()
        return [s for s in extractor.extract_from_source(source) if s.type == "import"]

    def test_simple_import(self) -> None:
        syms = self._import_syms("import os\n")
        m = _build_import_alias_map(syms)
        assert m["os"] == "os"

    def test_aliased_import(self) -> None:
        syms = self._import_syms("import numpy as np\n")
        m = _build_import_alias_map(syms)
        assert m["np"] == "numpy"

    def test_from_import(self) -> None:
        syms = self._import_syms("from math import sin\n")
        m = _build_import_alias_map(syms)
        assert m["sin"] == "math.sin"

    def test_wildcard_excluded(self) -> None:
        syms = self._import_syms("from os import *\n")
        m = _build_import_alias_map(syms)
        assert "*" not in m
