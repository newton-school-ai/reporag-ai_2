"""Unit tests for the CallGraphBuilder (Issue 9).

Covers the acceptance criteria -- direct calls, method calls (self / obj),
cross-file resolution via imports, and recursion -- plus constructor calls,
module-scope callers, wildcard imports, unresolved external calls, and the
public :class:`CallEdge` contract.
"""

from __future__ import annotations

import pytest

from src.reporag.graph.call_graph import (
    MODULE_SCOPE,
    CallEdge,
    CallGraphBuilder,
)


@pytest.fixture
def builder() -> CallGraphBuilder:
    """Provide a fresh CallGraphBuilder for each test."""
    return CallGraphBuilder()


def _edge(edges: list[CallEdge], caller: str, callee: str) -> CallEdge:
    """Return the single edge matching *caller* -> *callee* (fails if absent)."""
    matches = [e for e in edges if e.caller == caller and e.callee == callee]
    assert matches, f"no edge {caller} -> {callee} in {[str(e) for e in edges]}"
    assert len(matches) == 1, f"expected one {caller} -> {callee}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Direct (same-file) function calls
# ---------------------------------------------------------------------------


def test_direct_function_call(builder: CallGraphBuilder) -> None:
    """A direct call resolves to a same-file function with a local edge."""
    code = (
        "def helper():\n" "    return 1\n" "\n" "def main():\n" "    return helper()\n"
    )
    edges = builder.build_from_sources({"m.py": code})
    edge = _edge(edges, "main", "helper")
    assert edge.call_type == "function"
    assert edge.resolution == "local"
    assert edge.resolved is True
    assert edge.caller_file == "m.py"
    assert edge.callee_file == "m.py"
    assert edge.call_site_line == 5
    assert edge.is_recursive is False


def test_direct_call_records_precise_line(builder: CallGraphBuilder) -> None:
    """Two call sites of the same target produce two edges with distinct lines."""
    code = (
        "def helper():\n"
        "    return 1\n"
        "\n"
        "def main():\n"
        "    helper()\n"
        "    return helper()\n"
    )
    edges = builder.build_from_sources({"m.py": code})
    lines = sorted(
        e.call_site_line for e in edges if e.caller == "main" and e.callee == "helper"
    )
    assert lines == [5, 6]


def test_nested_helper_call_resolves_locally(builder: CallGraphBuilder) -> None:
    """A call to a uniquely named nested function resolves within the file."""
    code = (
        "def outer():\n"
        "    def inner():\n"
        "        return 1\n"
        "    return inner()\n"
    )
    edges = builder.build_from_sources({"m.py": code})
    edge = _edge(edges, "outer", "outer.<locals>.inner")
    assert edge.resolution == "local"
    assert edge.call_type == "function"


# ---------------------------------------------------------------------------
# Recursion
# ---------------------------------------------------------------------------


def test_recursive_call_flagged(builder: CallGraphBuilder) -> None:
    """A function calling itself yields a self-edge flagged recursive."""
    code = (
        "def factorial(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return factorial(n - 1)\n"
    )
    edges = builder.build_from_sources({"m.py": code})
    edge = _edge(edges, "factorial", "factorial")
    assert edge.is_recursive is True
    assert edge.resolution == "local"
    assert edge.caller_file == edge.callee_file == "m.py"


# ---------------------------------------------------------------------------
# Method calls (self / cls / obj)
# ---------------------------------------------------------------------------


def test_self_method_call(builder: CallGraphBuilder) -> None:
    """``self.method()`` resolves against the enclosing class."""
    code = (
        "class Service:\n"
        "    def run(self):\n"
        "        return self.step()\n"
        "    def step(self):\n"
        "        return 1\n"
    )
    edges = builder.build_from_sources({"svc.py": code})
    edge = _edge(edges, "Service.run", "Service.step")
    assert edge.call_type == "method"
    assert edge.resolution == "local"
    assert edge.callee_file == "svc.py"


def test_cls_method_call(builder: CallGraphBuilder) -> None:
    """``cls.method()`` resolves like ``self`` against the enclosing class."""
    code = (
        "class Factory:\n"
        "    @classmethod\n"
        "    def create(cls):\n"
        "        return cls.build()\n"
        "    @classmethod\n"
        "    def build(cls):\n"
        "        return 1\n"
    )
    edges = builder.build_from_sources({"f.py": code})
    edge = _edge(edges, "Factory.create", "Factory.build")
    assert edge.call_type == "method"
    assert edge.resolution == "local"


def test_self_method_disambiguated_by_enclosing_class(
    builder: CallGraphBuilder,
) -> None:
    """Same-named methods in different classes resolve to the caller's class."""
    code = (
        "class A:\n"
        "    def run(self):\n"
        "        return self.work()\n"
        "    def work(self):\n"
        "        return 'a'\n"
        "\n"
        "class B:\n"
        "    def run(self):\n"
        "        return self.work()\n"
        "    def work(self):\n"
        "        return 'b'\n"
    )
    edges = builder.build_from_sources({"m.py": code})
    _edge(edges, "A.run", "A.work")
    _edge(edges, "B.run", "B.work")
    # No cross-class leakage.
    assert not any(e.caller == "A.run" and e.callee == "B.work" for e in edges)


def test_obj_method_inferred_from_constructor(builder: CallGraphBuilder) -> None:
    """``obj = User(); obj.method()`` resolves via the receiver's inferred type."""
    code = (
        "class Widget:\n"
        "    def render(self):\n"
        "        return 1\n"
        "\n"
        "def draw():\n"
        "    w = Widget()\n"
        "    return w.render()\n"
    )
    edges = builder.build_from_sources({"m.py": code})
    edge = _edge(edges, "draw", "Widget.render")
    assert edge.resolution == "inferred"
    assert edge.call_type == "method"
    assert edge.callee_file == "m.py"


def test_obj_method_inferred_from_annotation(builder: CallGraphBuilder) -> None:
    """A parameter annotation types the receiver so ``obj.method()`` resolves."""
    code = (
        "class Widget:\n"
        "    def render(self):\n"
        "        return 1\n"
        "\n"
        "def draw(w: Widget):\n"
        "    return w.render()\n"
    )
    edges = builder.build_from_sources({"m.py": code})
    edge = _edge(edges, "draw", "Widget.render")
    assert edge.resolution == "inferred"
    assert edge.call_type == "method"


def test_obj_method_untyped_receiver_stays_unresolved(
    builder: CallGraphBuilder,
) -> None:
    """An untyped receiver is never guessed by method name -- no false positive.

    Even though ``Widget.render`` is the only ``render`` in the project, an
    untyped ``w`` must not be attributed to it.
    """
    code = (
        "class Widget:\n"
        "    def render(self):\n"
        "        return 1\n"
        "\n"
        "def draw(w):\n"
        "    return w.render()\n"
    )
    edges = builder.build_from_sources({"m.py": code}, include_unresolved=True)
    edge = _edge(edges, "draw", "render")
    assert edge.resolution == "unresolved"
    assert edge.resolved is False
    assert edge.callee_file is None


def test_obj_method_reassigned_receiver_stays_unresolved(
    builder: CallGraphBuilder,
) -> None:
    """A receiver reassigned to an untyped value is left unresolved (sound)."""
    code = (
        "class Widget:\n"
        "    def render(self):\n"
        "        return 1\n"
        "\n"
        "def draw():\n"
        "    w = Widget()\n"
        "    w = fetch()\n"
        "    return w.render()\n"
    )
    edges = builder.build_from_sources({"m.py": code}, include_unresolved=True)
    edge = _edge(edges, "draw", "render")
    assert edge.resolution == "unresolved"


def test_obj_method_ambiguous_name_resolves_by_type(
    builder: CallGraphBuilder,
) -> None:
    """With two classes sharing a method name, the receiver's type disambiguates."""
    code = (
        "class A:\n"
        "    def render(self):\n"
        "        return 1\n"
        "\n"
        "class B:\n"
        "    def render(self):\n"
        "        return 2\n"
        "\n"
        "def draw():\n"
        "    b = B()\n"
        "    return b.render()\n"
    )
    edges = builder.build_from_sources({"m.py": code})
    edge = _edge(edges, "draw", "B.render")
    assert edge.resolution == "inferred"
    assert not any(e.callee == "A.render" for e in edges)


def test_inherited_self_method_resolves_via_base(
    builder: CallGraphBuilder,
) -> None:
    """``self.method()`` resolves to a method defined on a base class."""
    code = (
        "class Base:\n"
        "    def shared(self):\n"
        "        return 1\n"
        "\n"
        "class Derived(Base):\n"
        "    def run(self):\n"
        "        return self.shared()\n"
    )
    edges = builder.build_from_sources({"m.py": code})
    edge = _edge(edges, "Derived.run", "Base.shared")
    assert edge.call_type == "method"
    assert edge.resolution == "local"


def test_inferred_method_resolves_across_files(builder: CallGraphBuilder) -> None:
    """A receiver typed via an imported class resolves the method cross-file."""
    sources = {
        "models.py": "class User:\n    def touch(self):\n        return 1\n",
        "app.py": (
            "from models import User\n"
            "\n"
            "def handle(u: User):\n"
            "    return u.touch()\n"
        ),
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "handle", "User.touch")
    assert edge.resolution == "inferred"
    assert edge.call_type == "method"
    assert edge.callee_file == "models.py"


# ---------------------------------------------------------------------------
# Constructor calls
# ---------------------------------------------------------------------------


def test_constructor_call_same_file(builder: CallGraphBuilder) -> None:
    """Instantiating a class resolves to the class and is tagged constructor."""
    code = (
        "class Point:\n"
        "    def __init__(self, x):\n"
        "        self.x = x\n"
        "\n"
        "def make():\n"
        "    return Point(1)\n"
    )
    edges = builder.build_from_sources({"m.py": code})
    edge = _edge(edges, "make", "Point")
    assert edge.call_type == "constructor"
    assert edge.resolution == "local"


# ---------------------------------------------------------------------------
# Cross-file calls via imports
# ---------------------------------------------------------------------------


def test_cross_file_from_import(builder: CallGraphBuilder) -> None:
    """``from db import get_user`` then ``get_user()`` resolves across files."""
    sources = {
        "db.py": "def get_user(uid):\n    return uid\n",
        "app.py": (
            "from db import get_user\n"
            "\n"
            "def handle():\n"
            "    return get_user(1)\n"
        ),
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "handle", "get_user")
    assert edge.resolution == "imported"
    assert edge.call_type == "function"
    assert edge.caller_file == "app.py"
    assert edge.callee_file == "db.py"


def test_cross_file_aliased_import(builder: CallGraphBuilder) -> None:
    """An aliased ``from db import get_user as gu`` resolves through the alias."""
    sources = {
        "db.py": "def get_user(uid):\n    return uid\n",
        "app.py": (
            "from db import get_user as gu\n"
            "\n"
            "def handle():\n"
            "    return gu(1)\n"
        ),
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "handle", "get_user")
    assert edge.resolution == "imported"
    assert edge.callee_file == "db.py"


def test_cross_file_module_attribute_call(builder: CallGraphBuilder) -> None:
    """``import db`` then ``db.get_user()`` resolves through the module import."""
    sources = {
        "db.py": "def get_user(uid):\n    return uid\n",
        "app.py": ("import db\n" "\n" "def handle():\n" "    return db.get_user(1)\n"),
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "handle", "get_user")
    assert edge.resolution == "imported"
    assert edge.call_type == "function"
    assert edge.callee_file == "db.py"


def test_cross_file_constructor(builder: CallGraphBuilder) -> None:
    """An imported class instantiation is a cross-file constructor edge."""
    sources = {
        "models.py": "class User:\n    def __init__(self):\n        pass\n",
        "app.py": (
            "from models import User\n" "\n" "def make():\n" "    return User()\n"
        ),
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "make", "User")
    assert edge.call_type == "constructor"
    assert edge.resolution == "imported"
    assert edge.callee_file == "models.py"


def test_relative_import_resolution(builder: CallGraphBuilder) -> None:
    """A relative ``from .db import get_user`` resolves to the sibling module."""
    sources = {
        "pkg/db.py": "def get_user(uid):\n    return uid\n",
        "pkg/app.py": (
            "from .db import get_user\n"
            "\n"
            "def handle():\n"
            "    return get_user(1)\n"
        ),
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "handle", "get_user")
    assert edge.resolution == "imported"
    assert edge.callee_file == "pkg/db.py"


def test_wildcard_import_resolution(builder: CallGraphBuilder) -> None:
    """A name reached via ``from db import *`` resolves to the source module."""
    sources = {
        "db.py": "def get_user(uid):\n    return uid\n",
        "app.py": (
            "from db import *\n" "\n" "def handle():\n" "    return get_user(1)\n"
        ),
    }
    edges = builder.build_from_sources(sources)
    edge = _edge(edges, "handle", "get_user")
    assert edge.resolution == "imported"
    assert edge.callee_file == "db.py"


# ---------------------------------------------------------------------------
# Module-scope callers and unresolved calls
# ---------------------------------------------------------------------------


def test_module_scope_caller(builder: CallGraphBuilder) -> None:
    """A call at module level is attributed to the MODULE_SCOPE sentinel."""
    code = "def setup():\n    return 1\n\nCONFIG = setup()\n"
    edges = builder.build_from_sources({"m.py": code})
    edge = _edge(edges, MODULE_SCOPE, "setup")
    assert edge.caller == "<module>"
    assert edge.resolution == "local"
    assert edge.call_site_line == 4


def test_external_call_dropped_by_default(builder: CallGraphBuilder) -> None:
    """Builtins / third-party calls are dropped unless explicitly requested."""
    code = (
        "import hashlib\n"
        "\n"
        "def digest(x):\n"
        "    print(x)\n"
        "    return hashlib.sha256(x).hexdigest()\n"
    )
    default_edges = builder.build_from_sources({"m.py": code})
    assert default_edges == []

    all_edges = builder.build_from_sources({"m.py": code}, include_unresolved=True)
    callees = {e.callee for e in all_edges}
    assert {"print", "sha256", "hexdigest"} <= callees
    assert all(
        e.resolution == "unresolved" and e.callee_file is None for e in all_edges
    )


# ---------------------------------------------------------------------------
# build_from_files / build_from_symbols entry points
# ---------------------------------------------------------------------------


def test_build_from_files_sample_repo(builder: CallGraphBuilder) -> None:
    """End-to-end build over the sample repo yields the known call chains."""
    edges = builder.build_from_files(
        [
            "examples/sample_repo/app.py",
            "examples/sample_repo/auth.py",
            "examples/sample_repo/db.py",
        ]
    )
    pairs = {(e.caller, e.callee) for e in edges}
    assert ("handle_login", "authenticate_user") in pairs
    assert ("handle_login", "create_token") in pairs
    assert ("handle_login", "save_session") in pairs
    assert ("handle_profile", "get_user_by_email") in pairs
    assert ("authenticate_user", "get_user_by_email") in pairs
    assert ("authenticate_user", "hash_password") in pairs

    # authenticate_user -> get_user_by_email is a cross-file edge.
    cross = _edge(edges, "authenticate_user", "get_user_by_email")
    assert cross.caller_file.endswith("auth.py")
    assert cross.callee_file.endswith("db.py")
    assert cross.resolution == "imported"


def test_build_from_symbols_matches_issue_signature(
    builder: CallGraphBuilder,
) -> None:
    """The lower-level build_from_symbols(symbols, file_asts) API works."""
    from src.reporag.ingestion.parser import ASTParser
    from src.reporag.ingestion.symbol_extractor import SymbolExtractor

    parser = ASTParser()
    extractor = SymbolExtractor(parser)

    sources = {
        "db.py": "def get_user(uid):\n    return uid\n",
        "app.py": "from db import get_user\n\ndef handle():\n    return get_user(1)\n",
    }
    symbols = []
    file_asts = {}
    for path, src in sources.items():
        tree = parser.parse(src, language="python")
        file_asts[path] = tree
        symbols.extend(extractor.extract_from_tree(tree, path, src, language="python"))

    edges = builder.build_from_symbols(symbols, file_asts)
    edge = _edge(edges, "handle", "get_user")
    assert edge.resolution == "imported"
    assert edge.callee_file == "db.py"


# ---------------------------------------------------------------------------
# Robustness and the CallEdge contract
# ---------------------------------------------------------------------------


def test_empty_and_no_calls(builder: CallGraphBuilder) -> None:
    """Empty sources and call-free sources produce no edges."""
    assert builder.build_from_sources({"a.py": ""}) == []
    assert builder.build_from_sources({"a.py": "x = 1\ny = x + 2\n"}) == []


def test_syntax_error_is_tolerated(builder: CallGraphBuilder) -> None:
    """A file with a syntax error still yields resolvable edges from valid parts."""
    code = (
        "def helper():\n"
        "    return 1\n"
        "\n"
        "def main():\n"
        "    helper()\n"
        "    return broken(\n"
    )
    edges = builder.build_from_sources({"m.py": code})
    assert any(e.caller == "main" and e.callee == "helper" for e in edges)


def test_edges_are_deterministically_ordered(builder: CallGraphBuilder) -> None:
    """Edges are sorted by (caller_file, call_site_line, callee)."""
    sources = {
        "b.py": "def b():\n    return 1\n",
        "a.py": (
            "from b import b\n"
            "\n"
            "def one():\n"
            "    return b()\n"
            "\n"
            "def two():\n"
            "    return b()\n"
        ),
    }
    edges = builder.build_from_sources(sources)
    keys = [(e.caller_file, e.call_site_line, e.callee) for e in edges]
    assert keys == sorted(keys)


def test_call_edge_to_dict_is_json_primitive() -> None:
    """CallEdge.to_dict emits only JSON-primitive values and derives resolved."""
    edge = CallEdge(
        caller="a",
        callee="b",
        caller_file="m.py",
        call_site_line=3,
        callee_file="m.py",
        call_type="function",
        resolution="local",
    )
    assert edge.resolved is True
    payload = edge.to_dict()
    assert payload["caller"] == "a"
    assert payload["callee"] == "b"
    assert payload["resolved"] is True
    assert payload["resolution"] == "local"
    for value in payload.values():
        assert isinstance(value, str | int | bool | None)


def test_unresolved_edge_marks_resolved_false() -> None:
    """An unresolved CallEdge reports resolved=False via __post_init__."""
    edge = CallEdge(
        caller="a",
        callee="mystery",
        caller_file="m.py",
        call_site_line=1,
        resolution="unresolved",
    )
    assert edge.resolved is False
    assert edge.callee_file is None
