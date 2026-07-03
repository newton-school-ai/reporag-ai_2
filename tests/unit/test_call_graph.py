"""Unit tests for call_graph module (Issue 9).

Coverage
--------
* Direct function calls
* Method calls (``self.method()``, ``obj.method()``)
* Constructor calls (``MyClass()``) -- resolved from local class symbol, not name casing
* Chained calls (``get_service().run()``)
* Cross-file calls (resolved via import map)
* Recursive / self-referential calls
* Module-level calls (caller = ``<module:...>``)
* CallEdge metadata: caller, callee, call_site_line, caller_file, callee_file
* Unresolved / third-party callees (callee_file is None)
* Dual local index: qualified-name index + simple-name index (no silent overwrites)
* Same-class self./cls. resolution using parent-class metadata
"""

from __future__ import annotations

import pytest

from src.reporag.graph.call_graph import CallEdge, CallGraphBuilder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def builder() -> CallGraphBuilder:
    """Shared CallGraphBuilder instance."""
    return CallGraphBuilder()


def edges_for(
    source: str, builder: CallGraphBuilder, file_path: str = "test.py"
) -> list[CallEdge]:
    """Helper: extract edges from a source string."""
    return builder.extract_from_source(source, file_path=file_path)


def find_edge(edges: list[CallEdge], *, callee_contains: str) -> CallEdge | None:
    """Find the first edge whose callee contains *callee_contains*."""
    for e in edges:
        if callee_contains in e.callee:
            return e
    return None


def find_edges(
    edges: list[CallEdge], *, caller_contains: str = "", callee_contains: str = ""
) -> list[CallEdge]:
    """Filter edges by partial caller/callee match."""
    result = []
    for e in edges:
        if caller_contains and caller_contains not in e.caller:
            continue
        if callee_contains and callee_contains not in e.callee:
            continue
        result.append(e)
    return result


# ---------------------------------------------------------------------------
# 1. Direct function calls
# ---------------------------------------------------------------------------


class TestDirectCalls:
    def test_single_direct_call(self, builder: CallGraphBuilder) -> None:
        """A function that calls another function emits one direct edge."""
        source = """\
def helper():
    pass

def main():
    helper()
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="helper")
        assert e is not None, "Expected an edge for helper()"
        assert e.call_type == "direct"
        assert e.caller == "main"
        assert e.call_site_line == 5

    def test_multiple_direct_calls(self, builder: CallGraphBuilder) -> None:
        """Multiple calls from the same function produce multiple edges."""
        source = """\
def a(): pass
def b(): pass
def c(): pass

def orchestrate():
    a()
    b()
    c()
"""
        edges = edges_for(source, builder)
        callees = {e.callee for e in edges if e.caller == "orchestrate"}
        assert "a" in callees
        assert "b" in callees
        assert "c" in callees

    def test_direct_call_metadata_caller_file(self, builder: CallGraphBuilder) -> None:
        """CallEdge carries the correct caller_file."""
        source = "def foo(): pass\ndef bar(): foo()\n"
        edges = edges_for(source, builder, file_path="/repo/mymodule.py")
        e = find_edge(edges, callee_contains="foo")
        assert e is not None
        assert e.caller_file == "/repo/mymodule.py"

    def test_direct_call_callee_same_file(self, builder: CallGraphBuilder) -> None:
        """When the callee is defined in the same file, callee_file matches caller_file."""
        source = "def compute(): pass\ndef run(): compute()\n"
        edges = edges_for(source, builder, file_path="/repo/ops.py")
        e = find_edge(edges, callee_contains="compute")
        assert e is not None
        assert e.callee_file == "/repo/ops.py"

    def test_call_line_number_is_correct(self, builder: CallGraphBuilder) -> None:
        """call_site_line is the 1-based line of the call expression."""
        source = """\
def helper(): pass

def main():
    x = 1
    y = 2
    helper()
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="helper")
        assert e is not None
        assert e.call_site_line == 6

    def test_nested_function_direct_call(self, builder: CallGraphBuilder) -> None:
        """Calls inside nested functions are attributed to the inner scope."""
        source = """\
def outer():
    def inner():
        helper()
    inner()
"""
        edges = edges_for(source, builder)
        inner_calls = [e for e in edges if "inner" in e.caller and "helper" in e.callee]
        assert len(inner_calls) >= 1

    def test_unresolved_builtin_callee_file_is_none(
        self, builder: CallGraphBuilder
    ) -> None:
        """Calls to built-ins like len() have callee_file=None."""
        source = "def process(items):\n    return len(items)\n"
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="len")
        assert e is not None
        assert e.callee_file is None


# ---------------------------------------------------------------------------
# 2. Method calls
# ---------------------------------------------------------------------------


class TestMethodCalls:
    def test_self_method_call(self, builder: CallGraphBuilder) -> None:
        """self.method() is detected as a method call."""
        source = """\
class MyService:
    def _helper(self):
        pass

    def run(self):
        self._helper()
"""
        edges = edges_for(source, builder)
        method_edges = [e for e in edges if "method" in e.call_type]
        assert len(method_edges) >= 1
        callee_names = {e.callee for e in method_edges}
        assert any("_helper" in c for c in callee_names)

    def test_obj_method_call(self, builder: CallGraphBuilder) -> None:
        """obj.method() is captured as a method call with proper callee text."""
        source = """\
def process(db):
    result = db.query()
    return result
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="query")
        assert e is not None
        assert e.call_type == "method"

    def test_method_call_type_flag(self, builder: CallGraphBuilder) -> None:
        """call_type is 'method' for attribute access calls."""
        source = """\
def run(obj):
    obj.execute()
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="execute")
        assert e is not None
        assert e.call_type == "method"

    def test_cls_method_call(self, builder: CallGraphBuilder) -> None:
        """cls.method() in classmethods is treated as a method call."""
        source = """\
class Factory:
    @classmethod
    def create(cls):
        return cls._build()

    @classmethod
    def _build(cls):
        pass
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="_build")
        assert e is not None
        assert e.call_type == "method"

    def test_method_call_line_metadata(self, builder: CallGraphBuilder) -> None:
        """Method call has correct call_site_line."""
        source = """\
class A:
    def go(self):
        self.helper()

    def helper(self):
        pass
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="helper")
        assert e is not None
        assert e.call_site_line == 3


# ---------------------------------------------------------------------------
# 3. Constructor calls
# ---------------------------------------------------------------------------


class TestConstructorCalls:
    def test_uppercase_call_without_local_class_is_direct(
        self, builder: CallGraphBuilder
    ) -> None:
        """Calling an uppercase name NOT defined locally stays 'direct'.

        Constructor detection is based on the *resolved symbol type*, not on
        name capitalisation.  When ``MyClass`` has no local definition the
        resolver falls back to ``_resolve_unknown`` which preserves the
        preliminary ``'direct'`` call type.
        """
        source = """\
def build():
    obj = MyClass()
    return obj
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="MyClass")
        assert e is not None
        # No local class definition -> cannot confirm it is a constructor
        assert e.call_type == "direct"

    def test_local_class_call_is_constructor(self, builder: CallGraphBuilder) -> None:
        """Calling a name whose local symbol has type 'class' is classified as constructor."""
        source = """\
class MyClass:
    pass

def build():
    obj = MyClass()
    return obj
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="MyClass")
        assert e is not None
        assert e.call_type == "constructor"

    def test_constructor_of_local_class(self, builder: CallGraphBuilder) -> None:
        """Constructing a class defined in the same file resolves callee_file."""
        source = """\
class Config:
    pass

def setup():
    cfg = Config()
    return cfg
"""
        edges = edges_for(source, builder, file_path="/repo/config.py")
        e = find_edge(edges, callee_contains="Config")
        assert e is not None
        assert e.call_type == "constructor"
        # Config is defined in the same file
        assert e.callee_file == "/repo/config.py"

    def test_constructor_call_metadata(self, builder: CallGraphBuilder) -> None:
        """Constructor edge includes correct caller and line number."""
        source = """\
class Widget:
    pass

def make_widget():
    w = Widget()
    return w
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="Widget")
        assert e is not None
        assert e.caller == "make_widget"
        assert e.call_site_line == 5

    def test_lowercase_class_call_is_constructor(
        self, builder: CallGraphBuilder
    ) -> None:
        """A class starting with a lowercase letter is still a constructor when locally defined."""
        source = """\
class myConfig:
    pass

def setup():
    cfg = myConfig()
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="myConfig")
        assert e is not None
        assert e.call_type == "constructor"


# ---------------------------------------------------------------------------
# 4. Chained calls
# ---------------------------------------------------------------------------


class TestChainedCalls:
    def test_simple_chained_call(self, builder: CallGraphBuilder) -> None:
        """get_service().run() is detected (at least the outer or inner call)."""
        source = """\
def start():
    get_service().run()
"""
        edges = edges_for(source, builder)
        # At minimum, the inner call get_service() should be captured
        assert any("get_service" in e.callee or "run" in e.callee for e in edges)

    def test_chained_call_type(self, builder: CallGraphBuilder) -> None:
        """Chained call is classified as 'chained' or 'method'."""
        source = """\
def process():
    builder().build().execute()
"""
        edges = edges_for(source, builder)
        types = {e.call_type for e in edges}
        assert types & {"chained", "method", "direct"}

    def test_method_chain_on_return_value(self, builder: CallGraphBuilder) -> None:
        """Fluent / builder pattern: each step in the chain produces an edge."""
        source = """\
def run():
    result = get_query().filter(active=True).all()
"""
        edges = edges_for(source, builder)
        callee_names = {e.callee for e in edges}
        assert any("get_query" in c for c in callee_names)


# ---------------------------------------------------------------------------
# 5. Cross-file calls (via import map)
# ---------------------------------------------------------------------------


class TestCrossFileCalls:
    def test_imported_function_call(self, builder: CallGraphBuilder) -> None:
        """Calling an imported name resolves callee via the import map."""
        source = """\
from mypackage.utils import compute

def run():
    compute(42)
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="compute")
        assert e is not None
        assert "mypackage" in e.callee or "compute" in e.callee

    def test_imported_aliased_function(self, builder: CallGraphBuilder) -> None:
        """Aliased imports resolve to the original qualified name."""
        source = """\
from mypackage.math import add_numbers as add

def calculate():
    add(1, 2)
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="add")
        assert e is not None
        # The callee should reference the module path
        assert "mypackage" in e.callee or "add" in e.callee

    def test_imported_module_method_call(self, builder: CallGraphBuilder) -> None:
        """``import os; os.path.join(...)`` resolves the root to the import map."""
        source = """\
import os

def build_path(base, name):
    return os.path.join(base, name)
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="join")
        assert e is not None
        # os is in the import map, callee should include os somewhere
        assert "os" in e.callee

    def test_cross_file_callee_file_populated(
        self, builder: CallGraphBuilder, tmp_path
    ) -> None:
        """When a helper file exists on disk, callee_file is resolved."""
        # Create a helper module on disk
        helper_dir = tmp_path / "mypackage"
        helper_dir.mkdir()
        (helper_dir / "__init__.py").write_text("")
        (helper_dir / "utils.py").write_text("def compute(x): return x * 2\n")

        source = "from mypackage.utils import compute\ndef run():\n    compute(1)\n"
        edges = builder.extract_from_source(
            source,
            file_path=str(tmp_path / "main.py"),
            project_root=str(tmp_path),
        )
        e = find_edge(edges, callee_contains="compute")
        assert e is not None
        assert e.callee_file is not None
        assert "utils.py" in e.callee_file

    def test_unresolved_cross_file_callee_file_is_none(
        self, builder: CallGraphBuilder
    ) -> None:
        """If the imported module cannot be found on disk, callee_file is None."""
        source = """\
from nonexistent_lib import magic_func

def run():
    magic_func()
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="magic_func")
        assert e is not None
        assert e.callee_file is None


# ---------------------------------------------------------------------------
# 6. Recursive calls
# ---------------------------------------------------------------------------


class TestRecursiveCalls:
    def test_direct_recursion(self, builder: CallGraphBuilder) -> None:
        """A function that calls itself is marked is_recursive=True."""
        source = """\
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
        edges = edges_for(source, builder)
        recursive = [e for e in edges if e.is_recursive]
        assert len(recursive) >= 1
        assert any("factorial" in e.callee for e in recursive)

    def test_method_recursion(self, builder: CallGraphBuilder) -> None:
        """A method calling itself is also recursive."""
        source = """\
class Worker:
    def work(self):
        self.work()
"""
        edges = edges_for(source, builder)

        recursive = [e for e in edges if e.is_recursive]
        assert len(recursive) == 1
        assert recursive[0].caller == "Worker.work"
        assert recursive[0].callee == "Worker.work"

    def test_non_recursive_call_is_not_flagged(self, builder: CallGraphBuilder) -> None:
        """A non-recursive call has is_recursive=False."""
        source = """\
def helper(): pass

def main():
    helper()
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="helper")
        assert e is not None
        assert e.is_recursive is False

    def test_mutual_recursion_edges_present(self, builder: CallGraphBuilder) -> None:
        """Mutually recursive functions both appear in the edge list."""
        source = """\
def is_even(n):
    if n == 0:
        return True
    return is_odd(n - 1)

def is_odd(n):
    if n == 0:
        return False
    return is_even(n - 1)
"""
        edges = edges_for(source, builder)
        callees = {e.callee for e in edges}
        assert any("is_odd" in c for c in callees)
        assert any("is_even" in c for c in callees)


# ---------------------------------------------------------------------------
# 7. Module-level calls
# ---------------------------------------------------------------------------


class TestModuleLevelCalls:
    def test_module_level_call_attributed_to_module_scope(
        self, builder: CallGraphBuilder
    ) -> None:
        """Calls at module level are attributed to a ``<module:...>`` scope."""
        source = "import os\nos.getcwd()\n"
        edges = edges_for(source, builder, file_path="script.py")
        module_edges = [e for e in edges if "<module:" in e.caller]
        assert len(module_edges) >= 1

    def test_module_level_call_line_number(self, builder: CallGraphBuilder) -> None:
        """Module-level call has correct call_site_line."""
        source = "\n\nprint('hello')\n"
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="print")
        assert e is not None
        assert e.call_site_line == 3


# ---------------------------------------------------------------------------
# 8. Edge metadata completeness
# ---------------------------------------------------------------------------


class TestEdgeMetadata:
    def test_all_required_fields_present(self, builder: CallGraphBuilder) -> None:
        """Every CallEdge has all required metadata fields populated."""
        source = "def foo(): pass\ndef bar(): foo()\n"
        edges = edges_for(source, builder, file_path="sample.py")
        for e in edges:
            assert isinstance(e.caller, str) and e.caller
            assert isinstance(e.callee, str) and e.callee
            assert isinstance(e.caller_file, str) and e.caller_file
            # callee_file can be None for unresolved
            assert isinstance(e.call_site_line, int) and e.call_site_line >= 1
            assert e.call_type in (
                "direct",
                "method",
                "constructor",
                "chained",
                "unknown",
            )
            assert isinstance(e.is_recursive, bool)

    def test_empty_source_no_edges(self, builder: CallGraphBuilder) -> None:
        """Empty source produces no edges."""
        edges = edges_for("", builder)
        assert edges == []

    def test_source_with_no_calls(self, builder: CallGraphBuilder) -> None:
        """Source with definitions but no call sites produces no edges."""
        source = "def foo(): pass\ndef bar(): pass\n"
        edges = edges_for(source, builder)
        assert edges == []

    def test_calledge_is_dataclass(self, builder: CallGraphBuilder) -> None:
        """CallEdge is a proper dataclass with expected attributes."""
        e = CallEdge(
            caller="main",
            callee="helper",
            caller_file="a.py",
            callee_file="b.py",
            call_site_line=10,
            call_type="direct",
            is_recursive=False,
        )
        assert e.caller == "main"
        assert e.callee == "helper"
        assert e.call_site_line == 10

    def test_multiple_files_via_build_graph(
        self, builder: CallGraphBuilder, tmp_path
    ) -> None:
        """build_graph merges edges from multiple files."""
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("def alpha(): pass\ndef run_a():\n    alpha()\n")
        f2.write_text("def beta(): pass\ndef run_b():\n    beta()\n")
        edges = builder.build_graph([str(f1), str(f2)])
        caller_files = {e.caller_file for e in edges}
        assert str(f1) in caller_files
        assert str(f2) in caller_files


# ---------------------------------------------------------------------------
# 9. Complex / real-world patterns
# ---------------------------------------------------------------------------


class TestComplexPatterns:
    def test_async_function_call(self, builder: CallGraphBuilder) -> None:
        """Calls inside async functions are captured."""
        source = """\
async def fetch_data():
    pass

async def main():
    await fetch_data()
"""
        edges = edges_for(source, builder)
        # fetch_data is called (the call node wraps await)
        e = find_edge(edges, callee_contains="fetch_data")
        assert e is not None

    def test_class_method_calls_sibling_method(self, builder: CallGraphBuilder) -> None:
        """A method calling another method of the same class is recorded."""
        source = """\
class Pipeline:
    def step_one(self):
        pass

    def step_two(self):
        self.step_one()

    def run(self):
        self.step_two()
"""
        edges = edges_for(source, builder)
        step_two_callers = [e for e in edges if "step_two" in e.callee]
        step_one_callers = [e for e in edges if "step_one" in e.callee]
        assert len(step_two_callers) >= 1
        assert len(step_one_callers) >= 1

    def test_decorated_function_call(self, builder: CallGraphBuilder) -> None:
        """Decorated functions are still tracked as callees."""
        source = """\
def decorator(fn):
    return fn

@decorator
def my_func():
    pass

def entry():
    my_func()
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="my_func")
        assert e is not None

    def test_lambda_call_site_not_attributed_to_lambda(
        self, builder: CallGraphBuilder
    ) -> None:
        """Calls within a function body (even via lambda) are still captured."""
        source = """\
def transform(items):
    return list(map(lambda x: x + 1, items))
"""
        edges = edges_for(source, builder)
        # list() and map() are both calls
        callee_names = {e.callee for e in edges}
        assert any("list" in c or "map" in c for c in callee_names)

    def test_conditional_call_both_branches(self, builder: CallGraphBuilder) -> None:
        """Calls in both branches of an if/else are captured."""
        source = """\
def on_true(): pass
def on_false(): pass

def dispatch(flag):
    if flag:
        on_true()
    else:
        on_false()
"""
        edges = edges_for(source, builder)
        callees = {e.callee for e in edges}
        assert "on_true" in callees
        assert "on_false" in callees

    def test_calls_inside_list_comprehension(self, builder: CallGraphBuilder) -> None:
        """Calls embedded in list comprehensions are captured."""
        source = """\
def process(x):
    return x * 2

def run(items):
    return [process(i) for i in items]
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="process")
        assert e is not None


# ---------------------------------------------------------------------------
# 10. Dual local index (no silent overwrites)
# ---------------------------------------------------------------------------


class TestDualLocalIndex:
    def test_same_name_methods_in_different_classes_both_resolved(
        self, builder: CallGraphBuilder
    ) -> None:
        """Two classes defining 'helper' do not overwrite each other in the index.

        When class A calls ``self.helper()`` the callee should resolve to
        ``A.helper``, not to ``B.helper``.
        """
        source = """\
class A:
    def helper(self):
        return 1

    def run(self):
        self.helper()

class B:
    def helper(self):
        return 2

    def run(self):
        self.helper()
"""
        edges = edges_for(source, builder)
        # Edge from A.run should resolve to A.helper
        a_edges = [e for e in edges if e.caller == "A.run"]
        assert a_edges, "Expected at least one edge from A.run"
        a_callee = a_edges[0].callee
        assert "A" in a_callee, f"Expected A.helper, got {a_callee!r}"
        assert "B" not in a_callee, f"A.run resolved to B's helper: {a_callee!r}"

        # Edge from B.run should resolve to B.helper
        b_edges = [e for e in edges if e.caller == "B.run"]
        assert b_edges, "Expected at least one edge from B.run"
        b_callee = b_edges[0].callee
        assert "B" in b_callee, f"Expected B.helper, got {b_callee!r}"

    def test_module_function_and_class_method_same_name(
        self, builder: CallGraphBuilder
    ) -> None:
        """A module-level function and a class method may share a name without collision."""
        source = """\
def save():
    pass

class Repo:
    def save(self):
        pass

    def commit(self):
        self.save()

def backup():
    save()
"""
        edges = edges_for(source, builder)

        # backup() calls the module-level save()
        backup_edges = [e for e in edges if e.caller == "backup"]
        assert backup_edges
        assert "save" in backup_edges[0].callee

        # Repo.commit() calls Repo.save via self.save()
        commit_edges = [e for e in edges if e.caller == "Repo.commit"]
        assert commit_edges
        # Should resolve to Repo.save, not the module-level save
        assert "Repo" in commit_edges[0].callee


# ---------------------------------------------------------------------------
# 11. Same-class self./cls. resolution
# ---------------------------------------------------------------------------


class TestSameClassMethodResolution:
    def test_self_call_resolves_to_own_class_method(
        self, builder: CallGraphBuilder
    ) -> None:
        """self.method() resolves to the method of the *calling* class."""
        source = """\
class Service:
    def _do_work(self):
        pass

    def execute(self):
        self._do_work()
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="_do_work")
        assert e is not None
        assert e.callee == "Service._do_work"
        assert e.caller == "Service.execute"

    def test_cls_call_resolves_to_own_classmethod(
        self, builder: CallGraphBuilder
    ) -> None:
        """cls.method() resolves to the correct classmethod of the calling class."""
        source = """\
class Builder:
    @classmethod
    def _make(cls):
        pass

    @classmethod
    def create(cls):
        return cls._make()
"""
        edges = edges_for(source, builder)
        e = find_edge(edges, callee_contains="_make")
        assert e is not None
        assert e.callee == "Builder._make"

    def test_self_call_sibling_method_across_class_boundary(
        self, builder: CallGraphBuilder
    ) -> None:
        """self.save() in class A does not resolve to B.save even when B.save exists."""
        source = """\
    class A:
        def save(self):
            pass

        def persist(self):
            self.save()

    class B:
        def save(self):
            pass
    """
        edges = edges_for(source, builder)
        persist_edges = [e for e in edges if e.caller == "A.persist"]
        assert persist_edges
        assert persist_edges[0].callee == "A.save"

    def test_ambiguous_method_resolution_returns_unresolved(
        self, builder: CallGraphBuilder
    ):
        source = """
    class A:
        def save(self):
            pass

    class B:
        def save(self):
            pass

    def process(obj):
        obj.save()
    """

        edges = edges_for(source, builder)
        edge = find_edge(edges, callee_contains="save")

        assert edge is not None
        assert edge.callee == "obj.save"
        assert edge.callee_file is None
