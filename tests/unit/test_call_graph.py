"""Unit tests for the call graph builder (Issue 9)."""

from __future__ import annotations

import pytest

from reporag.graph.call_graph import CallGraphBuilder


@pytest.fixture
def builder() -> CallGraphBuilder:
    """Fixture to provide a CallGraphBuilder instance."""
    return CallGraphBuilder()


def test_direct_function_call(builder: CallGraphBuilder) -> None:
    """Identifies a direct function call at the module level."""
    sources = {
        "app.py": ("def foo():\n" "    pass\n" "\n" "def bar():\n" "    foo()\n")
    }
    edges = builder.build_from_sources(sources)
    assert len(edges) == 1
    edge = edges[0]
    assert edge.caller == "bar"
    assert edge.callee == "foo"
    assert edge.file_path == "app.py"
    assert edge.line == 5


def test_recursive_function_call(builder: CallGraphBuilder) -> None:
    """Identifies a recursive function calling itself."""
    sources = {
        "math_utils.py": (
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        )
    }
    edges = builder.build_from_sources(sources)
    assert len(edges) == 1
    edge = edges[0]
    assert edge.caller == "factorial"
    assert edge.callee == "factorial"
    assert edge.file_path == "math_utils.py"
    assert edge.line == 4


def test_method_call_on_self(builder: CallGraphBuilder) -> None:
    """Resolves method calls on self within the same class definition."""
    sources = {
        "service.py": (
            "class MyService:\n"
            "    def start(self):\n"
            "        self.setup()\n"
            "\n"
            "    def setup(self):\n"
            "        pass\n"
        )
    }
    edges = builder.build_from_sources(sources)
    assert len(edges) == 1
    edge = edges[0]
    assert edge.caller == "MyService.start"
    assert edge.callee == "MyService.setup"
    assert edge.file_path == "service.py"
    assert edge.line == 3


def test_constructor_call(builder: CallGraphBuilder) -> None:
    """Resolves class constructor instantiation to the class qualified name."""
    sources = {
        "main.py": (
            "class Config:\n" "    pass\n" "\n" "def run():\n" "    c = Config()\n"
        )
    }
    edges = builder.build_from_sources(sources)
    assert len(edges) == 1
    edge = edges[0]
    assert edge.caller == "run"
    assert edge.callee == "Config"
    assert edge.file_path == "main.py"
    assert edge.line == 5


def test_cross_file_function_call(builder: CallGraphBuilder) -> None:
    """Resolves target definitions imported from other modules."""
    sources = {
        "db.py": ("def connect():\n" "    pass\n"),
        "app.py": ("from db import connect\n" "def start_app():\n" "    connect()\n"),
    }
    edges = builder.build_from_sources(sources)
    # Filter for calls inside app.py
    app_edges = [e for e in edges if e.file_path == "app.py"]
    assert len(app_edges) == 1
    edge = app_edges[0]
    assert edge.caller == "start_app"
    assert edge.callee == "connect"
    assert edge.file_path == "app.py"
    assert edge.line == 3


def test_cross_file_method_call(builder: CallGraphBuilder) -> None:
    """Resolves method calls on imported classes/modules using global lookup."""
    sources = {
        "engine.py": ("class Motor:\n" "    def run(self):\n" "        pass\n"),
        "car.py": (
            "from engine import Motor\n"
            "def drive():\n"
            "    m = Motor()\n"
            "    m.run()\n"
        ),
    }
    edges = builder.build_from_sources(sources)
    car_edges = [e for e in edges if e.file_path == "car.py"]
    assert len(car_edges) == 2

    # m = Motor()
    constructor = next(e for e in car_edges if e.line == 3)
    assert constructor.caller == "drive"
    assert constructor.callee == "Motor"

    # m.run()
    method_call = next(e for e in car_edges if e.line == 4)
    assert method_call.caller == "drive"
    assert method_call.callee == "Motor.run"


def test_method_inheritance_resolution(builder: CallGraphBuilder) -> None:
    """Resolves method calls to base classes defined in same or other files."""
    sources = {
        "base.py": ("class BaseHandler:\n" "    def handle(self):\n" "        pass\n"),
        "handler.py": (
            "from base import BaseHandler\n"
            "class SpecialHandler(BaseHandler):\n"
            "    def process(self):\n"
            "        self.handle()\n"
        ),
    }
    edges = builder.build_from_sources(sources)
    handler_edges = [e for e in edges if e.file_path == "handler.py"]
    assert len(handler_edges) == 1
    edge = handler_edges[0]
    assert edge.caller == "SpecialHandler.process"
    assert edge.callee == "BaseHandler.handle"
    assert edge.line == 4


def test_nested_classes_and_functions(builder: CallGraphBuilder) -> None:
    """Resolves calls inside nested functions and nested classes correctly."""
    sources = {
        "nested.py": (
            "def outer():\n" "    def inner():\n" "        pass\n" "    inner()\n"
        )
    }
    edges = builder.build_from_sources(sources)
    assert len(edges) == 1
    edge = edges[0]
    assert edge.caller == "outer"
    assert edge.callee == "outer.<locals>.inner"
    assert edge.line == 4


def test_chained_method_calls(builder: CallGraphBuilder) -> None:
    """Resolves final method name for chained call expressions."""
    sources = {
        "chained.py": (
            "class Builder:\n"
            "    def build(self):\n"
            "        pass\n"
            "\n"
            "def get_builder():\n"
            "    return Builder()\n"
            "\n"
            "def run():\n"
            "    get_builder().build()\n"
        )
    }
    edges = builder.build_from_sources(sources)
    run_edges = [e for e in edges if e.caller == "run"]
    assert len(run_edges) == 2  # get_builder() call and build() call

    build_call = next(e for e in run_edges if e.callee == "Builder.build")
    assert build_call.line == 9
