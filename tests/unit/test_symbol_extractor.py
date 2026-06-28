"""Unit tests for the SymbolExtractor (Issue 7)."""

from __future__ import annotations

import pytest

from src.reporag.ingestion.parser import UnsupportedLanguageError
from src.reporag.ingestion.symbol_extractor import SymbolExtractor


@pytest.fixture
def extractor() -> SymbolExtractor:
    """Fixture to provide a SymbolExtractor instance."""
    return SymbolExtractor()


def test_extract_empty_source(extractor: SymbolExtractor) -> None:
    """Extracting symbols from empty source returns an empty list."""
    symbols = extractor.extract_from_source("", language="python")
    assert symbols == []


def test_extract_simple_function(extractor: SymbolExtractor) -> None:
    """Extracting a simple function fetches name, type, and signature."""
    code = (
        "def compute_value(x: int, y: int = 10) -> int:\n"
        '    """Compute the result."""\n'
        "    return x + y\n"
    )
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    sym = symbols[0]
    assert sym.name == "compute_value"
    assert sym.type == "function"
    assert sym.signature == "def compute_value(x: int, y: int = 10) -> int"
    assert sym.docstring == "Compute the result."
    assert sym.is_async is False
    assert sym.decorators == []


def test_extract_async_and_nested_functions(extractor: SymbolExtractor) -> None:
    """Async and nested functions are extracted with proper properties."""
    code = (
        "async def outer_func():\n"
        "    def inner_func():\n"
        "        pass\n"
        "    return 42\n"
    )
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    outer = symbols[0]

    assert outer.name == "outer_func"
    assert outer.type == "function"
    assert outer.is_async is True
    assert len(outer.children) == 1

    inner = outer.children[0]
    assert inner.name == "inner_func"
    assert inner.type == "function"
    assert inner.is_async is False
    assert inner.parent_symbol == "outer_func"
    assert inner.qualified_name == "outer_func.<locals>.inner_func"


def test_extract_class_with_methods(extractor: SymbolExtractor) -> None:
    """Classes and their methods are parsed correctly with decorators."""
    code = (
        "class Calculator:\n"
        '    """Perform math operations."""\n'
        "    @staticmethod\n"
        "    def add(a, b):\n"
        "        return a + b\n"
        "\n"
        "    @property\n"
        "    def value(self):\n"
        "        return 0\n"
    )
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    cls_sym = symbols[0]

    assert cls_sym.name == "Calculator"
    assert cls_sym.type == "class"
    assert cls_sym.docstring == "Perform math operations."
    assert len(cls_sym.methods) == 2

    m1 = cls_sym.methods[0]
    assert m1.name == "add"
    assert m1.type == "method"
    assert m1.decorators == ["staticmethod"]

    m2 = cls_sym.methods[1]
    assert m2.name == "value"
    assert m2.type == "method"
    assert m2.decorators == ["property"]


def test_extract_class_inheritance(extractor: SymbolExtractor) -> None:
    """Superclass bases are captured on the class symbol."""
    code = "class CustomDict(dict, UserDict):\n    pass\n"
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    cls_sym = symbols[0]
    assert cls_sym.name == "CustomDict"
    assert cls_sym.bases == ["dict", "UserDict"]


def test_extract_various_imports(extractor: SymbolExtractor) -> None:
    """Different styles of import statements are extracted."""
    code = (
        "import os\n"
        "import sys as s\n"
        "from math import sin, cos\n"
        "from typing import (\n"
        "    List,\n"
        "    Dict as D,\n"
        ")\n"
        "from collections import *\n"
    )
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 7

    assert symbols[0].name == "os"
    assert symbols[0].type == "import"
    assert symbols[0].import_source == "os"

    assert symbols[1].name == "s"
    assert symbols[1].type == "import"
    assert symbols[1].import_source == "sys"
    assert symbols[1].import_alias == "s"

    assert symbols[2].name == "sin"
    assert symbols[2].type == "import"
    assert symbols[2].import_source == "math"

    assert symbols[3].name == "cos"
    assert symbols[3].type == "import"
    assert symbols[3].import_source == "math"

    assert symbols[4].name == "List"
    assert symbols[4].type == "import"
    assert symbols[4].import_source == "typing"

    assert symbols[5].name == "D"
    assert symbols[5].type == "import"
    assert symbols[5].import_source == "typing.Dict"
    assert symbols[5].import_alias == "D"

    assert symbols[6].name == "*"
    assert symbols[6].type == "import"
    assert symbols[6].import_source == "collections"
    assert symbols[6].is_wildcard_import is True


def test_unsupported_language_raises(extractor: SymbolExtractor) -> None:
    """Extracting symbols from an unsupported language raises UnsupportedLanguageError."""
    with pytest.raises(UnsupportedLanguageError):
        extractor.extract_from_source("const x = 1;", language="rust")


def test_edge_case_nested_classes(extractor: SymbolExtractor) -> None:
    """Nested classes are correctly resolved with clean qualified names."""
    code = "class Outer:\n" "    class Inner:\n" "        pass\n"
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    outer = symbols[0]
    assert outer.name == "Outer"
    assert len(outer.children) == 1

    inner = outer.children[0]
    assert inner.name == "Inner"
    assert inner.type == "class"
    assert inner.parent_symbol == "Outer"
    assert inner.qualified_name == "Outer.Inner"


def test_edge_case_nested_function_in_method(extractor: SymbolExtractor) -> None:
    """A nested function inside a method has <locals> in qualified name."""
    code = (
        "class MyClass:\n"
        "    def my_method(self):\n"
        "        def internal():\n"
        "            pass\n"
    )
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    cls_sym = symbols[0]
    assert len(cls_sym.methods) == 1

    method = cls_sym.methods[0]
    assert method.name == "my_method"
    assert len(method.children) == 1

    internal = method.children[0]
    assert internal.name == "internal"
    assert internal.type == "function"
    assert internal.parent_symbol == "MyClass.my_method"
    assert internal.qualified_name == "MyClass.my_method.<locals>.internal"


def test_edge_case_multiple_and_multiline_decorators(
    extractor: SymbolExtractor,
) -> None:
    """Multiple and multiline decorators are extracted correctly."""
    code = (
        "@dec1\n" "@dec2(a=1,\n" "      b=2)\n" "def decorated_func():\n" "    pass\n"
    )
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    sym = symbols[0]
    assert len(sym.decorators) == 2
    assert sym.decorators[0] == "dec1"
    assert "dec2" in sym.decorators[1]


def test_edge_case_async_methods(extractor: SymbolExtractor) -> None:
    """Async methods inside class definitions are identified correctly."""
    code = "class API:\n" "    async def fetch(self):\n" "        pass\n"
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    cls_sym = symbols[0]
    assert len(cls_sym.methods) == 1
    method = cls_sym.methods[0]
    assert method.name == "fetch"
    assert method.type == "method"
    assert method.is_async is True


def test_edge_case_property_getter_setter(extractor: SymbolExtractor) -> None:
    """Property getter and setter methods are extracted with accurate decorators."""
    code = (
        "class Person:\n"
        "    @property\n"
        "    def name(self):\n"
        "        return self._name\n"
        "    @name.setter\n"
        "    def name(self, val):\n"
        "        self._name = val\n"
    )
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    cls_sym = symbols[0]
    assert len(cls_sym.methods) == 2
    assert cls_sym.methods[0].decorators == ["property"]
    assert cls_sym.methods[1].decorators == ["name.setter"]


def test_edge_case_relative_imports(extractor: SymbolExtractor) -> None:
    """Relative module imports are correctly parsed."""
    code = "from . import sibling\n" "from ..parent import grandparent\n"
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 2
    assert symbols[0].import_source == "."
    assert symbols[0].name == "sibling"
    assert symbols[1].import_source == "..parent"
    assert symbols[1].name == "grandparent"


def test_edge_case_empty_definitions(extractor: SymbolExtractor) -> None:
    """Empty functions and classes do not fail parsing."""
    code = "class Empty:\n" "    pass\n" "def empty_func():\n" "    pass\n"
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 2
    assert symbols[0].name == "Empty"
    assert symbols[1].name == "empty_func"


def test_edge_case_syntax_errors(extractor: SymbolExtractor) -> None:
    """Malformed syntax is tolerated gracefully, dropping invalid nodes and keeping valid ones."""
    code = "def valid_one():\n" "    pass\n" "def broken(\n"
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) >= 1
    assert any(s.name == "valid_one" for s in symbols)
    # The malformed "broken" function definition is explicitly skipped because name is incomplete.
    assert not any(s.name == "broken" for s in symbols)


def test_decorated_async_functions(extractor: SymbolExtractor) -> None:
    """Decorated async functions are correctly parsed with async=True and decorators."""
    code = "@my_decorator\n" "async def run_task():\n" "    pass\n"
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    sym = symbols[0]
    assert sym.name == "run_task"
    assert sym.decorators == ["my_decorator"]
    assert sym.is_async is True


def test_decorated_classes(extractor: SymbolExtractor) -> None:
    """Decorated classes are correctly parsed with decorators."""
    code = "@singleton\n" "class AppContext:\n" "    pass\n"
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    sym = symbols[0]
    assert sym.name == "AppContext"
    assert sym.decorators == ["singleton"]


def test_raw_and_concatenated_docstrings(extractor: SymbolExtractor) -> None:
    """Raw, unicode, bytes-prefixed, and concatenated docstring literals are extracted."""
    code = (
        "def f_raw():\n"
        '    r"""raw string doc\\n"""\n'
        "def f_concat():\n"
        '    "hello " "world"\n'
        "def f_bytes():\n"
        '    b"bytes doc"\n'
    )
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 3
    assert symbols[0].docstring == "raw string doc\\n"
    assert symbols[1].docstring == "hello world"
    assert symbols[2].docstring == "bytes doc"


def test_precise_multiline_import_lines(extractor: SymbolExtractor) -> None:
    """Individual line numbers are recorded precisely for multiline from-imports."""
    code = "from my_module import (\n" "    first_val,\n" "    second_val,\n" ")\n"
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 2
    first, second = symbols[0], symbols[1]
    assert first.name == "first_val"
    assert second.name == "second_val"

    assert first.start_line == 2
    assert first.end_line == 2
    assert second.start_line == 3
    assert second.end_line == 3


def test_malformed_function_with_parse_error(extractor: SymbolExtractor) -> None:
    """A function with syntactic errors is extracted with has_parse_error=True."""
    code = (
        "def broken_func(x,):\n"
        "    # syntax error (trailing comma with nothing)\n"
        "    x +\n"
    )
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    sym = symbols[0]
    assert sym.name == "broken_func"
    assert sym.has_parse_error is True


def test_malformed_class_with_parse_error(extractor: SymbolExtractor) -> None:
    """A class containing parse errors is extracted with has_parse_error=True."""
    code = "class BrokenClass:\n" "    def method(self):\n" "        x =\n"
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    sym = symbols[0]
    assert sym.name == "BrokenClass"
    assert sym.has_parse_error is True


def test_inheritance_bases_generic(extractor: SymbolExtractor) -> None:
    """Bases extraction correctly fetches Generic[T]."""
    code = "class MyGeneric(Generic[T]):\n    pass\n"
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    assert symbols[0].bases == ["Generic[T]"]


def test_inheritance_bases_subscript(extractor: SymbolExtractor) -> None:
    """Bases extraction correctly fetches list[int]."""
    code = "class IntList(list[int]):\n    pass\n"
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    assert symbols[0].bases == ["list[int]"]


def test_metaclass_omitted_from_bases(extractor: SymbolExtractor) -> None:
    """Bases extraction skips metaclass keyword arguments."""
    code = "class MetaClass(Base, metaclass=ABCMeta):\n    pass\n"
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    assert symbols[0].bases == ["Base"]


def test_docstring_with_leading_comments(extractor: SymbolExtractor) -> None:
    """Comments preceding a docstring are skipped, extracting docstring correctly."""
    code = (
        "def doc_with_comments():\n"
        "    # Leading comment 1\n"
        "    # Leading comment 2\n"
        '    """This is the actual docstring"""\n'
        "    pass\n"
    )
    symbols = extractor.extract_from_source(code, language="python")
    assert len(symbols) == 1
    assert symbols[0].docstring == "This is the actual docstring"
