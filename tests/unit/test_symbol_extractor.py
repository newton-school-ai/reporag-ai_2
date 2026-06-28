"""Unit tests for the symbol extractor (Issue 7).

Covers every symbol type the extractor produces -- functions, classes, methods,
and imports -- plus the trickier Python shapes (async, nested functions, nested
classes, property/static/class methods, decorated definitions, aliased and
wildcard imports), error tolerance, and file-based extraction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.reporag.ingestion.parser import ParseError, UnsupportedLanguageError
from src.reporag.ingestion.symbol_extractor import (
    Symbol,
    SymbolExtractor,
    extract_symbols,
)


@pytest.fixture
def extractor() -> SymbolExtractor:
    """Return a fresh extractor for each test."""
    return SymbolExtractor()


def _by_name(symbols: list[Symbol], name: str) -> Symbol:
    """Return the single symbol with the given name (fails if absent/ambiguous)."""
    matches = [s for s in symbols if s.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r}, got {len(matches)}"
    return matches[0]


# ----- Functions -----


def test_extract_simple_function(extractor: SymbolExtractor) -> None:
    """A top-level function yields one function symbol with full metadata."""
    source = 'def greet(name):\n    """Say hi."""\n    return f"hi {name}"\n'
    symbols = extractor.extract(source)

    assert len(symbols) == 1
    func = symbols[0]
    assert func.name == "greet"
    assert func.type == "function"
    assert func.is_method is False
    assert func.signature == "def greet(name):"
    assert func.docstring == "Say hi."
    assert func.decorators == ()
    assert func.parent_class is None
    assert func.start_line == 1
    assert func.end_line == 3


def test_function_return_type_hint(extractor: SymbolExtractor) -> None:
    """A return annotation is captured and reflected in the signature."""
    func = extractor.extract("def total(a, b) -> int:\n    return a + b\n")[0]
    assert func.return_type == "int"
    assert func.signature == "def total(a, b) -> int:"


def test_async_function(extractor: SymbolExtractor) -> None:
    """Async functions are flagged and keep the async keyword in the signature."""
    func = extractor.extract("async def fetch(url):\n    return url\n")[0]
    assert func.is_async is True
    assert func.signature == "async def fetch(url):"


def test_decorated_function_line_range_covers_decorators(
    extractor: SymbolExtractor,
) -> None:
    """Decorators are collected and the line range starts at the first one."""
    source = '@app.route("/x")\n@cached\ndef view():\n    return 1\n'
    func = extractor.extract(source)[0]

    assert func.decorators == ('app.route("/x")', "cached")
    assert func.start_line == 1  # the @app.route line, not the def line
    assert func.end_line == 4


def test_no_docstring_is_none(extractor: SymbolExtractor) -> None:
    """A function whose first statement is not a string has no docstring."""
    func = extractor.extract("def f():\n    return 1\n")[0]
    assert func.docstring is None


def test_multiline_docstring_is_cleaned(extractor: SymbolExtractor) -> None:
    """Multi-line docstrings are dedented and stripped."""
    source = (
        "def f():\n"
        '    """First line.\n'
        "\n"
        "    Second line.\n"
        '    """\n'
        "    return 1\n"
    )
    func = extractor.extract(source)[0]
    assert func.docstring == "First line.\n\nSecond line."


# ----- Nested functions -----


def test_nested_function_is_not_a_method(extractor: SymbolExtractor) -> None:
    """A function defined inside another function is a function, not a method."""
    source = "def outer():\n    def inner():\n        return 1\n    return inner\n"
    symbols = extractor.extract(source)

    names = {s.name: s for s in symbols}
    assert set(names) == {"outer", "inner"}
    assert names["inner"].type == "function"
    assert names["inner"].parent_class is None


# ----- Classes and methods -----


def test_extract_class_with_methods(extractor: SymbolExtractor) -> None:
    """A class exposes name, bases, methods, docstring, and a line range."""
    source = (
        "class Greeter(Base):\n"
        '    """A greeter."""\n'
        "\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "\n"
        "    def greet(self):\n"
        "        return self.name\n"
    )
    symbols = extractor.extract(source)

    cls = _by_name(symbols, "Greeter")
    assert cls.type == "class"
    assert cls.bases == ("Base",)
    assert cls.methods == ("__init__", "greet")
    assert cls.docstring == "A greeter."
    assert cls.signature == "class Greeter(Base):"
    assert cls.start_line == 1

    init = _by_name(symbols, "__init__")
    assert init.type == "method"
    assert init.is_method is True
    assert init.parent_class == "Greeter"


def test_class_without_bases(extractor: SymbolExtractor) -> None:
    """A base-less class has empty bases and a bare signature."""
    cls = extractor.extract("class Plain:\n    pass\n")[0]
    assert cls.bases == ()
    assert cls.signature == "class Plain:"


def test_class_keyword_arguments_excluded_from_bases(
    extractor: SymbolExtractor,
) -> None:
    """metaclass=... and similar keyword args are not treated as base classes."""
    cls = extractor.extract("class C(Base, metaclass=Meta):\n    pass\n")[0]
    assert cls.bases == ("Base",)


def test_property_static_class_method_decorators(extractor: SymbolExtractor) -> None:
    """property, staticmethod, and classmethod decorators are captured."""
    source = (
        "class C:\n"
        "    @property\n"
        "    def value(self):\n"
        "        return self._v\n"
        "\n"
        "    @staticmethod\n"
        "    def helper():\n"
        "        return 1\n"
        "\n"
        "    @classmethod\n"
        "    def make(cls):\n"
        "        return cls()\n"
    )
    symbols = extractor.extract(source)

    assert _by_name(symbols, "value").decorators == ("property",)
    assert _by_name(symbols, "helper").decorators == ("staticmethod",)
    assert _by_name(symbols, "make").decorators == ("classmethod",)
    assert all(s.parent_class == "C" for s in symbols if s.type == "method")


def test_async_method(extractor: SymbolExtractor) -> None:
    """An async method is both a method and flagged async."""
    source = "class C:\n    async def fetch(self):\n        return 1\n"
    method = _by_name(extractor.extract(source), "fetch")
    assert method.type == "method"
    assert method.is_async is True


def test_nested_classes(extractor: SymbolExtractor) -> None:
    """Nested classes are discovered with the outer class as parent."""
    source = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def method(self):\n"
        "            return 1\n"
    )
    symbols = extractor.extract(source)

    outer = _by_name(symbols, "Outer")
    inner = _by_name(symbols, "Inner")
    method = _by_name(symbols, "method")
    assert outer.parent_class is None
    assert inner.parent_class == "Outer"
    assert method.parent_class == "Inner"
    # The outer class lists only its own direct members.
    assert outer.methods == ()


# ----- Imports -----


def test_plain_import(extractor: SymbolExtractor) -> None:
    """``import os`` binds the module name to itself."""
    sym = extractor.extract("import os\n")[0]
    assert sym.type == "import"
    assert sym.name == "os"
    assert sym.module == "os"


def test_aliased_import(extractor: SymbolExtractor) -> None:
    """``import numpy as np`` binds the alias and records the source module."""
    sym = extractor.extract("import numpy as np\n")[0]
    assert sym.name == "np"
    assert sym.module == "numpy"


def test_from_import_multiple_names(extractor: SymbolExtractor) -> None:
    """``from m import a, b`` yields one symbol per imported name."""
    symbols = extractor.extract("from collections import OrderedDict, defaultdict\n")
    assert [s.name for s in symbols] == ["OrderedDict", "defaultdict"]
    assert all(s.type == "import" and s.module == "collections" for s in symbols)


def test_from_import_aliased(extractor: SymbolExtractor) -> None:
    """``from m import x as y`` binds the alias, not the original name."""
    sym = extractor.extract("from os import path as p\n")[0]
    assert sym.name == "p"
    assert sym.module == "os"


def test_wildcard_import(extractor: SymbolExtractor) -> None:
    """``from m import *`` is represented with a ``*`` name."""
    sym = extractor.extract("from typing import *\n")[0]
    assert sym.name == "*"
    assert sym.module == "typing"


def test_relative_import(extractor: SymbolExtractor) -> None:
    """A relative import records the dotted prefix as its module."""
    sym = extractor.extract("from . import sibling\n")[0]
    assert sym.name == "sibling"
    assert sym.module == "."


# ----- Ordering, empties, and error tolerance -----


def test_empty_source_yields_no_symbols(extractor: SymbolExtractor) -> None:
    """Empty input produces an empty list, not an error."""
    assert extractor.extract("") == []


def test_module_level_code_is_ignored(extractor: SymbolExtractor) -> None:
    """Assignments and bare expressions are not extracted as symbols."""
    source = "x = 1\nprint(x)\n\ndef f():\n    return x\n"
    symbols = extractor.extract(source)
    assert [s.name for s in symbols] == ["f"]


def test_symbols_returned_in_source_order(extractor: SymbolExtractor) -> None:
    """Symbols come back in the order they appear in the source."""
    source = "import os\n\ndef a():\n    pass\n\nclass B:\n    pass\n"
    names = [s.name for s in extractor.extract(source)]
    assert names == ["os", "a", "B"]


def test_partial_extraction_on_syntax_error(extractor: SymbolExtractor) -> None:
    """A broken tail does not stop extraction of valid earlier definitions."""
    source = "def ok():\n    return 1\n\ndef broken(:\n    x =\n"
    symbols = extractor.extract(source)
    assert any(s.name == "ok" and s.type == "function" for s in symbols)


# ----- File-based extraction -----


def test_extract_from_file_infers_language(
    extractor: SymbolExtractor, tmp_path: Path
) -> None:
    """extract_from_file infers Python from the .py extension and sets file_path."""
    file_path = tmp_path / "module.py"
    file_path.write_text("def loaded():\n    return True\n")

    symbols = extractor.extract_from_file(file_path)
    assert len(symbols) == 1
    assert symbols[0].name == "loaded"
    assert symbols[0].file_path == str(file_path)


def test_extract_from_file_unknown_extension_raises(
    extractor: SymbolExtractor, tmp_path: Path
) -> None:
    """An unmappable extension raises UnsupportedLanguageError."""
    file_path = tmp_path / "notes.txt"
    file_path.write_text("just text")
    with pytest.raises(UnsupportedLanguageError):
        extractor.extract_from_file(file_path)


def test_extract_from_sample_repo(extractor: SymbolExtractor) -> None:
    """The bundled sample app extracts its known imports, functions, and methods."""
    symbols = extractor.extract_from_file("examples/sample_repo/app.py")

    imports = [s.name for s in symbols if s.type == "import"]
    functions = [s.name for s in symbols if s.type == "function"]
    assert imports == [
        "authenticate_user",
        "create_token",
        "get_user_by_email",
        "save_session",
    ]
    assert functions == ["handle_login", "handle_profile"]


# ----- Language guarding -----


def test_non_python_language_raises(extractor: SymbolExtractor) -> None:
    """Only Python is supported today; other languages raise clearly."""
    with pytest.raises(UnsupportedLanguageError):
        extractor.extract("function f() {}\n", language="javascript")

    # UnsupportedLanguageError is a ParseError subclass (shared with the parser).
    assert issubclass(UnsupportedLanguageError, ParseError)


def test_language_name_is_case_insensitive(extractor: SymbolExtractor) -> None:
    """Language names are normalized so casing does not matter."""
    symbols = extractor.extract("def f():\n    pass\n", language="PYTHON")
    assert symbols[0].name == "f"


# ----- Convenience wrapper -----


def test_extract_symbols_convenience_wrapper() -> None:
    """The module-level helper mirrors SymbolExtractor.extract."""
    symbols = extract_symbols("def f():\n    pass\n", file_path="x.py")
    assert symbols[0].name == "f"
    assert symbols[0].file_path == "x.py"
