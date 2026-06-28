"""Symbol extractor: walks a tree-sitter AST and produces structured Symbol objects.

Extracts meaningful code entities from parsed Python source files:
- Functions (sync and async)
- Classes (with their methods)
- Module-level imports (import X, from X import Y, from X import *)

Each symbol carries rich metadata: name, type, file_path, start_line, end_line,
signature, docstring, decorators, parent_class (if method), return_type_hint.

Usage::

    from src.reporag.ingestion.symbol_extractor import SymbolExtractor

    extractor = SymbolExtractor()
    symbols = extractor.extract_from_file('src/reporag/ingestion/cloner.py')
    for s in symbols:
        print(f'{s.type}: {s.name} [{s.start_line}-{s.end_line}]')

    # Or parse in-memory source:
    symbols = extractor.extract_from_source(source_code, language='python')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Node, Tree

from src.reporag.ingestion.parser import ASTParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol types
# ---------------------------------------------------------------------------

FUNCTION = "function"
ASYNC_FUNCTION = "async_function"
CLASS = "class"
METHOD = "method"
ASYNC_METHOD = "async_method"
STATIC_METHOD = "static_method"
CLASS_METHOD = "class_method"
PROPERTY = "property"
IMPORT = "import"


# ---------------------------------------------------------------------------
# Symbol dataclass
# ---------------------------------------------------------------------------


@dataclass
class Symbol:
    """A meaningful code entity extracted from a source file.

    Attributes:
        name:            Identifier name (e.g. ``"my_func"``).
        type:            One of the module-level constants: ``"function"``,
                         ``"async_function"``, ``"class"``, ``"method"``,
                         ``"static_method"``, ``"class_method"``,
                         ``"property"``, ``"import"``.
        file_path:       Absolute or relative path to the source file, or
                         ``""`` when extracted from an in-memory string.
        start_line:      1-based line number where this symbol begins.
        end_line:        1-based line number where this symbol ends (inclusive).
        signature:       Full parameter signature string, e.g.
                         ``"(self, x: int = 0) -> str"``.  Empty string for
                         imports and classes.
        docstring:       Content of the first expression-statement string
                         in the body (the docstring), or ``""`` if absent.
        decorators:      List of decorator strings in declaration order,
                         e.g. ``["staticmethod", "functools.wraps(fn)"]``.
        parent_class:    Name of the enclosing class when ``type`` is a method
                         variant; ``""`` otherwise.
        return_type_hint: Text of the return-type annotation (``-> X``), or
                          ``""`` when absent.
        module:          Dot-separated module path of imported name(s).
                         Populated only for ``type == "import"``; ``""`` otherwise.
        names:           List of imported names.  For ``import X`` it is
                         ``["X"]``; for ``from X import a, b`` it is
                         ``["a", "b"]``; for ``from X import *`` it is
                         ``["*"]``.  Empty for non-import symbols.
    """

    name: str
    type: str
    file_path: str
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)
    parent_class: str = ""
    return_type_hint: str = ""
    module: str = ""
    names: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        loc = f"[{self.start_line}-{self.end_line}]"
        parent = f" in {self.parent_class}" if self.parent_class else ""
        return f"Symbol({self.type}: {self.name}{parent} {loc})"


# ---------------------------------------------------------------------------
# SymbolExtractor
# ---------------------------------------------------------------------------


class SymbolExtractor:
    """Walk a tree-sitter AST and extract structured :class:`Symbol` objects.

    A single instance can be reused across many files; it internally caches
    one :class:`ASTParser` so grammar loading happens at most once per language.

    Args:
        parser: Optional pre-built :class:`ASTParser`.  When ``None``, a new
                one is created.  Inject a custom parser in tests.
    """

    def __init__(self, parser: ASTParser | None = None) -> None:
        self._parser: ASTParser = parser if parser is not None else ASTParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_file(
        self,
        file_path: str | Path,
        language: str | None = None,
    ) -> list[Symbol]:
        """Parse *file_path* from disk and extract all symbols.

        Args:
            file_path: Path to the source file.
            language:  Language override.  When ``None``, inferred from the
                       file extension via the parser's settings map.

        Returns:
            List of :class:`Symbol` objects in source order.

        Raises:
            UnsupportedLanguageError: If the language cannot be determined.
            ParseError: If the file cannot be read.
        """
        path = Path(file_path)
        tree: Tree = self._parser.parse_file(path, language=language)
        resolved_lang = language or _infer_language(path)
        return self._extract(tree, file_path=str(path), language=resolved_lang)

    def extract_from_source(
        self,
        source: str | bytes,
        *,
        language: str = "python",
        file_path: str = "",
    ) -> list[Symbol]:
        """Parse in-memory *source* and extract all symbols.

        Args:
            source:    Source code as ``str`` or ``bytes``.
            language:  Language name (default ``"python"``).
            file_path: Optional path label stored in each :class:`Symbol`.

        Returns:
            List of :class:`Symbol` objects in source order.
        """
        tree: Tree = self._parser.parse(source, language=language)
        return self._extract(tree, file_path=file_path, language=language)

    # ------------------------------------------------------------------
    # Internal extraction logic
    # ------------------------------------------------------------------

    def _extract(
        self,
        tree: Tree,
        *,
        file_path: str,
        language: str,
    ) -> list[Symbol]:
        """Walk *tree* and return all extracted symbols."""
        if language != "python":
            # Future: add JS/TS extraction here
            logger.debug("Symbol extraction for '%s' not yet implemented.", language)
            return []

        symbols: list[Symbol] = []
        root = tree.root_node
        self._visit_module(root, file_path=file_path, symbols=symbols)
        return symbols

    # ------------------------------------------------------------------
    # Module-level visitor
    # ------------------------------------------------------------------

    def _visit_module(
        self,
        module_node: Node,
        *,
        file_path: str,
        symbols: list[Symbol],
    ) -> None:
        """Iterate over top-level children of the module node."""
        for child in module_node.children:
            if child.type == "import_statement":
                symbols.extend(_extract_import(child, file_path=file_path))
            elif child.type == "import_from_statement":
                symbols.extend(_extract_import_from(child, file_path=file_path))
            elif child.type == "function_definition":
                sym = _extract_function(
                    child,
                    file_path=file_path,
                    parent_class="",
                    decorators=[],
                )
                symbols.append(sym)
                # Recursively extract nested functions from the body
                block = _get_child_by_type(child, ("block",))
                if block:
                    symbols.extend(
                        _extract_nested_functions(block, file_path=file_path)
                    )
            elif child.type == "class_definition":
                class_sym, method_syms = _extract_class(child, file_path=file_path)
                symbols.append(class_sym)
                symbols.extend(method_syms)
            elif child.type == "decorated_definition":
                decorators = _collect_decorators(child)
                inner = _get_child_by_type(
                    child, ("function_definition", "class_definition")
                )
                if inner is None:
                    continue
                if inner.type == "function_definition":
                    sym = _extract_function(
                        inner,
                        file_path=file_path,
                        parent_class="",
                        decorators=decorators,
                    )
                    symbols.append(sym)
                    block = _get_child_by_type(inner, ("block",))
                    if block:
                        symbols.extend(
                            _extract_nested_functions(block, file_path=file_path)
                        )
                elif inner.type == "class_definition":
                    class_sym, method_syms = _extract_class(
                        inner, file_path=file_path, decorators=decorators
                    )
                    symbols.append(class_sym)
                    symbols.extend(method_syms)


# ---------------------------------------------------------------------------
# Pure helpers (module-level functions for testability)
# ---------------------------------------------------------------------------


def _node_text(node: Node) -> str:
    """Decode a node's source bytes to a UTF-8 string."""
    raw = node.text if node.text is not None else b""
    return raw.decode("utf-8", errors="replace")


def _get_child_by_type(node: Node, types: tuple[str, ...]) -> Node | None:
    """Return the first direct child whose type is in *types*, or ``None``."""
    for child in node.children:
        if child.type in types:
            return child
    return None


def _get_named_children_by_type(node: Node, node_type: str) -> list[Node]:
    """Return all direct children matching *node_type*."""
    return [c for c in node.children if c.type == node_type]


def _collect_decorators(decorated_node: Node) -> list[str]:
    """Return decorator strings from a ``decorated_definition`` node.

    Each entry is the text after the ``@``, e.g. ``"staticmethod"`` or
    ``"functools.wraps(fn)"``.
    """
    result: list[str] = []
    for child in decorated_node.children:
        if child.type == "decorator":
            # The decorator node's text includes the leading '@'.
            dec_text = _node_text(child).lstrip("@").strip()
            result.append(dec_text)
    return result


def _extract_docstring(block_node: Node) -> str:
    """Extract the docstring from a function/class body block node.

    The docstring is the first statement in the block when it is a bare
    string literal (expression_statement containing a string).

    Returns:
        Docstring content (stripped), or ``""`` if absent.
    """
    if block_node.type != "block" or not block_node.children:
        return ""
    first = block_node.children[0]
    if first.type != "expression_statement":
        return ""
    string_node = _get_child_by_type(first, ("string",))
    if string_node is None:
        return ""
    # Collect string_content children (handles multi-line docstrings)
    parts = [_node_text(c) for c in string_node.children if c.type == "string_content"]
    return "".join(parts).strip()


def _extract_signature(func_node: Node) -> str:
    """Build the full signature string for a function node.

    Combines the parameters node and return-type annotation into a string
    like ``"(self, x: int) -> str"``.
    """
    params_node = _get_child_by_type(func_node, ("parameters",))
    params = _node_text(params_node) if params_node else "()"

    ret_node = _get_child_by_type(func_node, ("type",))
    ret = f" -> {_node_text(ret_node)}" if ret_node else ""

    return f"{params}{ret}"


def _extract_return_type(func_node: Node) -> str:
    """Return the return-type annotation text, or ``""`` if absent."""
    ret_node = _get_child_by_type(func_node, ("type",))
    return _node_text(ret_node) if ret_node else ""


def _is_async(func_node: Node) -> bool:
    """Return True if the function node is preceded by the ``async`` keyword.

    tree-sitter places ``async`` as an anonymous child of the function node
    when the function is ``async def``.
    """
    return any(c.type == "async" for c in func_node.children)


def _classify_method(decorators: list[str]) -> str:
    """Determine the method type from its decorator list."""
    for dec in decorators:
        name = dec.split("(")[0].strip()  # strip args: wraps(fn) -> wraps
        if name == "staticmethod":
            return STATIC_METHOD
        if name == "classmethod":
            return CLASS_METHOD
        if name == "property":
            return PROPERTY
    return METHOD


def _extract_function(
    func_node: Node,
    *,
    file_path: str,
    parent_class: str,
    decorators: list[str],
) -> Symbol:
    """Build a :class:`Symbol` from a ``function_definition`` node."""
    name_node = _get_child_by_type(func_node, ("identifier",))
    name = _node_text(name_node) if name_node else "<unknown>"

    block = _get_child_by_type(func_node, ("block",))
    docstring = _extract_docstring(block) if block else ""

    signature = _extract_signature(func_node)
    return_type = _extract_return_type(func_node)

    is_async = _is_async(func_node)
    if parent_class:
        base_type = _classify_method(decorators)
        # Preserve async distinction for plain methods (not static/class/property)
        sym_type = ASYNC_METHOD if (is_async and base_type == METHOD) else base_type
    else:
        sym_type = ASYNC_FUNCTION if is_async else FUNCTION

    # 1-based lines (parser already adds +1)
    start_line = func_node.start_point[0] + 1
    end_line = func_node.end_point[0] + 1

    return Symbol(
        name=name,
        type=sym_type,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        docstring=docstring,
        decorators=decorators,
        parent_class=parent_class,
        return_type_hint=return_type,
    )


def _extract_class(
    class_node: Node,
    *,
    file_path: str,
    decorators: list[str] | None = None,
) -> tuple[Symbol, list[Symbol]]:
    """Build a :class:`Symbol` for the class and its methods.

    Returns:
        A tuple ``(class_symbol, [method_symbols...])``.
    """
    decorators = decorators or []

    name_node = _get_child_by_type(class_node, ("identifier",))
    class_name = _node_text(name_node) if name_node else "<unknown>"

    # Base classes: argument_list node under class_definition
    bases: list[str] = []
    arg_list = _get_child_by_type(class_node, ("argument_list",))
    if arg_list:
        for child in arg_list.children:
            if child.is_named and child.type not in ("comment",):
                bases.append(_node_text(child))

    block = _get_child_by_type(class_node, ("block",))
    docstring = _extract_docstring(block) if block else ""

    start_line = class_node.start_point[0] + 1
    end_line = class_node.end_point[0] + 1

    class_sym = Symbol(
        name=class_name,
        type=CLASS,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        docstring=docstring,
        decorators=decorators,
        signature="(" + ", ".join(bases) + ")" if bases else "",
    )

    method_syms: list[Symbol] = []
    if block:
        method_syms = _extract_methods(
            block, file_path=file_path, class_name=class_name
        )

    return class_sym, method_syms


def _extract_methods(
    block_node: Node,
    *,
    file_path: str,
    class_name: str,
) -> list[Symbol]:
    """Extract all method symbols from a class body block."""
    methods: list[Symbol] = []
    for child in block_node.children:
        if child.type == "function_definition":
            sym = _extract_function(
                child,
                file_path=file_path,
                parent_class=class_name,
                decorators=[],
            )
            methods.append(sym)
            # Extract nested functions defined inside this method body
            inner_block = _get_child_by_type(child, ("block",))
            if inner_block:
                methods.extend(
                    _extract_nested_functions(inner_block, file_path=file_path)
                )
        elif child.type == "decorated_definition":
            decs = _collect_decorators(child)
            inner = _get_child_by_type(child, ("function_definition",))
            if inner:
                sym = _extract_function(
                    inner,
                    file_path=file_path,
                    parent_class=class_name,
                    decorators=decs,
                )
                methods.append(sym)
                inner_block = _get_child_by_type(inner, ("block",))
                if inner_block:
                    methods.extend(
                        _extract_nested_functions(inner_block, file_path=file_path)
                    )
        elif child.type == "class_definition":
            # Nested class -- extract it recursively
            nested_sym, nested_methods = _extract_class(child, file_path=file_path)
            # Mark nested class with parent info
            nested_sym.parent_class = class_name
            methods.append(nested_sym)
            methods.extend(nested_methods)
    return methods


def _extract_nested_functions(block_node: Node, *, file_path: str) -> list[Symbol]:
    """Recursively extract nested ``function_definition`` nodes from a block.

    Walks only the direct children of *block_node*, then recurses into each
    nested function's own block so arbitrary nesting depth is handled without
    hitting Python's recursion limit (tree-sitter trees are shallow in practice).

    Args:
        block_node: A ``block`` node (function or method body).
        file_path:  Path label stored in each :class:`Symbol`.

    Returns:
        List of :class:`Symbol` objects for all nested functions found.
    """
    nested: list[Symbol] = []
    for child in block_node.children:
        if child.type == "function_definition":
            sym = _extract_function(
                child,
                file_path=file_path,
                parent_class="",
                decorators=[],
            )
            nested.append(sym)
            inner_block = _get_child_by_type(child, ("block",))
            if inner_block:
                nested.extend(
                    _extract_nested_functions(inner_block, file_path=file_path)
                )
        elif child.type == "decorated_definition":
            decs = _collect_decorators(child)
            inner = _get_child_by_type(child, ("function_definition",))
            if inner:
                sym = _extract_function(
                    inner,
                    file_path=file_path,
                    parent_class="",
                    decorators=decs,
                )
                nested.append(sym)
                inner_block = _get_child_by_type(inner, ("block",))
                if inner_block:
                    nested.extend(
                        _extract_nested_functions(inner_block, file_path=file_path)
                    )
    return nested


def _extract_import(import_node: Node, *, file_path: str) -> list[Symbol]:
    """Build :class:`Symbol` objects from an ``import_statement`` node.

    ``import os, sys`` becomes two separate Import symbols.
    """
    symbols: list[Symbol] = []
    start_line = import_node.start_point[0] + 1
    end_line = import_node.end_point[0] + 1

    # Children of import_statement are dotted_name / aliased_import nodes
    for child in import_node.children:
        if child.type in ("dotted_name", "aliased_import"):
            name = _node_text(child).split(" as ")[0].strip()
            alias = (
                _node_text(child).split(" as ")[1].strip()
                if " as " in _node_text(child)
                else ""
            )
            sym_name = alias if alias else name.split(".")[-1]
            symbols.append(
                Symbol(
                    name=sym_name,
                    type=IMPORT,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    module=name,
                    names=[sym_name],
                )
            )
    return symbols


def _extract_import_from(import_from_node: Node, *, file_path: str) -> list[Symbol]:
    """Build a single :class:`Symbol` from a ``from X import Y`` node.

    All imported names are collected into :attr:`Symbol.names`.
    A ``from X import *`` produces ``names=["*"]``.
    """
    start_line = import_from_node.start_point[0] + 1
    end_line = import_from_node.end_point[0] + 1

    # First dotted_name child is the module
    children = import_from_node.children
    module_node = next(
        (c for c in children if c.type in ("dotted_name", "relative_import")), None
    )
    module = _node_text(module_node).strip() if module_node else ""

    # Collect imported names
    names: list[str] = []
    for child in children:
        if child.type == "wildcard_import":
            names.append("*")
        elif child.type in ("dotted_name", "aliased_import"):
            # Skip the module dotted_name (first occurrence)
            if child is module_node:
                continue
            raw = _node_text(child)
            alias_name = raw.split(" as ")[1].strip() if " as " in raw else raw.strip()
            names.append(alias_name)

    sym_name = f"from {module} import {', '.join(names)}" if names else module

    return [
        Symbol(
            name=sym_name,
            type=IMPORT,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            module=module,
            names=names,
        )
    ]


def _infer_language(path: Path) -> str:
    """Infer language from file extension for error messages."""
    from src.reporag.config import settings

    lang = settings.extension_map.get(path.suffix.lower())
    return lang or "python"
