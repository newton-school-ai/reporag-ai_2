"""Shared module-path resolution for the code knowledge graph.

Both the call graph (Issue 9) and the import dependency graph (Issue 10) need to
map a written import -- ``from db import get_user`` or ``from .utils import x`` --
to the project file that defines the module, and to a single *canonical* dotted
name for that module.  :class:`ModuleIndex` centralises that logic so the two
builders resolve modules identically.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath


def module_parts(file_path: str) -> list[str]:
    """Return the dotted-module path components for *file_path*.

    Strips the file extension and folds ``__init__`` into its package directory
    (``pkg/__init__.py`` -> ``["pkg"]``).  Uses POSIX path semantics regardless
    of host OS so module names are stable across platforms.
    """
    pure = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(pure.parts[:-1]) + [pure.stem]
    if pure.stem == "__init__":
        parts = parts[:-1]
    return [p for p in parts if p not in ("", ".", "/")]


class ModuleIndex:
    """Resolves import module names to project file paths and canonical names.

    - **Why it exists**: Cross-file resolution and cycle detection hinge on
      mapping an import like ``from db import get_user`` (module ``"db"``) to the
      file that defines ``db`` -- here ``examples/sample_repo/db.py`` -- and on
      giving every module a single canonical name so a bare ``import db`` and a
      fully qualified ``import examples.sample_repo.db`` line up.
    - **Algorithm**: For every file it precomputes all dotted *suffixes* of the
      path (``examples.sample_repo.db``, ``sample_repo.db``, ``db``) and maps
      each to the file.  A bare ``import db`` matches the shortest suffix; a fully
      qualified import matches the longest.  The *canonical* name of a file is
      always its longest (fully qualified) form.
    - **Edge cases**: ``__init__.py`` collapses to its package directory so a
      package import resolves to the package.  Relative imports (``.``, ``..``)
      are rebuilt into an absolute dotted path against the importing file.
    """

    def __init__(self, files: Iterable[str]) -> None:
        """Build the suffix map and canonical-name table from *files*."""
        self._suffix_map: dict[str, list[str]] = {}
        self._canonical: dict[str, str] = {}
        for file_path in files:
            parts = module_parts(file_path)
            self._canonical[file_path] = ".".join(parts)
            for candidate in self._candidates(parts):
                bucket = self._suffix_map.setdefault(candidate, [])
                if file_path not in bucket:
                    bucket.append(file_path)

    @staticmethod
    def _parts(file_path: str) -> list[str]:
        """Return the dotted-module path components for *file_path*."""
        return module_parts(file_path)

    @staticmethod
    def _candidates(parts: list[str]) -> list[str]:
        """Return every dotted suffix of a module's *parts*."""
        return [".".join(parts[i:]) for i in range(len(parts)) if parts[i:]]

    def module_name(self, file_path: str) -> str:
        """Return the canonical (fully qualified) dotted name of *file_path*."""
        cached = self._canonical.get(file_path)
        if cached is not None:
            return cached
        return ".".join(module_parts(file_path))

    def normalize(self, module: str, importing_file: str) -> str:
        """Return the absolute dotted form of *module* seen from *importing_file*.

        Absolute modules are returned unchanged.  A relative module (leading
        dots) is rebuilt against the importing file's package: one dot means
        "this package", each extra dot ascends one level, and the remainder is
        appended.
        """
        if not module.startswith("."):
            return module

        level = len(module) - len(module.lstrip("."))
        remainder = module[level:]

        pkg = module_parts(importing_file)[:-1]  # importing module's package
        ascend = level - 1  # a single dot means "this package"
        if ascend > 0:
            pkg = pkg[:-ascend] if ascend <= len(pkg) else []

        parts = pkg + (remainder.split(".") if remainder else [])
        return ".".join(parts)

    def resolve_absolute(self, module: str) -> list[str]:
        """Return files matching an absolute dotted *module* name."""
        return list(self._suffix_map.get(module, []))

    def resolve_relative(self, module: str, importing_file: str) -> list[str]:
        """Resolve a relative import (``.``, ``..pkg``) to project files.

        - **Algorithm**: Counts leading dots to find how many package levels to
          ascend from *importing_file*'s package, appends the remaining dotted
          remainder, then looks the rebuilt absolute name up in the suffix map.
        - **Edge cases**: A bare ``from . import x`` (empty remainder) resolves
          to the importing file's own package directory.
        """
        target = self.normalize(module, importing_file)
        if not target:
            return []
        return self.resolve_absolute(target)

    def resolve(self, module: str, importing_file: str) -> list[str]:
        """Resolve *module* (absolute or relative) to candidate files."""
        if module.startswith("."):
            return self.resolve_relative(module, importing_file)
        return self.resolve_absolute(module)

    def resolve_file(self, module: str, importing_file: str) -> str | None:
        """Resolve *module* to a single project file, or ``None``.

        When several files share a suffix the lowest path (sorted) wins, so
        resolution is deterministic.
        """
        files = self.resolve(module, importing_file)
        if not files:
            return None
        return sorted(files)[0]
