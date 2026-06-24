"""Git repository cloner and file discovery service (Issue 5).

The ingestion pipeline begins here: clone a Git repository (a remote HTTPS URL
or a local path) into a temporary directory, then walk the working tree and
return a manifest of the source files worth parsing.

The manifest is a list of :class:`FileInfo` records -- one per discovered file,
carrying its repo-relative path, detected language, and size in bytes. Files
are matched by extension; the default extension set is derived from
``settings.supported_languages`` so discovery stays in sync with configuration.

Temporary clone directories are removed automatically if anything fails; on
success the directory is kept (downstream stages read from it) and the caller
can release it via :meth:`RepoCloner.cleanup`.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from git import Repo

from reporag.config import settings

# Canonical language -> file extensions. Discovery maps a file's extension back
# to one of these language names.
_LANGUAGE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "python": (".py", ".pyi"),
    "javascript": (".js", ".jsx", ".mjs", ".cjs"),
    "typescript": (".ts", ".tsx"),
}

# Directories that are never source worth indexing: VCS metadata, vendored
# dependencies, build output, caches, and editor folders.
_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".tox",
        ".idea",
        ".vscode",
    }
)

# Matches an explicit URL scheme ("https://", "git://", "ssh://", ...).
_URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


class CloneError(RuntimeError):
    """Raised when cloning fails or a cloned repository is rejected.

    Whenever this is raised the temporary clone directory has already been
    removed, so the caller never has to clean up after a failed clone.
    """


@dataclass(frozen=True)
class FileInfo:
    """A single discovered source file in a clone manifest."""

    path: str  # repo-relative, POSIX-style ("pkg/util.py")
    language: str  # canonical language name ("python")
    size_bytes: int


def _extensions_for_languages(languages: list[str]) -> dict[str, str]:
    """Build an extension -> language map for the given language names."""
    extensions: dict[str, str] = {}
    for language in languages:
        for extension in _LANGUAGE_EXTENSIONS.get(language, ()):
            extensions[extension] = language
    return extensions


class RepoCloner:
    """Clone Git repositories and discover their parseable source files.

    Defaults come from :mod:`reporag.config` (the languages to discover, the
    shallow-clone depth, and the maximum repository size), so a zero-argument
    ``RepoCloner()`` is configured the same way as the rest of the app. Any of
    those can be overridden per instance.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        extensions: dict[str, str] | None = None,
        max_repo_size_mb: float | None = None,
    ) -> None:
        if extensions is not None:
            self.extensions = dict(extensions)
        else:
            chosen = (
                languages
                if languages is not None
                else settings.supported_languages_list
            )
            self.extensions = _extensions_for_languages(chosen)
        self.max_repo_size_mb = (
            max_repo_size_mb
            if max_repo_size_mb is not None
            else settings.max_repo_size_mb
        )
        self.last_clone_path: Path | None = None

    # ----- Public API -----

    def clone_and_discover(
        self,
        source: str,
        branch: str | None = None,
        shallow: bool = True,
        depth: int | None = None,
    ) -> list[FileInfo]:
        """Clone ``source`` and return the manifest of its source files."""
        repo_path = self.clone(source, branch=branch, shallow=shallow, depth=depth)
        return self.discover_files(repo_path)

    def clone(
        self,
        source: str,
        branch: str | None = None,
        shallow: bool = True,
        depth: int | None = None,
    ) -> Path:
        """Clone ``source`` into a fresh temp directory and return its path.

        ``branch`` selects a branch (default: the remote's default branch).
        When ``shallow`` is true the clone is truncated to ``depth`` commits
        (default: ``settings.clone_depth``), which is what makes cloning large
        histories fast. The temp directory is removed if the clone fails or the
        repository exceeds the configured size limit.
        """
        destination = Path(tempfile.mkdtemp(prefix="reporag-clone-"))
        try:
            clone_kwargs: dict[str, object] = {}
            if branch is not None:
                clone_kwargs["branch"] = branch
            if shallow:
                clone_kwargs["depth"] = (
                    depth if depth is not None else settings.clone_depth
                )
                clone_kwargs["single_branch"] = True
            Repo.clone_from(
                self._normalize_source(source, shallow=shallow),
                str(destination),
                **clone_kwargs,
            )
            self._enforce_size_limit(destination)
        except Exception as exc:
            shutil.rmtree(destination, ignore_errors=True)
            if isinstance(exc, CloneError):
                raise
            raise CloneError(f"Failed to clone {source!r}: {exc}") from exc
        self.last_clone_path = destination
        return destination

    def discover_files(self, repo_path: str | Path) -> list[FileInfo]:
        """Walk ``repo_path`` and return a manifest of matching source files.

        Files are included when their extension maps to a configured language;
        ignored directories (``.git``, ``node_modules``, caches, ...) are
        skipped. The manifest is sorted by path for deterministic output.
        """
        root = Path(repo_path)
        manifest: list[FileInfo] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in _IGNORED_DIRS for part in relative.parts):
                continue
            language = self.extensions.get(path.suffix.lower())
            if language is None:
                continue
            manifest.append(
                FileInfo(
                    path=relative.as_posix(),
                    language=language,
                    size_bytes=path.stat().st_size,
                )
            )
        manifest.sort(key=lambda info: info.path)
        return manifest

    def cleanup(self, repo_path: str | Path | None = None) -> None:
        """Remove a cloned temp directory (defaults to the last clone)."""
        target = Path(repo_path) if repo_path is not None else self.last_clone_path
        if target is None:
            return
        shutil.rmtree(target, ignore_errors=True)
        if self.last_clone_path is not None and target == self.last_clone_path:
            self.last_clone_path = None

    # ----- Internals -----

    @staticmethod
    def _normalize_source(source: str, shallow: bool) -> str:
        """Return a clone-ready source string.

        Git ignores ``--depth`` for local-path clones unless the path is given
        as a ``file://`` URL, so for shallow clones an existing local path is
        converted to its file URI. Remote URLs are returned unchanged.
        """
        source = str(source)
        if _URL_SCHEME.match(source) or source.startswith("git@"):
            return source
        candidate = Path(source).expanduser()
        if shallow and candidate.exists():
            return candidate.resolve().as_uri()
        return source

    def _enforce_size_limit(self, repo_path: Path) -> None:
        """Raise :class:`CloneError` if the clone exceeds the size limit."""
        if self.max_repo_size_mb is None:
            return
        total_bytes = sum(
            path.stat().st_size for path in repo_path.rglob("*") if path.is_file()
        )
        size_mb = total_bytes / (1024 * 1024)
        if size_mb > self.max_repo_size_mb:
            raise CloneError(
                f"Repository size {size_mb:.1f} MB exceeds the configured "
                f"limit of {self.max_repo_size_mb} MB."
            )
