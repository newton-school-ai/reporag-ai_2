"""Git repository cloner and file discovery service.

Clones a Git repository (HTTPS URL or local path) to a temporary directory
and discovers all parseable source files, returning a typed manifest of
``FileInfo`` objects: (file_path, language, size_bytes).

Usage::

    from reporag.ingestion.cloner import RepoCloner

    # As a context manager (recommended - auto cleanup)
    with RepoCloner() as cloner:
        manifest = cloner.clone_and_discover(
            "https://github.com/pallets/click", branch="main"
        )
        for f in manifest:
            print(f.file_path, f.language, f.size_bytes)

    # Manual lifecycle
    cloner = RepoCloner()
    manifest = cloner.clone_and_discover("https://github.com/pallets/click")
    # ... use manifest and cloner.clone_dir ...
    cloner.cleanup()
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from dataclasses import dataclass

import git

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maps file extensions to canonical language names.
#: Used as the default filter for :class:`RepoCloner`.
LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
}

#: Branch names tried in order when no branch is specified for a remote URL.
_BRANCH_CANDIDATES = ("main", "master")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ClonerError(RuntimeError):
    """Raised when cloning or file discovery fails.

    Wraps lower-level ``git`` exceptions so callers can catch cloner
    failures distinctly without depending on gitpython internals.
    """


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FileInfo:
    """A single entry in the repository file manifest.

    Attributes:
        file_path:   Absolute path to the file inside the cloned temp dir.
        language:    Canonical language name (e.g. ``"python"``).
        size_bytes:  File size in bytes at the time of discovery.
    """

    file_path: str
    language: str
    size_bytes: int


# ---------------------------------------------------------------------------
# RepoCloner
# ---------------------------------------------------------------------------


class RepoCloner:
    """Clone a Git repository and discover its parseable source files.

    Args:
        extensions: Mapping of file extension (e.g. ``".py"``) to language
            name (e.g. ``"python"``).  Defaults to :data:`LANGUAGE_MAP`
            which covers Python, JavaScript, and TypeScript.

    Examples:
        >>> cloner = RepoCloner()
        >>> manifest = cloner.clone_and_discover(
        ...     "https://github.com/pallets/click", branch="main"
        ... )
        >>> cloner.cleanup()
    """

    def __init__(self, extensions: dict[str, str] | None = None) -> None:
        self._extensions: dict[str, str] = (
            extensions if extensions is not None else LANGUAGE_MAP
        )
        self._tmpdir: tempfile.TemporaryDirectory | None = None  # type: ignore[type-arg]
        #: Root directory of the cloned repository. Set after a successful
        #: :meth:`clone_and_discover` call; ``None`` before or after cleanup.
        self.clone_dir: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_and_discover(
        self,
        repo_url: str,
        branch: str | None = None,
        depth: int = 1,
    ) -> list[FileInfo]:
        """Clone *repo_url* and return a manifest of parseable source files.

        Args:
            repo_url: HTTPS URL (e.g. ``"https://github.com/org/repo"``) **or**
                an absolute local filesystem path to an existing git repository.
            branch:   Branch to clone.  When ``None`` and *repo_url* is a
                remote URL, the branch is auto-detected (tries ``main`` then
                ``master``).  For local paths the branch is left unspecified.
            depth:    Shallow-clone depth.  ``1`` (the default) fetches only
                the latest commit, which is 5x+ faster for large repos.
                Ignored for local-path sources (local clones are always full).

        Returns:
            List of :class:`FileInfo` objects, one per matching file.

        Raises:
            ClonerError: If cloning fails (bad URL, invalid branch, network
                error) or if file discovery encounters an unrecoverable error.
                The temporary directory is always cleaned up before raising.
        """
        self._tmpdir = tempfile.TemporaryDirectory()
        try:
            source, is_local = self._resolve_source(repo_url)

            resolved_branch: str | None = branch
            if not is_local and resolved_branch is None:
                resolved_branch = self._auto_detect_branch(source)
                logger.info("Auto-detected branch '%s' for %s", resolved_branch, source)

            clone_depth = None if is_local else depth
            self._clone(source, resolved_branch, clone_depth, self._tmpdir.name)

            self.clone_dir = self._tmpdir.name
            manifest = self._discover_files(self._tmpdir.name)
            logger.info("Discovered %d file(s) in '%s'", len(manifest), repo_url)
            return manifest

        except ClonerError:
            # Already a ClonerError - cleanup and re-raise as-is
            self._cleanup_tmpdir()
            raise
        except Exception as exc:
            self._cleanup_tmpdir()
            raise ClonerError(
                f"Failed to clone or discover files from '{repo_url}': {exc}"
            ) from exc

    def cleanup(self) -> None:
        """Delete the cloned repository's temporary directory.

        Safe to call multiple times.  After calling this, :attr:`clone_dir`
        is reset to ``None`` and the cloned files are no longer accessible.
        """
        self._cleanup_tmpdir()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> RepoCloner:
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_tmpdir(self) -> None:
        """Internal: delete temp dir and reset state."""
        if self._tmpdir is not None:
            with contextlib.suppress(Exception):
                self._tmpdir.cleanup()
            self._tmpdir = None
            self.clone_dir = None

    def _resolve_source(self, repo_url: str) -> tuple[str, bool]:
        """Determine whether *repo_url* is a local path or a remote URL.

        Returns:
            ``(source, is_local)`` where *is_local* is ``True`` when
            *repo_url* points to a directory on the local filesystem.
        """
        is_local = os.path.isdir(repo_url)
        return repo_url, is_local

    def _auto_detect_branch(self, repo_url: str) -> str:
        """Try ``main`` then ``master``; raise :exc:`ClonerError` if neither exists.

        Uses ``git ls-remote`` which only downloads a few bytes and does not
        require a full clone.

        Raises:
            ClonerError: If neither ``main`` nor ``master`` is found.
        """
        g = git.cmd.Git()
        for candidate in _BRANCH_CANDIDATES:
            try:
                output = g.ls_remote("--heads", repo_url, candidate)
                if candidate in output:
                    return candidate
            except git.exc.GitCommandError:
                continue

        raise ClonerError(
            f"Could not find 'main' or 'master' branch in '{repo_url}'. "
            "Specify the branch explicitly via the 'branch' argument."
        )

    def _clone(
        self,
        source: str,
        branch: str | None,
        depth: int | None,
        target_dir: str,
    ) -> git.Repo:
        """Run ``git clone`` into *target_dir*.

        Args:
            source:     URL or local path to clone from.
            branch:     Branch to clone, or ``None`` to use the remote default.
            depth:      Shallow clone depth, or ``None`` for a full clone.
            target_dir: Directory to clone into.

        Returns:
            The cloned :class:`git.Repo` object.

        Raises:
            ClonerError: Wraps any :class:`git.exc.GitCommandError`.
        """
        kwargs: dict[str, object] = {"to_path": target_dir}
        if branch is not None:
            kwargs["branch"] = branch
        if depth is not None:
            kwargs["depth"] = depth

        try:
            repo = git.Repo.clone_from(source, **kwargs)
            logger.debug(
                "Cloned '%s' (branch=%s, depth=%s) to '%s'",
                source,
                branch,
                depth,
                target_dir,
            )
            return repo
        except git.exc.GitCommandError as exc:
            raise ClonerError(
                f"git clone failed for '{source}' (branch={branch!r}): {exc.stderr.strip()}"
            ) from exc

    def _discover_files(self, root_dir: str) -> list[FileInfo]:
        """Walk *root_dir* and return :class:`FileInfo` for every matching file.

        The ``.git`` directory is pruned from traversal so its internal files
        are never returned.

        Args:
            root_dir: Absolute path to the root of the cloned repository.

        Returns:
            Sorted list of :class:`FileInfo` objects.
        """
        results: list[FileInfo] = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Prune .git directory in-place so os.walk never descends into it.
            dirnames[:] = [d for d in dirnames if d != ".git"]

            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self._extensions:
                    continue
                full_path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    size = 0
                results.append(
                    FileInfo(
                        file_path=full_path,
                        language=self._extensions[ext],
                        size_bytes=size,
                    )
                )

        results.sort(key=lambda f: f.file_path)
        return results
