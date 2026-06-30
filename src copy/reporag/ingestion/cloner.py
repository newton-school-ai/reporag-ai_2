"""Git repository cloner and file discovery service.

Clones a Git repository to a temporary directory and discovers parseable source
files, returning a deterministic manifest of :class:`FileInfo` records.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from git import GitCommandError, Repo

from reporag.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Default extension map keyed by language name. Only languages listed in
# ``settings.supported_languages`` are included unless overridden.
DEFAULT_LANGUAGE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "python": (".py",),
    "javascript": (".js", ".jsx", ".mjs", ".cjs"),
    "typescript": (".ts", ".tsx"),
}

# Directory names skipped during file discovery.
IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".cache",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "htmlcov",
        "coverage",
    }
)


class CloneError(Exception):
    """Raised when cloning or validating a repository fails."""


@dataclass(frozen=True)
class FileInfo:
    """A discovered source file in a cloned repository."""

    path: str
    language: str
    size_bytes: int


def normalize_repo_url(url: str) -> str:
    """Normalize a repository location for Git.

    Local filesystem paths are converted to ``file://`` URIs so Git honors
    shallow-clone depth consistently across platforms.

    Args:
        url: HTTPS clone URL or local repository path.

    Returns:
        A URL string suitable for :meth:`git.Repo.clone_from`.
    """
    candidate = Path(url).expanduser()
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve().as_uri()
    return url


class RepoCloner:
    """Clone repositories and discover parseable source files."""

    def __init__(
        self,
        settings: Settings | None = None,
        extensions: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Initialize the cloner.

        Args:
            settings: Application settings. Defaults to the global singleton.
            extensions: Optional override for language-to-extension mapping.
        """
        self._settings = settings or get_settings()
        self._extensions_override = extensions
        self.last_clone_path: Path | None = None

    def extension_map(self) -> dict[str, tuple[str, ...]]:
        """Return the active language-to-extension mapping.

        Uses ``extensions`` passed to the constructor when set; otherwise
        filters :data:`DEFAULT_LANGUAGE_EXTENSIONS` by
        ``settings.supported_languages``.
        """
        if self._extensions_override is not None:
            return dict(self._extensions_override)

        supported = {lang.lower() for lang in self._settings.supported_languages_list}
        return {
            language: extensions
            for language, extensions in DEFAULT_LANGUAGE_EXTENSIONS.items()
            if language in supported
        }

    def clone(
        self,
        repo_url_or_path: str,
        *,
        branch: str | None = None,
        depth: int | None = None,
    ) -> Path:
        """Clone a repository into a new temporary directory.

        Args:
            repo_url_or_path: HTTPS URL or local path to the repository.
            branch: Optional branch name to check out after cloning.
            depth: Optional shallow-clone depth. When ``None``, uses
                ``settings.clone_depth``. A value of ``0`` performs a full
                clone.

        Returns:
            Path to the cloned working tree.

        Raises:
            CloneError: If cloning or post-clone validation fails. The
                temporary directory is removed before the error is raised.
        """
        clone_url = normalize_repo_url(repo_url_or_path)
        dest = Path(tempfile.mkdtemp(prefix="reporag_clone_"))

        clone_kwargs: dict[str, object] = {}
        resolved_depth = self._resolve_depth(depth)
        if resolved_depth is not None:
            clone_kwargs["depth"] = resolved_depth
        if branch:
            clone_kwargs["branch"] = branch

        try:
            Repo.clone_from(clone_url, dest, **clone_kwargs)
            self._validate_repo_size(dest)
        except (GitCommandError, OSError, ValueError) as exc:
            shutil.rmtree(dest, ignore_errors=True)
            raise CloneError(
                f"Failed to clone repository {repo_url_or_path!r}: {exc}"
            ) from exc

        self.last_clone_path = dest
        logger.info("Cloned %s to %s", repo_url_or_path, dest)
        return dest

    def discover(self, root: Path | str) -> list[FileInfo]:
        """Walk a repository tree and return a sorted source-file manifest.

        Args:
            root: Root directory of the cloned repository.

        Returns:
            Sorted list of :class:`FileInfo` entries for matching files.
        """
        root_path = Path(root)
        extension_to_language = self._extension_to_language_map()
        manifest: list[FileInfo] = []

        for dirpath, _dirnames, filenames in self._walk_dirs(root_path):
            current_dir = Path(dirpath)

            for filename in filenames:
                file_path = current_dir / filename
                language = extension_to_language.get(file_path.suffix.lower())
                if language is None:
                    continue

                relative_path = file_path.relative_to(root_path).as_posix()
                manifest.append(
                    FileInfo(
                        path=relative_path,
                        language=language,
                        size_bytes=file_path.stat().st_size,
                    )
                )

        manifest.sort(key=lambda item: item.path)
        return manifest

    def clone_and_discover(
        self,
        repo_url_or_path: str,
        *,
        branch: str | None = None,
        depth: int | None = None,
    ) -> list[FileInfo]:
        """Clone a repository and return its source-file manifest.

        Args:
            repo_url_or_path: HTTPS URL or local path to the repository.
            branch: Optional branch name to check out after cloning.
            depth: Optional shallow-clone depth override.

        Returns:
            Sorted manifest of discovered source files.

        Raises:
            CloneError: If cloning or validation fails.
        """
        root = self.clone(repo_url_or_path, branch=branch, depth=depth)
        return self.discover(root)

    def cleanup(self) -> None:
        """Remove the most recently cloned repository directory, if any."""
        if self.last_clone_path is None:
            return

        shutil.rmtree(self.last_clone_path, ignore_errors=True)
        logger.info("Removed cloned repository at %s", self.last_clone_path)
        self.last_clone_path = None

    def _resolve_depth(self, depth: int | None) -> int | None:
        """Map caller/config depth values to Git clone depth."""
        if depth is not None:
            return depth if depth > 0 else None
        configured = self._settings.clone_depth
        return configured if configured > 0 else None

    def _extension_to_language_map(self) -> dict[str, str]:
        """Invert the extension map for suffix lookups."""
        mapping: dict[str, str] = {}
        for language, extensions in self.extension_map().items():
            for extension in extensions:
                mapping[extension.lower()] = language
        return mapping

    def _walk_dirs(self, root: Path):
        """Yield os.walk tuples while skipping ignored directory names."""
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in IGNORED_DIR_NAMES]
            yield dirpath, dirnames, filenames

    def _validate_repo_size(self, root: Path) -> None:
        """Ensure the cloned repository does not exceed the configured limit."""
        max_bytes = self._settings.max_repo_size_mb * 1024 * 1024
        total_size = sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        )
        if total_size > max_bytes:
            raise ValueError(
                f"Repository size {total_size} bytes exceeds limit of "
                f"{max_bytes} bytes ({self._settings.max_repo_size_mb} MB)"
            )
