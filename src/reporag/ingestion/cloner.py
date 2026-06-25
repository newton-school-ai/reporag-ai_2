"""Git repository cloner and file discovery service.

Clones a Git repository to a temp directory and discovers all parseable
source files, returning a manifest of (file_path, language, size_bytes).
"""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from git import Repo

from reporag.config import settings


class CloneError(Exception):
    """Raised when repository cloning fails."""


IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".next",
    "target",
    "vendor",
}


@dataclass
class FileInfo:
    path: str
    language: str
    size_bytes: int


class RepoCloner:
    """Clone repositories and discover source files."""

    def __init__(
        self,
        extensions: dict[str, str] | None = None,
    ) -> None:
        self.extensions = (
            extensions if extensions is not None else settings.language_extensions
        )

        self.last_clone_path: str | None = None

    def _repository_size_mb(self, repo_path: str) -> float:
        """Return repository size in megabytes."""

        total_bytes = sum(
            path.stat().st_size for path in Path(repo_path).rglob("*") if path.is_file()
        )

        return total_bytes / (1024 * 1024)

    def cleanup(self) -> None:
        """Remove the last cloned repository."""

        if self.last_clone_path:
            shutil.rmtree(
                self.last_clone_path,
                ignore_errors=True,
            )
            self.last_clone_path = None

    def discover_files(
        self,
        repo_path: str,
        extensions: dict[str, str] | None = None,
    ) -> list[FileInfo]:
        """Discover supported source files in a repository."""

        active_extensions = extensions if extensions is not None else self.extensions

        manifest: list[FileInfo] = []

        for file_path in Path(repo_path).rglob("*"):
            if not file_path.is_file():
                continue

            if any(part in IGNORED_DIRS for part in file_path.parts):
                continue

            language = active_extensions.get(file_path.suffix)

            if language is None:
                continue

            manifest.append(
                FileInfo(
                    path=str(file_path),
                    language=language,
                    size_bytes=file_path.stat().st_size,
                )
            )
        manifest.sort(key=lambda file: file.path)
        return manifest

    def clone_repository(
        self,
        repo_source: str,
        branch: str | None = None,
        depth: int | None = None,
    ) -> str:
        """Clone a repository and return local path."""

        temp_dir = tempfile.mkdtemp(prefix="reporag_")
        depth = depth if depth is not None else settings.clone_depth

        try:

            clone_args: dict[str, object] = {
                "to_path": temp_dir,
                "depth": depth,
            }

            if branch:
                clone_args["branch"] = branch
                clone_args["single_branch"] = True

            if Path(repo_source).exists():
                Repo.clone_from(
                    Path(repo_source).resolve().as_uri(),
                    **clone_args,
                )
            else:
                Repo.clone_from(
                    repo_source,
                    **clone_args,
                )

            if self._repository_size_mb(temp_dir) > settings.max_repo_size_mb:
                raise CloneError(
                    f"Repository exceeds maximum size of "
                    f"{settings.max_repo_size_mb} MB"
                )

            self.last_clone_path = temp_dir

            return temp_dir
        except CloneError:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise CloneError(f"Failed to clone repository: {repo_source}") from exc

    def clone_and_discover(
        self,
        repo_source: str,
        branch: str | None = None,
        depth: int | None = None,
        extensions: dict[str, str] | None = None,
    ) -> list[FileInfo]:
        repo_path = self.clone_repository(
            repo_source,
            branch=branch,
            depth=depth,
        )

        try:
            return self.discover_files(
                repo_path,
                extensions=extensions,
            )
        except Exception:
            shutil.rmtree(repo_path, ignore_errors=True)
            raise
