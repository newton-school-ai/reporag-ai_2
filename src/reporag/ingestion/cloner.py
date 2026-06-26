"""Git repository cloner and file discovery service.

Clones a Git repository to a temp directory and discovers all parseable
source files, returning a manifest of FileEntry(path, language, size_bytes).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.reporag.config import settings

logger = logging.getLogger(__name__)

# Default language extension mappings from settings config
SUPPORTED_EXTENSIONS = getattr(
    settings,
    "extension_map",
    {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
    },
)


class CloneError(Exception):
    """Raised when a repository cloning or discovery operation fails."""

    pass


@dataclass(frozen=True)
class FileEntry:
    """Deterministic structural metadata for an ingested source file."""

    path: str
    language: str
    size_bytes: int


# Backwards-compatible alias for FileEntry
FileInfo = FileEntry


class RepoCloner:
    """A repository cloner and file discovery service.

    Fully integrated with the application configuration settings.
    """

    def __init__(self) -> None:
        """Initialize the repository cloner."""
        self.max_repo_size_bytes = settings.max_repo_size_mb * 1024 * 1024
        self.default_depth = settings.clone_depth
        self.extension_map = SUPPORTED_EXTENSIONS
        self.last_clone_path: Path | None = None
        self.ignored_dirs: set[str] = {
            ".git",
            "node_modules",
            "dist",
            "build",
            "__pycache__",
            ".venv",
            "venv",
            "env",
            ".pytest_cache",
            ".eggs",
        }

    def clone_and_discover(
        self,
        repo_url: str,
        branch: str | None = None,
        shallow: bool = True,
        extensions: dict[str, str] | None = None,
    ) -> list[FileEntry]:
        """Clones a repository, validates size constraints, and returns a manifest.

        Args:
            repo_url: Remote HTTPS URL or local repository path.
            branch: Optional branch name. If not provided, clones the default branch.
            shallow: If True, uses shallow clone depth. Defaults to True.
            extensions: Optional custom extension dictionary mappings.

        Returns:
            A list of FileEntry objects representing discovered source files.

        Raises:
            CloneError: If cloning or discovery fails.
        """
        # Convert local path to file:// URI for depth to be honored consistently by Git
        if os.path.exists(repo_url):
            local_path = Path(repo_url).resolve()
            repo_url = local_path.as_uri()

        # Create a temporary directory in a safe workspace location
        base_temp = Path(tempfile.gettempdir())
        target_dir = Path(
            shutil.os.path.join(base_temp, f"reporag_{os.urandom(4).hex()}")
        )
        self.last_clone_path = target_dir

        try:
            self._execute_clone(repo_url, target_dir, branch, shallow)
            manifest = self._discover_files(target_dir, extensions)
            return manifest
        except Exception as e:
            logger.error(f"Ingestion pipeline failed for repository {repo_url}: {e}")
            self.cleanup()
            if not isinstance(e, CloneError):
                raise CloneError(f"Internal ingestion failure: {e}") from e
            raise

    def _execute_clone(
        self,
        repo_url: str,
        target_dir: Path,
        branch: str | None,
        shallow: bool,
    ) -> None:
        """Executes the git clone subprocess command synchronously."""
        cmd = ["git", "clone"]
        if shallow:
            # Respect configured default depth if available, otherwise default to 1
            depth_val = self.default_depth if self.default_depth > 0 else 1
            cmd.extend(["--depth", str(depth_val)])
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([repo_url, str(target_dir)])

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            raise CloneError(f"Failed to initiate git subprocess: {e}") from e

        if process.returncode != 0:
            error_msg = (
                process.stderr.strip() if process.stderr else "Unknown git clone error"
            )
            raise CloneError(
                f"Git clone failed with code {process.returncode}: {error_msg}"
            )

    def _discover_files(
        self,
        base_path: Path,
        extensions: dict[str, str] | None = None,
    ) -> list[FileEntry]:
        """Discovers files in the cloned directory and filters by extension.

        Args:
            base_path: Path to the cloned repository root.
            extensions: Optional custom extension mappings.

        Returns:
            A sorted list of FileEntry objects.
        """
        manifest: list[FileEntry] = []
        total_size = 0

        # Resolve extension map prioritizing dynamic parameter, then instance map, then global
        ext_map = extensions or self.extension_map or settings.extension_map

        for root, dirs, files in os.walk(base_path):
            # Prune ignored directories and hidden directories in-place
            dirs[:] = [
                d for d in dirs if d not in self.ignored_dirs and not d.startswith(".")
            ]

            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                if ext in ext_map:
                    try:
                        size = file_path.stat().st_size
                        total_size += size

                        if total_size > self.max_repo_size_bytes:
                            raise CloneError(
                                f"Repository size exceeds maximum limit of "
                                f"{settings.max_repo_size_mb} MB."
                            )

                        rel_path = file_path.relative_to(base_path).as_posix()
                        manifest.append(
                            FileEntry(
                                path=rel_path,
                                language=ext_map[ext],
                                size_bytes=size,
                            )
                        )
                    except OSError as e:
                        logger.warning(f"Skipping inaccessible file {file_path}: {e}")
                        continue

        return sorted(manifest, key=lambda x: x.path)

    def cleanup(self) -> None:
        """Deletes the temporary directory created for the clone."""
        if self.last_clone_path and self.last_clone_path.exists():
            try:
                shutil.rmtree(self.last_clone_path)
                logger.info(f"Cleaned up temporary directory {self.last_clone_path}")
            except Exception as e:
                logger.critical(
                    f"Failed to delete temporary directory {self.last_clone_path}: {e}"
                )
            finally:
                self.last_clone_path = None
