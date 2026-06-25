"""Git repository cloner and file discovery service.

Clones a Git repository to a temp directory and discovers all parseable
source files, returning a manifest of FileInfo(file_path, language, size_bytes).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.reporag.config import settings

logger = logging.getLogger(__name__)


class CloneError(Exception):
    """Raised when a repository cloning or discovery operation fails."""

    pass


@dataclass(frozen=True)
class FileInfo:
    """Deterministic structural metadata for an ingested source file."""

    path: str
    language: str
    size_bytes: int


class RepoCloner:
    """An asynchronous repository cloner and file discovery service.

    Fully integrated with the application configuration settings.
    """

    def __init__(self) -> None:
        """Initialize the repository cloner."""
        self.max_repo_size_bytes = settings.max_repo_size_mb * 1024 * 1024
        self.default_depth = settings.clone_depth
        self.extension_map = settings.extension_map
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

    async def clone_and_discover(
        self,
        repo_url: str,
        branch: str | None = None,
        depth: int | None = None,
    ) -> list[FileInfo]:
        """Clones a repository, validates size constraints, and returns a manifest.

        Args:
            repo_url: Remote HTTPS URL or local repository path.
            branch: Optional branch name. If not provided, clones the default branch.
            depth: Optional clone depth. Defaults to settings.clone_depth.

        Returns:
            A list of FileInfo objects representing discovered source files.

        Raises:
            CloneError: If cloning or discovery fails.
        """
        target_depth = depth if depth is not None else self.default_depth

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
            await self._execute_async_clone(repo_url, target_dir, branch, target_depth)
            manifest = self._discover_files(target_dir)
            return manifest
        except Exception as e:
            logger.error(f"Ingestion pipeline failed for repository {repo_url}: {e}")
            self.cleanup()
            if not isinstance(e, CloneError):
                raise CloneError(f"Internal ingestion failure: {e}") from e
            raise

    async def _execute_async_clone(
        self,
        repo_url: str,
        target_dir: Path,
        branch: str | None,
        depth: int,
    ) -> None:
        """Executes the git clone subprocess command asynchronously."""
        cmd = ["git", "clone"]
        if depth > 0:
            cmd.extend(["--depth", str(depth)])
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([repo_url, str(target_dir)])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
        except Exception as e:
            raise CloneError(f"Failed to initiate git subprocess: {e}") from e

        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown git clone error"
            raise CloneError(
                f"Git clone failed with code {process.returncode}: {error_msg}"
            )

    def _discover_files(self, base_path: Path) -> list[FileInfo]:
        """Discovers files in the cloned directory and filters by extension.

        Args:
            base_path: Path to the cloned repository root.

        Returns:
            A sorted list of FileInfo objects.
        """
        manifest: list[FileInfo] = []
        total_size = 0

        # Reload extension_map dynamically from settings if needed
        ext_map = self.extension_map or settings.extension_map

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
                            FileInfo(
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
