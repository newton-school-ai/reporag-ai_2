"""Git repository cloner and file discovery service.

Clones a Git repository to a temp directory and discovers all parseable
source files, returning a manifest of (file_path, language, size_bytes).
"""

# TODO: Implement in Issue 5
# - Clone public repos via HTTPS (gitpython)
# - Support branch selection (default: main/master auto-detect)
# - Shallow clone option (depth=1) for performance
# - Walk file tree, filter by language extensions
# - Return manifest: list of FileInfo(path, language, size)
# - Clean up temp directory on error


import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from git import Repo


@dataclass
class FileInfo:
    path: str
    language: str
    size_bytes: int


LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
}


class RepoCloner:
    """Clone repositories and discover source files."""

    def __init__(self) -> None:
        self.extensions = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
        }

    def discover_files(self, repo_path: str) -> list[FileInfo]:
        """Discover supported source files in a repository."""

        manifest: list[FileInfo] = []

        for file_path in Path(repo_path).rglob("*"):
            if not file_path.is_file():
                continue

            language = self.extensions.get(file_path.suffix)

            if language is None:
                continue

            manifest.append(
                FileInfo(
                    path=str(file_path),
                    language=language,
                    size_bytes=file_path.stat().st_size,
                )
            )

        return manifest

    def clone_repository(
        self,
        repo_source: str,
        branch: str | None = None,
        depth: int = 1,
    ) -> str:
        """Clone a repository and return local path."""

        temp_dir = tempfile.mkdtemp(prefix="reporag_")

        try:
            if Path(repo_source).exists():
                shutil.copytree(
                    repo_source,
                    temp_dir,
                    dirs_exist_ok=True,
                )
                return temp_dir

            clone_args = {
                "to_path": temp_dir,
                "depth": depth,
            }

            if branch:
                clone_args["branch"] = branch
                clone_args["single_branch"] = True

            Repo.clone_from(repo_source, **clone_args)

            return temp_dir

        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def clone_and_discover(
        self,
        repo_source: str,
        branch: str | None = None,
        depth: int = 1,
    ) -> list[FileInfo]:
        """Clone repository and discover source files."""

        raise NotImplementedError
