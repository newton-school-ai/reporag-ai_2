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


from dataclasses import dataclass


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
