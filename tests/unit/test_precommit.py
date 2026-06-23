"""Unit tests for pre-commit hooks (Issue 2).

Tests verify that the pre-commit configuration is valid and that
the ASCII guard and internal-data guard scripts work correctly.
"""

import os
import subprocess
from pathlib import Path

import yaml

# Root of the repository
REPO_ROOT = Path(__file__).resolve().parents[2]
ASCII_GUARD = REPO_ROOT / "scripts" / "ascii_guard.sh"
INTERNAL_GUARD = REPO_ROOT / "scripts" / "internal_data_guard.sh"
# Use a temp directory inside the workspace (avoids macOS sandbox issues)
SCRATCH_DIR = REPO_ROOT / ".test_scratch"


def setup_module() -> None:
    """Create a scratch directory for temp files."""
    SCRATCH_DIR.mkdir(exist_ok=True)


def teardown_module() -> None:
    """Remove the scratch directory after tests."""
    import shutil

    if SCRATCH_DIR.exists():
        shutil.rmtree(SCRATCH_DIR)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestPreCommitConfig:
    """Tests for .pre-commit-config.yaml validity."""

    def test_config_file_exists(self) -> None:
        """Pre-commit config file must exist at repo root."""
        config = REPO_ROOT / ".pre-commit-config.yaml"
        assert config.exists(), ".pre-commit-config.yaml not found"

    def test_config_is_valid_yaml(self) -> None:
        """Config must parse as valid YAML without errors."""
        config_path = REPO_ROOT / ".pre-commit-config.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert data is not None, "YAML parsed to None"
        assert "repos" in data, "Missing top-level 'repos' key"

    def test_config_has_required_hooks(self) -> None:
        """Config must include all required hook IDs."""
        config_path = REPO_ROOT / ".pre-commit-config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        hook_ids: list[str] = []
        for repo in config["repos"]:
            for hook in repo["hooks"]:
                hook_ids.append(hook["id"])

        required = [
            "trailing-whitespace",
            "end-of-file-fixer",
            "ruff",
            "black",
            "ascii-only",
            "internal-data-guard",
        ]
        for hook_id in required:
            assert hook_id in hook_ids, f"Missing required hook: {hook_id}"

    def test_pre_commit_validate_config(self) -> None:
        """pre-commit validate-config must pass."""
        result = subprocess.run(
            [
                "pre-commit",
                "validate-config",
                str(REPO_ROOT / ".pre-commit-config.yaml"),
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert (
            result.returncode == 0
        ), f"pre-commit validate-config failed: {result.stderr}"


# ---------------------------------------------------------------------------
# ASCII guard
# ---------------------------------------------------------------------------


class TestASCIIGuard:
    """Tests for the ASCII-only guard script."""

    def test_guard_script_exists(self) -> None:
        """ASCII guard script must exist and be executable."""
        assert ASCII_GUARD.exists(), "scripts/ascii_guard.sh not found"
        assert os.access(ASCII_GUARD, os.X_OK), "ascii_guard.sh is not executable"

    def test_clean_ascii_file_passes(self) -> None:
        """A file with only ASCII characters should pass."""
        tmp = SCRATCH_DIR / "clean.py"
        tmp.write_text('def hello():\n    return "world"\n')
        result = subprocess.run(
            ["bash", str(ASCII_GUARD), str(tmp)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Clean ASCII file should pass: {result.stdout}"

    def test_non_ascii_smart_quote_fails(self) -> None:
        """A file with smart quotes (Unicode) should fail."""
        tmp = SCRATCH_DIR / "smartquote.py"
        # Write smart quote (U+201C left double quotation mark)
        tmp.write_bytes(b"x = \xe2\x80\x9chello\xe2\x80\x9d\n")
        result = subprocess.run(
            ["bash", str(ASCII_GUARD), str(tmp)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "Smart quotes should be caught by ASCII guard"

    def test_non_ascii_em_dash_fails(self) -> None:
        """A file with an em dash should fail."""
        tmp = SCRATCH_DIR / "emdash.md"
        # Write em dash (U+2014)
        tmp.write_bytes(b"some text \xe2\x80\x94 more text\n")
        result = subprocess.run(
            ["bash", str(ASCII_GUARD), str(tmp)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "Em dashes should be caught by ASCII guard"

    def test_non_checked_extension_passes(self) -> None:
        """Files with non-checked extensions should pass even with non-ASCII."""
        tmp = SCRATCH_DIR / "binary.png"
        tmp.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG-like binary header
        result = subprocess.run(
            ["bash", str(ASCII_GUARD), str(tmp)],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
        ), "Non-text extensions should be skipped by ASCII guard"


# ---------------------------------------------------------------------------
# Internal data guard
# ---------------------------------------------------------------------------


class TestInternalDataGuard:
    """Tests for the internal-data guard script."""

    def test_guard_script_exists(self) -> None:
        """Internal data guard script must exist and be executable."""
        assert INTERNAL_GUARD.exists(), "scripts/internal_data_guard.sh not found"
        assert os.access(
            INTERNAL_GUARD, os.X_OK
        ), "internal_data_guard.sh is not executable"

    def test_project_context_allowed(self) -> None:
        """_internal/PROJECT_CONTEXT.md should be allowed."""
        result = subprocess.run(
            ["bash", str(INTERNAL_GUARD), "_internal/PROJECT_CONTEXT.md"],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
        ), "PROJECT_CONTEXT.md should be allowed through guard"

    def test_other_internal_file_blocked(self) -> None:
        """Any other file under _internal/ should be blocked."""
        result = subprocess.run(
            ["bash", str(INTERNAL_GUARD), "_internal/secret_notes.md"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "_internal/secret_notes.md should be BLOCKED"
        assert "BLOCKED" in result.stdout

    def test_internal_subdirectory_blocked(self) -> None:
        """Files in subdirectories of _internal/ should be blocked."""
        result = subprocess.run(
            ["bash", str(INTERNAL_GUARD), "_internal/data/credentials.json"],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode != 0
        ), "_internal/data/credentials.json should be BLOCKED"

    def test_regular_files_pass(self) -> None:
        """Files outside _internal/ should pass without issue."""
        result = subprocess.run(
            [
                "bash",
                str(INTERNAL_GUARD),
                "src/reporag/config.py",
                "tests/unit/test_precommit.py",
                "README.md",
            ],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
        ), "Regular source files should pass the internal data guard"

    def test_mixed_files_blocks_internal(self) -> None:
        """If a mix of regular and _internal/ files, the guard should block."""
        result = subprocess.run(
            [
                "bash",
                str(INTERNAL_GUARD),
                "src/reporag/config.py",
                "_internal/secrets.env",
            ],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode != 0
        ), "Guard should block when any _internal/ file is in the list"


# ---------------------------------------------------------------------------
# Ruff & Black on clean scaffold
# ---------------------------------------------------------------------------


class TestRuffBlackCleanScaffold:
    """Tests that ruff and black pass on the current scaffold."""

    def test_ruff_passes(self) -> None:
        """ruff check should pass on src/ and tests/ directories."""
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "src", "tests", "alembic"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"ruff check failed:\n{result.stdout}"

    def test_black_passes(self) -> None:
        """black --check should pass on src/ and tests/ directories."""
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "black", "--check", "src", "tests", "alembic"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"black --check failed:\n{result.stderr}"
