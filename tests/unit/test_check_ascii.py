"""Unit tests for the ASCII-only pre-commit hook script."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_ascii.py"

# Non-ASCII test characters built via escapes so this source file
# itself stays pure ASCII (it gets scanned by the same hook).
E_ACUTE = "\u00e9"  # e with acute accent
I_DIAERESIS = "\u00ef"  # i with diaeresis


def run_check(*file_paths):
    """Run check_ascii.py against the given files and return the result."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, file_paths)],
        capture_output=True,
        text=True,
    )


def test_ascii_only_file_passes(tmp_path):
    f = tmp_path / "clean.py"
    f.write_text("print('hello world')\n", encoding="utf-8")

    result = run_check(f)

    assert result.returncode == 0
    assert result.stdout == ""


def test_non_ascii_file_fails(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text(f"# caf{E_ACUTE}\nprint('na{I_DIAERESIS}ve')\n", encoding="utf-8")

    result = run_check(f)

    assert result.returncode == 1
    assert "Non-ASCII characters found" in result.stdout
    assert "bad.py" in result.stdout


def test_multiple_files_one_bad(tmp_path):
    good = tmp_path / "good.py"
    bad = tmp_path / "bad.py"
    good.write_text("x = 1\n", encoding="utf-8")
    bad.write_text(f"y = 'r{E_ACUTE}sum{E_ACUTE}'\n", encoding="utf-8")

    result = run_check(good, bad)

    assert result.returncode == 1
    assert "good.py" not in result.stdout
    assert "bad.py" in result.stdout


def test_empty_file_passes(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("", encoding="utf-8")

    result = run_check(f)

    assert result.returncode == 0


def test_reports_correct_line_number(tmp_path):
    f = tmp_path / "multiline.py"
    f.write_text(f"line one\nline two caf{E_ACUTE}\nline three\n", encoding="utf-8")

    result = run_check(f)

    assert result.returncode == 1
    assert "multiline.py:2" in result.stdout
