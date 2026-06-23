#!/usr/bin/env bash
# ASCII Guard - Blocks non-ASCII characters in source files.
# Used by pre-commit hook (id: ascii-only).
#
# Scans staged files for characters outside the ASCII range (0x00-0x7F).
# Catches: em dashes, smart quotes, curly apostrophes, non-breaking spaces,
# Unicode arrows, box-drawing characters, etc.
#
# Compatible with both macOS (BSD grep) and Linux (GNU grep).

set -euo pipefail

FAILED=0

for file in "$@"; do
    # Skip binary files and files that don't exist
    [ -f "$file" ] || continue

    # Only check relevant text file extensions
    case "$file" in
        *.py|*.md|*.sh|*.yml|*.yaml|*.toml|*.json|*.jsx|*.js|*.ts|*.tsx|*.txt|*.cfg|*.ini|*.rst)
            ;;
        *)
            continue
            ;;
    esac

    # Check for non-ASCII characters.
    # Use LC_ALL=C so that grep treats bytes literally, and match any byte
    # outside the printable ASCII + common control character range.
    # This works on both macOS (BSD) and Linux (GNU) grep.
    if LC_ALL=C grep -n '[^	 -~]' "$file" 2>/dev/null; then
        echo "ERROR: Non-ASCII characters found in $file (see above)"
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo ""
    echo "Fix: Replace non-ASCII characters with ASCII equivalents."
    echo "  - Smart quotes -> straight quotes"
    echo "  - Em dash -> --"
    echo "  - Non-breaking space -> regular space"
    exit 1
fi
