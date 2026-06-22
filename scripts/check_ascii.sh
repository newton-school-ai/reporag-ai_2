#!/usr/bin/env bash
# ascii-guard: reject any staged file that contains non-ASCII bytes.
# Pre-commit calls this with the staged filenames as positional arguments.
set -euo pipefail

failed=0
for filepath in "$@"; do
    [ -f "$filepath" ] || continue
    if grep -Pn '[^\x00-\x7F]' "$filepath" 2>/dev/null; then
        echo "  ^ Non-ASCII bytes in: $filepath"
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    echo ""
    echo "ERROR [ascii-guard]: non-ASCII characters found."
    echo "Only plain ASCII (0x00-0x7F) is permitted in source files."
    echo "Common offenders: em dashes, smart quotes, non-breaking spaces, arrows."
    exit 1
fi
