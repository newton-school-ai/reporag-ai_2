#!/usr/bin/env bash
# Internal Data Guard - Blocks commits that include files under _internal/
# except for PROJECT_CONTEXT.md.
# Used by pre-commit hook (id: internal-data-guard).
#
# The _internal/ directory contains NST internal planning documents that
# must not be committed to the repository (except PROJECT_CONTEXT.md which
# is the approved public-facing project context file).

set -euo pipefail

FAILED=0

for file in "$@"; do
    case "$file" in
        _internal/PROJECT_CONTEXT.md)
            # Allowed -- this is the only _internal/ file that can be committed
            ;;
        _internal/*)
            echo "BLOCKED: $file"
            echo "  Files under _internal/ cannot be committed (except PROJECT_CONTEXT.md)."
            FAILED=1
            ;;
    esac
done

if [ "$FAILED" -ne 0 ]; then
    echo ""
    echo "Fix: Remove _internal/ files from staging with: git reset HEAD _internal/"
    exit 1
fi
