#!/usr/bin/env bash
# internal-data-guard: block commits of any file under _internal/
# except for _internal/PROJECT_CONTEXT.md which is explicitly allowed.
# Pre-commit calls this with staged filenames as positional arguments.
set -euo pipefail

blocked=0
for filepath in "$@"; do
    case "$filepath" in
        _internal/PROJECT_CONTEXT.md)
            # Allowed: this is the only whitelisted _internal file
            ;;
        _internal/*)
            echo "BLOCKED [internal-data-guard]: $filepath"
            echo "  Files under _internal/ must not be committed (internal NST data)."
            echo "  Only _internal/PROJECT_CONTEXT.md is permitted."
            blocked=1
            ;;
    esac
done

if [ "$blocked" -ne 0 ]; then
    exit 1
fi
