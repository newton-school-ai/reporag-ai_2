## Summary

Closes #<issue-number>

## What changed

- bullet
- bullet

## How to test

```bash
<runnable commands>
```

## Checklist

- [ ] Branch follows `feature/issue-N-short-name` naming
- [ ] Commits follow `[Issue N] description` format
- [ ] All acceptance criteria from the linked issue are met
- [ ] Unit tests added/updated and passing (`pytest tests/unit/ -v`)
- [ ] `pre-commit run --all-files` passes
- [ ] No non-ASCII characters introduced
- [ ] No files under `_internal/` modified (except PROJECT_CONTEXT.md)
- [ ] No hardcoded API keys, secrets, or internal data
- [ ] Type hints on all new function signatures
- [ ] Docstrings on all new public functions/classes
