# CONTRIBUTING.md - RepoRAG AI

## Branch Strategy

```
main  (faculty-only merge, protected: 1 approval, no force-push)
  |
  +-- dev  (maintainer + faculty merge, protected: 2 approvals, no force-push)
        |
        +-- feature/issue-N-short-name  (contributor branches)
```

All work happens on feature branches created from `dev`. Feature branches are
merged into `dev` via Pull Request. `dev` is merged into `main` by faculty only,
at milestone boundaries.

## Commit Format

```
[Issue N] Short imperative description

Optional longer body explaining what and why (not how).
```

Examples:
- `[Issue 6] Add tree-sitter AST parser for Python`
- `[Issue 19] Implement Reciprocal Rank Fusion for retrieval`

## PR Workflow

1. Create branch from `dev`: `git checkout dev && git pull && git checkout -b feature/issue-N-short-name`
2. Write code, commit with format above
3. Push: `git push -u origin feature/issue-N-short-name`
4. Open PR to `dev` using the PR template
5. Request 2 reviewers (at least 1 must be maintainer or faculty)
6. Address review feedback, push fixes
7. Maintainer or faculty merges (squash merge preferred)

## PR Checklist (from template)

- [ ] Branch follows naming convention `feature/issue-N-short-name`
- [ ] Commits follow format `[Issue N] description`
- [ ] All acceptance criteria from the issue are met
- [ ] Unit tests added/updated and passing
- [ ] `pre-commit run --all-files` passes
- [ ] No non-ASCII characters introduced
- [ ] No files under `_internal/` modified (except PROJECT_CONTEXT.md)
- [ ] No hardcoded API keys, secrets, or internal data

## Coding Standards

### Python

- Formatter: black (line length 88)
- Linter: ruff (default ruleset)
- Type hints: required on all function signatures
- Docstrings: Google style on all public functions/classes
- Imports: sorted by ruff (isort-compatible)

### JavaScript / React

- Formatter: Prettier (default config)
- Linter: ESLint with React plugin
- Components: functional components with hooks only (no class components)
- Styling: TailwindCSS utility classes

### General

- ASCII only: no em dashes, smart quotes, arrows, box-drawing characters
- No `print()` in production code; use `logging` module
- No hardcoded secrets; use `settings` from config.py
- Tests next to code or in tests/ directory

## Pod Roles

| Role | Can merge to dev | Can merge to main | Reviews PRs |
|------|:---:|:---:|:---:|
| Faculty (Admin) | Yes | Yes | Yes |
| Maintainer (Maintain) | Yes | No | Yes (required) |
| Contributor (Write) | No | No | Yes (optional) |

## Code Review Guidelines

Reviewers check:
1. Does the PR satisfy the issue's acceptance criteria?
2. Are there unit tests for new logic?
3. Is the code readable without excessive comments?
4. Are error cases handled?
5. No security issues (hardcoded secrets, SQL injection, unvalidated input)?

---

NST Engineering | RepoRAG AI Contributing Guide | 2026
