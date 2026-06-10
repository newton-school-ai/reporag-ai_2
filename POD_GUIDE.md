# POD_GUIDE.md - RepoRAG AI

## Pod Members

| # | Role | GitHub Role | Primary Ownership |
|---|------|------------|-------------------|
| 1 | Faculty | Admin | Merges to main, milestone sign-off, defense Q&A |
| 2 | Maintainer | Maintain | Merges to dev, code review, CI health, release |
| 3 | Contributor 1 | Write | Ingestion engine: cloner, parser, symbol extractor, chunker |
| 4 | Contributor 2 | Write | Graph + embedding: call graph, dep graph, Neo4j, embedders, index |
| 5 | Contributor 3 | Write | Retrieval + agent: search, fusion, reranker, planner, router |
| 6 | Contributor 4 | Write | Generation + API + frontend: generator, citations, FastAPI, React UI |

Note: Ownership defines primary responsibility. Every contributor must understand
every milestone (defense questions apply to all).

## Responsibilities

**Faculty**:
- Reviews and merges dev -> main at milestone boundaries
- Conducts Q&A sessions using defense questions from MILESTONES.md
- Resolves architectural disputes
- Signs off on milestone acceptance criteria

**Maintainer**:
- Reviews and merges feature -> dev PRs
- Keeps CI green; fixes broken builds
- Manages GitHub labels, milestones, project board
- Runs `scripts/create_github_issues.sh` during setup

**Contributors**:
- Pick up issues from their ownership area (or any available issue)
- Write code, tests, and documentation
- Open PRs with the standard template
- Review at least 1 PR per sprint from another contributor
- Prepare for defense questions on all milestones

## Sprint Timeline

| Week | Milestone | Focus |
|------|-----------|-------|
| 1 | M1 | Scaffold, CI, Docker, config |
| 2 | M2 | Ingestion + AST parsing |
| 3 | M3 | Knowledge graph construction |
| 4 | M4 | Embedding pipeline + indexing |
| 5 | M5 | Hybrid retrieval engine |
| 6 | M6 | Agentic query planner |
| 7 | M7 | Generation, citations, API |
| 8 | M8 | Auth + middleware |
| 9 | M9 | React frontend |
| 10 | M10 | Eval, testing, demo |

Each week ends with:
- Milestone acceptance criteria verified
- Defense Q&A (faculty asks, any contributor answers)
- Sprint retro (15 min)

## Collaboration Model

**Daily standup** (async in GitHub Discussions or Slack):
- What I did yesterday
- What I'm doing today
- Blockers

**PR review turnaround**: 24 hours max.

**Conflict resolution**: Maintainer decides technical disputes. Faculty decides
architectural disputes. If maintainer and faculty disagree, faculty wins.

**Cross-review requirement**: Every contributor must review at least 1 PR per sprint
outside their primary ownership area. This ensures everyone understands the full system.

## PR Review Checklist

When reviewing a PR, check:

- [ ] Branch named `feature/issue-N-short-name`
- [ ] Commits follow `[Issue N] description` format
- [ ] All acceptance criteria from the linked issue are met
- [ ] Unit tests added for new logic
- [ ] No hardcoded secrets, API keys, or internal data
- [ ] No non-ASCII characters (pre-commit should catch this)
- [ ] Code is readable; complex logic has comments
- [ ] Error cases handled (no bare except, no silent failures)
- [ ] Type hints on all function signatures
- [ ] Docstrings on public functions/classes

## Q&A Session Format

Each milestone ends with a Q&A session. Faculty picks 2-3 defense questions from
MILESTONES.md and directs them at random contributors (not necessarily the one who
wrote the code).

Format:
1. Faculty asks question
2. Contributor answers (2-3 minutes)
3. Faculty follow-up if needed
4. Next question

Every contributor should be able to answer every defense question for every completed
milestone. This is the core learning model.

---

NST Engineering | RepoRAG AI Pod Guide | 2026
