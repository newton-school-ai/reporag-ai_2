# GITHUB_ISSUES.md - RepoRAG AI

Source file for `scripts/create_github_issues.sh`. Each issue below is created
via `gh issue create` with the exact body shown.

Total: 37 issues across 10 milestones (M1-M10).

See ISSUES_TRACKER.md for the full narrative version with context.
See MILESTONES.md for milestone details, acceptance criteria, and defense questions.

---

## M1: Repo Scaffold, CI & Docker Setup (Issues 1-4)

### Issue 1 - Initialize repo scaffold, CI workflow, Docker setup
**Milestone**: M1: Repo Scaffold, CI & Docker Setup
**Labels**: m1, infra
**Body**: See ISSUES_TRACKER.md Issue 1

### Issue 2 - Pre-commit hooks: ASCII guard, internal-data guard, ruff, black
**Milestone**: M1: Repo Scaffold, CI & Docker Setup
**Labels**: m1, infra
**Body**: See ISSUES_TRACKER.md Issue 2

### Issue 3 - Database setup: SQLite default, Postgres-ready, Alembic migrations
**Milestone**: M1: Repo Scaffold, CI & Docker Setup
**Labels**: m1, infra, backend
**Body**: See ISSUES_TRACKER.md Issue 3

### Issue 4 - Project configuration module: .env loading, settings validation
**Milestone**: M1: Repo Scaffold, CI & Docker Setup
**Labels**: m1, infra
**Body**: See ISSUES_TRACKER.md Issue 4

---

## M2: Repository Ingestion & AST Parsing Engine (Issues 5-8)

### Issue 5 - Git repository cloner and file discovery service
**Milestone**: M2: Repository Ingestion & AST Parsing
**Labels**: m2, engine, data
**Body**: See ISSUES_TRACKER.md Issue 5

### Issue 6 - Tree-sitter AST parser for Python
**Milestone**: M2: Repository Ingestion & AST Parsing
**Labels**: m2, engine
**Body**: See ISSUES_TRACKER.md Issue 6

### Issue 7 - Symbol extractor: functions, classes, methods, imports
**Milestone**: M2: Repository Ingestion & AST Parsing
**Labels**: m2, engine
**Body**: See ISSUES_TRACKER.md Issue 7

### Issue 8 - Semantic code chunker: AST-aware, respects scope boundaries
**Milestone**: M2: Repository Ingestion & AST Parsing
**Labels**: m2, engine
**Body**: See ISSUES_TRACKER.md Issue 8

---

## M3: Code Knowledge Graph Construction (Issues 9-12)

### Issue 9 - Call graph builder from AST
**Milestone**: M3: Code Knowledge Graph
**Labels**: m3, engine
**Body**: See ISSUES_TRACKER.md Issue 9

### Issue 10 - Import dependency graph builder
**Milestone**: M3: Code Knowledge Graph
**Labels**: m3, engine
**Body**: See ISSUES_TRACKER.md Issue 10

### Issue 11 - Symbol table / global registry with metadata
**Milestone**: M3: Code Knowledge Graph
**Labels**: m3, engine
**Body**: See ISSUES_TRACKER.md Issue 11

### Issue 12 - Neo4j graph store with Cypher query layer
**Milestone**: M3: Code Knowledge Graph
**Labels**: m3, engine, infra
**Body**: See ISSUES_TRACKER.md Issue 12

---

## M4: Code Embedding Pipeline & Hybrid Index (Issues 13-15)

### Issue 13 - Code embedding pipeline: CodeBERT/UniXcoder
**Milestone**: M4: Embedding Pipeline & Hybrid Index
**Labels**: m4, engine
**Body**: See ISSUES_TRACKER.md Issue 13

### Issue 14 - Docstring and comment embedding pipeline
**Milestone**: M4: Embedding Pipeline & Hybrid Index
**Labels**: m4, engine
**Body**: See ISSUES_TRACKER.md Issue 14

### Issue 15 - Hybrid index builder: Qdrant vector + BM25 sparse
**Milestone**: M4: Embedding Pipeline & Hybrid Index
**Labels**: m4, engine, infra
**Body**: See ISSUES_TRACKER.md Issue 15

---

## M5: Hybrid Retrieval Engine (Issues 16-19)

### Issue 16 - Vector semantic search with configurable top-k
**Milestone**: M5: Hybrid Retrieval Engine
**Labels**: m5, engine
**Body**: See ISSUES_TRACKER.md Issue 16

### Issue 17 - BM25 sparse keyword search for code identifiers
**Milestone**: M5: Hybrid Retrieval Engine
**Labels**: m5, engine
**Body**: See ISSUES_TRACKER.md Issue 17

### Issue 18 - Graph-based retrieval: neighbor traversal, path queries
**Milestone**: M5: Hybrid Retrieval Engine
**Labels**: m5, engine
**Body**: See ISSUES_TRACKER.md Issue 18

### Issue 19 - Reciprocal Rank Fusion + cross-encoder reranker
**Milestone**: M5: Hybrid Retrieval Engine
**Labels**: m5, engine
**Body**: See ISSUES_TRACKER.md Issue 19

---

## M6: Agentic Query Planner & Multi-Hop Decomposition (Issues 20-22)

### Issue 20 - Query classifier: simple lookup vs. multi-hop vs. exploratory
**Milestone**: M6: Agentic Query Planner
**Labels**: m6, agent
**Body**: See ISSUES_TRACKER.md Issue 20

### Issue 21 - Agentic query decomposer: break complex queries into sub-queries
**Milestone**: M6: Agentic Query Planner
**Labels**: m6, agent
**Body**: See ISSUES_TRACKER.md Issue 21

### Issue 22 - Strategy router + sub-query executor
**Milestone**: M6: Agentic Query Planner
**Labels**: m6, agent
**Body**: See ISSUES_TRACKER.md Issue 22

---

## M7: Answer Generation, Citation & FastAPI Serving (Issues 23-26)

### Issue 23 - Context assembler: retrieved code -> structured prompt context
**Milestone**: M7: Generation, Citation & API
**Labels**: m7, engine
**Body**: See ISSUES_TRACKER.md Issue 23

### Issue 24 - Prompt builder with code-aware templates
**Milestone**: M7: Generation, Citation & API
**Labels**: m7, engine
**Body**: See ISSUES_TRACKER.md Issue 24

### Issue 25 - LLM generation with line-level citation extraction
**Milestone**: M7: Generation, Citation & API
**Labels**: m7, engine
**Body**: See ISSUES_TRACKER.md Issue 25

### Issue 26 - FastAPI application: repo, query, and health endpoints
**Milestone**: M7: Generation, Citation & API
**Labels**: m7, backend, infra
**Body**: See ISSUES_TRACKER.md Issue 26

---

## M8: Google OAuth, JWT Auth & API Middleware (Issues 27-29)

### Issue 27 - Google OAuth 2.0 login flow
**Milestone**: M8: Auth & API Middleware
**Labels**: m8, backend
**Body**: See ISSUES_TRACKER.md Issue 27

### Issue 28 - JWT token issuance, validation, refresh middleware
**Milestone**: M8: Auth & API Middleware
**Labels**: m8, backend
**Body**: See ISSUES_TRACKER.md Issue 28

### Issue 29 - Rate limiting, request logging, error handling middleware
**Milestone**: M8: Auth & API Middleware
**Labels**: m8, backend, infra
**Body**: See ISSUES_TRACKER.md Issue 29

---

## M9: React Frontend -- Code Explorer & Q&A Interface (Issues 30-33)

### Issue 30 - React + Vite scaffold, routing, auth flow (Google login)
**Milestone**: M9: React Frontend
**Labels**: m9, frontend
**Body**: See ISSUES_TRACKER.md Issue 30

### Issue 31 - Repository explorer: file tree + syntax-highlighted code viewer
**Milestone**: M9: React Frontend
**Labels**: m9, frontend
**Body**: See ISSUES_TRACKER.md Issue 31

### Issue 32 - Conversational Q&A interface with citation highlights
**Milestone**: M9: React Frontend
**Labels**: m9, frontend
**Body**: See ISSUES_TRACKER.md Issue 32

### Issue 33 - Interactive graph visualizer: call graph / dependency graph
**Milestone**: M9: React Frontend
**Labels**: m9, frontend
**Body**: See ISSUES_TRACKER.md Issue 33

---

## M10: Evaluation Harness, E2E Testing & Demo (Issues 34-37)

### Issue 34 - Retrieval evaluation suite: context recall, precision, MRR
**Milestone**: M10: Eval, Testing & Demo
**Labels**: m10, eval
**Body**: See ISSUES_TRACKER.md Issue 34

### Issue 35 - Generation evaluation: faithfulness, relevance, hallucination
**Milestone**: M10: Eval, Testing & Demo
**Labels**: m10, eval
**Body**: See ISSUES_TRACKER.md Issue 35

### Issue 36 - End-to-end integration tests with sample repositories
**Milestone**: M10: Eval, Testing & Demo
**Labels**: m10, eval, test
**Body**: See ISSUES_TRACKER.md Issue 36

### Issue 37 - Performance benchmarks, load testing, demo script
**Milestone**: M10: Eval, Testing & Demo
**Labels**: m10, eval, test
**Body**: See ISSUES_TRACKER.md Issue 37

---

NST Engineering | RepoRAG AI GitHub Issues | 2026
