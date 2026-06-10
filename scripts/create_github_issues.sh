#!/usr/bin/env bash
# RepoRAG AI - Create all 37 GitHub issues
# Pre-req: labels + milestones already exist (see README setup commands)
# Usage:   bash scripts/create_github_issues.sh
set -euo pipefail
REPO="${REPO:-newton-school-ai/reporag-ai}"

create_issue() {
  local num=$1; local title=$2; local milestone=$3; local labels=$4; local body=$5
  echo "Creating: Issue $num - $title"
  gh issue create --repo "$REPO" \
    --title "Issue $num - $title" \
    --milestone "$milestone" \
    --label "$labels" \
    --body "$body"
  sleep 1
}

# =============================================================================
# M1: Repo Scaffold, CI & Docker Setup
# =============================================================================

create_issue 1 "Initialize repo scaffold, CI workflow, Docker setup" \
  "M1: Repo Scaffold, CI & Docker Setup" "m1,infra" \
  "## Why
Every NST project starts with a reproducible scaffold: CI that catches problems before review, Docker that gives every pod member the same environment, and a directory layout that matches the architecture.

## What to build
- GitHub Actions CI workflow: ASCII guard, ruff lint, black format check, pytest
- Dockerfile for the API server (Python 3.11-slim base)
- docker-compose.yml with services: api, neo4j, qdrant, postgres
- Full directory tree with __init__.py files in every package

## Files to create
- .github/workflows/ci.yml
- Dockerfile
- docker-compose.yml
- All src/reporag/**/__init__.py files

## How to test locally
\`\`\`bash
docker-compose up -d
docker-compose ps
python -c \"import src.reporag; print('import OK')\"
\`\`\`

## Acceptance Criteria
- [ ] CI workflow triggers on PR to dev and push to dev
- [ ] ASCII guard step fails on non-ASCII characters
- [ ] docker-compose up -d starts all 3+ services
- [ ] All __init__.py files exist in every src package

## Branch: feature/issue-1-scaffold
## Dependencies: None"

create_issue 2 "Pre-commit hooks: ASCII guard, internal-data guard, ruff, black" \
  "M1: Repo Scaffold, CI & Docker Setup" "m1,infra" \
  "## Why
Pre-commit catches formatting and policy violations before they hit CI, saving review cycles.

## What to build
- .pre-commit-config.yaml with: trailing-whitespace, end-of-file-fixer, ruff, black, ASCII-only check, internal-data path guard

## Files to create
- .pre-commit-config.yaml

## How to test locally
\`\`\`bash
pre-commit install
pre-commit run --all-files
\`\`\`

## Acceptance Criteria
- [ ] pre-commit install succeeds
- [ ] Non-ASCII characters are caught and blocked
- [ ] Files in _internal/ (except PROJECT_CONTEXT.md) are blocked
- [ ] ruff and black run without errors on clean scaffold

## Branch: feature/issue-2-precommit
## Dependencies: Issue 1"

create_issue 3 "Database setup: SQLite default, Postgres-ready, Alembic migrations" \
  "M1: Repo Scaffold, CI & Docker Setup" "m1,infra,backend" \
  "## Why
Persistent storage for repo metadata, ingestion status, and user data. SQLite for dev, Postgres for production.

## What to build
- SQLAlchemy models: Repository, IngestionJob, User, QueryLog
- Alembic migration setup with initial schema
- Database session factory switching SQLite/Postgres via env var

## Files to create
- src/reporag/db/__init__.py
- src/reporag/db/models.py
- src/reporag/db/session.py
- alembic.ini, alembic/env.py, alembic/versions/

## How to test locally
\`\`\`bash
alembic upgrade head
python -c \"from src.reporag.db.session import get_db; print('DB OK')\"
\`\`\`

## Acceptance Criteria
- [ ] alembic upgrade head creates all tables in SQLite
- [ ] DATABASE_URL switch to Postgres works without code changes
- [ ] Models include: Repository, IngestionJob, User, QueryLog
- [ ] Session factory provides async-compatible sessions

## Branch: feature/issue-3-database
## Dependencies: Issue 1"

create_issue 4 "Project configuration module: .env loading, settings validation" \
  "M1: Repo Scaffold, CI & Docker Setup" "m1,infra" \
  "## Why
Centralized config prevents scattered os.getenv calls and validates required vars at startup.

## What to build
- Pydantic Settings class loading from .env
- Sections: database, neo4j, qdrant, llm, auth, app
- .env.example with all variables documented

## Files to create
- src/reporag/config.py
- .env.example

## How to test locally
\`\`\`bash
cp .env.example .env
python -c \"from src.reporag.config import settings; print(settings.model_dump())\"
\`\`\`

## Acceptance Criteria
- [ ] Missing required vars raise ValidationError at import time
- [ ] .env.example documents every variable with comments
- [ ] Sensitive fields marked as SecretStr

## Branch: feature/issue-4-config
## Dependencies: Issue 1"

# =============================================================================
# M2: Repository Ingestion & AST Parsing Engine
# =============================================================================

create_issue 5 "Git repository cloner and file discovery service" \
  "M2: Repository Ingestion & AST Parsing" "m2,engine,data" \
  "## Why
Ingestion pipeline starts with cloning a repository and discovering all parseable source files.

## What to build
- Clone a Git repo (URL or local path) to a temp directory
- Support branch selection and shallow cloning
- Walk file tree, filter by language extensions
- Return manifest: list of (file_path, language, size_bytes)

## Files to create
- src/reporag/ingestion/cloner.py

## How to test locally
\`\`\`bash
python -c \"
from src.reporag.ingestion.cloner import RepoCloner
cloner = RepoCloner()
manifest = cloner.clone_and_discover('https://github.com/pallets/click', branch='main')
print(f'Found {len(manifest)} files')
\"
\`\`\`

## Acceptance Criteria
- [ ] Clones public repos via HTTPS
- [ ] Supports branch selection
- [ ] Shallow clone option
- [ ] File discovery filters by configurable extensions
- [ ] Temp directory cleaned up on error

## Branch: feature/issue-5-cloner
## Dependencies: Issue 4"

create_issue 6 "Tree-sitter AST parser for Python" \
  "M2: Repository Ingestion & AST Parsing" "m2,engine" \
  "## Why
tree-sitter provides fast, incremental, error-tolerant parsing across many languages.

## What to build
- Parse Python source files into tree-sitter AST
- Walk AST and return structured node data
- Handle parse errors gracefully (partial ASTs)
- Language-agnostic interface

## Files to create
- src/reporag/ingestion/parser.py

## How to test locally
\`\`\`bash
python -c \"
from src.reporag.ingestion.parser import ASTParser
parser = ASTParser()
tree = parser.parse('def hello():\n    return 42\n', language='python')
print(tree.root_node.children)
\"
\`\`\`

## Acceptance Criteria
- [ ] Parses valid Python files into tree-sitter AST
- [ ] Returns partial AST for syntax errors
- [ ] Language-agnostic interface
- [ ] Node data includes type, text, start/end lines

## Branch: feature/issue-6-ast-parser
## Dependencies: Issue 5"

create_issue 7 "Symbol extractor: functions, classes, methods, imports" \
  "M2: Repository Ingestion & AST Parsing" "m2,engine" \
  "## Why
Raw ASTs are too granular. The symbol extractor produces structured inventory of meaningful code entities.

## What to build
- Extract: functions, classes, methods, module-level imports
- Metadata: name, type, file_path, start_line, end_line, signature, docstring, decorators
- Handle: nested functions, static/class methods, async, property decorators

## Files to create
- src/reporag/ingestion/symbol_extractor.py

## Acceptance Criteria
- [ ] Extracts functions with name, signature, docstring, decorators, line range
- [ ] Extracts classes with name, bases, methods, docstring
- [ ] Extracts imports: import X, from X import Y, from X import *
- [ ] Handles nested/async functions, property decorators

## Branch: feature/issue-7-symbol-extractor
## Dependencies: Issue 6"

create_issue 8 "Semantic code chunker: AST-aware, respects scope boundaries" \
  "M2: Repository Ingestion & AST Parsing" "m2,engine" \
  "## Why
Naive text chunking splits functions mid-body. AST-aware chunking respects function/class boundaries.

## What to build
- Chunk at AST-node boundaries
- Configurable max chunk size (tokens)
- Metadata: file_path, start_line, end_line, parent_symbol, language
- Overlap: include function signature in continuation chunks

## Files to create
- src/reporag/ingestion/chunker.py

## Acceptance Criteria
- [ ] Never splits function/class mid-body (unless exceeds max_tokens)
- [ ] Large functions split at logical points with signature overlap
- [ ] Chunk metadata complete
- [ ] Sizes within max_tokens +/- 10%

## Branch: feature/issue-8-chunker
## Dependencies: Issue 7"

# =============================================================================
# M3: Code Knowledge Graph Construction
# =============================================================================

create_issue 9 "Call graph builder from AST" \
  "M3: Code Knowledge Graph" "m3,engine" \
  "## Why
Call graph captures which functions call which -- essential for end-to-end flow questions.

## What to build
- Walk ASTs to identify function call expressions
- Resolve calls to target symbols (same file, cross-file)
- Build directed edges: caller -> callee with call site metadata
- Handle method calls, chained calls, constructor calls

## Files to create
- src/reporag/graph/call_graph.py

## Acceptance Criteria
- [ ] Identifies direct function calls and resolves targets
- [ ] Handles method calls and cross-file calls
- [ ] Edge metadata: caller, callee, call site file + line
- [ ] Unit tests: direct, method, cross-file, recursive calls

## Branch: feature/issue-9-call-graph
## Dependencies: Issue 7"

create_issue 10 "Import dependency graph builder" \
  "M3: Code Knowledge Graph" "m3,engine" \
  "## Why
Import graph captures module-level dependencies for architecture understanding.

## What to build
- Directed edges: importing_module -> imported_module
- Resolve relative imports to absolute paths
- Handle: import X, from X import Y, from .sibling import Z, star imports
- Detect circular imports

## Files to create
- src/reporag/graph/dependency_graph.py

## Acceptance Criteria
- [ ] Resolves absolute and relative imports
- [ ] Handles star imports with warning
- [ ] Detects circular import chains
- [ ] Edge metadata: source, target, import type, imported names

## Branch: feature/issue-10-dependency-graph
## Dependencies: Issue 7"

create_issue 11 "Symbol table / global registry with metadata" \
  "M3: Code Knowledge Graph" "m3,engine" \
  "## Why
Central lookup index: given a symbol name, returns file, line, type, signature.

## What to build
- Registry: symbol_id -> SymbolRecord
- Fully qualified names for disambiguation
- Lookup by: exact name, qualified name, regex, file path
- JSON serialization

## Files to create
- src/reporag/graph/symbol_table.py

## Acceptance Criteria
- [ ] Registers all symbols with fully qualified names
- [ ] Multiple lookup modes (exact, qualified, regex)
- [ ] JSON serialization works
- [ ] Handles name collisions across files

## Branch: feature/issue-11-symbol-table
## Dependencies: Issue 7"

create_issue 12 "Neo4j graph store with Cypher query layer" \
  "M3: Code Knowledge Graph" "m3,engine,infra" \
  "## Why
Neo4j enables complex graph traversals (shortest path, subgraph) at scale.

## What to build
- Neo4j driver wrapper: connect, create nodes/edges, query, clear
- Node types: Function, Class, Module
- Edge types: CALLS, IMPORTS, INHERITS, CONTAINS
- Cypher query helpers + NetworkX fallback

## Files to create
- src/reporag/graph/neo4j_store.py

## Acceptance Criteria
- [ ] Creates nodes with correct labels and properties
- [ ] CALLS, IMPORTS, INHERITS, CONTAINS edges work
- [ ] Cypher queries return correct results
- [ ] NetworkX fallback passes same test suite
- [ ] Bulk insert handles 10K+ nodes

## Branch: feature/issue-12-neo4j-store
## Dependencies: Issues 9, 10, 11"

# =============================================================================
# M4: Code Embedding Pipeline & Hybrid Index
# =============================================================================

create_issue 13 "Code embedding pipeline: CodeBERT/UniXcoder" \
  "M4: Embedding Pipeline & Hybrid Index" "m4,engine" \
  "## Why
Code-specific embeddings understand programming semantics that general-purpose embedders miss.

## What to build
- Load CodeBERT or UniXcoder from Hugging Face
- Batch embedding with GPU support (CPU fallback)
- L2-normalize embeddings
- Embedding cache

## Files to create
- src/reporag/embedding/code_embedder.py

## Acceptance Criteria
- [ ] Produces 768-dim vectors from code strings
- [ ] Batch embedding with configurable batch size
- [ ] GPU support with CPU fallback
- [ ] Embeddings L2-normalized
- [ ] Cache avoids re-computation

## Branch: feature/issue-13-code-embedder
## Dependencies: Issue 8"

create_issue 14 "Docstring and comment embedding pipeline" \
  "M4: Embedding Pipeline & Hybrid Index" "m4,engine" \
  "## Why
Docstrings describe intent in natural language, enabling natural language queries to match code documentation.

## What to build
- Load sentence-transformers model
- Embed docstrings, comments, README sections
- Link doc embeddings to parent code symbols
- Batch processing with progress

## Files to create
- src/reporag/embedding/doc_embedder.py

## Acceptance Criteria
- [ ] Produces vectors from natural language text
- [ ] Each embedding links to parent symbol ID
- [ ] Handles empty docstrings gracefully
- [ ] Batch processing with progress callback

## Branch: feature/issue-14-doc-embedder
## Dependencies: Issue 8"

create_issue 15 "Hybrid index builder: Qdrant vector + BM25 sparse" \
  "M4: Embedding Pipeline & Hybrid Index" "m4,engine,infra" \
  "## Why
Vector search + BM25 enables hybrid retrieval outperforming either alone.

## What to build
- Create Qdrant collection with payload schema
- Upsert code + doc embeddings with metadata
- Build BM25 index with code-aware tokenizer
- Code tokenizer: split camelCase, snake_case identifiers

## Files to create
- src/reporag/embedding/index_builder.py

## Acceptance Criteria
- [ ] Qdrant collection with correct schema
- [ ] All embeddings upserted with metadata
- [ ] BM25 index with code-aware tokenization
- [ ] Supports incremental updates

## Branch: feature/issue-15-index-builder
## Dependencies: Issues 13, 14"

# =============================================================================
# M5: Hybrid Retrieval Engine
# =============================================================================

create_issue 16 "Vector semantic search with configurable top-k" \
  "M5: Hybrid Retrieval Engine" "m5,engine" \
  "## Why
Vector search finds semantically similar code even when exact terms differ.

## What to build
- Query Qdrant with embedded query vector
- Return top-k with scores and payloads
- Filter by language, file path, symbol type
- Search code and doc collections separately, merge

## Files to create
- src/reporag/retrieval/vector_search.py

## Acceptance Criteria
- [ ] Returns top-k ranked by cosine similarity
- [ ] Payloads include file, lines, symbol, chunk text
- [ ] Filtering works
- [ ] Latency under 100ms for top-20

## Branch: feature/issue-16-vector-search
## Dependencies: Issue 15"

create_issue 17 "BM25 sparse keyword search for code identifiers" \
  "M5: Hybrid Retrieval Engine" "m5,engine" \
  "## Why
BM25 ensures exact identifier matches are found even if vector search misses them.

## What to build
- Query BM25 index with tokenized query
- Code-aware query tokenization
- Return top-k with BM25 scores
- Boost exact function/class name matches

## Files to create
- src/reporag/retrieval/bm25_search.py

## Acceptance Criteria
- [ ] Exact identifier queries return defining function as top-1
- [ ] Code-aware tokenization matches indexing
- [ ] Boosting exact name matches works
- [ ] Same result schema as vector search

## Branch: feature/issue-17-bm25-search
## Dependencies: Issue 15"

create_issue 18 "Graph-based retrieval: neighbor traversal, path queries" \
  "M5: Hybrid Retrieval Engine" "m5,engine" \
  "## Why
Structural questions require graph traversal, not text similarity.

## What to build
- N-hop neighbor query from a symbol
- Path query between two symbols
- Subgraph extraction around a set of symbols
- Convert graph results to common result schema

## Files to create
- src/reporag/retrieval/graph_traversal.py

## Acceptance Criteria
- [ ] N-hop neighbor query correct
- [ ] Shortest path between symbols
- [ ] Subgraph extraction works
- [ ] NetworkX fallback
- [ ] Common RetrievalResult schema

## Branch: feature/issue-18-graph-retrieval
## Dependencies: Issue 12"

create_issue 19 "Reciprocal Rank Fusion + cross-encoder reranker" \
  "M5: Hybrid Retrieval Engine" "m5,engine" \
  "## Why
RRF normalizes and fuses rankings from different sources. Cross-encoder provides final high-quality ranking.

## What to build
- RRF fusion from vector, BM25, graph ranked lists
- Configurable RRF constant k
- Cross-encoder reranker scoring (query, chunk) pairs
- Final top-k with reranked scores

## Files to create
- src/reporag/retrieval/fusion.py
- src/reporag/retrieval/reranker.py

## Acceptance Criteria
- [ ] RRF fuses 2-3 ranked lists correctly
- [ ] Handles items missing from some lists
- [ ] Cross-encoder reranks and reorders
- [ ] Reranked outperforms RRF-only
- [ ] Reranking under 500ms for 20 candidates

## Branch: feature/issue-19-fusion-reranker
## Dependencies: Issues 16, 17, 18"

# =============================================================================
# M6: Agentic Query Planner & Multi-Hop Decomposition
# =============================================================================

create_issue 20 "Query classifier: simple lookup vs. multi-hop vs. exploratory" \
  "M6: Agentic Query Planner" "m6,agent" \
  "## Why
Not every query needs the full agentic pipeline. Classification saves latency and cost.

## What to build
- LLM-based query classifier with few-shot examples
- Categories: simple-lookup, multi-hop, exploratory
- Confidence score, fallback to multi-hop on low confidence

## Files to create
- src/reporag/agent/planner.py

## Acceptance Criteria
- [ ] Correct classification for each query type
- [ ] Confidence score 0-1
- [ ] Low-confidence fallback to multi-hop
- [ ] Unit tests with 10+ queries

## Branch: feature/issue-20-query-classifier
## Dependencies: Issue 19"

create_issue 21 "Agentic query decomposer: break complex queries into sub-queries" \
  "M6: Agentic Query Planner" "m6,agent" \
  "## Why
Multi-hop questions require multiple retrieval steps with ordered sub-queries.

## What to build
- LLM-based decomposition using LangGraph state machine
- Ordered sub-queries with dependency edges
- Each sub-query: text, expected_answer_type, context_from

## Files to create
- src/reporag/agent/planner.py (decomposer section)

## Acceptance Criteria
- [ ] Decomposes into 2-5 ordered sub-queries
- [ ] Sub-queries have dependency edges
- [ ] Uses repo context for informed decomposition
- [ ] Handles queries that do not need decomposition
- [ ] LangGraph state machine with clear transitions

## Branch: feature/issue-21-query-decomposer
## Dependencies: Issue 20"

create_issue 22 "Strategy router + sub-query executor" \
  "M6: Agentic Query Planner" "m6,agent" \
  "## Why
Different sub-queries benefit from different retrieval strategies.

## What to build
- Route sub-queries to: graph, vector, bm25, or hybrid
- Routing based on sub-query characteristics
- Execute sub-queries in dependency order, passing context forward

## Files to create
- src/reporag/agent/router.py
- src/reporag/agent/executor.py

## Acceptance Criteria
- [ ] Routes identifier lookups to BM25
- [ ] Routes structural queries to graph
- [ ] Routes semantic queries to vector
- [ ] Executor respects dependency order
- [ ] Context forwarded between steps

## Branch: feature/issue-22-router-executor
## Dependencies: Issues 20, 21"

# =============================================================================
# M7: Answer Generation, Citation & FastAPI Serving
# =============================================================================

create_issue 23 "Context assembler: retrieved code -> structured prompt context" \
  "M7: Generation, Citation & API" "m7,engine" \
  "## Why
Raw retrieval results need ordering, deduplication, and formatting for the LLM prompt.

## What to build
- Order chunks by file path, then line number
- Deduplicate overlapping chunks
- Format: file header + line-numbered code
- Truncate to context window with priority ranking

## Files to create
- src/reporag/generation/context_assembler.py

## Acceptance Criteria
- [ ] Chunks ordered by file, then line
- [ ] Overlapping chunks merged
- [ ] Each chunk prefixed with file + line range
- [ ] Total tokens within max_tokens
- [ ] Highest-ranked prioritized when truncating

## Branch: feature/issue-23-context-assembler
## Dependencies: Issue 19"

create_issue 24 "Prompt builder with code-aware templates" \
  "M7: Generation, Citation & API" "m7,engine" \
  "## Why
Code-aware templates include citation instructions and examples for better answer quality.

## What to build
- Template per query type: simple-lookup, multi-hop, exploratory
- Citation format: [file_path:start_line-end_line]
- Few-shot examples of good cited answers

## Files to create
- src/reporag/generation/prompt_builder.py

## Acceptance Criteria
- [ ] Templates for all 3 query types
- [ ] Citation format clearly instructed
- [ ] Few-shot examples included
- [ ] Prompt fits within model context window

## Branch: feature/issue-24-prompt-builder
## Dependencies: Issue 23"

create_issue 25 "LLM generation with line-level citation extraction" \
  "M7: Generation, Citation & API" "m7,engine" \
  "## Why
Generate answers and extract/validate citations linking back to source code.

## What to build
- Call LLM (OpenAI/Anthropic, configurable)
- Parse citation markers [file:line-line]
- Validate citations against retrieved context
- Return structured {answer, citations}

## Files to create
- src/reporag/generation/generator.py
- src/reporag/generation/citation.py

## Acceptance Criteria
- [ ] Calls configurable LLM API
- [ ] Extracts citation markers
- [ ] Validates citations (flags invalid)
- [ ] Citation coverage >= 90%
- [ ] Handles LLM errors gracefully

## Branch: feature/issue-25-generator-citation
## Dependencies: Issue 24"

create_issue 26 "FastAPI application: repo, query, and health endpoints" \
  "M7: Generation, Citation & API" "m7,backend,infra" \
  "## Why
API layer exposes the pipeline over HTTP for the frontend and external consumers.

## What to build
- POST /api/v1/repos/ingest
- GET /api/v1/repos
- POST /api/v1/query
- GET /api/v1/health
- OpenAPI docs

## Files to create
- src/reporag/api/main.py
- src/reporag/api/routes/repos.py
- src/reporag/api/routes/query.py
- src/reporag/api/routes/health.py

## Acceptance Criteria
- [ ] Ingest endpoint triggers async ingestion
- [ ] Query endpoint returns {answer, citations, metadata}
- [ ] Health endpoint shows component status
- [ ] OpenAPI docs at /docs
- [ ] Pydantic request/response validation

## Branch: feature/issue-26-fastapi
## Dependencies: Issue 25"

# =============================================================================
# M8: Google OAuth, JWT Auth & API Middleware
# =============================================================================

create_issue 27 "Google OAuth 2.0 login flow" \
  "M8: Auth & API Middleware" "m8,backend" \
  "## Why
NST standard auth pattern. Secure, passwordless authentication.

## What to build
- GET /auth/google redirect to consent screen
- GET /auth/google/callback exchange code for tokens
- Create/update User record
- Return JWT access + refresh tokens

## Files to create
- src/reporag/api/routes/auth.py
- src/reporag/api/middleware/auth.py

## Acceptance Criteria
- [ ] Redirect to Google OAuth works
- [ ] Callback exchanges code for tokens
- [ ] User record created/updated
- [ ] JWT tokens returned
- [ ] OAuth errors handled

## Branch: feature/issue-27-google-oauth
## Dependencies: Issue 26"

create_issue 28 "JWT token issuance, validation, refresh middleware" \
  "M8: Auth & API Middleware" "m8,backend" \
  "## Why
Stateless authentication with JWT, validated on every request.

## What to build
- JWT encode/decode with user_id, email, roles, expiry
- Validation middleware on protected routes
- Refresh endpoint
- get_current_user dependency injection

## Files to create
- src/reporag/api/middleware/auth.py

## Acceptance Criteria
- [ ] JWT contains user_id, email, exp, iat
- [ ] Protected routes return 401 without valid token
- [ ] Refresh endpoint works
- [ ] get_current_user injectable

## Branch: feature/issue-28-jwt-middleware
## Dependencies: Issue 27"

create_issue 29 "Rate limiting, request logging, error handling middleware" \
  "M8: Auth & API Middleware" "m8,backend,infra" \
  "## Why
Production APIs need rate limiting, structured logging, and consistent error handling.

## What to build
- Per-user rate limiter (configurable, default 60/min)
- Structured JSON logging
- Global exception handler with consistent error shape
- Request ID middleware

## Files to create
- src/reporag/api/middleware/rate_limiter.py
- src/reporag/api/middleware/logging.py
- src/reporag/api/middleware/error_handler.py

## Acceptance Criteria
- [ ] Rate limiter returns 429 after limit
- [ ] Per-user rate limiting
- [ ] JSON structured logs
- [ ] Request ID in response header
- [ ] Clean error responses (no stack traces)

## Branch: feature/issue-29-middleware
## Dependencies: Issue 28"

# =============================================================================
# M9: React Frontend
# =============================================================================

create_issue 30 "React + Vite scaffold, routing, auth flow (Google login)" \
  "M9: React Frontend" "m9,frontend" \
  "## Why
Visual interface for exploring repos, asking questions, and seeing cited answers.

## What to build
- React 18 + Vite + TailwindCSS scaffold
- React Router: /, /login, /repos, /repos/:id, /query
- Google login -> backend OAuth
- JWT in memory, auto-attach to API requests
- Auth context, protected routes

## Files to create
- frontend/package.json, vite.config.js, tailwind.config.js
- frontend/src/App.jsx, main.jsx
- frontend/src/pages/Login.jsx, Dashboard.jsx
- frontend/src/context/AuthContext.jsx

## Acceptance Criteria
- [ ] Vite dev server with hot reload
- [ ] TailwindCSS works
- [ ] Google login redirects to OAuth
- [ ] JWT stored in memory, attached to requests
- [ ] Protected routes redirect unauthenticated users

## Branch: feature/issue-30-frontend-scaffold
## Dependencies: Issue 28"

create_issue 31 "Repository explorer: file tree + syntax-highlighted code viewer" \
  "M9: React Frontend" "m9,frontend" \
  "## Why
Visual file browsing with syntax highlighting, supporting citation cross-references.

## What to build
- File tree component (collapsible, file icons)
- Code viewer with syntax highlighting + line numbers
- Line range highlighting for citations
- Fetch file contents from API

## Files to create
- frontend/src/pages/RepoExplorer.jsx
- frontend/src/components/FileTree.jsx
- frontend/src/components/CodeViewer.jsx

## Acceptance Criteria
- [ ] File tree loads and renders collapsible directories
- [ ] Syntax-highlighted code with line numbers
- [ ] Line range highlighting works
- [ ] Supports Python, JS, TS syntax

## Branch: feature/issue-31-repo-explorer
## Dependencies: Issue 30"

create_issue 32 "Conversational Q&A interface with citation highlights" \
  "M9: React Frontend" "m9,frontend" \
  "## Why
Core UX: ask questions, get cited answers, click citations to jump to code.

## What to build
- Chat-style Q&A interface
- Inline citation links [file:line-line]
- Click citation -> navigate to code with highlights
- Loading state, query history

## Files to create
- frontend/src/pages/QueryInterface.jsx
- frontend/src/components/QueryInput.jsx
- frontend/src/components/AnswerDisplay.jsx
- frontend/src/components/CitationLink.jsx

## Acceptance Criteria
- [ ] Chat input sends query to API
- [ ] Answer renders with markdown
- [ ] Citations clickable, navigate to code
- [ ] Loading indicator
- [ ] Query history in session

## Branch: feature/issue-32-qa-interface
## Dependencies: Issues 30, 31"

create_issue 33 "Interactive graph visualizer: call graph / dependency graph" \
  "M9: React Frontend" "m9,frontend" \
  "## Why
Visual graph helps users understand code architecture at a glance.

## What to build
- Force-directed graph (react-force-graph or D3)
- Node types: functions, classes, modules with distinct shapes
- Edge types: calls, imports, inherits with distinct styles
- Click node -> details + navigate to code
- Zoom, pan, filter, search

## Files to create
- frontend/src/components/GraphVisualizer.jsx
- frontend/src/components/RepoSelector.jsx

## Acceptance Criteria
- [ ] Graph renders with correct node/edge types
- [ ] Force layout readable for 100+ nodes
- [ ] Click node shows details
- [ ] Filter by module
- [ ] Search highlights nodes

## Branch: feature/issue-33-graph-viz
## Dependencies: Issues 30, 31"

# =============================================================================
# M10: Evaluation Harness, E2E Testing & Demo
# =============================================================================

create_issue 34 "Retrieval evaluation suite: context recall, precision, MRR" \
  "M10: Eval, Testing & Demo" "m10,eval" \
  "## Why
Retrieval is the #1 RAG failure point. Measuring with standard IR metrics guides tuning.

## What to build
- Metrics: context recall@k, precision@k, MRR, NDCG
- Evaluation dataset: (query, relevant_chunks[]) pairs
- Batch runner with aggregate + per-query metrics
- Comparison mode across retrieval strategies

## Files to create
- src/reporag/evaluation/metrics.py
- src/reporag/evaluation/retrieval_eval.py
- examples/eval_dataset.json

## Acceptance Criteria
- [ ] All metrics computed correctly
- [ ] 20+ query-relevance pairs in eval dataset
- [ ] Batch runner with aggregate metrics
- [ ] Comparison mode with delta

## Branch: feature/issue-34-retrieval-eval
## Dependencies: Issue 19"

create_issue 35 "Generation evaluation: faithfulness, relevance, hallucination" \
  "M10: Eval, Testing & Demo" "m10,eval" \
  "## Why
Measure whether answers are grounded in retrieved context and relevant to queries.

## What to build
- Faithfulness (LLM-as-judge)
- Answer relevance (LLM-as-judge)
- Hallucination detection
- Citation coverage metric

## Files to create
- src/reporag/evaluation/generation_eval.py

## Acceptance Criteria
- [ ] Faithfulness score 0-1
- [ ] Answer relevance score 0-1
- [ ] Hallucination flags unsupported claims
- [ ] Citation coverage computed
- [ ] Configurable judge model

## Branch: feature/issue-35-generation-eval
## Dependencies: Issue 25"

create_issue 36 "End-to-end integration tests with sample repositories" \
  "M10: Eval, Testing & Demo" "m10,eval,test" \
  "## Why
Integration tests verify the full pipeline works end-to-end with known inputs.

## What to build
- Sample Python project in examples/sample_repo/
- Integration test: clone -> parse -> graph -> index -> query -> verify
- Test queries with expected answers and citations
- CI-compatible (Docker, NetworkX fallback)

## Files to create
- examples/sample_repo/ (5-10 Python files)
- examples/sample_queries.json
- tests/integration/test_e2e_pipeline.py

## Acceptance Criteria
- [ ] sample_repo with known call graph structure
- [ ] E2E test produces correct cited answers
- [ ] 5+ test queries with expected results
- [ ] CI-compatible (no external services required)

## Branch: feature/issue-36-integration-tests
## Dependencies: Issues 26, 34, 35"

create_issue 37 "Performance benchmarks, load testing, demo script" \
  "M10: Eval, Testing & Demo" "m10,eval,test" \
  "## Why
Demonstrate production readiness with benchmarks and a polished demo.

## What to build
- Benchmark: ingestion throughput, indexing time, query latency
- Load test: 50 concurrent queries, measure p50/p95/p99
- Demo script: 5 progressively harder queries
- Failure analyzer: retrieval vs generation vs context overflow

## Files to create
- src/reporag/evaluation/benchmarks.py
- src/reporag/evaluation/failure_analyzer.py
- scripts/run_benchmarks.sh
- scripts/demo.sh

## Acceptance Criteria
- [ ] Benchmark reports ingestion, indexing, query metrics
- [ ] Load test p95 under 5 seconds at 50 users
- [ ] Demo runs 5 queries with commentary
- [ ] Failure analyzer classifies failure modes

## Branch: feature/issue-37-benchmarks-demo
## Dependencies: Issue 36"

echo ""
echo "All 37 issues created successfully!"
echo "Verify: gh issue list --repo $REPO --limit 40"
