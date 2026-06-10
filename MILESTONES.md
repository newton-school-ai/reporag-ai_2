# MILESTONES.md - RepoRAG AI

10 milestones, 37 issues. Each milestone is roughly 1 sprint (1 week).

---

## M1: Repo Scaffold, CI & Docker Setup

**Key Output**: Working CI pipeline (ASCII guard + lint + tests), Docker stack
(Neo4j + Qdrant + app), project config module, pre-commit hooks.

**Acceptance Criteria**:
- `docker-compose up -d` brings up Neo4j, Qdrant, and the API container
- `pytest` passes with zero tests collected (scaffold only)
- CI runs on every PR to dev: ASCII guard, ruff, black --check, pytest
- Pre-commit hooks block non-ASCII, internal data leaks, lint failures
- `.env.example` documents all required environment variables
- `python -c "from src.reporag.config import settings; print(settings)"` succeeds

**Defense Questions**:
1. Why separate offline (ingestion) and online (retrieval) pipelines in production RAG?
2. Explain the trade-off between SQLite and PostgreSQL for this project.
3. What does the ASCII guard CI step prevent and why does it matter?
4. How does pre-commit differ from CI -- when does each catch errors?
5. Why pin dependency versions in requirements.txt instead of using ranges?

**Issues**: 1, 2, 3, 4

---

## M2: Repository Ingestion & AST Parsing Engine

**Key Output**: Given a Git URL, the system clones the repo, parses every Python
file into an AST via tree-sitter, and extracts all symbols (functions, classes,
methods, imports) with metadata (line ranges, signatures, docstrings, decorators).

**Acceptance Criteria**:
- `cloner.py` clones a public GitHub repo to a temp directory and lists all `.py` files
- `parser.py` parses a Python file and returns the tree-sitter AST
- `symbol_extractor.py` extracts functions, classes, imports with line ranges and docstrings
- `chunker.py` produces AST-aware chunks that never split a function/class mid-body
- Unit tests cover: empty file, single function, nested classes, decorators, async functions
- Processing Flask's codebase (~200 files) completes in under 60 seconds

**Defense Questions**:
1. Why use tree-sitter instead of Python's built-in `ast` module?
2. What is an AST and how does it differ from a CST (Concrete Syntax Tree)?
3. Explain why naive text chunking (e.g., 512-token windows) fails for code.
4. How would you extend the parser to support JavaScript/TypeScript?
5. What metadata do you attach to each extracted symbol and why?

**Issues**: 5, 6, 7, 8

---

## M3: Code Knowledge Graph Construction

**Key Output**: Neo4j graph populated with nodes (functions, classes, modules) and
edges (calls, imports, inherits). Queryable via Cypher. In-memory NetworkX fallback
for testing without Neo4j.

**Acceptance Criteria**:
- `call_graph.py` builds edges: function A -> calls -> function B (intra and cross-module)
- `dependency_graph.py` builds edges: module A -> imports -> module B
- `symbol_table.py` maintains a global registry: symbol name -> (file, line, type, signature)
- `neo4j_store.py` persists the graph and supports Cypher queries
- `MATCH (f:Function)-[:CALLS]->(g:Function) RETURN f.name, g.name` returns valid results
- NetworkX fallback passes same query interface without Neo4j running
- Unit tests cover: recursive calls, circular imports, star imports, re-exports

**Defense Questions**:
1. What is a call graph and how do you handle dynamic dispatch / monkey-patching?
2. Explain the difference between a static call graph and a dynamic call graph.
3. Why store the graph in Neo4j rather than just NetworkX?
4. How do you resolve `from module import *` when building the dependency graph?
5. What graph traversal algorithms are useful for answering "what breaks if I change X?"

**Issues**: 9, 10, 11, 12

---

## M4: Code Embedding Pipeline & Hybrid Index

**Key Output**: Every code chunk and docstring is embedded. Qdrant collection holds
vector embeddings; BM25 index holds sparse representations. Both are queryable.

**Acceptance Criteria**:
- `code_embedder.py` produces 768-dim vectors from code blocks using CodeBERT/UniXcoder
- `doc_embedder.py` produces vectors from docstrings/comments using sentence-transformers
- `index_builder.py` creates a Qdrant collection with proper payload schema
- `index_builder.py` creates a BM25 index over tokenized code identifiers
- Vector search for "authentication middleware" returns auth-related code chunks
- BM25 search for "def authenticate" returns exact function matches
- Embedding 10K chunks completes in under 5 minutes on CPU

**Defense Questions**:
1. Why use a code-specific embedding model (CodeBERT) instead of a general one?
2. What is the difference between dense (vector) and sparse (BM25) retrieval?
3. Explain how Qdrant stores and searches vectors (HNSW algorithm basics).
4. Why embed docstrings separately from code -- when would you query each?
5. How would you handle embedding drift if the model is updated?

**Issues**: 13, 14, 15

---

## M5: Hybrid Retrieval Engine

**Key Output**: Given a query, the system runs vector search, BM25 search, and
graph traversal in parallel, fuses results via RRF, and reranks with a cross-encoder.

**Acceptance Criteria**:
- `vector_search.py` returns top-k semantically similar code chunks with scores
- `bm25_search.py` returns top-k keyword-matched chunks with BM25 scores
- `graph_traversal.py` returns neighbors/paths from the knowledge graph
- `fusion.py` merges results from all three via Reciprocal Rank Fusion
- `reranker.py` reranks fused results using cross-encoder and returns final top-k
- End-to-end: query -> fused + reranked results in under 2 seconds
- Reranked results measurably outperform any single retrieval method (A/B eval)

**Defense Questions**:
1. What is Reciprocal Rank Fusion and why is it better than score normalization?
2. Explain how a cross-encoder reranker differs from a bi-encoder retriever.
3. When would graph-based retrieval outperform vector search? Give examples.
4. What is the latency/accuracy trade-off of reranking and how do you tune top-k?
5. How would you add a learned sparse model (SPLADE) alongside BM25?

**Issues**: 16, 17, 18, 19

---

## M6: Agentic Query Planner & Multi-Hop Decomposition

**Key Output**: An LLM-powered agent that classifies query complexity, decomposes
multi-hop questions into sub-queries, routes each to the optimal retrieval strategy,
executes them, and synthesizes a coherent plan.

**Acceptance Criteria**:
- `planner.py` classifies queries into: simple-lookup, multi-hop, exploratory
- `planner.py` decomposes "How does auth work end-to-end?" into sub-queries like
  ["find auth entry point", "trace auth middleware chain", "find token validation"]
- `router.py` assigns each sub-query a strategy: graph / vector / hybrid
- `executor.py` runs sub-queries in dependency order, passing context forward
- `synthesizer.py` merges sub-query results into a coherent execution plan
- Multi-hop query produces better results than single-shot retrieval (eval metric)

**Defense Questions**:
1. What makes a query "multi-hop" and why can single retrieval not answer it?
2. Explain the ReAct (Reason + Act) pattern and how it applies here.
3. How does the router decide between graph vs. vector vs. hybrid retrieval?
4. What happens when a sub-query returns no results -- how does the agent recover?
5. Compare this agentic approach to Chain-of-Thought prompting -- trade-offs?

**Issues**: 20, 21, 22

---

## M7: Answer Generation, Citation & FastAPI Serving

**Key Output**: FastAPI application that accepts queries, runs the full pipeline,
and returns answers with line-level citations linking to source files.

**Acceptance Criteria**:
- `context_assembler.py` orders retrieved chunks by relevance and dependency
- `prompt_builder.py` builds prompts with code context, file paths, line numbers
- `generator.py` calls LLM and returns structured answer with citation markers
- `citation.py` extracts `[file:line_start-line_end]` citations from generated text
- `POST /api/v1/query` returns `{answer, citations: [{file, start, end, snippet}]}`
- `POST /api/v1/repos/ingest` triggers async ingestion pipeline
- `GET /api/v1/health` returns pipeline status
- Every claim in the answer has at least one citation (citation coverage >= 90%)

**Defense Questions**:
1. How do you order retrieved chunks in the prompt to minimize lost-in-the-middle?
2. What prompt engineering techniques improve citation accuracy?
3. Explain the difference between extractive and abstractive citation.
4. How do you handle the LLM context window limit with large retrieved contexts?
5. Why use async ingestion and how do you communicate progress to the client?

**Issues**: 23, 24, 25, 26

---

## M8: Google OAuth, JWT Auth & API Middleware

**Key Output**: Secure API with Google OAuth login, JWT tokens, rate limiting,
structured logging, and error handling middleware.

**Acceptance Criteria**:
- Google OAuth 2.0 login returns a JWT access token + refresh token
- JWT middleware validates tokens on every protected endpoint
- Token refresh endpoint issues new access tokens from valid refresh tokens
- Rate limiter caps requests per user (configurable, default 60/min)
- Structured JSON logging for every request (method, path, status, latency)
- Global error handler returns consistent error shapes `{error, detail, status}`

**Defense Questions**:
1. Explain the OAuth 2.0 authorization code flow step by step.
2. What is the difference between access tokens and refresh tokens?
3. How do you securely store JWT secrets and what happens if they leak?
4. What rate limiting algorithms exist and which did you choose? Why?
5. Why structured logging and how would you query logs in production?

**Issues**: 27, 28, 29

---

## M9: React Frontend -- Code Explorer & Q&A Interface

**Key Output**: React + Vite frontend with Google login, repository file explorer
with syntax-highlighted code, conversational Q&A with citation highlights, and an
interactive graph visualizer.

**Acceptance Criteria**:
- Google login redirects to OAuth, stores JWT, attaches to API requests
- Dashboard shows ingested repositories with status (indexing / ready / error)
- RepoExplorer shows file tree; clicking a file shows syntax-highlighted source
- QueryInterface accepts natural language; response shows answer + highlighted citations
- Clicking a citation scrolls to the exact lines in the code viewer
- GraphVisualizer renders the call graph / dependency graph with zoom + pan + click
- Responsive layout; works on desktop browsers (mobile not required)

**Defense Questions**:
1. How do you securely store and refresh JWT tokens in a React SPA?
2. Explain how the citation highlighting works (data flow from API to UI).
3. What library did you use for graph visualization and why?
4. How do you handle streaming responses from the LLM in the frontend?
5. What accessibility considerations did you address?

**Issues**: 30, 31, 32, 33

---

## M10: Evaluation Harness, E2E Testing & Demo

**Key Output**: Automated evaluation suite, end-to-end integration tests against
real open-source repos, performance benchmarks, and a polished demo script.

**Acceptance Criteria**:
- `metrics.py` computes: context recall@10, precision@10, MRR, NDCG
- `metrics.py` computes: faithfulness, answer relevance, hallucination rate
- `benchmarks.py` runs eval against 3+ open-source repos with ground-truth Q&A pairs
- `failure_analyzer.py` classifies failures as retrieval / generation / context overflow
- Integration tests: ingest -> index -> query -> verify citation for sample_repo
- API load test: 50 concurrent queries, p95 latency under 5 seconds
- Demo script walks through ingestion + 5 progressively harder queries

**Defense Questions**:
1. What is context recall@k and why is retrieval failure the #1 RAG failure mode?
2. How do you measure faithfulness -- is the answer grounded in retrieved context?
3. Explain NDCG and when it matters more than simple precision.
4. How do you create ground-truth Q&A pairs for evaluation?
5. What is the difference between unit, integration, and end-to-end tests in RAG?

**Issues**: 34, 35, 36, 37

---

NST Engineering | RepoRAG AI Milestones | 2026
