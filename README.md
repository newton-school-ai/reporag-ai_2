# RepoRAG AI

**Code-Aware Repository Intelligence -- Agentic RAG for Codebases**

RepoRAG is a production-grade Retrieval-Augmented Generation system purpose-built for
source code. It parses repositories into ASTs, builds call graphs and dependency graphs,
creates hybrid indices (vector + BM25), and uses an agentic query planner to answer
complex multi-hop code questions with line-level citations.

Unlike generic RAG systems that treat code as text, RepoRAG understands code structure:
function boundaries, call relationships, import chains, and scope hierarchies. It
answers questions like "How does the auth flow work end-to-end?" or "What would break
if I change this interface?" by combining graph traversal with semantic retrieval.

---

## Pipeline

```
clone repo
  -> discover files
    -> parse AST (tree-sitter)
      -> extract symbols (functions, classes, imports)
        -> build call graph + dependency graph (Neo4j)
          -> embed code blocks (CodeBERT/UniXcoder) + docstrings (sentence-transformers)
            -> build hybrid index (Qdrant vector + BM25 sparse)

user query
  -> classify (simple / multi-hop / exploratory)
    -> decompose into sub-queries (agentic planner)
      -> route each sub-query (graph / vector / hybrid)
        -> retrieve + fuse (Reciprocal Rank Fusion)
          -> rerank (cross-encoder)
            -> assemble code context
              -> generate answer (LLM)
                -> extract line-level citations
                  -> return cited answer
```

---

## Key Features

- **AST-Aware Parsing**: tree-sitter-based parsing that understands code structure,
  not just text. Extracts functions, classes, methods, decorators, imports with full
  metadata (line ranges, docstrings, signatures, return types).

- **Code Knowledge Graph**: Neo4j-backed graph with call relationships, import
  dependencies, inheritance chains, and a global symbol table. Enables multi-hop
  traversal queries.

- **Hybrid Retrieval**: Vector semantic search (CodeBERT/UniXcoder embeddings) +
  BM25 sparse search for identifier matching + graph traversal for structural queries.
  Fused via Reciprocal Rank Fusion and reranked with a cross-encoder.

- **Agentic Query Planner**: LLM-powered agent that classifies query complexity,
  decomposes multi-hop questions into sub-queries, routes each to the optimal
  retrieval strategy, and synthesizes a coherent answer.

- **Line-Level Citations**: Every claim in the generated answer links back to specific
  file paths and line ranges in the source repository.

- **Production Architecture**: Offline pipeline (ingestion, indexing) separated from
  online pipeline (retrieval, generation). FastAPI serving, Google OAuth + JWT auth,
  rate limiting, health checks.

- **Evaluation Harness**: Built-in metrics -- context recall, precision, MRR,
  faithfulness, relevance, hallucination detection -- with benchmark suites against
  real open-source repositories.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AST Parsing | tree-sitter, tree-sitter-python, tree-sitter-javascript |
| Knowledge Graph | Neo4j, NetworkX (in-memory fallback) |
| Code Embeddings | CodeBERT / UniXcoder (Hugging Face) |
| Doc Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | Qdrant |
| Sparse Index | rank-bm25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Agent Framework | LangGraph |
| LLM | OpenAI GPT-4o / Anthropic Claude (configurable) |
| API | FastAPI, Uvicorn |
| Auth | Google OAuth 2.0, PyJWT |
| Database | SQLite (default), PostgreSQL (production) |
| Frontend | React 18, Vite, TailwindCSS |
| Code Viewer | Monaco Editor / react-syntax-highlighter |
| Graph Viz | react-force-graph / D3.js |
| Containerization | Docker, docker-compose |
| CI | GitHub Actions |
| Eval | RAGAS, custom metrics |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker and docker-compose
- Neo4j (or use Docker)
- A Qdrant instance (or use Docker)

### Setup

```bash
# Clone
git clone git@github.com:newton-school-ai/reporag-ai.git
cd reporag-ai

# Environment
cp .env.example .env
# Edit .env with your API keys and DB credentials

# Backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install

# Infrastructure (Neo4j + Qdrant + PostgreSQL)
docker-compose up -d

# Run API server
uvicorn src.reporag.api.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Ingest a Repository

```bash
curl -X POST http://localhost:8000/api/v1/repos/ingest \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/pallets/flask", "branch": "main"}'
```

### Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the request routing work end-to-end?", "repo_id": "flask"}'
```

---

## Project Structure

```
reporag-ai/
  .github/
    workflows/ci.yml              CI: ASCII guard + lint + tests
    PULL_REQUEST_TEMPLATE.md
    ISSUE_TEMPLATE/
      feature.md
      bug.md
  src/reporag/
    __init__.py
    config.py                     Settings, env loading, validation
    ingestion/
      cloner.py                   Git repo cloning + file discovery
      parser.py                   tree-sitter AST parsing
      symbol_extractor.py         Function/class/import extraction
      chunker.py                  AST-aware semantic code chunking
    graph/
      call_graph.py               Build call graphs from AST
      dependency_graph.py         Import/module dependency graph
      symbol_table.py             Global symbol registry
      neo4j_store.py              Neo4j persistence + Cypher queries
    embedding/
      code_embedder.py            CodeBERT/UniXcoder code embeddings
      doc_embedder.py             Docstring/comment embeddings
      index_builder.py            Build Qdrant vector + BM25 indices
    retrieval/
      vector_search.py            Semantic vector search
      bm25_search.py              Sparse keyword search
      graph_traversal.py          Graph-based multi-hop retrieval
      fusion.py                   Reciprocal Rank Fusion
      reranker.py                 Cross-encoder reranking
    agent/
      planner.py                  Query decomposition agent
      router.py                   Strategy routing per sub-query
      executor.py                 Sub-query execution orchestrator
      synthesizer.py              Answer synthesis with citations
    generation/
      context_assembler.py        Retrieved code -> prompt context
      prompt_builder.py           Code-aware prompt templates
      generator.py                LLM answer generation
      citation.py                 Line-level citation extraction
    evaluation/
      metrics.py                  Retrieval + generation metrics
      benchmarks.py               Code QA benchmark suites
      failure_analyzer.py         Failure mode classification
    api/
      main.py                     FastAPI application
      routes/
        repos.py                  Repo ingestion endpoints
        query.py                  Query endpoints
        health.py                 Health check
      middleware/
        auth.py                   Google OAuth + JWT
  frontend/
    src/
      pages/
        Dashboard.jsx
        RepoExplorer.jsx
        QueryInterface.jsx
        Login.jsx
      components/
        CodeViewer.jsx            Syntax-highlighted code + citations
        GraphVisualizer.jsx       Interactive call/dependency graph
        SearchResults.jsx
        QueryInput.jsx
        RepoSelector.jsx
  tests/
    unit/                         Per-module unit tests
    integration/                  Pipeline integration tests
  docs/
    ARCHITECTURE.md               System design + data flow
    API_CONTRACT.md               OpenAPI spec + endpoint docs
    RUNBOOK.md                    Ops playbook
  notebooks/                      EDA + tuning notebooks
  scripts/
    create_github_issues.sh       Bulk issue creation via gh CLI
  examples/
    sample_repo/                  Tiny Python project for testing
    sample_queries.json           Example queries + expected results
  _internal/
    PROJECT_CONTEXT.md            Internal planning doc (gitignored)
```

---

## Milestones

| # | Milestone | Key Output | Issues |
|---|-----------|-----------|--------|
| M1 | Repo Scaffold, CI & Docker | Working CI + Docker stack | 1-4 |
| M2 | Repository Ingestion & AST Parsing | AST parser extracts symbols from any Python repo | 5-8 |
| M3 | Code Knowledge Graph | Neo4j graph with call + dependency edges | 9-12 |
| M4 | Embedding Pipeline & Hybrid Index | Searchable vector + BM25 index of code | 13-15 |
| M5 | Hybrid Retrieval Engine | Fused retrieval with reranking | 16-19 |
| M6 | Agentic Query Planner | Multi-hop query decomposition + routing | 20-22 |
| M7 | Generation, Citation & API | FastAPI serving cited answers | 23-26 |
| M8 | Auth & API Middleware | Google OAuth + JWT + rate limiting | 27-29 |
| M9 | React Frontend | Code explorer + conversational Q&A UI | 30-33 |
| M10 | Eval, Testing & Demo | Eval harness + benchmarks + demo | 34-37 |

---

## Pod

| Role | GitHub Role | Responsibilities |
|------|------------|-----------------|
| Faculty | Admin | Merges to main, milestone sign-off, defense Q&A |
| Maintainer | Maintain | Merges to dev, code review, CI health |
| Contributor 1 | Write | Ingestion + AST parsing + graph |
| Contributor 2 | Write | Embedding + retrieval + reranking |
| Contributor 3 | Write | Agent + generation + API |
| Contributor 4 | Write | Frontend + evaluation + integration |

---

## Documentation

- [MILESTONES.md](MILESTONES.md) -- Milestone details with defense questions
- [ISSUES_TRACKER.md](ISSUES_TRACKER.md) -- Full issue breakdown
- [CONTRIBUTING.md](CONTRIBUTING.md) -- Branch strategy, PR workflow, coding standards
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) -- Setup, daily workflow, troubleshooting
- [POD_GUIDE.md](POD_GUIDE.md) -- Pod roles, sprint timeline, review checklist

---

## License

For educational use within Newton School of Technology pod projects.

---

NST Engineering | RepoRAG AI | 2026
