# ISSUES_TRACKER.md - RepoRAG AI

37 issues across 10 milestones. Each issue has: Why, What to build,
Files to create/update, How to test locally, Acceptance Criteria, Branch, Dependencies.

---

## M1: Repo Scaffold, CI & Docker Setup

### Issue 1 - Initialize repo scaffold, CI workflow, Docker setup

**Why**: Every NST project starts with a reproducible scaffold: CI that catches
problems before review, Docker that gives every pod member the same environment,
and a directory layout that matches the architecture.

**What to build**:
- GitHub Actions CI workflow: ASCII guard, ruff lint, black format check, pytest
- Dockerfile for the API server (Python 3.11-slim base)
- docker-compose.yml with services: api, neo4j, qdrant, postgres (optional)
- Full directory tree with __init__.py files in every package

**Files to create**:
- .github/workflows/ci.yml
- Dockerfile
- docker-compose.yml
- All src/reporag/**/__init__.py files

**How to test locally**:
```bash
docker-compose up -d
docker-compose ps  # all services healthy
python -c "import src.reporag; print('import OK')"
```

**Acceptance Criteria**:
- [ ] CI workflow triggers on PR to dev and push to dev
- [ ] ASCII guard step fails on non-ASCII characters
- [ ] docker-compose up -d starts all 3+ services
- [ ] All __init__.py files exist in every src package

**Branch**: `feature/issue-1-scaffold`
**Dependencies**: None

---

### Issue 2 - Pre-commit hooks: ASCII guard, internal-data guard, ruff, black

**Why**: Pre-commit catches formatting and policy violations before they hit CI,
saving review cycles. The ASCII guard and internal-data guard are NST non-negotiables.

**What to build**:
- .pre-commit-config.yaml with: trailing-whitespace, end-of-file-fixer, ruff, black,
  ASCII-only check, internal-data path guard (blocks _internal/ except PROJECT_CONTEXT.md)

**Files to create**:
- .pre-commit-config.yaml

**How to test locally**:
```bash
pre-commit install
echo "smart quote: \xe2\x80\x9c" > test_ascii.py
pre-commit run --all-files  # should fail on ASCII guard
rm test_ascii.py
```

**Acceptance Criteria**:
- [ ] pre-commit install succeeds
- [ ] Non-ASCII characters are caught and blocked
- [ ] Files in _internal/ (except PROJECT_CONTEXT.md) are blocked from commit
- [ ] ruff and black run without errors on clean scaffold

**Branch**: `feature/issue-2-precommit`
**Dependencies**: Issue 1

---

### Issue 3 - Database setup: SQLite default, Postgres-ready, Alembic migrations

**Why**: The project needs persistent storage for repo metadata, ingestion status,
and user data. SQLite for local dev, Postgres for production. Alembic ensures
schema changes are versioned and reproducible.

**What to build**:
- SQLAlchemy models: Repository, IngestionJob, User, QueryLog
- Alembic migration setup with initial schema
- Database session factory that switches SQLite/Postgres via env var

**Files to create/update**:
- src/reporag/db/__init__.py (new package)
- src/reporag/db/models.py
- src/reporag/db/session.py
- alembic.ini
- alembic/env.py
- alembic/versions/ (initial migration)

**How to test locally**:
```bash
alembic upgrade head
python -c "from src.reporag.db.session import get_db; print('DB OK')"
sqlite3 reporag.db ".tables"  # should list tables
```

**Acceptance Criteria**:
- [ ] alembic upgrade head creates all tables in SQLite
- [ ] DATABASE_URL=postgresql://... switches to Postgres without code changes
- [ ] Models include: Repository, IngestionJob, User, QueryLog
- [ ] Session factory provides async-compatible sessions

**Branch**: `feature/issue-3-database`
**Dependencies**: Issue 1

---

### Issue 4 - Project configuration module: .env loading, settings validation

**Why**: Centralized config prevents scattered os.getenv calls, validates required
vars at startup, and provides type-safe settings access throughout the codebase.

**What to build**:
- Pydantic Settings class loading from .env
- Sections: database, neo4j, qdrant, llm, auth, app
- .env.example with all variables documented

**Files to create/update**:
- src/reporag/config.py
- .env.example

**How to test locally**:
```bash
cp .env.example .env
python -c "from src.reporag.config import settings; print(settings.model_dump())"
```

**Acceptance Criteria**:
- [ ] Missing required vars raise ValidationError at import time
- [ ] .env.example documents every variable with comments
- [ ] settings object is importable from any module
- [ ] Sensitive fields (API keys, secrets) are marked as SecretStr

**Branch**: `feature/issue-4-config`
**Dependencies**: Issue 1

---

## M2: Repository Ingestion & AST Parsing Engine

### Issue 5 - Git repository cloner and file discovery service

**Why**: The ingestion pipeline starts with cloning a repository and discovering
all parseable source files. Needs to handle large repos, specific branches, and
shallow clones for performance.

**What to build**:
- Clone a Git repo (URL or local path) to a temp directory
- Support branch selection and shallow cloning (depth=1)
- Walk the file tree, filter by language extensions (.py, .js, .ts)
- Return a manifest: list of (file_path, language, size_bytes)

**Files to create**:
- src/reporag/ingestion/cloner.py

**How to test locally**:
```bash
python -c "
from src.reporag.ingestion.cloner import RepoCloner
cloner = RepoCloner()
manifest = cloner.clone_and_discover('https://github.com/pallets/click', branch='main')
print(f'Found {len(manifest)} files')
print(manifest[:5])
"
```

**Acceptance Criteria**:
- [ ] Clones public repos via HTTPS
- [ ] Supports branch selection (default: main/master auto-detect)
- [ ] Shallow clone option reduces clone time by 5x+
- [ ] File discovery filters by configurable extensions
- [ ] Manifest includes file path, language, size for each file
- [ ] Temp directory is cleaned up on error

**Branch**: `feature/issue-5-cloner`
**Dependencies**: Issue 4

---

### Issue 6 - Tree-sitter AST parser for Python

**Why**: tree-sitter provides fast, incremental, error-tolerant parsing across
many languages. Starting with Python; the same interface will support JS/TS later.
Unlike Python's ast module, tree-sitter preserves comments and whitespace positions.

**What to build**:
- Parse a Python source file into a tree-sitter AST
- Walk the AST and return structured node data (type, text, start/end lines)
- Handle parse errors gracefully (partial ASTs for broken files)
- Language-agnostic interface: Parser.parse(source, language) -> Tree

**Files to create**:
- src/reporag/ingestion/parser.py

**How to test locally**:
```bash
python -c "
from src.reporag.ingestion.parser import ASTParser
parser = ASTParser()
tree = parser.parse('def hello():\n    return 42\n', language='python')
print(tree.root_node.children)
"
```

**Acceptance Criteria**:
- [ ] Parses valid Python files into tree-sitter AST
- [ ] Returns partial AST for files with syntax errors (not crash)
- [ ] Parser interface is language-agnostic (parser.parse(source, lang))
- [ ] Node data includes: type, text, start_line, end_line, start_col, end_col
- [ ] Unit tests: empty file, single function, class with methods, syntax error

**Branch**: `feature/issue-6-ast-parser`
**Dependencies**: Issue 5

---

### Issue 7 - Symbol extractor: functions, classes, methods, imports

**Why**: Raw ASTs are too granular. The symbol extractor walks the AST and produces
a structured inventory of meaningful code entities that become nodes in the knowledge
graph and units for embedding.

**What to build**:
- Extract from AST: functions, classes, methods, module-level imports
- For each symbol: name, type, file_path, start_line, end_line, signature,
  docstring, decorators, parent_class (if method), return_type_hint
- Handle: nested functions, static/class methods, async functions, property decorators

**Files to create**:
- src/reporag/ingestion/symbol_extractor.py

**How to test locally**:
```bash
python -c "
from src.reporag.ingestion.symbol_extractor import SymbolExtractor
extractor = SymbolExtractor()
symbols = extractor.extract_from_file('examples/sample_repo/app.py')
for s in symbols:
    print(f'{s.type}: {s.name} [{s.start_line}-{s.end_line}]')
"
```

**Acceptance Criteria**:
- [ ] Extracts functions with: name, signature, docstring, decorators, line range
- [ ] Extracts classes with: name, bases, methods, docstring, line range
- [ ] Extracts imports: import X, from X import Y, from X import *
- [ ] Handles nested functions, async functions, property decorators
- [ ] Returns structured Symbol dataclass objects
- [ ] Unit tests cover all symbol types

**Branch**: `feature/issue-7-symbol-extractor`
**Dependencies**: Issue 6

---

### Issue 8 - Semantic code chunker: AST-aware, respects scope boundaries

**Why**: Naive text chunking splits functions mid-body, breaking semantic coherence.
AST-aware chunking uses the parse tree to produce chunks that respect function/class
boundaries, making embeddings more meaningful.

**What to build**:
- Chunk code at AST-node boundaries (functions, classes, top-level blocks)
- Configurable max chunk size (tokens); split large functions at logical points
- Each chunk carries metadata: file_path, start_line, end_line, parent_symbol, language
- Overlap strategy: include function signature in continuation chunks

**Files to create**:
- src/reporag/ingestion/chunker.py

**How to test locally**:
```bash
python -c "
from src.reporag.ingestion.chunker import SemanticChunker
chunker = SemanticChunker(max_tokens=512)
chunks = chunker.chunk_file('examples/sample_repo/app.py')
for c in chunks:
    print(f'[{c.start_line}-{c.end_line}] {c.parent_symbol} ({c.token_count} tokens)')
"
```

**Acceptance Criteria**:
- [ ] Never splits a function/class mid-body (unless exceeds max_tokens)
- [ ] Large functions are split at logical points with signature overlap
- [ ] Each chunk has metadata: file, lines, parent symbol, language, token count
- [ ] Chunk sizes stay within configurable max_tokens +/- 10%
- [ ] Unit tests: small function (1 chunk), large class (multiple chunks), module-level code

**Branch**: `feature/issue-8-chunker`
**Dependencies**: Issue 7

---

## M3: Code Knowledge Graph Construction

### Issue 9 - Call graph builder from AST

**Why**: The call graph captures which functions call which other functions. This
is essential for answering "how does X work end-to-end?" and "what breaks if I
change Y?" -- questions that pure vector search cannot answer.

**What to build**:
- Walk ASTs to identify function call expressions
- Resolve calls to their target symbols (same file, cross-file via imports)
- Build directed edges: caller -> callee with metadata (call site line number)
- Handle: method calls (self.method), chained calls, constructor calls

**Files to create**:
- src/reporag/graph/call_graph.py

**How to test locally**:
```bash
python -c "
from src.reporag.graph.call_graph import CallGraphBuilder
builder = CallGraphBuilder()
edges = builder.build_from_symbols(symbols, file_asts)
for e in edges[:10]:
    print(f'{e.caller} -> {e.callee} (line {e.call_site_line})')
"
```

**Acceptance Criteria**:
- [ ] Identifies direct function calls and resolves to target symbol
- [ ] Handles method calls (self.method, obj.method)
- [ ] Handles cross-file calls via import resolution
- [ ] Edge metadata includes: caller, callee, call site file + line
- [ ] Unit tests: direct call, method call, cross-file call, recursive call

**Branch**: `feature/issue-9-call-graph`
**Dependencies**: Issue 7

---

### Issue 10 - Import dependency graph builder

**Why**: The import graph captures module-level dependencies. Combined with the call
graph, it enables answering "which modules depend on this one?" and "what is the
import chain from A to B?"

**What to build**:
- Build directed edges: importing_module -> imported_module
- Resolve relative imports to absolute module paths
- Handle: import X, from X import Y, from .sibling import Z, from X import *
- Detect circular imports and flag them

**Files to create**:
- src/reporag/graph/dependency_graph.py

**How to test locally**:
```bash
python -c "
from src.reporag.graph.dependency_graph import DependencyGraphBuilder
builder = DependencyGraphBuilder()
edges = builder.build(symbols_by_file)
for e in edges[:10]:
    print(f'{e.source_module} -> {e.target_module} ({e.import_type})')
"
```

**Acceptance Criteria**:
- [ ] Resolves absolute imports (import os, from flask import Flask)
- [ ] Resolves relative imports (from .utils import helper)
- [ ] Handles from X import * with warning flag
- [ ] Detects circular import chains
- [ ] Edge metadata: source module, target module, import type, imported names

**Branch**: `feature/issue-10-dependency-graph`
**Dependencies**: Issue 7

---

### Issue 11 - Symbol table / global registry with metadata

**Why**: The symbol table is the central lookup index. Given a symbol name, it returns
the defining file, line range, type, and signature. The call graph and dependency graph
reference symbols by ID; the symbol table resolves them.

**What to build**:
- Global registry: symbol_id -> SymbolRecord (file, line, type, signature, docstring)
- Disambiguate same-name symbols across files/classes (fully qualified names)
- Lookup by: exact name, fully qualified name, regex pattern, file path
- Serializable to JSON for persistence and debugging

**Files to create**:
- src/reporag/graph/symbol_table.py

**How to test locally**:
```bash
python -c "
from src.reporag.graph.symbol_table import SymbolTable
table = SymbolTable()
table.register_symbols(all_symbols)
results = table.lookup('authenticate')
for r in results:
    print(f'{r.qualified_name} @ {r.file_path}:{r.start_line}')
"
```

**Acceptance Criteria**:
- [ ] Registers all symbols with fully qualified names (module.class.method)
- [ ] Lookup by exact name returns all matches across files
- [ ] Lookup by qualified name returns unique match
- [ ] Regex lookup works (e.g., `test_.*` finds all test functions)
- [ ] Serializes to/from JSON
- [ ] Unit tests: name collision across files, nested class methods, module-level vars

**Branch**: `feature/issue-11-symbol-table`
**Dependencies**: Issue 7

---

### Issue 12 - Neo4j graph store with Cypher query layer

**Why**: Neo4j provides persistent, indexed graph storage with the Cypher query
language. It enables complex traversal queries that NetworkX cannot efficiently
handle at scale (e.g., "shortest path from module A to module B through call edges").

**What to build**:
- Neo4j driver wrapper: connect, create nodes/edges, query, clear
- Node types: Function, Class, Module with properties from symbol table
- Edge types: CALLS, IMPORTS, INHERITS, CONTAINS (class -> method)
- Cypher query helpers: neighbors, shortest path, subgraph extraction
- NetworkX fallback implementing the same interface for testing

**Files to create**:
- src/reporag/graph/neo4j_store.py

**How to test locally**:
```bash
# With Neo4j running (docker-compose up neo4j)
python -c "
from src.reporag.graph.neo4j_store import GraphStore
store = GraphStore(uri='bolt://localhost:7687')
store.persist_graph(call_edges, dep_edges, symbols)
result = store.query('MATCH (f:Function)-[:CALLS]->(g:Function) RETURN f.name, g.name LIMIT 5')
print(result)
"
```

**Acceptance Criteria**:
- [ ] Creates nodes with correct labels and properties
- [ ] Creates CALLS, IMPORTS, INHERITS, CONTAINS edges
- [ ] Cypher queries return correct results (neighbors, paths)
- [ ] NetworkX fallback passes same test suite without Neo4j
- [ ] Bulk insert handles 10K+ nodes efficiently (batch transactions)
- [ ] Connection errors are handled gracefully with retry

**Branch**: `feature/issue-12-neo4j-store`
**Dependencies**: Issues 9, 10, 11

---

## M4: Code Embedding Pipeline & Hybrid Index

### Issue 13 - Code embedding pipeline: CodeBERT/UniXcoder

**Why**: Code-specific embedding models understand programming language semantics
(variable naming, control flow patterns, API usage) that general-purpose text
embedders miss. This produces better retrieval for code queries.

**What to build**:
- Load CodeBERT or UniXcoder from Hugging Face
- Embed code chunks in batches with GPU support (falls back to CPU)
- Normalize embeddings to unit vectors for cosine similarity
- Cache embeddings to avoid re-computation on re-index

**Files to create**:
- src/reporag/embedding/code_embedder.py

**How to test locally**:
```bash
python -c "
from src.reporag.embedding.code_embedder import CodeEmbedder
embedder = CodeEmbedder(model_name='microsoft/unixcoder-base')
vectors = embedder.embed_batch(['def hello(): return 42', 'class Foo: pass'])
print(f'Shape: {vectors.shape}')  # (2, 768)
print(f'Norm: {vectors[0].dot(vectors[0]):.4f}')  # ~1.0
"
```

**Acceptance Criteria**:
- [ ] Produces 768-dim vectors from code strings
- [ ] Batch embedding with configurable batch size
- [ ] GPU support with automatic CPU fallback
- [ ] Embeddings are L2-normalized
- [ ] Embedding cache avoids re-computation for unchanged chunks
- [ ] Unit test: similar code produces high cosine similarity (> 0.8)

**Branch**: `feature/issue-13-code-embedder`
**Dependencies**: Issue 8

---

### Issue 14 - Docstring and comment embedding pipeline

**Why**: Docstrings and comments describe intent in natural language. Embedding them
separately allows queries phrased in natural language ("how does auth work?") to
match documentation even when the code itself uses different terminology.

**What to build**:
- Load sentence-transformers model (all-MiniLM-L6-v2 or similar)
- Extract and embed: function docstrings, class docstrings, inline comments, README sections
- Link each doc embedding back to its parent code symbol
- Batch processing with progress tracking

**Files to create**:
- src/reporag/embedding/doc_embedder.py

**How to test locally**:
```bash
python -c "
from src.reporag.embedding.doc_embedder import DocEmbedder
embedder = DocEmbedder()
vectors = embedder.embed_batch(['Authenticate user with JWT token', 'Parse request body'])
print(f'Shape: {vectors.shape}')  # (2, 384)
"
```

**Acceptance Criteria**:
- [ ] Produces vectors from natural language text (docstrings, comments)
- [ ] Each doc embedding links to parent symbol ID
- [ ] Handles empty docstrings gracefully (skip, do not embed empty string)
- [ ] Batch processing with progress callback
- [ ] Unit test: "authentication" query is close to "verify JWT token" docstring

**Branch**: `feature/issue-14-doc-embedder`
**Dependencies**: Issue 8

---

### Issue 15 - Hybrid index builder: Qdrant vector + BM25 sparse

**Why**: Vector search excels at semantic similarity; BM25 excels at exact identifier
matching. Building both indices enables hybrid retrieval that outperforms either alone.

**What to build**:
- Create Qdrant collection with payload schema (file, lines, symbol, language)
- Upsert code + doc embeddings with metadata payloads
- Build BM25 index from tokenized code (split on camelCase, snake_case, operators)
- Code-aware tokenizer that handles identifiers (split `getUserById` -> `get user by id`)

**Files to create**:
- src/reporag/embedding/index_builder.py

**How to test locally**:
```bash
python -c "
from src.reporag.embedding.index_builder import IndexBuilder
builder = IndexBuilder(qdrant_url='localhost:6333')
builder.build_vector_index(chunks, code_embeddings, doc_embeddings)
builder.build_bm25_index(chunks)
print(f'Vector index: {builder.vector_count()} points')
print(f'BM25 index: {builder.bm25_doc_count()} documents')
"
```

**Acceptance Criteria**:
- [ ] Qdrant collection created with correct vector size and payload schema
- [ ] All code + doc embeddings upserted with metadata
- [ ] BM25 index built with code-aware tokenization
- [ ] Code tokenizer splits camelCase and snake_case identifiers
- [ ] Index supports incremental updates (add new files without full rebuild)
- [ ] Unit test: search "authenticate" returns auth-related chunks in both indices

**Branch**: `feature/issue-15-index-builder`
**Dependencies**: Issues 13, 14

---

## M5: Hybrid Retrieval Engine

### Issue 16 - Vector semantic search with configurable top-k

**Why**: Vector search finds semantically similar code even when the exact terms differ.
"How does the app handle unauthorized requests?" should match code that checks tokens
even if the word "unauthorized" never appears in the source.

**What to build**:
- Query Qdrant with an embedded query vector
- Return top-k results with scores, payloads (file, lines, symbol)
- Support filtering by: language, file path glob, symbol type
- Separate search over code embeddings and doc embeddings, merge results

**Files to create**:
- src/reporag/retrieval/vector_search.py

**How to test locally**:
```bash
python -c "
from src.reporag.retrieval.vector_search import VectorSearch
searcher = VectorSearch(qdrant_url='localhost:6333')
results = searcher.search('authentication middleware', top_k=10)
for r in results:
    print(f'{r.score:.3f} | {r.file_path}:{r.start_line} | {r.symbol_name}')
"
```

**Acceptance Criteria**:
- [ ] Returns top-k results ranked by cosine similarity
- [ ] Payloads include file_path, start_line, end_line, symbol_name, chunk_text
- [ ] Filtering by language and file path works
- [ ] Searches code and doc collections, merges results
- [ ] Latency under 100ms for top-20 search
- [ ] Unit test with known-good query-result pair

**Branch**: `feature/issue-16-vector-search`
**Dependencies**: Issue 15

---

### Issue 17 - BM25 sparse keyword search for code identifiers

**Why**: Vector search can miss exact identifier matches. BM25 ensures that searching
for "def authenticate_user" finds that exact function, even if the embedding model
does not surface it in top-k.

**What to build**:
- Query the BM25 index with a tokenized query
- Code-aware query tokenization (same tokenizer as indexing)
- Return top-k results with BM25 scores
- Support boosting exact function/class name matches

**Files to create**:
- src/reporag/retrieval/bm25_search.py

**How to test locally**:
```bash
python -c "
from src.reporag.retrieval.bm25_search import BM25Search
searcher = BM25Search()
searcher.load_index()
results = searcher.search('authenticate_user', top_k=10)
for r in results:
    print(f'{r.score:.3f} | {r.file_path}:{r.start_line} | {r.symbol_name}')
"
```

**Acceptance Criteria**:
- [ ] Exact identifier queries return the defining function as top-1
- [ ] Code-aware tokenization matches indexing tokenizer
- [ ] Boosting exact name matches works (configurable boost factor)
- [ ] Returns same result schema as vector search
- [ ] Unit test: "authenticate_user" returns the function, not just mentions

**Branch**: `feature/issue-17-bm25-search`
**Dependencies**: Issue 15

---

### Issue 18 - Graph-based retrieval: neighbor traversal, path queries

**Why**: Some questions require structural knowledge: "What functions call
authenticate_user?" or "Trace the request from router to database." Graph traversal
answers these directly; vector search cannot.

**What to build**:
- Given a symbol, return its N-hop neighbors in the call/dependency graph
- Path query: find paths between two symbols (shortest, all up to depth N)
- Subgraph extraction: return the induced subgraph around a set of symbols
- Convert graph results to the same result schema as vector/BM25 search

**Files to create**:
- src/reporag/retrieval/graph_traversal.py

**How to test locally**:
```bash
python -c "
from src.reporag.retrieval.graph_traversal import GraphRetriever
retriever = GraphRetriever(neo4j_uri='bolt://localhost:7687')
# Who calls authenticate_user?
callers = retriever.get_callers('authenticate_user', depth=2)
for c in callers:
    print(f'{c.symbol_name} -> authenticate_user (depth {c.depth})')
# Path from router to database
paths = retriever.find_paths('handle_request', 'execute_query', max_depth=5)
for p in paths:
    print(' -> '.join(p.symbols))
"
```

**Acceptance Criteria**:
- [ ] N-hop neighbor query returns correct callers/callees
- [ ] Path query finds shortest path between two symbols
- [ ] Subgraph extraction returns the relevant neighborhood
- [ ] Results converted to common RetrievalResult schema
- [ ] Falls back to NetworkX if Neo4j is unavailable
- [ ] Unit tests with known graph topology

**Branch**: `feature/issue-18-graph-retrieval`
**Dependencies**: Issue 12

---

### Issue 19 - Reciprocal Rank Fusion + cross-encoder reranker

**Why**: Each retrieval method returns scores on different scales. RRF normalizes
and fuses rankings without needing score calibration. The cross-encoder then reranks
the top candidates with a more powerful model that sees query + document together.

**What to build**:
- RRF fusion: take ranked lists from vector, BM25, graph; produce fused ranking
- Configurable RRF constant k (default 60)
- Cross-encoder reranker: score (query, chunk) pairs with cross-encoder model
- Return final top-k with reranked scores

**Files to create**:
- src/reporag/retrieval/fusion.py
- src/reporag/retrieval/reranker.py

**How to test locally**:
```bash
python -c "
from src.reporag.retrieval.fusion import reciprocal_rank_fusion
from src.reporag.retrieval.reranker import CrossEncoderReranker

# Fuse three ranked lists
fused = reciprocal_rank_fusion([vector_results, bm25_results, graph_results], k=60)
print(f'Fused: {len(fused)} results')

# Rerank top candidates
reranker = CrossEncoderReranker(model='cross-encoder/ms-marco-MiniLM-L-6-v2')
reranked = reranker.rerank(query='auth flow', candidates=fused[:20])
for r in reranked[:5]:
    print(f'{r.rerank_score:.3f} | {r.file_path}:{r.start_line}')
"
```

**Acceptance Criteria**:
- [ ] RRF correctly fuses 2-3 ranked lists into a single ranking
- [ ] RRF handles missing items (item in one list but not another)
- [ ] Cross-encoder reranks candidates and reorders by rerank score
- [ ] Reranked results measurably outperform RRF-only (eval on test queries)
- [ ] Reranking latency under 500ms for 20 candidates
- [ ] Unit tests: known rankings produce expected fusion order

**Branch**: `feature/issue-19-fusion-reranker`
**Dependencies**: Issues 16, 17, 18

---

## M6: Agentic Query Planner & Multi-Hop Decomposition

### Issue 20 - Query classifier: simple lookup vs. multi-hop vs. exploratory

**Why**: Not every query needs the full agentic pipeline. Simple lookups ("where is
function X defined?") should go straight to BM25/graph. Multi-hop queries need
decomposition. Exploratory queries ("explain the architecture") need broad retrieval.
Classifying first saves latency and cost.

**What to build**:
- LLM-based query classifier with few-shot examples
- Categories: simple-lookup, multi-hop, exploratory
- Confidence score for each classification
- Fallback: if confidence < threshold, treat as multi-hop (safest default)

**Files to create**:
- src/reporag/agent/planner.py (query_classifier section)

**How to test locally**:
```bash
python -c "
from src.reporag.agent.planner import QueryClassifier
classifier = QueryClassifier()
result = classifier.classify('Where is the authenticate function defined?')
print(f'Type: {result.query_type}, Confidence: {result.confidence:.2f}')
result = classifier.classify('How does the auth flow work end-to-end?')
print(f'Type: {result.query_type}, Confidence: {result.confidence:.2f}')
"
```

**Acceptance Criteria**:
- [ ] Classifies "where is X defined?" as simple-lookup
- [ ] Classifies "how does X work end-to-end?" as multi-hop
- [ ] Classifies "explain the architecture" as exploratory
- [ ] Returns confidence score (0-1)
- [ ] Low-confidence falls back to multi-hop
- [ ] Unit tests with 10+ example queries

**Branch**: `feature/issue-20-query-classifier`
**Dependencies**: Issue 19

---

### Issue 21 - Agentic query decomposer: break complex queries into sub-queries

**Why**: Multi-hop questions like "How does a request go from the API endpoint to the
database?" require multiple retrieval steps. The decomposer breaks this into ordered
sub-queries that each retrieve one piece of the answer.

**What to build**:
- LLM-based decomposition using LangGraph state machine
- Input: complex query + repo context (available modules, key symbols)
- Output: ordered list of sub-queries with dependency edges
- Each sub-query includes: text, expected_answer_type (code/explanation/list), context_from (prior sub-query IDs)

**Files to create**:
- src/reporag/agent/planner.py (decomposer section)

**How to test locally**:
```bash
python -c "
from src.reporag.agent.planner import QueryDecomposer
decomposer = QueryDecomposer()
plan = decomposer.decompose(
    'How does a request go from the API endpoint to the database?',
    repo_context={'modules': ['api', 'routes', 'db', 'models']}
)
for step in plan.steps:
    print(f'Step {step.id}: {step.query} (depends on: {step.depends_on})')
"
```

**Acceptance Criteria**:
- [ ] Decomposes multi-hop queries into 2-5 ordered sub-queries
- [ ] Sub-queries have dependency edges (step 2 depends on step 1)
- [ ] Uses repo context (module names, key symbols) to inform decomposition
- [ ] Handles edge case: query that does not need decomposition (returns single step)
- [ ] LangGraph state machine with clear state transitions
- [ ] Unit tests with 5+ multi-hop query examples

**Branch**: `feature/issue-21-query-decomposer`
**Dependencies**: Issue 20

---

### Issue 22 - Strategy router: graph vs. vector vs. hybrid per sub-query

**Why**: Different sub-queries benefit from different retrieval strategies. "Find the
authenticate function" is best served by BM25. "What does the auth middleware do?" is
best served by vector search. "What calls authenticate?" is best served by graph traversal.

**What to build**:
- Route each sub-query to: graph, vector, bm25, or hybrid (combination)
- Routing based on sub-query characteristics (identifier mention -> BM25, structural -> graph, semantic -> vector)
- LLM-assisted routing with rule-based fallback
- Execute routed sub-queries via the retrieval engine

**Files to create**:
- src/reporag/agent/router.py
- src/reporag/agent/executor.py

**How to test locally**:
```bash
python -c "
from src.reporag.agent.router import StrategyRouter
from src.reporag.agent.executor import SubQueryExecutor
router = StrategyRouter()
strategy = router.route('What functions call authenticate_user?')
print(f'Strategy: {strategy}')  # graph
executor = SubQueryExecutor(retrieval_engine)
results = executor.execute(plan.steps)
for step_id, step_results in results.items():
    print(f'Step {step_id}: {len(step_results)} results')
"
```

**Acceptance Criteria**:
- [ ] Routes identifier lookups to BM25
- [ ] Routes structural queries ("what calls X") to graph
- [ ] Routes semantic queries to vector search
- [ ] Hybrid route runs multiple strategies and fuses
- [ ] Executor runs sub-queries in dependency order, passing context forward
- [ ] Unit tests: 10+ sub-queries with expected routing decisions

**Branch**: `feature/issue-22-router-executor`
**Dependencies**: Issues 20, 21

---

## M7: Answer Generation, Citation & FastAPI Serving

### Issue 23 - Context assembler: retrieved code -> structured prompt context

**Why**: Raw retrieval results are unordered chunks. The context assembler orders them
by file, deduplicates overlapping chunks, and formats them into a structured context
block that the LLM can reason over effectively.

**What to build**:
- Order chunks by: file path, then line number (reading order)
- Deduplicate overlapping chunks (merge if overlap > 50%)
- Format: file header + line-numbered code blocks
- Truncate to fit context window with priority: highest-ranked chunks first

**Files to create**:
- src/reporag/generation/context_assembler.py

**How to test locally**:
```bash
python -c "
from src.reporag.generation.context_assembler import ContextAssembler
assembler = ContextAssembler(max_tokens=4000)
context = assembler.assemble(retrieval_results)
print(context[:500])
"
```

**Acceptance Criteria**:
- [ ] Chunks ordered by file path, then line number
- [ ] Overlapping chunks merged (no duplicate code in context)
- [ ] Each chunk prefixed with file path and line range
- [ ] Total tokens stay within configurable max_tokens
- [ ] Highest-ranked chunks prioritized when truncating
- [ ] Unit test: overlapping chunks produce clean, non-redundant output

**Branch**: `feature/issue-23-context-assembler`
**Dependencies**: Issue 19

---

### Issue 24 - Prompt builder with code-aware templates

**Why**: The prompt template determines answer quality. Code-aware templates include
file structure context, citation format instructions, and examples of good code
explanations. Different query types need different templates.

**What to build**:
- Template per query type: simple-lookup, multi-hop, exploratory
- Include: system prompt, code context block, query, citation format instructions
- Citation format: `[file_path:start_line-end_line]`
- Few-shot examples of good cited answers

**Files to create**:
- src/reporag/generation/prompt_builder.py

**How to test locally**:
```bash
python -c "
from src.reporag.generation.prompt_builder import PromptBuilder
builder = PromptBuilder()
prompt = builder.build(
    query='How does auth work?',
    query_type='multi-hop',
    context=assembled_context,
    sub_query_answers=prior_answers
)
print(prompt[:1000])
"
```

**Acceptance Criteria**:
- [ ] Templates for all 3 query types
- [ ] Citation format clearly instructed in system prompt
- [ ] Few-shot examples included for multi-hop and exploratory
- [ ] Sub-query answers from prior steps injected for multi-hop
- [ ] Total prompt fits within model context window
- [ ] Unit test: generated prompt contains all expected sections

**Branch**: `feature/issue-24-prompt-builder`
**Dependencies**: Issue 23

---

### Issue 25 - LLM generation with line-level citation extraction

**Why**: The generator calls the LLM and the citation extractor parses the response
to identify file:line references, validating them against the actual retrieved context
to ensure citations are real.

**What to build**:
- Call LLM (OpenAI / Anthropic, configurable) with the built prompt
- Parse response for citation markers `[file:line-line]`
- Validate each citation: does the referenced code exist in the context?
- Return structured response: {answer_text, citations: [{file, start, end, snippet}]}

**Files to create**:
- src/reporag/generation/generator.py
- src/reporag/generation/citation.py

**How to test locally**:
```bash
python -c "
from src.reporag.generation.generator import AnswerGenerator
gen = AnswerGenerator(provider='openai', model='gpt-4o')
result = gen.generate(prompt, context_chunks)
print(result.answer_text[:500])
for c in result.citations:
    print(f'  [{c.file_path}:{c.start_line}-{c.end_line}]')
"
```

**Acceptance Criteria**:
- [ ] Calls OpenAI or Anthropic API (configurable via env)
- [ ] Extracts citation markers from response text
- [ ] Validates citations against retrieved context (flags invalid ones)
- [ ] Returns structured result with answer + validated citations
- [ ] Handles LLM errors gracefully (timeout, rate limit, invalid response)
- [ ] Citation coverage: >= 90% of claims have at least one citation

**Branch**: `feature/issue-25-generator-citation`
**Dependencies**: Issue 24

---

### Issue 26 - FastAPI application: repo, query, and health endpoints

**Why**: The API layer exposes the pipeline over HTTP. Repo endpoints trigger
ingestion; query endpoints run the full retrieval + generation pipeline; health
endpoints enable monitoring.

**What to build**:
- POST /api/v1/repos/ingest: accepts repo_url, branch; triggers async ingestion
- GET /api/v1/repos: list ingested repos with status
- POST /api/v1/query: accepts question, repo_id; returns cited answer
- GET /api/v1/health: returns pipeline component status
- OpenAPI docs auto-generated

**Files to create**:
- src/reporag/api/main.py
- src/reporag/api/routes/repos.py
- src/reporag/api/routes/query.py
- src/reporag/api/routes/health.py

**How to test locally**:
```bash
uvicorn src.reporag.api.main:app --reload --port 8000
# In another terminal:
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/docs  # OpenAPI UI
```

**Acceptance Criteria**:
- [ ] POST /repos/ingest triggers async background ingestion
- [ ] GET /repos returns list with status (queued/indexing/ready/error)
- [ ] POST /query returns {answer, citations, metadata}
- [ ] GET /health returns status of each component (neo4j, qdrant, llm)
- [ ] OpenAPI docs available at /docs
- [ ] Request validation with Pydantic models, clear error responses

**Branch**: `feature/issue-26-fastapi`
**Dependencies**: Issue 25

---

## M8: Google OAuth, JWT Auth & API Middleware

### Issue 27 - Google OAuth 2.0 login flow

**Why**: NST standard auth pattern. Google OAuth provides secure, passwordless
authentication. Users log in with their Google account; the API receives a verified
email and profile.

**What to build**:
- GET /auth/google: redirect to Google OAuth consent screen
- GET /auth/google/callback: exchange code for tokens, extract user info
- Create or update User record in database
- Return JWT access token + refresh token

**Files to create/update**:
- src/reporag/api/routes/auth.py (new)
- src/reporag/api/middleware/auth.py

**How to test locally**:
```bash
# Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env
# Open browser:
open http://localhost:8000/auth/google
# After consent, callback returns JWT
```

**Acceptance Criteria**:
- [ ] Redirect to Google OAuth consent screen works
- [ ] Callback exchanges code for Google access token
- [ ] Extracts email + profile from Google userinfo endpoint
- [ ] Creates User record on first login, updates on subsequent
- [ ] Returns JWT access + refresh tokens
- [ ] Handles OAuth errors (denied consent, invalid code)

**Branch**: `feature/issue-27-google-oauth`
**Dependencies**: Issue 26

---

### Issue 28 - JWT token issuance, validation, refresh middleware

**Why**: JWT enables stateless authentication. The middleware validates tokens on every
request, extracts the user, and makes it available to route handlers.

**What to build**:
- JWT issuance: encode user_id, email, roles, expiry
- Validation middleware: decode + verify on every protected route
- Refresh endpoint: issue new access token from valid refresh token
- Dependency injection: `current_user = Depends(get_current_user)`

**Files to create/update**:
- src/reporag/api/middleware/auth.py

**How to test locally**:
```bash
# Get a token from OAuth flow, then:
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/repos
# Expired token should return 401
curl -X POST http://localhost:8000/auth/refresh -d '{"refresh_token": "..."}'
```

**Acceptance Criteria**:
- [ ] JWT contains: user_id, email, exp, iat
- [ ] Protected routes return 401 without valid token
- [ ] Expired tokens return 401 with "token expired" message
- [ ] Refresh endpoint issues new access token
- [ ] Refresh tokens have longer expiry (7 days default)
- [ ] get_current_user dependency injectable in any route

**Branch**: `feature/issue-28-jwt-middleware`
**Dependencies**: Issue 27

---

### Issue 29 - Rate limiting, request logging, error handling middleware

**Why**: Production APIs need rate limiting (prevent abuse), structured logging
(debuggability), and consistent error handling (good DX for consumers).

**What to build**:
- Rate limiter: per-user, configurable (default 60 req/min), returns 429
- Structured JSON logging: method, path, status, latency_ms, user_id, request_id
- Global exception handler: all errors return {error, detail, status_code}
- Request ID middleware: generates UUID per request, includes in response headers

**Files to create/update**:
- src/reporag/api/middleware/rate_limiter.py (new)
- src/reporag/api/middleware/logging.py (new)
- src/reporag/api/middleware/error_handler.py (new)
- src/reporag/api/main.py (register middleware)

**How to test locally**:
```bash
# Rapid-fire requests:
for i in $(seq 1 65); do curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/api/v1/health; done
# Should see 429 after 60 requests
# Check logs:
docker-compose logs api | jq .
```

**Acceptance Criteria**:
- [ ] Rate limiter returns 429 after exceeding limit
- [ ] Per-user rate limiting (keyed by user_id from JWT)
- [ ] JSON structured logs for every request
- [ ] Request ID in response header (X-Request-ID)
- [ ] Unhandled exceptions return clean JSON error, not stack trace
- [ ] Unit test: rate limiter triggers at configured threshold

**Branch**: `feature/issue-29-middleware`
**Dependencies**: Issue 28

---

## M9: React Frontend -- Code Explorer & Q&A Interface

### Issue 30 - React + Vite scaffold, routing, auth flow (Google login)

**Why**: The frontend gives users a visual interface to explore repos, ask questions,
and see cited answers. Google login matches the backend OAuth flow.

**What to build**:
- React 18 + Vite + TailwindCSS scaffold
- React Router with routes: /, /login, /repos, /repos/:id, /query
- Google login button that redirects to backend OAuth endpoint
- JWT storage in memory (not localStorage), auto-attach to API requests
- Auth context provider, protected route wrapper

**Files to create**:
- frontend/package.json
- frontend/vite.config.js
- frontend/tailwind.config.js
- frontend/src/App.jsx
- frontend/src/main.jsx
- frontend/src/pages/Login.jsx
- frontend/src/pages/Dashboard.jsx
- frontend/src/context/AuthContext.jsx

**How to test locally**:
```bash
cd frontend && npm install && npm run dev
# Open http://localhost:5173
# Click "Login with Google" -> redirects to backend OAuth
```

**Acceptance Criteria**:
- [ ] Vite dev server starts with hot reload
- [ ] TailwindCSS utility classes work
- [ ] Google login button redirects to /auth/google
- [ ] After OAuth callback, JWT stored in memory and attached to API calls
- [ ] Unauthenticated users redirected to /login
- [ ] Protected routes render only when authenticated

**Branch**: `feature/issue-30-frontend-scaffold`
**Dependencies**: Issue 28

---

### Issue 31 - Repository explorer: file tree + syntax-highlighted code viewer

**Why**: Users need to browse the ingested repository visually. The file tree provides
navigation; the code viewer shows syntax-highlighted source with line numbers that
correspond to citation references.

**What to build**:
- File tree component (collapsible directories, file icons by language)
- Code viewer with syntax highlighting (react-syntax-highlighter or Monaco)
- Line numbers that match the source (for citation cross-reference)
- Highlight specific line ranges (used by citation clicks)
- Fetch file contents from API

**Files to create**:
- frontend/src/pages/RepoExplorer.jsx
- frontend/src/components/FileTree.jsx
- frontend/src/components/CodeViewer.jsx

**How to test locally**:
```bash
# With backend running and a repo ingested:
# Navigate to http://localhost:5173/repos/<repo-id>
# Click files in tree -> code appears with syntax highlighting
```

**Acceptance Criteria**:
- [ ] File tree loads from API and renders collapsible directories
- [ ] Clicking a file shows syntax-highlighted code with line numbers
- [ ] Line range highlighting works (highlight lines 10-25)
- [ ] Supports Python, JavaScript, TypeScript syntax
- [ ] Handles large files without freezing (virtualized rendering if needed)

**Branch**: `feature/issue-31-repo-explorer`
**Dependencies**: Issue 30

---

### Issue 32 - Conversational Q&A interface with citation highlights

**Why**: The core user experience: ask a question about the repo, get an answer with
citations, and click citations to jump to the exact code.

**What to build**:
- Chat-style Q&A interface (input box, message history)
- Display answers with inline citation links [file:line-line]
- Click a citation -> navigate to RepoExplorer with that file + highlighted lines
- Loading state with streaming indicator
- Query history sidebar

**Files to create**:
- frontend/src/pages/QueryInterface.jsx
- frontend/src/components/QueryInput.jsx
- frontend/src/components/AnswerDisplay.jsx
- frontend/src/components/CitationLink.jsx
- frontend/src/components/SearchResults.jsx

**How to test locally**:
```bash
# Navigate to http://localhost:5173/query?repo=<repo-id>
# Type: "How does the auth flow work?"
# Answer appears with clickable citations
# Click citation -> jumps to code
```

**Acceptance Criteria**:
- [ ] Chat input sends query to POST /api/v1/query
- [ ] Answer renders with markdown formatting
- [ ] Citation links are clickable and styled distinctly
- [ ] Clicking citation navigates to code viewer with highlighted lines
- [ ] Loading indicator while waiting for response
- [ ] Query history persists during session

**Branch**: `feature/issue-32-qa-interface`
**Dependencies**: Issues 30, 31

---

### Issue 33 - Interactive graph visualizer: call graph / dependency graph

**Why**: Visualizing the code graph helps users understand architecture at a glance.
Seeing call chains and module dependencies as an interactive graph is far more intuitive
than reading Cypher query results.

**What to build**:
- Force-directed graph layout (react-force-graph or D3)
- Node types: functions (circles), classes (squares), modules (hexagons)
- Edge types: calls (solid), imports (dashed), inherits (dotted)
- Click node -> show symbol details + navigate to code
- Zoom, pan, filter by module, search for symbol

**Files to create**:
- frontend/src/components/GraphVisualizer.jsx
- frontend/src/components/RepoSelector.jsx

**How to test locally**:
```bash
# Navigate to http://localhost:5173/repos/<repo-id>/graph
# Interactive graph renders with zoom/pan
# Click a function node -> details panel + link to code
```

**Acceptance Criteria**:
- [ ] Graph renders with correct node/edge types and visual styling
- [ ] Force-directed layout produces readable graph for 100+ nodes
- [ ] Click node shows details panel (name, file, signature, docstring)
- [ ] Filter by module collapses/expands subgraphs
- [ ] Search highlights matching nodes
- [ ] Zoom + pan smooth on desktop

**Branch**: `feature/issue-33-graph-viz`
**Dependencies**: Issues 30, 31

---

## M10: Evaluation Harness, E2E Testing & Demo

### Issue 34 - Retrieval evaluation suite: context recall, precision, MRR

**Why**: Retrieval is the #1 failure point in RAG (73% of failures). Measuring
retrieval quality with standard IR metrics tells you exactly where the pipeline
is weak and guides tuning.

**What to build**:
- Metrics: context recall@k, precision@k, MRR (Mean Reciprocal Rank), NDCG
- Evaluation dataset: (query, relevant_chunks[]) pairs for sample repos
- Batch evaluation runner: compute metrics across all queries
- Comparison mode: evaluate vector-only vs. BM25-only vs. hybrid vs. hybrid+rerank

**Files to create**:
- src/reporag/evaluation/metrics.py
- src/reporag/evaluation/retrieval_eval.py (new)
- examples/eval_dataset.json (new)

**How to test locally**:
```bash
python -c "
from src.reporag.evaluation.retrieval_eval import RetrievalEvaluator
evaluator = RetrievalEvaluator()
results = evaluator.evaluate('examples/eval_dataset.json')
print(f'Recall@10: {results.recall_at_10:.3f}')
print(f'Precision@10: {results.precision_at_10:.3f}')
print(f'MRR: {results.mrr:.3f}')
"
```

**Acceptance Criteria**:
- [ ] Computes context recall@k, precision@k, MRR, NDCG correctly
- [ ] Evaluation dataset with 20+ query-relevance pairs
- [ ] Batch runner produces aggregate + per-query metrics
- [ ] Comparison mode shows delta between retrieval strategies
- [ ] Output as JSON and human-readable table
- [ ] Unit tests with known-answer metric calculations

**Branch**: `feature/issue-34-retrieval-eval`
**Dependencies**: Issue 19

---

### Issue 35 - Generation evaluation: faithfulness, relevance, hallucination

**Why**: Even with good retrieval, the LLM can hallucinate or produce irrelevant
answers. Generation metrics measure whether the answer is grounded in the retrieved
context and relevant to the query.

**What to build**:
- Faithfulness: is every claim in the answer supported by retrieved context? (LLM-as-judge)
- Answer relevance: does the answer address the query? (LLM-as-judge)
- Hallucination detection: identify claims not grounded in any retrieved chunk
- Citation coverage: percentage of claims with valid citations

**Files to create**:
- src/reporag/evaluation/generation_eval.py (new)

**How to test locally**:
```bash
python -c "
from src.reporag.evaluation.generation_eval import GenerationEvaluator
evaluator = GenerationEvaluator()
results = evaluator.evaluate(query, answer, retrieved_context, citations)
print(f'Faithfulness: {results.faithfulness:.3f}')
print(f'Relevance: {results.relevance:.3f}')
print(f'Hallucination rate: {results.hallucination_rate:.3f}')
print(f'Citation coverage: {results.citation_coverage:.3f}')
"
```

**Acceptance Criteria**:
- [ ] Faithfulness score between 0-1 using LLM-as-judge
- [ ] Answer relevance score between 0-1
- [ ] Hallucination detector flags unsupported claims
- [ ] Citation coverage computed as (cited claims / total claims)
- [ ] Configurable judge model (can use cheaper model for eval)
- [ ] Unit tests with known-faithful and known-hallucinated answers

**Branch**: `feature/issue-35-generation-eval`
**Dependencies**: Issue 25

---

### Issue 36 - End-to-end integration tests with sample repositories

**Why**: Unit tests verify components; integration tests verify the pipeline works
end-to-end. Using a sample repository with known structure ensures the full path
from ingestion to cited answer produces correct results.

**What to build**:
- Sample Python project in examples/sample_repo/ (5-10 files, known call graph)
- Integration test: clone -> parse -> graph -> index -> query -> verify citations
- Test queries with expected answers and citation file/line ranges
- CI-compatible: runs in Docker without external services

**Files to create**:
- examples/sample_repo/ (multiple .py files with known structure)
- examples/sample_queries.json
- tests/integration/test_e2e_pipeline.py
- tests/integration/test_ingestion_pipeline.py
- tests/integration/test_retrieval_pipeline.py

**How to test locally**:
```bash
pytest tests/integration/ -v --timeout=120
```

**Acceptance Criteria**:
- [ ] sample_repo has 5-10 Python files with functions, classes, imports, call chains
- [ ] Integration test: ingest sample_repo -> query -> answer has correct citations
- [ ] At least 5 test queries with expected results
- [ ] Tests pass in CI Docker environment (NetworkX fallback, no Neo4j required)
- [ ] Test timeout under 120 seconds total

**Branch**: `feature/issue-36-integration-tests`
**Dependencies**: Issues 26, 34, 35

---

### Issue 37 - Performance benchmarks, load testing, demo script

**Why**: The final milestone demonstrates production readiness: the system handles
concurrent load, maintains latency SLAs, and a polished demo script showcases the
full capability.

**What to build**:
- Benchmark script: measure ingestion throughput (files/sec), indexing time, query latency
- Load test: 50 concurrent queries via locust or similar, measure p50/p95/p99 latency
- Demo script: 5 progressively harder queries against a well-known open-source repo
- Results summary: table of benchmark numbers + comparison to baseline

**Files to create**:
- src/reporag/evaluation/benchmarks.py
- scripts/run_benchmarks.sh (new)
- scripts/demo.sh (new)
- src/reporag/evaluation/failure_analyzer.py

**How to test locally**:
```bash
# Run benchmarks
bash scripts/run_benchmarks.sh
# Run demo
bash scripts/demo.sh
# Load test
locust -f tests/load/locustfile.py --headless -u 50 -r 10 --run-time 60s
```

**Acceptance Criteria**:
- [ ] Benchmark script reports: ingestion files/sec, indexing time, avg query latency
- [ ] Load test: p95 latency under 5 seconds at 50 concurrent users
- [ ] Demo script runs 5 queries with commentary, shows cited answers
- [ ] Failure analyzer classifies failures as: retrieval / generation / context overflow
- [ ] Results written to JSON + printed as human-readable table
- [ ] Demo completes without errors on a fresh Docker setup

**Branch**: `feature/issue-37-benchmarks-demo`
**Dependencies**: Issue 36

---

NST Engineering | RepoRAG AI Issues Tracker | 2026
