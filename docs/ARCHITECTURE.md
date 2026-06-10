# Architecture - RepoRAG AI

## System Overview

RepoRAG has two separate pipelines: offline (ingestion) and online (query).

### Offline Pipeline (Ingestion)

```
Git URL
  -> RepoCloner (clone, discover files)
    -> ASTParser (tree-sitter parse each file)
      -> SymbolExtractor (functions, classes, imports)
        -> CallGraphBuilder + DependencyGraphBuilder
          -> Neo4j GraphStore (persist nodes + edges)
        -> SemanticChunker (AST-aware chunks)
          -> CodeEmbedder (CodeBERT/UniXcoder vectors)
          -> DocEmbedder (sentence-transformers vectors)
            -> IndexBuilder (Qdrant vector + BM25 sparse)
```

### Online Pipeline (Query)

```
User Query
  -> QueryClassifier (simple / multi-hop / exploratory)
    -> QueryDecomposer (break into sub-queries if multi-hop)
      -> StrategyRouter (graph / vector / bm25 / hybrid per sub-query)
        -> SubQueryExecutor
          -> VectorSearch (Qdrant)
          -> BM25Search (rank-bm25)
          -> GraphTraversal (Neo4j/NetworkX)
        -> ReciprocralRankFusion (merge results)
          -> CrossEncoderReranker (rerank top candidates)
            -> ContextAssembler (order, dedup, format)
              -> PromptBuilder (code-aware templates)
                -> AnswerGenerator (LLM call)
                  -> CitationExtractor (parse + validate citations)
                    -> API Response {answer, citations}
```

## Data Flow

See README.md pipeline diagram for the simplified version.

## Key Design Decisions

1. tree-sitter over Python ast: supports multiple languages, preserves comments
2. Neo4j over pure NetworkX: scalable graph queries, Cypher language
3. Hybrid retrieval (vector + BM25 + graph): each method has blind spots
4. RRF over learned fusion: no training data needed, works out of the box
5. Cross-encoder reranker: 2x accuracy gain at ~500ms latency cost
6. Agentic planner: multi-hop questions need decomposition, not bigger context
7. Offline/online split: ingestion is slow (minutes), queries must be fast (seconds)

## API Contract

See docs/API_CONTRACT.md for the full OpenAPI specification.
