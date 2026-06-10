# PROJECT_CONTEXT.md - RepoRAG AI (Internal)

## Project Summary

RepoRAG is a production-grade, code-aware RAG system that ingests Git repositories,
builds knowledge graphs + hybrid indices, and uses an agentic query planner to answer
complex multi-hop code questions with line-level citations.

## Pipeline

```
clone -> parse AST (tree-sitter) -> extract symbols -> build call graph + dep graph (Neo4j)
  -> embed code (CodeBERT) + docs (sentence-transformers) -> hybrid index (Qdrant + BM25)

query -> classify -> decompose (agentic) -> route (graph/vector/bm25/hybrid)
  -> retrieve -> fuse (RRF) -> rerank (cross-encoder) -> assemble context
  -> generate (LLM) -> extract citations -> return
```

## What Makes This Industry-Level

1. AST-aware parsing (not naive text chunking)
2. Code knowledge graph (call graph + dependency graph in Neo4j)
3. Triple hybrid retrieval (vector + BM25 + graph) with RRF fusion
4. Cross-encoder reranking
5. Agentic query decomposition for multi-hop questions
6. Line-level citation extraction with validation
7. Offline/online pipeline separation
8. Built-in eval harness (RAGAS metrics + failure analysis)

## Architecture Decisions

- tree-sitter: multi-language, error-tolerant, preserves comments (vs Python ast)
- Neo4j: scalable graph queries, Cypher (vs pure NetworkX)
- Qdrant: fast vector search, filtering, payloads (vs ChromaDB/Pinecone)
- LangGraph: state machine for agent planning (vs raw LangChain chains)
- RRF: training-free fusion (vs learned fusion models)
- Cross-encoder: ~2x accuracy over bi-encoder alone

## Milestone Plan

10 milestones, 37 issues, 10-week sprint timeline.
See MILESTONES.md for details.

## Pod Assignment Strategy

- Contributor 1 -> Ingestion (M2): cloner, parser, extractor, chunker
- Contributor 2 -> Graph + Embedding (M3, M4): call graph, dep graph, Neo4j, embedders
- Contributor 3 -> Retrieval + Agent (M5, M6): search, fusion, reranker, planner
- Contributor 4 -> Generation + API + Frontend (M7, M8, M9): generator, API, React

## Risk Areas

- tree-sitter setup can be tricky across OS (pre-built wheels help)
- Neo4j memory usage with large repos (tune heap settings)
- Embedding model download size (~500MB for CodeBERT)
- LLM API costs for agentic planner (use cheaper model for classification)
- Cross-encoder latency on large candidate sets (cap at 20 candidates)
