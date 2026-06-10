# Runbook - RepoRAG AI

## Starting the Stack

```bash
docker-compose up -d
uvicorn src.reporag.api.main:app --reload --port 8000
cd frontend && npm run dev
```

## Common Operations

### Re-index a repository
```bash
curl -X POST http://localhost:8000/api/v1/repos/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"repo_url": "...", "force_reindex": true}'
```

### Clear Neo4j graph
```bash
docker exec -it reporag-ai-neo4j-1 cypher-shell -u neo4j -p reporag123 \
  "MATCH (n) DETACH DELETE n"
```

### Clear Qdrant collections
```bash
curl -X DELETE http://localhost:6333/collections/reporag_code
curl -X DELETE http://localhost:6333/collections/reporag_docs
```

## Troubleshooting

See DEVELOPMENT_GUIDE.md troubleshooting section.
