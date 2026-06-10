# API Contract - RepoRAG AI

Base URL: `http://localhost:8000/api/v1`

## Endpoints

### Health
- `GET /health` - Component status (neo4j, qdrant, llm, db)

### Authentication
- `GET /auth/google` - Redirect to Google OAuth consent
- `GET /auth/google/callback` - OAuth callback, returns JWT
- `POST /auth/refresh` - Refresh access token

### Repositories
- `POST /repos/ingest` - Trigger async repo ingestion
- `GET /repos` - List ingested repos with status
- `GET /repos/{repo_id}` - Repo details + file tree
- `GET /repos/{repo_id}/files/{file_path}` - File contents
- `GET /repos/{repo_id}/graph` - Knowledge graph data

### Query
- `POST /query` - Ask a question, get cited answer

## Request/Response Schemas

See OpenAPI docs at `/docs` when the server is running.

## Authentication

All endpoints except `/health` and `/auth/*` require:
`Authorization: Bearer <jwt_token>`
