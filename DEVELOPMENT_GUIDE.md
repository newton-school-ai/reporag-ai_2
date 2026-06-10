# DEVELOPMENT_GUIDE.md - RepoRAG AI

## Prerequisites

- Python 3.11+ (`python --version`)
- Node.js 20+ (`node --version`)
- Docker + docker-compose (`docker --version`)
- Git (`git --version`)
- GitHub CLI (`gh --version`)
- pre-commit (`pip install pre-commit`)

## Initial Setup

```bash
# 1. Clone and enter
git clone git@github.com:newton-school-ai/reporag-ai.git
cd reporag-ai
git checkout dev

# 2. Python environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Pre-commit hooks
pre-commit install

# 4. Environment variables
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET

# 5. Start infrastructure
docker-compose up -d
# Verify:
docker-compose ps
# Expected: neo4j (7474, 7687), qdrant (6333, 6334), postgres (5432) all healthy

# 6. Run database migrations
alembic upgrade head

# 7. Start API server
uvicorn src.reporag.api.main:app --reload --port 8000
# Verify: curl http://localhost:8000/api/v1/health

# 8. Start frontend (separate terminal)
cd frontend
npm install
npm run dev
# Verify: open http://localhost:5173
```

## Daily Workflow

```bash
# Start of day
git checkout dev && git pull
source .venv/bin/activate
docker-compose up -d

# Start your feature
git checkout -b feature/issue-N-short-name

# ... write code ...

# Before committing
pre-commit run --all-files
pytest tests/unit/ -v

# Commit
git add .
git commit -m "[Issue N] Short description"

# Push and open PR
git push -u origin feature/issue-N-short-name
gh pr create --base dev --title "[Issue N] Short description" --body "Closes #N"
```

## Running Tests

```bash
# All unit tests
pytest tests/unit/ -v

# Specific test file
pytest tests/unit/test_parser.py -v

# Integration tests (requires Docker services)
pytest tests/integration/ -v --timeout=120

# With coverage
pytest tests/ --cov=src/reporag --cov-report=html
open htmlcov/index.html
```

## API Testing

```bash
# Health check
curl http://localhost:8000/api/v1/health | python -m json.tool

# Ingest a repo (requires auth token)
TOKEN="<your-jwt-token>"
curl -X POST http://localhost:8000/api/v1/repos/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/pallets/click", "branch": "main"}'

# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does command routing work?", "repo_id": "click"}'

# OpenAPI docs
open http://localhost:8000/docs
```

## Docker Services

| Service | Port | UI |
|---------|------|-----|
| API | 8000 | http://localhost:8000/docs |
| Neo4j | 7474 (HTTP), 7687 (Bolt) | http://localhost:7474 (neo4j/neo4j) |
| Qdrant | 6333 (gRPC), 6334 (HTTP) | http://localhost:6334/dashboard |
| PostgreSQL | 5432 | psql -h localhost -U reporag |
| Frontend | 5173 | http://localhost:5173 |

## Troubleshooting

**docker-compose up fails with port conflict**:
```bash
lsof -i :7474   # find what's using the port
docker-compose down && docker-compose up -d  # restart
```

**Pre-commit ASCII guard fails**:
```bash
grep -rPn '[^\x00-\x7F]' --include="*.py" --include="*.md" .
# Find and replace non-ASCII characters (em dashes, smart quotes, etc.)
```

**Neo4j connection refused**:
```bash
docker-compose logs neo4j  # check logs
# Neo4j takes ~30 seconds to start; wait and retry
```

**Qdrant collection not found**:
```bash
# Re-run indexing pipeline; collections are created on first index build
curl http://localhost:6334/collections | python -m json.tool
```

**alembic upgrade fails**:
```bash
alembic history   # check migration chain
alembic current   # check current state
alembic downgrade -1 && alembic upgrade head  # retry
```

**tree-sitter build errors**:
```bash
pip install tree-sitter tree-sitter-python --force-reinstall
python -c "import tree_sitter_python"  # verify
```

---

NST Engineering | RepoRAG AI Development Guide | 2026
