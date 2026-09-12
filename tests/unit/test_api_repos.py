"""Tests for POST /api/v1/repos/ingest, GET /api/v1/repos, GET /api/v1/repos/{id}.

Acceptance criteria exercised:
* POST /api/v1/repos/ingest returns 202 and triggers async ingestion
* GET  /api/v1/repos    lists repos with pagination
* GET  /api/v1/repos/{id} returns a single repo or 404

All tests are fully offline:
* The background ingestion job (_run_ingestion_job) is mocked so no git
  clone or real DB cross-session logic runs inside these unit tests.
* The DB is an in-memory SQLite engine provided by conftest.async_client.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

# ---------------------------------------------------------------------------
# POST /api/v1/repos/ingest
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    # --- Happy path --------------------------------------------------------

    async def test_returns_202_on_valid_request(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            resp = await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/octocat/Hello-World"},
            )
        assert resp.status_code == 202

    async def test_response_contains_required_fields(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            resp = await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/octocat/Hello-World"},
            )
        data = resp.json()
        assert "repository_id" in data
        assert "job_id" in data
        assert "status" in data
        assert "message" in data

    async def test_status_is_queued_on_response(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            resp = await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/octocat/Hello-World"},
            )
        assert resp.json()["status"] == "queued"

    async def test_repository_id_and_job_id_are_positive_integers(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            resp = await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/example/repo"},
            )
        data = resp.json()
        assert isinstance(data["repository_id"], int) and data["repository_id"] > 0
        assert isinstance(data["job_id"], int) and data["job_id"] > 0

    async def test_background_task_is_triggered(self, async_client):
        """The background ingestion job must be scheduled on a valid request."""
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ) as mock_job:
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/octocat/Hello-World"},
            )
        mock_job.assert_called_once()

    async def test_background_task_called_with_correct_url(self, async_client):
        url = "https://github.com/octocat/Hello-World"
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ) as mock_job:
            await async_client.post("/api/v1/repos/ingest", json={"url": url})
        _, args, _ = mock_job.mock_calls[0]
        # signature: _run_ingestion_job(repository_id, job_id, url, branch)
        assert args[2] == url

    async def test_name_derived_from_url_when_omitted(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            resp = await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/octocat/Hello-World"},
            )
        repo_id = resp.json()["repository_id"]

        # Fetch the created repo to verify its name
        detail_resp = await async_client.get(f"/api/v1/repos/{repo_id}")
        assert detail_resp.json()["name"] == "Hello-World"

    async def test_custom_name_is_stored(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            resp = await async_client.post(
                "/api/v1/repos/ingest",
                json={
                    "url": "https://github.com/octocat/Hello-World",
                    "name": "My Repo",
                },
            )
        repo_id = resp.json()["repository_id"]
        detail_resp = await async_client.get(f"/api/v1/repos/{repo_id}")
        assert detail_resp.json()["name"] == "My Repo"

    async def test_branch_forwarded_to_background_task(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ) as mock_job:
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/octocat/Hello-World", "branch": "dev"},
            )
        _, args, _ = mock_job.mock_calls[0]
        assert args[3] == "dev"  # branch argument

    async def test_branch_none_when_not_provided(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ) as mock_job:
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/octocat/Hello-World"},
            )
        _, args, _ = mock_job.mock_calls[0]
        assert args[3] is None

    async def test_demo_user_created_on_first_ingest(self, async_client, db_session):
        from sqlalchemy import select

        from reporag.db.models import User

        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/octocat/Hello-World"},
            )
        result = await db_session.execute(select(User))
        users = result.scalars().all()
        assert len(users) >= 1
        assert any(u.username == "demo-user" for u in users)

    async def test_demo_user_reused_on_second_ingest(self, async_client, db_session):
        from sqlalchemy import select

        from reporag.db.models import User

        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/a/b"},
            )
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/c/d"},
            )
        result = await db_session.execute(select(User))
        users = result.scalars().all()
        assert sum(1 for u in users if u.username == "demo-user") == 1

    # --- Validation / edge cases ------------------------------------------

    async def test_missing_url_returns_422(self, async_client):
        resp = await async_client.post("/api/v1/repos/ingest", json={})
        assert resp.status_code == 422

    async def test_blank_url_returns_422(self, async_client):
        resp = await async_client.post("/api/v1/repos/ingest", json={"url": "   "})
        assert resp.status_code == 422

    async def test_empty_string_url_returns_422(self, async_client):
        resp = await async_client.post("/api/v1/repos/ingest", json={"url": ""})
        assert resp.status_code == 422

    async def test_name_max_length_100_enforced(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            resp = await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/a/b", "name": "x" * 101},
            )
        assert resp.status_code == 422

    async def test_name_exactly_100_chars_accepted(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            resp = await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/a/b", "name": "x" * 100},
            )
        assert resp.status_code == 202

    async def test_git_suffix_stripped_from_derived_name(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            resp = await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/octocat/Hello-World.git"},
            )
        repo_id = resp.json()["repository_id"]
        detail = await async_client.get(f"/api/v1/repos/{repo_id}")
        assert detail.json()["name"] == "Hello-World"

    async def test_non_json_body_returns_422(self, async_client):
        resp = await async_client.post(
            "/api/v1/repos/ingest",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/repos
# ---------------------------------------------------------------------------


class TestListReposEndpoint:
    async def test_empty_db_returns_empty_list(self, async_client):
        resp = await async_client.get("/api/v1/repos")
        assert resp.status_code == 200
        data = resp.json()
        assert data["repositories"] == []
        assert data["total"] == 0

    async def test_response_schema_has_repositories_and_total(self, async_client):
        resp = await async_client.get("/api/v1/repos")
        data = resp.json()
        assert "repositories" in data
        assert "total" in data

    async def test_after_ingest_total_is_one(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/a/b"},
            )
        resp = await async_client.get("/api/v1/repos")
        data = resp.json()
        assert data["total"] == 1
        assert len(data["repositories"]) == 1

    async def test_repo_fields_present_in_list(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/a/b"},
            )
        resp = await async_client.get("/api/v1/repos")
        repo = resp.json()["repositories"][0]
        for field in ("id", "name", "url", "status", "created_at", "updated_at"):
            assert field in repo, f"missing field: {field}"

    async def test_pagination_limit(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            for i in range(5):
                await async_client.post(
                    "/api/v1/repos/ingest",
                    json={"url": f"https://github.com/a/repo-{i}"},
                )
        resp = await async_client.get("/api/v1/repos?limit=2")
        data = resp.json()
        assert data["total"] == 5
        assert len(data["repositories"]) == 2

    async def test_pagination_offset(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            for i in range(5):
                await async_client.post(
                    "/api/v1/repos/ingest",
                    json={"url": f"https://github.com/a/repo-{i}"},
                )
        resp_all = await async_client.get("/api/v1/repos?limit=5&offset=0")
        assert len(resp_all.json()["repositories"]) == 5
        resp_offset = await async_client.get("/api/v1/repos?limit=5&offset=3")
        assert resp_offset.json()["total"] == 5
        assert len(resp_offset.json()["repositories"]) == 2

    async def test_limit_clamped_to_200(self, async_client):
        """A limit above 200 is silently clamped -- should not error."""
        resp = await async_client.get("/api/v1/repos?limit=9999")
        assert resp.status_code == 200

    async def test_limit_below_1_clamped_to_1(self, async_client):
        """A limit of 0 is silently clamped to 1 -- should not error."""
        resp = await async_client.get("/api/v1/repos?limit=0")
        assert resp.status_code == 200

    async def test_negative_offset_clamped_to_0(self, async_client):
        resp = await async_client.get("/api/v1/repos?offset=-10")
        assert resp.status_code == 200

    async def test_repos_ordered_by_created_at_desc(self, async_client):
        """Most-recently created repository appears first."""
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            for i in range(3):
                await async_client.post(
                    "/api/v1/repos/ingest",
                    json={"url": f"https://github.com/a/repo-{i}"},
                )
        resp = await async_client.get("/api/v1/repos")
        ids = [r["id"] for r in resp.json()["repositories"]]
        assert ids == sorted(ids, reverse=True)


# ---------------------------------------------------------------------------
# GET /api/v1/repos/{id}
# ---------------------------------------------------------------------------


class TestGetRepoEndpoint:
    async def test_returns_200_for_existing_repo(self, async_client):
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            create_resp = await async_client.post(
                "/api/v1/repos/ingest",
                json={"url": "https://github.com/a/b"},
            )
        repo_id = create_resp.json()["repository_id"]
        resp = await async_client.get(f"/api/v1/repos/{repo_id}")
        assert resp.status_code == 200

    async def test_returns_correct_repo_fields(self, async_client):
        url = "https://github.com/octocat/Hello-World"
        with patch(
            "reporag.api.routes.repos._run_ingestion_job", new_callable=AsyncMock
        ):
            create_resp = await async_client.post(
                "/api/v1/repos/ingest", json={"url": url}
            )
        repo_id = create_resp.json()["repository_id"]
        resp = await async_client.get(f"/api/v1/repos/{repo_id}")
        data = resp.json()
        assert data["id"] == repo_id
        assert data["url"] == url
        assert data["name"] == "Hello-World"
        assert data["status"] == "queued"

    async def test_nonexistent_id_returns_404(self, async_client):
        resp = await async_client.get("/api/v1/repos/999999")
        assert resp.status_code == 404

    async def test_404_detail_message_present(self, async_client):
        resp = await async_client.get("/api/v1/repos/999999")
        assert "detail" in resp.json()

    async def test_zero_id_returns_404(self, async_client):
        resp = await async_client.get("/api/v1/repos/0")
        assert resp.status_code == 404

    async def test_string_id_returns_422(self, async_client):
        resp = await async_client.get("/api/v1/repos/abc")
        assert resp.status_code == 422
