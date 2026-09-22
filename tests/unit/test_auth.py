"""Tests for the Google OAuth login routes (Issue 27)."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from reporag.api.routes import auth as auth_routes
from reporag.config import settings
from reporag.db.models import User

GOOGLE_INFO = {
    "sub": "google-123",
    "email": "Ada@Example.com",
    "email_verified": True,
    "name": "Ada Lovelace",
    "picture": "https://example.com/ada.png",
}


@pytest.fixture(autouse=True)
def google_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "google_client_id", "client-id")
    monkeypatch.setattr(
        settings, "google_client_secret", type(settings.google_client_secret)("s3")
    )


@pytest.fixture
def stub_google(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the two Google HTTP calls; tests tweak the returned dict."""
    info = dict(GOOGLE_INFO)

    async def fake_exchange(code: str) -> str:
        return "google-access"

    async def fake_userinfo(token: str) -> dict[str, Any]:
        return info

    monkeypatch.setattr(auth_routes, "_exchange_code", fake_exchange)
    monkeypatch.setattr(auth_routes, "_fetch_userinfo", fake_userinfo)
    return info


def _login(client: TestClient) -> str:
    """Start the flow and return the state Google would echo back."""
    resp = client.get("/auth/google", follow_redirects=False)
    return parse_qs(urlparse(resp.headers["location"]).query)["state"][0]


def test_login_redirects_to_google(client: TestClient) -> None:
    resp = client.get("/auth/google", follow_redirects=False)
    assert resp.status_code == 307
    url = urlparse(resp.headers["location"])
    assert url.netloc == "accounts.google.com"
    q = parse_qs(url.query)
    assert q["client_id"] == ["client-id"]
    assert q["response_type"] == ["code"]
    assert q["redirect_uri"] == [settings.google_redirect_uri]
    assert "email" in q["scope"][0]
    assert q["state"][0] == resp.cookies["oauth_state"]


def test_login_503_when_unconfigured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "google_client_id", "")
    resp = client.get("/auth/google", follow_redirects=False)
    assert resp.status_code == 503
    assert resp.json()["error"] == "http_error"


def test_callback_creates_user_and_returns_tokens(
    client: TestClient, stub_google: dict[str, Any], api_app: Any
) -> None:
    state = _login(client)
    resp = client.get("/auth/google/callback", params={"code": "c", "state": state})
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == "ada@example.com"

    key = settings.jwt_secret_key.get_secret_value()
    access = jwt.decode(body["access_token"], key, algorithms=["HS256"])
    refresh = jwt.decode(body["refresh_token"], key, algorithms=["HS256"])
    assert access["type"] == "access" and refresh["type"] == "refresh"
    assert access["sub"] == str(body["user"]["id"])
    assert access["email"] == "ada@example.com"
    assert refresh["exp"] > access["exp"]


async def test_callback_updates_existing_user(
    async_client: Any, stub_google: dict[str, Any], db_session: Any
) -> None:
    async def login() -> dict[str, Any]:
        r = await async_client.get("/auth/google", follow_redirects=False)
        state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
        r = await async_client.get(
            "/auth/google/callback", params={"code": "c", "state": state}
        )
        assert r.status_code == 200
        return r.json()

    first = await login()
    stub_google["name"] = "Ada King"
    second = await login()

    assert first["user"]["id"] == second["user"]["id"]
    users = (await db_session.execute(select(User))).scalars().all()
    assert len(users) == 1
    assert users[0].full_name == "Ada King"
    assert users[0].google_id == "google-123"
    assert users[0].hashed_password is None


async def test_callback_adopts_existing_email_account(
    async_client: Any, stub_google: dict[str, Any], db_session: Any
) -> None:
    db_session.add(User(username="ada", email="ada@example.com", hashed_password="!"))
    await db_session.commit()

    r = await async_client.get("/auth/google", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    r = await async_client.get(
        "/auth/google/callback", params={"code": "c", "state": state}
    )
    assert r.status_code == 200
    users = (await db_session.execute(select(User))).scalars().all()
    assert len(users) == 1 and users[0].google_id == "google-123"


def test_callback_rejects_state_mismatch(
    client: TestClient, stub_google: dict[str, Any]
) -> None:
    _login(client)
    resp = client.get("/auth/google/callback", params={"code": "c", "state": "evil"})
    assert resp.status_code == 400


def test_callback_rejects_missing_state_cookie(
    client: TestClient, stub_google: dict[str, Any]
) -> None:
    resp = client.get("/auth/google/callback", params={"code": "c", "state": "x"})
    assert resp.status_code == 400


def test_callback_consent_denied(client: TestClient) -> None:
    resp = client.get("/auth/google/callback", params={"error": "access_denied"})
    assert resp.status_code == 403
    assert "access_denied" in resp.json()["detail"]


def test_callback_missing_code(client: TestClient) -> None:
    state = _login(client)
    resp = client.get("/auth/google/callback", params={"state": state})
    assert resp.status_code == 400


def _mock_client(handler: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    real = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(auth_routes.httpx, "AsyncClient", factory)


async def test_exchange_code_invalid_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_client(
        lambda r: httpx.Response(400, json={"error": "invalid_grant"}), monkeypatch
    )
    with pytest.raises(auth_routes.HTTPException) as exc:
        await auth_routes._exchange_code("bad")
    assert exc.value.status_code == 400


async def test_exchange_code_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_client(
        lambda r: httpx.Response(200, json={"access_token": "tok"}), monkeypatch
    )
    assert await auth_routes._exchange_code("good") == "tok"


async def test_exchange_code_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    _mock_client(boom, monkeypatch)
    with pytest.raises(auth_routes.HTTPException) as exc:
        await auth_routes._exchange_code("x")
    assert exc.value.status_code == 502


async def test_userinfo_unverified_email(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_client(
        lambda r: httpx.Response(
            200, json={"sub": "1", "email": "a@b.c", "email_verified": False}
        ),
        monkeypatch,
    )
    with pytest.raises(auth_routes.HTTPException) as exc:
        await auth_routes._fetch_userinfo("tok")
    assert exc.value.status_code == 403


async def test_exchange_code_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_client(lambda r: httpx.Response(200, content=b"invalid json"), monkeypatch)
    with pytest.raises(auth_routes.HTTPException) as exc:
        await auth_routes._exchange_code("code")
    assert exc.value.status_code == 502
    assert "Invalid response" in exc.value.detail


async def test_userinfo_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_client(lambda r: httpx.Response(200, content=b"invalid json"), monkeypatch)
    with pytest.raises(auth_routes.HTTPException) as exc:
        await auth_routes._fetch_userinfo("tok")
    assert exc.value.status_code == 502
    assert "Invalid response" in exc.value.detail
