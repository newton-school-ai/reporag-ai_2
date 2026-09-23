"""Unit tests for authentication and Google OAuth 2.0 flow (Issue 27)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.api.middleware.auth import (
    InvalidTokenError,
    TokenExpiredError,
    create_access_token,
    create_refresh_token,
    create_state_token,
    decode_token,
    validate_state_token,
)
from reporag.config import settings
from reporag.db.models import User

# ===========================================================================
# 1. JWT and State Token Utilities Tests
# ===========================================================================


class TestJwtUtilities:
    """Unit tests for JWT creation, validation, expiration, and state tokens."""

    def test_create_and_decode_access_token(self) -> None:
        token = create_access_token(user_id=42, email="developer@example.com")
        assert isinstance(token, str)

        payload = decode_token(token)
        assert payload["sub"] == "42"
        assert payload["email"] == "developer@example.com"
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload

    def test_create_and_decode_refresh_token(self) -> None:
        token = create_refresh_token(user_id=99, email="refresh@example.com")
        assert isinstance(token, str)

        payload = decode_token(token)
        assert payload["sub"] == "99"
        assert payload["email"] == "refresh@example.com"
        assert payload["type"] == "refresh"

    def test_token_expiration(self) -> None:
        token = create_access_token(
            user_id=1,
            email="expired@example.com",
            expires_delta=timedelta(seconds=-10),
        )
        with pytest.raises(TokenExpiredError):
            decode_token(token)

    def test_invalid_token_tampered(self) -> None:
        token = create_access_token(user_id=1, email="tampered@example.com")
        tampered = token[:-4] + "xxxx"
        with pytest.raises(InvalidTokenError):
            decode_token(tampered)

    def test_invalid_token_missing_claims(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import jwt

        raw_token = jwt.encode(
            {"some_key": "some_value"},
            settings.jwt_secret_key.get_secret_value(),
            algorithm="HS256",
        )
        with pytest.raises(InvalidTokenError, match="missing required claims"):
            decode_token(raw_token)

    def test_state_token_create_and_validate(self) -> None:
        state = create_state_token()
        assert isinstance(state, str)

        payload = validate_state_token(state)
        assert payload["type"] == "oauth_state"
        assert "nonce" in payload
        assert "exp" in payload

    def test_state_token_expiration(self) -> None:
        expired_state = create_state_token(expires_delta=timedelta(seconds=-10))
        with pytest.raises(TokenExpiredError):
            validate_state_token(expired_state)

    def test_state_token_invalid_or_missing(self) -> None:
        with pytest.raises(InvalidTokenError, match="Missing state token"):
            validate_state_token("")

        with pytest.raises(InvalidTokenError):
            validate_state_token("not-a-valid-token")


# ===========================================================================
# 2. Google OAuth Route Tests
# ===========================================================================


class TestGoogleOAuthRoutes:
    """Tests for GET /auth/google and GET /auth/google/callback."""

    def test_google_login_redirect_success(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            settings, "google_client_id", "test-client-id.apps.googleusercontent.com"
        )
        monkeypatch.setattr(
            settings,
            "google_redirect_uri",
            "http://localhost:8000/auth/google/callback",
        )

        response = client.get("/auth/google", follow_redirects=False)
        assert response.status_code == status.HTTP_302_FOUND
        redirect_url = response.headers["location"]
        assert redirect_url.startswith("https://accounts.google.com/o/oauth2/v2/auth")

        parsed = urlparse(redirect_url)
        params = parse_qs(parsed.query)

        assert params["client_id"] == ["test-client-id.apps.googleusercontent.com"]
        assert params["redirect_uri"] == ["http://localhost:8000/auth/google/callback"]
        assert params["response_type"] == ["code"]
        assert "openid email profile" in params["scope"][0]
        assert "state" in params

        # Verify state is valid and cookie is set
        state_param = params["state"][0]
        payload = validate_state_token(state_param)
        assert payload["type"] == "oauth_state"
        assert "oauth_state" in response.cookies
        assert response.cookies["oauth_state"] == state_param

    def test_google_login_unconfigured(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "google_client_id", "")
        response = client.get("/auth/google")
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "not configured" in response.json()["detail"]

    def test_callback_oauth_error(self, client: TestClient) -> None:
        response = client.get(
            "/auth/google/callback",
            params={"error": "access_denied", "error_description": "User cancelled"},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "User cancelled" in response.json()["detail"]

    def test_callback_missing_state(self, client: TestClient) -> None:
        response = client.get("/auth/google/callback", params={"code": "some_code"})
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Missing OAuth state parameter" in response.json()["detail"]

    def test_callback_invalid_state(self, client: TestClient) -> None:
        response = client.get(
            "/auth/google/callback",
            params={"code": "some_code", "state": "invalid-tampered-state"},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid OAuth state parameter" in response.json()["detail"]

    def test_callback_expired_state(self, client: TestClient) -> None:
        expired_state = create_state_token(expires_delta=timedelta(seconds=-10))
        response = client.get(
            "/auth/google/callback",
            params={"code": "some_code", "state": expired_state},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "OAuth state parameter has expired" in response.json()["detail"]

    def test_callback_mismatched_cookie_state(self, client: TestClient) -> None:
        state_one = create_state_token()
        state_two = create_state_token()
        client.cookies.set("oauth_state", state_one)

        response = client.get(
            "/auth/google/callback",
            params={"code": "some_code", "state": state_two},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "does not match session state" in response.json()["detail"]
        client.cookies.clear()

    def test_callback_missing_code(self, client: TestClient) -> None:
        state = create_state_token()
        response = client.get("/auth/google/callback", params={"state": state})
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Missing authorization code" in response.json()["detail"]

    def test_callback_token_exchange_network_error(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "google_client_id", "test-client-id")
        state = create_state_token()

        async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        with patch("httpx.AsyncClient.post", side_effect=mock_post):
            response = client.get(
                "/auth/google/callback",
                params={"code": "test_code", "state": state},
            )
            assert response.status_code == status.HTTP_502_BAD_GATEWAY
            assert "Network error" in response.json()["detail"]

    def test_callback_token_exchange_failure(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "google_client_id", "test-client-id")
        state = create_state_token()

        mock_resp = httpx.Response(
            status_code=400,
            json={"error": "invalid_grant", "error_description": "Code expired"},
            request=httpx.Request("POST", "https://oauth2.googleapis.com/token"),
        )

        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            response = client.get(
                "/auth/google/callback",
                params={"code": "expired_code", "state": state},
            )
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "Failed to exchange authorization code" in response.json()["detail"]

    def test_callback_userinfo_missing_email(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "google_client_id", "test-client-id")
        state = create_state_token()

        token_resp = httpx.Response(
            status_code=200,
            json={"access_token": "mock_access_token"},
            request=httpx.Request("POST", "https://oauth2.googleapis.com/token"),
        )
        userinfo_resp = httpx.Response(
            status_code=200,
            json={"sub": "123456", "name": "No Email User"},
            request=httpx.Request(
                "GET", "https://www.googleapis.com/oauth2/v3/userinfo"
            ),
        )

        with (
            patch("httpx.AsyncClient.post", return_value=token_resp),
            patch("httpx.AsyncClient.get", return_value=userinfo_resp),
        ):
            response = client.get(
                "/auth/google/callback",
                params={"code": "valid_code", "state": state},
            )
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "did not provide an email" in response.json()["detail"]

    def test_callback_userinfo_unverified_email_false(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "google_client_id", "test-client-id")
        state = create_state_token()

        token_resp = httpx.Response(
            status_code=200,
            json={"access_token": "mock_access_token"},
            request=httpx.Request("POST", "https://oauth2.googleapis.com/token"),
        )
        userinfo_resp = httpx.Response(
            status_code=200,
            json={"email": "unverified@example.com", "email_verified": False},
            request=httpx.Request(
                "GET", "https://www.googleapis.com/oauth2/v3/userinfo"
            ),
        )

        with (
            patch("httpx.AsyncClient.post", return_value=token_resp),
            patch("httpx.AsyncClient.get", return_value=userinfo_resp),
        ):
            response = client.get(
                "/auth/google/callback",
                params={"code": "valid_code", "state": state},
            )
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "not verified" in response.json()["detail"]

    def test_callback_userinfo_missing_email_verified_field(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject Google logins where email_verified is omitted and not explicitly True."""
        monkeypatch.setattr(settings, "google_client_id", "test-client-id")
        state = create_state_token()

        token_resp = httpx.Response(
            status_code=200,
            json={"access_token": "mock_access_token"},
            request=httpx.Request("POST", "https://oauth2.googleapis.com/token"),
        )
        userinfo_resp = httpx.Response(
            status_code=200,
            json={"email": "unspecified@example.com"},
            request=httpx.Request(
                "GET", "https://www.googleapis.com/oauth2/v3/userinfo"
            ),
        )

        with (
            patch("httpx.AsyncClient.post", return_value=token_resp),
            patch("httpx.AsyncClient.get", return_value=userinfo_resp),
        ):
            response = client.get(
                "/auth/google/callback",
                params={"code": "valid_code", "state": state},
            )
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "not verified" in response.json()["detail"]

    def test_callback_success_creates_new_user(
        self,
        client: TestClient,
        api_app: FastAPI,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "google_client_id", "test-client-id")
        state = create_state_token()
        client.cookies.set("oauth_state", state)

        token_resp = httpx.Response(
            status_code=200,
            json={"access_token": "mock_google_token"},
            request=httpx.Request("POST", "https://oauth2.googleapis.com/token"),
        )
        userinfo_resp = httpx.Response(
            status_code=200,
            json={
                "sub": "google-101",
                "email": "newuser@example.com",
                "name": "New User",
                "email_verified": True,
            },
            request=httpx.Request(
                "GET", "https://www.googleapis.com/oauth2/v3/userinfo"
            ),
        )

        with (
            patch("httpx.AsyncClient.post", return_value=token_resp),
            patch("httpx.AsyncClient.get", return_value=userinfo_resp),
        ):
            response = client.get(
                "/auth/google/callback",
                params={"code": "valid_code", "state": state},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == "newuser@example.com"
        assert data["user"]["username"] == "new_user"

        # Validate issued tokens
        access_payload = decode_token(data["access_token"])
        assert access_payload["email"] == "newuser@example.com"
        assert access_payload["type"] == "access"

        refresh_payload = decode_token(data["refresh_token"])
        assert refresh_payload["email"] == "newuser@example.com"
        assert refresh_payload["type"] == "refresh"

        client.cookies.clear()

    @pytest.mark.asyncio
    async def test_callback_success_updates_existing_user(
        self,
        client: TestClient,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "google_client_id", "test-client-id")
        state = create_state_token()

        # Seed existing user
        user = User(
            username="existing_dev",
            email="existing@example.com",
            hashed_password="!",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        existing_id = user.id

        token_resp = httpx.Response(
            status_code=200,
            json={"access_token": "mock_google_token"},
            request=httpx.Request("POST", "https://oauth2.googleapis.com/token"),
        )
        userinfo_resp = httpx.Response(
            status_code=200,
            json={
                "sub": "google-102",
                "email": "existing@example.com",
                "name": "Existing Developer",
                "email_verified": True,
            },
            request=httpx.Request(
                "GET", "https://www.googleapis.com/oauth2/v3/userinfo"
            ),
        )

        with (
            patch("httpx.AsyncClient.post", return_value=token_resp),
            patch("httpx.AsyncClient.get", return_value=userinfo_resp),
        ):
            response = client.get(
                "/auth/google/callback",
                params={"code": "valid_code", "state": state},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["user"]["id"] == existing_id
        assert data["user"]["email"] == "existing@example.com"
        assert data["user"]["username"] == "existing_dev"
