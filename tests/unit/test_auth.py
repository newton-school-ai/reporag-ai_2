"""Unit tests for the Google OAuth login flow (Issue 27).

Covers every acceptance criterion:

* ``GET /auth/google`` redirects to Google's consent screen.
* ``GET /auth/google/callback`` exchanges the code for tokens.
* Email and profile are extracted from Google's response.
* A ``User`` row is created on first login and updated on later ones.
* The callback returns JWT access and refresh tokens.
* OAuth errors (declined consent, invalid code) are handled.

Beyond those, the suite pins the security properties the flow depends on:
the ``state`` check that stops login CSRF, the PKCE binding that makes an
intercepted code useless, the nonce that stops an ID token being replayed,
and the signature verification that makes Google's claims trustworthy at
all.

Nothing here reaches the network. Google's HTTP endpoints are served by an
``httpx.MockTransport``, and ID tokens are signed with an RSA key generated
in-process and served through a stand-in for PyJWT's JWKS client -- so the
real verification code path runs against real cryptography.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import SecretStr

from reporag.api.middleware.auth import TokenType, decode_token
from reporag.api.routes import auth as auth_routes
from reporag.api.routes.auth import (
    GOOGLE_TOKEN_ENDPOINT,
    GOOGLE_USERINFO_ENDPOINT,
    _pkce_challenge,
    _seal_transaction,
    _unique_username,
    _username_from_email,
)
from reporag.config import settings as live_settings
from reporag.db.models import User

_CLIENT_ID = "test-client-id.apps.googleusercontent.com"
_REDIRECT_URI = "http://localhost:8000/auth/google/callback"
_TX_COOKIE = auth_routes._TX_COOKIE


# ---------------------------------------------------------------------------
# Google stand-ins
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_key() -> Any:
    """An RSA key standing in for Google's ID token signing key.

    Module-scoped because generating one costs about a second and every
    test in the file wants the same key.
    """
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(autouse=True)
def oauth_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the app credentials, as a deployed instance would have."""
    monkeypatch.setattr(live_settings, "google_client_id", _CLIENT_ID)
    monkeypatch.setattr(
        live_settings, "google_client_secret", SecretStr("test-client-secret")
    )
    monkeypatch.setattr(live_settings, "google_redirect_uri", _REDIRECT_URI)
    monkeypatch.setattr(live_settings, "jwt_secret_key", SecretStr("test-jwt-secret"))


@pytest.fixture(autouse=True)
def signing_keys(monkeypatch: pytest.MonkeyPatch, rsa_key: Any) -> None:
    """Serve the test public key where PyJWT would fetch Google's JWKS.

    Only the key source is replaced. Signature checking, issuer, audience,
    expiry and nonce validation all run as written.
    """

    class _Key:
        key = rsa_key.public_key()

    class _FakeJWKClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get_signing_key_from_jwt(self, token: str) -> Any:
            return _Key()

    monkeypatch.setattr(jwt, "PyJWKClient", _FakeJWKClient)


def make_id_token(rsa_key: Any, **overrides: Any) -> str:
    """Build an ID token the way Google would sign one."""
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "iss": "https://accounts.google.com",
        "aud": _CLIENT_ID,
        "sub": "google-sub-12345",
        "email": "ada@example.com",
        "email_verified": True,
        "name": "Ada Lovelace",
        "picture": "https://lh3.googleusercontent.com/ada",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    claims.update(overrides)
    for key, value in list(claims.items()):
        if value is None:
            del claims[key]
    return jwt.encode(claims, rsa_key, algorithm="RS256")


class GoogleStub:
    """A stand-in for Google's token and userinfo endpoints.

    Records what it was asked so tests can assert on the request the flow
    actually made -- the PKCE verifier, the client credentials, the code.
    """

    def __init__(
        self,
        *,
        id_token: str | None = None,
        token_status: int = 200,
        token_body: dict | None = None,
        userinfo_status: int = 200,
        userinfo_body: dict | None = None,
        network_error: bool = False,
    ) -> None:
        self.id_token = id_token
        self.token_status = token_status
        self.token_body = token_body
        self.userinfo_status = userinfo_status
        self.userinfo_body = userinfo_body if userinfo_body is not None else {}
        self.network_error = network_error
        self.token_requests: list[dict[str, str]] = []
        self.userinfo_requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.network_error:
            raise httpx.ConnectError("connection refused")

        if str(request.url).startswith(GOOGLE_TOKEN_ENDPOINT):
            self.token_requests.append(
                dict(parse_qs(request.content.decode(), keep_blank_values=True))  # type: ignore[arg-type]
            )
            body = self.token_body
            if body is None:
                body = {"access_token": "google-access-token", "token_type": "Bearer"}
                if self.id_token is not None:
                    body["id_token"] = self.id_token
            return httpx.Response(self.token_status, json=body)

        if str(request.url).startswith(GOOGLE_USERINFO_ENDPOINT):
            self.userinfo_requests.append(request)
            return httpx.Response(self.userinfo_status, json=self.userinfo_body)

        return httpx.Response(404, json={"error": "unexpected endpoint"})

    def last_token_request(self) -> dict[str, str]:
        """The most recent token-exchange form, flattened to single values."""
        return {k: v[0] for k, v in self.token_requests[-1].items()}


@pytest.fixture
def google(api_app: Any, rsa_key: Any) -> GoogleStub:
    """Install a Google stand-in and wire it into the app's HTTP client."""
    stub = GoogleStub(id_token=make_id_token(rsa_key))

    async def override_client() -> Any:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(stub.handler)
        ) as client:
            yield client

    api_app.dependency_overrides[auth_routes.get_http_client] = override_client
    return stub


def _cookie_is_cleared(response: Any) -> bool:
    """True when *response* tells the browser to drop the transaction."""
    header = response.headers.get("set-cookie", "")
    return _TX_COOKIE in header and ("max-age=0" in header.lower())


def _unseal(cookie: str) -> dict[str, Any]:
    """Read a transaction cookie the way the callback does."""
    return jwt.decode(
        cookie,
        "test-jwt-secret",
        algorithms=["HS256"],
        issuer=auth_routes.ISSUER,
        audience=auth_routes._TX_AUDIENCE,
    )


def start_login(client: Any) -> dict[str, str]:
    """Run ``GET /auth/google`` and return what the browser would now hold.

    Returns the ``state`` and ``nonce`` Google was sent and the sealed
    transaction cookie, which together are the inputs to a real callback.
    """
    response = client.get("/auth/google", follow_redirects=False)
    params = parse_qs(urlparse(response.headers["location"]).query)
    return {
        "state": params["state"][0],
        "nonce": params["nonce"][0],
        "cookie": response.cookies[_TX_COOKIE],
    }


def complete_login(client: Any, session: dict[str, str], **params: Any) -> Any:
    """Call the callback with a started login's state and cookie."""
    client.cookies.set(_TX_COOKIE, session["cookie"], path="/auth")
    query = {"code": "auth-code-abc", "state": session["state"]}
    query.update(params)
    return client.get("/auth/google/callback", params=query, follow_redirects=False)


# ---------------------------------------------------------------------------
# GET /auth/google
# ---------------------------------------------------------------------------


class TestLoginRedirect:
    def test_redirects_to_google_consent_screen(self, client: Any) -> None:
        response = client.get("/auth/google", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"].startswith(auth_routes.GOOGLE_AUTH_ENDPOINT)

    def test_sends_the_parameters_google_requires(self, client: Any) -> None:
        response = client.get("/auth/google", follow_redirects=False)
        params = parse_qs(urlparse(response.headers["location"]).query)
        assert params["client_id"] == [_CLIENT_ID]
        assert params["redirect_uri"] == [_REDIRECT_URI]
        assert params["response_type"] == ["code"]
        assert params["code_challenge_method"] == ["S256"]
        assert set(params["scope"][0].split()) == {"openid", "email", "profile"}

    def test_sets_a_signed_transaction_cookie(self, client: Any) -> None:
        response = client.get("/auth/google", follow_redirects=False)
        assert _TX_COOKIE in response.cookies
        header = response.headers["set-cookie"].lower()
        # HttpOnly keeps the verifier away from any script on the page;
        # Lax is required for the cross-site callback navigation to carry
        # it, where Strict would withhold it and break every login.
        assert "httponly" in header
        assert "samesite=lax" in header
        assert "path=/auth" in header

    def test_each_attempt_gets_fresh_secrets(self, client: Any) -> None:
        first = start_login(client)
        second = start_login(client)
        # Reusing state or nonce across logins would make both replayable.
        assert first["state"] != second["state"]
        assert first["nonce"] != second["nonce"]

    def test_the_challenge_matches_the_sealed_verifier(self, client: Any) -> None:
        response = client.get("/auth/google", follow_redirects=False)
        params = parse_qs(urlparse(response.headers["location"]).query)
        sealed = _unseal(response.cookies[_TX_COOKIE])

        # Google is sent the challenge; only the cookie carries the verifier
        # that produces it. A mismatch would make every exchange fail.
        assert params["code_challenge"] == [_pkce_challenge(sealed["verifier"])]
        assert sealed["state"] == params["state"][0]
        assert sealed["nonce"] == params["nonce"][0]

    def test_unconfigured_server_returns_503(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(live_settings, "google_client_id", "")
        response = client.get("/auth/google", follow_redirects=False)
        # Not a redirect to Google with an empty client id, which would put
        # the real cause two systems away from the error the user sees.
        assert response.status_code == 503

    def test_missing_secret_also_returns_503(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(live_settings, "google_client_secret", SecretStr(""))
        assert client.get("/auth/google", follow_redirects=False).status_code == 503


# ---------------------------------------------------------------------------
# GET /auth/google/callback -- the happy path
# ---------------------------------------------------------------------------


class TestCallbackSuccess:
    def test_returns_access_and_refresh_tokens(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=session["nonce"])

        response = complete_login(client, session)
        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert decode_token(body["access_token"], TokenType.ACCESS).email == (
            "ada@example.com"
        )
        assert decode_token(body["refresh_token"], TokenType.REFRESH).user_id == (
            body["user"]["id"]
        )

    def test_reports_the_access_token_lifetime(
        self, client: Any, google: GoogleStub, rsa_key: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(live_settings, "jwt_access_token_expire_minutes", 45)
        session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=session["nonce"])
        assert complete_login(client, session).json()["expires_in"] == 45 * 60

    def test_returns_the_profile_google_supplied(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=session["nonce"])
        user = complete_login(client, session).json()["user"]
        assert user["email"] == "ada@example.com"
        assert user["full_name"] == "Ada Lovelace"
        assert user["avatar_url"].endswith("/ada")

    def test_sends_the_pkce_verifier_in_the_exchange(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=session["nonce"])
        complete_login(client, session)

        form = google.last_token_request()
        assert form["grant_type"] == "authorization_code"
        assert form["code"] == "auth-code-abc"
        assert form["client_secret"] == "test-client-secret"
        # The verifier is what binds this code to this transaction; without
        # it an intercepted code is enough to complete someone's login.
        assert form["code_verifier"] == _unseal(session["cookie"])["verifier"]

    def test_clears_the_transaction_cookie(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=session["nonce"])
        response = complete_login(client, session)
        # Single use: a live verifier left in the browser stays replayable
        # for the rest of its ten minutes.
        assert _cookie_is_cleared(response)

    def test_profile_falls_back_to_userinfo(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        session = start_login(client)
        # An ID token without profile claims: the scope may not have been
        # granted, and userinfo is the documented place to look.
        google.id_token = make_id_token(
            rsa_key, nonce=session["nonce"], name=None, picture=None
        )
        google.userinfo_body = {"name": "Ada L", "picture": "https://example.com/pic"}
        user = complete_login(client, session).json()["user"]
        assert user["full_name"] == "Ada L"
        assert user["avatar_url"] == "https://example.com/pic"

    def test_a_failing_userinfo_call_does_not_fail_the_login(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=session["nonce"])
        google.userinfo_status = 500
        # The ID token already established who this is; losing a display
        # name is not a reason to refuse entry.
        assert complete_login(client, session).status_code == 200


# ---------------------------------------------------------------------------
# User records
# ---------------------------------------------------------------------------


class TestUserPersistence:
    async def test_creates_a_user_on_first_login(
        self, client: Any, google: GoogleStub, rsa_key: Any, db_session: Any
    ) -> None:
        session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=session["nonce"])
        body = complete_login(client, session).json()

        user = await db_session.get(User, body["user"]["id"])
        assert user.google_id == "google-sub-12345"
        assert user.email == "ada@example.com"
        assert user.last_login_at is not None

    async def test_an_oauth_user_has_no_password(
        self, client: Any, google: GoogleStub, rsa_key: Any, db_session: Any
    ) -> None:
        session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=session["nonce"])
        body = complete_login(client, session).json()

        user = await db_session.get(User, body["user"]["id"])
        # A placeholder hash would be indistinguishable from a real one at
        # the point a local login flow checks it.
        assert user.hashed_password is None
        assert user.is_federated

    def test_second_login_updates_rather_than_duplicates(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        first_session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=first_session["nonce"])
        first = complete_login(client, first_session).json()

        second_session = start_login(client)
        google.id_token = make_id_token(
            rsa_key, nonce=second_session["nonce"], name="Ada Byron"
        )
        second = complete_login(client, second_session).json()

        assert first["user"]["id"] == second["user"]["id"]
        assert second["user"]["full_name"] == "Ada Byron"

    def test_identity_follows_sub_not_email(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        first_session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=first_session["nonce"])
        first = complete_login(client, first_session).json()

        # Same person, new primary address -- the case that makes `sub` the
        # right key. Matching on email would strand their repositories.
        second_session = start_login(client)
        google.id_token = make_id_token(
            rsa_key, nonce=second_session["nonce"], email="ada@newdomain.com"
        )
        second = complete_login(client, second_session).json()

        assert second["user"]["id"] == first["user"]["id"]
        assert second["user"]["email"] == "ada@newdomain.com"

    async def test_adopts_an_existing_account_with_the_same_email(
        self, client: Any, google: GoogleStub, rsa_key: Any, db_session: Any
    ) -> None:
        existing = User(
            username="ada", email="ada@example.com", hashed_password="local-hash"
        )
        db_session.add(existing)
        await db_session.commit()
        await db_session.refresh(existing)

        session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=session["nonce"])
        body = complete_login(client, session).json()

        # Creating a second row would leave the person unable to see the
        # repositories the first one owns.
        assert body["user"]["id"] == existing.id

    async def test_refuses_an_email_owned_by_another_google_account(
        self, client: Any, google: GoogleStub, rsa_key: Any, db_session: Any
    ) -> None:
        db_session.add(
            User(
                username="ada",
                email="ada@example.com",
                hashed_password=None,
                google_id="some-other-google-sub",
            )
        )
        await db_session.commit()

        session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=session["nonce"])
        response = complete_login(client, session)

        # Adopting the row would hand the first account's repositories to
        # whoever later re-registered the released address.
        assert response.status_code == 409

    def test_distinct_accounts_get_distinct_usernames(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        first_session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=first_session["nonce"])
        first = complete_login(client, first_session).json()

        second_session = start_login(client)
        google.id_token = make_id_token(
            rsa_key,
            nonce=second_session["nonce"],
            sub="google-sub-99999",
            email="ada@other.com",
        )
        second = complete_login(client, second_session).json()

        # Same local part, different people: username is unique, so the
        # second has to be varied rather than collide.
        assert first["user"]["username"] != second["user"]["username"]


# ---------------------------------------------------------------------------
# Callback failures
# ---------------------------------------------------------------------------


class TestCallbackRejection:
    def test_declined_consent_is_a_400(self, client: Any, google: GoogleStub) -> None:
        session = start_login(client)
        response = complete_login(client, session, error="access_denied", code=None)
        # The user pressed Cancel. Nothing malfunctioned, so not a 500.
        assert response.status_code == 400
        assert "access_denied" in response.json()["detail"]

    def test_missing_code_is_a_400(self, client: Any, google: GoogleStub) -> None:
        session = start_login(client)
        assert complete_login(client, session, code=None).status_code == 400

    def test_missing_transaction_cookie_is_a_400(
        self, client: Any, google: GoogleStub
    ) -> None:
        session = start_login(client)
        client.cookies.clear()
        response = client.get(
            "/auth/google/callback",
            params={"code": "abc", "state": session["state"]},
            follow_redirects=False,
        )
        assert response.status_code == 400

    def test_state_mismatch_is_rejected(self, client: Any, google: GoogleStub) -> None:
        session = start_login(client)
        # The login CSRF case: an attacker sends a victim a callback URL
        # carrying the attacker's code, hoping to bind the victim's browser
        # to the attacker's Google account.
        response = complete_login(client, session, state="attacker-state")
        assert response.status_code == 400
        assert not google.token_requests

    def test_a_forged_cookie_is_rejected(self, client: Any, google: GoogleStub) -> None:
        forged = jwt.encode(
            {
                "state": "s",
                "verifier": "v",
                "nonce": "n",
                "iss": auth_routes.ISSUER,
                "aud": auth_routes._TX_AUDIENCE,
                "exp": datetime.now(UTC) + timedelta(minutes=5),
            },
            "not-the-real-secret",
            algorithm="HS256",
        )
        client.cookies.set(_TX_COOKIE, forged, path="/auth")
        response = client.get(
            "/auth/google/callback",
            params={"code": "abc", "state": "s"},
            follow_redirects=False,
        )
        # The signature is the only thing making the returned state
        # trustworthy; without this check state offers no protection at all.
        assert response.status_code == 400
        assert not google.token_requests

    def test_an_expired_transaction_is_rejected(
        self, client: Any, google: GoogleStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(auth_routes, "_TX_TTL_SECONDS", -1)
        stale = _seal_transaction("s", "v", "n")
        client.cookies.set(_TX_COOKIE, stale, path="/auth")
        response = client.get(
            "/auth/google/callback",
            params={"code": "abc", "state": "s"},
            follow_redirects=False,
        )
        assert response.status_code == 400

    def test_an_access_token_cannot_be_used_as_a_transaction_cookie(
        self, client: Any, google: GoogleStub
    ) -> None:
        from reporag.api.middleware.auth import create_access_token

        client.cookies.set(_TX_COOKIE, create_access_token(1, "a@b.com"), path="/auth")
        response = client.get(
            "/auth/google/callback",
            params={"code": "abc", "state": "s"},
            follow_redirects=False,
        )
        # Same secret, different audience -- which is why the audience is
        # checked rather than assumed.
        assert response.status_code == 400

    @pytest.mark.parametrize(
        "failure",
        ["bad_state", "google_rejects", "unreachable"],
    )
    def test_a_failed_attempt_still_burns_the_transaction(
        self, client: Any, google: GoogleStub, failure: str
    ) -> None:
        session = start_login(client)
        params: dict[str, Any] = {}
        if failure == "bad_state":
            params["state"] = "wrong"
        elif failure == "google_rejects":
            google.token_status = 400
            google.token_body = {"error": "invalid_grant"}
        else:
            google.network_error = True

        response = complete_login(client, session, **params)
        assert response.status_code >= 400
        # FastAPI discards the injected response when an exception is
        # raised, so without explicitly re-attaching this header a failed
        # attempt would leave a usable verifier in the browser.
        assert _cookie_is_cleared(response)

    def test_google_rejecting_the_code_is_a_400(
        self, client: Any, google: GoogleStub
    ) -> None:
        session = start_login(client)
        google.token_status = 400
        google.token_body = {"error": "invalid_grant"}
        response = complete_login(client, session)
        assert response.status_code == 400
        # Google's wording goes to the log, not to the caller.
        assert "invalid_grant" not in response.text

    def test_google_being_unreachable_is_a_502(
        self, client: Any, google: GoogleStub
    ) -> None:
        session = start_login(client)
        google.network_error = True
        # Retryable, and not this service's fault -- which a 500 would not
        # convey.
        assert complete_login(client, session).status_code == 502

    def test_a_response_without_an_id_token_is_a_502(
        self, client: Any, google: GoogleStub
    ) -> None:
        session = start_login(client)
        google.token_body = {"access_token": "x"}
        assert complete_login(client, session).status_code == 502


# ---------------------------------------------------------------------------
# ID token verification
# ---------------------------------------------------------------------------


class TestIdTokenVerification:
    def test_a_token_signed_by_someone_else_is_rejected(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        session = start_login(client)
        attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        google.id_token = jwt.encode(
            {
                "iss": "https://accounts.google.com",
                "aud": _CLIENT_ID,
                "sub": "attacker",
                "email": "attacker@example.com",
                "email_verified": True,
                "nonce": session["nonce"],
                "iat": datetime.now(UTC),
                "exp": datetime.now(UTC) + timedelta(hours=1),
            },
            attacker_key,
            algorithm="RS256",
        )
        assert complete_login(client, session).status_code == 401

    def test_a_token_for_another_client_is_rejected(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        session = start_login(client)
        # A valid Google token issued to a different application. Without
        # the audience check, any app's token would log someone in here.
        google.id_token = make_id_token(
            rsa_key, nonce=session["nonce"], aud="someone-elses-client-id"
        )
        assert complete_login(client, session).status_code == 401

    def test_a_token_from_another_issuer_is_rejected(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        session = start_login(client)
        google.id_token = make_id_token(
            rsa_key, nonce=session["nonce"], iss="https://evil.example.com"
        )
        assert complete_login(client, session).status_code == 401

    def test_an_expired_token_is_rejected(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        session = start_login(client)
        google.id_token = make_id_token(
            rsa_key,
            nonce=session["nonce"],
            iat=datetime.now(UTC) - timedelta(hours=2),
            exp=datetime.now(UTC) - timedelta(hours=1),
        )
        assert complete_login(client, session).status_code == 401

    def test_a_replayed_token_from_another_login_is_rejected(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        other_login = start_login(client)
        stolen = make_id_token(rsa_key, nonce=other_login["nonce"])

        session = start_login(client)
        google.id_token = stolen
        # The nonce binds an ID token to the one login that asked for it.
        assert complete_login(client, session).status_code == 401

    def test_a_token_with_no_nonce_is_rejected(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        session = start_login(client)
        google.id_token = make_id_token(rsa_key)
        assert complete_login(client, session).status_code == 401

    def test_an_unverified_email_is_refused(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        session = start_login(client)
        google.id_token = make_id_token(
            rsa_key, nonce=session["nonce"], email_verified=False
        )
        # Accounts are adopted by email below, so an address the person may
        # not control is a route into someone else's account.
        assert complete_login(client, session).status_code == 403

    def test_a_token_with_no_email_is_refused(
        self, client: Any, google: GoogleStub, rsa_key: Any
    ) -> None:
        session = start_login(client)
        google.id_token = make_id_token(rsa_key, nonce=session["nonce"], email=None)
        assert complete_login(client, session).status_code == 401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestPkce:
    def test_matches_the_rfc_7636_test_vector(self) -> None:
        verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        # Appendix B of RFC 7636.
        assert (
            _pkce_challenge(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        )

    def test_is_unpadded_base64url(self) -> None:
        challenge = _pkce_challenge("a" * 64)
        assert "=" not in challenge and "+" not in challenge and "/" not in challenge

    def test_is_a_one_way_function_of_the_verifier(self) -> None:
        verifier = "some-verifier-value"
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        assert _pkce_challenge(verifier) == expected


class TestUsernameDerivation:
    @pytest.mark.parametrize(
        ("email", "expected"),
        [
            ("ada@example.com", "ada"),
            ("Ada.Lovelace@example.com", "ada.lovelace"),
            ("ada+tag@example.com", "adatag"),
            ("a" * 80 + "@example.com", "a" * 50),
            ("+++@example.com", "user"),
        ],
    )
    def test_derives_a_usable_handle(self, email: str, expected: str) -> None:
        assert _username_from_email(email) == expected

    def test_never_exceeds_the_column_width(self) -> None:
        assert len(_username_from_email("x" * 200 + "@example.com")) <= 50

    async def test_unique_username_avoids_a_collision(self, db_session: Any) -> None:
        db_session.add(User(username="ada", email="a@x.com", hashed_password=None))
        await db_session.commit()
        assert await _unique_username(db_session, "ada") == "ada2"

    async def test_unique_username_keeps_trying(self, db_session: Any) -> None:
        for name, email in [("ada", "a@x.com"), ("ada2", "b@x.com")]:
            db_session.add(User(username=name, email=email, hashed_password=None))
        await db_session.commit()
        assert await _unique_username(db_session, "ada") == "ada3"

    async def test_a_suffixed_username_still_fits_the_column(
        self, db_session: Any
    ) -> None:
        base = "a" * 50
        db_session.add(User(username=base, email="a@x.com", hashed_password=None))
        await db_session.commit()
        # Appending rather than trimming would overflow String(50).
        assert len(await _unique_username(db_session, base)) <= 50


class TestOpenApi:
    def test_auth_endpoints_are_documented(self, client: Any) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/auth/google" in paths
        assert "/auth/google/callback" in paths

    def test_no_secret_appears_in_the_schema(self, client: Any) -> None:
        schema = json.dumps(client.get("/openapi.json").json())
        assert "test-client-secret" not in schema
        assert "test-jwt-secret" not in schema
