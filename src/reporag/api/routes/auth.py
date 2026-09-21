"""Google OAuth 2.0 login routes.

Why
---
Passwordless sign-in: Google verifies the person, and the API receives a
verified email and profile instead of storing a credential.

Flow
----
1. ``GET /auth/google`` redirects the browser to Google's consent screen.
2. Google redirects back to ``GET /auth/google/callback`` with a ``code``.
3. The callback exchanges the code for a Google access token, reads the
   userinfo endpoint, creates or updates the :class:`User`, and returns a
   JWT access + refresh token pair.

Design
------
* **CSRF protection via ``state``.** A random value is sent to Google and
  also set as an HttpOnly cookie; the callback rejects the request unless
  both match. Without it an attacker could feed a victim's browser the
  attacker's authorization code.
* **Accounts are keyed by Google ``sub``, then email.** ``sub`` is the
  stable identifier; email is the fallback so an existing account (such as
  one created before OAuth existed) is adopted rather than duplicated.
* **Only verified emails sign in.** Linking by email is only safe when
  Google vouches for the address.
* **Every failure is an ``HTTPException``**, so it renders in the shared
  ``{error, detail, status_code}`` shape from :mod:`reporag.api.main`.
"""

from __future__ import annotations

import logging
import re
import secrets
from datetime import datetime
from typing import Annotated, Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.api.middleware.auth import create_access_token, create_refresh_token
from reporag.config import settings
from reporag.db.models import User
from reporag.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_SCOPES = "openid email profile"

STATE_COOKIE = "oauth_state"
STATE_COOKIE_MAX_AGE_SECONDS = 600
_HTTP_TIMEOUT_SECONDS = 10.0
_USERNAME_MAX = 50


class TokenResponse(BaseModel):
    """JWT pair returned after a successful login."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(BaseModel):
    """The account that just logged in."""

    id: int
    email: str
    username: str
    created_at: datetime


class LoginResponse(TokenResponse):
    """Tokens plus the user they were issued to."""

    user: UserOut


def _require_configured() -> None:
    """Fail with 503 when Google credentials are missing."""
    if not settings.google_client_id or not (
        settings.google_client_secret.get_secret_value()
    ):
        raise HTTPException(
            status_code=503,
            detail="Google OAuth is not configured on this server.",
        )


@router.get(
    "/google",
    summary="Start Google login",
    response_class=RedirectResponse,
    status_code=307,
)
async def google_login() -> RedirectResponse:
    """Redirect the browser to Google's consent screen."""
    _require_configured()

    state = secrets.token_urlsafe(32)
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    response = RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}")
    response.set_cookie(
        STATE_COOKIE,
        state,
        max_age=STATE_COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        secure=settings.is_production,
        samesite="lax",
    )
    return response


async def _exchange_code(code: str) -> str:
    """Trade an authorization code for a Google access token."""
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret.get_secret_value(),
                    "redirect_uri": settings.google_redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
        if resp.status_code in (400, 401):
            # invalid_grant: the code is wrong, expired, or already used.
            raise HTTPException(400, "Invalid or expired authorization code.")
        if resp.status_code != 200:
            logger.warning("Google token endpoint returned %s", resp.status_code)
            raise HTTPException(502, "Google rejected the token exchange.")

        data = resp.json()
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        logger.warning("Google token endpoint unreachable: %s", exc)
        raise HTTPException(502, "Could not reach Google to complete login.") from exc
    except ValueError as exc:
        logger.warning("Google token endpoint returned malformed JSON: %s", exc)
        raise HTTPException(
            502, "Invalid response from Google token endpoint."
        ) from exc

    access_token = data.get("access_token")
    if not access_token:
        raise HTTPException(502, "Google returned no access token.")
    return access_token


async def _fetch_userinfo(google_access_token: str) -> dict[str, Any]:
    """Fetch the verified profile for the signed-in Google account."""
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
            resp = await client.get(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {google_access_token}"},
            )
        if resp.status_code != 200:
            logger.warning("Google userinfo returned %s", resp.status_code)
            raise HTTPException(502, "Could not read the Google profile.")

        info = resp.json()
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        logger.warning("Google userinfo endpoint unreachable: %s", exc)
        raise HTTPException(502, "Could not reach Google to read the profile.") from exc
    except ValueError as exc:
        logger.warning("Google userinfo endpoint returned malformed JSON: %s", exc)
        raise HTTPException(
            502, "Invalid response from Google userinfo endpoint."
        ) from exc

    if not info.get("sub") or not info.get("email"):
        raise HTTPException(502, "Google profile is missing an email address.")
    if not info.get("email_verified"):
        raise HTTPException(403, "Your Google email address is not verified.")
    return info


async def _unique_username(session: AsyncSession, email: str) -> str:
    """Derive an unused username from the email's local part."""
    base = re.sub(r"[^a-zA-Z0-9_.-]", "", email.split("@", 1)[0]) or "user"
    base = base[: _USERNAME_MAX - 5]
    candidate = base
    while (
        await session.execute(select(User.id).where(User.username == candidate))
    ).first():
        candidate = f"{base}-{secrets.token_hex(2)}"
    return candidate


async def upsert_google_user(session: AsyncSession, info: dict[str, Any]) -> User:
    """Create the user on first login, or refresh their record on later ones."""
    sub = str(info["sub"])
    email = str(info["email"]).lower().strip()
    name = (info.get("name") or "")[:255] or None
    picture = (info.get("picture") or "")[:2048] or None

    for attempt in range(2):
        try:
            result = await session.execute(select(User).where(User.google_id == sub))
            user = result.scalar_one_or_none()
            if user is None:
                result = await session.execute(select(User).where(User.email == email))
                user = result.scalar_one_or_none()

            if user is None:
                user = User(
                    username=await _unique_username(session, email),
                    email=email,
                    google_id=sub,
                    full_name=name,
                    picture_url=picture,
                )
                session.add(user)
            else:
                user.google_id = sub
                user.email = email
                if name:
                    user.full_name = name
                if picture:
                    user.picture_url = picture

            await session.commit()
            await session.refresh(user)
            return user
        except IntegrityError as exc:
            await session.rollback()
            if attempt == 1:
                logger.exception(
                    "Failed to upsert Google user %s due to integrity conflict", email
                )
                raise HTTPException(
                    409, "User account conflict during authentication."
                ) from exc


@router.get(
    "/google/callback",
    response_model=LoginResponse,
    summary="Finish Google login",
)
async def google_callback(
    request: Request,
    response: Response,
    session: Annotated[AsyncSession, Depends(get_db)],
    code: Annotated[str | None, Query()] = None,
    state: Annotated[str | None, Query()] = None,
    error: Annotated[str | None, Query()] = None,
) -> LoginResponse:
    """Exchange Google's authorization code for a JWT access + refresh pair."""
    _require_configured()

    if error:
        # e.g. ``access_denied`` when the user declines consent.
        status = 403 if error == "access_denied" else 400
        raise HTTPException(status, f"Google login failed: {error}.")

    expected = request.cookies.get(STATE_COOKIE)
    if not state or not expected or not secrets.compare_digest(state, expected):
        raise HTTPException(400, "Invalid or missing OAuth state.")
    if not code:
        raise HTTPException(400, "Missing authorization code.")

    google_token = await _exchange_code(code)
    info = await _fetch_userinfo(google_token)
    user = await upsert_google_user(session, info)

    # The state is single-use.
    response.delete_cookie(STATE_COOKIE)
    return LoginResponse(
        access_token=create_access_token(user.id, user.email),
        refresh_token=create_refresh_token(user.id, user.email),
        expires_in=settings.jwt_access_token_expire_minutes * 60,
        user=UserOut(
            id=user.id,
            email=user.email,
            username=user.username,
            created_at=user.created_at,
        ),
    )
