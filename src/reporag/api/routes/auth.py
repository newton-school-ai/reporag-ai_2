"""Google OAuth 2.0 authentication endpoints.

GET /auth/google          - Redirect to Google OAuth consent screen with CSRF state.
GET /auth/google/callback - Exchange authorization code for tokens, validate state,
                            upsert user, and return JWT access + refresh tokens.
"""

from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime
from typing import Annotated
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.api.middleware.auth import (
    AuthError,
    InvalidTokenError,
    TokenExpiredError,
    create_access_token,
    create_refresh_token,
    create_state_token,
    validate_state_token,
)
from reporag.config import settings
from reporag.db.models import User
from reporag.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class UserProfileResponse(BaseModel):
    """User profile data returned upon successful authentication."""

    id: int
    email: str
    username: str

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """Access and refresh token pair with user profile."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserProfileResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _upsert_user(session: AsyncSession, email: str, name: str | None) -> User:
    """Find an existing user by email or create a new user record."""
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is not None:
        user.updated_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(user)
        return user

    # Derive unique username (max 50 chars as defined in User model)
    base_username = (
        name.strip().replace(" ", "_").lower() if name else email.split("@")[0]
    )[:40]
    candidate_username = base_username or "user"

    # Check for username collision
    check_stmt = select(User).where(User.username == candidate_username)
    res = await session.execute(check_stmt)
    if res.scalar_one_or_none() is not None:
        candidate_username = f"{candidate_username[:40]}_{secrets.token_hex(4)}"

    user = User(
        username=candidate_username,
        email=email,
        hashed_password="oauth:google",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/google")
async def login_google() -> RedirectResponse:
    """Redirect to Google OAuth 2.0 consent screen.

    Builds the authorization URL with required scopes (openid, email, profile)
    and a generated CSRF state parameter. Also sets a secure HTTP-only cookie.
    """
    if not settings.google_client_id or settings.google_client_id.strip() in (
        "",
        "change-me",
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured on this server.",
        )

    state = create_state_token()
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    redirect_url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    response = RedirectResponse(url=redirect_url, status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key="oauth_state",
        value=state,
        httponly=True,
        max_age=600,
        samesite="lax",
    )
    return response


@router.get("/google/callback", response_model=TokenResponse)
async def google_callback(
    session: Annotated[AsyncSession, Depends(get_db)],
    request: Request,
    response: Response,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    error_description: str | None = Query(default=None),
) -> TokenResponse:
    """Exchange authorization code for tokens, fetch user info, and issue JWTs.

    Args:
        session: Database session dependency.
        request: FastAPI HTTP request to access cookies and client state.
        response: FastAPI HTTP response to clear cookies.
        code: Authorization code returned by Google.
        state: CSRF state parameter.
        error: OAuth error code (if user denied consent or error occurred).
        error_description: Human-readable error description from Google.

    Returns:
        TokenResponse containing JWT access and refresh tokens along with user info.
    """
    if error:
        detail = error_description or error
        logger.warning("Google OAuth error received in callback: %s", detail)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Google OAuth error: {detail}",
        )

    # 1. Validate state parameter to protect against CSRF and login hijacking
    if not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing OAuth state parameter",
        )

    try:
        validate_state_token(state)
    except TokenExpiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth state parameter has expired",
        ) from exc
    except (InvalidTokenError, AuthError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid OAuth state parameter: {exc}",
        ) from exc

    cookie_state = request.cookies.get("oauth_state")
    if cookie_state and not secrets.compare_digest(cookie_state, state):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth state does not match session state",
        )
    if "oauth_state" in request.cookies:
        response.delete_cookie(key="oauth_state")

    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization code",
        )

    # 2. Exchange authorization code for Google access token
    token_data = {
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret.get_secret_value(),
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": settings.google_redirect_uri,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            token_response = await client.post(GOOGLE_TOKEN_URL, data=token_data)
            if token_response.status_code != status.HTTP_200_OK:
                logger.error(
                    "Google token exchange failed (%s): %s",
                    token_response.status_code,
                    token_response.text,
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange authorization code with Google",
                )
            token_json = token_response.json()
            google_access_token = token_json.get("access_token")

            if not google_access_token:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No access token returned from Google",
                )

            # 3. Fetch user profile from Google userinfo endpoint
            headers = {"Authorization": f"Bearer {google_access_token}"}
            userinfo_response = await client.get(GOOGLE_USERINFO_URL, headers=headers)
            if userinfo_response.status_code != status.HTTP_200_OK:
                logger.error(
                    "Google userinfo request failed (%s): %s",
                    userinfo_response.status_code,
                    userinfo_response.text,
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to retrieve user information from Google",
                )
            userinfo = userinfo_response.json()

    except httpx.RequestError as exc:
        logger.error("Network error communicating with Google OAuth: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Network error communicating with Google authentication service",
        ) from exc

    email = userinfo.get("email")
    email_verified = userinfo.get("email_verified")

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account did not provide an email address",
        )

    # Reject logins where email_verified isn't explicitly True
    if email_verified is not True:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account email is not verified",
        )

    name = userinfo.get("name")

    # 4. Create or update user in database
    user = await _upsert_user(session, email=email, name=name)

    # 5. Issue JWT access and refresh tokens
    access_token = create_access_token(user_id=user.id, email=user.email)
    refresh_token = create_refresh_token(user_id=user.id, email=user.email)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=UserProfileResponse(
            id=user.id,
            email=user.email,
            username=user.username,
        ),
    )
