from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.api.middleware.auth import create_access_token, create_refresh_token
from reporag.config import settings
from reporag.db.models import User
from reporag.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"

COOKIE_NAME = "reporag_oauth_tx"
COOKIE_MAX_AGE = 600

# Cache the PyJWKClient instance across requests
_jwks_client = jwt.PyJWKClient(GOOGLE_JWKS_URL)


async def get_http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provide an HTTP client that gets closed with the request."""
    async with httpx.AsyncClient() as client:
        yield client


def _pkce_challenge(verifier: str) -> str:
    """Generate a PKCE S256 challenge from a verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _seal_transaction(state: str, verifier: str, nonce: str) -> str:
    """Seal the OAuth transaction state into a signed JWT."""
    now = datetime.now(UTC)
    payload = {
        "state": state,
        "verifier": verifier,
        "nonce": nonce,
        "iss": "reporag",
        "aud": "reporag-oauth-tx",
        "iat": now,
        "exp": now + timedelta(seconds=COOKIE_MAX_AGE),
    }
    return jwt.encode(
        payload, settings.jwt_secret_key.get_secret_value(), algorithm="HS256"
    )


def _open_transaction(token: str) -> dict:
    """Verify and decode a transaction JWT."""
    try:
        return jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=["HS256"],
            issuer="reporag",
            audience="reporag-oauth-tx",
            leeway=10,
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid transaction cookie: {e}",
        ) from e


def _require_oauth_configured() -> None:
    """Ensure Google OAuth is configured before proceeding."""
    if (
        not settings.google_client_id
        or not settings.google_client_secret.get_secret_value()
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured on this server.",
        )


def _unique_username(email: str) -> str:
    """Derive a base username from the email local part."""
    base = email.split("@")[0]
    # Remove special characters
    return "".join(c.lower() for c in base if c.isalnum())


async def _upsert_user(
    db: AsyncSession,
    google_id: str,
    email: str,
    full_name: str | None,
    avatar_url: str | None,
) -> User:
    """Create or update a User based on their Google identity."""
    # 1. Match on google_id
    stmt = select(User).where(User.google_id == google_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user:
        user.email = email
        user.full_name = full_name
        user.avatar_url = avatar_url
        user.last_login_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(user)
        return user

    # 2. Match on email (adoption)
    stmt = select(User).where(User.email == email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user:
        if user.google_id and user.google_id != google_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email is associated with a different Google account.",
            )
        user.google_id = google_id
        user.full_name = full_name
        user.avatar_url = avatar_url
        user.last_login_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(user)
        return user

    # 3. Create a new user
    base_username = _unique_username(email)
    username = base_username
    counter = 1

    while True:
        # Check username uniqueness
        u_stmt = select(User).where(User.username == username)
        u_res = await db.execute(u_stmt)
        if not u_res.scalar_one_or_none():
            break
        username = f"{base_username[:45]}{counter}"  # Trim to fit String(50)
        counter += 1

    user = User(
        email=email,
        username=username,
        google_id=google_id,
        full_name=full_name,
        avatar_url=avatar_url,
        last_login_at=datetime.now(UTC),
    )
    db.add(user)

    try:
        await db.commit()
    except IntegrityError:
        # Handle race condition where another request created the user
        await db.rollback()
        # Fall back to adopting the newly created user
        return await _upsert_user(db, google_id, email, full_name, avatar_url)

    await db.refresh(user)
    return user


async def _exchange_code(client: httpx.AsyncClient, code: str, verifier: str) -> dict:
    """Exchange the authorization code for an access token and ID token."""
    data = {
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret.get_secret_value(),
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": settings.google_redirect_uri,
        "code_verifier": verifier,
    }
    try:
        response = await client.post(GOOGLE_TOKEN_URL, data=data)
    except httpx.RequestError as e:
        logger.error(f"Failed to reach Google token endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach identity provider.",
        ) from e

    if response.status_code != 200:
        logger.error(f"Google token exchange rejected: {response.text}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to exchange authorization code.",
        )
    return response.json()


def _verify_id_token(id_token: str, nonce: str) -> dict:
    """Verify the ID token cryptographically using Google's JWKS."""
    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(id_token)
        payload = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.google_client_id,
            issuer=["accounts.google.com", "https://accounts.google.com"],
            leeway=10,
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid ID token: {e}",
        ) from e

    if payload.get("nonce") != nonce:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid nonce in ID token.",
        )

    email = payload.get("email")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ID token does not contain an email.",
        )

    if not payload.get("email_verified"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email is not verified by Google.",
        )

    return payload


async def _fetch_profile(client: httpx.AsyncClient, access_token: str) -> dict:
    """Fetch profile details from Google userinfo."""
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        response = await client.get(GOOGLE_USERINFO_URL, headers=headers)
        if response.status_code == 200:
            return response.json()
    except httpx.RequestError:
        pass
    return {}


@router.get("/auth/google")
async def google_login() -> Response:
    """Redirect to Google OAuth consent screen."""
    _require_oauth_configured()

    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    nonce = secrets.token_urlsafe(32)

    challenge = _pkce_challenge(verifier)

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "prompt": "select_account",
    }

    url = f"{GOOGLE_AUTH_URL}?" + "&".join(f"{k}={v}" for k, v in params.items())
    response = RedirectResponse(url, status_code=307)

    tx_token = _seal_transaction(state, verifier, nonce)
    response.set_cookie(
        key=COOKIE_NAME,
        value=tx_token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=settings.is_production,
        path="/auth",
    )
    return response


@router.get("/auth/google/callback")
async def google_callback(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),  # noqa: B008
    http_client: httpx.AsyncClient = Depends(get_http_client),  # noqa: B008
) -> dict:
    """Exchange code for tokens and authenticate user."""
    try:
        content = await _complete_login(request, db, http_client)
        response.delete_cookie(COOKIE_NAME, path="/auth", httponly=True, samesite="lax")
        return content
    except HTTPException as e:
        e.headers = e.headers or {}
        e.headers["Set-Cookie"] = (
            f"{COOKIE_NAME}=; Path=/auth; Max-Age=0; HttpOnly; SameSite=lax"
        )
        raise e


async def _complete_login(
    request: Request, db: AsyncSession, http_client: httpx.AsyncClient
) -> dict:
    # Check for error from Google
    error = request.query_params.get("error")
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth error: {error}",
        )

    code = request.query_params.get("code")
    query_state = request.query_params.get("state")
    if not code or not query_state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing code or state.",
        )

    tx_cookie = request.cookies.get(COOKIE_NAME)
    if not tx_cookie:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transaction cookie missing.",
        )

    tx = _open_transaction(tx_cookie)
    if not secrets.compare_digest(tx["state"], query_state):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="State mismatch.",
        )

    token_data = await _exchange_code(http_client, code, tx["verifier"])

    id_token = token_data.get("id_token")
    if not id_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing ID token.",
        )

    id_payload = _verify_id_token(id_token, tx["nonce"])
    google_id = id_payload["sub"]
    email = id_payload["email"]

    # Fall back to userinfo if profile data isn't in ID token
    full_name = id_payload.get("name")
    avatar_url = id_payload.get("picture")

    access_token = token_data.get("access_token")
    if access_token and (not full_name or not avatar_url):
        profile = await _fetch_profile(http_client, access_token)
        full_name = full_name or profile.get("name")
        avatar_url = avatar_url or profile.get("picture")

    user = await _upsert_user(db, google_id, email, full_name, avatar_url)

    jwt_access_token = create_access_token(user)
    jwt_refresh_token = create_refresh_token(user)

    # Note: we need to delete the cookie on success as well.
    # We can use a Response object.
    response_content = {
        "access_token": jwt_access_token,
        "refresh_token": jwt_refresh_token,
        "token_type": "bearer",
        "expires_in": settings.jwt_access_token_expire_minutes * 60,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "avatar_url": user.avatar_url,
        },
    }

    # We will return the dict and rely on a middleware or let the user fetch it.
    # Wait, FastAPI doesn't easily let us set cookies when returning a dict,
    # unless we use Response as a parameter and modify it. Let's do that.
    # We'll pass `response: Response` to the route instead.

    return response_content
