"""Authentication middleware and JWT utilities.

Google OAuth 2.0 login flow + JWT token issuance, validation, and refresh.
Provides token creation and decoding utilities, and in Issue 28, the
get_current_user dependency for protected routes.
"""

from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.config import settings
from reporag.db.models import User
from reporag.db.session import get_db

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"

# Default role granted to a token when the caller doesn't specify one. The
# User model has no roles column yet, so this is a claim on the token, not
# something read back off the database row -- it's forward compatible with
# a future roles/permissions table without another migration to this module.
DEFAULT_ROLES = ["user"]


class AuthError(Exception):
    """Base exception for authentication errors."""

    pass


class TokenExpiredError(AuthError):
    """Raised when a JWT token has expired."""

    pass


class InvalidTokenError(AuthError):
    """Raised when a JWT token is invalid, malformed, or has an invalid signature."""

    pass


def create_access_token(
    user_id: int,
    email: str,
    roles: list[str] | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a short-lived signed JWT access token.

    Args:
        user_id: Unique database ID of the authenticated user.
        email: Verified email address of the user.
        roles: Roles to embed in the token, used for coarse authorization
            checks without a database round trip. Defaults to
            ``DEFAULT_ROLES``.
        expires_delta: Optional custom lifetime. Defaults to
            ``settings.jwt_access_token_expire_minutes``.

    Returns:
        Encoded JWT string.
    """
    now = datetime.now(UTC)
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.jwt_access_token_expire_minutes)
    expire = now + expires_delta

    payload = {
        "sub": str(user_id),
        "email": email,
        "roles": roles if roles is not None else list(DEFAULT_ROLES),
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(
        payload,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=ALGORITHM,
    )


def create_refresh_token(
    user_id: int,
    email: str,
    roles: list[str] | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a long-lived signed JWT refresh token.

    Args:
        user_id: Unique database ID of the authenticated user.
        email: Verified email address of the user.
        roles: Roles to carry over onto the next access token minted from
            this refresh token. Defaults to ``DEFAULT_ROLES``.
        expires_delta: Optional custom lifetime. Defaults to
            ``settings.jwt_refresh_token_expire_days``.

    Returns:
        Encoded JWT string.
    """
    now = datetime.now(UTC)
    if expires_delta is None:
        expires_delta = timedelta(days=settings.jwt_refresh_token_expire_days)
    expire = now + expires_delta

    payload = {
        "sub": str(user_id),
        "email": email,
        "roles": roles if roles is not None else list(DEFAULT_ROLES),
        "type": "refresh",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(
        payload,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=ALGORITHM,
    )


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a signed JWT token.

    Args:
        token: Encoded JWT string.

    Returns:
        Decoded payload dictionary.

    Raises:
        TokenExpiredError: If the token's ``exp`` timestamp is in the past.
        InvalidTokenError: If the token signature is invalid, claims are
            missing, or formatting is malformed.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[ALGORITHM],
        )
        if "sub" not in payload or "email" not in payload or "type" not in payload:
            raise InvalidTokenError("Token payload missing required claims")
        return payload
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpiredError("Token has expired") from exc
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(f"Invalid token: {exc}") from exc


def create_state_token(expires_delta: timedelta | None = None) -> str:
    """Create a signed, tamper-proof OAuth state token with expiration.

    Args:
        expires_delta: Optional custom lifetime. Defaults to 10 minutes.

    Returns:
        Encoded JWT state string.
    """
    now = datetime.now(UTC)
    if expires_delta is None:
        expires_delta = timedelta(minutes=10)
    expire = now + expires_delta

    payload = {
        "type": "oauth_state",
        "nonce": secrets.token_urlsafe(16),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(
        payload,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=ALGORITHM,
    )


def validate_state_token(state: str | None) -> dict[str, Any]:
    """Validate a signed OAuth state token.

    Args:
        state: State token string received in OAuth callback.

    Returns:
        Decoded payload dictionary.

    Raises:
        InvalidTokenError: If the state is missing, invalid, or type claim is incorrect.
        TokenExpiredError: If the state token has expired.
    """
    if not state or not state.strip():
        raise InvalidTokenError("Missing state token")
    try:
        payload = jwt.decode(
            state,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[ALGORITHM],
        )
        if payload.get("type") != "oauth_state":
            raise InvalidTokenError("Invalid state token type")
        return payload
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpiredError("OAuth state parameter has expired") from exc
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(f"Invalid OAuth state parameter: {exc}") from exc


# ---------------------------------------------------------------------------
# Request-time validation: get_current_user dependency (Issue 28)
# ---------------------------------------------------------------------------

# ``auto_error=False`` so a missing header falls through to our own 401 with
# a consistent error body, instead of FastAPI's default "Not authenticated"
# shape from HTTPBearer's built-in error handling.
_bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    """Build a 401 with the ``WWW-Authenticate`` header a bearer scheme owes.

    Every rejection in ``get_current_user`` -- missing header, malformed
    token, expired token, unknown user -- returns exactly this shape so a
    client can branch on status code alone instead of parsing ``detail``.
    """
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)
    ],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Resolve the authenticated user from the request's bearer token.

    Injectable into any route via ``Depends(get_current_user)`` to make it
    a protected route: FastAPI resolves this dependency before the route
    body runs, so a request with no token, an expired token, or a token for
    a deleted user never reaches the handler -- it short-circuits with 401.

    Only access tokens are accepted here; a refresh token presented as a
    bearer credential is rejected, since refresh tokens are only meant to
    be exchanged at ``POST /auth/refresh``, not used to authenticate a
    request directly.

    Args:
        credentials: Bearer credentials extracted from the ``Authorization``
            header, or ``None`` when the header is absent or malformed.
        session: Database session dependency, used to load the user the
            token's ``sub`` claim refers to.

    Returns:
        The authenticated :class:`~reporag.db.models.User`.

    Raises:
        HTTPException: 401 if the token is missing, invalid, expired, not
            an access token, or no longer refers to an existing user.
    """
    if credentials is None or not credentials.credentials:
        raise _unauthorized("Not authenticated")

    try:
        payload = decode_token(credentials.credentials)
    except TokenExpiredError as exc:
        raise _unauthorized("Token has expired") from exc
    except InvalidTokenError as exc:
        raise _unauthorized(f"Invalid authentication token: {exc}") from exc

    if payload.get("type") != "access":
        raise _unauthorized("Token is not an access token")

    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _unauthorized("Invalid authentication token") from exc

    user = await session.get(User, user_id)
    if user is None:
        raise _unauthorized("User no longer exists")

    return user
