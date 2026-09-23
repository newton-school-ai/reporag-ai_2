"""Authentication middleware and JWT utilities.

Google OAuth 2.0 login flow + JWT token issuance, validation, and refresh.
Provides token creation and decoding utilities, and in Issue 28, the
get_current_user dependency for protected routes.
"""

from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from reporag.config import settings

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"


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
    expires_delta: timedelta | None = None,
) -> str:
    """Create a short-lived signed JWT access token.

    Args:
        user_id: Unique database ID of the authenticated user.
        email: Verified email address of the user.
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
    expires_delta: timedelta | None = None,
) -> str:
    """Create a long-lived signed JWT refresh token.

    Args:
        user_id: Unique database ID of the authenticated user.
        email: Verified email address of the user.
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
