"""Authentication.

Google OAuth 2.0 login flow + JWT token issuance, validation, and refresh.
Provides get_current_user dependency for protected routes.

Scope
-----
Issue 27 (this slice) only *issues* tokens: ``create_access_token`` and
``create_refresh_token``, called by ``routes/auth.py`` once Google OAuth has
confirmed an identity. Issue 28 depends on this file and adds the other
half -- ``decode_token``, the ``get_current_user`` dependency that protects
every other route, 401 handling for missing/expired/invalid tokens, and
``POST /auth/refresh``.

Design
------
* **`type` is part of the payload, not an afterthought.** An access token
  and a refresh token are minted by the same function shape and share every
  claim except this one. Without it, Issue 28's refresh endpoint would have
  no way to reject an access token presented as a refresh token (or vice
  versa) -- a leaked short-lived access token could otherwise be replayed
  indefinitely against ``POST /auth/refresh``.
* **Claim names match the tracker's acceptance criteria literally**
  (``user_id``, ``email``, not the JWT-conventional ``sub``), since Issue 28
  will read these fields by name.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import jwt

from reporag.config import settings
from reporag.db.models import User

# HS256 (symmetric, JWT_SECRET_KEY) rather than RS256: there is exactly one
# issuer and one verifier (this API), so there's nothing an asymmetric key
# pair would buy that a shared secret doesn't already provide more simply.
JWT_ALGORITHM = "HS256"

TokenType = Literal["access", "refresh"]


class TokenError(Exception):
    """Base class for token validation errors."""

    pass


class TokenExpiredError(TokenError):
    """Raised when a token has expired."""

    pass


class InvalidTokenError(TokenError):
    """Raised when a token is invalid (forged, wrong issuer, bad audience, etc.)."""

    pass


@dataclass(frozen=True)
class TokenClaims:
    """The claims contained in a validated JWT."""

    sub: str
    email: str
    type: TokenType


def _create_token(user: User, token_type: TokenType, expires_delta: timedelta) -> str:
    """Encode a signed JWT identifying *user*.

    Args:
        user: The account the token authenticates.
        token_type: ``"access"`` or ``"refresh"`` -- carried in the payload
            so a future validator can tell them apart.
        expires_delta: How long the token is valid for.

    Returns:
        A signed, encoded JWT string.
    """
    now = datetime.now(UTC)
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "type": token_type,
        "iss": "reporag",
        "aud": "reporag-api",
        # HS256 signing is deterministic, and `iat` only has second
        # precision -- two tokens minted for the same user in the same
        # second would otherwise be byte-identical. `jti` also gives
        # Issue 28 a stable per-token id to key a revocation list on,
        # if/when one is needed.
        "jti": secrets.token_hex(16),
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(
        payload, settings.jwt_secret_key.get_secret_value(), algorithm=JWT_ALGORITHM
    )


def create_access_token(user: User) -> str:
    """Mint a short-lived token proving *user*'s identity on API requests.

    Expiry is ``JWT_ACCESS_TOKEN_EXPIRE_MINUTES`` (default 30) -- short on
    purpose, since this is the token attached to every request and the one
    most exposed to leaking (logs, browser storage, a proxy).
    """
    return _create_token(
        user, "access", timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )


def create_refresh_token(user: User) -> str:
    """Mint a long-lived token exchangeable for a new access token.

    Expiry is ``JWT_REFRESH_TOKEN_EXPIRE_DAYS`` (default 7). Issue 28's
    ``POST /auth/refresh`` is what actually exchanges one of these for a new
    access token; nothing here validates a token, only issues one.
    """
    return _create_token(
        user, "refresh", timedelta(days=settings.jwt_refresh_token_expire_days)
    )


def decode_token(token: str, expected_type: TokenType) -> TokenClaims:
    """Verify a token's signature and claims.

    Args:
        token: The encoded JWT string.
        expected_type: The expected token type ("access" or "refresh").

    Raises:
        TokenExpiredError: If the token has expired.
        InvalidTokenError: If the token is invalid (forged, bad issuer, etc.).

    Returns:
        The validated claims.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[JWT_ALGORITHM],
            issuer="reporag",
            audience="reporag-api",
            leeway=10,  # 10 seconds leeway for clock skew
        )
    except jwt.ExpiredSignatureError as e:
        raise TokenExpiredError("Token has expired") from e
    except jwt.InvalidTokenError as e:
        raise InvalidTokenError(f"Invalid token: {e}") from e

    token_type = payload.get("type")
    if token_type != expected_type:
        raise InvalidTokenError(
            f"Expected token type '{expected_type}', got '{token_type}'"
        )

    sub = payload.get("sub")
    email = payload.get("email")
    if not sub or not email:
        raise InvalidTokenError("Token missing required 'sub' or 'email' claims")

    return TokenClaims(sub=sub, email=email, type=expected_type)
