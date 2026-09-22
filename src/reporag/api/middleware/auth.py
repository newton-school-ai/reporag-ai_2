"""Authentication middleware.

Google OAuth 2.0 login flow + JWT token issuance, validation, and refresh.
Provides get_current_user dependency for protected routes.

Why
---
Issue 27 needs to hand a caller a session after Google confirms who they
are, so this module owns token *issuance*. Validation, the refresh endpoint
and the ``get_current_user`` dependency belong to Issue 28 and are not
built here.

Design
------
* **Two token types, one secret.** Access and refresh tokens are both HS256
  JWTs signed with ``JWT_SECRET_KEY``, told apart by a ``type`` claim so a
  long-lived refresh token can never be replayed as an access token.
* **Claims are minimal.** ``sub`` (user id), ``email``, ``iat``, ``exp`` and
  ``type``. Anything else would go stale the moment the user record changes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import jwt

from reporag.config import settings

JWT_ALGORITHM = "HS256"

TokenType = Literal["access", "refresh"]


def _create_token(
    user_id: int, email: str, token_type: TokenType, lifetime: timedelta
) -> str:
    """Sign a JWT for ``user_id`` that expires ``lifetime`` from now."""
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        # RFC 7519 requires ``sub`` to be a string.
        "sub": str(user_id),
        "email": email,
        "type": token_type,
        "iat": now,
        "exp": now + lifetime,
    }
    return jwt.encode(
        payload, settings.jwt_secret_key.get_secret_value(), algorithm=JWT_ALGORITHM
    )


def create_access_token(user_id: int, email: str) -> str:
    """Issue a short-lived access token (``JWT_ACCESS_TOKEN_EXPIRE_MINUTES``)."""
    return _create_token(
        user_id,
        email,
        "access",
        timedelta(minutes=settings.jwt_access_token_expire_minutes),
    )


def create_refresh_token(user_id: int, email: str) -> str:
    """Issue a long-lived refresh token (``JWT_REFRESH_TOKEN_EXPIRE_DAYS``)."""
    return _create_token(
        user_id,
        email,
        "refresh",
        timedelta(days=settings.jwt_refresh_token_expire_days),
    )
