"""JWT issuance and verification.

The token layer behind the Google OAuth login flow in
:mod:`reporag.api.routes.auth`. Issue 27 needs it to mint the access and
refresh tokens the callback returns; Issue 28 builds the validation
middleware, the ``get_current_user`` dependency and the refresh endpoint on
top of the primitives here.

Why
---
Sessions would mean shared server-side state, which the API deliberately
does not have -- it can run as several replicas behind a load balancer with
no sticky routing. A signed token carries the identity instead, so any
replica can verify a request without a lookup.

Design
------
* **Two token types, one secret, one signature check.** An access token is
  short-lived and authorises requests; a refresh token is long-lived and
  does nothing but obtain a new access token. Both carry a ``type`` claim
  and :func:`decode_token` demands the one the caller expects. Without that
  check a refresh token -- which lives for days, by design -- would be
  accepted as a bearer credential, quietly turning the short access
  lifetime into a week.
* **Claims are the contract.** ``sub`` (user id, a string per RFC 7519),
  ``email``, ``iat``, ``exp``, ``iss``, ``aud``, ``jti``, ``type``. Issuer
  and audience are verified, not merely present: a token minted by another
  service that happens to share the secret must not authenticate here.
* **Failures are typed, not boolean.** An expired token and a forged one
  need different responses -- the first tells a client to refresh, the
  second is an attack or a bug -- so they raise distinct exceptions rather
  than both returning ``None``.
* **Clock skew is tolerated, expiry is not.** A few seconds of leeway
  absorbs drift between hosts; beyond that an expired token is expired.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import jwt

from reporag.config import settings

logger = logging.getLogger(__name__)

# HMAC rather than RS256: the API is both the issuer and the only verifier,
# so there is no third party needing a public key, and a shared secret has
# no key distribution problem to solve.
ALGORITHM = "HS256"

# Verified on every decode. A token minted for another service, or by an
# older deployment with a different audience, is rejected even when the
# signature checks out.
ISSUER = "reporag"
AUDIENCE = "reporag-api"

# Absorbs clock drift between the host that signed a token and the host
# verifying it. Small enough that an expired token stays expired.
_LEEWAY_SECONDS = 10


class TokenType(StrEnum):
    """What a token is allowed to do.

    Attributes:
        ACCESS: Authorises API requests. Short-lived.
        REFRESH: Obtains a new access token, and nothing else.
    """

    ACCESS = "access"
    REFRESH = "refresh"


class TokenError(Exception):
    """Base class for every token failure."""


class TokenExpiredError(TokenError):
    """The token was well-formed and correctly signed, but has expired.

    Distinct from :class:`InvalidTokenError` because the remedy differs: a
    client seeing this should refresh, not re-authenticate from scratch.
    """


class InvalidTokenError(TokenError):
    """The token is malformed, mis-signed, or not the type that was expected."""


@dataclass(frozen=True)
class TokenClaims:
    """The verified contents of a token.

    Attributes:
        user_id: Database id of the authenticated user.
        email: Email recorded at issuance time.
        token_type: Which kind of token this was.
        issued_at: When it was minted.
        expires_at: When it stops being accepted.
        jti: Unique token id. Issue 29 can use it for revocation lists.
    """

    user_id: int
    email: str
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime
    jti: str


def _create_token(
    *,
    user_id: int,
    email: str,
    token_type: TokenType,
    lifetime: timedelta,
) -> str:
    """Mint a signed token. Shared by both public factories.

    Args:
        user_id: Database id to record as ``sub``.
        email: Email to record.
        token_type: Which kind of token to mint.
        lifetime: How long it stays valid.

    Returns:
        The encoded JWT.
    """
    now = datetime.now(UTC)
    payload = {
        # RFC 7519 requires `sub` to be a string; PyJWT enforces this on
        # decode, so an int here would mint tokens this module cannot read.
        "sub": str(user_id),
        "email": email,
        "type": token_type.value,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + lifetime,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(
        payload, settings.jwt_secret_key.get_secret_value(), algorithm=ALGORITHM
    )


def create_access_token(user_id: int, email: str) -> str:
    """Mint a short-lived token that authorises API requests.

    Args:
        user_id: Database id of the authenticated user.
        email: Their email, carried so common authorisation checks need no
            database round trip.

    Returns:
        The encoded JWT, valid for ``JWT_ACCESS_TOKEN_EXPIRE_MINUTES``.
    """
    return _create_token(
        user_id=user_id,
        email=email,
        token_type=TokenType.ACCESS,
        lifetime=timedelta(minutes=settings.jwt_access_token_expire_minutes),
    )


def create_refresh_token(user_id: int, email: str) -> str:
    """Mint a long-lived token whose only use is obtaining an access token.

    Args:
        user_id: Database id of the authenticated user.
        email: Their email at issuance time.

    Returns:
        The encoded JWT, valid for ``JWT_REFRESH_TOKEN_EXPIRE_DAYS``.
    """
    return _create_token(
        user_id=user_id,
        email=email,
        token_type=TokenType.REFRESH,
        lifetime=timedelta(days=settings.jwt_refresh_token_expire_days),
    )


def decode_token(token: str, expected_type: TokenType) -> TokenClaims:
    """Verify a token and return its claims.

    Checks the signature, the issuer, the audience, the expiry and the token
    type. *expected_type* is required rather than optional so that accepting
    the wrong kind of token has to be a deliberate act: a refresh token
    presented as a bearer credential is rejected here, not several layers
    later where the distinction is easy to forget.

    Args:
        token: The encoded JWT.
        expected_type: The kind of token the caller requires.

    Returns:
        The verified claims.

    Raises:
        TokenExpiredError: The token is past its expiry.
        InvalidTokenError: The signature, issuer, audience, structure or
            type is wrong.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            audience=AUDIENCE,
            leeway=_LEEWAY_SECONDS,
            options={"require": ["sub", "exp", "iat", "iss", "aud", "jti"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpiredError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        # Covers a bad signature, a wrong issuer or audience, a missing
        # required claim, and anything malformed.
        raise InvalidTokenError(f"Token is invalid: {exc}") from exc

    actual_type = payload.get("type")
    if actual_type != expected_type.value:
        raise InvalidTokenError(
            f"Expected a {expected_type.value} token, got {actual_type!r}."
        )

    try:
        user_id = int(payload["sub"])
    except (TypeError, ValueError) as exc:
        raise InvalidTokenError("Token subject is not a user id.") from exc

    return TokenClaims(
        user_id=user_id,
        email=payload.get("email", ""),
        token_type=expected_type,
        issued_at=datetime.fromtimestamp(payload["iat"], tz=UTC),
        expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC),
        jti=payload["jti"],
    )
