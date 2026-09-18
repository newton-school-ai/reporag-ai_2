"""Google OAuth 2.0 login.

GET /auth/google          - Redirect to Google's consent screen.
GET /auth/google/callback - Exchange the code, upsert the user, return JWTs.

Why
---
Everything before this issue attributed repositories to a placeholder
account, because ``Repository.owner_id`` is a non-nullable foreign key and
there was no authenticated principal. This is where a real one arrives:
Google verifies the human, and the callback trades that for the API's own
tokens, after which the platform is the only thing holding a credential a
user has to trust.

Design
------
* **Authorization Code with PKCE.** The proof key is not optional here even
  though this is a confidential client with a secret. An authorization code
  travels through the user's browser and is written to server logs,
  ``Referer`` headers and browser history along the way; PKCE binds the code
  to the transaction that requested it, so a code captured in transit is
  useless to anyone who cannot also produce the verifier.
* **The OAuth transaction is a signed cookie, not a session.** ``state``,
  the PKCE verifier and the nonce have to survive a round trip through
  Google and come back trustworthy. A server-side session would give the
  API the shared state it is built to avoid -- with several replicas behind
  a load balancer, the callback can land on a different process than the
  redirect. Signing the transaction and handing it to the browser keeps
  every replica able to verify it with no store and no sticky routing.
* **The ID token is the identity; userinfo only decorates it.** The ID
  token is verified against Google's published keys, so its ``sub`` and
  ``email`` are proof. The userinfo response is an ordinary HTTP body with
  no signature of its own, so it is used for display name and avatar and
  never for identity, and a failure to fetch it does not fail the login.
* **Accounts are keyed on ``sub``, not email.** Google promises ``sub`` is
  stable and never reused; an email address is neither. Matching on email
  would hand one person's repositories to whoever inherits a recycled
  Workspace address.
* **Errors distinguish the user from the provider.** A declined consent is
  a 400 the user caused; Google being unreachable is a 502 they can retry.
  Collapsing both into 500 would tell the caller nothing about which.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from urllib.parse import urlencode

import httpx
import jwt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Query, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from reporag.api.middleware.auth import (
    ALGORITHM,
    ISSUER,
    create_access_token,
    create_refresh_token,
)
from reporag.config import settings
from reporag.db.models import User
from reporag.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Google's OpenID Connect endpoints. Hard-coded rather than read from the
# discovery document: fetching /.well-known/openid-configuration on every
# login would add a network round trip to the critical path to learn URLs
# that have not changed in a decade, and a discovery failure would take the
# login flow down with it.
GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_JWKS_URI = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"

# Google signs ID tokens with this issuer. Both spellings are documented as
# valid and appear in the wild, so both are accepted.
_GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")

# openid gives the ID token, email gives an identity worth keying on, and
# profile gives a display name. Nothing more is requested: an authorisation
# screen asking for more than the product uses is the fastest way to make
# people decline it.
_SCOPES = ("openid", "email", "profile")

# Name of the cookie holding the signed OAuth transaction.
_TX_COOKIE = "reporag_oauth_tx"

# Distinct audience so a transaction cookie can never be replayed as an API
# access token, even though both are signed with the same secret.
_TX_AUDIENCE = "reporag-oauth-tx"

# How long a login attempt may sit on the consent screen. Long enough for
# someone to pick an account and read the prompt, short enough that an
# abandoned transaction cookie stops being useful quickly.
_TX_TTL_SECONDS = 600

# Bounds every call to Google. Without one a hung provider would hold the
# request open until the client gives up.
_HTTP_TIMEOUT_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    """The authenticated user, as returned to the client.

    Attributes:
        id: Database id, used as ``sub`` in the issued tokens.
        email: Verified email address from Google.
        username: Generated handle, unique across accounts.
        full_name: Display name, when Google supplied one.
        avatar_url: Profile picture URL, when Google supplied one.
    """

    id: int
    email: str
    username: str
    full_name: str | None = None
    avatar_url: str | None = None

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """Body of a successful ``GET /auth/google/callback``.

    Attributes:
        access_token: Bearer token authorising API requests.
        refresh_token: Long-lived token used to obtain a new access token.
        token_type: Always ``bearer``, per RFC 6750.
        expires_in: Access token lifetime in seconds.
        user: The authenticated account.
    """

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(
        ..., description="Access token lifetime in seconds.", examples=[1800]
    )
    user: UserProfile


@dataclass(frozen=True)
class GoogleIdentity:
    """A verified Google account.

    Attributes:
        sub: Google's stable, never-reused account identifier.
        email: Verified email address.
        full_name: Display name, if Google supplied one.
        avatar_url: Profile picture URL, if Google supplied one.
    """

    sub: str
    email: str
    full_name: str | None = None
    avatar_url: str | None = None


# ---------------------------------------------------------------------------
# HTTP client (injectable so tests never reach the network)
# ---------------------------------------------------------------------------


async def get_http_client() -> Any:
    """Yield the HTTP client used to talk to Google.

    A dependency rather than a module-level client so tests can override it
    with a transport that answers without a socket, and so the client is
    closed when the request finishes.
    """
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
        yield client


HttpClient = Annotated[httpx.AsyncClient, Depends(get_http_client)]


# ---------------------------------------------------------------------------
# OAuth transaction: state, PKCE and nonce
# ---------------------------------------------------------------------------


def _pkce_challenge(verifier: str) -> str:
    """Derive the S256 PKCE challenge from *verifier*.

    Base64url of the SHA-256 digest with padding stripped, as RFC 7636
    requires. ``plain`` is not offered: it would put the verifier itself in
    the redirect URL, defeating the point.
    """
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _seal_transaction(state: str, verifier: str, nonce: str) -> str:
    """Sign the transaction secrets for the round trip through the browser.

    The browser holds this but cannot read anything into it that would
    verify: the signature is what makes the returned ``state`` trustworthy.
    """
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "state": state,
            "verifier": verifier,
            "nonce": nonce,
            "iss": ISSUER,
            "aud": _TX_AUDIENCE,
            "iat": now,
            "exp": now + timedelta(seconds=_TX_TTL_SECONDS),
        },
        settings.jwt_secret_key.get_secret_value(),
        algorithm=ALGORITHM,
    )


def _open_transaction(sealed: str) -> dict[str, Any]:
    """Verify and unseal a transaction cookie.

    Raises:
        HTTPException: 400 when the cookie is expired, forged or malformed.
            All three are one message to the caller -- a legitimate user can
            only act on "start again", and distinguishing them would tell an
            attacker which part of the forgery failed.
    """
    try:
        return jwt.decode(
            sealed,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            audience=_TX_AUDIENCE,
            options={"require": ["state", "verifier", "nonce", "exp"]},
        )
    except jwt.InvalidTokenError as exc:
        logger.warning("Rejected OAuth transaction cookie: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Login session expired or invalid. Start again at /auth/google.",
        ) from exc


def _require_oauth_configured() -> None:
    """Fail loudly when the deployment has no Google credentials.

    Raises:
        HTTPException: 503, matching how the query route reports an
            unconfigured component. Letting the flow proceed would send the
            user to Google only for Google to reject an empty client id,
            with the real cause two systems away from the error they see.
    """
    if (
        not settings.google_client_id
        or not settings.google_client_secret.get_secret_value()
    ):
        logger.error("Google OAuth is not configured; refusing to start a login")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Google sign-in is not configured on this server. "
                "See GET /api/v1/health for component status."
            ),
        )


# ---------------------------------------------------------------------------
# Google calls
# ---------------------------------------------------------------------------


async def _exchange_code(client: httpx.AsyncClient, code: str, verifier: str) -> dict:
    """Trade the authorization code for tokens.

    Args:
        client: HTTP client to use.
        code: The authorization code Google sent to the callback.
        verifier: The PKCE verifier proving this is the same transaction
            that requested the code.

    Returns:
        Google's token response.

    Raises:
        HTTPException: 400 when Google rejects the code, 502 when Google
            cannot be reached.
    """
    try:
        response = await client.post(
            GOOGLE_TOKEN_ENDPOINT,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret.get_secret_value(),
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
                "code_verifier": verifier,
            },
            headers={"Accept": "application/json"},
        )
    except httpx.HTTPError as exc:
        logger.warning("Token exchange could not reach Google: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach Google to complete sign-in. Try again.",
        ) from exc

    if response.status_code != 200:
        # Google's body names the reason (invalid_grant for a reused or
        # expired code, redirect_uri_mismatch for a misconfigured console).
        # It goes to the log; the caller gets a message they can act on.
        logger.warning(
            "Google rejected the authorization code (%s): %s",
            response.status_code,
            response.text[:500],
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google rejected the sign-in attempt. Start again at /auth/google.",
        )

    return response.json()


def _verify_id_token(id_token: str, nonce: str) -> dict:
    """Verify an ID token against Google's published signing keys.

    Checks the signature, issuer, audience, expiry and the nonce bound to
    this transaction. The nonce check is what stops a valid ID token
    obtained elsewhere from being replayed into someone else's login.

    Blocking: fetches and caches Google's JWKS over the network, so callers
    must dispatch it to a worker thread.

    Args:
        id_token: The encoded ID token from the token response.
        nonce: The nonce this transaction sent to Google.

    Returns:
        The verified claims.

    Raises:
        HTTPException: 401 when the token does not verify.
    """
    try:
        # PyJWKClient caches the key set, so a second login reuses it.
        signing_key = jwt.PyJWKClient(GOOGLE_JWKS_URI).get_signing_key_from_jwt(
            id_token
        )
        claims = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.google_client_id,
            issuer=list(_GOOGLE_ISSUERS),
            options={"require": ["sub", "aud", "iss", "exp", "iat"]},
        )
    except jwt.InvalidTokenError as exc:
        logger.warning("Google ID token failed verification: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google sign-in could not be verified.",
        ) from exc
    except Exception as exc:
        # A JWKS fetch failure is Google being unreachable, not a bad token.
        logger.warning("Could not verify the Google ID token: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach Google to verify sign-in. Try again.",
        ) from exc

    if claims.get("nonce") != nonce:
        logger.warning("Google ID token nonce did not match the transaction")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google sign-in could not be verified.",
        )

    if not claims.get("email"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google did not return an email address for this account.",
        )

    if not claims.get("email_verified", False):
        # An unverified address is one the person may not control, and it is
        # what accounts are linked on below.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This Google account's email address is not verified.",
        )

    return claims


async def _fetch_profile(client: httpx.AsyncClient, access_token: str) -> dict:
    """Fetch display name and avatar from Google's userinfo endpoint.

    Never fatal: the ID token already established who this is, and losing a
    display name is not a reason to refuse a login.

    Args:
        client: HTTP client to use.
        access_token: Google's access token from the exchange.

    Returns:
        The userinfo body, or an empty mapping if it could not be fetched.
    """
    try:
        response = await client.get(
            GOOGLE_USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.info("Could not fetch Google profile details: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# User persistence
# ---------------------------------------------------------------------------


def _username_from_email(email: str) -> str:
    """Derive a candidate handle from an email address.

    Keeps the local part, strips anything that is not alphanumeric, dot,
    underscore or hyphen, and fits it to the column. A local part that
    survives as empty (an address of only symbols) falls back to a generic
    stem, so the result is never blank.
    """
    local = email.split("@", 1)[0]
    cleaned = "".join(c for c in local if c.isalnum() or c in "._-").strip("._-")
    return (cleaned[:50] or "user").lower()


async def _unique_username(session: AsyncSession, base: str) -> str:
    """Return *base*, or the first numbered variant not already taken.

    ``username`` is unique, and two people called ``alex`` at different
    domains is ordinary rather than exceptional. The suffix is trimmed out
    of the base, not appended past the column width.
    """
    existing = set(
        (
            await session.scalars(
                select(User.username).where(User.username.like(f"{base}%"))
            )
        ).all()
    )
    if base not in existing:
        return base
    for suffix in range(2, 1000):
        tail = str(suffix)
        candidate = f"{base[: 50 - len(tail)]}{tail}"
        if candidate not in existing:
            return candidate
    # 998 collisions on one stem is not a real scenario; a random tail ends
    # the loop rather than letting it fail.
    return f"{base[:42]}{secrets.token_hex(4)}"


async def _upsert_user(session: AsyncSession, identity: GoogleIdentity) -> User:
    """Create or update the account behind *identity*.

    Resolution order is deliberate:

    1. ``google_id`` -- the account has signed in before.
    2. Verified email, on an account not already federated -- one exists
       that was created some other way (the placeholder owner from Issue
       26, or a future local signup), so this login adopts it rather than
       creating a duplicate that cannot see its own repositories. Only
       reachable because the caller has already rejected unverified
       addresses.
    3. Neither -- a new account.

    Args:
        session: Request-scoped database session.
        identity: The verified Google account.

    Returns:
        The persisted user.

    Raises:
        HTTPException: 409 when the email belongs to an account already
            federated to a *different* Google identity.
    """
    user = await session.scalar(select(User).where(User.google_id == identity.sub))

    if user is None:
        claimant = await session.scalar(
            select(User).where(User.email == identity.email)
        )
        if claimant is not None and claimant.google_id is not None:
            # A different Google account already owns this row. Reachable
            # when an address is released and later re-registered: adopting
            # it here would hand the first person's repositories to the
            # second. Refusing needs a human to resolve it.
            logger.error(
                "Google sub %s claims an email already federated to another account",
                identity.sub,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This email address is already linked to a different "
                    "Google account. Contact support to resolve it."
                ),
            )
        user = claimant

    now = datetime.now(UTC)
    if user is None:
        user = User(
            username=await _unique_username(
                session, _username_from_email(identity.email)
            ),
            email=identity.email,
            hashed_password=None,
        )
        session.add(user)

    # Applied on creation and on every subsequent login, so a changed name,
    # avatar or primary address on the Google side propagates here.
    user.google_id = identity.sub
    user.email = identity.email
    user.full_name = identity.full_name or user.full_name
    user.avatar_url = identity.avatar_url or user.avatar_url
    user.last_login_at = now

    try:
        await session.commit()
    except IntegrityError:
        # Two first logins for the same account can race between the SELECT
        # above and this commit. The unique index is what actually decides;
        # the loser rolls back and reads the row the winner wrote.
        await session.rollback()
        existing = await session.scalar(
            select(User).where(User.google_id == identity.sub)
        )
        if existing is None:
            raise
        logger.info(
            "Concurrent first login for %s resolved to an existing row", user.email
        )
        return existing

    await session.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/google",
    summary="Start Google sign-in",
    response_class=RedirectResponse,
    status_code=status.HTTP_307_TEMPORARY_REDIRECT,
    response_description="Redirect to Google's consent screen.",
    responses={503: {"description": "Google sign-in is not configured."}},
)
async def google_login() -> RedirectResponse:
    """Redirect the browser to Google's consent screen.

    Mints the CSRF state, PKCE verifier and nonce for this attempt, seals
    them into a short-lived signed cookie, and sends the matching public
    halves to Google.
    """
    _require_oauth_configured()

    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    nonce = secrets.token_urlsafe(32)

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": " ".join(_SCOPES),
        "state": state,
        "nonce": nonce,
        "code_challenge": _pkce_challenge(verifier),
        "code_challenge_method": "S256",
        # Forces the account chooser rather than silently reusing whichever
        # Google account the browser last used -- the common complaint of
        # being logged in as the wrong identity with no way to switch.
        "prompt": "select_account",
    }
    response = RedirectResponse(
        url=f"{GOOGLE_AUTH_ENDPOINT}?{urlencode(params)}",
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
    )
    response.set_cookie(
        _TX_COOKIE,
        _seal_transaction(state, verifier, nonce),
        max_age=_TX_TTL_SECONDS,
        httponly=True,
        # Lax, not Strict: the callback arrives as a top-level navigation
        # from accounts.google.com, and Strict would withhold the cookie on
        # exactly that request, breaking every login.
        samesite="lax",
        # Off outside production only because http://localhost cannot store
        # a Secure cookie, which would make the flow untestable locally.
        secure=settings.is_production,
        path="/auth",
    )
    return response


@router.get(
    "/google/callback",
    response_model=TokenResponse,
    summary="Complete Google sign-in",
    response_description="Access and refresh tokens for the signed-in user.",
    responses={
        400: {"description": "Consent was declined, or the request did not verify."},
        401: {"description": "Google's response could not be verified."},
        403: {"description": "The Google account's email is not verified."},
        409: {"description": "The email is linked to a different Google account."},
        502: {"description": "Google could not be reached."},
        503: {"description": "Google sign-in is not configured."},
    },
)
async def google_callback(
    response: Response,
    session: Annotated[AsyncSession, Depends(get_db)],
    client: HttpClient,
    code: Annotated[str | None, Query(description="Authorization code.")] = None,
    state: Annotated[str | None, Query(description="CSRF state echo.")] = None,
    error: Annotated[str | None, Query(description="Google error code.")] = None,
    transaction: Annotated[str | None, Cookie(alias=_TX_COOKIE)] = None,
) -> TokenResponse:
    """Complete the login and issue this API's own tokens.

    Verifies the transaction, exchanges the code, verifies Google's ID
    token, creates or updates the account, and returns an access and a
    refresh token.

    Raises:
        HTTPException: 400 if consent was declined or the transaction does
            not verify, 401 if Google's ID token does not verify, 403 if the
            account's email is unverified, 502 if Google is unreachable,
            503 if the server has no Google credentials.
    """
    # The transaction is single-use whatever happens next. On the success
    # path the injected response carries the deletion; on a failure path
    # FastAPI discards that response and renders the exception instead, so
    # the header is attached to the exception as well. Leaving a live
    # verifier in the browser after a failed attempt would keep it
    # replayable for the rest of its ten minutes.
    response.delete_cookie(_TX_COOKIE, path="/auth")
    try:
        return await _complete_login(
            session=session,
            client=client,
            code=code,
            state=state,
            error=error,
            transaction=transaction,
        )
    except HTTPException as exc:
        exc.headers = {
            **(exc.headers or {}),
            "set-cookie": _clear_tx_cookie_header(),
        }
        raise


def _clear_tx_cookie_header() -> str:
    """Render the ``Set-Cookie`` directive that expires the transaction.

    Built through Starlette's own cookie writer rather than by hand so the
    attributes always match the ones :func:`google_login` set -- a deletion
    whose ``Path`` disagrees is silently ignored by the browser.
    """
    eraser = Response()
    eraser.delete_cookie(_TX_COOKIE, path="/auth")
    return eraser.headers["set-cookie"]


async def _complete_login(
    *,
    session: AsyncSession,
    client: httpx.AsyncClient,
    code: str | None,
    state: str | None,
    error: str | None,
    transaction: str | None,
) -> TokenResponse:
    """Verify the callback and issue tokens. See :func:`google_callback`."""
    _require_oauth_configured()

    if error:
        # The documented value for "the user pressed Cancel". Reported as
        # 400, not 500: nothing malfunctioned.
        logger.info("Google sign-in was not completed: %s", error)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Google sign-in was not completed ({error}).",
        )

    if not code or not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization code or state.",
        )

    if transaction is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Login session expired or invalid. Start again at /auth/google.",
        )

    sealed = _open_transaction(transaction)
    # compare_digest rather than ==: the comparison is against a value an
    # attacker controls and can retry, which is the shape a timing oracle
    # needs.
    if not secrets.compare_digest(str(sealed["state"]), state):
        logger.warning("OAuth state mismatch; rejecting callback")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Login session expired or invalid. Start again at /auth/google.",
        )

    tokens = await _exchange_code(client, code, str(sealed["verifier"]))

    id_token = tokens.get("id_token")
    if not id_token:
        logger.warning("Google token response carried no id_token")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Google did not return an identity token.",
        )

    # Verification fetches Google's key set over the network, so it must not
    # run on the event loop.
    claims = await run_in_threadpool(_verify_id_token, id_token, str(sealed["nonce"]))

    profile = {}
    if tokens.get("access_token"):
        profile = await _fetch_profile(client, tokens["access_token"])

    identity = GoogleIdentity(
        sub=claims["sub"],
        email=claims["email"],
        full_name=claims.get("name") or profile.get("name"),
        avatar_url=claims.get("picture") or profile.get("picture"),
    )
    user = await _upsert_user(session, identity)

    logger.info("Google sign-in succeeded for user %s", user.id)
    return TokenResponse(
        access_token=create_access_token(user.id, user.email),
        refresh_token=create_refresh_token(user.id, user.email),
        expires_in=settings.jwt_access_token_expire_minutes * 60,
        user=UserProfile.model_validate(user),
    )
