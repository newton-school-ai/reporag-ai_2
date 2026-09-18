"""Unit tests for the JWT layer (Issue 27).

Covers :mod:`reporag.api.middleware.auth`: minting access and refresh
tokens, and the verification that decides whether one is trustworthy.

The emphasis is on what verification must *reject*. A token layer that
accepts every token it issued is trivially correct; the cases that matter
are the ones an attacker constructs -- a forged signature, a stripped
algorithm, a swapped audience, and above all a refresh token presented as a
bearer credential, which is the mistake that silently converts a 30-minute
access lifetime into a 7-day one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from pydantic import SecretStr

from reporag.api.middleware.auth import (
    ALGORITHM,
    AUDIENCE,
    ISSUER,
    InvalidTokenError,
    TokenExpiredError,
    TokenType,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from reporag.config import settings as live_settings

_SECRET = "test-jwt-secret-value"


@pytest.fixture(autouse=True)
def _known_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the signing secret so tests can forge tokens deliberately."""
    monkeypatch.setattr(live_settings, "jwt_secret_key", SecretStr(_SECRET))


def _raw_claims(token: str) -> dict:
    """Read a token's payload without verifying it."""
    return jwt.decode(token, options={"verify_signature": False}, audience=AUDIENCE)


def _forge(**overrides: object) -> str:
    """Build a token with arbitrary claims, signed with the real secret.

    Used to construct the tokens an attacker would, which the public
    factories deliberately cannot produce.
    """
    now = datetime.now(UTC)
    payload: dict = {
        "sub": "1",
        "email": "user@example.com",
        "type": TokenType.ACCESS.value,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "jti": "forged",
    }
    payload.update(overrides)
    for key, value in list(payload.items()):
        if value is None:
            del payload[key]
    return jwt.encode(payload, _SECRET, algorithm=ALGORITHM)


class TestTokenIssuance:
    def test_access_token_round_trips(self) -> None:
        claims = decode_token(create_access_token(7, "a@b.com"), TokenType.ACCESS)
        assert claims.user_id == 7
        assert claims.email == "a@b.com"
        assert claims.token_type is TokenType.ACCESS

    def test_refresh_token_round_trips(self) -> None:
        claims = decode_token(create_refresh_token(7, "a@b.com"), TokenType.REFRESH)
        assert claims.user_id == 7
        assert claims.token_type is TokenType.REFRESH

    def test_subject_is_a_string(self) -> None:
        # RFC 7519 requires it, and PyJWT enforces it on decode: an int here
        # would mint tokens this module could not read back.
        assert _raw_claims(create_access_token(7, "a@b.com"))["sub"] == "7"

    def test_carries_issuer_and_audience(self) -> None:
        payload = _raw_claims(create_access_token(1, "a@b.com"))
        assert payload["iss"] == ISSUER
        assert payload["aud"] == AUDIENCE

    def test_each_token_has_a_unique_jti(self) -> None:
        first = decode_token(create_access_token(1, "a@b.com"), TokenType.ACCESS)
        second = decode_token(create_access_token(1, "a@b.com"), TokenType.ACCESS)
        # Issue 29 can build a revocation list on this; duplicate ids would
        # make revoking one token revoke another.
        assert first.jti != second.jti

    def test_access_lifetime_follows_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(live_settings, "jwt_access_token_expire_minutes", 15)
        claims = decode_token(create_access_token(1, "a@b.com"), TokenType.ACCESS)
        assert abs(
            (claims.expires_at - claims.issued_at) - timedelta(minutes=15)
        ) < timedelta(seconds=2)

    def test_refresh_lifetime_follows_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(live_settings, "jwt_refresh_token_expire_days", 3)
        claims = decode_token(create_refresh_token(1, "a@b.com"), TokenType.REFRESH)
        assert abs(
            (claims.expires_at - claims.issued_at) - timedelta(days=3)
        ) < timedelta(seconds=2)

    def test_refresh_outlives_access(self) -> None:
        access = decode_token(create_access_token(1, "a@b.com"), TokenType.ACCESS)
        refresh = decode_token(create_refresh_token(1, "a@b.com"), TokenType.REFRESH)
        assert refresh.expires_at > access.expires_at


class TestTokenTypeConfusion:
    """The separation that makes two token lifetimes mean anything."""

    def test_refresh_token_is_not_accepted_as_access(self) -> None:
        token = create_refresh_token(1, "a@b.com")
        # Without this check a refresh token -- valid for days by design --
        # would work as a bearer credential.
        with pytest.raises(InvalidTokenError):
            decode_token(token, TokenType.ACCESS)

    def test_access_token_is_not_accepted_as_refresh(self) -> None:
        with pytest.raises(InvalidTokenError):
            decode_token(create_access_token(1, "a@b.com"), TokenType.REFRESH)

    def test_missing_type_claim_is_rejected(self) -> None:
        with pytest.raises(InvalidTokenError):
            decode_token(_forge(type=None), TokenType.ACCESS)

    def test_unknown_type_claim_is_rejected(self) -> None:
        with pytest.raises(InvalidTokenError):
            decode_token(_forge(type="admin"), TokenType.ACCESS)


class TestTokenRejection:
    def test_expired_token_raises_expired_not_invalid(self) -> None:
        stale = _forge(
            iat=datetime.now(UTC) - timedelta(hours=2),
            exp=datetime.now(UTC) - timedelta(hours=1),
        )
        # A distinct type because the remedy differs: refresh, rather than
        # re-authenticate from scratch.
        with pytest.raises(TokenExpiredError):
            decode_token(stale, TokenType.ACCESS)

    def test_token_signed_with_another_secret_is_rejected(self) -> None:
        now = datetime.now(UTC)
        forged = jwt.encode(
            {
                "sub": "1",
                "type": "access",
                "iss": ISSUER,
                "aud": AUDIENCE,
                "iat": now,
                "exp": now + timedelta(minutes=5),
                "jti": "x",
            },
            "not-the-real-secret",
            algorithm=ALGORITHM,
        )
        with pytest.raises(InvalidTokenError):
            decode_token(forged, TokenType.ACCESS)

    def test_unsigned_token_is_rejected(self) -> None:
        now = datetime.now(UTC)
        none_alg = jwt.encode(
            {
                "sub": "1",
                "type": "access",
                "iss": ISSUER,
                "aud": AUDIENCE,
                "iat": now,
                "exp": now + timedelta(minutes=5),
                "jti": "x",
            },
            key="",
            algorithm="none",
        )
        # The classic JWT attack: strip the algorithm and the signature.
        # decode() pins algorithms, so it never gets considered.
        with pytest.raises(InvalidTokenError):
            decode_token(none_alg, TokenType.ACCESS)

    def test_wrong_issuer_is_rejected(self) -> None:
        with pytest.raises(InvalidTokenError):
            decode_token(_forge(iss="someone-else"), TokenType.ACCESS)

    def test_wrong_audience_is_rejected(self) -> None:
        # A token minted for a sibling service that shares the secret must
        # not authenticate here.
        with pytest.raises(InvalidTokenError):
            decode_token(_forge(aud="another-service"), TokenType.ACCESS)

    @pytest.mark.parametrize("claim", ["sub", "exp", "iat", "iss", "aud", "jti"])
    def test_every_required_claim_is_required(self, claim: str) -> None:
        with pytest.raises(InvalidTokenError):
            decode_token(_forge(**{claim: None}), TokenType.ACCESS)

    def test_non_numeric_subject_is_rejected(self) -> None:
        with pytest.raises(InvalidTokenError):
            decode_token(_forge(sub="not-an-id"), TokenType.ACCESS)

    @pytest.mark.parametrize("garbage", ["", "not.a.token", "a.b", "..."])
    def test_malformed_tokens_are_rejected(self, garbage: str) -> None:
        with pytest.raises(InvalidTokenError):
            decode_token(garbage, TokenType.ACCESS)

    def test_tampered_payload_is_rejected(self) -> None:
        header, payload, signature = create_access_token(1, "a@b.com").split(".")
        # Swap in another user's payload, keep the original signature.
        other = create_access_token(999, "attacker@example.com").split(".")[1]
        with pytest.raises(InvalidTokenError):
            decode_token(f"{header}.{other}.{signature}", TokenType.ACCESS)

    def test_a_secret_rotation_invalidates_old_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = create_access_token(1, "a@b.com")
        monkeypatch.setattr(live_settings, "jwt_secret_key", SecretStr("rotated"))
        # Rotating JWT_SECRET_KEY is the operator's break-glass for a leaked
        # secret; it has to actually log everyone out.
        with pytest.raises(InvalidTokenError):
            decode_token(token, TokenType.ACCESS)


class TestClockSkew:
    def test_a_token_a_few_seconds_past_expiry_is_still_accepted(self) -> None:
        barely = _forge(exp=datetime.now(UTC) - timedelta(seconds=3))
        # Hosts drift. Rejecting on three seconds would produce 401s that
        # no client could reproduce or act on.
        assert decode_token(barely, TokenType.ACCESS).user_id == 1

    def test_a_token_well_past_expiry_is_not(self) -> None:
        with pytest.raises(TokenExpiredError):
            decode_token(
                _forge(exp=datetime.now(UTC) - timedelta(minutes=5)), TokenType.ACCESS
            )
