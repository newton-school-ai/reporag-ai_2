import jwt
import pytest

from reporag.api.middleware.auth import (
    InvalidTokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from reporag.db.models import User


@pytest.fixture
def test_user():
    user = User(id=1, email="test@example.com")
    return user


def test_returns_access_and_refresh_tokens(test_user):
    access_token = create_access_token(test_user)
    refresh_token = create_refresh_token(test_user)

    assert access_token is not None
    assert refresh_token is not None

    claims = decode_token(access_token, "access")
    assert claims.sub == "1"
    assert claims.email == "test@example.com"
    assert claims.type == "access"

    claims = decode_token(refresh_token, "refresh")
    assert claims.type == "refresh"


def test_type_confusion_rejected(test_user):
    access_token = create_access_token(test_user)
    with pytest.raises(InvalidTokenError):
        decode_token(access_token, "refresh")


def test_forged_token_rejected(test_user):
    payload = {"sub": "1", "email": "x", "type": "access"}
    forged = jwt.encode(payload, "wrong_secret", algorithm="HS256")
    with pytest.raises(InvalidTokenError):
        decode_token(forged, "access")
