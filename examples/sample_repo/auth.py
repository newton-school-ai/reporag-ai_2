"""Authentication module for the sample app."""

import hashlib

from db import get_user_by_email


def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def authenticate_user(email, password):
    """Authenticate a user by email and password.

    Returns the user object if credentials are valid, None otherwise.
    """
    user = get_user_by_email(email)
    if user is None:
        return None
    if user.password_hash != hash_password(password):
        return None
    return user


def create_token(user):
    """Create a session token for an authenticated user."""
    payload = f"{user.id}:{user.email}"
    return hashlib.sha256(payload.encode()).hexdigest()
