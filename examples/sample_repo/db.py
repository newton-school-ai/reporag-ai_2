"""Database module for the sample app.

Simple in-memory store for testing purposes.
"""


class User:
    """User model."""

    def __init__(self, id, name, email, password_hash):
        self.id = id
        self.name = name
        self.email = email
        self.password_hash = password_hash


# In-memory store
_users = {}
_sessions = {}


def get_user_by_email(email):
    """Look up a user by email address."""
    return _users.get(email)


def save_session(user_id, token):
    """Persist a session token for a user."""
    _sessions[token] = user_id


def get_session(token):
    """Look up a session by token."""
    return _sessions.get(token)
