"""Sample Flask-like application for testing RepoRAG.

This tiny app has known call chains and import dependencies
so integration tests can verify correct graph construction
and retrieval.
"""

from auth import authenticate_user, create_token
from db import get_user_by_email, save_session


def handle_login(request):
    """Handle user login request.

    Authenticates credentials, creates a session token,
    and saves the session to the database.
    """
    email = request.get("email")
    password = request.get("password")

    user = authenticate_user(email, password)
    if user is None:
        return {"error": "Invalid credentials"}, 401

    token = create_token(user)
    save_session(user.id, token)
    return {"token": token}, 200


def handle_profile(request):
    """Return the authenticated user's profile."""
    user = get_user_by_email(request.get("email"))
    if user is None:
        return {"error": "User not found"}, 404
    return {"name": user.name, "email": user.email}, 200
