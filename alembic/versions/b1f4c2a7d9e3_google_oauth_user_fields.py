"""Add Google OAuth identity fields to users

Adds the columns the OAuth login flow (Issue 27) needs and relaxes
``hashed_password`` to nullable, because an account created through Google
never has one.

``batch_alter_table`` is used for the nullability change: SQLite cannot
``ALTER COLUMN``, so Alembic rebuilds the table instead. On Postgres the
same call compiles to a plain ``ALTER TABLE``, so one code path serves both.

Revision ID: b1f4c2a7d9e3
Revises: da46327d8019
Create Date: 2026-09-18 10:12:03.114927

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b1f4c2a7d9e3"
down_revision: str | None = "da46327d8019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("google_id", sa.String(length=255), nullable=True)
        )
        batch_op.add_column(
            sa.Column("full_name", sa.String(length=255), nullable=True)
        )
        batch_op.add_column(
            sa.Column("avatar_url", sa.String(length=2048), nullable=True)
        )
        batch_op.add_column(
            sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.alter_column(
            "hashed_password", existing_type=sa.String(length=255), nullable=True
        )
        batch_op.create_index(
            batch_op.f("ix_users_google_id"), ["google_id"], unique=True
        )


def downgrade() -> None:
    # Rows created through OAuth have no password. Giving them one that
    # cannot match any hash keeps the NOT NULL constraint satisfiable
    # without inventing a credential that could authenticate.
    op.execute("UPDATE users SET hashed_password = '!' WHERE hashed_password IS NULL")

    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_users_google_id"))
        batch_op.alter_column(
            "hashed_password", existing_type=sa.String(length=255), nullable=False
        )
        batch_op.drop_column("last_login_at")
        batch_op.drop_column("avatar_url")
        batch_op.drop_column("full_name")
        batch_op.drop_column("google_id")
