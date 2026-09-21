"""Google OAuth user fields

Revision ID: a27c0ffee001
Revises: da46327d8019
Create Date: 2026-09-18 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a27c0ffee001"
down_revision: str | None = "da46327d8019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("google_id", sa.String(length=255), nullable=True))
        batch.add_column(sa.Column("full_name", sa.String(length=255), nullable=True))
        batch.add_column(
            sa.Column("picture_url", sa.String(length=2048), nullable=True)
        )
        batch.alter_column(
            "hashed_password", existing_type=sa.String(length=255), nullable=True
        )
        batch.create_index(op.f("ix_users_google_id"), ["google_id"], unique=True)


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_index(op.f("ix_users_google_id"))
        batch.alter_column(
            "hashed_password", existing_type=sa.String(length=255), nullable=False
        )
        batch.drop_column("picture_url")
        batch.drop_column("full_name")
        batch.drop_column("google_id")
