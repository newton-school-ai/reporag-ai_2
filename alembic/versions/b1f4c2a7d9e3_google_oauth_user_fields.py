"""google oauth user fields

Revision ID: b1f4c2a7d9e3
Revises: da46327d8019
Create Date: 2026-09-19 22:54:00.000000

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
    # Use batch_alter_table for SQLite compatibility
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
    # We must backfill null passwords before restoring NOT NULL
    # '!' is used because it's not a valid bcrypt/argon hash
    user_table = sa.table("users", sa.column("hashed_password", sa.String))
    op.execute(
        user_table.update()
        .where(user_table.c.hashed_password.is_(None))
        .values(hashed_password="!")
    )

    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_users_google_id"))
        batch_op.alter_column(
            "hashed_password", existing_type=sa.String(length=255), nullable=False
        )
        batch_op.drop_column("last_login_at")
        batch_op.drop_column("avatar_url")
        batch_op.drop_column("full_name")
        batch_op.drop_column("google_id")
