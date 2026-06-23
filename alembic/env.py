"""Alembic migration environment (async).

URL and metadata are sourced from the RepoRAG application code so that
migrations and the running app always agree on the database. The engine is
async (aiosqlite / asyncpg), so ``alembic upgrade head`` works against both
SQLite and Postgres without a sync driver.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy.engine import Connection

from alembic import context

# RepoRAG application imports (repo root is on sys.path via prepend_sys_path).
from src.reporag.config import settings
from src.reporag.db.models import Base
from src.reporag.db.session import create_engine, make_async_url

# Alembic Config object, providing access to values in alembic.ini.
config = context.config

# Resolve the database URL at runtime from application settings rather than
# hard-coding it in alembic.ini.
config.set_main_option("sqlalchemy.url", make_async_url(settings.database_url))

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for 'autogenerate' support.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL, no DBAPI needed)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # Batch mode lets SQLite emulate ALTER TABLE in future migrations.
        render_as_batch=connection.dialect.name == "sqlite",
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations against a live connection."""
    connectable = create_engine()

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
