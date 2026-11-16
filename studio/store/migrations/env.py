# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Alembic environment.

The URL comes from ``studio.store.engine.database_url`` rather than ``alembic.ini`` so that one
committed configuration serves a developer machine, CI and a Postgres deployment -- the difference
is ``STUDIO_DATABASE_URL``, not an edited file.

``render_as_batch=True`` is set for SQLite only. SQLite cannot ALTER most things in place, so
Alembic rebuilds the table; without it, the first migration that drops a column would fail on the
Phase-0 backend and pass on Postgres. That divergence is exactly what ADR-007 says to avoid.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context

from studio.store.engine import create_db_engine
from studio.store.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL without a connection, for review or for a DBA to apply."""
    url = config.get_main_option("sqlalchemy.url") or None
    from studio.store.engine import database_url

    context.configure(
        url=database_url(url),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_db_engine(config.get_main_option("sqlalchemy.url") or None)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
