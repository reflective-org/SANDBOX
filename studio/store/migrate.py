# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Running migrations from code, so nothing depends on the ``alembic`` CLI being on PATH.

``Base.metadata.create_all`` is deliberately **not** used anywhere, including in tests. It would
produce a schema that no migration ever created, so the migrations would be exercised for the first
time on someone's real database -- which is the failure mode Alembic-from-the-first-migration exists
to prevent (ADR-007). Tests upgrade to head like everything else.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Engine

#: The committed configuration; the URL is supplied at call time, never from the file.
ALEMBIC_INI = Path(__file__).resolve().parent / "alembic.ini"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(url: str) -> Config:
    """An Alembic config pointed at this package's migrations and the given URL."""
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", url)
    return config


def upgrade_to_head(url: str) -> None:
    """Bring a database up to the latest revision. Idempotent."""
    command.upgrade(alembic_config(url), "head")


def current_revision(engine: Engine) -> str | None:
    """The revision a database is at, or ``None`` if it has never been migrated."""
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


__all__ = [
    "ALEMBIC_INI",
    "MIGRATIONS_DIR",
    "alembic_config",
    "current_revision",
    "upgrade_to_head",
]
