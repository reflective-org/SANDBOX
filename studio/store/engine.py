# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Engine and session factory.

SQLite in Phase 0; a Postgres URL substitutes without code changes, which is the whole point of
putting Alembic in from the first migration (ADR-007).

Two SQLite-specific pragmas are set **for SQLite only**, and neither is a workaround leaking into
the models:

* ``foreign_keys=ON`` -- SQLite ignores foreign keys unless asked, so without this the constraints
  in ``models.py`` would be documentation on that backend and enforced on Postgres. Silent
  divergence between environments is worse than either behaviour.
* ``journal_mode=WAL`` -- readers stop blocking the writer. ADR-007 flags SQLite's single-writer
  model as a real constraint, and the API reads job state far more often than it writes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

#: Read at import of a session factory, never at module import, so a test can set it first.
DATABASE_URL_ENV = "STUDIO_DATABASE_URL"

#: Where a local SQLite database lives when nothing says otherwise. Matches studio/.env.example.
DEFAULT_SQLITE_PATH = Path("var/studio/studio.db")


def database_url(url: str | None = None) -> str:
    """Resolve the database URL: explicit argument, then environment, then the local default."""
    if url:
        return url
    from_env = os.environ.get(DATABASE_URL_ENV)
    if from_env:
        return from_env
    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_SQLITE_PATH}"


def create_db_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """An engine with SQLite's footguns disarmed."""
    resolved = database_url(url)
    engine = create_engine(resolved, echo=echo, future=True)
    if resolved.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Sessions that do not expire objects on commit.

    ``expire_on_commit=False`` because callers read attributes off returned rows after the
    transaction closes; the alternative is a lazy reload per attribute, which on SQLite means
    re-acquiring the very lock ADR-007 warns about.
    """
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """A transaction that commits on success and rolls back on any exception.

    Short by design: SQLite has one writer, so a session held open across a model run would block
    every other job's state update.
    """
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = [
    "DATABASE_URL_ENV",
    "DEFAULT_SQLITE_PATH",
    "create_db_engine",
    "database_url",
    "session_factory",
    "session_scope",
]
