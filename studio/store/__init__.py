# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Persistence for run metadata (ADR-004, ADR-007).

Three stores, matched to three shapes. This package is the relational one: run sets, runs, configs,
jobs and pointers to output. **Arrays live on disk behind ``artifacts.ArtifactStore``**, never in
the database, and climatology is a separate Phase-1 concern.

* ``models.py`` -- the tables. No ``AUTOINCREMENT``, no reliance on SQLite type affinity,
  timezone-aware timestamps: moving to Postgres must be a connection-string change.
* ``engine.py`` -- engine and short-transaction session scope; SQLite gets ``foreign_keys=ON`` and
  WAL, because otherwise its constraints would be documentation while Postgres enforced them.
* ``artifacts.py`` -- the storage interface. A directory today; MinIO or S3 later without touching
  callers. Checksums are computed on write so "still the file that was written" is checkable.
* ``repository.py`` -- the operations, and no update path for a config: immutability is structural.
* ``migrations/`` -- Alembic from the first migration, so the second one is routine.

This package must not import ``coupled``.
"""

from __future__ import annotations

from studio.store.artifacts import ArtifactStore, LocalDirectoryStore, StoredArtifact, sha256_of
from studio.store.engine import (
    DATABASE_URL_ENV,
    create_db_engine,
    database_url,
    session_factory,
    session_scope,
)
from studio.store.migrate import current_revision, upgrade_to_head
from studio.store.models import (
    Base,
    DatasetVersionRow,
    JobRow,
    JobTransitionRow,
    ResultArtifactRow,
    RunConfigRow,
    RunRow,
    RunSetRow,
    RunSummaryRow,
)
from studio.store.repository import (
    artifact_for,
    create_run,
    create_run_set,
    ensure_config,
    record_artifact,
    record_job,
    record_summary,
    record_transition,
    runs_for_config,
)

__all__ = [
    "DATABASE_URL_ENV",
    "ArtifactStore",
    "Base",
    "DatasetVersionRow",
    "JobRow",
    "JobTransitionRow",
    "LocalDirectoryStore",
    "ResultArtifactRow",
    "RunConfigRow",
    "RunRow",
    "RunSetRow",
    "RunSummaryRow",
    "StoredArtifact",
    "artifact_for",
    "create_db_engine",
    "create_run",
    "create_run_set",
    "current_revision",
    "database_url",
    "ensure_config",
    "record_artifact",
    "record_job",
    "record_summary",
    "record_transition",
    "runs_for_config",
    "session_factory",
    "session_scope",
    "sha256_of",
    "upgrade_to_head",
]
