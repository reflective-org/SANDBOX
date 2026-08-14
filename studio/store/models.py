# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Relational models for run metadata (ADR-004).

**The database stores pointers, never arrays.** Raw output is object storage keyed by run id -- a
directory on disk today, MinIO or S3 later without touching callers -- and what lives here is the
path, size, checksum and content type. A 3.6 MB npz per run times 810 runs is not a database's job.

Two rules from ADR-004 are structural rather than conventional:

* **Configs are immutable once submitted.** ``run_config`` is insert-only, keyed by ``config_hash``:
  the same config submitted twice is the same row, and an edited config is a *different* row and a
  *different* run. There is deliberately no update path -- see ``repository.py``, which offers
  get-or-create and nothing else.
* **Results are never overwritten.** An artefact row is written once per run.

Portability is a requirement, not an aspiration (ADR-007): moving to Postgres must be a
connection-string change. So:

* **No ``AUTOINCREMENT``.** Primary keys are explicit strings -- content hashes where a natural one
  exists, UUID hex otherwise. This also makes a run's identity meaningful rather than positional,
  and lets a row be written before the database has ever seen it.
* **No reliance on SQLite's type affinity.** Every column has a real type, and timestamps go
  through :class:`UtcDateTime` -- because ``DateTime(timezone=True)`` alone returns *naive*
  datetimes on SQLite and aware ones on Postgres, which is a comparison bug that only appears in
  production.
* **JSON columns via SQLAlchemy's portable ``JSON``**, which is native on both backends.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, TypeDecorator


class UtcDateTime(TypeDecorator[datetime]):
    """A timestamp that is timezone-aware on **every** backend.

    ``DateTime(timezone=True)`` is not enough. Postgres stores the offset and returns an aware
    datetime; **SQLite stores a string and hands back a naive one**, so the same code compares
    correctly on one backend and silently wrongly on the other -- the exact cross-backend divergence
    ADR-007 says to avoid, and the reason this class exists rather than a convention that everyone
    remembers to call ``.replace(tzinfo=UTC)``.

    Naive input **raises**: a caller who does not know their own timezone cannot be given one by
    guessing (ADR-005).
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                f"naive datetime {value!r} cannot be stored: its timezone is unknown, and assuming "
                f"one would be wrong on every machine but the one that wrote it. Use "
                f"datetime.now(UTC)."
            )
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    """Declarative base. Alembic's autogenerate compares against this metadata."""


#: Identifier column width: a SHA-256 hex digest is 64, a UUID hex 32. Fixed rather than unbounded
#: so Postgres gets a sensible column and the intent is visible.
_ID = String(64)


class RunSetRow(Base):
    """A sweep. A single run is the N = 1 case of one (see ``studio.schema.runset``)."""

    __tablename__ = "run_set"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    label: Mapped[str] = mapped_column(String(255), default="")
    #: The axes, as submitted. Stored whole because a RunSet is small and re-expanding it must give
    #: exactly the runs that were created, not today's interpretation of the axes.
    axes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)
    #: Nullable owner, present from the first migration so adopting tenancy later is a backfill
    #: rather than a migration of every query (BLOCKING-2 / ASSUMPTION-3).
    owner: Mapped[str | None] = mapped_column(String(255), nullable=True)

    runs: Mapped[list[RunRow]] = relationship(
        back_populates="run_set", cascade="all, delete-orphan"
    )


class RunConfigRow(Base):
    """A resolved configuration, keyed by its own hash. **Insert-only.**

    The primary key IS the identity (ADR-006): the same configuration submitted from the CLI and
    from the API is one row, which is what makes the hash usable as a cache key. Two runs of the
    same config share this row and differ only in their run id.
    """

    __tablename__ = "run_config"

    config_hash: Mapped[str] = mapped_column(_ID, primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32))
    #: The resolved, post-derivation parameter set -- what the model actually received.
    resolved_config: Mapped[dict[str, Any]] = mapped_column(JSON)
    #: Derived fields the user overrode, with the value in force.
    overrides: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)


class RunRow(Base):
    """One simulation: its config, its lineage, and where its outputs live."""

    __tablename__ = "run"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    run_set_id: Mapped[str] = mapped_column(_ID, ForeignKey("run_set.id"), index=True)
    config_hash: Mapped[str] = mapped_column(_ID, ForeignKey("run_config.config_hash"), index=True)
    #: The ensemble-style case label (``30N_20km__sabr220__…``). A label, never an identity -- it
    #: captures the axes rather than the resolved configuration, so two sweeps differing only in a
    #: "fixed" value would collide on it (ADR-006).
    label: Mapped[str] = mapped_column(String(255), default="")
    #: Lineage for an edited config: this run supersedes that one (ADR-004). Never a mutation.
    derived_from_run_id: Mapped[str | None] = mapped_column(
        _ID, ForeignKey("run.id"), nullable=True
    )
    #: The provenance record (ADR-006) as written at submit time, stored verbatim.
    provenance: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    #: False when any checkout was dirty: the SHAs do not describe the code that ran.
    reproducible: Mapped[bool | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)

    run_set: Mapped[RunSetRow] = relationship(back_populates="runs")
    jobs: Mapped[list[JobRow]] = relationship(back_populates="run", cascade="all, delete-orphan")
    artifacts: Mapped[list[ResultArtifactRow]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_run_config_hash_created", "config_hash", "created_at"),)


class JobRow(Base):
    """Execution of a run. State lives here, not in memory, so a handle survives an API restart."""

    __tablename__ = "job"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    run_id: Mapped[str] = mapped_column(_ID, ForeignKey("run.id"), index=True)
    #: Current lifecycle state (``studio.runner.JobState``). Stored as its string value so the
    #: database is readable without the enum, and a new state is a data change not a schema one.
    state: Mapped[str] = mapped_column(String(32), index=True)
    backend: Mapped[str] = mapped_column(String(64), default="local-subprocess")
    work_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime)

    run: Mapped[RunRow] = relationship(back_populates="jobs")
    transitions: Mapped[list[JobTransitionRow]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobTransitionRow.at"
    )


class JobTransitionRow(Base):
    """One timestamped state change. The audit trail, kept rather than collapsed to a status.

    "It failed" is not debuggable; "QUEUED 14:02:11, RUNNING 14:02:11, FAILED 14:06:48 exit 1" is.
    """

    __tablename__ = "job_transition"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    job_id: Mapped[str] = mapped_column(_ID, ForeignKey("job.id"), index=True)
    state: Mapped[str] = mapped_column(String(32))
    at: Mapped[datetime] = mapped_column(UtcDateTime)
    detail: Mapped[str] = mapped_column(Text, default="")

    job: Mapped[JobRow] = relationship(back_populates="transitions")


class ResultArtifactRow(Base):
    """A pointer to one output file. **The bytes are not here.**

    Path, size, checksum and content type -- enough to find it, verify it, and notice when it has
    gone missing. ``sha256`` is what makes "the file is still the one that was written" checkable
    rather than assumed.
    """

    __tablename__ = "result_artifact"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    run_id: Mapped[str] = mapped_column(_ID, ForeignKey("run.id"), index=True)
    #: ``state``, ``summary``, ``provenance``, ``stdout``, ``stderr``, ``input``.
    kind: Mapped[str] = mapped_column(String(32))
    #: Relative to the artifact store's root, never absolute: the root moves between deployments.
    path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    content_type: Mapped[str] = mapped_column(String(64), default="application/octet-stream")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)

    run: Mapped[RunRow] = relationship(back_populates="artifacts")

    __table_args__ = (UniqueConstraint("run_id", "kind", name="uq_artifact_run_kind"),)


class DatasetVersionRow(Base):
    """An input dataset consulted by a run, with its checksum (ADR-006).

    Empty in Phase 0 -- the schema carries no dataset inputs yet -- but present from the first
    migration, because "which ERA5 product was this run built on?" is a question Phase 1 must be
    able to answer about runs made before it existed.
    """

    __tablename__ = "dataset_version"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    identifier: Mapped[str] = mapped_column(String(255), index=True)
    version: Mapped[str] = mapped_column(String(64))
    sha256: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)

    __table_args__ = (
        UniqueConstraint("identifier", "version", name="uq_dataset_identifier_version"),
    )


class RunDatasetRow(Base):
    """Which datasets a run consulted. Many-to-many, so a dataset is recorded once."""

    __tablename__ = "run_dataset"

    run_id: Mapped[str] = mapped_column(_ID, ForeignKey("run.id"), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(_ID, ForeignKey("dataset_version.id"), primary_key=True)


class RunSummaryRow(Base):
    """The versioned reduction (ADR-004), stored so comparison views never open the raw npz.

    Kept as JSON rather than shredded into columns: it is read whole, its shape is versioned, and
    normalising it would make ``schema_version`` a migration problem instead of a field.
    """

    __tablename__ = "run_summary"

    run_id: Mapped[str] = mapped_column(_ID, ForeignKey("run.id"), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32))
    termination: Mapped[str] = mapped_column(String(32), index=True)
    flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    #: Headline scalars promoted out of the JSON for querying and sorting without deserialising:
    #: "show me every run where peak number exceeded X" must not require reading 810 blobs.
    final_so2_pptv: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_h2so4_pptv: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_number_cm3: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_surface_area: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)


__all__ = [
    "Base",
    "DatasetVersionRow",
    "JobRow",
    "JobTransitionRow",
    "ResultArtifactRow",
    "RunConfigRow",
    "RunDatasetRow",
    "RunRow",
    "RunSetRow",
    "RunSummaryRow",
    "UtcDateTime",
]
