# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The operations the CLI, the API and the runner need -- and deliberately no others.

**There is no update path for a config.** ADR-004 says a submitted config is immutable and an edit
produces a new config and a new run; a repository that offered ``update_config`` would make that a
convention people remember rather than a property of the system. :func:`ensure_config` is
get-or-create, keyed by the config's own hash, and is the only way a config enters the database.

Everything here takes a ``Session`` rather than opening its own. Transaction boundaries belong to
the caller, because SQLite has a single writer (ADR-007) and only the caller knows what else belongs
in the same short transaction.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from studio.resolve import ResolvedConfig
from studio.store.artifacts import ArtifactStore
from studio.store.models import (
    JobRow,
    JobTransitionRow,
    ResultArtifactRow,
    RunConfigRow,
    RunRow,
    RunSetRow,
    RunSummaryRow,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


def ensure_config(session: Session, config: ResolvedConfig) -> RunConfigRow:
    """Get-or-create the config row. **The only way a config enters the database.**

    Idempotent by construction: the primary key is the config's own hash, so submitting the same
    configuration from the CLI and from the API converges on one row rather than creating a second
    identity for the same computation.

    Raises:
        InconsistentConfigError: If the config has stale overrides. Persisting one would give an
            inconsistent set of numbers a permanent identity.
    """
    config.require_consistent()
    config_hash = config.config.config_hash()
    existing = session.get(RunConfigRow, config_hash)
    if existing is not None:
        return existing
    row = RunConfigRow(
        config_hash=config_hash,
        schema_version=config.config.schema_version,
        resolved_config=config.config.model_dump(mode="json"),
        overrides={path: record.value for path, record in sorted(config.overrides.items())},
        created_at=_now(),
    )
    session.add(row)
    session.flush()
    return row


def create_run_set(
    session: Session,
    *,
    label: str = "",
    axes: dict[str, Any] | None = None,
    owner: str | None = None,
) -> RunSetRow:
    """Create a sweep. A single run is the N = 1 case of one, with no axes."""
    row = RunSetRow(id=_new_id(), label=label, axes=axes or {}, owner=owner, created_at=_now())
    session.add(row)
    session.flush()
    return row


def create_run(
    session: Session,
    *,
    run_set: RunSetRow,
    config: ResolvedConfig,
    label: str = "",
    provenance: dict[str, Any] | None = None,
    derived_from_run_id: str | None = None,
) -> RunRow:
    """Create a run, ensuring its config exists first.

    ``provenance`` is the record written at submit time (ADR-006), stored verbatim. ``reproducible``
    is derived from it here rather than recomputed later, because it is a property of the moment the
    run started, not of the checkout as it stands now.
    """
    config_row = ensure_config(session, config)
    reproducible: bool | None = None
    if provenance is not None:
        dirty = bool(provenance.get("sandbox", {}).get("dirty")) or any(
            bool(sub.get("dirty")) for sub in provenance.get("submodules", {}).values()
        )
        reproducible = not dirty
    row = RunRow(
        id=_new_id(),
        run_set_id=run_set.id,
        config_hash=config_row.config_hash,
        label=label,
        derived_from_run_id=derived_from_run_id,
        provenance=provenance,
        reproducible=reproducible,
        created_at=_now(),
    )
    session.add(row)
    session.flush()
    return row


def record_job(
    session: Session,
    *,
    run: RunRow,
    state: str,
    backend: str = "local-subprocess",
    work_dir: Path | str | None = None,
    detail: str = "",
) -> JobRow:
    """Create a job in its first state, with the matching transition."""
    now = _now()
    job = JobRow(
        id=_new_id(),
        run_id=run.id,
        state=state,
        backend=backend,
        work_dir=str(work_dir) if work_dir is not None else None,
        detail=detail,
        created_at=now,
        updated_at=now,
    )
    session.add(job)
    session.add(JobTransitionRow(id=_new_id(), job_id=job.id, state=state, at=now, detail=detail))
    session.flush()
    return job


def record_transition(
    session: Session, *, job: JobRow, state: str, detail: str = "", exit_code: int | None = None
) -> JobRow:
    """Advance a job and append the transition.

    Appends rather than replaces: the current state is a convenience column, and the transition list
    is the record. A job whose state went backwards is visible here instead of being overwritten.
    """
    now = _now()
    job.state = state
    job.detail = detail or job.detail
    job.updated_at = now
    if exit_code is not None:
        job.exit_code = exit_code
    session.add(JobTransitionRow(id=_new_id(), job_id=job.id, state=state, at=now, detail=detail))
    session.flush()
    return job


def record_artifact(
    session: Session, *, run: RunRow, kind: str, source: Path, store: ArtifactStore
) -> ResultArtifactRow:
    """Store a file and record the pointer. The bytes never enter the database.

    Raises:
        FileNotFoundError: If the file is missing -- surfaced rather than recorded as an absent row.
    """
    stored = store.put(run.id, kind, Path(source))
    row = ResultArtifactRow(
        id=_new_id(),
        run_id=run.id,
        kind=kind,
        path=stored.key,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        content_type=stored.content_type,
        created_at=_now(),
    )
    session.add(row)
    session.flush()
    return row


def record_summary(session: Session, *, run: RunRow, summary: dict[str, Any]) -> RunSummaryRow:
    """Store the RunSummary, promoting four headline scalars into columns.

    Promoted so "every run whose peak number exceeded X" is a query rather than 810
    deserialisations. The scalars are read from the summary rather than recomputed, so the column
    and the JSON can never disagree.
    """
    series = summary.get("series", {})

    def last(name: str) -> float | None:
        values = series.get(name, {}).get("values") or []
        return float(values[-1]) if values else None

    def peak(name: str) -> float | None:
        values = series.get(name, {}).get("values") or []
        return float(max(values)) if values else None

    row = RunSummaryRow(
        run_id=run.id,
        schema_version=str(summary.get("schema_version", "")),
        termination=str(summary.get("termination", "unknown")),
        flags=list(summary.get("flags", [])),
        final_so2_pptv=last("SO2"),
        peak_h2so4_pptv=peak("H2SO4"),
        peak_number_cm3=peak("total_n"),
        final_surface_area=last("SA"),
        summary=summary,
        created_at=_now(),
    )
    session.add(row)
    session.flush()
    return row


def runs_for_config(session: Session, config_hash: str) -> list[RunRow]:
    """Every run of one configuration, newest first. The cache lookup (ADR-006).

    Identical hash is necessary but **not sufficient** to reuse a result: the caller must also
    compare the model version in each run's provenance, which is why this returns the runs rather
    than a verdict.
    """
    statement = (
        select(RunRow).where(RunRow.config_hash == config_hash).order_by(RunRow.created_at.desc())
    )
    return list(session.scalars(statement))


def artifact_for(session: Session, run_id: str, kind: str) -> ResultArtifactRow | None:
    statement = select(ResultArtifactRow).where(
        ResultArtifactRow.run_id == run_id, ResultArtifactRow.kind == kind
    )
    return session.scalars(statement).one_or_none()


__all__ = [
    "artifact_for",
    "create_run",
    "create_run_set",
    "ensure_config",
    "record_artifact",
    "record_job",
    "record_summary",
    "record_transition",
    "runs_for_config",
]
