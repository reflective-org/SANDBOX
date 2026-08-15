# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Submitting a run: the one flow the CLI and the API both use.

Extracted the moment there were two callers. A second copy of "resolve, record provenance, persist,
submit, record every transition, store the artefacts" would drift within a week, and the drift would
be invisible -- both paths would keep working, and only their database rows would disagree. ADR-002
requires that a sweep launched from a terminal and one launched from the web produce identical rows
and identical provenance; that is a property of there being *one function*, not of two being written
carefully.

The two callers differ in what they do *around* this -- the CLI blocks and prints, the API returns a
handle and streams progress -- and neither difference reaches in here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sqlalchemy.orm import sessionmaker

from studio.modelio.provenance import record_for
from studio.resolve import ResolvedConfig
from studio.runner import LocalSubprocessRunner
from studio.store import (
    LocalDirectoryStore,
    create_db_engine,
    create_run,
    database_url,
    record_artifact,
    record_job,
    record_summary,
    record_transition,
    session_factory,
    session_scope,
    upgrade_to_head,
)
from studio.store.models import JobRow, RunRow, RunSetRow

#: How often a running job's state is copied into the database. Short against a 4-minute run, long
#: against a single SELECT.
_FOLLOW_INTERVAL_S = 0.2

#: Artefacts recorded for every run, by kind -> filename in the job's work directory. A fixed set
#: rather than "whatever the directory contains", so a missing one is a visible gap rather than a
#: silently shorter list.
ARTIFACTS: dict[str, str] = {
    "input": "input.json",
    "provenance": "provenance.json",
    "state": "state.npz",
    "summary": "summary.json",
    "stdout": "stdout.log",
    "stderr": "stderr.log",
}


def prepare(database: str | None, out: Path) -> tuple[sessionmaker[Any], LocalDirectoryStore]:
    """Migrate the database and open the artefact store.

    Idempotent, so both entry points can call it unconditionally at startup: a fresh clone must not
    need a manual ``alembic upgrade`` before the first run.
    """
    url = database_url(database)
    upgrade_to_head(url)
    return session_factory(create_db_engine(url)), LocalDirectoryStore(Path(out) / "artifacts")


def _require(row: Any, what: str, key: str) -> Any:
    """Return ``row``, or raise saying what went missing.

    ``Session.get`` returns ``None`` for a row that is not there, and passing that on would fail
    several frames later as an ``AttributeError`` about ``None``. Here it means the database changed
    underneath a run that is mid-flight -- rare, and worth naming precisely when it happens.
    """
    if row is None:
        raise RuntimeError(
            f"{what} {key!r} vanished from the database while its run was in progress; "
            f"the run may have been deleted concurrently"
        )
    return row


def create_pending_run(
    factory: sessionmaker[Any],
    *,
    run_set: RunSetRow,
    config: ResolvedConfig,
    label: str = "",
) -> str:
    """Record the run and its provenance **before** anything is submitted.

    Provenance first, deliberately (ADR-006): if the model cannot be pinned, the run must not start,
    and a run that dies in its first second still says exactly what produced it.
    """
    provenance = record_for(config)
    with session_scope(factory) as session:
        run = create_run(
            session,
            run_set=run_set,
            config=config,
            label=label,
            provenance=provenance.model_dump(mode="json"),
        )
        return str(run.id)


def submit(
    factory: sessionmaker[Any],
    runner: LocalSubprocessRunner,
    *,
    run_id: str,
    config: ResolvedConfig,
    label: str = "",
) -> tuple[str, str]:
    """Submit a recorded run. Returns ``(job_id, runner_job_id)``.

    Returns both because they are different things: the database's job id is durable and survives a
    restart, while the runner's is a handle into this process's pool.
    """
    record = runner.submit(config, label=label)
    with session_scope(factory) as session:
        run_row = _require(session.get(RunRow, run_id), "run", run_id)
        job = record_job(
            session,
            run=run_row,
            state=record.state.value,
            work_dir=record.work_dir,
            detail=record.detail,
        )
        return str(job.id), record.job_id


def finalise(
    factory: sessionmaker[Any],
    store: LocalDirectoryStore,
    runner: LocalSubprocessRunner,
    *,
    run_id: str,
    job_id: str,
    runner_job_id: str,
) -> str:
    """Follow a job to completion, persisting each transition **as it happens**, then store its
    artefacts. Returns the final state.

    Polling the runner rather than only waiting, because the database is what the API streams from
    (a stream reading the worker pool would be blind to runs submitted by the CLI or by a previous
    process). Writing only at the end would leave a job showing ``queued`` for its whole four
    minutes and then jumping to ``succeeded`` -- technically a trail, useless as progress.

    ``_FOLLOW_INTERVAL_S`` is short relative to a run and long relative to a SELECT; SQLite has one
    writer, and each write here is a single short transaction.
    """
    seen = 1  # the queued transition is already persisted by submit()
    while True:
        record = runner.poll(runner_job_id)
        new_transitions = record.transitions[seen:]
        if new_transitions:
            with session_scope(factory) as session:
                job_row = _require(session.get(JobRow, job_id), "job", job_id)
                for transition in new_transitions:
                    record_transition(
                        session,
                        job=job_row,
                        state=transition.state.value,
                        detail=transition.detail,
                        exit_code=record.exit_code if transition.state is record.state else None,
                    )
            seen = len(record.transitions)
        if record.is_terminal:
            break
        time.sleep(_FOLLOW_INTERVAL_S)

    final = runner.wait(runner_job_id)
    with session_scope(factory) as session:
        run_row = _require(session.get(RunRow, run_id), "run", run_id)
        work_dir = Path(final.work_dir or ".")
        for kind, filename in ARTIFACTS.items():
            source = work_dir / filename
            if source.is_file():
                record_artifact(session, run=run_row, kind=kind, source=source, store=store)
        summary_path = work_dir / "summary.json"
        if summary_path.is_file():
            record_summary(
                session, run=run_row, summary=json.loads(summary_path.read_text(encoding="utf-8"))
            )
    return str(final.state.value)


def submit_and_record(
    factory: sessionmaker[Any],
    store: LocalDirectoryStore,
    runner: LocalSubprocessRunner,
    *,
    config: ResolvedConfig,
    run_set: RunSetRow,
    label: str = "",
    wait: bool = True,
) -> tuple[str, str]:
    """The whole flow, for a caller that just wants a run to happen. Returns ``(run_id, state)``."""
    run_id = create_pending_run(factory, run_set=run_set, config=config, label=label)
    job_id, runner_job_id = submit(factory, runner, run_id=run_id, config=config, label=label)
    if not wait:
        return run_id, "queued"
    return run_id, finalise(
        factory, store, runner, run_id=run_id, job_id=job_id, runner_job_id=runner_job_id
    )


__all__ = [
    "ARTIFACTS",
    "create_pending_run",
    "finalise",
    "prepare",
    "submit",
    "submit_and_record",
]
