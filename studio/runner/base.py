# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The job lifecycle and the ``JobRunner`` interface.

**Nothing above this interface may assume an execution backend** (ADR-008). Today there is one
implementation, ``LocalSubprocessRunner``; Slurm and cloud-batch adapters do not exist and raise
rather than degrade. That constraint is why the lifecycle lives here as data rather than inside the
local runner: a scheduler-backed runner would report the same states and the same transitions.

Every transition is timestamped, and the record keeps all of them rather than just the current
state. "It failed" is not debuggable; "QUEUED at 14:02:11, RUNNING at 14:02:11, FAILED at 14:06:48
with exit code 1" is. The same reasoning drives what a finished job keeps on disk: the resolved
input, the captured stdout, the captured stderr and the exit code, so **a failed run can be
diagnosed without re-running it** -- which matters when re-running costs four minutes and the
failure is intermittent.

``run_coupled`` prints diagnostics to stdout rather than logging them (``[gas] dt0 stall...``,
``[coupled] NOTE:...``, ``[stop]...``), so stdout IS the run's log stream and is captured as such.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class JobState(StrEnum):
    """Where a job is. The terminal states are distinguished on purpose.

    ``FAILED`` and ``TERMINATED_ON_LIMIT`` are not the same thing and must never be collapsed: the
    first means the model could not produce a result, the second means it was still going when we
    stopped it. A run stopped on a limit has partial output that may look complete, so it is flagged
    and **never presented as converged**.
    """

    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TERMINATED_ON_LIMIT = "terminated_on_limit"


#: States from which no further transition is possible.
TERMINAL_STATES = frozenset(
    {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLED,
        JobState.TERMINATED_ON_LIMIT,
    }
)

#: The only transitions the lifecycle allows. Enforced rather than documented: a job that went
#: RUNNING -> QUEUED, or that reported SUCCEEDED twice, means the runner lost track of a process,
#: and silently accepting it would make the record a story rather than a log.
_ALLOWED: dict[JobState, frozenset[JobState]] = {
    JobState.DRAFT: frozenset({JobState.QUEUED, JobState.CANCELLED}),
    JobState.QUEUED: frozenset({JobState.RUNNING, JobState.CANCELLED, JobState.FAILED}),
    JobState.RUNNING: frozenset(
        {
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.TERMINATED_ON_LIMIT,
        }
    ),
    JobState.SUCCEEDED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(),
    JobState.TERMINATED_ON_LIMIT: frozenset(),
}


class InvalidTransitionError(ValueError):
    """An illegal lifecycle transition. Raised, never tolerated."""


def _now() -> datetime:
    """Timezone-aware UTC. Naive timestamps compare wrongly across a DST boundary."""
    return datetime.now(UTC)


class Transition(BaseModel):
    """One state change, with when and why."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: JobState
    at: datetime
    detail: str = ""


class JobRecord(BaseModel):
    """Everything known about one job. Immutable; a transition produces a new record.

    Immutable because this is the audit trail. A record that can be edited in place is a record that
    can quietly disagree with what happened.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    #: Identity of the config being run (ADR-006). Links the job to its inputs and its cache entry.
    config_hash: str
    label: str = ""
    state: JobState = JobState.DRAFT
    transitions: tuple[Transition, ...] = ()
    #: Where the resolved input, the logs and the output live. Present from submission, so a job
    #: that dies early still says where to look.
    work_dir: Path | None = None
    input_path: Path | None = None
    #: The provenance record (ADR-006), written at submit time -- before execution -- so a run that
    #: dies in minute three of four still says exactly what produced it.
    provenance_path: Path | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    exit_code: int | None = None
    #: Set when the job ends for any reason; a short human-facing explanation.
    detail: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def _first_time(self, state: JobState) -> datetime | None:
        """When the job FIRST entered ``state``, or ``None`` if it never did."""
        for transition in self.transitions:
            if transition.state is state:
                return transition.at
        return None

    @property
    def submitted_at(self) -> datetime | None:
        return self._first_time(JobState.QUEUED)

    @property
    def started_at(self) -> datetime | None:
        return self._first_time(JobState.RUNNING)

    @property
    def ended_at(self) -> datetime | None:
        for transition in reversed(self.transitions):
            if transition.state in TERMINAL_STATES:
                return transition.at
        return None

    @property
    def duration_s(self) -> float | None:
        """Wall-clock from RUNNING to the terminal state, or ``None`` if it has not run yet."""
        started, ended = self.started_at, self.ended_at
        if started is None or ended is None:
            return None
        return (ended - started).total_seconds()

    def transition_to(self, state: JobState, detail: str = "", **updates: object) -> JobRecord:
        """Return a new record in ``state``.

        Raises:
            InvalidTransitionError: If the lifecycle does not allow it.
        """
        if state not in _ALLOWED[self.state]:
            allowed = sorted(s.value for s in _ALLOWED[self.state])
            raise InvalidTransitionError(
                f"job {self.job_id} cannot go {self.state.value} -> {state.value}; "
                f"allowed from {self.state.value}: {allowed or ['(terminal)']}"
            )
        return self.model_copy(
            update={
                "state": state,
                "transitions": (
                    *self.transitions,
                    Transition(state=state, at=_now(), detail=detail),
                ),
                "detail": detail or self.detail,
                **updates,
            }
        )


@runtime_checkable
class JobRunner(Protocol):
    """How Studio executes runs. The only assumption anything above may make.

    Deliberately small. ``submit`` takes a resolved config and returns a record; everything else
    operates on a job id. A Slurm implementation would satisfy this without any caller changing.
    """

    def submit(self, config: object, *, label: str = "") -> JobRecord:
        """Queue a run. Returns immediately with a record in QUEUED."""
        ...

    def poll(self, job_id: str) -> JobRecord:
        """Current record, advancing the state if the process has finished or exceeded its limit."""
        ...

    def cancel(self, job_id: str) -> JobRecord:
        """Stop a queued or running job. Terminal jobs are returned unchanged."""
        ...

    def artifacts(self, job_id: str) -> Sequence[Path]:
        """Files this job produced, if any."""
        ...


class NotImplementedRunner:
    """Base for backends that do not exist yet (ADR-008).

    Present so that "Slurm is not supported" is a class you can point at rather than a gap in a
    dispatch table. Every method raises with the same message; nothing degrades to local execution,
    because a job silently running somewhere other than where it was sent is worse than an error.
    """

    backend_name = "unimplemented"

    def _raise(self) -> None:
        raise NotImplementedError(
            f"the {self.backend_name} execution backend is not implemented. ADR-008 records local "
            f"subprocesses as the only supported backend; nothing falls back to local execution, "
            f"because a job running somewhere other than where it was sent is worse than an error."
        )

    def submit(self, config: object, *, label: str = "") -> JobRecord:
        self._raise()
        raise AssertionError("unreachable")  # pragma: no cover

    def poll(self, job_id: str) -> JobRecord:
        self._raise()
        raise AssertionError("unreachable")  # pragma: no cover

    def cancel(self, job_id: str) -> JobRecord:
        self._raise()
        raise AssertionError("unreachable")  # pragma: no cover

    def artifacts(self, job_id: str) -> Sequence[Path]:
        self._raise()
        raise AssertionError("unreachable")  # pragma: no cover


class SlurmRunner(NotImplementedRunner):
    """Not implemented (ADR-008, BLOCKING-1)."""

    backend_name = "Slurm"


class CloudBatchRunner(NotImplementedRunner):
    """Not implemented (ADR-008, BLOCKING-1)."""

    backend_name = "cloud-batch"


class JobRegistry(BaseModel):
    """In-memory job records, keyed by id.

    Phase 0 keeps this in the process that owns the worker pool (ADR-007). ADR-004 puts job state in
    the database so a handle survives an API restart; this class is the seam that will move there,
    and it is deliberately dumb so that swap is a swap rather than a rewrite.
    """

    model_config = ConfigDict(extra="forbid")

    records: dict[str, JobRecord] = Field(default_factory=dict)

    def put(self, record: JobRecord) -> JobRecord:
        self.records[record.job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord:
        try:
            return self.records[job_id]
        except KeyError:
            raise KeyError(f"unknown job id {job_id!r}") from None

    def __contains__(self, job_id: object) -> bool:
        return job_id in self.records


__all__ = [
    "TERMINAL_STATES",
    "CloudBatchRunner",
    "InvalidTransitionError",
    "JobRecord",
    "JobRegistry",
    "JobRunner",
    "JobState",
    "NotImplementedRunner",
    "SlurmRunner",
    "Transition",
]
