# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""``LocalSubprocessRunner`` -- the only execution backend that exists (ADR-008).

Each run is one thread-pinned subprocess. The pinning is not incidental: setting
``OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1`` and disabling XLA's multithreaded Eigen is what
gives ~N times the throughput for N workers on this workload -- **not** ``vmap``. It is copied from
``coupled/paper_ensemble/launch_parallel.py:26-30``, which is the configuration the 810-run ensemble
was actually produced with.

Why a subprocess at all: ``run_coupled`` is a library call with no ``__main__`` of its own, it
returns arrays in memory and writes nothing, and it prints its diagnostics. So something has to be
the process, that something is ``python -m studio.cli.run``, and its stdout is the run's log.

What is on disk when a job ends, whatever the outcome:

* ``input.json`` -- the RESOLVED config that was actually run
* ``provenance.json`` -- what produced it: config hash, app version, SANDBOX and submodule SHAs,
  and whether any checkout was dirty (ADR-006). Written BEFORE the process starts.
* ``stdout.log`` / ``stderr.log`` -- captured in full
* the exit code and every state transition, in the record

That set is chosen so a failure can be diagnosed **without re-running it**, which matters when a
re-run costs minutes and the failure is intermittent.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final

from studio.modelio.provenance import record_for
from studio.resolve import ResolvedConfig
from studio.runner.base import JobRecord, JobRegistry, JobState

#: Single-thread pinning, copied from ``launch_parallel.py:26-30``. This is what makes N concurrent
#: runs ~N times the throughput; without it they fight over cores and each one gets slower.
THREAD_PINNING: Final[dict[str, str]] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
}

#: Default concurrent subprocesses (ASSUMPTION-4). Each run is single-threaded by the pinning above,
#: so this is a core count rather than a guess about memory. Existing practice is 10.
DEFAULT_MAX_WORKERS = 4

#: Grace period between SIGTERM and SIGKILL when stopping a run, in seconds. Long enough for Python
#: to unwind and flush the log; short enough that a wedged process does not hold a worker.
_TERMINATE_GRACE_S = 5.0


class LocalSubprocessRunner:
    """Run jobs as local subprocesses from a bounded pool.

    Args:
        work_root: Directory under which each job gets ``<work_root>/<job_id>/``.
        max_workers: Concurrent subprocesses (ASSUMPTION-4).
        entry_module: The module launched with ``-m``. Overridable so the LIFECYCLE can be tested
            without a four-minute model run -- the default is the real path, and nothing in this
            class branches on the value. It is a parameter, not a test hook.
        repo_root: Which checkout to record in each run's provenance. Defaults to the one this code
            came from, which is what a local runner should record. It is a parameter because
            "which checkout produced this?" is a real question a runner has to answer -- a worker
            executing code from elsewhere would answer it differently -- and because CI has no
            submodules, so the tests point it at a synthetic checkout. **It does not weaken the
            guarantee**: a run still cannot start unless the checkout it names can be pinned.
        python_executable: Interpreter for the subprocess; defaults to the current one, so a job
            inherits the environment that submitted it rather than whatever is first on PATH.
    """

    def __init__(
        self,
        work_root: Path,
        *,
        max_workers: int = DEFAULT_MAX_WORKERS,
        entry_module: str = "studio.cli.run",
        python_executable: str | None = None,
        repo_root: Path | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.entry_module = entry_module
        self.python_executable = python_executable or sys.executable
        self.repo_root = repo_root
        self.registry = JobRegistry()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="studio-run")
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._futures: dict[str, Future[None]] = {}

    # -- submission ------------------------------------------------------------------------

    def submit(self, config: ResolvedConfig, *, label: str = "") -> JobRecord:
        """Write the resolved input, queue the run, return immediately.

        Raises:
            InconsistentConfigError: If the config has stale overrides. A run started from one would
                produce results that do not follow from their own inputs, and nothing downstream
                could tell.
        """
        config.require_consistent()
        job_id = uuid.uuid4().hex[:12]
        work_dir = self.work_root / job_id
        work_dir.mkdir(parents=True, exist_ok=False)

        input_path = work_dir / "input.json"
        input_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

        # Provenance BEFORE execution (ADR-006). Deliberately not in a try/except: if the model
        # cannot be pinned, the run must not start. A result whose origin is unknown is worth less
        # than no result, because it looks like the others.
        provenance = record_for(config, repo_root=self.repo_root)
        provenance_path = provenance.write(work_dir / "provenance.json")

        record = JobRecord(
            job_id=job_id,
            config_hash=config.config.config_hash(),
            label=label,
            work_dir=work_dir,
            input_path=input_path,
            provenance_path=provenance_path,
            stdout_path=work_dir / "stdout.log",
            stderr_path=work_dir / "stderr.log",
        ).transition_to(JobState.QUEUED, detail=f"queued for {self.entry_module}")
        self.registry.put(record)

        max_wall_time_s = config.config.termination.max_wall_time_s
        self._futures[job_id] = self._pool.submit(self._execute, job_id, max_wall_time_s)
        return record

    # -- execution -------------------------------------------------------------------------

    def _execute(self, job_id: str, max_wall_time_s: float) -> None:
        """Run one job to completion. Runs on a pool thread; never raises into the pool."""
        record = self.registry.get(job_id)
        if record.state is JobState.CANCELLED:
            return  # cancelled while queued
        work_dir = record.work_dir
        assert work_dir is not None and record.stdout_path and record.stderr_path

        command = [
            self.python_executable,
            "-m",
            self.entry_module,
            str(record.input_path),
            str(work_dir),
        ]
        env = {**os.environ, **THREAD_PINNING}
        try:
            with (
                record.stdout_path.open("wb") as stdout,
                record.stderr_path.open("wb") as stderr,
            ):
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
                self._processes[job_id] = process
                self.registry.put(
                    self.registry.get(job_id).transition_to(
                        JobState.RUNNING, detail=" ".join(command)
                    )
                )
                try:
                    exit_code = process.wait(timeout=max_wall_time_s)
                except subprocess.TimeoutExpired:
                    self._stop(process)
                    self._finish(
                        job_id,
                        JobState.TERMINATED_ON_LIMIT,
                        detail=(
                            f"exceeded max_wall_time_s = {max_wall_time_s}; partial output is "
                            f"NOT a converged result"
                        ),
                        exit_code=process.returncode,
                    )
                    return
        except Exception as exc:
            self._finish(job_id, JobState.FAILED, detail=f"{type(exc).__name__}: {exc}")
            return
        finally:
            self._processes.pop(job_id, None)

        current = self.registry.get(job_id)
        if current.state is JobState.CANCELLED:
            return
        if exit_code == 0:
            self._finish(job_id, JobState.SUCCEEDED, detail="completed", exit_code=exit_code)
        else:
            self._finish(
                job_id,
                JobState.FAILED,
                detail=(
                    f"exit code {exit_code}; see {record.stderr_path.name} in {work_dir} -- the "
                    f"resolved input and both log streams are kept so this needs no re-run"
                ),
                exit_code=exit_code,
            )

    def _finish(
        self, job_id: str, state: JobState, *, detail: str, exit_code: int | None = None
    ) -> None:
        record = self.registry.get(job_id)
        if record.is_terminal:
            return
        self.registry.put(record.transition_to(state, detail=detail, exit_code=exit_code))

    @staticmethod
    def _stop(process: subprocess.Popen[Any]) -> None:
        """SIGTERM, then SIGKILL if it will not go. Python gets a chance to flush its log first."""
        process.terminate()
        try:
            process.wait(timeout=_TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    # -- inspection ------------------------------------------------------------------------

    def poll(self, job_id: str) -> JobRecord:
        """The current record. State is advanced by the worker thread, not by polling.

        Polling is therefore free and side-effect-free, which matters because the API pushes
        progress over SSE (spec 7.3) and would otherwise poll this on a timer per connected client.
        """
        return self.registry.get(job_id)

    def cancel(self, job_id: str) -> JobRecord:
        """Stop a queued or running job. A terminal job is returned unchanged, not an error."""
        record = self.registry.get(job_id)
        if record.is_terminal:
            return record
        process = self._processes.get(job_id)
        if process is not None:
            self._stop(process)
        cancelled = record.transition_to(JobState.CANCELLED, detail="cancelled by request")
        return self.registry.put(cancelled)

    def artifacts(self, job_id: str) -> Sequence[Path]:
        """Files this job produced, sorted. Includes the logs and the resolved input on failure."""
        record = self.registry.get(job_id)
        if record.work_dir is None or not record.work_dir.is_dir():
            return ()
        return tuple(sorted(p for p in record.work_dir.iterdir() if p.is_file()))

    def wait(self, job_id: str, timeout: float | None = None) -> JobRecord:
        """Block until the job reaches a terminal state. For the CLI and for tests, not the API."""
        future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.registry.get(job_id)

    def shutdown(self, *, cancel_running: bool = False) -> None:
        """Stop accepting work; optionally stop what is already running."""
        if cancel_running:
            for job_id in list(self._processes):
                self.cancel(job_id)
        self._pool.shutdown(wait=True)


__all__ = ["DEFAULT_MAX_WORKERS", "THREAD_PINNING", "LocalSubprocessRunner"]
