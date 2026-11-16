# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Job execution: the ``JobRunner`` Protocol and its local-subprocess implementation (ADR-008).

NOTHING ABOVE THIS INTERFACE MAY ASSUME AN EXECUTION BACKEND.

Contents (task 0.6, not yet implemented):

* ``JobRunner`` -- ``submit`` / ``poll`` / ``cancel`` / ``fetch_artifacts``.
* ``LocalSubprocessRunner`` -- the only implementation. Launches ``python -m studio.cli.run`` from a
  bounded pool (default 4 workers, ASSUMPTION-4), with the thread-pinning environment proven in
  ``coupled/paper_ensemble/launch_parallel.py:26-30``: OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1
  and XLA_FLAGS=--xla_cpu_multi_thread_eigen=false. That pinning -- not vmap -- is what gives ~Nx
  CPU throughput for this workload.

Slurm and cloud-batch runners are NOT implemented and raise ``NotImplementedError`` (ADR-005).

Lifecycle: DRAFT -> QUEUED -> RUNNING -> (SUCCEEDED | FAILED | CANCELLED | TERMINATED_ON_LIMIT),
every transition timestamped and logged.

Two facts about the model this package has to accommodate rather than fix:

* ``run_coupled`` PRINTS diagnostics to stdout rather than logging them (``[gas] dt0 stall...``,
  ``[coupled] NOTE:...``, ``[stop]...``), so stdout is captured as the run's log stream.
* It is a library call that returns arrays in memory and writes nothing -- hence the CLI shim.

A failed run must be debuggable WITHOUT re-running: capture stderr, exit code, and the resolved
input file. ``max_wall_time`` and ``max_sim_time`` are required config fields, enforced here; a run
stopped by either is flagged TERMINATED_ON_LIMIT and never presented as converged.
"""

from __future__ import annotations

from studio.runner.base import (
    TERMINAL_STATES,
    CloudBatchRunner,
    InvalidTransitionError,
    JobRecord,
    JobRegistry,
    JobRunner,
    JobState,
    NotImplementedRunner,
    SlurmRunner,
    Transition,
)
from studio.runner.local import DEFAULT_MAX_WORKERS, THREAD_PINNING, LocalSubprocessRunner

__all__ = [
    "DEFAULT_MAX_WORKERS",
    "TERMINAL_STATES",
    "THREAD_PINNING",
    "CloudBatchRunner",
    "InvalidTransitionError",
    "JobRecord",
    "JobRegistry",
    "JobRunner",
    "JobState",
    "LocalSubprocessRunner",
    "NotImplementedRunner",
    "SlurmRunner",
    "Transition",
]
