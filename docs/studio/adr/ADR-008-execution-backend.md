# ADR-008 — Execution via local subprocesses, behind a `JobRunner` interface

**Status:** Accepted (2026-08-13) · Answers BLOCKING-1

## Context

BLOCKING-1 offered three execution environments: same host as the API, a Slurm/PBS cluster, or cloud
batch. The relevant facts:

- A 10-day / 80-bin run takes **~3–5 min**; a 60-day run **~30–40 min**
  (`coupled/paper_ensemble/README.md:33`).
- The existing launcher, `coupled/paper_ensemble/launch_parallel.py`, slices the case list and
  `subprocess.Popen`s one worker per slice, **each pinned to a single thread** via
  `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`,
  `NUMEXPR_NUM_THREADS=1` and `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` (`:26-30`). Its docstring
  is emphatic that this pinning — **not `vmap`** — is what delivers ~N× CPU throughput for this
  workload.
- The 810-run paper ensemble was produced this way.
- There is no cluster in evidence, and no containerisation of JAX + the three submodules exists.

`run_coupled` is a **library function**, not an executable (`coupled/driver.py:216`). It returns
arrays in memory and writes nothing. Every existing runner script supplies its own `main` with the
same CLI shape: `plan | one <i> | run <lo> <hi>`.

## Decision

All execution goes through the `JobRunner` Protocol:

```python
class JobRunner(Protocol):
    def submit(self, run: Run) -> JobHandle: ...
    def poll(self, handle: JobHandle) -> JobState: ...
    def cancel(self, handle: JobHandle) -> None: ...
    def fetch_artifacts(self, handle: JobHandle) -> list[Artifact]: ...
```

**`LocalSubprocessRunner` is the only implementation.** It launches
`python -m studio.cli.run <config.yaml> <outdir>` — a thin shim added in Phase 0 so that a library
call has a process to be — with the thread-pinning environment above, from a bounded worker pool
(default 4).

`SlurmRunner` and `CloudBatchRunner` are **not written**. Should they be stubbed at all, they raise
`NotImplementedError` (ADR-005). **Nothing above the `JobRunner` interface may assume an execution
backend.**

Lifecycle: `DRAFT → QUEUED → RUNNING → (SUCCEEDED | FAILED | CANCELLED | TERMINATED_ON_LIMIT)`, every
transition timestamped and logged.

## Consequences

- **A failed run must be debuggable without re-running.** The runner captures stderr, exit code, and
  the resolved input file. Note that `run_coupled` **prints** diagnostics to stdout rather than
  logging them (`[gas] dt0 stall…`, `[coupled] NOTE:…`, `[stop]…`), so stdout is captured as the run's
  log stream — this is a fact about the model that the runner has to accommodate, not a thing to fix
  in the model.
- `max_wall_time` and `max_sim_time` are **required** config fields, enforced by the runner. A run
  stopped by either is flagged `TERMINATED_ON_LIMIT` and never presented as converged (ADR-005).
- The worker pool is in-process (ADR-007), so cancelling means terminating a child process; the
  handle must survive an API restart, which means job state lives in the database, not in memory.
- Subprocess isolation is a genuine benefit beyond parallelism: `coupled.tomas_bridge` sets
  `jax_enable_x64` at import and `coupled/model_bridge.py` mutates `sys.path` globally. Keeping that
  out of the API process is worth doing regardless of concurrency.
- Because Studio and the model share a machine, a runaway run degrades the UI. The bounded pool is
  the only defence in Phase 0; this is accepted.

## Alternatives rejected

**Slurm/PBS submission adapter.** Needs `sbatch` templating, `sacct` polling, shared-filesystem
assumptions and a no-outbound-internet compute-node story — none of which can be designed without a
cluster to design against.

**Cloud batch / Kubernetes Jobs.** Largest lift: container images for JAX plus three submodules, and
object-storage-only I/O from day one. Reasonable if this ever needs to run thousands of cases; there
is no such requirement.

**In-process execution, no subprocess.** Simplest, and avoids the CLI shim. Rejected: a JAX run in
the API process cannot be cancelled or wall-clock-capped reliably, and the import-time global state
noted above would contaminate the server.
