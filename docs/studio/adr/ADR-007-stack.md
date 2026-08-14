# ADR-007 — Stack: lightweight first, with the migration path kept open

**Status:** Accepted (2026-08-13) · Amends the draft spec's ADR-007

## Context

The spec proposed, as an `[ASSUMPTION]`: FastAPI + Pydantic v2 + SQLAlchemy 2.x/Alembic + **Postgres
15**, a **Celery or arq** worker with **Redis** as broker, React/Vite frontend, and **Docker Compose**
for local dev.

Weighed against what this actually has to serve (spec §2: a small trusted group, single-tenant), and
against BLOCKING-1's answer (local subprocess execution only — ADR-008), most of that infrastructure
is carrying capacity that nothing is currently asking for. Redis and Celery exist to distribute work
across machines; there is one machine. Postgres exists for concurrency and scale; the write rate is a
handful of runs an hour by one person.

The cost of the heavy stack is not licence fees, it is that Phase 0's exit criterion — *one real run,
end to end, with full provenance* — recedes behind four services that have to be stood up and kept
running first.

## Decision

**Backend.** Python 3.11+ (3.12.12 is what is installed), FastAPI, Pydantic v2, SQLAlchemy 2.x with
**Alembic from the first migration**, on **SQLite**.

**Worker.** An in-process bounded pool behind the `JobRunner` interface (ADR-008). No Redis, no
Celery, no arq in Phase 0.

**Frontend.** TypeScript, React, Vite. A plotting library capable of fast interactive updates, chosen
when the first interactive panel is built rather than now.

**Packaging.** `uv` with a fully pinned, committed lockfile.

**No Docker Compose in Phase 0.** A local venv and `uvicorn` is the dev loop.

The migration path is what makes this safe rather than merely cheap:

- **Alembic from day one** means Postgres is a connection-string change plus a migration run, not a
  rewrite. No SQLite-specific SQL, no reliance on its type affinity, no `AUTOINCREMENT` quirks.
- **The `JobRunner` Protocol** means Celery/arq/Slurm is a new implementation class, not a
  refactor of everything that submits work.
- **Object storage behind an interface** (ADR-004) means a directory today, MinIO or S3 later.

Revisit at Phase 6, when the real execution backend lands and concurrency is known.

## Consequences

- SQLite's single-writer model is a real constraint. Job-state writes must be short transactions, and
  WAL mode is on. If write contention appears before Phase 6, that is the trigger to move early.
- Progress reporting uses **server-sent events**, not client polling (spec §7.3), which works fine
  without a broker because the API process owns the worker pool.
- The dev environment is a project-local venv (`.venv`), per the engineering rules. **Never install
  into the system interpreter, never `sudo`.** Note the current machine has no `.venv` at the SANDBOX
  root and the model's dependencies are installed in the pyenv global environment — Studio does not
  adopt that; it creates and uses its own.
- Some dependencies the app needs are not currently installed anywhere: `pint`, `fastapi`,
  `sqlalchemy`, `alembic`, `uvicorn`, `mypy`, `black`. `pydantic` 2.13.4, `pytest` 9.0.2 and `ruff`
  0.15.12 are present.
- Choosing not to containerise means the reproducibility guarantee rests entirely on the lockfile and
  the recorded git SHAs (ADR-006). That is a weaker guarantee than an image digest, and it is
  accepted deliberately.

## Alternatives rejected

**The full spec stack from Phase 0.** No migration later, but four services to stand up before the
first real run works end to end, for a single-tenant tool. If usage grows past one machine this
becomes right; the Alembic/`JobRunner`/storage-interface seams are there so that switch stays cheap.

**No database at all** — runs as directories, configs as YAML, index rebuilt by scanning. Closest to
how `runs/` works today and genuinely tempting. Rejected because it gives up queryability and, more
importantly, the immutable-config guarantee: a directory of YAML has nothing preventing an edit in
place, which is precisely the reproducibility failure ADR-006 exists to close.
