# Architecture decision records — Plume Studio

Format: context, decision, consequences, alternatives rejected. Newest additions at the bottom.

Three of these **amend or supersede** the corresponding record in the draft architecture spec. Where
that happens the ADR says so explicitly and gives the reason, so that a reader holding the spec can
see the divergence rather than discover it.

| ADR | Title | Status |
|---|---|---|
| [001](ADR-001-monorepo-in-tree-model.md) | Plume Studio lives in this repository; the model is consumed in-tree | Accepted — **supersedes** spec ADR-001 |
| [002](ADR-002-schema-is-source-of-truth.md) | The run configuration schema is the single source of truth | Accepted |
| [003](ADR-003-units.md) | Units are explicit; the canonical set is the model's native units, not SI | Accepted — **amends** spec ADR-003 |
| [004](ADR-004-data-layer.md) | Gridded climatology does not live in a relational database | Accepted |
| [005](ADR-005-fail-loud.md) | Fail loud, never fall back silently | Accepted |
| [006](ADR-006-provenance.md) | Every run records its provenance, immutably | Accepted |
| [007](ADR-007-stack.md) | Stack: lightweight first, with the migration path kept open | Accepted — **amends** spec ADR-007 |
| [008](ADR-008-execution-backend.md) | Execution via local subprocesses, behind a `JobRunner` interface | Accepted — answers BLOCKING-1 |
| [009](ADR-009-golden-file-strategy.md) | Two-tier golden-file regression against the existing ensemble | Accepted |

## Divergences from the spec, in one place

- **ADR-001** — the spec assumes the model is packageable as a pinned artefact. It is not: `coupled/`
  is not pip-installable, reaches its three submodules by `sys.path` injection, and nothing is
  published or versioned. Studio is built in-tree instead, and pins the model by recording the
  SANDBOX SHA plus three submodule SHAs.
- **ADR-003** — the spec prefers SI. Studio uses the model's native units as the canonical set, to
  guarantee lossless float passthrough at the model seam and so protect golden reproduction.
- **ADR-007** — the spec's stack assumes Postgres, Redis, Celery and Docker. Studio starts on SQLite
  with an in-process worker pool, keeping the migration seams (Alembic, `JobRunner`, storage
  interface) explicit.

## Writing a new one

Number sequentially; never renumber or delete. A decision that is later reversed gets a new ADR that
supersedes the old one, and the old one's status changes to *Superseded by ADR-nnn* — the record of
having believed the earlier thing is part of the value.
