# ADR-004 — Gridded climatology does not live in a relational database

**Status:** Accepted (2026-08-13)

## Context

Plume Studio needs three quite different kinds of data:

1. **Gridded climatology** — ERA5 fields on (month × latitude × pressure level), possibly longitude-
   resolved (SCIENCE-1). Read-mostly, array-shaped, interpolated.
2. **Run metadata** — configurations, job state, run sets, provenance. Small, relational, queried by
   identity and by attribute.
3. **Raw model output** — `state.npz` per run, **~3.6 MB for a 10-day case and ~6.1 MB for a 60-day
   case**, with the full ensemble running to 1–2 GB (`coupled/paper_ensemble/README.md:38`).

Putting any of these in the wrong store is expensive to undo.

## Decision

Three stores, matched to the three shapes:

- **Climatology → NetCDF / Zarr, read via `xarray`.** A *precomputed reduced* product, not raw ERA5.
  The preprocessing lives in `data/pipelines/` as a committed, re-runnable pipeline with recorded
  input identifiers and output checksums — **not a one-off notebook**. The API exposes derived
  quantities (tropopause height per definition; T, p, H₂O at a target level), never raw grids.
- **Run metadata → a relational database**, via SQLAlchemy 2.x with Alembic migrations (ADR-007).
  Tables: `run_set`, `run`, `run_config`, `job`, `result_artifact`, `dataset_version`,
  `reference_distribution`, `mechanism_version`.
- **Raw output → object storage, keyed by `run_id`.** The database stores pointers, sizes, checksums
  and content types — never the arrays. In the Phase-0 local deployment "object storage" is a
  directory on disk behind an interface; MinIO or S3 substitutes without touching callers.

Two rules on top:

- **Configs are immutable once submitted.** Editing produces a *new* config and a *new* run, linked
  by `derived_from_run_id`. Results are never overwritten.
- **A standardised, versioned `RunSummary`** is derived per run — small and queryable: scalar time
  series, final size distribution, integrated diagnostics, termination reason, flags. Comparison
  plots and figures read the summary, **never** the raw output. Figure generation is a deterministic
  function of the summary, so figures are cacheable and regenerable and a lost figure is never a lost
  result.

## Consequences

- The interpolation method (in pressure level and in latitude) is a documented, unit-tested decision
  against hand-computed cases, not an `xarray` default that happens to be in force.
- Each climatology dataset entry carries source, version, temporal coverage, citation, licence and
  **known limitations** — for ERA5 specifically, the stratospheric water-vapour dry bias (BLOCKING-5).
- `RunSummary` needs a `schema_version` of its own and its own migration story, because it is a
  persisted contract that outlives the code that wrote it.
- Golden-file tests must key on the **raw npz**, not on the summary, since the summary is a lossy
  reduction (ADR-009).

## Alternatives rejected

**Climatology in Postgres** (PostGIS or array columns). Rejected: the access pattern is
multi-dimensional slicing and interpolation, which is what `xarray` exists for and what SQL is worst
at. It would also make the derived product uncommittable and unversioned.

**Raw `state.npz` as a database BLOB.** Rejected on size: at 3.6 MB per run, a 810-run ensemble is a
2.9 GB table, backed up and replicated on every operation, for data that is written once and read by
path.

**No summary layer — figures read the raw npz directly.** This is what the repository does today, and
it is instructive: `make_paper_candidate_plots.py:102` reduces all 810 npz files into a memoised
cache that has **no version stamp and no input-hash key**, invalidated only by manual deletion
(`paper_ensemble/README.md:119-121`). Golden tests keyed on a stale cache would pass vacuously. The
versioned summary exists specifically so that cannot happen.
