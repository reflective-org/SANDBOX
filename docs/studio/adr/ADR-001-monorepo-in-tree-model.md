# ADR-001 — Plume Studio lives in this repository; the model is consumed in-tree

**Status:** Accepted (2026-08-13) · Supersedes the draft spec's ADR-001

## Context

The architecture spec called for a monorepo containing the schema package, API, worker and web
client, with the scientific model consumed as *a pinned versioned dependency (release tag or built
artefact), not a git submodule*. The stated rationale was that submodules break CI caching, make
reproducible builds harder, and encourage accidental in-place edits of the model.

That rationale is sound in general but does not survive contact with this codebase:

- The thing Plume Studio must call is `coupled.driver.run_coupled` (`coupled/driver.py:216`), which
  lives in **this** repository, not in a submodule.
- It is **not pip-installable**. `coupled/model_bridge.py:22-31` and `coupled/tomas_bridge.py:26-32`
  place `stratchem-jax/`, `tuvx-jax/` and `tomas-jax/` on `sys.path` at import time. The package
  works because the process runs from the SANDBOX root, not because anything is installed.
- **No component is published anywhere.** There is no release tag, no wheel, no registry entry, and
  no `__version__` on the coupling layer. "Pin a versioned artefact" is not a configuration choice
  here; it is a packaging project that the spec did not budget for.
- Plume Studio needs a small number of changes *inside* `coupled/` to function at all — widening the
  `stop_condition` callback, hoisting `BACKGROUND_MODES` out of the JAX-importing module, accepting
  user-defined lognormal background modes. Under a two-repo split each of these becomes a release
  cycle.

## Decision

Plume Studio is built **in this repository**, as a top-level `studio/` package alongside `coupled/`.
The model is consumed **in-tree** by direct import.

Run provenance therefore records the **SANDBOX commit SHA plus the three submodule SHAs**
(`tuvx-jax`, `stratchem-jax`, `tomas-jax`), which together pin the model exactly. This gives the
reproducibility guarantee the spec's ADR-001 was reaching for, without the packaging work.

Two structural adjustments follow from the repository's existing layout:

- **Studio docs live under `docs/studio/`.** A top-level `docs/` already exists and holds the
  model's own `ARCHITECTURE.md`, `ASSUMPTIONS.md`, `CAVEATS.md`, `DECISIONS.md` and `PROGRESS.md`,
  all explicitly scoped "(SANDBOX)". Writing the spec's `docs/` tree at top level would have
  overwritten them.
- **Studio tests live under `studio/tests/`**, matching the repository's existing convention
  (`coupled/tests`, and one suite per submodule) rather than introducing a competing root `tests/`.

## Consequences

- Changes to `coupled/` and to `studio/` can ship in one reviewed change, but **must not ship in one
  PR** — model-side changes go in their own PR with their own tests, so that a model behaviour change
  is never buried inside an app feature.
- `studio/schema` and `studio/science` **must not import `coupled`**. Importing it pulls in JAX and
  TOMAS, which an API validating a form on every keystroke cannot afford. These two packages must be
  usable from a bare Python session. Enforced by an import-linter test.
- Root `pytest` currently sets `testpaths = ["coupled/tests"]`. Studio's suite is added explicitly
  rather than by widening that default, so the existing "three separate test suites" contract in
  `CLAUDE.md` still holds.
- The spec's warning about accidental in-place model edits is real and is not mitigated by structure
  here. It is mitigated by process: golden-file tests (ADR-009) fail loudly if model behaviour moves.
- If Plume Studio is ever extracted to its own repository, the seam to sever is `studio/modelio/` —
  the only package permitted to import `coupled`.

## Alternatives rejected

**Separate repo, model pinned as a versioned artefact** (the spec as written). Requires first
packaging `coupled` and the three submodules as installable, versioned distributions and standing up
somewhere to publish them. Every model-side fix becomes a two-repo release cycle. Rejected as
disproportionate to a single-tenant tool for a small group.

**Separate repo, model vendored at a pinned commit** (the spec's own fallback). Cleaner boundary and
honest provenance, but co-development friction is high precisely where the work is densest in Phase
0, and it does not remove the packaging problem — it relocates it.
