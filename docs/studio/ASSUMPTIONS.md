# Assumptions — Plume Studio

Every `[ASSUMPTION]` made to unblock work, with the code location it affects and what would settle
it. An assumption is a provisional choice made **visibly**; it is not a licence to guess.

If implementing something requires a number, a formula, a units convention, or a scientific
definition that is not stated in the spec or a cited reference, it belongs here or in
[`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md). It is never a thing to quietly pick.

Model-side assumptions inherited from the coupled model are in
[`../ASSUMPTIONS.md`](../ASSUMPTIONS.md) and are not repeated here.

---

## ASSUMPTION-1 — Canonical units are the model's native units, not SI

**Made:** 2026-08-13 · **Affects:** `studio/schema/units.py`, `studio/modelio/` ·
**Recorded in:** [ADR-003](adr/ADR-003-units.md)

The spec's ADR-003 says "SI (or a documented canonical set)". Studio takes the second branch and
declares the canonical set to be the model's own units — mbar, ppm, pptv, K, s, µm² cm⁻³, cm⁻³ s⁻¹.

**Why.** The existing ensemble's constants are decimal values in native units, e.g. `WTR = 6.9104`
ppm (`coupled/paper_ensemble/run_ensemble.py:62`). Round-tripping through SI (`6.9104e-6 mol mol⁻¹ →
×1e6`) is not guaranteed to return the same float64. Since we are committing to reproduce the
existing runs within a measured tolerance ([ADR-009](adr/ADR-009-golden-file-strategy.md)), a
canonical set that guarantees lossless passthrough at the model seam is worth more than SI purity —
otherwise a units convention could perturb the golden runs and make "the app converted badly"
indistinguishable from "the model changed".

**What would settle it.** Nothing needs to; this is a deliberate, documented choice. It would only be
revisited if Studio ever drives a model that does not use these units, at which point SI-canonical
plus a conversion layer becomes the better trade.

---

## ASSUMPTION-2 — The archived `state.npz` files are the golden reference, at a tolerance yet to be measured

**Made:** 2026-08-13 · **Settled:** 2026-08-13 (#70) · **Affects:** `studio/tests/golden/` ·
**Recorded in:** [ADR-009](adr/ADR-009-golden-file-strategy.md)

> **Settled.** The measurement exists:
> [`studio/tests/golden/REFERENCE_TOLERANCES.md`](../../studio/tests/golden/REFERENCE_TOLERANCES.md).
> Reproduction is **close but not bit-for-bit** — every headline quantity within 2.1e-12, worst
> deviation anywhere 3.4e-12, but only ~31 % of gas state-vector elements bit-identical. The archive
> is usable as a golden reference at ~1e-12 (endpoints) / ~1e-10 (series and per-bin size
> distribution); exact equality is not. The paragraphs below stand as the reasoning that made the
> measurement necessary; the *consequence* below still holds and cannot be retrofitted.

Golden fixtures are derived from the existing `coupled/paper_ensemble/runs*/` outputs. Whether
re-running those cases **today** reproduces them bit-for-bit is *unverified*: the submodule commits
at which they were produced were never recorded (there is no provenance record for the existing
ensemble — [ADR-006](adr/ADR-006-provenance.md)), and `docs/DECISIONS.md` documents at least one
solver-level change since, the 2026-07-08 day-12.139 first-step fix after which the SciPy-BDF
fallback was removed.

The first golden-harness task is therefore to **measure** the deviation, not assume it, and commit
the measurement together with the SHAs it was taken at.

**Consequence.** Golden fixtures record the SHA at which the reference was *measured*, not the SHA at
which the data was originally produced. This is an honest limitation and cannot be retrofitted.

**What settled it.** The measurement itself, in `studio/tests/golden/REFERENCE_TOLERANCES.md`
(2026-08-13, issue #70). It also produced a result nobody had asked for: the model is bit-for-bit
deterministic run-to-run *today*, so the residual is drift between the archive's toolchain and this
one — which is what makes a 1e-12 tolerance defensible rather than arbitrary.

---

## ASSUMPTION-3 — Single-tenant deployment, reverse-proxy auth, no in-app user model

**Made:** 2026-08-13 · **Affects:** the `run` table schema ·
**Tracks:** [BLOCKING-2](OPEN_QUESTIONS.md#blocking-2--tenancy-and-auth-model--open-provisional-assumption-in-force)

Provisional per spec §2, pending BLOCKING-2. Authentication is delegated to a reverse proxy; there is
no user model, no per-user quota, and no row-level access control in v1.

Every `run` row nevertheless carries a **nullable `owner` string**, so the column exists and adding
ownership later is a backfill rather than a migration touching every query.

**What would settle it.** An answer to BLOCKING-2, required before Phase 6.

---

## ASSUMPTION-4 — Default worker pool of 4 and a 60-minute per-run wall-clock cap

**Made:** 2026-08-13 · **Affects:** `studio/runner/local.py` config defaults ·
**Tracks:** [BLOCKING-4](OPEN_QUESTIONS.md#blocking-4--wall-time-and-concurrency-budget--answered-2026-08-13)

BLOCKING-4's *measurements* are known (3–5 min for a 10-day/80-bin case; 30–40 min for 60 days; the
existing launcher uses 10 thread-pinned subprocesses). The *policy* ceiling is not: how long a single
user-submitted run may occupy a worker is a deployment choice.

Defaults chosen: **4 workers**, **60 min** per-run wall clock. Four rather than ten because, unlike
the batch launcher, Studio shares the machine with an interactive API; sixty minutes because it
admits a 60-day run without admitting an unbounded one.

Both are configuration, not constants, and are surfaced in the settings rather than buried.

**What would settle it.** Observed usage — the runtime-estimation feature (spec §7.5) can be seeded
from the existing `runs*/worker_*.log` files and will make the right cap obvious.

---

## ASSUMPTION-5 — Phase 0 defaults follow the golden runs, not the spec's geometry

**Made:** 2026-08-13 · **Affects:** `studio/schema` defaults, `studio/science/plume_volume.py` ·
**Tracks:** [SCIENCE-2](OPEN_QUESTIONS.md#science-2--definition-of-t--0--open--blocks-phase-2-caveats-all-results)

The spec's §4.2 gives a default source cross-section of 10 m × 30 m. The existing 810-run ensemble
uses **10 m × 10 m × 15 km** (`run_ensemble.py:45`, giving V₀ = 1.5e12 cm³) and the D1 flagship run
uses a 30 km track (`run_dilution_d1_clean.py:61`, V₀ = 3e6 m³).

Phase 0 defaults follow the ensemble, because Phase 0's job is to reproduce existing trusted runs.
The spec's geometry is not implemented as a default until SCIENCE-2 resolves what t = 0 means.

**Note, and it matters for how this is presented:** V₀ **does not enter the dynamics**. It only sets
the initial SO₂ concentration; the model is intensive and volume-invariant
(`coupled/tests/test_boxvol_invariance.py`; the box-size sweep `run_boxsize.py:43` works purely by
scaling the initial concentration). The UI must say so rather than implying a geometric dependence
that does not exist.

**What would settle it.** SCIENCE-2.

---

## ASSUMPTION-6 — Studio docs live under `docs/studio/`, tests under `studio/tests/`

**Made:** 2026-08-13 · **Affects:** repository layout · **Recorded in:**
[ADR-001](adr/ADR-001-monorepo-in-tree-model.md)

The spec's §10 layout puts `ASSUMPTIONS.md`, `CAVEATS.md`, `PROGRESS.md`, `adr/` and `science/` at
`docs/` top level, and tests at a root `tests/`. Both collide with what is already here: `docs/`
holds the **model's** `ARCHITECTURE.md`, `ASSUMPTIONS.md`, `CAVEATS.md`, `DECISIONS.md` and
`PROGRESS.md`, all scoped "(SANDBOX)", and the repository's test convention is one suite per
component (`coupled/tests`, plus one per submodule).

Studio therefore namespaces itself: `docs/studio/` and `studio/tests/`.

**What would settle it.** Nothing pending; this is a layout decision, recorded because the spec says
otherwise and a reader comparing the two will notice.

## ASSUMPTION-7 — Platform speed of 250 m/s, chosen rather than sourced

**Where:** `studio/schema/config.py` — `Injection.platform_speed_m_s` (schema 0.3.0).

**What is assumed.** That 250 m/s is a reasonable ground speed for an emitting platform in the
lower stratosphere, and a defensible default for the `RATE_AND_SPEED` emission basis.

**Why it is an assumption and not a value.** Nothing in this repository carries a platform speed.
The paper ensemble specifies the release geometrically — 1 t into 10 m × 10 m × 15 km
(`run_ensemble.py:45`) — and neither the model nor the ensemble has any notion of an aircraft, a
speed, or an emission rate. So this default has no upstream source to cite, and 250 m/s is a round
number chosen in conversation (Ali, 2026-08-17), not an airframe specification.

**What it affects.** Only configurations that select `RATE_AND_SPEED`. Under the default
`MASS_AND_LENGTH` basis the field is inert, so no existing run or archived comparison is touched.
Where it does apply it is consequential: the track length is `speed × duration`, so the speed sets
the dilution of the release linearly, and a 2× error in speed is a 2× error in initial
concentration.

**How to remove it.** Cite an airframe and a cruise condition, and move the field's provenance from
`CONVENTION` to `LITERATURE` with that citation. `test_the_speed_default_is_marked_as_a_choice_not_a
_measurement` asserts the caveat is present, so it cannot be quietly dropped without the citation
that would justify dropping it.

