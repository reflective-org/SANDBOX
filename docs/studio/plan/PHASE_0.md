# Phase 0 — Skeleton and vertical slice

**Goal:** one hardcoded end-to-end path, real from top to bottom, however narrow.

**Exit criteria:** a run can be submitted from the CLI **and** from the web UI, produces a stored
result with full provenance, and the golden tests pass.

One task = one issue = one branch = one PR. Nothing is built on top of `studio/schema` until 0.2 is
reviewed and merged.

---

## 0.1 — Repo, CI, docs skeleton · *done*

`docs/studio/` tree; `OPEN_QUESTIONS.md` seeded from spec §13 with a GitHub issue per remaining row
(`blocking` / `science` labels); ADR-001…009; `studio/` package skeleton; ruff + black + `mypy
--strict` on `studio/schema` and `studio/science`, enforced in CI; committed lockfile.

Delivered as above, with two things worth stating plainly rather than leaving implied:

- **CI is scoped to `studio/`.** The repository had no CI at all before this. Putting the model's
  three suites under it needs the private submodules checked out with a deploy key, and is its own
  change — not something to bury in the app layer's skeleton.
- **Issues #52–#57** cover the six open rows (BLOCKING-2, SCIENCE-1…5), labelled `studio` +
  `blocking` / `science`. They deliberately do *not* reuse the `phase-N` labels: those belong to the
  coupled model's phases (issues #11–#28). `OPEN_QUESTIONS.md` remains the source of truth.

---

## 0.2 — `studio/schema` v0 · **review gate** · *implemented, awaiting review* (issue #61)

- `SciField(unit=…, range=…, provenance=…, cite=…, derived_from=[…])` over
  `Field(json_schema_extra=…)`.
- Minimum-viable `RunConfig` covering exactly what the Phase-0 slice needs — resist modelling stages
  4–7 fully before anything runs.
- `RunSet` + axes (`GRID` / `ZIP` / `LIST`) **from day one**. N = 1 is a RunSet with zero axes; there
  is no separate single-run code path. Bolting sweeps on later would mean rewriting the config layer,
  the results schema and the comparison views.
- Canonical JSON serialisation → stable SHA-256. **Hash stability tested across dict insertion order
  and across Python versions** — a drifting hash silently invalidates every cache and fixture.
- JSON Schema export; round-trip tests.
- Field `provenance` populated from the sources that already exist:
  `paper_ensemble/TABLE_microphysics_parameters.md`, `TABLE_dilution_parameters.md`, and the
  `CoupledScenario` docstrings, which are unusually good help text already.

**Request review explicitly before building on it.**

Delivered as specified, with two divergences recorded in `PROGRESS.md` rather than absorbed
silently: `max_sim_time` is optional (simulated time is already bounded by `schedule.duration_days`,
and a required second bound would need an invented default), and `DilutionRegime.CONSTANT` is
spelled `"constant"` where the model spells it `""` — the only enum value that is not the model's
own string, and one the 0.4 equivalence test must cover explicitly.

---

## 0.3 — Dependency-graph engine and override semantics

The mechanism that makes "go back and edit stage 1 without losing your stage 6 choices" correct by
construction. It is a data-model problem, not a UI problem.

- Build an explicit DAG from `derived_from` metadata.
- On a change to field X, compute the downstream closure. For each downstream field: **auto** →
  recompute silently; **user_override** → do not overwrite, mark **stale**, and present the old
  value, the newly-derived value, and a choice.
- Never persist an internally inconsistent config without an explicit `stale_fields` list attached.
- Tests assert that changing an upstream field recomputes *exactly* the expected downstream set and
  preserves overrides.

---

## 0.4 — `studio/modelio` seam + `RunSummary`

- `to_scenario(RunConfig) -> CoupledScenario` — the single conversion point, and the **only** package
  permitted to import `coupled`.
- **The equivalence test.** `to_scenario` applied to a `RunConfig` describing case
  `30N_20km__sabr220__D2med__a1p0__nuc1__cg1` must produce a `CoupledScenario` **field-for-field
  identical** to `run_ensemble.build_scenario()` for that case. This is what proves the schema is a
  faithful superset *before* anything is run — cheaper and sharper than discovering it from a
  diverging result.
- `RunSummary` — versioned reduced structure (`schema_version`, scalar time series, final size
  distribution, termination reason, flags, conservation-check residuals), written next to
  `state.npz`. Every array declares wet-vs-dry basis. Comparison plots read this, never the raw npz.

---

## 0.5 — `studio/science` derivations · *done* (issue #64)

One cited, tested implementation of each derivation that currently exists several times over.

**Consolidate:**
- V₀ and mass → initial concentration — **six** copies today, not five: `run_ensemble.py:41-46,95`,
  `run_60day.py:37`, `make_rf_runs.py:44`, `viz/bake_plume_dynamics.py:62-63`,
  `coupled/run_dilution_d1_clean.py:61,130` (note: at `coupled/`, not `coupled/paper_ensemble/`),
  and `run_boxsize.py:43`. Two different V₀ values — a 15 km track and a 30 km one, a factor of two.
- dN/dlogDp and bin edges — four copies, written two ways. **Corrected by 0.5:** the two
  expressions (`run_ensemble.py:147-149` vs `coupled/run_dilution_d1_clean.py:587-589`) are
  *algebraically identical*, not two conventions — `10**(0.5*(log a + log b)) == sqrt(a*b)` — and
  measured agreement on an 80-bin grid is to a few ULP (≤ 7e-16 relative). The consolidation is
  still worth doing; the divergence claim was not accurate.

**Reuse, do not rewrite:** `coupled/dilution.py:62` `volume_ratio` and `:74` `kdil_from_regime` (both
tested); `air_number_density` (`stratchem-jax/config.py:55`); `coupled/aerosol_props.py`;
`coupled/units.py:26,35`.

**New:** a GCR ion-pair parameterisation. Today `ion_pair_rate` is a bare `30.0` with no derivation —
the default should not be zero, and it should be a cited function of altitude, latitude and
solar-cycle phase. If no citation is agreed, it raises `NotImplementedError` rather than returning a
plausible number. Either way it is a `[SCIENCE]` issue.

Note the physical-constants discipline here: `coupled/units.py:18-22` documents a deliberate ~0.036 %
Avogadro mismatch at the gas/TOMAS seam. Studio inherits it and does not silently "correct" it.

---

## 0.6 — `studio/runner` + job lifecycle

`JobRunner` Protocol; `LocalSubprocessRunner` launching `python -m studio.cli.run` with the
thread-pinning environment from `launch_parallel.py:26-30`. Lifecycle
`DRAFT → QUEUED → RUNNING → (SUCCEEDED | FAILED | CANCELLED | TERMINATED_ON_LIMIT)`, every transition
timestamped.

A failed run must be debuggable **without re-running**: capture stderr, exit code and the resolved
input file. `run_coupled` prints rather than logs, so stdout is captured as the run's log stream.
`max_wall_time` and `max_sim_time` are required and enforced here.

---

## 0.7 — Golden-file harness · *do this early*

**First task is a measurement, not an assertion.** Re-run two archived cases at today's submodule
SHAs and record the observed per-quantity deviation in
`studio/tests/golden/REFERENCE_TOLERANCES.md`, together with those SHAs. Bit-for-bit is a hypothesis
(ASSUMPTION-2), not a premise.

- **Tier A** (`-m tier_a`, CI, seconds): schema/units/DAG/hash/expansion tests + one 1-day, 40-bin run.
- **Tier B** (`-m tier_b`, nightly/manual): 4–6 curated cases across D1/D2/D3/burst × sabr220/sabr330.

Fixtures store a documented **uniform**-stride reduction, not the 3.6 MB npz. Uniform because an
adaptive grid aliases the nucleation burst by up to 8×.

---

## 0.8 — Four contained fixes in `coupled/` · *separate PR, own tests*

1. Hoist `BACKGROUND_MODES` out of `tomas_bridge` so scenario validation does not import JAX
   (`coupled_scenario.py:196`).
2. Widen `stop_condition` to receive a diagnostics dict rather than only `(t1, wet_SA)`, so
   termination criteria can reference SO₂ and particle number.
3. Accept user-defined lognormal background modes — `_seed_lognormal` (`tomas_bridge.py:110`) already
   takes arbitrary `(N, Dg, σg)` tuples, so this is small.
4. Remove or honour the dead `output_dir` field (`coupled_scenario.py:147`).

Model behaviour changes never ship inside an app-feature PR.

---

## 0.9 — Vertical slice

One case, one dilution curve, one background, 40 bins — submitted from **both** the CLI and a minimal
web UI, producing a stored result with full provenance (config hash, SANDBOX SHA + three submodule
SHAs, resolved parameter set) and one figure generated deterministically from `RunSummary`.

FastAPI + SQLAlchemy/Alembic on SQLite; React + Vite; SSE for progress.

---

## Deferred to Phase 1+

Climatology ingest and tropopause definitions (Phase 1, ERA5 via CDS); the full stage 2–7 schema;
comparison UI; caching; runtime estimation; the education layer. Planned one phase ahead, against
working code.
