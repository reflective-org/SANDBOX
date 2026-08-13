# Open questions register — Plume Studio

Seeded from the architecture spec §13. `[BLOCKING-n]` gates dependent work; `[SCIENCE-n]` affects
correctness, not architecture, so Phase 0–1 work may proceed alongside it.

**Rule:** if implementing something requires a number, a formula, a units convention, or a scientific
definition that is not stated in the spec or a cited reference, it is an entry here or in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md). It is never a thing to quietly pick.

Status values: **OPEN** · **ANSWERED** (decision recorded, link given) · **SUPERSEDED**.

Every **OPEN** row has a GitHub issue, linked in its heading, labelled `studio` + `blocking` /
`science`. Note the repository's existing `phase-1`…`phase-8` labels refer to the *coupled model's*
phases (issues #11–#28), not Studio's, so Studio issues name their phase in the text instead of
reusing those labels. This file stays the source of truth; the issues are where discussion happens.

---

## Blocking

### BLOCKING-1 — Execution environment · **ANSWERED** (2026-08-13)
*Where do simulations execute?*

**Answer: local subprocesses only.** All execution goes through the `JobRunner` Protocol
(`studio/runner/base.py`); `LocalSubprocessRunner` is the only implementation. It reuses the
thread-pinning environment already proven in `coupled/paper_ensemble/launch_parallel.py:26-30`
(`OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1`, `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`),
which is what delivers ~N× CPU throughput for this workload — not `vmap`.

Slurm and cloud-batch adapters are **not** implemented and must raise `NotImplementedError` rather
than degrade. Nothing above the `JobRunner` interface may assume an execution backend.

See [ADR-008](adr/ADR-008-execution-backend.md).

---

### BLOCKING-2 — Tenancy and auth model · **OPEN** (provisional assumption in force) · [#52](https://github.com/reflective-org/SANDBOX/issues/52)
*Single user, small group, or open?*

Provisional per spec §2 and recorded as [ASSUMPTION-3](ASSUMPTIONS.md#assumption-3): small trusted
group, single-tenant deployment, authentication delegated to a reverse proxy, **no in-app user model
in v1**. Every `run` row nevertheless carries a nullable `owner` string so the column exists and
retrofitting is a backfill rather than a migration of every query.

Must be answered before Phase 6 (real execution backend + result ownership). Retrofitting row-level
access control is expensive.

**Related constraint, already live in this repo:** `coupled/viz/README.md:38-39` records that pages
published to `gh-pages` are **public even though this repository is private**. Plume Studio is the
first component here with a server, and therefore the first place that boundary can be enforced by
code rather than by convention. Nothing is published to any public URL without explicit approval.

---

### BLOCKING-3 — Model identity, invocation, I/O, licence, versioning · **ANSWERED** (2026-08-13)
*What is the underlying model and how is it invoked?*

Answered from the code rather than by assumption:

| | |
|---|---|
| **Identity** | This repository. `coupled/` is the operator-split driver; the three models are the `tuvx-jax`, `stratchem-jax`, `tomas-jax` git submodules. |
| **Entry point** | **Not an executable.** Library call `run_coupled(scenario, ...)` — `coupled/driver.py:216`. It returns arrays in memory and writes nothing to disk. |
| **Input format** | `CoupledScenario` dataclass (`coupled/coupled_scenario.py:63`), loadable from YAML or JSON via `.load()` (`:220`). |
| **Output format** | `.npz`, written by the *caller*. Canonical key set at `coupled/paper_ensemble/run_ensemble.py:150-156`. Note `netCDF4` is a declared dependency but nothing in `coupled/` writes netCDF. |
| **Licence** | Apache-2.0, UCAR. No redistribution constraint. |
| **Versioning** | Git SHAs only — there is no `__version__` on the coupling layer, and none of the four components is published as an artefact. |

**Consequence:** Phase 0 adds a thin `python -m studio.cli.run <config.yaml> <outdir>` shim so
`LocalSubprocessRunner` has a process to launch, and provenance is recorded as the SANDBOX commit SHA
plus the three submodule SHAs. See [ADR-001](adr/ADR-001-monorepo-in-tree-model.md).

---

### BLOCKING-4 — Wall time and concurrency budget · **ANSWERED** (2026-08-13)
*What is the longest acceptable single-run wall time, and the concurrency budget?*

Measured values already documented at `coupled/paper_ensemble/README.md:33`:

- 10-day, 80-bin run: **~3–5 min** (CPU).
- 60-day run: **~30–40 min**.
- Full ensemble output: **~1–2 GB**.
- Existing practice: **10 concurrent thread-pinned subprocesses** (`launch_parallel.py`).

**Consequences.** (a) Per-commit CI cannot run a full case — hence the two-tier golden harness
(see [ADR-009](adr/ADR-009-golden-file-strategy.md)). (b) `max_wall_time` and `max_sim_time` are
**required** config fields, enforced by the runner. (c) The runtime-estimation feature (spec §7.5)
can be seeded empirically from the existing `runs*/worker_*.log` files rather than displaying
"insufficient data" from a cold start.

Still open in a narrow sense: the *policy* ceiling (how long a single user-submitted run may occupy a
worker) is a deployment choice, not a measurement. Default worker pool size and per-run wall-clock
cap are configuration, defaulting to 4 workers and 60 min.

---

### BLOCKING-5 — Climatology datasets and redistribution rights · **ANSWERED** (2026-08-13)
*Which datasets, and can they be redistributed?*

**Answer: ERA5 via the Copernicus CDS API; credentials are available.** The pipeline in
`data/pipelines/` fetches T / geopotential / specific humidity on pressure levels and reduces them to
a committed derived product. The Copernicus licence permits redistribution of derived products with
attribution, which is what is committed — never the raw ERA5 fields.

Carried into `CAVEATS.md` as a required user-facing caveat: **reanalysis stratospheric water vapour
is known to be biased dry**, so ERA5 H₂O is not the recommended source for the `h2o_mixing_ratio_ppmv`
field even though it is available. Whether to add Aura MLS as a dedicated H₂O source is deferred to
Phase 1 and tracked as SCIENCE-1's sibling.

Nothing about tropopause height exists in this repository today — grep for "tropopause" hits only
literature notes in `tomas-jax`. Phase 1 is genuinely new code.

---

## Science

### SCIENCE-1 — Climatology sampling convention · **OPEN** · blocks Phase 1 · [#53](https://github.com/reflective-org/SANDBOX/issues/53)
*Zonal mean vs. specific longitude; monthly climatology vs. daily vs. a specific reanalysis timestep.*

These give materially different tropopause heights and temperatures. Note that longitude and date are
required anyway for the solar zenith angle (`stratchem-jax/solar.py`, and `CoupledScenario.longitude`
/ `day_of_year` / `start_utc_hour`), so a longitude-resolved option should exist even if the default
is zonal-mean.

The reduced climatology's dimensions depend on the answer, so decide before the preprocessing
pipeline is written, not after.

---

### SCIENCE-2 — Definition of t = 0 · **OPEN** · blocks Phase 2, caveats **all** results · [#54](https://github.com/reflective-org/SANDBOX/issues/54)
*Does the box start at the engine exit plane, or after wake-vortex breakup?*

The most consequential unresolved item in the specification. The jet and vortex phases dilute the
plume by orders of magnitude within the first ~10–100 s, and nucleation is strongly nonlinear in
H₂SO₄ concentration, so this choice changes resulting particle number more than most parameters in
stages 4–7.

Requirements once answered:
- `t0_definition` is an explicit, **required** schema field.
- If `ENGINE_EXIT`, the early dilution regime needs a *citable* parameterisation, not the same curve
  used for the later diffusive regime. The existing regimes (`coupled/dilution.py:39`) follow a
  Schumann et al. (1998) volume expansion `V(t)/V₀ = max(1, t^0.8)` for t ≤ 10⁴ s — which is already
  a two-stage form, but is not a jet/vortex treatment.
- Appears as a top-level caveat in every results view and in `CAVEATS.md`.

**Complication found in the code:** the initial plume volume V₀ **does not enter the dynamics at
all**. It only sets the initial SO₂ concentration; the model is intensive and volume-invariant
(`coupled/tests/test_boxvol_invariance.py`; the box-size sweep `run_boxsize.py:43` works purely by
scaling the initial concentration). So the t=0 question is entirely a question about the *initial
concentration*, and the UI must not imply a geometric dependence that does not exist.

**Existing geometry is inconsistent with the spec.** The spec proposes a 10 m × 30 m cross-section;
the 810-run ensemble uses 10 m × 10 m × 15 km (`run_ensemble.py:45`) and the D1 flagship a 30 km
track (`run_dilution_d1_clean.py:61`). Phase 0 defaults follow the golden runs.

---

### SCIENCE-3 — Aerosol composition, mixing state, meteoric material · **OPEN** · blocks Phase 4 · [#55](https://github.com/reflective-org/SANDBOX/issues/55)

Required per background distribution: diameter basis (dry vs ambient, and at what water content if
ambient), composition and mixing state (internal vs external, sulfate mass fraction), and whether
meteoric material is represented at all.

**Partially constrained by the code already:** `coupled/tomas_bridge.py:92-105` defines six named
lognormal backgrounds as sulfate-only (`Mk[:, SRTSO4]`, `_seed_lognormal:131-132`), and
`AMBIENT_BACKGROUNDS` (`:107`) marks which mode sets are specified at ambient vs STP — so the
dry/ambient distinction exists but is per-dataset and implicit rather than a declared field. Meteoric
material is **not** represented.

---

### SCIENCE-4 — Box thermodynamics · **OPEN** · blocks Phase 5 · [#56](https://github.com/reflective-org/SANDBOX/issues/56)
*Is the box isobaric? isothermal? does it rise buoyantly? do particles sediment out?*

Absent from the original brief. Current behaviour, from the code:

- **Isobaric and isothermal** unless `switches.heating_to_t` is on.
- With heating on, the term is **shortwave-only — no longwave cooling** (`coupled_scenario.py:47-50`,
  AD-5.4), producing a one-sided ≈ +1.2 K / 10 d warm drift. Every science script leaves it off.
- **No buoyant rise.** No sedimentation.

Each must become a schema field with a documented default, and the SW-only asymmetry must warn in
the UI when the switch is enabled rather than silently producing a drifting temperature.

---

### SCIENCE-5 — Does the background reservoir evolve? · **OPEN in principle, ANSWERED in code** · Phase 3 · [#57](https://github.com/reflective-org/SANDBOX/issues/57)

The model is **one-box with a static background**. Specifically (`coupled/driver.py:261-276`) the
entrained background gas is the *initial plume state* with `dilution_zero_species` zeroed and
`dilution_background` pptv values applied as overrides; the background aerosol is the *initial*
`TomasState`, held immutable (`tstate_bg = tstate`, `:276`). It never evolves photochemically.

Entrainment itself is physically correct — background gases and background aerosol genuinely enter
the box at the entrainment rate (`coupled/dilution.py:81,91`), not merely a sink on plume species.

**Standing decision already recorded in this repo** (Ali, 2026-07-08, `run_ensemble.py:49-55`): all
future production runs should initialise from a **spun-up control run** rather than the static
background list, as `run_60day.py` does via `runs_60day/frank_control_ic.json`. The static list is
retained only to reproduce the existing 810-run ensemble.

The open part is whether one-box-with-spun-up-IC is sufficient, or whether a genuine two-box model is
required. Until then `background_evolves` is a schema field whose only accepted value is `false`.

---

## Register of capabilities the spec assumes but the model does not have

Not open questions — settled facts, listed here because the spec's stage descriptions imply
otherwise. Per ADR-005 each must raise `NotImplementedError` rather than quietly ignoring the field.
No issue is open for these: they are gaps to be closed when a phase needs them, and each becomes an
issue at that point rather than sitting in a backlog now.

| Spec field | Status in the model |
|---|---|
| `chemistry.rate_overrides[]` (general) | Only `so2_ho2_rate` is a knob (`coupled_scenario.py:130`). Arbitrary per-reaction overrides do not exist. |
| `chemistry.photolysis.tuvx_settings.{o3_column, albedo, aod}` | Not exposed. Only mode + lat/lon/doy/hour reach TUV-x (`model_bridge.py:46`). |
| `numerics.bin_scheme.{d_min, d_max, mass_doubling}` | Fixed by the TOMAS grid; only `tomas_nbins ∈ {40, 80, 160}` is selectable (`tomas_bridge.py:146`). Ratio = `2**(40/nbins)`; the top boundary is pinned. |
| `numerics.box_thermodynamics.*` | See SCIENCE-4. |
| `dilution.entrainment.{entrains_background_gases, entrains_background_aerosol}` | Entrainment is unconditional when `switches.dilution` is on. Separate flags are new code. |
| `dilution.background_evolves` | See SCIENCE-5. Only `false` is accepted. |
| `background.aerosol` custom lognormal modes | Six named modes + tabulated `redcircles` only — but `_seed_lognormal` (`tomas_bridge.py:110`) already accepts arbitrary `(N, Dg, σg)` tuples, so this is a small, worthwhile early addition. |
| `termination.criteria[]` (multi-quantity) | `stop_condition` exists (`driver.py:217`) but receives only `(t1_seconds, wet_SA)`. SO₂- or number-based criteria need the callback widened. |
