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

### SCIENCE-1 — Climatology sampling convention · **ANSWERED** (2026-08-18) · [#53](https://github.com/reflective-org/SANDBOX/issues/53)
*Zonal mean vs. specific longitude; monthly climatology vs. daily vs. a specific reanalysis timestep.*

**Answered (Ali, 2026-08-18): monthly climatology**, selected by the month of the release date. Not a
daily climatology and not a specific reanalysis timestep — so the product carries a **month** axis of
12, and the run's date picks the month rather than a date in a particular year.

Consequences to hold onto:

- **The year is meaningless to the meteorology.** A monthly climatology is an average over years, so
  a config must not imply it is using the weather of a particular year. Either the date field stays
  year-free, or a year is accepted and the schema states plainly that it affects nothing but the
  calendar arithmetic.
- **Month must be derived, not entered twice.** The solar zenith angle needs the day of year
  (`solar.py`), and the climatology needs the month; entering both invites a config whose month and
  day disagree. One is primary and the other is derived — which is the dependency engine's job.
- **The day↔month mapping needs a stated convention**, because it depends on leap years. Day 172 is
  21 June in a non-leap year and 20 June in a leap year. Fixing a non-leap mapping keeps the
  ensemble's `day_of_year = 172` reading as "late June" without introducing a year.

**Answered (Ali, 2026-08-18): zonal-mean**, with a longitude-resolved product possible later. So the
reduced climatology is **(lat × month × level)** — of order 86k values per field, a few MB, small
enough to commit with a checksum rather than fetched from anywhere at run time. That is also how the
paper ensemble reasons: a latitude band, not a place.

Longitude is still a required input, because the solar zenith angle needs it (`solar.py`); it simply
does not select the meteorology. The convention is recorded in the dataset identifier that goes into
provenance (ADR-006), so adding a longitude-resolved product later cannot silently reinterpret a
config made against this one.

**Date entry follows from this** (decided in the same conversation): a calendar date **without a
year**, entered as month + day of month. `day_of_year` becomes DERIVED — the solar declination needs
it and the climatology needs the month, and entering both would allow a config whose month and day
disagree. A monthly climatology is an average over years, so accepting a year would imply we used a
particular year's weather, which we do not. The day↔month mapping uses a fixed **non-leap** calendar,
which keeps the ensemble's `day_of_year = 172` as 21 June.

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

**Partially constrained by the code already:** `coupled/backgrounds.py` defines six named
lognormal backgrounds, seeded as sulfate-only (`Mk[:, SRTSO4]` in `tomas_bridge._seed_lognormal`),
and `AMBIENT_BACKGROUNDS` marks which mode sets are specified at ambient vs STP — so the
dry/ambient distinction exists but is per-dataset and implicit rather than a declared field. Meteoric
material is **not** represented.

---

### SCIENCE-4 — Box thermodynamics · **ANSWERED for heating and buoyancy** (2026-08-13); sedimentation open · [#56](https://github.com/reflective-org/SANDBOX/issues/56)
*Is the box isobaric? isothermal? does it rise buoyantly? do particles sediment out?*

Absent from the original brief. Current behaviour, from the code:

- **Isobaric and isothermal** unless `switches.heating_to_t` is on.
- With heating on, the term is **shortwave-only — no longwave cooling** (`coupled_scenario.py:47-50`,
  AD-5.4), producing a one-sided ≈ +1.2 K / 10 d warm drift. Every science script leaves it off.
- **No buoyant rise.** No sedimentation.

**Answered (Ali, 2026-08-13): heating and buoyancy are out of scope for this model.**

Not "undecided" — **out of scope**, which is a different status and is why this row is closed rather
than left open. Longwave radiation is not in the radiative calculation, so the heating term cannot
represent the box's energy balance: enabling it does not make the thermodynamics more complete, it
makes them one-sided, and the ~+1.2 K / 10 d drift is an artefact of the missing cooling rather than
a result. Buoyant rise follows the same logic — a parcel rises in response to a heating rate this
model cannot compute, so a rise velocity here would be a free parameter dressed as physics.

**Answering either question needs a different model**, one with longwave radiation and plume
dynamics. It is not a gap to be filled in by a later Studio phase, and Studio must not present a
knob implying otherwise:

- `switches.heating_to_t` is `Literal[False]` from schema 0.2.0 — `True` fails validation rather
  than being defaulted off, so it cannot be enabled by a form, a YAML file or a sweep axis.
- **No buoyancy or heating-rate fields are added to the schema at all.** A field for a capability
  the model does not have would advertise it; the absence is the honest interface (ADR-005).
- Every run is therefore **isobaric and isothermal at the configured temperature**, and results
  carry that as a top-level caveat rather than a footnote.

**Still open: sedimentation.** It is untouched by this decision — a particle-loss process, not a
thermodynamic response — and the model does not have it. It is deliberately left in this register
rather than swept in with the rest, because "we decided not to model heating" is not an argument
about gravitational settling.

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

### SCIENCE-6 — GCR ion-pair production rate has no derivation · **OPEN** · Phase 0/4 · [#63](https://github.com/reflective-org/SANDBOX/issues/63)
*What is the ion-pair production rate as a function of altitude, latitude and solar-cycle phase?*

Raised by task 0.5. The two values available in the repository are an **uncited constant** and a
value that **switches off a physical process**: the paper ensemble uses a bare `30.0` cm⁻³ s⁻¹
(`run_ensemble.py:102`, described in `TABLE_microphysics_parameters.md` as "galactic cosmic rays at
~20 km"), and the model defaults to `0.0`, which disables ion-induced nucleation entirely
(`coupled/coupled_scenario.py:117`).

It feeds the ion-induced channels of Dunne et al. (2016) nucleation — the most sensitive part of this
system. GCR ionisation varies by roughly a factor of two over the solar cycle and strongly with
latitude and altitude, so one number is wrong nearly everywhere except where it was read off.

`studio/science/gcr.py` therefore raises `NotImplementedError` rather than interpolating an uncited
number, and exposes `PAPER_ENSEMBLE_ION_PAIR_RATE = 30.0` as a constant with its provenance attached.

**Answered when** either a citable parameterisation is agreed and implemented with its reference, or
the decision is recorded that the fixed value stands, with its sensitivity quantified.

---

### SCIENCE-7 — Physical upper bounds: `temperature_k = 9999` validates · **OPEN** · Phase 1 · [#91](https://github.com/reflective-org/SANDBOX/issues/91)

Found by driving the wizard in a real browser and typing 9999 into the temperature box, expecting a
refusal. There was none: the schema declares only `gt=0`.

**17 of 22 numeric fields have a lower bound and no upper bound.** The 5 that are bounded on both
sides are bounded *definitionally* — latitude ±90, day-of-year <366, hour <24, an accommodation
coefficient in [0, 1] — so the line was never drawn deliberately; bounds appeared where the number's
own definition supplied one and were omitted where an upper limit would be a judgement about
plausible physics.

**Not resolved by picking numbers.** "Temperature ≤ 300 K" is a convention, and inventing it is the
fabricated-physics failure mode. The candidate with a real source is to bound temperature and
pressure by the validity range of the TUV-x tables and JPL rate fits — which needs someone who knows
those ranges to state them — optionally with soft, non-blocking warnings elsewhere so a deliberate
sensitivity test can still leave the envelope.

`test_which_numeric_fields_have_no_upper_bound` pins the current list, so a decision arrives as a
visible change to it rather than as a silent tightening.

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
| `numerics.box_thermodynamics.*` | **Not exposed, by decision.** Heating and buoyancy are out of scope (SCIENCE-4): the model cannot compute them and a field would imply it can. Isobaric + isothermal is the only behaviour. |
| `dilution.entrainment.{entrains_background_gases, entrains_background_aerosol}` | Entrainment is unconditional when `switches.dilution` is on. Separate flags are new code. |
| `dilution.background_evolves` | See SCIENCE-5. Only `false` is accepted. |
| `background.aerosol` custom lognormal modes | Six named modes + tabulated `redcircles` only — but `_seed_lognormal` (`tomas_bridge.py:110`) already accepts arbitrary `(N, Dg, σg)` tuples, so this is a small, worthwhile early addition. |
| `termination.criteria[]` (multi-quantity) | `stop_condition` exists (`driver.py:217`) but receives only `(t1_seconds, wet_SA)`. SO₂- or number-based criteria need the callback widened. |
