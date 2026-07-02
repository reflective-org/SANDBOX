# Ultra-Plan: Coupled stratospheric box — gas chemistry ⇄ TUV-x photolysis ⇄ TOMAS aerosol microphysics ⇄ dilution

## Context

We are building a self-consistent stratospheric (SAI-relevant) box model by coupling four pieces that
already exist in separate, validated codebases:

- **Gas-phase photochemistry** — `tuvx-photolysis/gas_phase_chemistry/` (frank-model): 34-species
  Cl/Br/NOx/HOx/Ox chemistry. Has a NumPy/SciPy reference (`reactions.py`, `rhs.py`, `driver.py`) and
  a **validated JAX backend `jaxmodel/`** (rates.py/chem.py/model.py, diffrax).
- **Photolysis** — `tuvx-photolysis/tuvx_photolysis/` (TUV-x JAX port): actinic-flux → J-values,
  validated ~1.8e-8 vs Fortran; O3→O(1D) J now wired (branch `o3-tuvx-photolysis`).
- **Aerosol microphysics** — `tomas-jax/` (branch `feat/marianna-dilution`): sectional TOMAS in JAX
  (40 bins × 44 species), `make_step([...])`, condensation/nucleation/coagulation, its own SO2 chem,
  dilution, and Mie optics + radiative forcing.

Today these do not talk to each other: photolysis is aerosol-free, the chemistry's aerosol surface
area is a prescribed constant, there is no gas-phase H2SO4 production feeding the aerosol, and there is
no dilution. The goal is a **single-input, fully coupled** box with **on/off switches** for each
process (photolysis mode, sulfur→H2SO4, nucleation, condensation, coagulation, dilution, aerosol→J
radiative feedback, aerosol→temperature heating), running on **consistent JAX solvers** with operator
splitting, so we can study sulfate evolution, its radiative effect, and dilution together.

## Confirmed decisions

1. **Sulfur ownership:** frank-model owns the gas phase — it computes OH and does SO2(+OH)→[SO3]→H2SO4
   **gas**; it hands gas H2SO4 to TOMAS, whose **internal SO2 chemistry is disabled**; TOMAS only
   nucleates/condenses/coagulates. Single gas-phase master (no double counting).
2. **Solvers:** consistency required → the coupled run uses the **JAX gas backend (`jaxmodel`)**, not
   SciPy BDF. **Operator splitting** at an outer coupling step between chemistry / microphysics /
   dilution / radiation update; each sub-process uses a JAX/diffrax solver.
3. **Radiation:** two-way, **including aerosol→temperature** (radiative heating), behind a switch.
   Aerosol→photolysis (optics→TUV-x→J) is the always-on part of two-way radiation.
4. **Dilution:** reuse the TOMAS formulation and the V(t)/k_dil environmental factor from
   `tomas-jax/experimental_case/run_marianna_dilution.py`; apply the shared k_dil(t) + background
   entrainment to **all** gas species (frank's 34 + SO3/H2SO4) **and** the aerosol.
5. **Single input file** for the whole system, with process on/off switches.
6. Sub-decisions already set: SO2+HO2 keeps `1e-18`; new sulfur species appended at the end;
   reference-mode chemistry stays MATLAB-faithful (gate new chemistry to non-reference modes).
7. **Tunable microphysics scaling knobs** (for the sensitivity study, Phase 8) are exposed in the
   single input as free multipliers defaulting to 1.0 (any value allowed, not just the sweep points):
   `coag_kernel_scale`, `nucleation_rate_scale`, and a condensation sticking knob (tentatively the
   mass-accommodation coefficient `alpha` — **to discuss/confirm**).

## Target architecture (the coupling loop, one outer step Δt_couple)

```
        ┌─────────────────────────────────────────────────────────────────┐
        │ SINGLE INPUT (YAML): environment, location/date, schedule,        │
        │ initial gas composition, initial+background aerosol, dilution      │
        │ regime, and SWITCHES (each process on/off)                         │
        └─────────────────────────────────────────────────────────────────┘
   state = { gas C[molec/cm3] (34 + SO3 + H2SO4), TOMAS (Nk,Mk,Gc), T }
   repeat each Δt_couple:
     1. RADIATION: aerosol size-dist → Mie (OD/SSA/g per λ) → TUV-x aerosol radiator
                   → J-values (incl. O3→O1D→OH). [switch: aerosol→J]
     2. GAS CHEM (jaxmodel/diffrax): advance 34 gas species + SO3 + H2SO4 using J and
                   aerosol SA/radius (heterogeneous rates). Produces gas H2SO4. [switch: sulfur]
     3. HANDOFF: gas H2SO4 (molec/cm3) → TOMAS Gc[SRTSO4] (kg/cell) via boxvol.
     4. MICROPHYSICS (TOMAS make_step, SO2-chem OFF): nucleation+condensation+coagulation
                   consume gas H2SO4. [switches: nucleation/condensation/coagulation]
        → return depleted H2SO4 to the gas master; derive SA, wet radius, composition.
     5. HEATING (switch): aerosol/gas absorption → heating rate → update box T.
     6. DILUTION (switch): shared k_dil(t) + background entrainment on ALL gas + aerosol.
```

Units bridge (one small module): gas `molec/cm3` ⇄ TOMAS `Gc` `kg/gridcell` via
`conc = Gc/boxvol * NA/(MW/1000)` (both directions), reusing `tomas-jax` `molec_cm3_to_kg_gridcell`.

## Key integration seams (from exploration, with references)

- **frank aerosol seam:** `gas_phase_chemistry/reactions.py:290-299` (`build_env`) — SA (`khet`,
  `mechanism.py:106-111`) and `hetgammas_jpl00` radius (hard-coded `0.1e-4`) + composition
  (`h2so4wp_at`). Replace prescribed SA/radius with TOMAS-derived values.
- **TUV-x aerosol seam:** add per-solve aerosol OD/SSA/g into `tuvx_photolysis/api.py` `_solve`
  (like the LA/SR O2 in-place mod), building `radiators.aerosol_radiator` (`radiators.py:97-115`);
  `from_tuvx_json:212-226` currently skips aerosol. Adapter uses the no-aerosol config
  (`tuvx_photolysis_adapter.py:70`).
- **TOMAS gas seam:** `make_step([...])` (`tomas_jax/solvers/condensation.py:753`) with
  `so2_chemistry` **omitted**; feed `Gc[SRTSO4]`; read back depleted `Gc` + `Nk/Mk`.
- **TOMAS optics:** `precompute_mie_properties(xk)` + `compute_optical_depth(Nk,mie,area)`
  (`tomas_jax/physics/radiative_forcing.py:231,643`); SSA = Qsca/Qext (derive), g = gsca.
- **Dilution:** `dilution_step(Nk,Mk,Gc,dt,kdil,*_bg)` (`tomas_jax/physics/dilution.py:19`);
  `build_kdil_array`/`V_ratio`/background (`experimental_case/run_marianna_dilution.py:262-345`).

## What must be ported from TUV-x (radiation / heating / feedback)

- **Radiation (photolysis with aerosol): NO new Fortran port.** Solver, radiators, spherical geometry,
  cross sections, quantum yields are already ported/validated. Aerosol-aware J is only *wiring*: feed
  TOMAS Mie optics (OD/SSA/g) into the existing generic `radiators.aerosol_radiator` +
  `api.py` `from_tuvx_json`/`_solve`. Exact `aerosol.F90` (fractional-source interp + Ångström) is
  **not needed** — TOMAS gives spectral optics directly (Ångström only stretches a single-λ OD).
- **Heating: one small port — `src/heating_rates.F90`.** Its core is
  `heating_rate(z,rxn) = dot(actinic_flux(:,z), σ·φ·energy(:)) · scaling` (heating_rates.F90:271-303) —
  structurally identical to the already-ported photolysis-rate dot product, just weighted by a
  per-reaction wavelength-resolved bond-dissociation `energy term` [J] + a scaling factor. Reuses the
  existing `actinic_flux` and σ·φ; the only new data is the per-reaction energy term (+ O2 special
  case). Gives **gas-phase photochemical heating** (O3/O2 UV), which already captures the aerosol's
  *indirect* effect (aerosol changes the flux).
- **NOT in TUV-x (needs a decision, not a port): aerosol *direct* radiative heating.** Sulfate SW
  absorption (Mie Qabs = Qext−Qsca) and especially **longwave** heating (the dominant stratospheric
  aerosol heating term) are outside any shortwave actinic-flux code — both TUV-x and TOMAS's RF are SW.
  Full "aerosol→temperature" likely needs a small dedicated heating term (Mie Qabs × flux and/or a LW
  parameterization) separate from TUV-x. Scope/fidelity to settle at Phase 5.
- **Feedback wiring (aerosol→J, heating→T) is coupling code, not a Fortran port.**

## Documentation (a living `docs/` folder of md files)

The coupled project keeps a **`docs/` folder** of focused, continuously-updated markdown files so
state is easy to track for both humans and agents. Each is append/maintained every phase (dated
entries), not written once. Planned files:
- `ARCHITECTURE.md` — the coupling design + data-flow diagram (the "target architecture" here, kept
  current) and the units bridge.
- `DECISIONS.md` — decision log (lightweight ADRs): each decision, date, rationale, alternatives
  (e.g. "frank owns gas sulfur", "JAX solver consistency", "aerosol→T on").
- `ASSUMPTIONS.md` — every modeling assumption made to get things running, with justification (so no
  silent simplifications) — e.g. SO3+H2O rate value, column placement, prescribed vs evolving.
- `PROGRESS.md` — phase/task status, what's done, what's next (mirrors the task list in prose).
- `DEFERRED.md` — deferred work / follow-ups (e.g. exact `aerosol.F90` port, LW heating fidelity).
- `CAVEATS.md` — known limitations / where results should not be over-trusted.
- `NICE_TO_HAVE.md` — wishlist / future enhancements.
- `VALIDATION.md` — what was checked against what (JPL 19-5 sulfur cross-check, Fortran parity,
  conservation tests) with results.
Phase-level sub-plans also live here (`docs/phase-N-plan.md`). Keeping these current is part of each
phase gate.

## Execution model (how we run this)

This document is the **master plan**. Each phase is substantial, so we do **not** implement it straight
from here. For every phase, when we reach it:
1. **Per-phase planning pass** — a focused (re-enter plan mode) design for that phase: explore the
   exact seams/files, resolve that phase's open decisions with you, and write a phase-level plan.
2. **Task breakdown** — turn the phase plan into concrete tasks/subtasks (TaskCreate), each a small,
   testable unit with a clear deliverable.
3. **Incremental implementation** — one subtask at a time, each with a passing test + a commit
   (the established "commit small stages, verify with plots/tests" workflow), on a per-phase branch.
4. **Phase gate** — validation (tests + consistency plots) + updating the `docs/` living files
   (PROGRESS, DECISIONS, ASSUMPTIONS, VALIDATION, etc.) + the 3-agent independent verification
   (below) before moving on. Phases are sequenced but their gates let us stop/review between them.

So: master plan (this file) → per-phase sub-plan → per-phase task list → incremental commits →
3-agent independent verification.

### Independent verification (3-agent gate)

At the end of each phase (and for any substantial task) we deploy **3 independent verification agents
in parallel**, each re-deriving whether the work is actually accomplished — they do NOT trust the
implementer's claim and each works from the spec/reference, not from my summary. To catch different
failure modes, each agent gets a **distinct lens**:
1. **Correctness vs reference** — does the code match the authoritative source (JPL 19-5 rates/product
   channels, the TUV-x Fortran, the TOMAS equations)?
2. **Tests & coverage** — do the tests actually pass, and do they genuinely exercise the new behavior
   (not vacuous / not over-fitted)? Re-run them.
3. **Physical & conservation sanity** — mass/number/sulfur conservation, units, limiting cases, and
   **no silent simplifying assumptions** (any assumption must be in `docs/ASSUMPTIONS.md`).
Each returns a structured verdict (pass/fail + evidence). The gate passes only on agreement
(target: unanimous; at minimum ≥2/3), and any dissent becomes a new task before proceeding. This can
be orchestrated with parallel subagents (or the Workflow tool when opted in). Verdicts are logged in
`docs/VALIDATION.md`.

## Phases (each ends with tests + a commit; incremental)

**Phase 1 — Gas-phase sulfur→H2SO4 in frank (NumPy + jaxmodel).**
Add `SO3`,`H2SO4` species (append 35,36; `config.py`). Add gated reactions
`SO2+OH→SO3+HO2`, `SO2+HO2→SO3+OH`, `SO3+H2O→H2SO4` (rates: existing `_k68`; `1e-18`;
Lovejoy/JPL water-catalyzed `k·[H2O]`), gate via `Env.sulfur_chain` (`photolysis!="reference"`);
keep the two legacy SO2 reactions reference-only. Mirror in `jaxmodel/rates.py`/`chem.py`
(parity), update `tests/test_config.py`, `rhs.csv` fixture (+2 zero rows), `test_rhs.py` shape.
New tests: sulfur mass conservation, gate off in reference (SO3=H2SO4=0), gate on in tuvx. Species
1–34 byte-identical in reference mode. *(This is the Stage A–D work already scoped.)*

**Phase 1 also includes a JPL cross-check of the whole H2SO4 production path** against the NASA-JPL
Evaluation 19-5 handbook (`frank-model/references/NASA-JPL_Evaluation_19-5.pdf`): verify each
reaction's **rate constant AND product channel** — `SO2+OH(+M)→HOSO2` (Table 2-1 I4, termolecular),
the `HOSO2+O2→SO3+HO2` step lumped into our `SO2+OH→SO3+HO2`, `SO2+HO2` (upper limit `1e-18`), and the
`SO3+H2O(+H2O)→H2SO4` water-catalyzed rate (confirm the Lovejoy/JPL value vs the handbook). Document
the exact JPL reference (table/section) next to each rate in `reactions.py`, and flag any reaction
where JPL gives no recommendation (e.g. SO2+HO2) so the choice is explicit. This check gates Phase 1
completion.

**Phase 2 — JAX operator-split coupling skeleton + single config.**
New coupling driver that runs the **jaxmodel** gas chemistry (diffrax) and steps the outer loop;
introduce the unified config dataclass (`CoupledScenario`) + one YAML, with all process switches and
the units bridge. No TOMAS yet — verify the JAX gas run reproduces the current tuvx results (parity
vs NumPy), so we have a trusted JAX baseline before adding aerosol.

**Phase 3 — Couple TOMAS microphysics (gas H2SO4 ⇄ aerosol; SA→chemistry).**
Add TOMAS as a dependency; init `TomasState` from the config's initial+background distribution
(`background_aerosol_distribution.py`). Each step: handoff gas H2SO4 → `make_step` (SO2-chem off) →
depleted H2SO4 back; compute SA + wet radius from `Nk/Mk`/`xk` and feed frank's `build_env`
(replace constant SA and hard-coded radius; keep the `SA = background + sulfate` split). Tests:
sulfur budget closes across gas+aerosol; SA responds to condensed mass.

**Phase 4 — Two-way radiation: aerosol → photolysis.**
Port the aerosol radiator into the TUV-x port (`from_tuvx_json` aerosol branch + per-solve inject in
`_solve`); optionally the exact `aerosol.F90` fractional-source/Ångström handling (follow-up #11).
Each step, convert TOMAS Mie (OD/SSA/g, spectral) to the port's wavelength grid and box layer; J now
responds to aerosol. Switch: aerosol→J. Validate J decreases with aerosol OD; conserve when OD=0.

**Phase 5 — Radiative heating → temperature (switchable).**
Two components: (a) **gas-phase photochemical heating** — port the small `heating_rates.F90` (reuses
the ported actinic flux + σ·φ, adds the per-reaction energy term); (b) **aerosol direct heating** —
NOT in TUV-x; decision at phase start on fidelity: Mie Qabs×flux (SW) only, or add a LW heating
parameterization (LW dominates stratospheric aerosol heating). Sum → dT/dt on box `T`. Switch:
heating→T (off ⇒ T constant, tested). Energy-bookkeeping test.

**Phase 6 — Dilution (shared k_dil + background entrainment).**
Port `V_ratio`/`build_kdil_array` + background state from `run_marianna_dilution.py` into the shared
config; apply `dilution_step`-style relaxation to **all** gas species and the aerosol each outer step.
*Open:* confirm the entrainment math (several valid ways to mix background air) — settle before coding.
Tests: passive-tracer decay matches analytic; regimes D1–D5 reproduce `run_marianna_dilution.py`.

**Phase 7 — Single unified input, end-to-end validation, plots, docs.**
One YAML drives everything; switches reproduce every sub-case (chemistry-only, +aerosol, +radiation,
+heating, +dilution). The three scaling knobs (item 7) live here as free multipliers. Per-species +
aerosol (size dist, SA, H2SO4, RF, T) plots into a configurable folder. Document the whole coupling in
`DEVELOPMENT.md`.

**Phase 8 — Sensitivity experiments (after the coupled model is validated).**
Drive sweeps from the single input by varying one scaling knob at a time (a small runner produces one
output folder per case + overlay/summary plots of the key responses: size distribution, SA, H2SO4,
particle number, RF, T). Each knob accepts **any** value; the standard sweeps are:
- **(1) Coagulation kernel** — `coag_kernel_scale ∈ {0.5, 1, 2}`. Multiplies the coagulation kernel
  `kij` (seam: `tomas_jax` coagulation-kernel calc / `calc_coagulation_rates`). Thread a scale factor
  through the coupling; expose in the input.
- **(2) Nucleation rate** — `nucleation_rate_scale ∈ {0.01, 0.1, 1, 10, 100}`. TOMAS's
  `nucleation_step(..., fn_scale=...)` already provides this multiplier — wire it to the input.
- **(3) Condensation sticking** — vary the accommodation coefficient `alpha` (TomasState.alpha /
  `calc_condensation_sink(..., alpha=...)`). **Needs discussion** to confirm `alpha` is the intended
  variable and the sweep range/values.
Deliverable: a reproducible sensitivity runner + committed comparison plots and a short write-up of
how each knob moves the sulfate size distribution / surface area / radiative effect.

## Open decisions to resolve at each phase (not blocking the plan)
- **Repo layout:** new coupling package inside `tuvx-photolysis/` importing `tomas-jax`, vs a new
  top-level repo. (Lean: a `coupled/` package in `tuvx-photolysis` depending on `tomas-jax`.)
- **jaxmodel gap:** confirm `jaxmodel` reproduces the NumPy tuvx run before Phase 3 (Phase 2 gate).
- **Heating method** (Phase 5, above).
- **Dilution entrainment formulation** (Phase 6, above) — you will confirm from
  `run_marianna_dilution.py`.
- **Aerosol column placement:** the box is one altitude; decide single-layer OD vs a scaled profile
  for the TUV-x column.
- **Δt_couple** and operator-split ordering/stability (H2SO4 transient needs small dt in TOMAS).

## Verification (end-to-end)
- Per-phase pytest (chemistry parity NumPy↔JAX; sulfur conservation; dilution analytic; aerosol→J
  monotonicity; T-switch off ⇒ T constant).
- Single-config runs with switches toggling each process; confirm reductions to known sub-cases.
- Conservation diagnostics: total sulfur (gas SO2+H2SO4 + particulate SO4) under condensation;
  dilution tracer; radiative energy.
- Consistency plots committed as artifacts (species trends, size distribution, SA(t), H2SO4(t),
  RF(t), T(t)); compare switch-on vs switch-off.
```
