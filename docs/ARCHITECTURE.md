# Architecture (SANDBOX)

Coupled stratospheric box. State = `{ gas C [molec/cm³] (34 + SO3 + H2SO4), TOMAS (Nk, Mk, Gc), T }`.
Operator-split loop, one outer step Δt_couple (all sub-processes on consistent JAX/diffrax solvers):

```
each Δt_couple:
  1. RADIATION: aerosol size-dist → Mie (OD/SSA/g per λ) → TUV-x aerosol radiator → J
                (incl. O3→O1D→OH).                              [switch: aerosol→J]
  2. GAS CHEM (jaxmodel/diffrax): advance gas species using J + aerosol SA/radius
                (heterogeneous rates); produce gas H2SO4.        [switch: sulfur]
  3. HANDOFF: gas H2SO4 (molec/cm³) → TOMAS Gc[SRTSO4] (kg/cell) via boxvol.
  4. MICROPHYSICS (TOMAS make_step, SO2-chem OFF): nucleation+condensation+coagulation
                consume gas H2SO4 → return depleted H2SO4; derive SA, wet radius, composition.
  5. HEATING: gas photochemical + aerosol direct → dT/dt on box T.  [switch: heating→T]
  6. DILUTION: shared k_dil(t) + background entrainment on ALL gas + aerosol. [switch: dilution]
```

## Component ownership
- **frank-model (gas master)** — owns all gas species incl. SO2→[SO3]→H2SO4 **gas**; computes OH.
- **TUV-x port** — photolysis J (now O3→O1D); aerosol radiator fed by TOMAS optics.
- **TOMAS** — sectional aerosol; consumes gas H2SO4; **its own SO2 chem is OFF**. Produces size dist
  → surface area (background + sulfate), wet radius, Mie optics.
- **Dilution** — reused from `tomas-jax` (`dilution_step`, V(t)→k_dil, background entrainment).

## Units bridge
gas `molec/cm³` ⇄ TOMAS `Gc` `kg/gridcell`: `conc = Gc/boxvol · N_A/(MW/1000)` (both directions;
reuse `tomas-jax` `molec_cm3_to_kg_gridcell`).

## Key integration seams (files)
- frank aerosol: `gas_phase_chemistry/reactions.py` `build_env` (SA via `khet`; radius+composition via
  `hetgammas_jpl00`/`h2so4wp_at`).
- TUV-x aerosol: `tuvx_photolysis/api.py` `_solve` + `radiators.aerosol_radiator`.
- TOMAS gas: `tomas_jax` `make_step([...])` (omit `so2_chemistry`); optics via
  `radiative_forcing.precompute_mie_properties` / `compute_optical_depth`.

## Implemented so far (Phases 1-3)
Steps 2-4 of the loop are live in `coupled/driver.py` (`run_coupled`); the NumPy mirror
(`coupled/reference_numpy.py`) reproduces it for cross-backend parity. Per outer interval:
- **Gas chem** integrates on the JAX backend with frozen midpoint J and the previous interval's
  aerosol het inputs (SA / effective radius / H2SO4 wt%).
- **Handoff**: `coupled/units.py` converts gas H2SO4 → `Gc[SRTSO4]`; `coupled/tomas_bridge.py` builds
  the initial `TomasState` (Marianna dist) and the SO2-off `make_step`.
- **Microphysics**: TOMAS advances the interval; `coupled/aerosol_props.py` derives SA (µm²/cm³),
  effective wet radius r_eff = ΣN r³/ΣN r² (cm), and H2SO4 wt% for the next interval; depleted H2SO4
  returns to the gas. A loud guard raises if TOMAS returns non-finite (dt_couple stability, AD-3.9).

**Step 1 (aerosol→J) is live (Phase 4):** `coupled/aerosol_optics.py` builds spectral OD/SSA/g from the
TomasState (per-wavelength Mie on fixed bin radii) over the `aerosol_band_km` slab; the driver injects
it into the TUV-x port `_solve` via the adapter behind `switches.aerosol_to_j`, so J responds to the
aerosol (scattering enhancement then shielding).

Steps 5 (heating), 6 (dilution) are Phases 5-6. TOMAS is active iff any of
`switches.{nucleation,condensation,coagulation}` is on; else the Phase-2 gas-only path (prescribed
`cfg.SA`) runs. Phase-3 design calls: `DECISIONS.md` + `AUTONOMOUS_DECISIONS.md` (AD-3.x).

See `docs/master-plan.md` for the full plan and phase breakdown.
