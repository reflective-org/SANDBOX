# Caveats / known limitations (SANDBOX)

- **Single box, single altitude.** Photolysis needs a column for radiative transfer; the box maps to
  one altitude/layer within it. Overhead atmosphere (for the RT column) is currently fixed.
- **Coupling not yet built.** Aerosol microphysics (TOMAS), aerosol→radiation, heating→T, and dilution
  are planned (Phases 3–6); today the repo is photolysis + gas chemistry only.
- **Branching quantum yields (HNO4, ClOOCl)** are opt-in extensions, NOT Fortran-comparable, and the
  ClOOCl→2ClO channel changes ClOOCl chemistry vs the reference MATLAB. Confirm splits vs JPL 19-5.
- **Heterogeneous chemistry** uses a prescribed constant surface area and a hard-coded 1 µm radius
  *only when TOMAS microphysics is off*. With Phase 3 coupling on, the surface area, effective radius,
  and H2SO4 weight-percent come from the evolving TOMAS aerosol.
- **TOMAS internal sulfur non-conservation (~1%/day).** With nucleation active, the tomas-jax
  `feat/marianna-dilution` microphysics loses ~1.3% of sulfur over ~40 steps in its OWN gas+particulate
  budget (MNFIX mass-number fixing, PPM condensation redistribution, nucleation cluster accounting).
  The SANDBOX coupling handoff is provably exact (a conserving-stub test conserves total S to ~1e-12),
  so any total-S drift in a coupled run is inherited from TOMAS, not the coupling. See AD-3.10 and the
  OPEN item in `AUTONOMOUS_DECISIONS.md` (flagged for user review).
- **dt_couple stability with TOMAS (AD-3.9).** The operator split produces all of an interval's H2SO4
  before TOMAS consumes it, so a large `dt_couple` overestimates the peak gas H2SO4; above ~1e9
  molec/cm³ TOMAS's nucleation rate overflows to NaN. `dt_couple` must be small enough (≲ a minute for
  high-SO2 cases) to keep the per-step slug in range; a loud guard raises if TOMAS returns non-finite.
- **Heterogeneous radius = effective radius (AD-3.4).** A single scalar radius (the surface-area-
  weighted r_eff) represents the whole distribution in `hetgammas_jpl00`; the per-bin size dependence
  of uptake is not resolved. Under heavy nucleation r_eff is pulled toward the fine mode.
- **γ water activity a_W stays thermodynamic (AD-3.3).** Composition (H2SO4 wt%) comes from TOMAS but
  a_W is still the gas-phase `pH2O/p0` value; a fully self-consistent aerosol-water reconciliation is
  deferred.
- **`switches.sulfur=False` is not "SO2 inert".** The gate toggles between *two chemistries*, not on/off
  of one. With the sulfur chain OFF, the model falls back to the reference-only legacy MATLAB SO2 lumps
  (`SO2+OH→HO2`, `SO2+HO2→`), which still destroy SO2 (~0.45 %/day) but produce no SO3/H2SO4. So
  **total sulfur is conserved only with the gate ON**; with it OFF, SO2 decays into unbudgeted products
  by design (MATLAB faithfulness). Verified at the Phase 2 gate (lens 3). See `docs/VALIDATION.md`.
