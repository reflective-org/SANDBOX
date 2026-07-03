# Caveats / known limitations (SANDBOX)

- **Single box, single altitude.** Photolysis needs a column for radiative transfer; the box maps to
  one altitude/layer within it. Overhead atmosphere (for the RT column) is currently fixed.
- **Coupling status.** Gas chemistry + photolysis + TOMAS microphysics (Phase 3) + aerosol→photolysis
  (Phase 4) + radiative heating→T (Phase 5) + dilution (Phase 6) are all built and switchable; Phase 7
  (single unified input + end-to-end validation) and Phase 8 (sensitivity sweeps) remain.
- **Branching quantum yields (HNO4, ClOOCl)** are opt-in extensions, NOT Fortran-comparable, and the
  ClOOCl→2ClO channel changes ClOOCl chemistry vs the reference MATLAB. Confirm splits vs JPL 19-5.
- **Heterogeneous chemistry** uses a prescribed constant surface area and a hard-coded 0.1 µm radius (0.1e-4 cm)
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
- **Dilution (Phase 6):** constant first-order relaxation to the INITIAL box state at
  `dilution_rate` [1/s]; the V(t)/volume-expansion time-varying rate is deferred. Box TEMPERATURE is
  NOT diluted (heating owns T); entrainment of background-temperature air is not modeled (AD-6.x,
  DEFERRED). Background = initial state (a plume-into-ambient assumption).
- **Heating→T is SW-only, NO radiative cooling (Phase 5, AD-5.4).** The box gains heat from O3
  photochemical + aerosol shortwave absorption but has NO longwave cooling term, so T rises
  monotonically in daylight and never relaxes — this is NOT a closed energy balance. Longwave aerosol
  heating (the DOMINANT stratospheric-sulfate term) is also not modeled. Both OPEN for the user. Use
  `heating_to_t` for sensitivity/relative studies, not absolute equilibrium temperatures.
- **O2 photochemical heating deferred (Phase 5, AD-5.1).** Only O3 heating is ported (dominant at
  ~19 km); O2 Schumann-Runge/Lyman-α heating (needs the LA/SR-corrected σ) is not included.
- **γ water activity a_W stays thermodynamic (AD-3.3).** Composition (H2SO4 wt%) comes from TOMAS but
  a_W is still the gas-phase `pH2O/p0` value; a fully self-consistent aerosol-water reconciliation is
  deferred.
- **Aerosol→photolysis optics use a fixed refractive index (AD-4.1).** No wavelength-dependent n(λ)
  for sulfate in the UV; the Mie table uses a constant 1.4+1e-8j and the fixed dry geometric-mean bin
  radii (matching TOMAS's own RF), so wet-growth optical effects are not captured. DEFERRED fidelity.
- **Aerosol vertical placement = pressure-anchored plume (AD-4.2, resolved).** The box aerosol fills a
  plume of thickness `aerosol_thickness_km` (default 1 km) centered on the box altitude (derived from
  P), so it tracks the pressure. The thickness — not an absolute km window — sets the feedback
  magnitude, so choose it as the plume's real vertical extent. `aerosol_band_km` overrides with an
  absolute band; `aerosol_to_j=False` turns the feedback off. NOTE: the box's own aerosol is a thin
  perturbation — it does NOT represent the full overhead background (Junge) layer, so the radiative
  self-feedback is modest by design; don't read the box-altitude J change as the total aerosol effect.
- **Aerosol→J is non-monotonic in loading.** A scattering sulfate aerosol ENHANCES J at low-moderate
  optical depth (added diffuse actinic flux) and only SHIELDS at high loading — physically correct, but
  do not expect "more aerosol ⇒ less J" at all loadings (see coupled/validation/phase4_j_vs_od.png).
- **`switches.sulfur=False` is not "SO2 inert".** The gate toggles between *two chemistries*, not on/off
  of one. With the sulfur chain OFF, the model falls back to the reference-only legacy MATLAB SO2 lumps
  (`SO2+OH→HO2`, `SO2+HO2→`), which still destroy SO2 (~0.45 %/day) but produce no SO3/H2SO4. So
  **total sulfur is conserved only with the gate ON**; with it OFF, SO2 decays into unbudgeted products
  by design (MATLAB faithfulness). Verified at the Phase 2 gate (lens 3). See `docs/VALIDATION.md`.
