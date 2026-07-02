# Progress (SANDBOX)

Phases from `docs/master-plan.md`. Status updated each phase gate.

## Done (pre-SANDBOX consolidation)
- TUV-x photolysis JAX port, validated ~1.8e-8 vs Fortran (incl. LA/SR bands).
- Gas-phase chemistry box model (NumPy ref + JAX `jaxmodel`), coupled via `tuvx_photolysis_adapter`.
- O3 photolysis driven by TUV-x O(1D) channel (Matsumi-2002 QY); comparison runs + plots.
- JPL 19-5 faithfulness review (`REVIEW_FINDINGS.md`).
- **Repo consolidated to `main` and pushed to `reflective-org/SANDBOX`; docs/ established.**

## Done
- Repo setup (README→SANDBOX, docs/ living files, CONTRIBUTING) — merged (#1).
- **Phase 1 — gas-phase SO2→SO3→H2SO4 — COMPLETE.** Species (#18), sulfur chain NumPy+JAX (#19),
  JPL 19-5 cross-check (#16), termolecular 298 K correction (#17→reverted by #20), run + conservation
  + 3-agent verification. H2SO4 produced; sulfur conserved to 3.5e-15; 92 tests pass.
- **Phase 2 — JAX operator-split coupling skeleton + single config — COMPLETE.** `CoupledScenario`
  single input + units bridge (2.1), data-driven JAX sulfur gate (2.2), absolute TUV-x J on the JAX
  backend (2.3), operator-split coupling driver — frozen midpoint J, terminator snapping, per-interval
  diffrax solve (2.4), and the Phase 2.5 gate: NumPy operator-split mirror (`reference_numpy.py`),
  genuine JAX-vs-NumPy cross-backend parity test, end-to-end real-port validation, 3-agent verification
  (3/3 PASS). H2SO4 produced; sulfur drift 1.3e-15; JAX-vs-NumPy worst rel diff 1.2e-6; 125 tests pass
  (coupled 25 + gas 100).

- **Phase 3 — Couple TOMAS microphysics — COMPLETE.** tomas_bridge (init from Marianna dist, SO2-chem
  off, 3.1), aerosol diagnostics (SA / effective wet radius / H2SO4 wt%, 3.2), gas het seam injecting
  TOMAS radius+composition with NumPy↔JAX parity (3.3), coupled driver with the H2SO4 handoff and
  per-interval SA/radius/wt% feedback (3.4), and the gate: coupling proven sulfur-exact via a
  conserving stub (~1e-12), real-TOMAS run with bounded drift + SA growth, TOMAS-on cross-backend
  parity, plots, 3-agent verification (3.5). SO2→sulfate demonstrated (SA 0.27→21 µm²/cm³ over 2 d);
  total-S drift is TOMAS-internal (~1e-3, see CAVEATS/AD-3.10).

- **Phase 4 — Aerosol → photolysis radiation — COMPLETE.** Spectral aerosol optics from the TomasState
  via per-wavelength Mie on the fixed bin radii (`aerosol_optics.py`, 4.1); dynamic aerosol radiator
  injected into the TUV-x port `_solve` (4.2); wired through the adapter + coupled driver behind
  `switches.aerosol_to_j` with a configurable altitude band (4.3); validation sweep + plot (4.4). J
  responds correctly (scattering enhancement then shielding). `aerosol_band_km` placement flagged OPEN
  (AD-4.2). NumPy mirror kept in parity.

- **Phase 5 — Radiative heating → temperature — COMPLETE.** Ported the `heating_rates.F90` kernel into
  the port (O3 both channels; O2 deferred) reusing actinic flux + xsqy (5.1); `coupled/heating.py`
  box dT/dt = O3 photochemical + aerosol SW absorption / (n_air·cp) (5.2); driver evolves box T behind
  `heating_to_t` (5.3); validation (5.4). **Caveat:** SW heating only, NO longwave cooling (AD-5.4,
  OPEN) — T rises monotonically; not a closed energy balance. Aerosol LW heating (the dominant
  strat-sulfate term) also OPEN.

- **Phase 6 — Dilution — COMPLETE.** coupled/dilution.py (first-order relaxation to the initial
  background, gas + aerosol via TOMAS dilution_step, 6.1); wired into driver + NumPy mirror behind
  switches.dilution (6.2); validation (6.3). All process switches now implemented. Constant rate;
  V(t) schedule + temperature dilution deferred.

- **Phase 7 — Single unified input + end-to-end — COMPLETE.** CoupledScenario is the single input
  (already YAML/JSON); added the 3 sensitivity knobs (nucleation_rate_scale->fn_scale,
  condensation_alpha->TomasState.alpha; coag_kernel_scale RAISES if !=1.0, not wired) (7.1); shippable
  coupled_full.yaml + roundtrip (7.2); validate_phase7 sub-case demo (one input, progressive physics)
  (7.3). All 7 switches implemented; end-to-end reproduces every sub-case.

## Next
- **Phase 8** (future) — Sensitivity sweeps over the knobs. Needs a tomas-jax coagulation-kernel-scale
  knob for coag_kernel_scale. Not started.

## OPEN for the user (see docs/AUTONOMOUS_DECISIONS.md)
- Phase 3: TOMAS internal sulfur loss (AD-3.10) + runaway nucleation (default nucleation off?).
- Phase 4: aerosol vertical placement / `aerosol_band_km` (AD-4.2).
- Phase 5: heating is SW-only, NO longwave cooling / no LW aerosol heating (AD-5.4) -- biggest one.
- Recurring: gammas divide-by-zero at HCl=0 (keep HCl>0); package gas_phase_chemistry + tomas_jax.

## Phase status
| Phase | Title | Status |
|---|---|---|
| 1 | Gas sulfur→H2SO4 (+JPL check) | ✅ done (3-agent verified) |
| 2 | JAX operator-split coupling skeleton + single config | ✅ done (3-agent verified) |
| 3 | Couple TOMAS microphysics (H2SO4 handoff; SA→chem) | ✅ done (3-agent verified) |
| 4 | Aerosol→photolysis radiation | ✅ done (pending 3-agent) |
| 5 | Radiative heating→T | ✅ done (pending 3-agent) |
| 6 | Dilution (gas+aerosol, entrainment) | ✅ done (verified) |
| 7 | Single unified input + validation + plots | ✅ done (verified; fixed 1 bug) |
| 8 | Sensitivity sweeps (coag/nucleation/condensation) | not started |
