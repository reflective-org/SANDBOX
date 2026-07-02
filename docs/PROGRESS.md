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

## Next
- **Phase 4** — Two-way radiation: aerosol → photolysis. Feed TOMAS Mie optics (OD/SSA/g) into the
  TUV-x port's aerosol radiator so J responds to aerosol. Gets its own sub-plan + tasks.

## Phase status
| Phase | Title | Status |
|---|---|---|
| 1 | Gas sulfur→H2SO4 (+JPL check) | ✅ done (3-agent verified) |
| 2 | JAX operator-split coupling skeleton + single config | ✅ done (3-agent verified) |
| 3 | Couple TOMAS microphysics (H2SO4 handoff; SA→chem) | ✅ done (3-agent verified) |
| 4 | Aerosol→photolysis radiation | not started |
| 5 | Radiative heating→T | not started |
| 6 | Dilution (gas+aerosol, entrainment) | not started |
| 7 | Single unified input + validation + plots | not started |
| 8 | Sensitivity sweeps (coag/nucleation/condensation) | not started |
