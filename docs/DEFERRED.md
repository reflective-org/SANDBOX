# Deferred / follow-ups (SANDBOX)

- **Phase 2.4 — sulfur gate source (mostly DONE 2.4b).** The coupled driver now gates on
  `CoupledScenario.switches.sulfur` (JAX side), and `build_env` gained a `sulfur_chain=` override so the
  NumPy side can read the SAME switch — making `switches.sulfur` the single gate source in coupled runs
  (`sulfur_chain_active(photolysis)` stays the default only for standalone NumPy). Verified OFF+ON in
  the driver tests. **Remaining for Phase 2.5:** the genuine cross-backend parity harness must drive
  BOTH backends from `switches.sulfur` (pass it to `build_env(sulfur_chain=...)`), and replace the
  helper-equality tuvx stand-in with an integrate-and-compare-species tuvx parity test.
- **Phase 2.4 — CoupledScenario integration debts** (from the #29 review):
  - ~~Validate `concentrations` species keys~~ **DONE (2.4a):** `model_bridge.initial_state` raises on
    unknown species.
  - ~~Single `CoupledScenario -> ModelConfig` mapping~~ **DONE (2.4a):** `model_bridge.to_model_config`.
  - `PHOTOLYSIS_MODES` dedup: **guarded (2.4a)** by an assertion in `model_bridge` that the coupled and
    gas lists match; a true single definition still needs gas_phase_chemistry packaged.
  - **Still open — package gas_phase_chemistry.** `model_bridge` uses an interim `sys.path` insert to
    import the gas model; make `gas_phase_chemistry`/`jaxmodel` an installable package so `coupled/`
    imports it cleanly (removes the path insert and the coupled/conftest dual-path setup).

- ~~**Phase 2 prerequisite — make the JAX sulfur gate data-driven.**~~ **DONE (Phase 2.2).** Both
  backends now derive the gate from `reactions.sulfur_chain_active(photolysis)` (single source of
  truth); `test_sulfur_gate_single_source_all_modes` covers reference/sza/tuvx, so the previous silent
  NumPy-on / JAX-off disagreement for `tuvx` can't recur.
- **Sulfur test coverage (minor):** no test pins the `SO2+HO2->SO3+OH` (I34, 1e-18) coefficient
  magnitude; the I79 analytic test reuses the implementation constant so it checks structure, not the
  constant's value. Low priority (both are cross-checked against JPL in docs/jpl19-5-sulfur-crosscheck.md).

- **Exact `aerosol.F90` port** (fractional-source OD interpolation + Ångström scaling). Not needed for
  the coupling (TOMAS supplies spectral optics directly), but required for config-static aerosol
  parity with Fortran TUV-x.
- **Longwave aerosol heating** fidelity (Phase 5) — dominant stratospheric aerosol heating term;
  outside any shortwave actinic code. Decide parameterization.
- **Trace the uniform ~1.8e-8 residual** in the extraterrestrial-flux normalization (negligible).
- **O3 QY 1:1 Fortran diagnostic** (currently validated against the analytic Matsumi recommendation).
- Remaining reaction-specific QY modules (RONO2/PAN/ketone families) for full TUV-x coverage.
