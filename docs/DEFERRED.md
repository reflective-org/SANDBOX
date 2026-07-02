# Deferred / follow-ups (SANDBOX)

- ~~**Phase 2.4 — sulfur gate source.**~~ **DONE (2.4b + 2.5).** The coupled driver gates on
  `CoupledScenario.switches.sulfur` (JAX side), and `build_env` gained a `sulfur_chain=` override so the
  NumPy side reads the SAME switch — making `switches.sulfur` the single gate source in coupled runs
  (`sulfur_chain_active(photolysis)` stays the default only for standalone NumPy). Phase 2.5 closed the
  remainder: the cross-backend parity harness (`reference_numpy.py`) drives BOTH backends from
  `switches.sulfur`, and `test_coupled_parity.py` is a genuine integrate-and-compare-species tuvx
  parity test (diffrax vs SciPy BDF, matched frozen J) replacing the helper-equality stand-in.
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
- **CI-cheap real-port parity smoke test (PR#35 review, minor).** The regression-guarded parity test
  (`test_coupled_parity.py`) runs the coupled backends with a *stubbed* adapter (fast). The real
  TUV-x-port JAX-vs-NumPy agreement (1.2e-6) is only produced by the `validate_coupled.py` script (too
  slow for CI) and lives as committed PNGs, not a test. A short real-port run (e.g. a few hours, 1
  outer step) small enough for CI would close the gap so real-port parity is regression-guarded too.
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
