# Deferred / follow-ups (SANDBOX)

- **Phase 2.4 — feed the gate from `switches.sulfur`.** Phase 2.2 made `sulfur_chain_active(photolysis)`
  the single source of truth shared by NumPy and JAX. The coupled driver should map
  `CoupledScenario.switches.sulfur` onto that (so the switch, not just the photolysis string, controls
  the gate) when `CoupledScenario -> ModelConfig` is wired.
- **Phase 2.4 — CoupledScenario integration debts** (from the #29 review):
  - Validate `concentrations` species keys against `config.SPECIES` when the driver wires composition
    (currently a typo'd species is silently accepted — the exact silent-assumption we want to avoid).
  - Dedupe `PHOTOLYSIS_MODES` (defined in both `coupled/coupled_scenario.py` and
    `gas_phase_chemistry/config.py`) to one source of truth once the import lands.
  - Single `CoupledScenario -> ModelConfig` mapping so the duplicated fields (T/P/SA/WTR/Yn2o5/opt/
    lat/lon/...) can't drift.
  - Reconcile the two `conftest.py` `sys.path` insertions when `coupled/` imports `gas_phase_chemistry`
    (make `jaxmodel` importable as a package rather than replicating the path hack — see packaging note).

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
