# Deferred / follow-ups (SANDBOX)

- **Phase 2 prerequisite — make the JAX sulfur gate data-driven.** The NumPy gate is derived from
  data (`Env.sulfur_chain = cfg.photolysis != "reference"`), but the JAX gate is hardcoded per driver
  (`make_vector_field`→0.0, `make_sza_vector_field`→1.0, `build_params` default 0.0). They agree today
  for reference/sza, but `jaxmodel` has **no tuvx driver**, so a `photolysis="tuvx"` run would have
  NumPy chain-ON vs a default JAX call chain-OFF — a silent disagreement (no test covers tuvx; the
  `jaxmodel` docstring already notes it isn't valid for tuvx). When the Phase 2 coupled JAX driver is
  built, drive `sulfur_chain` from `cfg.photolysis` (single source of truth) and add a tuvx parity
  test. (Raised by the Phase 1 3-agent verification.)
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
