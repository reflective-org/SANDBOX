# Validation log (SANDBOX)

What was checked against what, with results and dates. 3-agent verification verdicts are recorded here
at each phase gate.

## Photolysis port (pre-SANDBOX)
- Radiation field, geometry, O3 cross section, per-reaction J: **~1.8e-8 vs Fortran TUV-x**
  (no-aerosol config). LA/SR bands ported and validated. See `validation/` and `DEVELOPMENT.md`.
- O3 photolysis quantum yields (Matsumi 2002): validated against the analytic recommendation
  (`tests/test_o3_quantum_yield.py`).
- Branching QY (HNO4/ClOOCl): invariant checks (channels partition the total) — `tests/test_branching.py`.

## Chemistry
- Reference-mode reproduces the MATLAB/Octave trajectory; JAX `jaxmodel` matches the NumPy RHS.
- JPL 19-5 faithfulness review: `REVIEW_FINDINGS.md`.

## JPL 19-5 sulfur cross-check (Phase 1.4, done first) — `docs/jpl19-5-sulfur-crosscheck.md`
- SO2+OH+M→HOSO2 (Table 2-1 I4) + HOSO2+O2→HO2+SO3 (I92) ⇒ net **SO2+OH→SO3+HO2** product confirmed.
- SO3+2H2O→H2SO4 (I79): **kI = 8.5e-41·exp(+6540/T)·[H2O]²** — confirmed handbook value.
- SO2+HO2 (I34): rate **<1e-18** upper limit; **no JPL product recommendation** (SO3+OH is our choice).
- **Discrepancy found:** termolecular reference temperature is **300 K** in JPL 19-5 (Table 2-1
  k0(300)), but the code used `troe298` (298 K). Fixed on `fix/termolecular-ref-temp-300k` (see that PR).

## Phase gates (3-agent verification)

### Phase 1 — gas SO2→SO3→H2SO4 (2026-07-02) — **PASS**
Three independent agents, each working from primary sources (JPL 19-5 PDF / code / tests), not from a
summary:
- **Lens 1 — correctness vs JPL 19-5: PASS.** Pulled the Table 2-1 header verbatim
  (`k0(T)=k0_298 (T/298)^-n`) → termolecular reference is **298 K** (settling the 298↔300 dispute);
  SO2+OH = I4+I92 net; SO2+HO2 = I34 upper-limit with products correctly flagged as an assumption;
  SO3+H2O = I79, with the `[H2O]^2` reconstruction shown. No discrepancies.
- **Lens 2 — tests + NumPy↔JAX parity: PASS (with noted gaps, now addressed).** 92/92 tests pass;
  coefficient/order parity exact (max rel diff 2e-16); reference-mode faithfulness proven (species
  1–34 tendencies unchanged, chain coeffs exactly 0). Gaps it found → fixed: added
  `test_so2_to_so3_production_rate_matches_k68` and `test_chain_on_numpy_jax_parity` (chain-ON dC/dt
  parity). Remaining item logged: JAX gate is hardcoded per-driver (Phase 2 prerequisite → DEFERRED.md).
- **Lens 3 — physical/conservation + assumptions: PASS.** Sulfur conservation bit-exact (crafted
  state) and machine-precision end-to-end (drift 6.6e-16); units correct; gate has no double-counting.
  Concerns fixed: stale "300 K" row in the crosscheck doc; clarified the SO2+HO2 ~0.02% metric; added
  `ModelConfig.photolysis` enum validation (no silent mis-gate on a typo).

End-to-end run (`sulfur_budget.py`, 10-day tuvx, model_input): H2SO4 0 → ~4.8e4 pptv, SO2 −5.0%,
total-sulfur drift +3.5e-15. Plots in `gas_phase_chemistry/sulfur_budget/`.
