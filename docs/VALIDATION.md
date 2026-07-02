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
_(to be appended per phase — lens verdicts + evidence)_
