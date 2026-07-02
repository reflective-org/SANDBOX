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

## Phase gates (3-agent verification)
_(to be appended per phase — lens verdicts + evidence)_
