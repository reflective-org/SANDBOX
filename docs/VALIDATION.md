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
- **Termolecular reference temperature = 298 K** (resolved). An initial 298→300 K "fix" was **reverted**
  after the Phase 1 gate re-read the Table 2-1 header verbatim (`k0(T)=k0_298 (T/298)^-n`); the "300"
  seen in the PDF was the g-column / older-literature note, not the reference. Code stays at `troe298`.
  See the Phase 1 gate below and PR#20.

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

### Phase 2 — JAX operator-split coupling driver (2026-07-02) — **PASS (3/3)**
Three independent agents, each re-deriving from the code/spec (not from the implementer's summary),
one per lens. Full verdicts below; all three returned PASS.

- **Lens 1 — driver correctness: PASS (6/6).** Verified: (1) J is frozen at each outer-step
  **midpoint** (2nd-order) with a fresh diffrax solve per interval (never integrating across a J
  discontinuity); (2) day/night snapping subdivides intervals at terminator crossings incl. polar
  edge cases, so every interval is fully-day or fully-night and J≡0 at night; (3) `dt_couple` drives
  the J recompute via the uncached `_frozen_j_values`, bypassing the adapter's 120 s cache; (4) a
  single SZA source (gas `solar.py`) feeds both the gate and TUV-x; (5) `switches.sulfur` is the one
  gate source, passed to both backends; (6) J is consumed correctly incl. the O3 opt-split. Minor:
  no `days<1` guard → **fixed** (`coupled_scenario.py`, `test_days_must_be_at_least_one`).
- **Lens 2 — tests + parity genuineness: PASS.** Suites green (coupled 24→25, gas 100). Confirmed
  `test_coupled_parity` is a *genuine* cross-backend test: diffrax Kvaerno5 vs SciPy BDF integrated
  independently over the same operator-split grid with identical frozen J and sulfur gate, full
  species trajectories compared. Tolerances are honest/conservative (tightest bulk species `Cl` needs
  2.3e-4 vs the 2e-3 bound → 8.8× headroom). **Empirically falsified** the "could pass while
  disagreeing" failure mode: perturbing one backend's J by 10 % pushes required rtol to 0.134, so the
  assertion fails loudly. Day/night snapping and `switches.sulfur=False` both independently verified.
  Note (tracked): physical-J parity lives only in the `validate_coupled.py` harness (slow real port),
  not in CI.
- **Lens 3 — physical/conservation + no-silent-assumptions: PASS.** Sulfur conserved to machine
  precision with the chain ON (max rel drift 8.8e-16 sza; 2.0e-15 tuvx). Units bridge exactly
  invertible (≤4 ULP) and byte-matches TOMAS's formula/Avogadro; the ~0.036 % gas-model Avogadro seam
  is real and documented. Day/night intervals clean (0 straddling, 0 night intervals with nonzero
  j_scale). H2O single-sourced from WTR (inconsistent value raises); sulfur gate single-sourced; no
  `try/except` swallowing or "for simplicity" shortcuts anywhere in the 5 coupled files. **Design
  nuance flagged (not a defect):** `switches.sulfur=False` swaps in the reference-only legacy SO2
  lumps (`SO2+OH→HO2`, `SO2+HO2→`) rather than freezing SO2, so total-S is conserved only with the
  gate ON — MATLAB-faithful and intended → recorded in `CAVEATS.md`.

**End-to-end real-port run** (`validate_coupled.py`, 2-day tuvx, dt_couple=3600 s, 53 outer steps):
H2SO4 0 → 7.35e3 pptv, SO2 −0.77 %, sulfur drift +1.3e-15, and JAX-vs-NumPy worst relative species
diff **1.2e-6 (SO2)** under matched J (isolates the solver difference). Plots in `coupled/validation/`
(`coupled_sulfur.png`, `coupled_conservation.png`, `coupled_jax_vs_numpy.png`).
