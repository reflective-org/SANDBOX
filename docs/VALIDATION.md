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

### Phase 3 — TOMAS microphysics coupling (2026-07-02) — **PASS (3/3)**
Three independent agents, one per lens, each re-deriving from the code + the TOMAS/gas sources (not the
implementer's summary). All three returned PASS; two found real issues that were fixed before the gate.

- **Lens 1 — correctness vs TOMAS: PASS (bug found + fixed).** Verified make_step is called with
  'so2_chemistry' OMITTED (the only internal H2SO4 source, `so2_chemistry.py:245`, never runs); the
  units handoff is exact (Avogadro/MW match TOMAS; `.set` write-back, no double-count); `Mk[:,SRTSO4]`
  is H2SO4-equiv mass (1:1 condensation transfer, AD-3.8); SA/r_eff formulas match a hand recompute
  exactly; NumPy↔JAX het-seam parity holds; rh = a_W. **Found a real bug:** the diagnostics recomputed
  wet water with the ISORROPIA scheme while the step used Tabazadeh (1.3-1.8× SA error) → **fixed** to
  `calc_equilibrium_water_h2so4` (AD-3.11).
- **Lens 2 — tests & coverage: PASS.** Re-ran suites: coupled 47 passed (incl. slow real-TOMAS), gas
  100 passed, no regression. Confirmed the coupling-conservation test is genuine by injecting
  drop/double-count stubs (both drift 2.6e-2 → fail as they should); het-seam parity has ~4 orders of
  headroom and catches a 2× radius mismatch; TOMAS-on cross-backend parity tolerances are honest
  (tightest headroom 3.8×, ClOOCl). Flagged coverage gaps → **closed** with `test_phase3_coverage.py`
  (NaN guard both backends, pinned RH, run-level SO2-off, NumPy switch-off).
- **Lens 3 — physical & conservation: PASS (doc corrections).** Independently: sulfur budget drift
  −4.3e-4/day (bounded); H2SO4 is terminal in the gas mechanism with TOMAS its sole sink (no
  double-count); NaN guard verified firing (5.9e10 slug → clear RuntimeError); no try/except bypass;
  boxvol invariance exact for conc/radius/wt%, ~1e-4 for SA (TOMAS floor). **Corrected two doc claims:**
  the drift is the 96/98 nucleation clamp firing *every daytime step* (not generic MNFIX, not an edge
  case — AD-3.10/3.8 fixed), and r_eff *collapses to ~5 nm* under a runaway nucleation burst (N≈1.6e10
  cm⁻³, AD-3.4 fixed + flagged OPEN for user).

**End-to-end run** (`validate_phase3.py`, 2 d, dt_couple=3600 s, SO2=1e4 pptv, all microphysics on):
SO2 2.34e10→2.22e10, particulate S ×12, SA 0.39→30.8 µm²/cm³, max gas H2SO4 6.6e6 (stable), total-S
drift −8.9e-4 (TOMAS clamp, coupling handoff exact). Plots in `coupled/validation/phase3_*.png`.
**Two items flagged OPEN for the user** in AUTONOMOUS_DECISIONS.md: (1) TOMAS's every-step clamp
sulfur loss, (2) the runaway homogeneous nucleation (implausible N) — both tomas-jax behaviors, not
coupling bugs.

---

### Phase 4 — Aerosol → photolysis radiation (2026-07-02) — **PASS (2/2 lenses)**
Two independent agents (correctness; tests+physical), each re-deriving from the code + the TOMAS Mie
source + the TUV-x port.

- **Lens 1 — correctness: PASS (no bugs).** The Mie table (`bhmie_qsca_jax` on x=2πr/λ, fixed bin
  radii + fixed index) is **bit-exact** vs TOMAS's own `precompute_mie_properties` at 550 nm; units
  correct (r m→cm, λ nm→m, OD=b_ext·Δz with Δz km→cm); Qsca≤Qext, |g|≤1; SSA≈1 for sulfate. Port
  injection appends the aerosol radiator before `accumulate` (None reproduces the prior solve exactly);
  adapter clears `aerosol_props` in a `finally` that re-raises (no swallow, no stale state); NumPy/JAX
  build identical optics. Enhancement-then-shielding confirmed correct physics (pure absorber → J
  monotonically drops, so the enhancement is scattering-driven).
- **Lens 2+3 — tests + physical: PASS.** 62 coupled + 100 gas tests pass, no regression. Bug-injection
  confirms the tests are genuine (skipping the aerosol append → `test_port_aerosol` fails; ignoring Nk
  → `test_aerosol_optics` OD∝Nk fails). Physics re-derived: SSA=1.000000, OD∝N (×5.0) and ∝band, OD=0
  outside band, **boxvol cancels** (OD ratio 1.000000), driver-level J change up to +72 %/−98 %.
  No error bypass. Coverage gaps flagged → addressed: added a numeric SSA/g scattering-weighting test;
  the end-to-end `aerosol_to_j` on/off (agent measured a 1265× trajectory change) and the JAX/NumPy
  aerosol parity are NOT committed as CI tests (a real-TUV-x coupled run solves the full radiation
  field per interval ~30 s, and the runaway-nucleation OPEN item drives the stiff gas solver to its
  step cap) — exercised instead by `validate_phase4.py` and confirmed by the agent; the mirror runs
  byte-identical aerosol code.

**End-to-end** (`validate_phase4.py`): box-altitude J/J0 vs 550 nm column OD shows the scattering
sulfate aerosol enhancing J to ~1.4× at OD~1-2 then shielding at high OD; deep-UV HNO3 decreases
monotonically. Plot: `coupled/validation/phase4_j_vs_od.png`. Two items OPEN for the user
(AUTONOMOUS_DECISIONS.md): AD-4.2 vertical placement (slab bounds set the feedback magnitude) and the
Phase-3 runaway nucleation (which, with aerosol_to_j on, further stresses the gas solver).

---

### Phase 5 — Radiative heating → temperature (2026-07-02) — **PASS (2 independent lenses)**
Two independent agents each re-derived the kernel from `heating_rates.F90`, checked units, ran the 8
targeted tests, and did bug-injection. Both PASS, no bugs. (Two earlier agent runs were killed by the
harness exactly when they launched pytest — not a finding; re-run to completion.)

- **Lens 1 — correctness vs Fortran: PASS.** The energy formula `max(0, hc(1/λ − 1/λ_thr))` and the
  heating sum `Σ actinic·energy·σφ` are character-for-character the Fortran (`heating_rates.F90:224-226,
  298-303`); `hc` matches `constants.F90`; **negative actinic flux is zeroed** (matches Fortran); O3
  energy terms 310.32/1179.87 correct; both O3 channels used, O2 correctly deferred (AD-5.1).
- **Lens 2 (consolidated) — units + tests + physical: PASS.** dT/dt units verified K/s
  (H_gas=[O3]·Σheat, H_aer=Σflux·b_abs·E_photon, /(n_air·cp), cp=3.5kB correct); night gating via
  `compute_box_heating→None` with a `try/finally` that re-raises (no swallow); T-update + M recompute
  matched in the NumPy mirror. **8/8 targeted tests pass; bug-injection** (`box_dTdt→0`) makes the
  "T rises" test FAIL (non-vacuous), reverted clean. Diurnal O3 heating ≈0.24 K/day at noon/19 km
  (plausible); SW-only/no-LW-cooling/monotonic-T limitation prominently documented (CAVEATS + AD-5.4
  OPEN); `heating_to_t` defaults OFF (surfaced, not hidden).

**End-to-end** (`validate_phase5.py`): diurnal heating 0.24 K/day (0 at night); a 3-day coupled run
with `heating_to_t` warms the box 210.00→210.30 K. Plots `coupled/validation/phase5_*.png`.
**OPEN for the user (AD-5.4):** no longwave cooling (T not a closed energy balance) and no LW aerosol
heating (dominant strat-sulfate term) — heating is shortwave-only. Doc-wording nit found + fixed (the
formula text no longer reads as if etfl is applied twice; code applies it once, correctly).

---

### Phase 6 — Dilution (2026-07-02) — **PASS (independent verification)**
An independent agent re-derived the formula vs TOMAS, checked the wiring in both backends, ran the
targeted tests, and did bug-injection.
- **Formula:** `dilute_gas = conc_bg + (conc−conc_bg)·exp(−k·dt)` is byte-for-byte TOMAS's
  `dilution_step` exponential; `dilute_aerosol` calls the actual `tomas_jax` `dilution_step` on
  Nk/Mk/Gc toward the background TomasState. Exact for constant k over the interval.
- **Wiring:** both backends capture background = the INITIAL state before the loop and apply dilution
  LAST each interval behind `switches.dilution` (het inputs refreshed from the diluted aerosol);
  structural mirrors. `dilution_rate` validated ≥0; all 7 switches now implemented.
- **Tests:** 19 passed (dilution 5, driver_dilution 3, scenario 11). Genuine: dilution-off is
  byte-identical to no-switch; strong dilution suppresses H2SO4 to <1e-3 of no-dilution; passive tracer
  matches `exp(−k t)` to rtol 1e-12. **Bug-injection** (flip `−k`→`+k`) makes the decay + monotonic
  tests fail; reverted, `git diff` clean.
- **No silent assumptions:** no try/except; background=initial, constant rate (V(t) deferred), no
  temperature dilution — all documented (AD-6.x, CAVEATS, DEFERRED).
- Coverage gaps (minor): no JAX↔NumPy parity test with dilution ON; `dilute_aerosol` not exercised
  through a full TOMAS-active `run_coupled`.

**End-to-end** (`validate_phase6.py`): passive-tracer relaxation matches the analytic `exp(−k t)` to
5e-14; a 3-day coupled run with dilution keeps SO2 near background and suppresses H2SO4 (13947→3679
pptv). Plots `coupled/validation/phase6_*.png`.

---

**What is CI-enforced vs. a committed artifact** (PR#35 review): the automated test
(`test_coupled_parity.py`) proves **solver parity under matched J** using a *stubbed* adapter (fast) —
that is the regression gate. The **real-port** 1.2e-6 agreement above comes from `validate_coupled.py`,
a diagnostic **script** (a real 2-day TUV-x solve is too slow for CI), so it is a manually-run,
committed artifact (the PNGs), **not** a regression-guarded check. Don't read the committed PNG as a CI
pass. A CI-cheap real-port smoke test is tracked in `DEFERRED.md`.
