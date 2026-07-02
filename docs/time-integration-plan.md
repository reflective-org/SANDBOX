# Design plan — two-level time integration for the coupled driver

Status: PROPOSAL for review (nothing implemented yet). Base = phase-2 `main`.

## 1. Problem
The current coupled driver (`coupled/driver.py`) uses a **single operator-split step `dt_couple`**. Per
interval it: freezes J at the midpoint → integrates the gas ODE over the *whole* interval → hands the
*entire* interval's H2SO4 to TOMAS in **one** `make_step(dt_couple)` → applies heating → dilution.

At a coarse `dt_couple` with full-rate nucleation this over-drives nucleation (rate ∝ [H2SO4]^p sees a
whole interval's H2SO4 as a standing slug) → runaway (N≈1.6e10 cm⁻³, r_eff≈5 nm). It is a **numerical
operator-split artifact**, not physics; it also forced the awkward "nucleation off" workaround.

## 2. Goal
- **All processes on by default** (physically correct because in a clean background the nucleation rate
  is naturally tiny; with fine early stepping it self-limits and you recover r_eff ≈ 0.1 µm).
- **Finer time resolution at the beginning** (high SO2 → fast H2SO4 → burst nucleation; sub-second
  H2SO4 turnover in the first minutes), coarsening later.
- Keep the **expensive TUV-x radiative-transfer solve on a coarse cadence** (J varies on ~minutes with
  SZA; sub-second J is wasteful) while resolving the **stiff gas↔aerosol coupling finely**.

## 3. Proposed architecture — two nested time levels
```
for each OUTER (radiation) step [T0, T1], length dt_rad (coarse, ~minutes):
    solve J once (TUV-x, aerosol-aware) at the midpoint; freeze it for the whole outer step
    compute aerosol optics for aerosol->J once (aerosol state at T0), frozen across inner steps
    for each INNER micro-step [t0, t1] within [T0,T1], length dt_micro (fine->coarse schedule):
        gas chem over dt_micro (diffrax, frozen J)              # produces incremental H2SO4
        handoff: gas H2SO4 -> Gc[SRTSO4]; TOMAS make_step(dt_micro); depleted H2SO4 -> gas
        dilution over dt_micro (analytic)
    heating -> box T once per OUTER step (uses the outer J solve's actinic flux)
```
Nucleation now only ever sees `dt_micro`'s worth of H2SO4 → self-limits; radiation cost is bounded by
`dt_rad`. Gas + TOMAS + dilution run at `dt_micro` (cheap — no TUV-x inside the inner loop).

`dt_micro` schedule mirrors tomas-jax's own Marianna run
(`DEFAULT_DT_SCHEDULE = (120 s, 0.01), (1200 s, 0.1), (14400 s, 10), (>, 20)`), i.e. 0.01 s for the
first 2 min, 0.1 s to 20 min, 10 s to 4 h, 20 s after — capped within each outer interval.

## 4. DECISIONS (confirmed with user)
1. **Inner loop = approach B** (dense-output): ONE adaptive diffrax gas solve per outer step (H2SO4 is
   terminal in the gas phase, so removing it post-hoc is exact); TOMAS consumes the H2SO4 in fine
   micro-steps querying the dense gas solution. Frozen J + frozen het inputs (SA/radius/wt%) per outer.
2. **dt_rad = 600 s** default (300 s also fine), configurable.
3. **dt_micro = ADAPTIVE, not hard-coded.** Choose each micro-step so the fractional change in H2SO4
   and in particle number N stays <= eps (default **eps = 0.1**), bounded by [floor **1e-4 s**, cap
   **20 s**]; **raise a loud error if the floor is hit** (never silently under-resolve). It auto-shrinks
   to 1e-3 s or smaller exactly where nucleation is fast (e.g. high nucleation_rate_scale) and relaxes
   when slow -- no per-scenario tuning. The Marianna fixed schedule is used only as the cap ceiling.
4. **Heating: once per outer step** (SW-only, ~0.1 K/day; uses the outer J solve's flux).
5. **J + aerosol->J optics: recomputed once per outer step**, frozen across the inner micro-steps.
6. **Config:** `dt_couple` -> `dt_rad`; add `micro_eps`, `micro_floor_s`, `micro_cap_s`; `Switches`
   defaults flip to **all True**. Dilution applied once per outer step (analytic exponential is exact
   for the dilution operator; dilution is slow vs the microphysics burst).

## (original decision list, for reference)
1. **dt_rad (outer / J cadence).** Default **300 s** (5 min). J & aerosol optics recomputed once per
   outer step, frozen across the inner micro-steps. Alternative: 600 s (cheaper, coarser J).
2. **dt_micro schedule.** Adopt the Marianna fine→coarse schedule above (fine early). It is applied
   *relative to run start* and clipped to each outer interval.
3. **Efficiency of the inner loop (important).** Two implementations:
   - **(A) Interleaved (simplest):** one small diffrax gas solve per micro-step. Correct but many tiny
     stiff solves early (0.01 s × ~12000 in the first 2 min) — expensive.
   - **(B) Dense-output (recommended):** ONE adaptive gas solve per *outer* step with dense/saved
     output at the micro-times; feed TOMAS the *incremental* H2SO4 between micro-times, subtract what
     TOMAS removed. One stiff gas solve per outer step + cheap TOMAS micro-steps. Much faster, same
     operator-split semantics.
   Recommend **(B)**; fall back to (A) if the H2SO4↔TOMAS back-coupling within an outer step proves to
   need it.
4. **Frozen-J across micro-steps:** J frozen at the outer-step midpoint for all inner steps (2nd-order
   in the outer step). OK?
5. **Aerosol→J cadence:** aerosol optics recomputed once per outer step (aerosol state at T0), frozen
   across inner steps (re-solving J per micro-step is too expensive). OK?
6. **Heating cadence:** dT/dt applied once per outer step from the outer J solve's flux (not per
   micro-step). OK? (Heating is a slow, minor term — SW-only, ~0.1 K/day.)
7. **Config surface:** `CoupledScenario` gains `dt_rad` + a `dt_micro_schedule` (with the default
   above). `dt_couple` is repurposed as `dt_rad` (or kept as an alias). `DT` (output cadence) unchanged.
8. **Defaults:** flip `Switches` to **all True** (full coupled system is the default).

## 5. Validation plan (before re-presenting for review)
- **Convergence:** trajectories (N, r_eff, SO2, H2SO4) stable as `dt_micro` is refined further.
- **Nucleation physical:** all-on clean background → N ~ 1-100 cm⁻³ (not 1e10), r_eff ~0.1 µm,
  condensation-dominated.
- **Conservation:** gas+particulate sulfur drift small (and the 96/98 clamp rarely fires now).
- **Parity:** JAX driver vs NumPy mirror under the two-level scheme.
- Re-run the D1 clean-stratosphere case + the phase validations → **physical** headline numbers, with
  the regime caveat carried in plot titles / VALIDATION.md.

## 6. Scope of change (what moves)
- `coupled/driver.py` + `coupled/reference_numpy.py`: replace the single-step loop with the two-level
  loop (dense-output inner handoff).
- `coupled/coupled_scenario.py`: `dt_rad` + `dt_micro_schedule`; `Switches` defaults all-on.
- Everything else (tomas_bridge, aerosol_props, aerosol_optics, heating, dilution, units, het seam,
  knobs, D1 regimes) is **unchanged** — it is already verified correct.
- Re-validate + regenerate plots; then re-present the coupled model as ~3 coherent PRs.

## 7. Not in scope (tracked separately)
- The tomas-jax 96/98 nucleation-clamp MW bug (upstream fix).
- O2 photochemical heating; longwave cooling / LW aerosol heating (AD-5.4).
