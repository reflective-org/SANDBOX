# Autonomous decisions log (SANDBOX)

Decisions made **while the user was asleep** during the autonomous Phase 3→7 build (authorized
2026-07-01 evening: "go on through the plan and create the PRs and build all the way to phase 7. I
will kick off review when I am up").

Each entry states the **question**, the **options considered**, the **decision taken**, and the
**rationale** — so the user can review/override every judgment call at wake-up. Decisions that were
answered directly by the user (the four Phase-3 design questions) live in `DECISIONS.md`, not here;
this file is only for calls I made without the user in the loop. Anything genuinely high-stakes that
I could not resolve defensibly is listed under **OPEN — needs user** at the bottom.

Convention: pick the most defensible, reversible, well-documented option; prefer faithfulness to the
reference implementations (TOMAS equations, TUV-x Fortran, JPL) over convenience; never silently
simplify; never bypass an error.

---

## Phase 3 — TOMAS microphysics coupling

### AD-3.1 — How to depend on tomas-jax
**Q:** Install tomas-jax into the SANDBOX venv, or bridge via sys.path?
**Decision:** **sys.path bridge** to `../tomas-jax` (and `../tomas-jax/experimental_case` for the
Marianna distribution loader), mirroring the existing `model_bridge.py` gas-model bridge.
**Rationale:** pip-installing tomas-jax risks resolving different jax/diffrax versions and breaking
the validated Phase 1–2 stack. The bridge is the established pattern here. Verified the package
imports and a full `make_step` runs under the SANDBOX venv. Proper packaging tracked in `DEFERRED.md`.

### AD-3.2 — TOMAS branch pinned
**Q:** Which tomas-jax branch/commit does the coupling target?
**Decision:** `feat/marianna-dilution` (current checkout). Recorded so results are reproducible.
**Rationale:** it carries the Marianna distribution + dilution needed for Phases 3 and 6.

### AD-3.3 — Water activity a_W when composition comes from TOMAS
**Q:** `hetgammas_jpl00` needs both the H2SO4 weight-percent AND the water activity `a_W`. Decision 2
says derive the weight-percent from TOMAS. What about `a_W`?
**Decision:** Override only the **H2SO4 weight-percent** with the TOMAS value; keep **`a_W` from the
gas-phase thermodynamics** (`h2so4wp_at`'s `a_W = pH2O/p0H2O`).
**Rationale:** `a_W` is water *activity* — an RH-like gas-side quantity set by water vapour and T, not
a particle bulk-composition variable. TOMAS supplies particle composition (SO4 vs H2O mass), which is
exactly the weight-percent; it does not supply a gas-phase water activity. Mixing TOMAS composition
with the thermodynamic `a_W` is the minimal, physically-cleanest override. Flagged for review: a fully
self-consistent treatment would reconcile TOMAS's own water uptake with `a_W` (tracked in DEFERRED).

### AD-3.4 — Heterogeneous-chem radius definition (REVISED to effective radius)
**Q:** Which single radius represents the polydisperse TOMAS distribution for the reacto-diffusive
f-factor in `hetgammas_jpl00` (which takes one scalar radius)?
**Decision:** **Surface-area-weighted (effective) wet radius** `r_eff = Σ Nk·r_wet³ / Σ Nk·r_wet²`
(the 3rd/2nd moment), in cm. (Initially chose number-weighted; **revised** — see below.)
**Rationale:** heterogeneous uptake is carried by the surface-area-dominant (larger) particles, so the
surface-area-weighted mean is the physically correct single representative for the f-factor, and it is
exactly the standard aerosol *effective radius*. Wet (deliquesced) particle, matching the old 1-µm
value. **Why revised from number-weighted:** when nucleation is on, the number-weighted mean collapses
toward the ~1 nm nucleation mode — unphysical for uptake AND numerically fatal: the f-factor
`coth(r/l) − l/r` becomes `inf − inf → NaN` when r ≪ the reacto-diffusive length l. r_eff is far more
robust than the number-weighted mean and the f-factor stays finite.
**Correction (Lens-3):** the earlier claim that r_eff "stays in ~0.05-0.5 µm for realistic
distributions" is FALSE under the shipped `validate_phase3` config: TOMAS's Ricco-Dunne nucleation
produces a runaway burst (final total N ≈ 1.6e10 cm⁻³, implausibly high vs real stratosphere ~1-100
cm⁻³) that dominates BOTH number AND surface area, so r_eff collapses to ~5 nm (0.005 µm). The f-factor
is still finite at 5 nm (survivable), but surface-area weighting does not "protect" against the fine
mode when nucleation dominates the surface area itself. See the OPEN runaway-nucleation item.
Empty distribution → 0.1e-4 cm fallback.

### AD-3.9 — dt_couple / gas-H2SO4 stability limit + loud NaN guard
**Q:** How to handle TOMAS going non-finite under a large H2SO4 slug?
**Decision:** (a) rely on a **small `dt_couple`** (decision 3) to keep the per-step gas H2SO4 in TOMAS's
stable range; (b) add an explicit **NaN/inf guard after every TOMAS step** in both the JAX driver and
the NumPy mirror that RAISES a clear `RuntimeError` (never continue silently — project rule).
**Rationale:** TOMAS's Ricco-Dunne nucleation rate ∝ [H2SO4]^p overflows to NaN when the gas H2SO4
handed in is ≳1e9 molec/cm³. Because the operator split produces all of an interval's H2SO4 *before*
TOMAS consumes it, a large `dt_couple` overestimates the peak H2SO4 TOMAS sees. Realistic stratospheric
gas H2SO4 is ~1e6-1e8; a small `dt_couple` (≲ a minute for high-SO2 cases) keeps each step's slug well
below the limit. The guard converts a silent NaN cascade into an actionable error naming `dt_couple`.
Documented as a stability constraint in CAVEATS.md; a genuinely adaptive TOMAS sub-step schedule is a
DEFERRED robustness improvement.

### AD-3.5 — Relative humidity fed to TOMAS
**Q:** TOMAS water uptake needs an RH (0-1). The gas model specifies water as WTR (ppm). What RH?
**Decision:** `rh = a_W`, the gas-model **water activity** `pH2O/p0_liquid` from `aerosol.h2so4wp_at`,
clipped to [1e-4, 0.99].
**Rationale:** a particle in equilibrium sees the ambient water activity as its effective RH, and this
makes TOMAS's water uptake consume the SAME water field the gas heterogeneous chemistry already uses
(`p0h2o` in `aerosol.py` is the liquid-water saturation, so `a_W` is RH over liquid water — exactly
what TOMAS's `water_uptake_sulfate(rh)` fit expects). Clip range matches the fit's validity (~1-99%).
At T=210 K, P=68 mbar, WTR=5 ppm this gives a_W≈0.027 (2.7% RH), physically reasonable for the cold
dry lower stratosphere. Single-sources water across both models.

### AD-3.6 — TOMAS scheme choices
**Q:** `make_step` takes `cond_method`, `nucl_scheme`, `water_scheme`. Which?
**Decision:** `cond_method='ppm_jit'`, `nucl_scheme='ricco_dunne'`, `water_scheme='h2so4_tabazadeh'` —
the exact combination used by the validated tomas-jax Marianna run
(`experimental_case/run_marianna_dilution.py`).
**Rationale:** reuse the settings TOMAS itself validated for this stratospheric regime rather than
introduce an unvalidated combination. Any of these is a later sensitivity axis. `water_scheme=
'h2so4_tabazadeh'` (vs 'isorropia') is the stratospheric-sulfate choice, consistent with the gas
model's Tabazadeh-based `h2so4wp_at`.

### AD-3.7 — TOMAS grid-cell volume (boxvol)
**Q:** TOMAS's `Gc`/`Nk`/`Mk` are per grid cell; what boxvol?
**Decision:** `BOXVOL_CM3 = 1.0e6` (1 m³), the tomas-jax canonical value.
**Rationale:** boxvol is a reference volume that cancels for all intensive outputs (concentrations,
surface area per cm³, weight-percent); the units bridge already matches TOMAS's constant. 1 m³ matches
the tomas-jax examples so cross-checks against them are apples-to-apples.

### AD-3.8 — TOMAS `Mk[:,SRTSO4]` is H2SO4-equivalent mass (no 96/98 conversion)
**Q:** Is TOMAS's aerosol sulfate mass stored as SO4 (96 g/mol) or H2SO4 (98 g/mol)? This drives both
the heterogeneous-chem weight-percent and the Phase-3.5 sulfur budget.
**Decision:** Treat `Mk[:,SRTSO4]` as **H2SO4-equivalent mass** (MW 98).
**Rationale:** verified in `tomas_jax/solvers/condensation.py` `_condensation_step_core` (the JIT path
actually run, ~lines 246-308; the numpy path at 192-211 is deprecated) — condensation moves the gaseous
H2SO4 mass (`Gc[SRTSO4]`, MW 98) into `Mk[:,SRTSO4]` **1:1** with no MW conversion, and normal-branch
nucleation depletes gas by exactly the SO4 mass it adds. So particulate sulfur (kg) equals the H2SO4
mass condensed/nucleated; the weight-percent is `100·M_SO4/(M_SO4+M_H2O)` with `M_SO4` taken directly,
and the budget converts particulate mass → molec/cm³ with MW=98. **Clamp caveat (corrected by Lens-3 —
NOT edge-case-only):** the nucleation *clamp* branch (gas exhausted) applies a 96/98 factor; in the
coupled stratospheric regime the gas H2SO4 is fully depleted **every daytime step**, so the clamp fires
routinely and is the dominant sulfur-drift term (see AD-3.10). Flagged in CAVEATS.md.

### AD-3.10 — Coupled-run sulfur drift = the 96/98 nucleation-clamp firing every daytime step (CORRECTED by Lens-3)
**Q:** The coupled run does not conserve total (gas+particulate) sulfur to machine precision. Bug, or
inherent?
**Finding (corrected by the physical-conservation verification agent):** the coupled 1-day drift is
**−4.3e-4**, and it is **TOMAS-internal but NOT generic MNFIX drift**. The real mechanism, traced
per step: condensation+nucleation consume the gas H2SO4 to **exactly 0 every daytime step**, which
triggers the **nucleation *clamp* branch** (`tomas_jax/physics/nucleation.py:521-539`) — it depletes
gas by `gc_so4` but deposits only `gc_so4·(96/98)` into the aerosol, a **−2.04% loss on the clamped
fraction every daytime step**. At the coupled run's actual H2SO4 scale (~1e6/step) standalone TOMAS
otherwise conserves to ~2e-6; the earlier "~1.3e-2 over 40 steps" figure only reproduces at an
artificially large ~1e8/step slug. So the clamp (AD-3.8) is **not an edge case here — full depletion
is the normal operating regime**, and it is the dominant drift term.
**Decision:** (a) the **coupling layer is proven exact** — a perfectly-conserving stub conserves total
sulfur to ~1e-12 and drop/double-count stubs fail loudly (verified by Lens-2); (b) accept the clamp
loss as a **tomas-jax property** (fixing the 96/98 clamp is a tomas-jax change, DEFERRED), assert the
real-run drift is bounded (<2e-2), and correct AD-3.8/CAVEATS to state the clamp fires every daytime
step. **FLAGGED FOR USER** (see OPEN).

### AD-3.11 — aerosol_props must use the SAME water scheme as the TOMAS step (bug fix from Lens-1)
**Q:** The diagnostics (`aerosol_props._wet_diameters_m`) recompute equilibrium water to get wet SA /
radius / wt%. Which water scheme?
**Finding (Lens-1 verification):** the diagnostics initially used `calc_equilibrium_water` (the
ISORROPIA/NH4HSO4 default, ×1.2 factor) while the coupled TOMAS step evolves the particles with
`water_scheme='h2so4_tabazadeh'` (AD-3.6). Two different water fields → the SA/radius/wt% fed to the
het chemistry did not match the particles TOMAS carried (measured 1.3-1.8× SA error across RH).
**Decision/Fix:** `aerosol_props._wet_diameters_m` now uses `calc_equilibrium_water_h2so4(Mk, rh, temp)`
— the Tabazadeh binary H2SO4/H2O scheme — matching the step. A consequence: `h2so4_weight_pct` now
equals the Tabazadeh `wt%` exactly (self-consistent), and SA/radius are on the same water basis TOMAS
uses. Caught by the correctness-lens agent; fixed before the phase gate.

_(further Phase-3 decisions appended as they arise)_

---

## OPEN — needs user review at wake-up

- **[Phase 3] TOMAS internal sulfur non-conservation (~1%/day).** (AD-3.10) The tomas-jax
  `feat/marianna-dilution` microphysics loses ~1.3% of sulfur over ~40 nucleation-active steps in its
  OWN gas+particulate budget. The coupling layer is proven exact (stub tests), so total-S drift in a
  coupled run is inherited from TOMAS. **Question for you:** is this acceptable for SANDBOX's purposes,
  or should we (a) investigate/fix mass conservation in tomas-jax (MNFIX / PPM condensation / nucleation
  cluster accounting), or (b) pin a different tomas-jax branch/commit with better conservation? I have
  proceeded treating it as a documented caveat (CAVEATS.md) + DEFERRED item so Phases 4-7 can continue.

- **[Phase 3] Runaway homogeneous nucleation (N ≈ 1.6e10 cm⁻³).** (Lens-3, AD-3.4) With nucleation on,
  TOMAS's Ricco-Dunne scheme produces an implausibly large number of sub-10 nm particles at these
  stratospheric conditions (real background is ~1-100 cm⁻³), collapsing the effective radius to ~5 nm
  and driving the every-step 96/98 clamp loss (AD-3.10). This is likely an artifact of the operator
  split dumping a whole interval's H2SO4 into one nucleation step and/or nucleation being inappropriate
  for a quiescent stratospheric background (where condensation onto pre-existing aerosol dominates and
  homogeneous nucleation is rare outside fresh plumes). **Questions for you:** (a) should nucleation
  default OFF for background stratospheric runs (condensation+coagulation only give a physical r_eff
  ~0.1 µm)? (b) or use a much smaller dt_couple / an internal TOMAS sub-step schedule to tame the
  burst? (c) or a different nucleation scheme? I proceeded with all three microphysics ON in the
  validation for completeness, and documented the artifact; the coupling machinery itself is correct.
